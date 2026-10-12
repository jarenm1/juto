//! Authenticated signaling for remote desktop viewers.
//!
//! A minimal HTTP/1.1 + WebSocket server that serves the browser viewer page and
//! relays SDP/ICE between a [`PresentationWorker`] and one viewer. Access is
//! gated by a one-time attach token bound to an environment, viewer, and role
//! set; the token is consumed on use and expires after 60 seconds.
//!
//! TLS termination and public reachability belong to the deployment gateway;
//! this server binds a loopback address and never creates a world-readable
//! attach URL. See `docs/agent-compositor.md` sections 7 and 10.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use parking_lot::Mutex;

use crate::error::DesktopError;
use crate::protocol::GrantRole;
use crate::remote::{PresentationWorker, RemoteStreamConfig, SignalingMessage, VideoCodec};
use crate::supervisor::DesktopSupervisor;

/// Lifetime of an unconsumed attach token (spec section 10).
const TOKEN_TTL: Duration = Duration::from_secs(60);

/// A one-time attach grant for a viewer.
#[derive(Debug, Clone)]
pub struct AttachGrant {
    pub token: String,
    pub env_id: String,
    pub viewer_id: String,
    pub roles: Vec<GrantRole>,
    pub expires_at: u64,
}

/// What a viewer reported about its own session, used for verification.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViewerReport {
    pub connected: bool,
    pub video_width: u32,
    pub video_height: u32,
    pub sampled_pixel: Option<Vec<u8>>,
    pub input_sent: bool,
    pub error: Option<String>,
}

struct ServerState {
    supervisor: Arc<DesktopSupervisor>,
    grants: Mutex<HashMap<String, AttachGrant>>,
    reports: Mutex<HashMap<String, ViewerReport>>,
}

/// A running signaling server.
pub struct SignalingServer {
    addr: SocketAddr,
    state: Arc<ServerState>,
    shutdown: Arc<AtomicBool>,
    accept_thread: Option<std::thread::JoinHandle<()>>,
}

impl SignalingServer {
    /// Binds a loopback port and starts accepting connections.
    pub fn start(supervisor: Arc<DesktopSupervisor>) -> Result<Self, DesktopError> {
        let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| {
            DesktopError::TransportUnavailable(format!("cannot bind signaling socket: {e}"))
        })?;
        let addr = listener
            .local_addr()
            .map_err(|e| DesktopError::TransportUnavailable(format!("local addr: {e}")))?;
        let state = Arc::new(ServerState {
            supervisor,
            grants: Mutex::new(HashMap::new()),
            reports: Mutex::new(HashMap::new()),
        });
        let shutdown = Arc::new(AtomicBool::new(false));

        let accept_state = state.clone();
        let accept_shutdown = shutdown.clone();
        let accept_thread = std::thread::Builder::new()
            .name("juto-signaling".to_string())
            .spawn(move || {
                for stream in listener.incoming() {
                    if accept_shutdown.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(stream) = stream else { continue };
                    let state = accept_state.clone();
                    std::thread::Builder::new()
                        .name("juto-signaling-conn".to_string())
                        .spawn(move || {
                            if let Err(err) = handle_connection(stream, state) {
                                tracing::debug!(error = %err, "signaling connection ended");
                            }
                        })
                        .ok();
                }
            })
            .map_err(|e| DesktopError::TransportUnavailable(format!("signaling thread: {e}")))?;

        Ok(Self {
            addr,
            state,
            shutdown,
            accept_thread: Some(accept_thread),
        })
    }

    /// The loopback address the server is listening on.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Issues a one-time attach token for an environment.
    pub fn issue_token(&self, env_id: &str, viewer_id: &str, roles: Vec<GrantRole>) -> AttachGrant {
        let token = format!("attach-{}", uuid::Uuid::now_v7());
        let grant = AttachGrant {
            token: token.clone(),
            env_id: env_id.to_string(),
            viewer_id: viewer_id.to_string(),
            roles,
            expires_at: now_secs() + TOKEN_TTL.as_secs(),
        };
        self.state.grants.lock().insert(token, grant.clone());
        grant
    }

    /// The viewer URL for a token (loopback only).
    pub fn viewer_url(&self, token: &str) -> String {
        format!("http://{}/?token={token}", self.addr)
    }

    /// The last report posted by a viewer, if any.
    pub fn report(&self, token: &str) -> Option<ViewerReport> {
        self.state.reports.lock().get(token).cloned()
    }

    /// Stops accepting connections.
    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.addr);
        if let Some(handle) = self.accept_thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for SignalingServer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.addr);
        if let Some(handle) = self.accept_thread.take() {
            let _ = handle.join();
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn handle_connection(mut stream: TcpStream, state: Arc<ServerState>) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }
    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }

    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let target = parts.next().unwrap_or_default().to_string();
    let (path, query) = match target.split_once('?') {
        Some((path, query)) => (path.to_string(), query.to_string()),
        None => (target.clone(), String::new()),
    };
    let token = query
        .split('&')
        .find_map(|pair| pair.strip_prefix("token="))
        .unwrap_or_default()
        .to_string();

    match (method.as_str(), path.as_str()) {
        ("GET", "/") => {
            let body = viewer_page(&token);
            write_http(
                &mut stream,
                200,
                "text/html; charset=utf-8",
                body.as_bytes(),
            )
        }
        ("POST", "/report") => {
            let mut length = 0usize;
            if let Some(value) = headers.get("content-length") {
                length = value.parse().unwrap_or(0);
            }
            let mut body = vec![0u8; length.min(64 * 1024)];
            reader.read_exact(&mut body)?;
            if let Ok(report) = serde_json::from_slice::<ViewerReport>(&body) {
                state.reports.lock().insert(token.clone(), report);
            }
            write_http(&mut stream, 200, "application/json", b"{\"ok\":true}")
        }
        ("GET", "/ws") => {
            let key = headers
                .get("sec-websocket-key")
                .cloned()
                .unwrap_or_default();
            websocket_session(stream, state, token, key)
        }
        _ => write_http(&mut stream, 404, "text/plain", b"not found"),
    }
}

fn write_http(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        401 => "Unauthorized",
        _ => "Not Found",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn websocket_session(
    mut stream: TcpStream,
    state: Arc<ServerState>,
    token: String,
    key: String,
) -> std::io::Result<()> {
    // Consume the one-time token.
    let grant = {
        let mut grants = state.grants.lock();
        match grants.remove(&token) {
            Some(grant) if grant.expires_at >= now_secs() => grant,
            _ => {
                return write_http(&mut stream, 401, "text/plain", b"invalid or expired token");
            }
        }
    };

    let response = format!(
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {}\r\n\r\n",
        ws_accept(&key)
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()?;

    let worker = match start_presentation(&state.supervisor, &grant.env_id, &grant.viewer_id) {
        Ok(worker) => worker,
        Err(err) => {
            let _ = ws_send_text(
                &mut stream,
                &encode_message(&SignalingMessage::Status {
                    state: "failed".to_string(),
                    detail: Some(err.to_string()),
                }),
            );
            return Ok(());
        }
    };

    let state_for_reports = state.clone();
    // Read frames on a dedicated thread so the relay loop can flush candidates.
    let (tx, rx) = std::sync::mpsc::channel::<WsFrame>();
    let mut read_stream = stream.try_clone()?;
    std::thread::Builder::new()
        .name("juto-signaling-read".to_string())
        .spawn(move || {
            let mut reader = BufReader::new(&mut read_stream);
            while let Ok(frame) = read_ws_frame(&mut reader) {
                if tx.send(frame).is_err() {
                    break;
                }
            }
        })
        .ok();

    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(WsFrame::Text(text)) => match serde_json::from_str::<SignalingMessage>(&text) {
                Ok(SignalingMessage::Offer { sdp }) => match worker.answer_to(&sdp) {
                    Ok(answer) => {
                        ws_send_text(
                            &mut stream,
                            &encode_message(&SignalingMessage::Answer { sdp: answer }),
                        )?;
                    }
                    Err(err) => {
                        let _ = ws_send_text(
                            &mut stream,
                            &encode_message(&SignalingMessage::Status {
                                state: "failed".to_string(),
                                detail: Some(err.to_string()),
                            }),
                        );
                        break;
                    }
                },
                Ok(SignalingMessage::Candidate { candidate }) => {
                    let _ = worker.add_ice_candidate(&candidate);
                }
                Ok(SignalingMessage::Status { state, detail }) if state == "report" => {
                    if let Some(detail) = detail
                        && let Ok(report) = serde_json::from_str::<ViewerReport>(&detail)
                    {
                        state_for_reports
                            .reports
                            .lock()
                            .insert(token.clone(), report);
                    }
                }
                _ => {}
            },
            Ok(WsFrame::Close) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            Ok(WsFrame::Other) | Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
        }

        for candidate in worker.take_local_candidates() {
            ws_send_text(
                &mut stream,
                &encode_message(&SignalingMessage::Candidate { candidate }),
            )?;
        }
        if let Some(error) = worker.error() {
            let _ = ws_send_text(
                &mut stream,
                &encode_message(&SignalingMessage::Status {
                    state: "failed".to_string(),
                    detail: Some(error),
                }),
            );
            break;
        }
    }

    worker.shutdown();
    Ok(())
}

enum WsFrame {
    Text(String),
    Close,
    Other,
}

fn ws_accept(key: &str) -> String {
    use sha1::{Digest, Sha1};
    let mut hasher = Sha1::new();
    hasher.update(key.as_bytes());
    hasher.update(b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    base64::engine::general_purpose::STANDARD.encode(hasher.finalize())
}

fn read_ws_frame(reader: &mut impl Read) -> std::io::Result<WsFrame> {
    let mut header = [0u8; 2];
    reader.read_exact(&mut header)?;
    let opcode = header[0] & 0x0F;
    let masked = header[1] & 0x80 != 0;
    let mut length = u64::from(header[1] & 0x7F);
    if length == 126 {
        let mut ext = [0u8; 2];
        reader.read_exact(&mut ext)?;
        length = u64::from(u16::from_be_bytes(ext));
    } else if length == 127 {
        let mut ext = [0u8; 8];
        reader.read_exact(&mut ext)?;
        length = u64::from_be_bytes(ext);
    }
    if length > 1 << 20 {
        return Err(std::io::Error::other("websocket frame too large"));
    }
    let mut mask = [0u8; 4];
    if masked {
        reader.read_exact(&mut mask)?;
    }
    let mut payload = vec![0u8; length as usize];
    reader.read_exact(&mut payload)?;
    if masked {
        for (index, byte) in payload.iter_mut().enumerate() {
            *byte ^= mask[index % 4];
        }
    }
    Ok(match opcode {
        0x1 => WsFrame::Text(String::from_utf8_lossy(&payload).into_owned()),
        0x8 => WsFrame::Close,
        _ => WsFrame::Other,
    })
}

fn ws_send_text(stream: &mut TcpStream, text: &str) -> std::io::Result<()> {
    let payload = text.as_bytes();
    let mut frame = Vec::with_capacity(payload.len() + 10);
    frame.push(0x81);
    if payload.len() < 126 {
        frame.push(payload.len() as u8);
    } else if payload.len() <= u16::MAX as usize {
        frame.push(126);
        frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    } else {
        frame.push(127);
        frame.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    }
    frame.extend_from_slice(payload);
    stream.write_all(&frame)?;
    stream.flush()
}

fn viewer_page(token: &str) -> String {
    VIEWER_PAGE.replace("__TOKEN__", token)
}

const VIEWER_PAGE: &str = r#"<!doctype html>
<html><head><meta charset="utf-8"><title>Juto desktop</title>
<style>
body{margin:0;background:#0d0f17;color:#e8eaed;font:14px system-ui,sans-serif}
#wrap{display:flex;flex-direction:column;height:100vh}
#bar{padding:8px 12px;display:flex;gap:12px;align-items:center}
#stage{flex:1;display:flex;align-items:center;justify-content:center;background:#000}
video{max-width:100%;max-height:100%}
#status{color:#9aa0a6;font-size:12px}
</style></head>
<body><div id="wrap">
<div id="bar"><strong>Juto desktop</strong><span id="status">connecting…</span></div>
<div id="stage"><video id="v" autoplay playsinline muted></video></div>
</div>
<script>
const TOKEN = "__TOKEN__";
const statusEl = document.getElementById('status');
const video = document.getElementById('v');
let pc, dc, ws, desktop = {width:0,height:0}, inputSent = false;

function report(extra) {
  const payload = Object.assign({
    connected: pc && pc.connectionState === 'connected',
    videoWidth: video.videoWidth, videoHeight: video.videoHeight,
    sampledPixel: samplePixel(), inputSent: inputSent, error: null
  }, extra || {});
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({type:'status', state:'report', detail: JSON.stringify(payload)}));
  }
}

function samplePixel() {
  if (!video.videoWidth) return null;
  const c = document.createElement('canvas');
  c.width = video.videoWidth; c.height = video.videoHeight;
  const ctx = c.getContext('2d');
  ctx.drawImage(video, 0, 0);
  const d = ctx.getImageData(10, 10, 1, 1).data;
  return [d[0], d[1], d[2], d[3]];
}

function send(events) {
  if (dc && dc.readyState === 'open') { dc.send(JSON.stringify(events)); inputSent = true; }
}

async function start() {
  ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws?token=' + encodeURIComponent(TOKEN));
  pc = new RTCPeerConnection({iceServers: []});
  dc = pc.createDataChannel('input', {ordered: true});
  dc.onopen = () => { statusEl.textContent = 'input channel open'; };
  pc.addTransceiver('video', {direction: 'recvonly'});
  pc.ontrack = (ev) => { video.srcObject = ev.streams[0]; };
  pc.onicecandidate = (ev) => { if (ev.candidate) ws.send(JSON.stringify({type:'candidate', candidate: ev.candidate.candidate})); };
  pc.onconnectionstatechange = () => { statusEl.textContent = pc.connectionState; report(); };

  ws.onmessage = async (ev) => {
    const msg = JSON.parse(ev.data);
    statusEl.textContent = 'msg: ' + msg.type;
    if (msg.type === 'answer') {
      await pc.setRemoteDescription({type:'answer', sdp: msg.sdp});
    } else if (msg.type === 'candidate') {
      try { await pc.addIceCandidate({candidate: msg.candidate}); } catch (e) {}
    } else if (msg.type === 'status') {
      statusEl.textContent = msg.state + (msg.detail ? ' — ' + msg.detail : '');
    }
  };
  ws.onopen = async () => {
    statusEl.textContent = 'signaling open';
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    ws.send(JSON.stringify({type:'offer', sdp: offer.sdp}));
    report();
  };
  ws.onerror = () => { statusEl.textContent = 'signaling error'; report({error:'signaling error'}); };

  report();
  video.addEventListener('loadedmetadata', () => {
    desktop = {width: video.videoWidth, height: video.videoHeight};
    statusEl.textContent = 'live ' + desktop.width + 'x' + desktop.height;
    send([{kind:'pointerMoveAbs', x: 10, y: 10}]);
    report();
  });

  const toDesktop = (ev) => {
    const r = video.getBoundingClientRect();
    const x = Math.round((ev.clientX - r.left) / r.width * desktop.width);
    const y = Math.round((ev.clientY - r.top) / r.height * desktop.height);
    return {x: Math.max(0, Math.min(desktop.width - 1, x)), y: Math.max(0, Math.min(desktop.height - 1, y))};
  };
  video.addEventListener('mousemove', (ev) => { const p = toDesktop(ev); send([{kind:'pointerMoveAbs', x:p.x, y:p.y}]); });
  video.addEventListener('mousedown', (ev) => { const p = toDesktop(ev); send([{kind:'pointerMoveAbs', x:p.x, y:p.y},{kind:'pointerButton', button:272, pressed:true}]); });
  video.addEventListener('mouseup', (ev) => { const p = toDesktop(ev); send([{kind:'pointerMoveAbs', x:p.x, y:p.y},{kind:'pointerButton', button:272, pressed:false}]); });
  video.addEventListener('wheel', (ev) => { send([{kind:'pointerScroll', axis:1, value: ev.deltaY > 0 ? -1 : 1}]); ev.preventDefault(); }, {passive:false});
  window.addEventListener('keydown', (ev) => { const k = keycode(ev.code); if (k) send([{kind:'key', keycode:k, pressed:true}]); });
  window.addEventListener('keyup', (ev) => { const k = keycode(ev.code); if (k) send([{kind:'key', keycode:k, pressed:false}]); });
}

const KEYS = {KeyA:30,KeyB:48,KeyC:46,KeyD:32,KeyE:18,KeyF:33,KeyG:34,KeyH:35,KeyI:23,KeyJ:36,KeyK:37,KeyL:38,KeyM:50,KeyN:49,KeyO:24,KeyP:25,KeyQ:16,KeyR:19,KeyS:31,KeyT:20,KeyU:22,KeyV:47,KeyW:17,KeyX:45,KeyY:21,KeyZ:44,Digit1:2,Digit2:3,Digit3:4,Digit4:5,Digit5:6,Digit6:7,Digit7:8,Digit8:9,Digit9:10,Digit0:11,Enter:28,Escape:1,Space:57,Tab:15,Backspace:14,ArrowUp:103,ArrowDown:108,ArrowLeft:105,ArrowRight:106,ShiftLeft:42,ShiftRight:54,ControlLeft:29,ControlRight:97,AltLeft:56,AltRight:100};
function keycode(code) { return KEYS[code] || 0; }

start().catch((e) => { statusEl.textContent = 'error: ' + e; report({error: String(e)}); });
</script></body></html>
"#;

/// Builds the default remote stream configuration for an environment.
pub fn default_stream_config(env_id: &str, viewer_id: &str) -> RemoteStreamConfig {
    RemoteStreamConfig {
        env_id: env_id.to_string(),
        viewer_id: viewer_id.to_string(),
        video_codec: VideoCodec::Vp8,
        max_bitrate_bps: 4_000_000,
        max_fps: 30,
        negotiation: crate::remote::Negotiation::ViewerOffers,
        ice_servers: Vec::new(),
        audio_source: None,
    }
}

/// Creates a presentation worker for an environment.
pub fn start_presentation(
    supervisor: &DesktopSupervisor,
    env_id: &str,
    viewer_id: &str,
) -> Result<PresentationWorker, DesktopError> {
    let env = supervisor.get_environment(env_id)?;
    PresentationWorker::new(env, default_stream_config(env_id, viewer_id))
}

/// Encodes a signaling message for the wire.
pub fn encode_message(message: &SignalingMessage) -> String {
    serde_json::to_string(message).unwrap_or_default()
}
