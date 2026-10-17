//! Chrome DevTools Protocol client over one browser WebSocket.
//!
//! Commands are multiplexed by id; flattened target sessions (`sessionId`)
//! share the socket. Events fan out on a broadcast channel. When the socket
//! closes every pending command fails and [`Cdp::closed`] is cancelled.

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use futures::{SinkExt as _, StreamExt as _};
use parking_lot::Mutex;
use serde_json::{Value, json};
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio_tungstenite::tungstenite::{Message, client::IntoClientRequest as _};
use tokio_util::sync::CancellationToken;

/// One protocol event; `session` is `None` for browser-level events.
#[derive(Debug, Clone)]
pub(crate) struct CdpEvent {
    pub session: Option<String>,
    pub method: String,
    pub params: Value,
}

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>>;

pub(crate) struct Cdp {
    outgoing: mpsc::UnboundedSender<Message>,
    pending: Pending,
    next_id: AtomicU64,
    events: broadcast::Sender<CdpEvent>,
    closed: CancellationToken,
}

/// Events retained for slow subscribers before they observe a lag.
const EVENT_CAPACITY: usize = 8192;

impl Cdp {
    /// Connect to a `ws://` DevTools endpoint.
    pub(crate) async fn connect(ws_url: &str) -> Result<Arc<Self>, String> {
        let mut request = ws_url
            .into_client_request()
            .map_err(|e| format!("invalid DevTools URL {ws_url}: {e}"))?;
        // Chromium rejects cross-origin DevTools clients unless Origin is absent.
        request.headers_mut().remove("Origin");
        let config = tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default()
            .max_message_size(Some(256 << 20))
            .max_frame_size(Some(256 << 20));
        let (socket, _) =
            tokio_tungstenite::connect_async_with_config(request, Some(config), false)
                .await
                .map_err(|e| format!("cannot connect to DevTools at {ws_url}: {e}"))?;
        let (mut sink, mut stream) = socket.split();
        let (outgoing, mut outgoing_rx) = mpsc::unbounded_channel::<Message>();
        let pending: Pending = Arc::default();
        let (events, _) = broadcast::channel(EVENT_CAPACITY);
        let closed = CancellationToken::new();

        let writer_closed = closed.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = writer_closed.cancelled() => break,
                    message = outgoing_rx.recv() => match message {
                        Some(message) => {
                            if sink.send(message).await.is_err() {
                                writer_closed.cancel();
                                break;
                            }
                        }
                        None => break,
                    },
                }
            }
            let _ = sink.close().await;
        });

        let reader_pending = pending.clone();
        let reader_events = events.clone();
        let reader_closed = closed.clone();
        tokio::spawn(async move {
            loop {
                let frame = tokio::select! {
                    _ = reader_closed.cancelled() => break,
                    frame = stream.next() => frame,
                };
                let text = match frame {
                    Some(Ok(Message::Text(text))) => text,
                    Some(Ok(Message::Binary(bytes))) => match String::from_utf8(bytes.to_vec()) {
                        Ok(text) => text.into(),
                        Err(_) => continue,
                    },
                    Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                    Some(Ok(_)) => continue,
                };
                let Ok(mut value) = serde_json::from_str::<Value>(&text) else {
                    continue;
                };
                if let Some(id) = value.get("id").and_then(Value::as_u64) {
                    let Some(reply) = reader_pending.lock().remove(&id) else {
                        continue;
                    };
                    let result = match value.get_mut("error") {
                        Some(error) => Err(error
                            .get("message")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                            .unwrap_or_else(|| error.to_string())),
                        None => Ok(value
                            .get_mut("result")
                            .map(Value::take)
                            .unwrap_or(Value::Null)),
                    };
                    let _ = reply.send(result);
                } else if let Some(method) = value.get("method").and_then(Value::as_str) {
                    let _ = reader_events.send(CdpEvent {
                        session: value
                            .get("sessionId")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        method: method.to_string(),
                        params: value
                            .get_mut("params")
                            .map(Value::take)
                            .unwrap_or(Value::Null),
                    });
                }
            }
            reader_closed.cancel();
            for (_, reply) in reader_pending.lock().drain() {
                let _ = reply.send(Err("browser connection closed".to_string()));
            }
        });

        Ok(Arc::new(Self {
            outgoing,
            pending,
            next_id: AtomicU64::new(1),
            events,
            closed,
        }))
    }

    /// Send one command and await its result.
    pub(crate) async fn send(
        &self,
        session: Option<&str>,
        method: &str,
        params: Value,
    ) -> Result<Value, String> {
        if self.closed.is_cancelled() {
            return Err("browser connection closed".to_string());
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut message = json!({ "id": id, "method": method, "params": params });
        if let Some(session) = session {
            message["sessionId"] = Value::String(session.to_string());
        }
        let (reply, response) = oneshot::channel();
        self.pending.lock().insert(id, reply);
        if self
            .outgoing
            .send(Message::Text(message.to_string().into()))
            .is_err()
        {
            self.pending.lock().remove(&id);
            return Err("browser connection closed".to_string());
        }
        let result = tokio::select! {
            result = response => result.unwrap_or_else(|_| Err("browser connection closed".to_string())),
            _ = self.closed.cancelled() => Err("browser connection closed".to_string()),
        };
        if result.is_err() {
            self.pending.lock().remove(&id);
        }
        result.map_err(|e| format!("{method}: {e}"))
    }

    pub(crate) fn subscribe(&self) -> broadcast::Receiver<CdpEvent> {
        self.events.subscribe()
    }

    pub(crate) fn closed(&self) -> &CancellationToken {
        &self.closed
    }

    /// Drop the socket; pending commands fail.
    pub(crate) fn close(&self) {
        self.closed.cancel();
    }
}

impl Drop for Cdp {
    fn drop(&mut self) {
        self.closed.cancel();
    }
}
