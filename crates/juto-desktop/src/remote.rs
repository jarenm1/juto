use serde::{Deserialize, Serialize};

/// Session description for WebRTC signaling (SDP offer/answer).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebRtcSessionDescription {
    pub sdp_type: String, // "offer" or "answer"
    pub sdp: String,
}

/// An ICE candidate trickled during WebRTC connection negotiation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebRtcCandidate {
    pub candidate: String,
    pub sdp_mid: Option<String>,
    pub sdp_m_line_index: Option<u32>,
}

/// Configuration parameters for an authenticated remote WebRTC streaming session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteStreamConfig {
    pub session_id: String,
    pub viewer_id: String,
    pub video_codec: String, // "H264" or "VP8"
    pub audio_codec: String, // "Opus"
    pub max_bitrate_bps: u64,
    pub max_fps: u32,
    pub reliable_data_channel: bool,
    pub stun_turn_servers: Vec<String>,
}

impl Default for RemoteStreamConfig {
    fn default() -> Self {
        Self {
            session_id: uuid::Uuid::now_v7().to_string(),
            viewer_id: "default-viewer".to_string(),
            video_codec: "H264".to_string(),
            audio_codec: "Opus".to_string(),
            max_bitrate_bps: 12_000_000,
            max_fps: 60,
            reliable_data_channel: true,
            stun_turn_servers: vec!["stun:stun.l.google.com:19302".to_string()],
        }
    }
}

/// State tracking for an active WebRTC streaming session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteStreamSession {
    pub config: RemoteStreamConfig,
    pub local_description: Option<WebRtcSessionDescription>,
    pub remote_description: Option<WebRtcSessionDescription>,
    pub ice_candidates: Vec<WebRtcCandidate>,
    pub is_connected: bool,
}

impl RemoteStreamSession {
    pub fn new(config: RemoteStreamConfig) -> Self {
        Self {
            config,
            local_description: None,
            remote_description: None,
            ice_candidates: Vec::new(),
            is_connected: false,
        }
    }

    pub fn set_local_offer(&mut self, sdp: String) {
        self.local_description = Some(WebRtcSessionDescription {
            sdp_type: "offer".to_string(),
            sdp,
        });
    }

    pub fn set_remote_answer(&mut self, sdp: String) {
        self.remote_description = Some(WebRtcSessionDescription {
            sdp_type: "answer".to_string(),
            sdp,
        });
        self.is_connected = true;
    }

    pub fn add_ice_candidate(&mut self, candidate: WebRtcCandidate) {
        self.ice_candidates.push(candidate);
    }
}
