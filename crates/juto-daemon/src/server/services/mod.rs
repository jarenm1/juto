//! Daemon-side handlers for configuration, models, subagents, and credentials.

use crate::protocol::{ErrorCode, RpcError};

pub(crate) mod config;
pub(crate) mod credentials;
pub(crate) mod subagents;

pub(crate) use config::{get_config, list_models, set_model_role};
pub(crate) use credentials::{
    credential_status, login_cancel, login_start, remove_credential, set_api_key,
};
pub(crate) use subagents::{list_subagents, subagent_command, subagent_transcript};

/// Runs blocking file work off the async worker threads.
async fn run_blocking<T: Send + 'static>(
    work: impl FnOnce() -> T + Send + 'static,
) -> Result<T, RpcError> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|err| RpcError::new(ErrorCode::Failed, format!("blocking task failed: {err}")))
}
