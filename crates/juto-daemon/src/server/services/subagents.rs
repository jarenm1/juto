use juto_runtime::SubagentError;

use super::run_blocking;
use crate::protocol::{ErrorCode, Reply, RpcError, SubagentCommand};
use crate::server::Daemon;

pub(crate) fn list_subagents(d: &Daemon) -> Result<Reply, RpcError> {
    Ok(Reply::Subagents(d.runtime.subagent_manager().list()))
}

/// Reads the child's session file off the async workers.
pub(crate) async fn subagent_transcript(d: &Daemon, id: String) -> Result<Reply, RpcError> {
    let manager = d.runtime.subagent_manager();
    run_blocking(move || manager.transcript(&id))
        .await?
        .map(Reply::Transcript)
        .map_err(subagent_error)
}

/// `Send` is a user-typed message: it steers a live child, answers a blocked
/// one, or starts a user follow-up on a completed one. Each promotes the child
/// to a focus thread.
pub(crate) async fn subagent_command(
    d: &Daemon,
    id: String,
    command: SubagentCommand,
) -> Result<Reply, RpcError> {
    let manager = d.runtime.subagent_manager();
    let result = match command {
        SubagentCommand::Pause => manager.pause(&id),
        SubagentCommand::Resume => manager.resume(&id),
        SubagentCommand::Cancel => manager.cancel(&id),
        SubagentCommand::Unblock { resolution } => manager.unblock(&id, resolution),
        SubagentCommand::Release => manager.release(&id).await,
        SubagentCommand::Send { text } => manager.send_from_user(&id, text),
        SubagentCommand::SetArchived { archived } => manager.set_archived(&id, archived),
    };
    result.map_err(subagent_error)?;
    Ok(Reply::Ok)
}

fn subagent_error(err: SubagentError) -> RpcError {
    let code = match err {
        SubagentError::Unknown(_) => ErrorCode::NotFound,
        SubagentError::NoRuntime
        | SubagentError::Timeout(_)
        | SubagentError::Transcript(_)
        | SubagentError::WorkspaceRelease(_) => ErrorCode::Failed,
        _ => ErrorCode::Invalid,
    };
    RpcError::new(code, err.to_string())
}
