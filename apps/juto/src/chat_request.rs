//! Runs one daemon request on the Tokio runtime so GPUI's executor can await it.

use juto_daemon::client::{Client, ClientError};
use juto_daemon::protocol::{ErrorCode, Reply, Request, RpcError};

/// Sends `request` through `client` on `handle` and returns the daemon's answer.
pub async fn call(
    client: Client,
    handle: tokio::runtime::Handle,
    request: Request,
) -> Result<Reply, ClientError> {
    match handle
        .spawn(async move { client.request(request).await })
        .await
    {
        Ok(result) => result,
        Err(err) if err.is_panic() => std::panic::resume_unwind(err.into_panic()),
        Err(_) => Err(ClientError::Disconnected),
    }
}

/// Readable text for a failed daemon request. A headline names the kind of
/// failure, then the daemon's message follows verbatim, so ids and details
/// needed for debugging stay in the text.
pub fn describe_error(error: &ClientError) -> String {
    match error {
        ClientError::Rpc(RpcError { code, message }) => {
            format!("{}: {message}", error_headline(*code))
        }
        ClientError::Disconnected => "Lost the connection to the daemon".to_owned(),
        ClientError::Start(message) => message.clone(),
    }
}

fn error_headline(code: ErrorCode) -> &'static str {
    match code {
        ErrorCode::ProtocolMismatch => "Protocol mismatch",
        ErrorCode::NotFound => "Not found",
        ErrorCode::Busy => "Busy",
        ErrorCode::Invalid => "Invalid request",
        ErrorCode::Failed => "Failed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rpc_error_keeps_detail_under_readable_headline() {
        let error = ClientError::Rpc(RpcError::new(
            ErrorCode::NotFound,
            "unknown subagent id: 0195-abc",
        ));

        assert_eq!(
            describe_error(&error),
            "Not found: unknown subagent id: 0195-abc"
        );
    }

    #[test]
    fn disconnect_has_plain_sentence() {
        assert_eq!(
            describe_error(&ClientError::Disconnected),
            "Lost the connection to the daemon"
        );
    }
}
