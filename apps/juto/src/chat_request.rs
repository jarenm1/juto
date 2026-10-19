//! Runs one daemon request on the Tokio runtime so GPUI's executor can await it.

use juto_daemon::client::{Client, ClientError};
use juto_daemon::protocol::{Reply, Request};

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
