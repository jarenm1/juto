use std::sync::Arc;

use juto_ai::AuthError;
use tokio_util::sync::CancellationToken;

use super::run_blocking;
use crate::protocol::{CredentialStatus, ErrorCode, LoginEvent, Reply, RpcError, ServerEvent};
use crate::server::Daemon;

fn failed(err: impl std::fmt::Display) -> RpcError {
    RpcError::new(ErrorCode::Failed, err.to_string())
}

pub(crate) async fn credential_status(
    d: &Daemon,
    providers: Vec<String>,
) -> Result<Reply, RpcError> {
    let credentials = d.runtime.credentials();
    let status = run_blocking(move || -> Result<CredentialStatus, AuthError> {
        let summaries = credentials.list()?;
        let available = credentials
            .available_providers(providers.iter().map(String::as_str))?
            .into_iter()
            .collect();
        Ok(CredentialStatus {
            summaries,
            available,
        })
    })
    .await?
    .map_err(failed)?;
    Ok(Reply::Credentials(status))
}

/// The key is written to the store and dropped; it never reaches a reply, event, or log.
pub(crate) async fn set_api_key(
    d: &Daemon,
    provider: String,
    key: String,
) -> Result<Reply, RpcError> {
    let credentials = d.runtime.credentials();
    run_blocking(move || credentials.set_api_key(&provider, &key))
        .await?
        .map_err(failed)?;
    Ok(Reply::Ok)
}

pub(crate) async fn remove_credential(d: &Daemon, provider: String) -> Result<Reply, RpcError> {
    let credentials = d.runtime.credentials();
    run_blocking(move || credentials.remove(&provider))
        .await?
        .map_err(failed)?;
    Ok(Reply::Ok)
}

/// Starts an OAuth flow in a detached task. The authorization URL and the
/// completion arrive as `ServerEvent::Login` events for the returned `login_id`.
pub(crate) fn login_start(d: &Arc<Daemon>, provider: String) -> Result<Reply, RpcError> {
    let login_id = d.next_id();
    let cancel = CancellationToken::new();
    d.logins
        .lock()
        .expect("login registry poisoned")
        .insert(login_id, cancel.clone());

    let credentials = d.runtime.credentials();
    let daemon = Arc::clone(d);
    tokio::spawn(async move {
        let url_daemon = Arc::clone(&daemon);
        let on_url: Arc<dyn Fn(&str) + Send + Sync> = Arc::new(move |url: &str| {
            url_daemon.emit(ServerEvent::Login {
                event: LoginEvent::Url {
                    login_id,
                    url: url.to_string(),
                },
            });
        });
        let result = credentials.login(&provider, cancel, on_url).await;
        daemon
            .logins
            .lock()
            .expect("login registry poisoned")
            .remove(&login_id);
        daemon.emit(ServerEvent::Login {
            event: LoginEvent::Finished {
                login_id,
                error: result.err().map(|err| err.to_string()),
            },
        });
    });
    Ok(Reply::LoginStarted { login_id })
}

pub(crate) fn login_cancel(d: &Daemon, login_id: u64) -> Result<Reply, RpcError> {
    let cancel = d
        .logins
        .lock()
        .expect("login registry poisoned")
        .remove(&login_id)
        .ok_or_else(|| {
            RpcError::new(ErrorCode::NotFound, format!("no login with id {login_id}"))
        })?;
    cancel.cancel();
    Ok(Reply::Ok)
}
