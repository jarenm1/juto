//! GitHub event watches: the allowlist, the event ledger, and the poll loop.

use std::sync::Arc;
use std::time::Duration;

use juto_runtime::github::{GithubError, GithubWatchSpec};

use crate::protocol::{ErrorCode, Reply, RpcError};
use crate::server::Daemon;

/// The GitHub REST API root used unless `JUTO_GITHUB_API_BASE` overrides it.
const DEFAULT_API_BASE: &str = "https://api.github.com";

/// How often every watch is polled. A repeat poll starts nothing, so the interval
/// only bounds how late an event is noticed.
const POLL_INTERVAL: Duration = Duration::from_secs(60);

/// Polls every GitHub watch on a fixed interval until the daemon stops. Per-watch
/// failures are logged; a failed poll leaves each cursor where it was, so the next
/// poll retries the same items. Nothing logged contains the token.
pub(crate) async fn poll_loop(daemon: Arc<Daemon>) {
    let mut interval = tokio::time::interval(POLL_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        interval.tick().await;
        let _permit = daemon.admission.read().await;
        if daemon.draining.load(std::sync::atomic::Ordering::Acquire) {
            continue;
        }
        match daemon.runtime.poll_github(&api_base()).await {
            Ok(report) => {
                for error in report.errors {
                    eprintln!("juto-daemon: GitHub poll: {error}");
                }
            }
            Err(error) => eprintln!("juto-daemon: GitHub poll failed: {error}"),
        }
    }
}

/// The API root this daemon polls. `JUTO_GITHUB_API_BASE` lets a local fixture stand
/// in for GitHub; it is read on every call so the poll loop and requests agree.
pub(crate) fn api_base() -> String {
    match std::env::var("JUTO_GITHUB_API_BASE") {
        Ok(base) if !base.trim().is_empty() => base,
        _ => DEFAULT_API_BASE.to_owned(),
    }
}

pub(crate) fn create(d: &Daemon, spec: GithubWatchSpec) -> Result<Reply, RpcError> {
    d.runtime
        .create_github_watch(spec)
        .map(Reply::GithubWatch)
        .map_err(github_error)
}

pub(crate) fn list(d: &Daemon) -> Result<Reply, RpcError> {
    d.runtime
        .list_github_watches()
        .map(Reply::GithubWatches)
        .map_err(github_error)
}

pub(crate) fn delete(d: &Daemon, id: &str) -> Result<Reply, RpcError> {
    d.runtime
        .delete_github_watch(id)
        .map(|()| Reply::Ok)
        .map_err(github_error)
}

pub(crate) fn events(d: &Daemon) -> Result<Reply, RpcError> {
    d.runtime
        .github_events()
        .map(Reply::GithubEvents)
        .map_err(github_error)
}

pub(crate) async fn poll(d: &Daemon) -> Result<Reply, RpcError> {
    d.runtime
        .poll_github(&api_base())
        .await
        .map(Reply::GithubPoll)
        .map_err(github_error)
}

fn github_error(err: GithubError) -> RpcError {
    let code = match err {
        GithubError::NotFound(_) => ErrorCode::NotFound,
        GithubError::Invalid(_) => ErrorCode::Invalid,
        // The protocol has no separate auth code; the message says what to do.
        GithubError::Auth(_) | GithubError::Failed(_) => ErrorCode::Failed,
    };
    RpcError::new(code, err.to_string())
}
