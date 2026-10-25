use std::sync::Arc;
use std::time::Duration;

use juto_runtime::automations::{AutomationError, AutomationSpec};

use crate::protocol::{ErrorCode, Reply, RpcError};
use crate::server::Daemon;

/// How often due occurrences are checked. An occurrence fires within one tick of
/// its instant, and a tick never starts more than one job per automation.
const TICK: Duration = Duration::from_secs(1);

/// Fires due automation occurrences until the daemon stops. The clock is read once
/// per tick; `tick_automations` starts nothing twice for the same instant, so a
/// tick that lands late or repeats is harmless.
pub(crate) async fn tick_loop(daemon: Arc<Daemon>) {
    let mut interval = tokio::time::interval(TICK);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        interval.tick().await;
        let _permit = daemon.admission.read().await;
        if daemon.draining.load(std::sync::atomic::Ordering::Acquire) {
            continue;
        }
        if let Err(error) = daemon.runtime.tick_automations(juto_ai::timestamp_ms()) {
            eprintln!("juto-daemon: automation tick failed: {error}");
        }
    }
}

pub(crate) fn create(d: &Daemon, spec: AutomationSpec) -> Result<Reply, RpcError> {
    d.runtime
        .create_automation(spec)
        .map(Reply::Automation)
        .map_err(automation_error)
}

pub(crate) fn list(d: &Daemon) -> Result<Reply, RpcError> {
    d.runtime
        .list_automations()
        .map(Reply::Automations)
        .map_err(automation_error)
}

pub(crate) fn get(d: &Daemon, id: &str) -> Result<Reply, RpcError> {
    d.runtime
        .automation(id)
        .map(Reply::Automation)
        .map_err(automation_error)
}

pub(crate) fn history(d: &Daemon, id: &str) -> Result<Reply, RpcError> {
    d.runtime
        .automation_history(id)
        .map(Reply::AutomationHistory)
        .map_err(automation_error)
}

pub(crate) fn run_now(d: &Daemon, id: &str) -> Result<Reply, RpcError> {
    d.runtime
        .run_automation_now(id)
        .map(Reply::Job)
        .map_err(automation_error)
}

pub(crate) fn set_enabled(d: &Daemon, id: &str, enabled: bool) -> Result<Reply, RpcError> {
    d.runtime
        .set_automation_enabled(id, enabled)
        .map(Reply::Automation)
        .map_err(automation_error)
}

pub(crate) fn delete(d: &Daemon, id: &str) -> Result<Reply, RpcError> {
    d.runtime
        .delete_automation(id)
        .map(|()| Reply::Ok)
        .map_err(automation_error)
}

fn automation_error(err: AutomationError) -> RpcError {
    let code = match err {
        AutomationError::NotFound(_) => ErrorCode::NotFound,
        AutomationError::Invalid(_) => ErrorCode::Invalid,
        AutomationError::Failed(_) => ErrorCode::Failed,
    };
    RpcError::new(code, err.to_string())
}
