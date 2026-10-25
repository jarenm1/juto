//! Cooperative replacement: fence new work, let continuations finish, then exit.

use std::sync::Arc;

use super::*;

impl Daemon {
    /// The write permit excludes requests, scheduler ticks, and idle wakes while
    /// deciding to exit. Run completion itself never needs this permit.
    pub(super) async fn upgrade_ready(self: &Arc<Self>) -> Result<bool> {
        if !self.draining.load(Ordering::Acquire) {
            return Ok(false);
        }
        let _permit = self.admission.write().await;
        if self.state.lock().unwrap().active.is_some()
            || !self.logins.lock().unwrap().is_empty()
            || self.runtime.subagent_manager().has_active_children()
            || self
                .runtime
                .subagent_manager()
                .main_inbox()
                .has_pending_reports()
            || self.runtime.has_active_jobs()
        {
            return Ok(false);
        }
        // A completed child may have queued the next main turn. Start that turn
        // rather than exiting between child completion and delivery of its report.
        let ready = matches!(self.wake_idle()?, Wake::Idle);
        if ready {
            self.retiring.store(true, Ordering::Release);
        }
        Ok(ready)
    }
}
