//! Channel-backed approval: the agent loop asks, a host answers.
//!
//! The runtime never blocks on UI. `ChannelApprovalHandler` forwards each
//! Write/Exec tool call as an [`ApprovalRequest`] over an unbounded channel and
//! waits for the host to call [`ApprovalRequest::respond`]. Cancelling the run
//! or dropping the request without answering resolves the wait as an error, so
//! the tool call is rejected rather than left hanging.

use std::sync::Arc;

use async_trait::async_trait;
use juto_agent::ToolTier;
use juto_ai::ToolCall;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::ApprovalHandler;

/// One pending permission question for a tool call.
pub struct ApprovalRequest {
    call: ToolCall,
    tier: ToolTier,
    respond: oneshot::Sender<bool>,
}

impl ApprovalRequest {
    pub fn call(&self) -> &ToolCall {
        &self.call
    }

    pub fn tier(&self) -> ToolTier {
        self.tier
    }

    /// Whether the waiting tool call has already given up (run cancelled).
    /// Hosts use this to skip requests nobody is waiting on any more.
    pub fn is_abandoned(&self) -> bool {
        self.respond.is_closed()
    }

    /// Answer the request. `true` allows the call; `false` denies it. Answering
    /// a request that was already abandoned is a no-op.
    pub fn respond(self, allow: bool) {
        let _ = self.respond.send(allow);
    }
}

/// Forwards approval questions to a host over an unbounded channel.
pub struct ChannelApprovalHandler {
    requests: mpsc::UnboundedSender<ApprovalRequest>,
}

impl ChannelApprovalHandler {
    /// Returns the handler to install in the runtime and the receiver the host
    /// drains to show requests.
    pub fn channel() -> (
        Arc<dyn ApprovalHandler>,
        mpsc::UnboundedReceiver<ApprovalRequest>,
    ) {
        let (requests, receiver) = mpsc::unbounded_channel();
        let handler: Arc<dyn ApprovalHandler> = Arc::new(Self { requests });
        (handler, receiver)
    }
}

#[async_trait]
impl ApprovalHandler for ChannelApprovalHandler {
    async fn approve(
        &self,
        call: &ToolCall,
        tier: ToolTier,
        cancel: CancellationToken,
    ) -> Result<bool, String> {
        let (respond, decision) = oneshot::channel();
        let request = ApprovalRequest {
            call: call.clone(),
            tier,
            respond,
        };
        self.requests
            .send(request)
            .map_err(|_| "no approval host is listening".to_owned())?;
        tokio::select! {
            biased;
            _ = cancel.cancelled() => Err("run cancelled while waiting for approval".to_owned()),
            decision = decision => decision
                .map_err(|_| "approval request was discarded without a decision".to_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn bash_call() -> ToolCall {
        ToolCall {
            id: "call-1".into(),
            name: "bash".into(),
            arguments: json!({"command": "rm -rf target"}),
            thought_signature: None,
        }
    }

    #[tokio::test]
    async fn host_allow_resolves_approval_true() {
        let (handler, mut requests) = ChannelApprovalHandler::channel();
        let waiter = tokio::spawn({
            let handler = handler.clone();
            async move {
                handler
                    .approve(&bash_call(), ToolTier::Exec, CancellationToken::new())
                    .await
            }
        });
        let request = requests.recv().await.expect("request forwarded");
        assert_eq!(request.call(), &bash_call());
        assert_eq!(request.tier(), ToolTier::Exec);
        request.respond(true);
        assert_eq!(waiter.await.unwrap(), Ok(true));
    }

    #[tokio::test]
    async fn host_deny_resolves_approval_false() {
        let (handler, mut requests) = ChannelApprovalHandler::channel();
        let waiter = tokio::spawn({
            let handler = handler.clone();
            async move {
                handler
                    .approve(&bash_call(), ToolTier::Exec, CancellationToken::new())
                    .await
            }
        });
        requests
            .recv()
            .await
            .expect("request forwarded")
            .respond(false);
        assert_eq!(waiter.await.unwrap(), Ok(false));
    }

    #[tokio::test]
    async fn cancelling_the_run_fails_a_pending_approval() {
        let (handler, mut requests) = ChannelApprovalHandler::channel();
        let cancel = CancellationToken::new();
        let waiter = tokio::spawn({
            let handler = handler.clone();
            let cancel = cancel.clone();
            async move { handler.approve(&bash_call(), ToolTier::Exec, cancel).await }
        });
        let request = requests.recv().await.expect("request forwarded");
        cancel.cancel();
        assert!(waiter.await.unwrap().is_err());
        assert!(request.is_abandoned());
    }

    #[tokio::test]
    async fn dropping_the_request_without_answer_is_an_error() {
        let (handler, mut requests) = ChannelApprovalHandler::channel();
        let waiter = tokio::spawn({
            let handler = handler.clone();
            async move {
                handler
                    .approve(&bash_call(), ToolTier::Write, CancellationToken::new())
                    .await
            }
        });
        drop(requests.recv().await.expect("request forwarded"));
        assert!(waiter.await.unwrap().is_err());
    }

    #[tokio::test]
    async fn approval_fails_when_no_host_is_listening() {
        let (handler, requests) = ChannelApprovalHandler::channel();
        drop(requests);
        let result = handler
            .approve(&bash_call(), ToolTier::Exec, CancellationToken::new())
            .await;
        assert!(result.is_err());
    }
}
