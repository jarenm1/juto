//! Execution tiers: how much of the host a run's shell commands may touch.
//!
//! A tier is chosen per run. It decides whether `bash` is sandboxed and whether
//! it runs under resource governance. Sandbox and approval are separate axes:
//! a tier fixes the sandbox, and approvals still apply on top of it. A child run
//! never gets a tier above its parent's.

use serde::{Deserialize, Serialize};

use crate::workspace::{ResourceLimits, Sandbox};

/// How confined a run's shell commands are. Variants are ordered from most to
/// least confined, so `a < b` means `a` is the stricter tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionTier {
    /// Sandboxed, for work triggered by content from outside (issues, PRs from
    /// strangers). Writes are confined like `Workspace`.
    Untrusted,
    /// Sandboxed: writes are confined to the run's workspace and granted paths.
    /// Default for background and isolated work.
    Workspace,
    /// Interactive: the user watches the run. Commands are not sandboxed and
    /// approvals apply. Governed when `bashScoped` is set.
    Attended,
    /// No sandbox and no governance. Explicit opt-in for the top-level run only.
    Full,
}

impl ExecutionTier {
    /// The name used in config files, `task` arguments, and error messages.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Untrusted => "untrusted",
            Self::Workspace => "workspace",
            Self::Attended => "attended",
            Self::Full => "full",
        }
    }

    /// Whether `bash` in this tier runs inside the filesystem sandbox.
    pub fn sandboxed(self) -> bool {
        matches!(self, Self::Untrusted | Self::Workspace)
    }

    /// Resource and sandbox settings for `bash` in this tier, or `None` when the
    /// command runs bare. Sandboxed tiers are always governed. `governed` is the
    /// runtime's `bashScoped` setting and applies only to attended runs.
    pub fn bash_limits(self, governed: bool) -> Option<ResourceLimits> {
        match self {
            Self::Untrusted | Self::Workspace => {
                Some(ResourceLimits::default().with_sandbox(Sandbox::for_agent_workspace()))
            }
            Self::Attended => governed.then(ResourceLimits::default),
            Self::Full => None,
        }
    }

    /// The tier for a child run.
    ///
    /// `requested` is the tier the parent asked for, if any. `isolated` is
    /// whether the child gets its own workspace; isolated children default to
    /// `Workspace`, and others to `Attended`, both capped at the parent's tier.
    /// Asking for `Full` is an error because only the top-level run may be
    /// `Full`, and asking for a tier above the parent's is an error rather than
    /// a silent clamp.
    pub fn resolve_child(
        parent: Self,
        requested: Option<Self>,
        isolated: bool,
    ) -> Result<Self, String> {
        match requested {
            Some(Self::Full) => Err("`full` is only available to the top-level run".to_string()),
            Some(tier) if tier > parent => Err(format!(
                "tier `{}` exceeds this run's tier `{}`",
                tier.as_str(),
                parent.as_str()
            )),
            Some(tier) => Ok(tier),
            None => {
                let default = if isolated {
                    Self::Workspace
                } else {
                    Self::Attended
                };
                Ok(default.min(parent))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stricter_tiers_order_first() {
        assert!(ExecutionTier::Untrusted < ExecutionTier::Workspace);
        assert!(ExecutionTier::Workspace < ExecutionTier::Attended);
        assert!(ExecutionTier::Attended < ExecutionTier::Full);
    }

    #[test]
    fn only_untrusted_and_workspace_are_sandboxed() {
        assert!(ExecutionTier::Untrusted.sandboxed());
        assert!(ExecutionTier::Workspace.sandboxed());
        assert!(!ExecutionTier::Attended.sandboxed());
        assert!(!ExecutionTier::Full.sandboxed());
    }

    #[test]
    fn sandboxed_tiers_are_governed_whatever_the_setting() {
        for tier in [ExecutionTier::Untrusted, ExecutionTier::Workspace] {
            let limits = tier
                .bash_limits(false)
                .expect("sandboxed tiers always have limits");
            assert!(limits.sandbox.is_some(), "{tier:?} must be sandboxed");
        }
    }

    #[test]
    fn attended_is_governed_only_when_configured() {
        assert!(ExecutionTier::Attended.bash_limits(false).is_none());
        let governed = ExecutionTier::Attended
            .bash_limits(true)
            .expect("governed attended run has limits");
        assert!(governed.sandbox.is_none());
    }

    #[test]
    fn full_is_never_governed() {
        assert!(ExecutionTier::Full.bash_limits(true).is_none());
        assert!(ExecutionTier::Full.bash_limits(false).is_none());
    }

    #[test]
    fn unrequested_child_defaults_by_isolation_capped_at_parent() {
        assert_eq!(
            ExecutionTier::resolve_child(ExecutionTier::Full, None, true),
            Ok(ExecutionTier::Workspace)
        );
        assert_eq!(
            ExecutionTier::resolve_child(ExecutionTier::Full, None, false),
            Ok(ExecutionTier::Attended)
        );
        assert_eq!(
            ExecutionTier::resolve_child(ExecutionTier::Untrusted, None, false),
            Ok(ExecutionTier::Untrusted)
        );
    }

    #[test]
    fn child_cannot_loosen_its_parent_tier() {
        assert!(
            ExecutionTier::resolve_child(
                ExecutionTier::Workspace,
                Some(ExecutionTier::Attended),
                false
            )
            .is_err()
        );
        assert!(
            ExecutionTier::resolve_child(
                ExecutionTier::Untrusted,
                Some(ExecutionTier::Workspace),
                true
            )
            .is_err()
        );
        assert_eq!(
            ExecutionTier::resolve_child(
                ExecutionTier::Attended,
                Some(ExecutionTier::Untrusted),
                false
            ),
            Ok(ExecutionTier::Untrusted)
        );
    }

    #[test]
    fn child_cannot_be_full() {
        let err =
            ExecutionTier::resolve_child(ExecutionTier::Full, Some(ExecutionTier::Full), false)
                .expect_err("a child must not be full");
        assert!(err.contains("top-level"), "{err}");
    }
}
