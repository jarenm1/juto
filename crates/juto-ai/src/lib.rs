//! Provider-neutral messages, streaming transports, and credentials.
//! Behavioral reference: oh-my-pi 579da1d6; see licenses/OMP-MIT.txt.

pub mod auth;
pub mod judge;
pub mod provider;
mod sse;
mod types;

pub use auth::{
    AuthError, Credential, CredentialKind, CredentialStore, CredentialSummary, OAuthLogin,
};
pub use judge::{
    TextJudge, TypeSafeJudge, create_judge, judge_for_model, judge_role_accepts,
    supports_judge_model,
};
pub use provider::HttpProvider;
pub use types::*;
