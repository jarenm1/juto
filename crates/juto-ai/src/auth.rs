//! Credential storage and OAuth lifecycle.
//!
//! Ported from oh-my-pi (`packages/ai/src/registry/engine/{oauth-code,refresh,common}.ts`,
//! `registry/oauth/{callback-server,pkce,openai-codex,anthropic}.ts`, and the
//! `compat/rules/auth/{anthropic,openai-codex}.kdl` declarative policies).
//! Source license: licenses/OMP-MIT.txt.
//!
//! Layout: one JSON document at the store path protected by `0600`
//! permissions. Every read-modify-write is serialized cross-process by an
//! `fs2` exclusive lock on a sibling `.lock` file and committed by a
//! write-temp/fsync/rename so a crash never leaves a truncated store. The lock
//! file (not the renamed store file) is the lock target so the guard survives
//! atomic replacement.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::{ErrorKind, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use base64::engine::general_purpose::{STANDARD as B64, URL_SAFE_NO_PAD as B64URL};
use fs2::FileExt;
use juto_catalog::Model;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::Digest;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

/// Default token-endpoint timeout when a rule pins none (`DEFAULT_REQUEST_TIMEOUT_MS`).
const DEFAULT_TOKEN_TIMEOUT: Duration = Duration::from_secs(30);
/// How long the interactive loopback flow waits for the browser callback (source `DEFAULT_TIMEOUT`).
const CALLBACK_TIMEOUT: Duration = Duration::from_secs(300);
/// Skew applied so an access token is refreshed before it is actually stale.
/// The source refresher adds a 60 s skew on top of the per-rule mint skew.
const REFRESH_SKEW_MS: i64 = 60_000;
/// Claude Code SDK version interpolated into the Anthropic refresh User-Agent
/// (`anthropic-sdk-typescript/{claude_code_sdk_version} userOAuthProvider`).
const CLAUDE_CODE_SDK_VERSION: &str = "0.112.1";
/// Pinned Claude Code CLI version sent to the bootstrap endpoint; env-overridable
/// like the source (`PI_AI_CLAUDE_CODE_VERSION`).
const DEFAULT_CLAUDE_CODE_VERSION: &str = "2.1.280";
const ANTHROPIC_BOOTSTRAP_URL: &str = "https://api.anthropic.com/api/claude_cli/bootstrap";
const ANTHROPIC_BOOTSTRAP_MODEL: &str = "claude-opus-4-8";
/// Retries allowed when a random-port redraw collides only on the IPv6 companion.
const IPV6_COMPANION_ATTEMPTS: usize = 4;

fn timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Errors from credential storage, resolution, and OAuth.
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("credential store is corrupt: {0}")]
    Corrupt(String),
    #[error("provider \"{0}\" does not support OAuth login")]
    UnsupportedProvider(String),
    #[error("OAuth callback rejected: {0}")]
    Rejected(String),
    #[error("OAuth callback failed: {0}")]
    Callback(String),
    #[error("OAuth login cancelled")]
    Cancelled,
    #[error("timed out waiting for the OAuth callback")]
    Timeout,
    #[error("{provider} token exchange failed: {message}")]
    Exchange {
        provider: &'static str,
        message: String,
    },
    #[error("{provider} token refresh failed: {message}")]
    Refresh {
        provider: &'static str,
        message: String,
    },
    #[error("{0}")]
    Protocol(String),
}

/// Stored credential kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CredentialKind {
    ApiKey,
    OAuth,
}

/// A resolved credential: the API key or the current OAuth access token.
/// `Debug` never prints the secret.
#[derive(Clone)]
pub struct Credential {
    secret: String,
    pub kind: CredentialKind,
    pub account_id: Option<String>,
}

impl Credential {
    /// The wire secret (API key or OAuth access token).
    pub fn expose_secret(&self) -> &str {
        &self.secret
    }
}

impl fmt::Debug for Credential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Credential")
            .field("secret", &"<redacted>")
            .field("kind", &self.kind)
            .field("account_id", &self.account_id)
            .finish()
    }
}

/// Redacted, listable view of one stored credential — identity metadata only.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CredentialSummary {
    pub provider: String,
    pub kind: CredentialKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub org_name: Option<String>,
    /// Epoch ms at which the access token stops being usable, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires: Option<u64>,
}

/// A pending OAuth authorization-code login: the URL to open plus the loopback
/// redirect the provider will target. `Debug` redacts verifier and state.
pub struct OAuthLogin {
    pub authorization_url: String,
    pub redirect_uri: String,
    provider: &'static str,
    state: String,
    verifier: String,
}

impl fmt::Debug for OAuthLogin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The public URL necessarily carries `state`; mask its value so a
        // debug dump cannot be replayed against a pending login.
        let masked = self.authorization_url.replacen(
            &format!("state={}", self.state),
            "state=<redacted>",
            1,
        );
        f.debug_struct("OAuthLogin")
            .field("authorization_url", &masked)
            .field("redirect_uri", &self.redirect_uri)
            .field("provider", &self.provider)
            .field("state", &"<redacted>")
            .field("verifier", &"<redacted>")
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Persisted model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
enum StoredSecret {
    /// Plain API key.
    ApiKey {
        key: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        account_id: Option<String>,
    },
    /// OAuth token-response projection plus identity fields.
    OAuth {
        access: String,
        #[serde(default)]
        refresh: String,
        /// Epoch ms beyond which the access token must be refreshed.
        expires: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        account_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        email: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        org_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        org_name: Option<String>,
        /// Epoch ms of the interactive login that minted this grant family.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        authorized_at: Option<u64>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoreFile {
    version: u32,
    #[serde(default)]
    credentials: BTreeMap<String, StoredSecret>,
}

impl Default for StoreFile {
    fn default() -> Self {
        Self {
            version: 1,
            credentials: BTreeMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Env API-key fallbacks (catalog providers/*.kdl `env` lines)
// ---------------------------------------------------------------------------

/// First non-empty variable wins, mirroring `$pickenv` over KDL `env` lists.
fn env_vars_for(provider: &str) -> &'static [&'static str] {
    match provider {
        "abliteration" => &["ABLITERATION_API_KEY", "ABLIT_KEY"],
        "aiand" => &["AIAND_API_KEY"],
        "aimlapi" => &["AIMLAPI_API_KEY"],
        "alibaba-coding-plan" => &["ALIBABA_CODING_PLAN_API_KEY"],
        "alibaba-token-plan" => &["ALIBABA_TOKEN_PLAN_API_KEY", "BAILIAN_TOKEN_PLAN_API_KEY"],
        "azure" => &["AZURE_OPENAI_API_KEY"],
        "baseten" => &["BASETEN_API_KEY"],
        "bedrock-mantle" => &["AWS_BEARER_TOKEN_BEDROCK"],
        "cerebras" => &["CEREBRAS_API_KEY"],
        "charm-hyper" => &["CHARM_HYPER_API_KEY", "HYPER_API_KEY"],
        "cline-pass" => &["CLINE_API_KEY"],
        "cloudflare-ai-gateway" => &["CLOUDFLARE_AI_GATEWAY_API_KEY"],
        "commandcode" => &["COMMAND_CODE_API_KEY", "COMMANDCODE_API_KEY"],
        "coreweave" => &["COREWEAVE_API_KEY", "WANDB_API_KEY"],
        "cursor" => &["CURSOR_ACCESS_TOKEN", "CURSOR_API_KEY"],
        "deepinfra" => &["DEEPINFRA_API_KEY"],
        "deepseek" => &["DEEPSEEK_API_KEY"],
        "devin" => &["DEVIN_API_KEY"],
        "firepass" => &["FIREPASS_API_KEY"],
        "fireworks" => &["FIREWORKS_API_KEY"],
        "github-copilot" => &["COPILOT_GITHUB_TOKEN"],
        "gitlab-duo" | "gitlab-duo-agent" => &["GITLAB_TOKEN"],
        "gmi-cloud" => &["GMI_API_KEY"],
        "google" => &["GEMINI_API_KEY"],
        "groq" => &["GROQ_API_KEY"],
        "helmcode" => &["HELMCODE_API_KEY"],
        "huggingface" => &["HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"],
        "kilo" => &["KILO_API_KEY"],
        "kimi-code" => &["KIMI_API_KEY"],
        "litellm" => &["LITELLM_API_KEY"],
        "lm-studio" => &["LM_STUDIO_API_KEY"],
        "local" => &["LOCAL_API_KEY"],
        "meta" => &["MODEL_API_KEY", "META_API_KEY"],
        "minimax-code-cn" => &["MINIMAX_CODE_CN_API_KEY"],
        "minimax-code" => &["MINIMAX_CODE_API_KEY"],
        "minimax" => &["MINIMAX_API_KEY"],
        "mistral" => &["MISTRAL_API_KEY"],
        "moonshot" => &["MOONSHOT_API_KEY", "KIMI_API_KEY"],
        "nanogpt" => &["NANO_GPT_API_KEY"],
        "novita" => &["NOVITA_API_KEY"],
        "nvidia" => &["NVIDIA_API_KEY"],
        "ollama-cloud" => &["OLLAMA_CLOUD_API_KEY"],
        "ollama" => &["OLLAMA_API_KEY"],
        "openai-codex" => &["OPENAI_CODEX_OAUTH_TOKEN"],
        "openai" => &["OPENAI_API_KEY"],
        "opencode-go" | "opencode-zen" => &["OPENCODE_API_KEY"],
        "openrouter" => &["OPENROUTER_API_KEY"],
        "qianfan" => &["QIANFAN_API_KEY"],
        "qwen-portal" => &["QWEN_OAUTH_TOKEN", "QWEN_PORTAL_API_KEY"],
        "sakana" => &["SAKANA_API_KEY", "FUGU_API_KEY"],
        "siliconflow-cn" => &["SILICONFLOW_CN_API_KEY"],
        "siliconflow" => &["SILICONFLOW_API_KEY"],
        "singularityapi-dev" => &["SINGULARITYAPI_DEV_API_KEY"],
        "singularityapi-tech" => &["SINGULARITYAPI_TECH_API_KEY"],
        "snowflake" => &["SNOWFLAKE_PAT"],
        "stepfun" => &["STEPFUN_API_KEY"],
        "synthetic" => &["SYNTHETIC_API_KEY"],
        "together" => &["TOGETHER_API_KEY"],
        "typesafe" => &["TYPESAFE_API_KEY"],
        "umans" => &["UMANS_AI_CODING_PLAN_API_KEY"],
        "venice" => &["VENICE_API_KEY"],
        "vercel-ai-gateway" => &["AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_API_KEY"],
        "vllm" => &["VLLM_API_KEY"],
        "wafer-serverless" => &["WAFER_SERVERLESS_API_KEY"],
        "xai-oauth" => &["XAI_OAUTH_TOKEN", "XAI_API_KEY"],
        "xai" => &["XAI_API_KEY"],
        "xiaomi-token-plan-ams" => &["XIAOMI_TOKEN_PLAN_AMS_API_KEY"],
        "xiaomi-token-plan-cn" => &["XIAOMI_TOKEN_PLAN_CN_API_KEY"],
        "xiaomi-token-plan-sgp" => &["XIAOMI_TOKEN_PLAN_SGP_API_KEY"],
        "xiaomi" => &["XIAOMI_API_KEY"],
        "yolo-auto" => &["YOLO_AUTO_API_KEY"],
        "zai" => &["ZAI_API_KEY"],
        "zenmux" => &["ZENMUX_API_KEY"],
        "zhipu-coding-plan" => &["ZHIPU_API_KEY"],
        _ => &[],
    }
}

/// Anthropic's `env hook="anthropic-foundry"`: Foundry mode adds the enterprise
/// gateway key at the head of the list, otherwise OAuth token beats API key.
fn anthropic_env_key() -> Option<String> {
    let foundry = std::env::var("CLAUDE_CODE_USE_FOUNDRY")
        .map(|value| {
            matches!(
                value.trim().to_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false);
    let vars: &[&str] = if foundry {
        &[
            "ANTHROPIC_FOUNDRY_API_KEY",
            "ANTHROPIC_OAUTH_TOKEN",
            "ANTHROPIC_API_KEY",
        ]
    } else {
        &["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]
    };
    vars.iter()
        .find_map(|name| std::env::var(name).ok().filter(|v| !v.is_empty()))
}

fn env_api_key(provider: &str) -> Option<String> {
    if provider == "anthropic" {
        return anthropic_env_key();
    }
    env_vars_for(provider)
        .iter()
        .find_map(|name| std::env::var(name).ok().filter(|v| !v.is_empty()))
}

// ---------------------------------------------------------------------------
// OAuth policies, transcribed from compat/rules/auth/*.kdl
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum TokenBody {
    Json,
    Form,
}

/// Credential-field dot paths read from the token response body.
#[derive(Debug, Clone, Copy)]
struct CredentialMap {
    access: &'static str,
    refresh: &'static str,
    /// `(path, skew_ms)` on the `expires_in`-style field.
    expires: (&'static str, i64),
    account_id: Option<&'static str>,
    email: Option<&'static str>,
    org_id: Option<&'static str>,
    org_name: Option<&'static str>,
}

struct OAuthPolicy {
    id: &'static str,
    /// Client id, either literal or base64-wrapped per the KDL `encoding` attr.
    client_id: &'static str,
    client_id_base64: bool,
    authorize_url: &'static str,
    scopes: &'static [&'static str],
    /// Extra `authorize-params` entries appended after the standard set.
    authorize_params: &'static [(&'static str, &'static str)],
    callback_port: u16,
    callback_path: &'static str,
    /// Pinned redirect URI; `None` derives one from the bound port.
    redirect_uri: Option<&'static str>,
    /// `port-fallback`: providers with an allowlisted redirect must not redraw.
    port_fallback: bool,
    token_url: &'static str,
    token_body: TokenBody,
    token_timeout: Option<Duration>,
    /// Extra templated `params` merged into the exchange body.
    token_params: &'static [(&'static str, &'static str)],
    /// Extra refresh headers (`anthropic-beta`, User-Agent template).
    refresh_headers: &'static [(&'static str, &'static str)],
    login_map: CredentialMap,
    refresh_map: CredentialMap,
}

const CODEX: OAuthPolicy = OAuthPolicy {
    id: "openai-codex",
    client_id: "app_EMoamEEZ73f0CkXaXp7hrann",
    client_id_base64: false,
    authorize_url: "https://auth.openai.com/oauth/authorize",
    scopes: &[
        "openid",
        "profile",
        "email",
        "offline_access",
        "api.connectors.read",
        "api.connectors.invoke",
    ],
    authorize_params: &[
        ("id_token_add_organizations", "true"),
        ("codex_cli_simplified_flow", "true"),
        ("originator", "omp"),
    ],
    callback_port: 1455,
    callback_path: "/auth/callback",
    redirect_uri: Some("http://localhost:1455/auth/callback"),
    port_fallback: false,
    token_url: "https://auth.openai.com/oauth/token",
    token_body: TokenBody::Form,
    token_timeout: Some(Duration::from_millis(15_000)),
    token_params: &[],
    refresh_headers: &[],
    login_map: CredentialMap {
        access: "access_token",
        refresh: "refresh_token",
        expires: ("expires_in", 0),
        account_id: None,
        email: None,
        org_id: None,
        org_name: None,
    },
    refresh_map: CredentialMap {
        access: "access_token",
        refresh: "refresh_token",
        expires: ("expires_in", 0),
        account_id: None,
        email: None,
        org_id: None,
        org_name: None,
    },
};

const ANTHROPIC: OAuthPolicy = OAuthPolicy {
    id: "anthropic",
    // Public OAuth client id, stored base64 so secret scanners stay quiet.
    client_id: "OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl",
    client_id_base64: true,
    authorize_url: "https://claude.ai/oauth/authorize",
    scopes: &[
        "org:create_api_key",
        "user:profile",
        "user:inference",
        "user:sessions:claude_code",
        "user:mcp_servers",
        "user:file_upload",
    ],
    authorize_params: &[("code", "true")],
    callback_port: 54545,
    callback_path: "/callback",
    redirect_uri: None,
    port_fallback: true,
    token_url: "https://api.anthropic.com/v1/oauth/token",
    token_body: TokenBody::Json,
    token_timeout: None,
    token_params: &[("state", "{state}")],
    refresh_headers: &[
        ("anthropic-beta", "oauth-2025-04-20"),
        (
            "User-Agent",
            "anthropic-sdk-typescript/{claude_code_sdk_version} userOAuthProvider",
        ),
    ],
    login_map: CredentialMap {
        access: "access_token",
        refresh: "refresh_token",
        expires: ("expires_in", 300_000),
        account_id: Some("account.uuid"),
        email: Some("account.email_address"),
        org_id: Some("organization.uuid"),
        org_name: Some("organization.name"),
    },
    refresh_map: CredentialMap {
        access: "access_token",
        refresh: "refresh_token",
        expires: ("expires_in", 300_000),
        account_id: Some("account.uuid"),
        email: Some("account.email_address"),
        org_id: None,
        org_name: None,
    },
};

fn oauth_policy(provider: &str) -> Option<&'static OAuthPolicy> {
    match provider {
        "openai-codex" => Some(&CODEX),
        "anthropic" => Some(&ANTHROPIC),
        _ => None,
    }
}

impl OAuthPolicy {
    fn client_id(&self) -> Result<String, AuthError> {
        if self.client_id_base64 {
            let bytes = B64
                .decode(self.client_id)
                .map_err(|e| AuthError::Corrupt(format!("{} client id: {e}", self.id)))?;
            String::from_utf8(bytes)
                .map_err(|e| AuthError::Corrupt(format!("{} client id: {e}", self.id)))
        } else {
            Ok(self.client_id.to_string())
        }
    }
}

// ---------------------------------------------------------------------------
// PKCE, state, templating, JSON paths (source engine/common.ts helpers)
// ---------------------------------------------------------------------------

/// 96-byte base64url verifier, S256 challenge (source `generatePKCE`).
fn generate_pkce() -> (String, String) {
    let mut bytes = [0u8; 96];
    rand::rng().fill_bytes(&mut bytes);
    let verifier = B64URL.encode(bytes);
    let challenge = B64URL.encode(sha2::Sha256::digest(verifier.as_bytes()));
    (verifier, challenge)
}

/// 16 random bytes hex-encoded (source default `generateState`).
fn generate_state() -> String {
    let mut bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `{name}` placeholder substitution; unknown names become empty (source `template`).
fn template(text: &str, vars: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('{') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        match after.find('}') {
            Some(end)
                if after[..end]
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c == '_') =>
            {
                let name = &after[..end];
                out.push_str(
                    vars.iter()
                        .find(|(k, _)| *k == name)
                        .map(|(_, v)| *v)
                        .unwrap_or(""),
                );
                rest = &after[end + 1..];
            }
            _ => {
                out.push('{');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

fn json_path<'a>(body: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = body;
    for segment in path.split('.') {
        current = current.get(segment)?;
    }
    Some(current)
}

fn scalar_string(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(Value::Number(n)) => Some(n.to_string()),
        _ => None,
    }
}

fn decode_jwt_payload(token: &str) -> Option<Value> {
    let mut parts = token.split('.');
    parts.next()?;
    let payload = parts.next()?;
    if parts.next().is_none() {
        return None;
    }
    let bytes = B64URL.decode(payload).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn url_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn form_encode(params: &[(String, String)]) -> String {
    params
        .iter()
        .map(|(k, v)| format!("{}={}", url_encode(k), url_encode(v)))
        .collect::<Vec<_>>()
        .join("&")
}

fn json_body(params: &[(String, String)]) -> String {
    let map: serde_json::Map<String, Value> = params
        .iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect();
    Value::Object(map).to_string()
}

// ---------------------------------------------------------------------------
// Token endpoint + credential projection
// ---------------------------------------------------------------------------

/// Projection of a token response onto stored OAuth fields (source `mapCredentials`).
struct MappedCredential {
    access: String,
    refresh: Option<String>,
    expires: u64,
    account_id: Option<String>,
    email: Option<String>,
    org_id: Option<String>,
    org_name: Option<String>,
}

fn map_credentials(
    map: &CredentialMap,
    body: &Value,
    provider: &str,
) -> Result<MappedCredential, AuthError> {
    let access = scalar_string(json_path(body, map.access)).ok_or_else(|| {
        let excerpt = body.to_string();
        AuthError::Protocol(format!(
            "{provider} token response missing access token: {}",
            &excerpt[..excerpt.len().min(500)]
        ))
    })?;
    let (expires_path, skew_ms) = map.expires;
    let seconds = json_path(body, expires_path)
        .and_then(Value::as_f64)
        .ok_or_else(|| {
            AuthError::Protocol(format!("{provider} token response missing {expires_path}"))
        })?;
    let expires = (timestamp_ms() as i64 + (seconds * 1000.0) as i64 - skew_ms).max(0) as u64;
    Ok(MappedCredential {
        access,
        refresh: scalar_string(json_path(body, map.refresh)),
        expires,
        account_id: map
            .account_id
            .and_then(|p| scalar_string(json_path(body, p))),
        email: map.email.and_then(|p| scalar_string(json_path(body, p))),
        org_id: map.org_id.and_then(|p| scalar_string(json_path(body, p))),
        org_name: map.org_name.and_then(|p| scalar_string(json_path(body, p))),
    })
}

fn claude_code_version() -> String {
    std::env::var("PI_AI_CLAUDE_CODE_VERSION")
        .unwrap_or_else(|_| DEFAULT_CLAUDE_CODE_VERSION.to_string())
}

// ---------------------------------------------------------------------------
// Loopback callback server
// ---------------------------------------------------------------------------

/// One bound loopback listener pair (IPv4 plus optional IPv6 companion).
struct CallbackListeners {
    v4: tokio::net::TcpListener,
    v6: Option<tokio::net::TcpListener>,
    port: u16,
}

/// Whether `::1` exists on this host; without it the companion bind would
/// misreport a healthy IPv4 listener as a collision (source `ipv6LoopbackAvailable`).
fn ipv6_loopback_available() -> bool {
    std::net::TcpListener::bind((std::net::Ipv6Addr::LOCALHOST, 0)).is_ok()
}

/// Bind the callback listener pair on `port` (`0` = random). Mirrors the
/// source: IPv4 primary plus an IPv6 companion so `localhost` traffic resolved
/// to `::1` cannot be captured by a wildcard listener; a companion collision
/// on a random port redraws the pair, while a genuine same-address collision
/// surfaces `AddrInUse`.
async fn bind_callback(port: u16) -> Result<CallbackListeners, AuthError> {
    let dual = ipv6_loopback_available();
    for attempt in 0.. {
        let v4 = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], port))).await?;
        let bound = v4.local_addr()?.port();
        if !dual {
            return Ok(CallbackListeners {
                v4,
                v6: None,
                port: bound,
            });
        }
        match tokio::net::TcpListener::bind(SocketAddr::from((
            std::net::Ipv6Addr::LOCALHOST,
            bound,
        )))
        .await
        {
            Ok(v6) => {
                return Ok(CallbackListeners {
                    v4,
                    v6: Some(v6),
                    port: bound,
                });
            }
            Err(e)
                if e.kind() == ErrorKind::AddrInUse
                    && port == 0
                    && attempt < IPV6_COMPANION_ATTEMPTS =>
            {
                drop(v4);
                continue;
            }
            Err(e) if e.kind() == ErrorKind::AddrInUse => return Err(AuthError::Io(e)),
            // Non-collision companion failure (e.g. IPv6 vanished): IPv4 serves alone.
            Err(_) => {
                return Ok(CallbackListeners {
                    v4,
                    v6: None,
                    port: bound,
                });
            }
        }
    }
    unreachable!()
}

const CALLBACK_HTML_OK: &str = "<!doctype html><html><head><title>Signed in</title></head>\
    <body><h1>Signed in</h1><p>You can return to the terminal; this window can be closed.</p></body></html>";
const CALLBACK_HTML_ERR: &str = "<!doctype html><html><head><title>Sign-in failed</title></head>\
    <body><h1>Sign-in failed</h1><p>The authorization callback was rejected. Check the terminal for details.</p></body></html>";

enum CallbackOutcome {
    /// Code + state received; finish the exchange.
    Code { code: String, state: String },
    /// The redirect carried our state with an `error` param: fail fast.
    Denied { message: String },
    /// Missing code / state mismatch: answered 500, keep waiting (source behavior).
    KeepWaiting,
}

/// Serve one HTTP request against the callback route; returns how the flow
/// should proceed after the response has been written.
async fn serve_callback_request(
    stream: tokio::net::TcpStream,
    expected_state: &str,
    callback_path: &str,
) -> Result<CallbackOutcome, AuthError> {
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).await? == 0 {
        return Ok(CallbackOutcome::KeepWaiting);
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let target = parts.next().unwrap_or("");
    // Drain headers so the browser sees a clean close.
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 || line.trim().is_empty() {
            break;
        }
    }

    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let mut status = (404, "Not Found", "Not Found");
    let mut outcome = CallbackOutcome::KeepWaiting;

    if method == "GET" && path == callback_path {
        let mut code = String::new();
        let mut state = String::new();
        let mut error = String::new();
        let mut error_description = String::new();
        for pair in query.split('&') {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            match key {
                "code" => code = percent_decode(value),
                "state" => state = percent_decode(value),
                "error" => error = percent_decode(value),
                "error_description" => error_description = percent_decode(value),
                _ => {}
            }
        }
        if !error.is_empty() && (expected_state.is_empty() || state == expected_state) {
            // Error carrying our state nonce: genuine denial — fail the flow
            // now instead of waiting out the timeout (source #4106 fix).
            status = (500, "Internal Server Error", CALLBACK_HTML_ERR);
            outcome = CallbackOutcome::Denied {
                message: format!(
                    "Authorization failed: {}",
                    if error_description.is_empty() {
                        error
                    } else {
                        error_description
                    }
                ),
            };
        } else if code.is_empty() || (!expected_state.is_empty() && state != expected_state) {
            // Forged/error responses without our state stay ignored, exactly
            // like the source: 500 page, flow keeps waiting.
            status = (500, "Internal Server Error", CALLBACK_HTML_ERR);
        } else {
            status = (200, "OK", CALLBACK_HTML_OK);
            outcome = CallbackOutcome::Code { code, state };
        }
    }

    let (code_status, reason, body) = status;
    let response = format!(
        "HTTP/1.1 {code_status} {reason}\r\ncontent-type: text/html\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    reader.get_mut().write_all(response.as_bytes()).await?;
    Ok(outcome)
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hex = |b: u8| (b as char).to_digit(16);
                match (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                    (Some(h), Some(l)) => out.push((h * 16 + l) as u8),
                    _ => out.push(bytes[i]),
                }
                i += 3;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ---------------------------------------------------------------------------
// CredentialStore
// ---------------------------------------------------------------------------

/// JSON credential store protected by `0600` perms, an `fs2` lockfile for
/// cross-process mutation, and atomic rename commits.
pub struct CredentialStore {
    path: PathBuf,
    lock_path: PathBuf,
    /// In-process serialization for async read-modify-write and refresh; the
    /// fs2 lockfile orders both in-process threads and other processes.
    mutex: Mutex<()>,
    client: reqwest::Client,
    /// Endpoint overrides populated only by unit tests.
    #[cfg(test)]
    endpoint_overrides: std::collections::HashMap<&'static str, String>,
}

impl fmt::Debug for CredentialStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CredentialStore")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl CredentialStore {
    /// Open (or create) the store at `path`. Synchronous by contract; creates
    /// parent directories and enforces `0600` on the store file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, AuthError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = fs::set_permissions(parent, fs::Permissions::from_mode(0o700));
            }
        }
        if path.exists() {
            let text = fs::read_to_string(&path)?;
            let store: StoreFile = serde_json::from_str(&text)
                .map_err(|e| AuthError::Corrupt(format!("{}: {e}", path.display())))?;
            if store.version != 1 {
                return Err(AuthError::Corrupt(format!(
                    "{}: unsupported version {}",
                    path.display(),
                    store.version
                )));
            }
        } else {
            write_store_atomic(&path, &StoreFile::default())?;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
        }
        let mut lock_name = path.clone().into_os_string();
        lock_name.push(".lock");
        Ok(Self {
            path,
            lock_path: PathBuf::from(lock_name),
            mutex: Mutex::new(()),
            client: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .map_err(|error| AuthError::Protocol(error.without_url().to_string()))?,
            #[cfg(test)]
            endpoint_overrides: std::collections::HashMap::new(),
        })
    }

    fn read_store(&self) -> Result<StoreFile, AuthError> {
        match fs::read_to_string(&self.path) {
            Ok(text) => serde_json::from_str(&text)
                .map_err(|e| AuthError::Corrupt(format!("{}: {e}", self.path.display()))),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(StoreFile::default()),
            Err(e) => Err(AuthError::Io(e)),
        }
    }

    /// Exclusive cross-process lock on the sidecar lockfile. The returned
    /// `File` is the guard; dropping it releases the lock. Runs on the
    /// blocking pool so the scheduler is never stalled waiting on `flock`.
    async fn lock_exclusive(&self) -> Result<fs::File, AuthError> {
        let lock_path = self.lock_path.clone();
        tokio::task::spawn_blocking(move || -> Result<fs::File, AuthError> {
            let file = fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(&lock_path)?;
            file.lock_exclusive()?;
            Ok(file)
        })
        .await
        .map_err(|e| AuthError::Io(std::io::Error::other(e)))?
    }

    /// Synchronous variant for the sync API; `flock` contention between
    /// in-process threads is itself the serialization.
    fn lock_exclusive_blocking(&self) -> Result<fs::File, AuthError> {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&self.lock_path)?;
        file.lock_exclusive()?;
        Ok(file)
    }

    /// Store (or replace) an API key for `provider`. An empty/whitespace key
    /// removes the entry, matching the source `empty-fallback` opt-out.
    pub fn set_api_key(&self, provider: &str, key: &str) -> Result<(), AuthError> {
        let _file_lock = self.lock_exclusive_blocking()?;
        let mut store = self.read_store()?;
        if key.trim().is_empty() {
            store.credentials.remove(provider);
        } else {
            store.credentials.insert(
                provider.to_string(),
                StoredSecret::ApiKey {
                    key: key.to_string(),
                    account_id: None,
                },
            );
        }
        write_store_atomic(&self.path, &store)
    }

    /// Remove any credential stored for `provider`.
    pub fn remove(&self, provider: &str) -> Result<(), AuthError> {
        let _file_lock = self.lock_exclusive_blocking()?;
        let mut store = self.read_store()?;
        store.credentials.remove(provider);
        write_store_atomic(&self.path, &store)
    }

    /// List redacted identity metadata for every stored credential.
    pub fn list(&self) -> Result<Vec<CredentialSummary>, AuthError> {
        let store = self.read_store()?;
        Ok(store
            .credentials
            .iter()
            .map(|(provider, secret)| match secret {
                StoredSecret::ApiKey { account_id, .. } => CredentialSummary {
                    provider: provider.clone(),
                    kind: CredentialKind::ApiKey,
                    account_id: account_id.clone(),
                    email: None,
                    org_name: None,
                    expires: None,
                },
                StoredSecret::OAuth {
                    account_id,
                    email,
                    org_name,
                    expires,
                    ..
                } => CredentialSummary {
                    provider: provider.clone(),
                    kind: CredentialKind::OAuth,
                    account_id: account_id.clone(),
                    email: email.clone(),
                    org_name: org_name.clone(),
                    expires: Some(*expires),
                },
            })
            .collect())
    }

    /// Resolve a usable credential for `model`'s provider: explicit stored
    /// credential first (refreshing OAuth under the cross-process lock and
    /// durably committing rotated refresh tokens), then the provider's env
    /// vars, then `None` for local/auth-none providers.
    pub async fn resolve(&self, model: &Model) -> Result<Option<Credential>, AuthError> {
        let provider = model.provider.as_str();
        // Cheap unlocked read first; the refresh path re-checks under lock.
        let stored = self.read_store()?.credentials.get(provider).cloned();
        match stored {
            Some(StoredSecret::ApiKey { key, account_id }) => Ok(Some(Credential {
                secret: key,
                kind: CredentialKind::ApiKey,
                account_id,
            })),
            Some(StoredSecret::OAuth { .. }) => {
                let _in_process = self.mutex.lock().await;
                // Serialize refresh + commit across processes. The lock is
                // held over the refresh HTTP call on purpose — that is the
                // critical section it exists for; unrelated store I/O still
                // happens only inside short locked windows.
                let _file_lock = self.lock_exclusive().await?;
                let mut store = self.read_store()?;
                let Some(entry) = store.credentials.get_mut(provider) else {
                    // Another process removed it; fall through to env.
                    return Ok(env_api_key(provider).map(|key| Credential {
                        secret: key,
                        kind: CredentialKind::ApiKey,
                        account_id: None,
                    }));
                };
                let now = timestamp_ms() as i64;
                match entry {
                    // Another process rotated to an API key between reads.
                    StoredSecret::ApiKey { key, account_id } => Ok(Some(Credential {
                        secret: key.clone(),
                        kind: CredentialKind::ApiKey,
                        account_id: account_id.clone(),
                    })),
                    StoredSecret::OAuth {
                        access, expires, ..
                    } if (*expires as i64) - now > REFRESH_SKEW_MS => Ok(Some(Credential {
                        secret: access.clone(),
                        kind: CredentialKind::OAuth,
                        account_id: entry_account_id(entry),
                    })),
                    StoredSecret::OAuth { refresh, .. } if refresh.is_empty() => {
                        Err(AuthError::Refresh {
                            provider: policy_id(provider),
                            message: format!(
                                "{provider} credentials are missing refresh; sign in again"
                            ),
                        })
                    }
                    StoredSecret::OAuth { .. } => {
                        let policy = oauth_policy(provider).ok_or_else(|| AuthError::Refresh {
                            provider: policy_id(provider),
                            message: format!("no refresh policy for {provider}"),
                        })?;
                        let refreshed = self.refresh_tokens(policy, entry).await?;
                        let StoredSecret::OAuth {
                            access, account_id, ..
                        } = &refreshed
                        else {
                            unreachable!("refresh_tokens always returns OAuth")
                        };
                        let credential = Credential {
                            secret: access.clone(),
                            kind: CredentialKind::OAuth,
                            account_id: account_id.clone(),
                        };
                        *entry = refreshed;
                        write_store_atomic(&self.path, &store)?;
                        Ok(Some(credential))
                    }
                }
            }
            None => Ok(env_api_key(provider).map(|key| Credential {
                secret: key,
                kind: CredentialKind::ApiKey,
                account_id: None,
            })),
        }
    }

    // -- OAuth lifecycle ----------------------------------------------------

    /// Begin an authorization-code + PKCE login for `provider`. Synchronous:
    /// generates state/verifier and builds the authorization URL against the
    /// provider's advertised redirect URI without binding a socket.
    pub fn begin_login(&self, provider: &str) -> Result<OAuthLogin, AuthError> {
        let policy = oauth_policy(provider)
            .ok_or_else(|| AuthError::UnsupportedProvider(provider.to_string()))?;
        let state = generate_state();
        let (verifier, challenge) = generate_pkce();
        let redirect_uri = policy.redirect_uri.map(str::to_string).unwrap_or_else(|| {
            format!(
                "http://localhost:{}{}",
                policy.callback_port, policy.callback_path
            )
        });
        let authorization_url =
            self.authorization_url(policy, &state, &redirect_uri, &challenge)?;
        Ok(OAuthLogin {
            authorization_url,
            redirect_uri,
            provider: policy.id,
            state,
            verifier,
        })
    }

    /// Finish a `begin_login` flow with the redirect URL (or pasted code) the
    /// user supplies. Validates `state` against the pending login and performs
    /// the real token exchange before storing the credential.
    pub async fn complete_login(
        &self,
        login: OAuthLogin,
        callback_url: &str,
    ) -> Result<(), AuthError> {
        let policy = oauth_policy(login.provider).expect("begin_login validated the policy");
        let parsed = parse_callback_input(callback_url);
        let code = parsed.code.ok_or_else(|| {
            AuthError::Callback(
                "could not find an authorization code in the pasted input".to_string(),
            )
        })?;
        if let Some(state) = parsed.state {
            if state != login.state {
                return Err(AuthError::Callback(
                    "State mismatch - possible CSRF attack".to_string(),
                ));
            }
        }
        // Providers may echo `code#state`; the fragment wins over the callback
        // state, matching the source exchange path.
        let (exchange_code, exchange_state) = match code.split_once('#') {
            Some((c, s)) => (
                c.to_string(),
                if s.is_empty() {
                    login.state.clone()
                } else {
                    s.to_string()
                },
            ),
            None => (code, login.state.clone()),
        };
        let mapped = self
            .exchange_code(
                policy,
                &exchange_code,
                &exchange_state,
                &login.redirect_uri,
                &login.verifier,
            )
            .await?;
        self.store_oauth(policy, mapped).await
    }

    /// Interactive login: bind the loopback callback BEFORE publishing the
    /// authorization URL through `on_url`, wait for the redirect, then
    /// exchange and store. Honors `cancel` and the source's 5-minute timeout.
    pub async fn login(
        &self,
        provider: &str,
        cancel: CancellationToken,
        on_url: Arc<dyn Fn(&str) + Send + Sync>,
    ) -> Result<(), AuthError> {
        let policy = oauth_policy(provider)
            .ok_or_else(|| AuthError::UnsupportedProvider(provider.to_string()))?;
        if cancel.is_cancelled() {
            return Err(AuthError::Cancelled);
        }
        let state = generate_state();
        let (verifier, challenge) = generate_pkce();

        // Bind first so a busy pinned port fails before the browser opens.
        let listeners = match bind_callback(policy.callback_port).await {
            Ok(l) => l,
            Err(AuthError::Io(e)) if e.kind() == ErrorKind::AddrInUse => {
                if let Some(fixed) = policy.redirect_uri {
                    return Err(AuthError::Callback(format!(
                        "OAuth callback port {} is in use, but {} requires this exact redirect ({fixed}). Free the port and retry.",
                        policy.callback_port, policy.id
                    )));
                }
                if !policy.port_fallback {
                    return Err(AuthError::Callback(format!(
                        "OAuth callback port {} is in use and this provider does not allow a fallback port.",
                        policy.callback_port
                    )));
                }
                bind_callback(0).await?
            }
            Err(e) => return Err(e),
        };
        let bound_port = listeners.port;
        let redirect_uri = policy
            .redirect_uri
            .map(str::to_string)
            .unwrap_or_else(|| format!("http://localhost:{bound_port}{}", policy.callback_path));
        let authorization_url =
            self.authorization_url(policy, &state, &redirect_uri, &challenge)?;
        on_url(&authorization_url);

        let CallbackListeners { v4, mut v6, .. } = listeners;
        let callback_path = policy.callback_path;
        let expected = state.clone();
        let deadline = Instant::now() + CALLBACK_TIMEOUT;
        let (code, cb_state) = loop {
            let accepted = tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(AuthError::Cancelled),
                _ = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => return Err(AuthError::Timeout),
                accepted = v4.accept() => accepted.map_err(AuthError::Io)?,
                accepted = async {
                    match v6.as_mut() {
                        Some(l) => l.accept().await,
                        None => std::future::pending::<Result<(tokio::net::TcpStream, SocketAddr), std::io::Error>>().await,
                    }
                } => accepted.map_err(AuthError::Io)?,
            };
            let (stream, _) = accepted;
            match serve_callback_request(stream, &expected, callback_path).await {
                Ok(CallbackOutcome::Code { code, state }) => break (code, state),
                Ok(CallbackOutcome::Denied { message }) => {
                    return Err(AuthError::Rejected(message));
                }
                Ok(CallbackOutcome::KeepWaiting) => continue,
                Err(e) => return Err(e),
            }
        };
        if cancel.is_cancelled() {
            return Err(AuthError::Cancelled);
        }
        let (exchange_code, exchange_state) = match code.split_once('#') {
            Some((c, s)) => (
                c.to_string(),
                if s.is_empty() {
                    cb_state.clone()
                } else {
                    s.to_string()
                },
            ),
            None => (code, cb_state),
        };
        let mapped = self
            .exchange_code(
                policy,
                &exchange_code,
                &exchange_state,
                &redirect_uri,
                &verifier,
            )
            .await?;
        self.store_oauth(policy, mapped).await
    }

    fn authorization_url(
        &self,
        policy: &OAuthPolicy,
        state: &str,
        redirect_uri: &str,
        challenge: &str,
    ) -> Result<String, AuthError> {
        let client_id = policy.client_id()?;
        let scope = policy.scopes.join(" ");
        let vars: Vec<(&str, &str)> = vec![
            ("client_id", client_id.as_str()),
            ("redirect_uri", redirect_uri),
            ("scope", scope.as_str()),
            ("state", state),
            ("code_challenge", challenge),
        ];
        // Template the extras first so the vars borrows end before the moves.
        let extra: Vec<(String, String)> = policy
            .authorize_params
            .iter()
            .map(|(key, value)| (key.to_string(), template(value, &vars)))
            .collect();
        let mut params: Vec<(String, String)> = vec![
            ("client_id".into(), client_id),
            ("response_type".into(), "code".into()),
            ("redirect_uri".into(), redirect_uri.to_string()),
            ("scope".into(), scope),
            ("code_challenge".into(), challenge.to_string()),
            ("code_challenge_method".into(), "S256".into()),
            ("state".into(), state.to_string()),
        ];
        params.extend(extra);
        Ok(format!("{}?{}", policy.authorize_url, form_encode(&params)))
    }

    fn token_url(&self, policy: &OAuthPolicy) -> String {
        #[cfg(test)]
        if let Some(url) = self.endpoint_overrides.get(policy.id) {
            return url.clone();
        }
        policy.token_url.to_string()
    }

    /// POST the authorization-code exchange (source `postTokenRequest` +
    /// `exchangeToken`): standard grant params plus the rule's templated
    /// `params`, JSON or form body per the KDL `body` attribute.
    async fn exchange_code(
        &self,
        policy: &OAuthPolicy,
        code: &str,
        state: &str,
        redirect_uri: &str,
        verifier: &str,
    ) -> Result<MappedCredential, AuthError> {
        let client_id = policy.client_id()?;
        let vars: Vec<(&str, &str)> = vec![
            ("code", code),
            ("state", state),
            ("redirect_uri", redirect_uri),
            ("code_verifier", verifier),
            ("client_id", client_id.as_str()),
        ];
        // Template the extras first so the vars borrows end before the moves.
        let extra: Vec<(String, String)> = policy
            .token_params
            .iter()
            .map(|(key, value)| (key.to_string(), template(value, &vars)))
            .collect();
        let mut params: Vec<(String, String)> = vec![
            ("grant_type".into(), "authorization_code".into()),
            ("client_id".into(), client_id),
            ("code".into(), code.to_string()),
            ("redirect_uri".into(), redirect_uri.to_string()),
            ("code_verifier".into(), verifier.to_string()),
        ];
        params.extend(extra);
        let url = self.token_url(policy);
        let timeout = policy.token_timeout.unwrap_or(DEFAULT_TOKEN_TIMEOUT);
        let request = match policy.token_body {
            TokenBody::Json => self
                .client
                .post(&url)
                .timeout(timeout)
                .header("content-type", "application/json")
                .body(json_body(&params)),
            TokenBody::Form => self
                .client
                .post(&url)
                .timeout(timeout)
                .header("content-type", "application/x-www-form-urlencoded")
                .body(form_encode(&params)),
        };
        let response = request.send().await.map_err(|e| AuthError::Exchange {
            provider: policy.id,
            message: format!("{url}: {e}"),
        })?;
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(AuthError::Exchange {
                provider: policy.id,
                message: format!("{} {}", status.as_u16(), &text[..text.len().min(500)]),
            });
        }
        let body: Value = serde_json::from_str(&text).map_err(|_| {
            AuthError::Protocol(format!("{} token response was not JSON", policy.id))
        })?;
        let mut mapped = map_credentials(&policy.login_map, &body, policy.id)?;
        // after-exchange hooks
        match policy.id {
            "openai-codex" => apply_codex_profile(&mut mapped, &body, None, true)?,
            "anthropic" => self.apply_anthropic_identity(&mut mapped, true).await,
            _ => {}
        }
        Ok(mapped)
    }

    /// Refresh-token grant for a stored OAuth credential (source `refresh`
    /// rule): standard refresh params, rule headers, credential remap; an
    /// unrotated refresh token is preserved.
    async fn refresh_tokens(
        &self,
        policy: &OAuthPolicy,
        entry: &StoredSecret,
    ) -> Result<StoredSecret, AuthError> {
        let StoredSecret::OAuth {
            refresh,
            account_id,
            email,
            org_id,
            org_name,
            authorized_at,
            ..
        } = entry
        else {
            return Err(AuthError::Protocol(
                "refresh called on non-OAuth credential".into(),
            ));
        };
        let client_id = policy.client_id()?;
        let params: Vec<(String, String)> = vec![
            ("grant_type".into(), "refresh_token".into()),
            ("client_id".into(), client_id.clone()),
            ("refresh_token".into(), refresh.clone()),
        ];
        // The source also forwards `organization_id` for WorkOS-scoped grants;
        // neither ported policy mints one, so none is added.
        let url = self.token_url(policy);
        let timeout = policy.token_timeout.unwrap_or(DEFAULT_TOKEN_TIMEOUT);
        let vars: Vec<(&str, &str)> = vec![
            ("refresh_token", refresh.as_str()),
            ("client_id", client_id.as_str()),
            ("claude_code_sdk_version", CLAUDE_CODE_SDK_VERSION),
        ];
        let request = match policy.token_body {
            TokenBody::Json => self
                .client
                .post(&url)
                .timeout(timeout)
                .header("content-type", "application/json")
                .body(json_body(&params)),
            TokenBody::Form => self
                .client
                .post(&url)
                .timeout(timeout)
                .header("content-type", "application/x-www-form-urlencoded")
                .body(form_encode(&params)),
        };
        let mut request = request;
        for (key, value) in policy.refresh_headers {
            request = request.header(*key, template(value, &vars));
        }
        let response = request.send().await.map_err(|e| AuthError::Refresh {
            provider: policy.id,
            message: format!("{url}: {e}"),
        })?;
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(AuthError::Refresh {
                provider: policy.id,
                message: format!("{} {}", status.as_u16(), &text[..text.len().min(500)]),
            });
        }
        let body: Value = serde_json::from_str(&text).map_err(|_| {
            AuthError::Protocol(format!("{} refresh response was not JSON", policy.id))
        })?;
        let mut mapped = map_credentials(&policy.refresh_map, &body, policy.id)?;
        match policy.id {
            "openai-codex" => apply_codex_profile(
                &mut mapped,
                &body,
                Some((account_id, email, org_id, org_name)),
                false,
            )?,
            "anthropic" => self.apply_anthropic_identity(&mut mapped, false).await,
            _ => {}
        }
        Ok(StoredSecret::OAuth {
            access: mapped.access,
            // Unrotated refresh tokens keep the stored value.
            refresh: mapped.refresh.unwrap_or_else(|| refresh.clone()),
            expires: mapped.expires,
            account_id: mapped.account_id.or_else(|| account_id.clone()),
            email: mapped.email.or_else(|| email.clone()),
            org_id: mapped.org_id.or_else(|| org_id.clone()),
            org_name: mapped.org_name.or_else(|| org_name.clone()),
            authorized_at: *authorized_at,
        })
    }

    /// `anthropic-identity` hook: fill missing account (and at login, org)
    /// identity from the Claude Code bootstrap endpoint; failures degrade to
    /// the unenriched credential exactly like the source.
    async fn apply_anthropic_identity(&self, mapped: &mut MappedCredential, is_login: bool) {
        let org_satisfied = !is_login || mapped.org_id.is_some();
        if mapped.account_id.is_some() && mapped.email.is_some() && org_satisfied {
            return;
        }
        let url = format!(
            "{ANTHROPIC_BOOTSTRAP_URL}?entrypoint=cli&model={}",
            url_encode(ANTHROPIC_BOOTSTRAP_MODEL)
        );
        let result = self
            .client
            .get(&url)
            .timeout(DEFAULT_TOKEN_TIMEOUT)
            .header("accept", "application/json, text/plain, */*")
            .header("authorization", format!("Bearer {}", mapped.access))
            .header("content-type", "application/json")
            .header(
                "user-agent",
                format!("claude-code/{}", claude_code_version()),
            )
            .header("anthropic-beta", "oauth-2025-04-20")
            .send()
            .await;
        let Ok(response) = result else { return };
        if !response.status().is_success() {
            return;
        }
        let Ok(body) = response.json::<Value>().await else {
            return;
        };
        let account = body.get("oauth_account");
        let field = |name: &str| {
            account
                .and_then(|a| a.get(name))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };
        if mapped.account_id.is_none() {
            mapped.account_id = field("account_uuid");
        }
        if mapped.email.is_none() {
            mapped.email = field("account_email_address");
        }
        if is_login {
            if mapped.org_id.is_none() {
                mapped.org_id = field("organization_uuid");
            }
            if mapped.org_name.is_none() {
                mapped.org_name = field("organization_name");
            }
        }
    }

    /// Commit a freshly exchanged credential under the locks.
    async fn store_oauth(
        &self,
        policy: &OAuthPolicy,
        mapped: MappedCredential,
    ) -> Result<(), AuthError> {
        let _in_process = self.mutex.lock().await;
        let _file_lock = self.lock_exclusive().await?;
        let mut store = self.read_store()?;
        store.credentials.insert(
            policy.id.to_string(),
            StoredSecret::OAuth {
                access: mapped.access,
                refresh: mapped.refresh.unwrap_or_default(),
                expires: mapped.expires,
                account_id: mapped.account_id,
                email: mapped.email,
                org_id: mapped.org_id,
                org_name: mapped.org_name,
                authorized_at: Some(timestamp_ms()),
            },
        );
        write_store_atomic(&self.path, &store)
    }

    #[cfg(test)]
    fn override_token_endpoint(&mut self, provider: &'static str, url: String) {
        self.endpoint_overrides.insert(provider, url);
    }
}

fn entry_account_id(entry: &StoredSecret) -> Option<String> {
    match entry {
        StoredSecret::ApiKey { account_id, .. } => account_id.clone(),
        StoredSecret::OAuth { account_id, .. } => account_id.clone(),
    }
}

fn policy_id(provider: &str) -> &'static str {
    match provider {
        "openai-codex" => "openai-codex",
        "anthropic" => "anthropic",
        _ => "oauth",
    }
}

/// `openai-codex-profile` hook: derive account/org identity from the access or
/// id token's `https://api.openai.com/*` claims; login requires identity.
fn apply_codex_profile(
    mapped: &mut MappedCredential,
    body: &Value,
    stored: Option<(
        &Option<String>,
        &Option<String>,
        &Option<String>,
        &Option<String>,
    )>,
    is_login: bool,
) -> Result<(), AuthError> {
    const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";
    const JWT_PROFILE_CLAIM: &str = "https://api.openai.com/profile";
    let id_token = body.get("id_token").and_then(Value::as_str);
    let claims = |token: Option<&str>| -> Option<(Option<String>, Option<String>, Option<String>)> {
        let payload = decode_jwt_payload(token?)?;
        let auth = payload.get(JWT_CLAIM_PATH);
        let account = auth
            .and_then(|a| a.get("chatgpt_account_id"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        let plan = auth
            .and_then(|a| a.get("chatgpt_plan_type"))
            .and_then(Value::as_str)
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty());
        let email = payload
            .get(JWT_PROFILE_CLAIM)
            .and_then(|p| p.get("email"))
            .and_then(Value::as_str)
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty());
        Some((account, email, plan))
    };
    let access_claims = claims(Some(mapped.access.as_str()));
    let id_claims = claims(id_token);
    let account_id = access_claims
        .as_ref()
        .and_then(|c| c.0.clone())
        .or_else(|| id_claims.as_ref().and_then(|c| c.0.clone()));
    let email = access_claims
        .as_ref()
        .and_then(|c| c.1.clone())
        .or_else(|| id_claims.as_ref().and_then(|c| c.1.clone()));
    let plan = access_claims
        .as_ref()
        .and_then(|c| c.2.clone())
        .or_else(|| id_claims.as_ref().and_then(|c| c.2.clone()));
    if is_login && account_id.is_none() && email.is_none() {
        return Err(AuthError::Protocol(
            "Failed to extract account identity from token".into(),
        ));
    }
    let (stored_account, stored_email, stored_org, stored_org_name) = stored
        .map(|(a, e, o, n)| (a.clone(), e.clone(), o.clone(), n.clone()))
        .unwrap_or_default();
    let resolved_account = account_id.or(stored_account);
    if let Some(a) = &resolved_account {
        mapped.account_id = Some(a.clone());
    }
    if let Some(e) = email.or(stored_email) {
        mapped.email = Some(e);
    }
    if is_login {
        if let Some(a) = &resolved_account {
            mapped.org_id = Some(a.clone());
        }
        if mapped.org_name.is_none() {
            mapped.org_name = plan;
        }
    } else {
        if mapped.org_id.is_none() {
            mapped.org_id = stored_org;
        }
        if mapped.org_name.is_none() {
            mapped.org_name = stored_org_name;
        }
    }
    Ok(())
}

/// Parse a pasted redirect URL or bare code (source `parseCallbackInput`).
fn parse_callback_input(input: &str) -> ParsedCallback {
    let value = input.trim();
    if value.is_empty() {
        return ParsedCallback::default();
    }
    if value.contains("://") {
        if let Some(query) = value.split_once('?').map(|(_, q)| q) {
            let params = query_params(query);
            if params.contains_key("code") || params.contains_key("state") {
                return ParsedCallback {
                    code: params.get("code").cloned(),
                    state: params.get("state").cloned(),
                };
            }
        }
    }
    if value.contains("code=") {
        let trimmed = value.trim_start_matches(['?', '#']);
        let params = query_params(trimmed);
        return ParsedCallback {
            code: params.get("code").cloned(),
            state: params.get("state").cloned(),
        };
    }
    let (code, state) = match value.split_once('#') {
        Some((c, s)) => (c, Some(s.to_string())),
        None => (value, None),
    };
    ParsedCallback {
        code: Some(code.to_string()),
        state,
    }
}

#[derive(Default)]
struct ParsedCallback {
    code: Option<String>,
    state: Option<String>,
}

fn query_params(query: &str) -> BTreeMap<String, String> {
    query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .map(|(k, v)| (k.to_string(), percent_decode(v)))
        .collect()
}

/// fsync'd write to `path.tmp` followed by an atomic rename; mode 0600.
fn write_store_atomic(path: &Path, store: &StoreFile) -> Result<(), AuthError> {
    let tmp = path.with_extension("tmp");
    let data = serde_json::to_vec_pretty(store).map_err(|e| AuthError::Corrupt(e.to_string()))?;
    {
        let mut file = fs::File::create(&tmp)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(fs::Permissions::from_mode(0o600))?;
        }
        file.write_all(&data)?;
        file.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    // Best-effort directory fsync so the rename itself is durable.
    if let Some(parent) = path.parent() {
        if let Ok(dir) = fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use tokio::io::AsyncReadExt;

    /// Serializes env-mutating tests. (`std::sync::Mutex`: parking_lot is not
    /// a workspace dependency and manifests are parent-owned.)
    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    fn tempdir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    fn model(provider: &str) -> Model {
        Model {
            id: "m".into(),
            name: "m".into(),
            api: "anthropic-messages".into(),
            provider: provider.into(),
            base_url: "http://localhost".into(),
            reasoning: false,
            input: vec![],
            cost: Default::default(),
            context_window: None,
            max_tokens: None,
            kind: "chat".into(),
            supports_tools: false,
            compat: Value::Null,
        }
    }

    #[test]
    fn set_get_list_remove_api_key() {
        let dir = tempdir();
        let path = dir.path().join("auth.json");
        let store = CredentialStore::open(&path).unwrap();
        store.set_api_key("openai", "sk-test-1").unwrap();
        store.set_api_key("anthropic", "sk-ant").unwrap();
        assert_eq!(store.list().unwrap().len(), 2);
        // Reopen sees persisted state.
        let reopened = CredentialStore::open(&path).unwrap();
        assert_eq!(reopened.list().unwrap().len(), 2);
        reopened.remove("openai").unwrap();
        assert_eq!(reopened.list().unwrap().len(), 1);
        // Empty key removes (ollama empty-fallback behavior).
        reopened.set_api_key("anthropic", "  ").unwrap();
        assert!(reopened.list().unwrap().is_empty());
    }

    #[test]
    fn store_file_is_mode_600() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempdir();
        let path = dir.path().join("auth.json");
        let _store = CredentialStore::open(&path).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        // Tighten an existing loose file on open.
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        let _ = CredentialStore::open(&path).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    #[test]
    fn corrupt_store_fails_closed() {
        let dir = tempdir();
        let path = dir.path().join("auth.json");
        fs::write(&path, "{not json").unwrap();
        assert!(matches!(
            CredentialStore::open(&path),
            Err(AuthError::Corrupt(_))
        ));
    }

    #[test]
    fn credential_debug_redacts() {
        let cred = Credential {
            secret: "super-secret".into(),
            kind: CredentialKind::ApiKey,
            account_id: Some("acct".into()),
        };
        let debug = format!("{cred:?}");
        assert!(!debug.contains("super-secret"));
        assert!(debug.contains("redacted"));
        assert_eq!(cred.expose_secret(), "super-secret");
    }

    #[tokio::test]
    async fn env_api_key_fallback() {
        let _lock = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("ZENMUX_API_KEY", "env-key") };
        let dir = tempdir();
        let store = CredentialStore::open(dir.path().join("auth.json")).unwrap();
        let cred = store.resolve(&model("zenmux")).await.unwrap().unwrap();
        assert_eq!(cred.expose_secret(), "env-key");
        // Stored beats env.
        store.set_api_key("zenmux", "stored-key").unwrap();
        let cred = store.resolve(&model("zenmux")).await.unwrap().unwrap();
        assert_eq!(cred.expose_secret(), "stored-key");
        unsafe { std::env::remove_var("ZENMUX_API_KEY") };
    }

    #[tokio::test]
    async fn local_provider_no_env_returns_none() {
        let _lock = ENV_LOCK.lock().unwrap();
        unsafe { std::env::remove_var("OLLAMA_API_KEY") };
        let dir = tempdir();
        let store = CredentialStore::open(dir.path().join("auth.json")).unwrap();
        assert!(store.resolve(&model("ollama")).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn anthropic_env_prefers_oauth_token() {
        let _lock = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("ANTHROPIC_OAUTH_TOKEN", "oauth-tok");
            std::env::set_var("ANTHROPIC_API_KEY", "api-tok");
            std::env::remove_var("CLAUDE_CODE_USE_FOUNDRY");
        }
        let dir = tempdir();
        let store = CredentialStore::open(dir.path().join("auth.json")).unwrap();
        let cred = store.resolve(&model("anthropic")).await.unwrap().unwrap();
        assert_eq!(cred.expose_secret(), "oauth-tok");
        unsafe {
            std::env::remove_var("ANTHROPIC_OAUTH_TOKEN");
            std::env::remove_var("ANTHROPIC_API_KEY");
        }
    }

    #[test]
    fn begin_login_builds_codex_url() {
        let dir = tempdir();
        let store = CredentialStore::open(dir.path().join("auth.json")).unwrap();
        let login = store.begin_login("openai-codex").unwrap();
        assert_eq!(login.redirect_uri, "http://localhost:1455/auth/callback");
        assert!(
            login
                .authorization_url
                .starts_with("https://auth.openai.com/oauth/authorize?")
        );
        assert!(
            login
                .authorization_url
                .contains("client_id=app_EMoamEEZ73f0CkXaXp7hrann")
        );
        assert!(
            login
                .authorization_url
                .contains("code_challenge_method=S256")
        );
        assert!(
            login
                .authorization_url
                .contains("codex_cli_simplified_flow=true")
        );
        assert!(login.authorization_url.contains("originator=omp"));
        // Debug must not leak verifier/state.
        let debug = format!("{login:?}");
        assert!(debug.contains("redacted"));
        assert!(!debug.contains(&login.state));
    }

    #[test]
    fn begin_login_builds_anthropic_url() {
        let dir = tempdir();
        let store = CredentialStore::open(dir.path().join("auth.json")).unwrap();
        let login = store.begin_login("anthropic").unwrap();
        assert_eq!(login.redirect_uri, "http://localhost:54545/callback");
        assert!(
            login
                .authorization_url
                .starts_with("https://claude.ai/oauth/authorize?")
        );
        // Base64-decoded client id per the KDL `encoding="base64"`.
        assert!(
            login
                .authorization_url
                .contains("client_id=9d1c250a-e61b-44d9-88ed-5944d1962f5e")
        );
        assert!(login.authorization_url.contains("code=true"));
        assert!(login.authorization_url.contains("code_challenge="));
    }

    #[test]
    fn unsupported_login_errors() {
        let dir = tempdir();
        let store = CredentialStore::open(dir.path().join("auth.json")).unwrap();
        assert!(matches!(
            store.begin_login("openai"),
            Err(AuthError::UnsupportedProvider(_))
        ));
    }

    /// Minimal HTTP stub that records request bodies and always answers
    /// `body`. Returns `(url, recorded_requests)`.
    async fn spawn_token_stub(body: String) -> (String, Arc<StdMutex<Vec<String>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let requests: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
        let seen = requests.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let mut reader = BufReader::new(stream);
                let mut request = String::new();
                let mut line = String::new();
                let mut content_length = 0usize;
                loop {
                    line.clear();
                    if reader.read_line(&mut line).await.unwrap_or(0) == 0 {
                        break;
                    }
                    request.push_str(&line);
                    if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
                        content_length = v.trim().parse().unwrap_or(0);
                    }
                    if line.trim().is_empty() {
                        break;
                    }
                }
                let mut bodybuf = vec![0u8; content_length];
                let _ = reader.read_exact(&mut bodybuf).await;
                request.push_str(&String::from_utf8_lossy(&bodybuf));
                seen.lock().unwrap().push(request);
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = reader.get_mut().write_all(response.as_bytes()).await;
            }
        });
        (format!("http://127.0.0.1:{port}/token"), requests)
    }

    /// A JWT access token carrying the Codex claim set so the profile hook can
    /// mint account/org identity without network calls.
    fn codex_jwt(account: &str, plan: &str) -> String {
        let header = B64URL.encode(r#"{"alg":"none"}"#);
        let payload = B64URL.encode(format!(
            r#"{{"https://api.openai.com/auth":{{"chatgpt_account_id":"{account}","chatgpt_plan_type":"{plan}"}}}}"#
        ));
        format!("{header}.{payload}.sig")
    }

    #[tokio::test]
    async fn refresh_rotates_and_persists() {
        let (url, requests) = spawn_token_stub(
            r#"{"access_token":"new-access","refresh_token":"new-refresh","expires_in":3600}"#
                .to_string(),
        )
        .await;
        let dir = tempdir();
        let path = dir.path().join("auth.json");
        let mut store = CredentialStore::open(&path).unwrap();
        store.override_token_endpoint("openai-codex", url);
        // Seed an expired OAuth credential directly.
        {
            let mut file = StoreFile::default();
            file.credentials.insert(
                "openai-codex".into(),
                StoredSecret::OAuth {
                    access: "old-access".into(),
                    refresh: "old-refresh".into(),
                    expires: 1,
                    account_id: Some("acct-1".into()),
                    email: None,
                    org_id: None,
                    org_name: None,
                    authorized_at: Some(42),
                },
            );
            write_store_atomic(&path, &file).unwrap();
        }
        let cred = store
            .resolve(&model("openai-codex"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(cred.expose_secret(), "new-access");
        assert_eq!(cred.kind, CredentialKind::OAuth);
        assert_eq!(cred.account_id.as_deref(), Some("acct-1"));
        // Form-encoded refresh grant per the KDL `body="form"`.
        let sent = requests.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert!(sent[0].contains("grant_type=refresh_token"));
        assert!(sent[0].contains("refresh_token=old-refresh"));
        drop(sent);
        // Rotation durably committed: a fresh store reads the rotated token.
        let reopened = CredentialStore::open(&path).unwrap();
        match reopened
            .read_store()
            .unwrap()
            .credentials
            .get("openai-codex")
            .unwrap()
        {
            StoredSecret::OAuth {
                refresh,
                access,
                authorized_at,
                ..
            } => {
                assert_eq!(refresh, "new-refresh");
                assert_eq!(access, "new-access");
                assert_eq!(*authorized_at, Some(42));
            }
            _ => panic!("expected oauth"),
        }
    }

    #[tokio::test]
    async fn anthropic_refresh_sends_beta_header_and_json_body() {
        let (url, requests) =
            spawn_token_stub(r#"{"access_token":"new-access","expires_in":28800}"#.to_string())
                .await;
        let dir = tempdir();
        let path = dir.path().join("auth.json");
        let mut store = CredentialStore::open(&path).unwrap();
        store.override_token_endpoint("anthropic", url);
        {
            let mut file = StoreFile::default();
            file.credentials.insert(
                "anthropic".into(),
                StoredSecret::OAuth {
                    access: "old".into(),
                    refresh: "rt".into(),
                    expires: 1,
                    account_id: Some("acct".into()),
                    email: Some("e@x".into()),
                    org_id: Some("org".into()),
                    org_name: Some("Org".into()),
                    authorized_at: None,
                },
            );
            write_store_atomic(&path, &file).unwrap();
        }
        let cred = store.resolve(&model("anthropic")).await.unwrap().unwrap();
        assert_eq!(cred.expose_secret(), "new-access");
        let sent = requests.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert!(
            sent[0]
                .to_lowercase()
                .contains("anthropic-beta: oauth-2025-04-20")
        );
        assert!(sent[0].contains(r#""grant_type":"refresh_token""#));
        // Unrotated refresh token is preserved.
        match store
            .read_store()
            .unwrap()
            .credentials
            .get("anthropic")
            .unwrap()
        {
            StoredSecret::OAuth { refresh, .. } => assert_eq!(refresh, "rt"),
            _ => panic!("expected oauth"),
        }
    }

    #[tokio::test]
    async fn concurrent_refreshes_single_exchange() {
        let (url, requests) = spawn_token_stub(
            r#"{"access_token":"a","refresh_token":"r2","expires_in":3600}"#.to_string(),
        )
        .await;
        let dir = tempdir();
        let path = dir.path().join("auth.json");
        let mut store = CredentialStore::open(&path).unwrap();
        store.override_token_endpoint("openai-codex", url);
        {
            let mut file = StoreFile::default();
            file.credentials.insert(
                "openai-codex".into(),
                StoredSecret::OAuth {
                    access: "old".into(),
                    refresh: "r1".into(),
                    expires: 1,
                    account_id: Some("a".into()),
                    email: None,
                    org_id: None,
                    org_name: None,
                    authorized_at: None,
                },
            );
            write_store_atomic(&path, &file).unwrap();
        }
        let store = Arc::new(store);
        let mut handles = Vec::new();
        for _ in 0..8 {
            let s = store.clone();
            handles.push(tokio::spawn(async move {
                s.resolve(&model("openai-codex")).await
            }));
        }
        for h in handles {
            h.await.unwrap().unwrap().unwrap();
        }
        // Exactly one refresh exchange ran; followers re-read under the lock.
        assert_eq!(requests.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn complete_login_state_mismatch_rejected() {
        let dir = tempdir();
        let store = CredentialStore::open(dir.path().join("auth.json")).unwrap();
        let login = store.begin_login("anthropic").unwrap();
        let bad = format!("{}?code=abc&state=wrong", login.redirect_uri);
        assert!(matches!(
            store.complete_login(login, &bad).await,
            Err(AuthError::Callback(_))
        ));
    }

    #[tokio::test]
    async fn complete_login_stores_codex_credential() {
        let body = format!(
            r#"{{"access_token":"{}","refresh_token":"rt","expires_in":3600,"id_token":"{}"}}"#,
            codex_jwt("acct-9", "pro"),
            codex_jwt("acct-9", "pro")
        );
        let (url, _requests) = spawn_token_stub(body).await;
        let dir = tempdir();
        let mut store = CredentialStore::open(dir.path().join("auth.json")).unwrap();
        store.override_token_endpoint("openai-codex", url);
        let login = store.begin_login("openai-codex").unwrap();
        let callback = format!("{}?code=thecode&state={}", login.redirect_uri, login.state);
        store.complete_login(login, &callback).await.unwrap();
        let list = store.list().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].kind, CredentialKind::OAuth);
        assert_eq!(list[0].account_id.as_deref(), Some("acct-9"));
        assert_eq!(list[0].org_name.as_deref(), Some("pro"));
    }

    #[tokio::test]
    async fn login_cancel_aborts_callback_wait() {
        let dir = tempdir();
        let store = CredentialStore::open(dir.path().join("auth.json")).unwrap();
        let cancel = CancellationToken::new();
        let seen = Arc::new(StdMutex::new(String::new()));
        let seen2 = seen.clone();
        let c2 = cancel.clone();
        let task = tokio::spawn(async move {
            store
                .login(
                    "anthropic",
                    c2,
                    Arc::new(move |url| *seen2.lock().unwrap() = url.to_string()),
                )
                .await
        });
        for _ in 0..100 {
            if !seen.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(
            seen.lock()
                .unwrap()
                .starts_with("https://claude.ai/oauth/authorize?")
        );
        cancel.cancel();
        assert!(matches!(task.await.unwrap(), Err(AuthError::Cancelled)));
    }

    #[tokio::test]
    async fn loopback_callback_completes_login() {
        // Real loopback: `login` binds 1455, publishes the URL, receives the
        // browser redirect with code+state, exchanges against the stub, and
        // persists the OAuth credential.
        let body = format!(
            r#"{{"access_token":"{}","refresh_token":"rt","expires_in":3600}}"#,
            codex_jwt("acct-7", "pro")
        );
        let (url, _requests) = spawn_token_stub(body).await;
        let dir = tempdir();
        let path = dir.path().join("auth.json");
        let mut store = CredentialStore::open(&path).unwrap();
        store.override_token_endpoint("openai-codex", url);
        let seen = Arc::new(StdMutex::new(String::new()));
        let seen2 = seen.clone();
        let task = tokio::spawn(async move {
            store
                .login(
                    "openai-codex",
                    CancellationToken::new(),
                    Arc::new(move |u| *seen2.lock().unwrap() = u.to_string()),
                )
                .await
        });
        let mut auth_url = String::new();
        for _ in 0..200 {
            auth_url = seen.lock().unwrap().clone();
            if !auth_url.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(auth_url.starts_with("https://auth.openai.com/oauth/authorize?"));
        let params = query_params(auth_url.split('?').nth(1).unwrap());
        let state = params.get("state").unwrap().clone();
        let mut conn = tokio::net::TcpStream::connect(("127.0.0.1", 1455))
            .await
            .unwrap();
        conn.write_all(
            format!(
                "GET /auth/callback?code=abc&state={state} HTTP/1.1\r\nhost: localhost\r\n\r\n"
            )
            .as_bytes(),
        )
        .await
        .unwrap();
        let mut response = Vec::new();
        conn.read_to_end(&mut response).await.unwrap();
        assert!(String::from_utf8_lossy(&response).starts_with("HTTP/1.1 200"));
        task.await.unwrap().unwrap();
        let list = CredentialStore::open(&path).unwrap().list().unwrap();
        assert_eq!(list[0].kind, CredentialKind::OAuth);
        assert_eq!(list[0].account_id.as_deref(), Some("acct-7"));
    }

    #[test]
    fn pkce_verifier_challenge_relation() {
        let (verifier, challenge) = generate_pkce();
        assert_eq!(
            challenge,
            B64URL.encode(sha2::Sha256::digest(verifier.as_bytes()))
        );
        assert_eq!(verifier.len(), 128);
    }

    #[test]
    fn template_substitution() {
        let vars = vec![("state", "s1"), ("client_id", "c")];
        assert_eq!(
            template("x={state}&y={client_id}&z={unknown}", &vars),
            "x=s1&y=c&z="
        );
        assert_eq!(template("no {UPPER} match", &vars), "no {UPPER} match");
    }
}
