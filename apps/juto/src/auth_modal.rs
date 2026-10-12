//! Provider-first login with grouped authentication methods and background
//! credential I/O. Backend credential namespaces remain independent.

use std::collections::BTreeSet;
use std::sync::Arc;

use gpui::{
    App, ClickEvent, Context, Entity, FocusHandle, Focusable, KeyBinding, ScrollHandle,
    SharedString, Subscription, Task, Window, actions, div, prelude::*, px, rgb,
};
use juto_ai::{CredentialKind, CredentialSummary};
use juto_runtime::Runtime;
use tokio_util::sync::CancellationToken;

use crate::text_input::TextInput;

actions!(auth_modal, [CloseAuthModal]);

pub fn init(cx: &mut App) {
    cx.bind_keys([KeyBinding::new("escape", CloseAuthModal, Some("AuthModal"))]);
}

/// Authentication method classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthMethodKind {
    OAuth,
    ApiKey,
    Local,
}

/// A specific authentication mechanism supported by a provider.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthMethod {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub backend_id: &'static str,
    pub kind: AuthMethodKind,
    pub env_var: Option<&'static str>,
}

/// Top-level provider catalog definition for presentation and selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderDefinition {
    pub id: &'static str,
    pub name: &'static str,
    pub tagline: &'static str,
    pub methods: &'static [AuthMethod],
}

/// Resolved connection status for a specific authentication method.
#[derive(Clone, Debug, PartialEq)]
pub enum MethodStatus {
    /// Active stored OAuth session with optional identity details.
    StoredOAuth {
        email: Option<String>,
        org_name: Option<String>,
    },
    /// Stored API key with optional account identifier.
    StoredApiKey { account_id: Option<String> },
    /// Discovered via environment variable fallback.
    Environment { env_var: &'static str },
    /// Local daemon (always ready without credentials).
    LocalReady,
    /// Method is not currently configured.
    NotConfigured,
}

/// Cached redacted authentication status populated asynchronously.
#[derive(Clone, Debug, Default)]
pub struct CachedAuthStatus {
    pub stored_summaries: Vec<CredentialSummary>,
    pub available_backends: BTreeSet<String>,
    pub loaded: bool,
}

/// Events emitted by the AuthModal.
#[derive(Clone, Debug)]
pub enum AuthModalEvent {
    Close,
    CredentialsChanged,
}

/// Supported provider registry matching primary upstream OMP capabilities.
pub const PROVIDERS: &[ProviderDefinition] = &[
    ProviderDefinition {
        id: "openai",
        name: "OpenAI",
        tagline: "OpenAI Platform models and ChatGPT / Codex",
        methods: &[
            AuthMethod {
                id: "openai-codex-oauth",
                name: "ChatGPT / Codex OAuth",
                description: "Sign in with your ChatGPT account in the system browser.",
                backend_id: "openai-codex",
                kind: AuthMethodKind::OAuth,
                env_var: Some("OPENAI_CODEX_OAUTH_TOKEN"),
            },
            AuthMethod {
                id: "openai-api-key",
                name: "OpenAI API Key",
                description: "Standard OpenAI Platform developer API key (sk-...)",
                backend_id: "openai",
                kind: AuthMethodKind::ApiKey,
                env_var: Some("OPENAI_API_KEY"),
            },
        ],
    },
    ProviderDefinition {
        id: "anthropic",
        name: "Anthropic",
        tagline: "Claude models and Claude account access",
        methods: &[
            AuthMethod {
                id: "anthropic-oauth",
                name: "Claude Account OAuth",
                description: "Sign in with your Claude account in the system browser.",
                backend_id: "anthropic",
                kind: AuthMethodKind::OAuth,
                env_var: None,
            },
            AuthMethod {
                id: "anthropic-api-key",
                name: "Anthropic API Key",
                description: "Anthropic Console API key (sk-ant-...)",
                backend_id: "anthropic",
                kind: AuthMethodKind::ApiKey,
                env_var: Some("ANTHROPIC_API_KEY"),
            },
        ],
    },
    ProviderDefinition {
        id: "google",
        name: "Google Gemini",
        tagline: "Gemini models through Google AI Studio",
        methods: &[AuthMethod {
            id: "google-api-key",
            name: "Gemini API Key",
            description: "API key from Google AI Studio",
            backend_id: "google",
            kind: AuthMethodKind::ApiKey,
            env_var: Some("GEMINI_API_KEY"),
        }],
    },
    ProviderDefinition {
        id: "ollama",
        name: "Ollama (Local)",
        tagline: "Local open-weights daemon (http://localhost:11434)",
        methods: &[AuthMethod {
            id: "ollama-local",
            name: "Local Daemon",
            description: "Local daemon running without authentication",
            backend_id: "ollama",
            kind: AuthMethodKind::Local,
            env_var: None,
        }],
    },
    ProviderDefinition {
        id: "openrouter",
        name: "OpenRouter",
        tagline: "Unified gateway to 100+ AI models",
        methods: &[AuthMethod {
            id: "openrouter-api-key",
            name: "OpenRouter API Key",
            description: "Unified API key from openrouter.ai (sk-or-...)",
            backend_id: "openrouter",
            kind: AuthMethodKind::ApiKey,
            env_var: Some("OPENROUTER_API_KEY"),
        }],
    },
    ProviderDefinition {
        id: "deepseek",
        name: "DeepSeek",
        tagline: "DeepSeek-V3, DeepSeek-R1 reasoning models",
        methods: &[AuthMethod {
            id: "deepseek-api-key",
            name: "DeepSeek API Key",
            description: "API key from platform.deepseek.com (sk-...)",
            backend_id: "deepseek",
            kind: AuthMethodKind::ApiKey,
            env_var: Some("DEEPSEEK_API_KEY"),
        }],
    },
    ProviderDefinition {
        id: "groq",
        name: "Groq",
        tagline: "Ultra-low latency LPU inference",
        methods: &[AuthMethod {
            id: "groq-api-key",
            name: "Groq API Key",
            description: "API key from console.groq.com (gsk_...)",
            backend_id: "groq",
            kind: AuthMethodKind::ApiKey,
            env_var: Some("GROQ_API_KEY"),
        }],
    },
    ProviderDefinition {
        id: "mistral",
        name: "Mistral AI",
        tagline: "Mistral Large, Codestral, Pixtral",
        methods: &[AuthMethod {
            id: "mistral-api-key",
            name: "Mistral API Key",
            description: "API key from console.mistral.ai",
            backend_id: "mistral",
            kind: AuthMethodKind::ApiKey,
            env_var: Some("MISTRAL_API_KEY"),
        }],
    },
    ProviderDefinition {
        id: "xai",
        name: "xAI",
        tagline: "Grok models",
        methods: &[AuthMethod {
            id: "xai-api-key",
            name: "xAI API Key",
            description: "API key from console.x.ai (xai-...)",
            backend_id: "xai",
            kind: AuthMethodKind::ApiKey,
            env_var: Some("XAI_API_KEY"),
        }],
    },
    ProviderDefinition {
        id: "azure",
        name: "Azure OpenAI",
        tagline: "Enterprise Azure OpenAI Service",
        methods: &[AuthMethod {
            id: "azure-api-key",
            name: "Azure OpenAI Key",
            description: "API key from Microsoft Azure Portal",
            backend_id: "azure",
            kind: AuthMethodKind::ApiKey,
            env_var: Some("AZURE_OPENAI_API_KEY"),
        }],
    },
    ProviderDefinition {
        id: "cohere",
        name: "Cohere",
        tagline: "Command R+ and Command R models",
        methods: &[AuthMethod {
            id: "cohere-api-key",
            name: "Cohere API Key",
            description: "API key from dashboard.cohere.com",
            backend_id: "cohere",
            kind: AuthMethodKind::ApiKey,
            env_var: Some("COHERE_API_KEY"),
        }],
    },
    ProviderDefinition {
        id: "together",
        name: "Together AI",
        tagline: "Open-source model hosting and inference",
        methods: &[AuthMethod {
            id: "together-api-key",
            name: "Together AI Key",
            description: "API key from api.together.ai",
            backend_id: "together",
            kind: AuthMethodKind::ApiKey,
            env_var: Some("TOGETHER_API_KEY"),
        }],
    },
    ProviderDefinition {
        id: "perplexity",
        name: "Perplexity",
        tagline: "Sonar search-grounded models",
        methods: &[AuthMethod {
            id: "perplexity-api-key",
            name: "Perplexity API Key",
            description: "API key from perplexity.ai (pplx-...)",
            backend_id: "perplexity",
            kind: AuthMethodKind::ApiKey,
            env_var: Some("PERPLEXITY_API_KEY"),
        }],
    },
    ProviderDefinition {
        id: "devin",
        name: "Devin",
        tagline: "SWE models via Devin account or API key",
        methods: &[
            AuthMethod {
                id: "devin-oauth",
                name: "Devin Account OAuth",
                description: "Sign in with your Devin account in the system browser.",
                backend_id: "devin",
                kind: AuthMethodKind::OAuth,
                env_var: None,
            },
            AuthMethod {
                id: "devin-api-key",
                name: "Devin API Key",
                description: "Service credential API key from app.devin.ai",
                backend_id: "devin",
                kind: AuthMethodKind::ApiKey,
                env_var: Some("DEVIN_API_KEY"),
            },
        ],
    },
];

pub struct AuthModal {
    focus_handle: FocusHandle,
    runtime: Runtime,
    tokio_handle: tokio::runtime::Handle,
    providers: Vec<&'static ProviderDefinition>,
    search_input: Entity<TextInput>,
    api_key_input: Entity<TextInput>,
    selected_provider_id: String,
    selected_method_id: String,
    status_cache: CachedAuthStatus,
    status_message: Option<(SharedString, bool)>,
    in_flight_oauth: Option<String>,
    in_flight_mutation: bool,
    oauth_task: Option<Task<()>>,
    refresh_task: Option<Task<()>>,
    mutation_task: Option<Task<()>>,
    cancel_token: Option<CancellationToken>,
    scroll_handle: ScrollHandle,
    sidebar_scroll_handle: ScrollHandle,
    _subscriptions: Vec<Subscription>,
}

impl gpui::EventEmitter<AuthModalEvent> for AuthModal {}

impl Drop for AuthModal {
    fn drop(&mut self) {
        if let Some(cancel) = self.cancel_token.take() {
            cancel.cancel();
        }
    }
}

impl AuthModal {
    pub fn new(
        runtime: Runtime,
        tokio_handle: tokio::runtime::Handle,
        cx: &mut Context<Self>,
    ) -> Self {
        let search_input = cx.new(|cx| TextInput::new(cx).with_placeholder("Search providers…"));

        let api_key_input = cx.new(|cx| {
            TextInput::new(cx)
                .with_placeholder("Enter API key…")
                .with_masked(true)
        });

        let obs_search = cx.observe(&search_input, |_, _, cx| cx.notify());
        // Providers surface when the catalog carries chat models for them —
        // even ones whose transport is still pending, like devin — so login
        // can be configured before the wire protocol lands.
        let supported: BTreeSet<&str> = runtime
            .catalog()
            .iter()
            .filter(|model| matches!(model.kind.as_str(), "chat" | "tiny"))
            .map(|model| model.provider.as_str())
            .collect();
        let providers = PROVIDERS
            .iter()
            .filter(|provider| {
                provider
                    .methods
                    .iter()
                    .any(|method| supported.contains(method.backend_id))
            })
            .collect();

        let mut modal = Self {
            focus_handle: cx.focus_handle(),
            runtime,
            tokio_handle,
            providers,
            search_input,
            api_key_input,
            selected_provider_id: "openai".to_string(),
            selected_method_id: "openai-codex-oauth".to_string(),
            status_cache: CachedAuthStatus::default(),
            status_message: None,
            in_flight_oauth: None,
            in_flight_mutation: false,
            oauth_task: None,
            refresh_task: None,
            mutation_task: None,
            cancel_token: None,
            scroll_handle: ScrollHandle::new(),
            sidebar_scroll_handle: ScrollHandle::new(),
            _subscriptions: vec![obs_search],
        };

        modal.refresh_credentials(cx);
        modal
    }

    /// Refresh credential status off the GPUI thread.
    ///
    /// Reads redacted stored summaries and the environment cascade using
    /// `spawn_blocking`, then updates the UI cache.
    pub fn refresh_credentials(&mut self, cx: &mut Context<Self>) {
        let credentials = self.runtime.credentials();
        let tokio_handle = self.tokio_handle.clone();
        self.status_cache = CachedAuthStatus::default();
        let load_task = tokio_handle.spawn_blocking(move || {
            let summaries = credentials.list()?;
            let available =
                credentials.available_providers(PROVIDERS.iter().flat_map(|provider| {
                    provider.methods.iter().map(|method| method.backend_id)
                }))?;
            Ok::<_, juto_ai::AuthError>((summaries, available))
        });

        self.refresh_task = Some(cx.spawn(async move |this, cx| {
            let result = load_task.await;
            let _ = this.update(cx, |this, cx| {
                this.status_cache.loaded = true;
                match result {
                    Ok(Ok((summaries, available))) => {
                        this.status_cache.stored_summaries = summaries;
                        this.status_cache.available_backends = available;
                    }
                    Ok(Err(error)) => {
                        this.status_message =
                            Some((format!("Could not read credentials: {error}").into(), true));
                    }
                    Err(error) => {
                        this.status_message =
                            Some((format!("Credential lookup failed: {error}").into(), true));
                    }
                }
                this.refresh_task = None;
                cx.notify();
            });
        }));
    }

    pub fn focus(&self, window: &mut Window, cx: &App) {
        self.search_input.read(cx).focus_handle(cx).focus(window);
    }

    pub fn resolve_method_status(&self, method: &AuthMethod) -> MethodStatus {
        if method.kind == AuthMethodKind::Local {
            return MethodStatus::LocalReady;
        }

        // 1. Check for stored credentials matching method's backend_id
        let matching_summary = self
            .status_cache
            .stored_summaries
            .iter()
            .find(|s| s.provider.eq_ignore_ascii_case(method.backend_id));

        if let Some(s) = matching_summary {
            if method.kind == AuthMethodKind::OAuth && s.kind == CredentialKind::OAuth {
                return MethodStatus::StoredOAuth {
                    email: s.email.clone(),
                    org_name: s.org_name.clone(),
                };
            }
            if method.kind == AuthMethodKind::ApiKey && s.kind == CredentialKind::ApiKey {
                return MethodStatus::StoredApiKey {
                    account_id: s.account_id.clone(),
                };
            }
        }

        // 2. Check environment credentials
        if let Some(env_name) = method.env_var {
            let in_available = self
                .status_cache
                .available_backends
                .contains(method.backend_id);
            let has_other_stored = matching_summary.is_some();

            if !has_other_stored && in_available {
                return MethodStatus::Environment { env_var: env_name };
            }
        }

        MethodStatus::NotConfigured
    }

    pub fn resolve_provider_badge(
        &self,
        provider: &ProviderDefinition,
    ) -> (&'static str, gpui::Rgba, gpui::Rgba) {
        if !self.status_cache.loaded {
            return ("Loading…", rgb(0xaaaaaa), rgb(0x27272a));
        }
        let mut has_oauth = false;
        let mut has_key = false;
        let mut has_env = false;
        let mut is_local = false;

        for method in provider.methods {
            match self.resolve_method_status(method) {
                MethodStatus::StoredOAuth { .. } => has_oauth = true,
                MethodStatus::StoredApiKey { .. } => has_key = true,
                MethodStatus::Environment { .. } => has_env = true,
                MethodStatus::LocalReady => is_local = true,
                MethodStatus::NotConfigured => {}
            }
        }

        if has_oauth {
            ("OAuth ✓", rgb(0xd8b4fe), rgb(0x2e1065))
        } else if has_key {
            ("Key ✓", rgb(0x86efac), rgb(0x14532d))
        } else if has_env {
            ("Env ✓", rgb(0x93c5fd), rgb(0x1e3a8a))
        } else if is_local {
            ("Local", rgb(0x5eead4), rgb(0x134e4a))
        } else {
            ("Not set", rgb(0x71717a), rgb(0x27272a))
        }
    }

    pub fn selected_provider(&self) -> Option<&'static ProviderDefinition> {
        self.providers
            .iter()
            .copied()
            .find(|p| p.id == self.selected_provider_id)
    }

    pub fn selected_method(&self) -> Option<&'static AuthMethod> {
        let provider = self.selected_provider()?;
        provider
            .methods
            .iter()
            .find(|m| m.id == self.selected_method_id)
            .or_else(|| provider.methods.first())
    }

    pub fn filtered_providers(&self, cx: &App) -> Vec<&'static ProviderDefinition> {
        let query = self.search_input.read(cx).text().trim().to_lowercase();
        if query.is_empty() {
            return self.providers.clone();
        }

        self.providers
            .iter()
            .copied()
            .filter(|p| {
                p.name.to_lowercase().contains(&query)
                    || p.id.to_lowercase().contains(&query)
                    || p.tagline.to_lowercase().contains(&query)
                    || p.methods.iter().any(|m| {
                        m.name.to_lowercase().contains(&query)
                            || m.backend_id.to_lowercase().contains(&query)
                            || m.env_var
                                .is_some_and(|ev| ev.to_lowercase().contains(&query))
                    })
            })
            .collect()
    }

    fn select_provider(&mut self, id: String, cx: &mut Context<Self>) {
        if self.in_flight_mutation {
            return;
        }
        if self.selected_provider_id != id {
            if self.in_flight_oauth.is_some() {
                self.cancel_oauth_flow(cx);
            }
            self.selected_provider_id = id.clone();
            if let Some(provider) = PROVIDERS.iter().find(|p| p.id == id)
                && let Some(first_method) = provider.methods.first()
            {
                self.selected_method_id = first_method.id.to_string();
                let placeholder = format!("Enter {}…", first_method.name);
                self.api_key_input.update(cx, |input, cx| {
                    input.clear(cx);
                    input.set_masked(true, cx);
                    input.set_placeholder(placeholder, cx);
                });
            }
            self.status_message = None;
            cx.notify();
        }
    }

    fn select_method(&mut self, method_id: String, cx: &mut Context<Self>) {
        if self.in_flight_mutation {
            return;
        }
        if self.selected_method_id != method_id {
            if self.in_flight_oauth.is_some() {
                self.cancel_oauth_flow(cx);
            }
            self.selected_method_id = method_id.clone();
            if let Some(method) = self.selected_method() {
                let placeholder = format!("Enter {}…", method.name);
                self.api_key_input.update(cx, |input, cx| {
                    input.clear(cx);
                    input.set_masked(true, cx);
                    input.set_placeholder(placeholder, cx);
                });
            }
            self.status_message = None;
            cx.notify();
        }
    }

    fn toggle_api_key_mask(&mut self, cx: &mut Context<Self>) {
        let is_masked = self.api_key_input.read(cx).is_masked();
        self.api_key_input.update(cx, |input, cx| {
            input.set_masked(!is_masked, cx);
        });
    }

    fn save_api_key(&mut self, cx: &mut Context<Self>) {
        let key = self.api_key_input.read(cx).text().trim().to_string();
        if self.in_flight_mutation || self.in_flight_oauth.is_some() {
            return;
        }
        if key.is_empty() {
            self.status_message = Some(("Please enter an API key to save".into(), true));
            cx.notify();
            return;
        }

        let method = match self.selected_method() {
            Some(m) => m,
            None => return,
        };
        let backend_id = method.backend_id.to_string();
        let method_name = method.name.to_string();

        self.in_flight_mutation = true;
        self.status_message = Some((format!("Saving API key for {method_name}…").into(), false));
        cx.notify();

        let credentials = self.runtime.credentials();
        let tokio_handle = self.tokio_handle.clone();
        let save_task =
            tokio_handle.spawn_blocking(move || credentials.set_api_key(&backend_id, &key));

        self.mutation_task = Some(cx.spawn(async move |this, cx| {
            let result = save_task.await;
            this.update(cx, |this, cx| {
                this.in_flight_mutation = false;
                match result {
                    Ok(Ok(())) => {
                        this.status_message = Some((
                            format!("✓ API key stored securely for {method_name}").into(),
                            false,
                        ));
                        this.api_key_input.update(cx, |input, cx| input.clear(cx));
                        cx.emit(AuthModalEvent::CredentialsChanged);
                        this.refresh_credentials(cx);
                    }
                    Ok(Err(err)) => {
                        this.status_message =
                            Some((format!("Failed to save key: {err}").into(), true));
                    }
                    Err(join_err) => {
                        this.status_message =
                            Some((format!("Save task failed: {join_err}").into(), true));
                    }
                }
                cx.notify();
            })
            .ok();
        }));
    }

    fn remove_credential(&mut self, backend_id: &'static str, name: &str, cx: &mut Context<Self>) {
        if self.in_flight_mutation || self.in_flight_oauth.is_some() {
            return;
        }
        let backend_str = backend_id.to_string();
        let name_str = name.to_string();

        self.in_flight_mutation = true;
        self.status_message = Some((
            format!("Clearing credentials for {name_str}…").into(),
            false,
        ));
        cx.notify();

        let credentials = self.runtime.credentials();
        let tokio_handle = self.tokio_handle.clone();
        let remove_task = tokio_handle.spawn_blocking(move || credentials.remove(&backend_str));

        self.mutation_task = Some(cx.spawn(async move |this, cx| {
            let result = remove_task.await;
            this.update(cx, |this, cx| {
                this.in_flight_mutation = false;
                match result {
                    Ok(Ok(())) => {
                        this.status_message = Some((
                            format!("✓ Credentials cleared for {name_str}").into(),
                            false,
                        ));
                        this.api_key_input.update(cx, |input, cx| input.clear(cx));
                        cx.emit(AuthModalEvent::CredentialsChanged);
                        this.refresh_credentials(cx);
                    }
                    Ok(Err(err)) => {
                        this.status_message =
                            Some((format!("Failed to clear credentials: {err}").into(), true));
                    }
                    Err(join_err) => {
                        this.status_message =
                            Some((format!("Clear task failed: {join_err}").into(), true));
                    }
                }
                cx.notify();
            })
            .ok();
        }));
    }

    fn start_oauth_flow(
        &mut self,
        backend_id: &'static str,
        name: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.in_flight_oauth.is_some() || self.in_flight_mutation {
            return;
        }

        let backend_str = backend_id.to_string();
        let name_str = name.to_string();
        self.in_flight_oauth = Some(backend_str.clone());
        let cancel = CancellationToken::new();
        self.cancel_token = Some(cancel.clone());
        let credentials = self.runtime.credentials();
        let tokio_handle = self.tokio_handle.clone();

        self.status_message = Some((
            format!("Opening browser for {name_str}… Waiting for callback").into(),
            false,
        ));

        let oauth_future = tokio_handle.spawn(async move {
            let on_url = Arc::new(|url: &str| {
                let _ = std::process::Command::new("xdg-open").arg(url).spawn();
            });
            credentials.login(&backend_str, cancel, on_url).await
        });

        self.oauth_task = Some(cx.spawn_in(window, async move |this, cx| {
            let result = oauth_future.await;
            this.update_in(cx, |this, _window, cx| {
                this.in_flight_oauth = None;
                this.cancel_token = None;
                this.oauth_task = None;
                match result {
                    Ok(Ok(())) => {
                        this.status_message = Some((
                            format!("✓ Successfully logged in to {name_str} via OAuth!").into(),
                            false,
                        ));
                        cx.emit(AuthModalEvent::CredentialsChanged);
                        this.refresh_credentials(cx);
                    }
                    Ok(Err(err)) => {
                        this.status_message =
                            Some((format!("OAuth login failed: {err}").into(), true));
                    }
                    Err(join_err) => {
                        this.status_message =
                            Some((format!("OAuth task cancelled: {join_err}").into(), true));
                    }
                }
                cx.notify();
            })
            .ok();
        }));

        cx.notify();
    }

    fn cancel_oauth_flow(&mut self, cx: &mut Context<Self>) {
        if let Some(cancel) = self.cancel_token.take() {
            cancel.cancel();
        }
        self.in_flight_oauth = None;
        self.oauth_task = None;
        self.status_message = Some(("OAuth login cancelled".into(), true));
        cx.notify();
    }

    pub fn dismiss(&mut self, cx: &mut Context<Self>) {
        if self.in_flight_oauth.is_some() {
            self.cancel_oauth_flow(cx);
        }
        self.status_message = None;
        self.api_key_input.update(cx, |input, cx| {
            input.clear(cx);
            input.set_masked(true, cx);
        });
    }

    fn close(&mut self, _: &CloseAuthModal, _: &mut Window, cx: &mut Context<Self>) {
        self.dismiss(cx);
        cx.emit(AuthModalEvent::Close);
    }

    // --- RENDER HELPERS ---

    fn header(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .items_center()
            .justify_between()
            .px_4()
            .py_3()
            .border_b_1()
            .border_color(rgb(0x333333))
            .bg(rgb(0x202020))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(
                        div()
                            .text_base()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(rgb(0xffffff))
                            .child("Providers"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x888888))
                            .child("Connect an account or API key"),
                    ),
            )
            .child(
                div()
                    .id("close-auth-modal")
                    .px_2p5()
                    .py_1()
                    .rounded_sm()
                    .text_xs()
                    .text_color(rgb(0xaaaaaa))
                    .cursor_pointer()
                    .hover(|s| s.bg(rgb(0x303030)).text_color(rgb(0xffffff)))
                    .on_click(cx.listener(|this, _: &ClickEvent, window, cx| {
                        this.close(&CloseAuthModal, window, cx);
                    }))
                    .child("✕ Close (Esc)"),
            )
    }

    fn sidebar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let filtered = self.filtered_providers(cx);
        let selected_id = self.selected_provider_id.clone();
        let total_count = self.providers.len();

        div()
            .w(px(200.))
            .flex_none()
            .border_r_1()
            .border_color(rgb(0x2d2d2d))
            .bg(rgb(0x181818))
            .flex()
            .flex_col()
            .child(
                // Search Box
                div()
                    .p_2()
                    .border_b_1()
                    .border_color(rgb(0x262626))
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .px_2()
                            .py_1p5()
                            .rounded_md()
                            .border_1()
                            .border_color(rgb(0x3f3f46))
                            .bg(rgb(0x121214))
                            .child(self.search_input.clone()),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .px_1()
                            .text_size(px(10.))
                            .text_color(rgb(0x71717a))
                            .child(if filtered.len() == total_count {
                                format!("{total_count} providers")
                            } else {
                                format!("{}/{} matching", filtered.len(), total_count)
                            }),
                    ),
            )
            .child(
                // Provider List
                div()
                    .id("auth-provider-list")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .track_scroll(&self.sidebar_scroll_handle)
                    .p_2()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .when(filtered.is_empty(), |list| {
                        list.child(
                            div()
                                .px_3()
                                .py_4()
                                .text_center()
                                .text_xs()
                                .text_color(rgb(0x71717a))
                                .child("No matching providers"),
                        )
                    })
                    .children(filtered.into_iter().enumerate().map(|(index, provider)| {
                        let is_selected = provider.id == selected_id;
                        let id = provider.id.to_string();
                        let name = provider.name;
                        let (badge_text, badge_fg, badge_bg) =
                            self.resolve_provider_badge(provider);

                        div()
                            .id(("auth-sidebar-row", index))
                            .flex()
                            .items_center()
                            .justify_between()
                            .px_2p5()
                            .py_2()
                            .rounded_md()
                            .border_1()
                            .border_color(if is_selected {
                                rgb(0x444444)
                            } else {
                                rgb(0x222222)
                            })
                            .bg(if is_selected {
                                rgb(0x2a2a2a)
                            } else {
                                rgb(0x1d1d1d)
                            })
                            .cursor_pointer()
                            .hover(|s| if is_selected { s } else { s.bg(rgb(0x232323)) })
                            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                this.select_provider(id.clone(), cx);
                            }))
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(if is_selected {
                                        gpui::FontWeight::SEMIBOLD
                                    } else {
                                        gpui::FontWeight::NORMAL
                                    })
                                    .text_color(if is_selected {
                                        rgb(0xffffff)
                                    } else {
                                        rgb(0xd0d0d0)
                                    })
                                    .child(name),
                            )
                            .child(
                                div()
                                    .px_1p5()
                                    .py_0p5()
                                    .rounded_sm()
                                    .bg(badge_bg)
                                    .text_color(badge_fg)
                                    .text_size(px(10.))
                                    .child(badge_text),
                            )
                    })),
            )
    }

    fn detail_pane(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let provider = match self.selected_provider() {
            Some(p) => p,
            None => {
                return div()
                    .id("empty-auth-pane")
                    .flex_1()
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_xs()
                    .text_color(rgb(0x707070))
                    .child("Select a provider");
            }
        };

        let method = match self.selected_method() {
            Some(m) => m,
            None => {
                return div()
                    .id("empty-auth-method-pane")
                    .flex_1()
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_xs()
                    .text_color(rgb(0x707070))
                    .child("No method available for provider");
            }
        };

        let method_status = self.resolve_method_status(method);
        let in_flight_oauth = self.in_flight_oauth.as_deref() == Some(method.backend_id);
        let is_masked = self.api_key_input.read(cx).is_masked();

        let (provider_badge, badge_fg, badge_bg) = self.resolve_provider_badge(provider);

        div()
            .id("auth-detail-pane")
            .flex_1()
            .min_w_0()
            .min_h_0()
            .overflow_y_scroll()
            .track_scroll(&self.scroll_handle)
            .p_4()
            .flex()
            .flex_col()
            .gap_4()
            // Status Notification Banner
            .when_some(self.status_message.clone(), |pane, (msg, is_error)| {
                pane.child(
                    div()
                        .px_3()
                        .py_2()
                        .rounded_md()
                        .border_1()
                        .border_color(if is_error {
                            rgb(0x7f1d1d)
                        } else {
                            rgb(0x14532d)
                        })
                        .bg(if is_error {
                            rgb(0x450a0a)
                        } else {
                            rgb(0x052e16)
                        })
                        .text_xs()
                        .text_color(if is_error {
                            rgb(0xfca5a5)
                        } else {
                            rgb(0x86efac)
                        })
                        .child(msg),
                )
            })
            // Provider Header
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .gap_2()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .min_w_0()
                            .flex()
                            .flex_col()
                            .gap_0p5()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(
                                        div()
                                            .text_lg()
                                            .font_weight(gpui::FontWeight::BOLD)
                                            .text_color(rgb(0xffffff))
                                            .child(provider.name),
                                    )
                                    .child(
                                        div()
                                            .px_2()
                                            .py_0p5()
                                            .rounded_md()
                                            .bg(rgb(0x27272a))
                                            .text_xs()
                                            .text_color(rgb(0xa1a1aa))
                                            .child(provider.id),
                                    ),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(rgb(0xa1a1aa))
                                    .child(provider.tagline),
                            ),
                    )
                    .child(
                        div()
                            .px_2p5()
                            .py_1()
                            .rounded_md()
                            .bg(badge_bg)
                            .text_color(badge_fg)
                            .text_xs()
                            .font_weight(gpui::FontWeight::MEDIUM)
                            .child(provider_badge),
                    ),
            )
            // Method Selection Tabs (Segmented bar for providers with multiple methods)
            .when(provider.methods.len() > 1, |pane| {
                pane.child(
                    div()
                        .flex()
                        .flex_wrap()
                        .gap_2()
                        .p_1()
                        .rounded_lg()
                        .bg(rgb(0x1e1e20))
                        .border_1()
                        .border_color(rgb(0x2e2e32))
                        .children(provider.methods.iter().map(|m| {
                            let is_selected = m.id == method.id;
                            let m_id = m.id.to_string();
                            let status = self.resolve_method_status(m);
                            let status_chip = match status {
                                MethodStatus::StoredOAuth { .. } => {
                                    ("OAuth ✓", rgb(0xd8b4fe), rgb(0x2e1065))
                                }
                                MethodStatus::StoredApiKey { .. } => {
                                    ("Key ✓", rgb(0x86efac), rgb(0x14532d))
                                }
                                MethodStatus::Environment { .. } => {
                                    ("Env ✓", rgb(0x93c5fd), rgb(0x1e3a8a))
                                }
                                MethodStatus::LocalReady => ("Local", rgb(0x5eead4), rgb(0x134e4a)),
                                MethodStatus::NotConfigured => {
                                    ("Not set", rgb(0x71717a), rgb(0x27272a))
                                }
                            };

                            div()
                                .id(m.id)
                                .flex_1()
                                .min_w(px(240.))
                                .flex()
                                .items_center()
                                .justify_between()
                                .px_3()
                                .py_2()
                                .rounded_md()
                                .border_1()
                                .border_color(if is_selected {
                                    rgb(0x52525b)
                                } else {
                                    rgb(0x27272a)
                                })
                                .bg(if is_selected {
                                    rgb(0x27272a)
                                } else {
                                    rgb(0x18181b)
                                })
                                .cursor_pointer()
                                .hover(|s| if is_selected { s } else { s.bg(rgb(0x222225)) })
                                .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                    this.select_method(m_id.clone(), cx);
                                }))
                                .child(
                                    div()
                                        .text_xs()
                                        .font_weight(if is_selected {
                                            gpui::FontWeight::SEMIBOLD
                                        } else {
                                            gpui::FontWeight::NORMAL
                                        })
                                        .text_color(if is_selected {
                                            rgb(0xffffff)
                                        } else {
                                            rgb(0xd4d4d8)
                                        })
                                        .child(m.name),
                                )
                                .child(
                                    div()
                                        .px_1p5()
                                        .py_0p5()
                                        .rounded_sm()
                                        .bg(status_chip.2)
                                        .text_color(status_chip.1)
                                        .text_size(px(10.))
                                        .child(status_chip.0),
                                )
                        })),
                )
            })
            // Selected Method Configuration & Action Card
            .child(self.method_action_card(method, &method_status, in_flight_oauth, is_masked, cx))
    }

    fn method_action_card(
        &self,
        method: &'static AuthMethod,
        status: &MethodStatus,
        in_flight_oauth: bool,
        is_masked: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let backend_id = method.backend_id;
        let method_name = method.name;

        div()
            .w_full()
            .min_w_0()
            .flex()
            .flex_col()
            .gap_3()
            .p_4()
            .rounded_lg()
            .border_1()
            .border_color(match method.kind {
                AuthMethodKind::OAuth => rgb(0x3b0764),
                AuthMethodKind::ApiKey => rgb(0x27272a),
                AuthMethodKind::Local => rgb(0x1e3a8a),
            })
            .bg(match method.kind {
                AuthMethodKind::OAuth => rgb(0x190c29),
                AuthMethodKind::ApiKey => rgb(0x18181b),
                AuthMethodKind::Local => rgb(0x0f172a),
            })
            // Card Header
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .gap_2()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::BOLD)
                            .text_color(match method.kind {
                                AuthMethodKind::OAuth => rgb(0xc084fc),
                                AuthMethodKind::ApiKey => rgb(0xe4e4e7),
                                AuthMethodKind::Local => rgb(0x93c5fd),
                            })
                            .child(method.name.to_uppercase()),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x71717a))
                            .child(format!("Backend: {}", method.backend_id)),
                    ),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0xa1a1aa))
                    .child(method.description),
            )
            // Live Status Box
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .gap_2()
                    .items_center()
                    .justify_between()
                    .px_3()
                    .py_2()
                    .rounded_md()
                    .border_1()
                    .border_color(match status {
                        MethodStatus::StoredOAuth { .. } => rgb(0x7e22ce),
                        MethodStatus::StoredApiKey { .. } => rgb(0x166534),
                        MethodStatus::Environment { .. } => rgb(0x1e40af),
                        MethodStatus::LocalReady => rgb(0x0e7490),
                        MethodStatus::NotConfigured => rgb(0x3f3f46),
                    })
                    .bg(match status {
                        MethodStatus::StoredOAuth { .. } => rgb(0x2e1065),
                        MethodStatus::StoredApiKey { .. } => rgb(0x052e16),
                        MethodStatus::Environment { .. } => rgb(0x172554),
                        MethodStatus::LocalReady => rgb(0x083344),
                        MethodStatus::NotConfigured => rgb(0x27272a),
                    })
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .text_xs()
                            .text_color(match status {
                                MethodStatus::StoredOAuth { .. } => rgb(0xd8b4fe),
                                MethodStatus::StoredApiKey { .. } => rgb(0x86efac),
                                MethodStatus::Environment { .. } => rgb(0x93c5fd),
                                MethodStatus::LocalReady => rgb(0x67e8f9),
                                MethodStatus::NotConfigured => rgb(0xa1a1aa),
                            })
                            .child(match status {
                                MethodStatus::StoredOAuth { email, org_name } => {
                                    format!(
                                        "OAuth saved: {} (Org: {})",
                                        email.as_deref().unwrap_or("authenticated user"),
                                        org_name.as_deref().unwrap_or("default")
                                    )
                                }
                                MethodStatus::StoredApiKey { account_id } => {
                                    format!(
                                        "API Key configured ({})",
                                        account_id.as_deref().unwrap_or("stored securely")
                                    )
                                }
                                MethodStatus::Environment { env_var } => {
                                    format!("Connected via environment variable ({env_var})")
                                }
                                MethodStatus::LocalReady => {
                                    "Local service configuration; availability is not checked".to_string()
                                }
                                MethodStatus::NotConfigured => {
                                    "Not configured (not set in storage or environment)".to_string()
                                }
                            }),
                    )
                    // Clear / Remove Button if stored credentials exist
                    .when(
                        matches!(status, MethodStatus::StoredOAuth { .. } | MethodStatus::StoredApiKey { .. }),
                        |status_row| {
                            status_row.child(
                                div()
                                    .id("clear-credential-btn")
                                    .px_2p5()
                                    .py_1()
                                    .rounded_sm()
                                    .border_1()
                                    .border_color(rgb(0x7f1d1d))
                                    .bg(rgb(0x450a0a))
                                    .text_xs()
                                    .text_color(rgb(0xfca5a5))
                                    .cursor_pointer()
                                    .hover(|s| s.bg(rgb(0x7f1d1d)))
                                    .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                        this.remove_credential(backend_id, method_name, cx);
                                    }))
                                    .child("Remove Stored"),
                            )
                        },
                    ),
            )
            // Interactive Controls per Method Kind
            .child(match method.kind {
                AuthMethodKind::OAuth => {
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .when(!in_flight_oauth, |actions| {
                            actions.child(
                                div()
                                    .flex()
                                    .gap_2()
                                    .child(
                                        div()
                                            .id("trigger-oauth-btn")
                                            .flex()
                                            .items_center()
                                            .gap_1p5()
                                            .px_4()
                                            .py_2()
                                            .rounded_md()
                                            .bg(rgb(0x7e22ce))
                                            .text_xs()
                                            .font_weight(gpui::FontWeight::MEDIUM)
                                            .text_color(rgb(0xffffff))
                                            .cursor_pointer()
                                            .hover(|s| s.bg(rgb(0x9333ea)))
                                            .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                                                this.start_oauth_flow(backend_id, method_name, window, cx);
                                            }))
                                            .child("Sign in with browser"),
                                    ),
                            )
                        })
                        .when(in_flight_oauth, |actions| {
                            actions.child(
                                div()
                                    .flex()
                                    .flex_wrap()
                                    .items_center()
                                    .gap_3()
                                    .child(
                                        div()
                                            .px_3()
                                            .py_2()
                                            .rounded_md()
                                            .bg(rgb(0x581c87))
                                            .text_xs()
                                            .text_color(rgb(0xe9d5ff))
                                            .child("⏳ Waiting for browser authorization callback…"),
                                    )
                                    .child(
                                        div()
                                            .id("cancel-oauth-btn")
                                            .px_3()
                                            .py_2()
                                            .rounded_md()
                                            .border_1()
                                            .border_color(rgb(0x7f1d1d))
                                            .bg(rgb(0x450a0a))
                                            .text_xs()
                                            .text_color(rgb(0xfca5a5))
                                            .cursor_pointer()
                                            .hover(|s| s.bg(rgb(0x7f1d1d)))
                                            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                                this.cancel_oauth_flow(cx);
                                            }))
                                            .child("Cancel Login"),
                                    ),
                            )
                        })
                        .child(
                            div()
                                .text_xs()
                                .text_color(rgb(0x71717a))
                                .child("Starts a local loopback server, launches your system browser to sign in, and stores refreshed tokens with mode 0600."),
                        )
                }
                AuthMethodKind::ApiKey => {
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(
                            div()
                                .flex()
                                .flex_wrap()
                                .gap_2()
                                .child(
                                    div()
                                        .flex_1()
                                        .min_w(px(160.))
                                        .p_1()
                                        .rounded_md()
                                        .border_1()
                                        .border_color(rgb(0x3f3f46))
                                        .bg(rgb(0x121214))
                                        .child(self.api_key_input.clone()),
                                )
                                .child(
                                    div()
                                        .id("toggle-mask-auth")
                                        .flex()
                                        .items_center()
                                        .justify_center()
                                        .px_3()
                                        .rounded_md()
                                        .border_1()
                                        .border_color(rgb(0x3f3f46))
                                        .bg(rgb(0x27272a))
                                        .text_xs()
                                        .text_color(rgb(0xd4d4d8))
                                        .cursor_pointer()
                                        .hover(|s| s.bg(rgb(0x3f3f46)))
                                        .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                            this.toggle_api_key_mask(cx);
                                        }))
                                        .child(if is_masked { "Show" } else { "Hide" }),
                                )
                                .child(
                                    div()
                                        .id("save-auth-key-btn")
                                        .min_h(px(36.))
                                        .flex()
                                        .items_center()
                                        .justify_center()
                                        .px_4()
                                        .rounded_md()
                                        .bg(if self.in_flight_mutation {
                                            rgb(0x166534)
                                        } else {
                                            rgb(0x15803d)
                                        })
                                        .text_xs()
                                        .font_weight(gpui::FontWeight::MEDIUM)
                                        .text_color(rgb(0xffffff))
                                        .cursor_pointer()
                                        .hover(|s| s.bg(rgb(0x16a34a)))
                                        .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                            this.save_api_key(cx);
                                        }))
                                        .child(if self.in_flight_mutation {
                                            "Saving…"
                                        } else {
                                            "Save Key"
                                        }),
                                ),
                        )
                        .child(
                            div()
                                .flex()
                                .flex_wrap()
                                .gap_2()
                                .items_center()
                                .justify_between()
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(rgb(0x71717a))
                                        .child("Stored with mode 0600 and atomic replacement."),
                                )
                                .when_some(method.env_var, |row, env| {
                                    row.child(
                                        div()
                                            .text_xs()
                                            .text_color(rgb(0x60a5fa))
                                            .child(format!("Environment policy: {env}")),
                                    )
                                }),
                        )
                }
                AuthMethodKind::Local => {
                    div()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(
                            div()
                                .text_xs()
                                .text_color(rgb(0x93c5fd))
                                .child("Ollama is a local daemon running on your workstation (default: http://localhost:11434)."),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(rgb(0x71717a))
                                .child("Ensure the daemon is running with `ollama serve` and pull desired models with `ollama run <model>`."),
                        )
                }
            })
    }
}

impl Focusable for AuthModal {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for AuthModal {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .key_context("AuthModal")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(Self::close))
            .flex()
            .flex_col()
            .w(px(860.))
            .max_w(window.viewport_size().width - px(48.))
            .h(px(600.))
            .max_h(window.viewport_size().height - px(48.))
            .min_w_0()
            .min_h_0()
            .rounded_lg()
            .border_1()
            .border_color(rgb(0x4a4a4a))
            .bg(rgb(0x141414))
            .shadow_xl()
            .text_color(rgb(0xe4e4e7))
            .child(self.header(cx))
            .child(
                div()
                    .flex()
                    .flex_1()
                    .min_h_0()
                    .child(self.sidebar(cx))
                    .child(self.detail_pane(cx)),
            )
    }
}
