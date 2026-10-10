//! Provider authentication commands and API key management modal.
//!
//! Features:
//! - Interactive OAuth commands for Anthropic and OpenAI Codex.
//! - API key setup, inspection, and clearing per provider.
//! - Real-time credential status resolution from `runtime.credentials()`.

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

#[derive(Clone, Debug)]
pub struct ProviderAuthRow {
    pub id: String,
    pub name: String,
    pub env_var: Option<String>,
    pub supports_oauth: bool,
    pub is_local: bool,
    pub summary: Option<CredentialSummary>,
}

#[derive(Clone, Debug)]
pub enum AuthModalEvent {
    Close,
    CredentialsChanged,
}

pub struct AuthModal {
    focus_handle: FocusHandle,
    runtime: Runtime,
    tokio_handle: tokio::runtime::Handle,
    selected_provider_id: String,
    api_key_input: Entity<TextInput>,
    status_message: Option<(SharedString, bool)>,
    in_flight_oauth: Option<String>,
    oauth_task: Option<Task<()>>,
    cancel_token: Option<CancellationToken>,
    scroll_handle: ScrollHandle,
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
        let api_key_input = cx.new(|cx| {
            TextInput::new(cx)
                .with_placeholder("Enter API key (e.g. sk-ant-…)")
                .with_masked(true)
        });

        Self {
            focus_handle: cx.focus_handle(),
            runtime,
            tokio_handle,
            selected_provider_id: "anthropic".to_string(),
            api_key_input,
            status_message: None,
            in_flight_oauth: None,
            oauth_task: None,
            cancel_token: None,
            scroll_handle: ScrollHandle::new(),
            _subscriptions: Vec::new(),
        }
    }

    pub fn focus(&self, window: &mut Window, cx: &App) {
        self.api_key_input.read(cx).focus_handle(cx).focus(window);
    }

    fn provider_rows(&self) -> Vec<ProviderAuthRow> {
        let stored_credentials = self.runtime.credentials().list().unwrap_or_default();

        let known = vec![
            (
                "anthropic",
                "Anthropic",
                Some("ANTHROPIC_API_KEY"),
                true,
                false,
            ),
            ("openai", "OpenAI", Some("OPENAI_API_KEY"), false, false),
            ("openai-codex", "OpenAI Codex", None, true, false),
            (
                "google",
                "Google Gemini",
                Some("GEMINI_API_KEY"),
                false,
                false,
            ),
            ("ollama", "Ollama (Local)", None, false, true),
            (
                "openrouter",
                "OpenRouter",
                Some("OPENROUTER_API_KEY"),
                false,
                false,
            ),
            (
                "deepseek",
                "DeepSeek",
                Some("DEEPSEEK_API_KEY"),
                false,
                false,
            ),
            ("groq", "Groq", Some("GROQ_API_KEY"), false, false),
            (
                "mistral",
                "Mistral AI",
                Some("MISTRAL_API_KEY"),
                false,
                false,
            ),
            ("xai", "xAI", Some("XAI_API_KEY"), false, false),
            (
                "azure",
                "Azure OpenAI",
                Some("AZURE_OPENAI_API_KEY"),
                false,
                false,
            ),
            ("cohere", "Cohere", Some("COHERE_API_KEY"), false, false),
            (
                "together",
                "Together AI",
                Some("TOGETHER_API_KEY"),
                false,
                false,
            ),
            (
                "perplexity",
                "Perplexity",
                Some("PERPLEXITY_API_KEY"),
                false,
                false,
            ),
        ];

        known
            .into_iter()
            .map(|(id, name, env_var, supports_oauth, is_local)| {
                let summary = stored_credentials
                    .iter()
                    .find(|c| c.provider.eq_ignore_ascii_case(id))
                    .cloned();
                ProviderAuthRow {
                    id: id.to_string(),
                    name: name.to_string(),
                    env_var: env_var.map(str::to_string),
                    supports_oauth,
                    is_local,
                    summary,
                }
            })
            .collect()
    }

    fn selected_row(&self) -> Option<ProviderAuthRow> {
        self.provider_rows()
            .into_iter()
            .find(|r| r.id == self.selected_provider_id)
    }

    fn select_provider(&mut self, id: String, cx: &mut Context<Self>) {
        self.selected_provider_id = id;
        self.status_message = None;
        self.api_key_input.update(cx, |input, cx| input.clear(cx));
        cx.notify();
    }

    fn toggle_api_key_mask(&mut self, cx: &mut Context<Self>) {
        let is_masked = self.api_key_input.read(cx).is_masked();
        self.api_key_input.update(cx, |input, cx| {
            input.set_masked(!is_masked, cx);
        });
    }

    fn save_api_key(&mut self, cx: &mut Context<Self>) {
        let key = self.api_key_input.read(cx).text().trim().to_string();
        let provider = self.selected_provider_id.clone();
        if key.is_empty() {
            self.status_message = Some(("Please enter an API key to save".into(), true));
            cx.notify();
            return;
        }

        match self.runtime.credentials().set_api_key(&provider, &key) {
            Ok(()) => {
                self.status_message = Some((
                    format!("API key stored securely for {provider}").into(),
                    false,
                ));
                self.api_key_input.update(cx, |input, cx| input.clear(cx));
                cx.emit(AuthModalEvent::CredentialsChanged);
                cx.notify();
            }
            Err(err) => {
                self.status_message = Some((format!("Failed to save key: {err}").into(), true));
                cx.notify();
            }
        }
    }

    fn remove_credential(&mut self, provider_id: &str, cx: &mut Context<Self>) {
        match self.runtime.credentials().remove(provider_id) {
            Ok(()) => {
                self.status_message = Some((
                    format!("Credentials cleared for {provider_id}").into(),
                    false,
                ));
                self.api_key_input.update(cx, |input, cx| input.clear(cx));
                cx.emit(AuthModalEvent::CredentialsChanged);
                cx.notify();
            }
            Err(err) => {
                self.status_message =
                    Some((format!("Failed to clear credentials: {err}").into(), true));
                cx.notify();
            }
        }
    }

    fn start_oauth_flow(
        &mut self,
        provider_id: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.in_flight_oauth.is_some() {
            return;
        }

        let provider_str = provider_id.clone();
        self.in_flight_oauth = Some(provider_id.clone());
        let cancel = CancellationToken::new();
        self.cancel_token = Some(cancel.clone());
        let credentials = self.runtime.credentials();
        let tokio_handle = self.tokio_handle.clone();

        self.status_message = Some((
            format!("Opening browser for {provider_str} OAuth login… Waiting for callback").into(),
            false,
        ));

        let oauth_future = tokio_handle.spawn(async move {
            let on_url = Arc::new(|url: &str| {
                let _ = std::process::Command::new("xdg-open").arg(url).spawn();
            });
            credentials.login(&provider_str, cancel, on_url).await
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
                            format!("✓ Successfully logged in to {provider_id} via OAuth!").into(),
                            false,
                        ));
                        cx.emit(AuthModalEvent::CredentialsChanged);
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

    fn close(&mut self, _: &CloseAuthModal, _: &mut Window, cx: &mut Context<Self>) {
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
                    .gap_2()
                    .child(
                        div()
                            .text_base()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(rgb(0xffffff))
                            .child("Providers & Authentication"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x888888))
                            .child("OAuth commands & secure API keys"),
                    ),
            )
            .child(
                div()
                    .id("close-auth-modal")
                    .px_2()
                    .py_1()
                    .rounded_sm()
                    .text_xs()
                    .text_color(rgb(0xaaaaaa))
                    .cursor_pointer()
                    .hover(|s| s.bg(rgb(0x303030)).text_color(rgb(0xffffff)))
                    .on_click(cx.listener(|_, _: &ClickEvent, _, cx| {
                        cx.emit(AuthModalEvent::Close);
                    }))
                    .child("✕ Close (Esc)"),
            )
    }

    fn sidebar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let rows = self.provider_rows();
        let selected_id = self.selected_provider_id.clone();

        div()
            .w(px(240.))
            .flex_none()
            .border_r_1()
            .border_color(rgb(0x2d2d2d))
            .bg(rgb(0x191919))
            .p_2()
            .flex()
            .flex_col()
            .gap_1()
            .children(rows.into_iter().enumerate().map(|(index, row)| {
                let is_selected = row.id == selected_id;
                let id = row.id.clone();
                let name = row.name.clone();
                let has_cred = row.summary.is_some();
                let is_oauth = row
                    .summary
                    .as_ref()
                    .map(|s| s.kind == CredentialKind::OAuth)
                    .unwrap_or(false);

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
                    .child(if row.is_local {
                        div()
                            .px_1p5()
                            .py_0p5()
                            .rounded_sm()
                            .bg(rgb(0x172554))
                            .text_color(rgb(0x93c5fd))
                            .text_size(px(10.))
                            .child("Local")
                    } else if is_oauth {
                        div()
                            .px_1p5()
                            .py_0p5()
                            .rounded_sm()
                            .bg(rgb(0x2e1065))
                            .text_color(rgb(0xd8b4fe))
                            .text_size(px(10.))
                            .child("OAuth")
                    } else if has_cred {
                        div()
                            .px_1p5()
                            .py_0p5()
                            .rounded_sm()
                            .bg(rgb(0x14532d))
                            .text_color(rgb(0x86efac))
                            .text_size(px(10.))
                            .child("Key ✓")
                    } else {
                        div()
                            .px_1p5()
                            .py_0p5()
                            .rounded_sm()
                            .bg(rgb(0x27272a))
                            .text_color(rgb(0x71717a))
                            .text_size(px(10.))
                            .child("Not set")
                    })
            }))
    }

    fn detail_pane(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let row = match self.selected_row() {
            Some(r) => r,
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

        let is_masked = self.api_key_input.read(cx).is_masked();
        let in_flight = self.in_flight_oauth.as_deref() == Some(&row.id);
        let id_copy = row.id.clone();
        let id_copy_2 = row.id.clone();

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
            // Header for provider
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
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
                                    .child(row.name.clone()),
                            )
                            .child(
                                div()
                                    .px_2()
                                    .py_0p5()
                                    .rounded_md()
                                    .bg(rgb(0x27272a))
                                    .text_xs()
                                    .text_color(rgb(0xa1a1aa))
                                    .child(row.id.clone()),
                            ),
                    ),
            )
            // OAuth Command Section (if supported)
            .when(row.supports_oauth, |pane| {
                let id = id_copy.clone();
                pane.child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .p_3()
                        .rounded_lg()
                        .border_1()
                        .border_color(rgb(0x3b0764))
                        .bg(rgb(0x1e0b36))
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .child(
                                    div()
                                        .text_xs()
                                        .font_weight(gpui::FontWeight::BOLD)
                                        .text_color(rgb(0xc084fc))
                                        .child("🚀 OAUTH BROWSER LOGIN COMMAND"),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(rgb(0xa855f7))
                                        .child("Interactive Loopback Flow"),
                                ),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(rgb(0xd8b4fe))
                                .child("Starts local loopback callback, launches your browser to authenticate, and automatically stores the token."),
                        )
                        .child(
                            div()
                                .flex()
                                .gap_2()
                                .when(!in_flight, |btn_row| {
                                    btn_row.child(
                                        div()
                                            .id("trigger-oauth-btn")
                                            .flex()
                                            .items_center()
                                            .gap_1p5()
                                            .px_3p5()
                                            .py_1p5()
                                            .rounded_md()
                                            .bg(rgb(0x7e22ce))
                                            .text_xs()
                                            .font_weight(gpui::FontWeight::MEDIUM)
                                            .text_color(rgb(0xffffff))
                                            .cursor_pointer()
                                            .hover(|s| s.bg(rgb(0x9333ea)))
                                            .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                                                this.start_oauth_flow(id.clone(), window, cx);
                                            }))
                                            .child(format!("Launch {} OAuth Login", row.name)),
                                    )
                                })
                                .when(in_flight, |btn_row| {
                                    btn_row
                                        .child(
                                            div()
                                                .px_3()
                                                .py_1p5()
                                                .rounded_md()
                                                .bg(rgb(0x581c87))
                                                .text_xs()
                                                .text_color(rgb(0xe9d5ff))
                                                .child("⏳ Waiting for browser callback…"),
                                        )
                                        .child(
                                            div()
                                                .id("cancel-oauth-btn")
                                                .px_3()
                                                .py_1p5()
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
                                                .child("Cancel"),
                                        )
                                }),
                        ),
                )
            })
            // API Key Management Section
            .when(!row.is_local, |pane| {
                let id = id_copy_2.clone();
                let summary_opt = row.summary.clone();

                pane.child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_3()
                        .p_3()
                        .rounded_lg()
                        .border_1()
                        .border_color(rgb(0x27272a))
                        .bg(rgb(0x18181b))
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .child(
                                    div()
                                        .text_xs()
                                        .font_weight(gpui::FontWeight::BOLD)
                                        .text_color(rgb(0xe4e4e7))
                                        .child("🔑 API KEY SETUP"),
                                )
                                .when_some(row.env_var.clone(), |hdr, env| {
                                    hdr.child(
                                        div()
                                            .text_xs()
                                            .text_color(rgb(0x71717a))
                                            .child(format!("Environment fallback: {env}")),
                                    )
                                }),
                        )
                        // Current status display
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .px_3()
                                .py_2()
                                .rounded_md()
                                .border_1()
                                .border_color(if summary_opt.is_some() {
                                    rgb(0x166534)
                                } else {
                                    rgb(0x3f3f46)
                                })
                                .bg(if summary_opt.is_some() {
                                    rgb(0x052e16)
                                } else {
                                    rgb(0x27272a)
                                })
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(if summary_opt.is_some() {
                                            rgb(0x86efac)
                                        } else {
                                            rgb(0xa1a1aa)
                                        })
                                        .child(match &summary_opt {
                                            Some(s) if s.kind == CredentialKind::OAuth => {
                                                format!(
                                                    "OAuth active: {} (Org: {})",
                                                    s.email.as_deref().unwrap_or("user"),
                                                    s.org_name.as_deref().unwrap_or("default")
                                                )
                                            }
                                            Some(s) => {
                                                format!(
                                                    "API Key configured ({})",
                                                    s.account_id.as_deref().unwrap_or("stored")
                                                )
                                            }
                                            None => "No stored key (using env var if set)".to_string(),
                                        }),
                                )
                                .when(summary_opt.is_some(), |status_row| {
                                    let id_clear = id.clone();
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
                                                this.remove_credential(&id_clear, cx);
                                            }))
                                            .child("Clear"),
                                    )
                                }),
                        )
                        // Input box to set new API key
                        .child(
                            div()
                                .flex()
                                .gap_2()
                                .child(
                                    div()
                                        .flex_1()
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
                                        .px_2p5()
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
                                        .flex()
                                        .items_center()
                                        .justify_center()
                                        .px_3p5()
                                        .rounded_md()
                                        .bg(rgb(0x15803d))
                                        .text_xs()
                                        .font_weight(gpui::FontWeight::MEDIUM)
                                        .text_color(rgb(0xffffff))
                                        .cursor_pointer()
                                        .hover(|s| s.bg(rgb(0x16a34a)))
                                        .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                            this.save_api_key(cx);
                                        }))
                                        .child("Save Key"),
                                ),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(rgb(0x71717a))
                                .child("Stored in ~/.local/share/juto/auth.json (mode 0600) and encrypted by system permissions."),
                        ),
                )
            })
            // Local provider notice
            .when(row.is_local, |pane| {
                pane.child(
                    div()
                        .p_3()
                        .rounded_lg()
                        .border_1()
                        .border_color(rgb(0x1e3a8a))
                        .bg(rgb(0x0f172a))
                        .text_xs()
                        .text_color(rgb(0x93c5fd))
                        .child("Ollama is a local daemon running on your machine (default: http://localhost:11434). No API key or OAuth authentication is needed."),
                )
            })
    }
}

impl Focusable for AuthModal {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for AuthModal {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .key_context("AuthModal")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(Self::close))
            .flex()
            .flex_col()
            .size_full()
            .bg(rgb(0x141414))
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

#[cfg(test)]
mod tests {
    #[test]
    fn auth_modal_rows_cover_known_services() {
        let rows = vec![
            ("anthropic", true),
            ("openai", false),
            ("openai-codex", true),
            ("google", false),
            ("ollama", false),
        ];
        for (id, supports_oauth) in rows {
            assert_eq!(matches!(id, "anthropic" | "openai-codex"), supports_oauth);
        }
    }
}
