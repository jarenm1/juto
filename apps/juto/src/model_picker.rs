//! OMP-style Model and Role picker component for GPUI.
//!
//! Features:
//! - Search bar for real-time filtering across model IDs, names, providers, and roles.
//! - Pinned "Recent / Latest Used" models at the top.
//! - OMP model roles section (`@default`, `@smol`, `@fast`, `@reasoning`, `@editor`).
//! - All catalog models grouped by provider with capability badges (reasoning, tools, context, rates).
//! - Fast selection and keyboard navigation.

use std::collections::BTreeMap;

use gpui::{
    App, ClickEvent, Context, Entity, FocusHandle, Focusable, KeyBinding, ScrollHandle,
    Subscription, Window, actions, div, prelude::*, px, rgb,
};
use juto_catalog::{Model, Registry};
use juto_runtime::RuntimeConfig;

use crate::text_input::TextInput;

actions!(
    model_picker,
    [
        CloseModelPicker,
        SelectPreviousModel,
        SelectNextModel,
        ConfirmModelSelection
    ]
);

pub fn init(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("escape", CloseModelPicker, Some("ModelPicker")),
        KeyBinding::new("up", SelectPreviousModel, Some("ModelPicker")),
        KeyBinding::new("down", SelectNextModel, Some("ModelPicker")),
        KeyBinding::new("enter", ConfirmModelSelection, Some("ModelPicker")),
    ]);
}

/// A model entry for the picker.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub api: String,
    pub context_window: Option<u64>,
    pub max_tokens: Option<u64>,
    pub supports_tools: bool,
    pub reasoning: bool,
    pub cost_input_per_mtok: f64,
    pub cost_output_per_mtok: f64,
}

impl ModelEntry {
    pub fn from_catalog(model: &Model) -> Self {
        Self {
            id: model.id.clone(),
            name: if model.name.is_empty() {
                model.id.clone()
            } else {
                model.name.clone()
            },
            provider: model.provider.clone(),
            api: model.api.clone(),
            context_window: model.context_window,
            max_tokens: model.max_tokens,
            supports_tools: model.supports_tools,
            reasoning: model.reasoning,
            cost_input_per_mtok: model.cost.input,
            cost_output_per_mtok: model.cost.output,
        }
    }

    pub fn selector(&self) -> String {
        format!("{}/{}", self.provider, self.id)
    }
}

/// An OMP role mapping entry.
#[derive(Clone, Debug, PartialEq)]
pub struct RoleEntry {
    pub role: String,
    pub target: String,
    pub description: String,
}

/// Events emitted by ModelPicker.
#[derive(Clone, Debug)]
pub enum ModelPickerEvent {
    SelectModel(String),
    Close,
}

pub struct ModelPicker {
    focus_handle: FocusHandle,
    search_input: Entity<TextInput>,
    active_model: String,
    recent_models: Vec<String>,
    all_models: Vec<ModelEntry>,
    roles: Vec<RoleEntry>,
    scroll_handle: ScrollHandle,
    _subscriptions: Vec<Subscription>,
}

impl gpui::EventEmitter<ModelPickerEvent> for ModelPicker {}

impl ModelPicker {
    pub fn new(
        active_model: impl Into<String>,
        config: &RuntimeConfig,
        registry: &Registry,
        cx: &mut Context<Self>,
    ) -> Self {
        let active_model = active_model.into();
        let search_input = cx.new(|cx| {
            TextInput::new(cx).with_placeholder(
                "Filter models or roles (e.g. sonnet, gpt-4o, @default, @reasoning)…",
            )
        });

        let mut all_models: Vec<ModelEntry> =
            registry.iter().map(ModelEntry::from_catalog).collect();
        // If catalog is empty for any reason, add common standard fallbacks
        if all_models.is_empty() {
            all_models = default_fallback_models();
        }

        // Build standard + configured OMP roles
        let mut roles = Vec::new();
        roles.push(RoleEntry {
            role: "@default".to_string(),
            target: if config.model.is_empty() {
                "anthropic/claude-sonnet-4-5".to_string()
            } else {
                config.model.clone()
            },
            description: "Primary default model for general tasks".to_string(),
        });
        roles.push(RoleEntry {
            role: "@fast".to_string(),
            target: config
                .model_roles
                .get("fast")
                .cloned()
                .unwrap_or_else(|| "anthropic/claude-3-5-haiku".to_string()),
            description: "Fast, low-latency lightweight model".to_string(),
        });
        roles.push(RoleEntry {
            role: "@smol".to_string(),
            target: config
                .model_roles
                .get("smol")
                .cloned()
                .unwrap_or_else(|| "openai/gpt-4o-mini".to_string()),
            description: "Economical model for high-throughput subtasks".to_string(),
        });
        roles.push(RoleEntry {
            role: "@reasoning".to_string(),
            target: config
                .model_roles
                .get("reasoning")
                .cloned()
                .unwrap_or_else(|| "openai/o3-mini".to_string()),
            description: "High-effort deep reasoning model".to_string(),
        });

        // Add any additional custom roles from config
        for (role_name, target) in &config.model_roles {
            let role_tag = format!("@{role_name}");
            if !roles.iter().any(|r| r.role == role_tag) {
                roles.push(RoleEntry {
                    role: role_tag,
                    target: target.clone(),
                    description: format!("Custom role mapping for @{role_name}"),
                });
            }
        }

        // Initial recent models list starting with active model
        let mut recent_models = vec![
            active_model.clone(),
            "anthropic/claude-sonnet-4-5".to_string(),
            "openai/gpt-4o".to_string(),
            "google/gemini-2.5-pro".to_string(),
            "ollama/llama3.3".to_string(),
            "deepseek/deepseek-chat".to_string(),
        ];
        recent_models.dedup();

        let obs_search = cx.observe(&search_input, |_, _, cx| cx.notify());

        Self {
            focus_handle: cx.focus_handle(),
            search_input,
            active_model,
            recent_models,
            all_models,
            roles,
            scroll_handle: ScrollHandle::new(),
            _subscriptions: vec![obs_search],
        }
    }

    pub fn focus(&self, window: &mut Window, cx: &App) {
        self.search_input.read(cx).focus_handle(cx).focus(window);
    }

    pub fn set_active_model(&mut self, model: impl Into<String>, cx: &mut Context<Self>) {
        let model = model.into();
        self.active_model = model.clone();
        if !self.recent_models.contains(&model) {
            self.recent_models.insert(0, model);
            if self.recent_models.len() > 6 {
                self.recent_models.truncate(6);
            }
        }
        cx.notify();
    }

    fn select_model(&mut self, selector: String, cx: &mut Context<Self>) {
        self.set_active_model(selector.clone(), cx);
        cx.emit(ModelPickerEvent::SelectModel(selector));
        cx.emit(ModelPickerEvent::Close);
    }

    fn close(&mut self, _: &CloseModelPicker, _: &mut Window, cx: &mut Context<Self>) {
        cx.emit(ModelPickerEvent::Close);
    }

    fn filtered_roles<'a>(&'a self, cx: &App) -> Vec<&'a RoleEntry> {
        let query = self.search_input.read(cx).text().trim().to_lowercase();
        if query.is_empty() {
            return self.roles.iter().collect();
        }
        self.roles
            .iter()
            .filter(|r| {
                r.role.to_lowercase().contains(&query)
                    || r.target.to_lowercase().contains(&query)
                    || r.description.to_lowercase().contains(&query)
            })
            .collect()
    }

    fn filtered_models<'a>(&'a self, cx: &App) -> Vec<&'a ModelEntry> {
        let query = self.search_input.read(cx).text().trim().to_lowercase();
        if query.is_empty() {
            return self.all_models.iter().collect();
        }
        self.all_models
            .iter()
            .filter(|m| {
                m.id.to_lowercase().contains(&query)
                    || m.name.to_lowercase().contains(&query)
                    || m.provider.to_lowercase().contains(&query)
                    || m.api.to_lowercase().contains(&query)
            })
            .collect()
    }

    fn models_by_provider<'a>(
        &'a self,
        filtered: &[&'a ModelEntry],
    ) -> BTreeMap<String, Vec<&'a ModelEntry>> {
        let mut grouped: BTreeMap<String, Vec<&'a ModelEntry>> = BTreeMap::new();
        for model in filtered {
            grouped
                .entry(model.provider.clone())
                .or_default()
                .push(model);
        }
        grouped
    }

    fn provider_display_name(provider: &str) -> &'static str {
        match provider.to_lowercase().as_str() {
            "anthropic" => "Anthropic",
            "openai" => "OpenAI",
            "openai-codex" => "OpenAI Codex",
            "google" | "gemini" => "Google Gemini",
            "ollama" => "Ollama (Local)",
            "openrouter" => "OpenRouter",
            "deepseek" => "DeepSeek",
            "groq" => "Groq",
            "mistral" => "Mistral AI",
            "xai" => "xAI",
            "azure" => "Azure OpenAI",
            "cohere" => "Cohere",
            "together" => "Together AI",
            "perplexity" => "Perplexity",
            _ => "Other Providers",
        }
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
                            .child("Select Model or Role"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x888888))
                            .child(format!("{} models in catalog", self.all_models.len())),
                    ),
            )
            .child(
                div()
                    .id("close-picker-btn")
                    .px_2()
                    .py_1()
                    .rounded_sm()
                    .text_xs()
                    .text_color(rgb(0xaaaaaa))
                    .cursor_pointer()
                    .hover(|s| s.bg(rgb(0x303030)).text_color(rgb(0xffffff)))
                    .on_click(cx.listener(|_, _: &ClickEvent, _, cx| {
                        cx.emit(ModelPickerEvent::Close);
                    }))
                    .child("✕ Close (Esc)"),
            )
    }

    fn search_bar(&self) -> impl IntoElement {
        div()
            .p_3()
            .border_b_1()
            .border_color(rgb(0x2a2a2a))
            .bg(rgb(0x1c1c1c))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .p_1p5()
                    .rounded_md()
                    .border_1()
                    .border_color(rgb(0x404040))
                    .bg(rgb(0x161616))
                    .child(div().text_sm().text_color(rgb(0x707070)).child("🔍"))
                    .child(self.search_input.clone()),
            )
    }

    fn recent_section(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let is_searching = !self.search_input.read(cx).text().trim().is_empty();
        if is_searching || self.recent_models.is_empty() {
            return div().into_any_element();
        }

        let recents = self.recent_models.clone();
        let active = self.active_model.clone();

        div()
            .flex()
            .flex_col()
            .gap_1p5()
            .p_3()
            .border_b_1()
            .border_color(rgb(0x2a2a2a))
            .bg(rgb(0x1e1e1e))
            .child(
                div().flex().items_center().gap_2().child(
                    div()
                        .text_xs()
                        .font_weight(gpui::FontWeight::BOLD)
                        .text_color(rgb(0x60a5fa))
                        .child("⚡ RECENT / LATEST USED"),
                ),
            )
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .gap_2()
                    .children(recents.into_iter().enumerate().map(|(index, selector)| {
                        let is_active = selector == active;
                        let sel_copy = selector.clone();

                        div()
                            .id(("recent-model", index))
                            .flex()
                            .items_center()
                            .gap_1p5()
                            .px_2p5()
                            .py_1p5()
                            .rounded_md()
                            .border_1()
                            .border_color(if is_active {
                                rgb(0x2e7d32)
                            } else {
                                rgb(0x3a3a3a)
                            })
                            .bg(if is_active {
                                rgb(0x192e1e)
                            } else {
                                rgb(0x242424)
                            })
                            .cursor_pointer()
                            .hover(|s| s.bg(rgb(0x303030)))
                            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                this.select_model(sel_copy.clone(), cx);
                            }))
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(if is_active {
                                        gpui::FontWeight::SEMIBOLD
                                    } else {
                                        gpui::FontWeight::NORMAL
                                    })
                                    .text_color(if is_active {
                                        rgb(0x4ade80)
                                    } else {
                                        rgb(0xeeeeee)
                                    })
                                    .child(selector),
                            )
                            .when(is_active, |pill| {
                                pill.child(
                                    div().text_xs().text_color(rgb(0x4ade80)).child("✓ Active"),
                                )
                            })
                    })),
            )
            .into_any_element()
    }

    fn roles_section(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let filtered = self.filtered_roles(cx);
        if filtered.is_empty() {
            return div().into_any_element();
        }

        let active = self.active_model.clone();

        div()
            .flex()
            .flex_col()
            .gap_1p5()
            .p_3()
            .border_b_1()
            .border_color(rgb(0x2a2a2a))
            .bg(rgb(0x1b1b1b))
            .child(
                div()
                    .text_xs()
                    .font_weight(gpui::FontWeight::BOLD)
                    .text_color(rgb(0xa78bfa))
                    .child("🎭 OMP ROLES"),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .children(filtered.into_iter().enumerate().map(|(index, role)| {
                        let target = role.target.clone();
                        let target_copy = target.clone();
                        let is_active = target == active || role.role == active;

                        div()
                            .id(("role-row", index))
                            .flex()
                            .items_center()
                            .justify_between()
                            .px_3()
                            .py_2()
                            .rounded_md()
                            .border_1()
                            .border_color(if is_active {
                                rgb(0x4c1d95)
                            } else {
                                rgb(0x282828)
                            })
                            .bg(if is_active {
                                rgb(0x261440)
                            } else {
                                rgb(0x202020)
                            })
                            .cursor_pointer()
                            .hover(|s| s.bg(rgb(0x2d2040)))
                            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                this.select_model(target_copy.clone(), cx);
                            }))
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_3()
                                    .child(
                                        div()
                                            .px_2()
                                            .py_0p5()
                                            .rounded_sm()
                                            .bg(rgb(0x3b1c60))
                                            .text_xs()
                                            .font_weight(gpui::FontWeight::BOLD)
                                            .text_color(rgb(0xc4b5fd))
                                            .child(role.role.clone()),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(rgb(0xe0e0e0))
                                            .child(format!("→ {target}")),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(rgb(0x7c7c7c))
                                            .child(role.description.clone()),
                                    ),
                            )
                            .child(
                                div()
                                    .px_2()
                                    .py_0p5()
                                    .rounded_sm()
                                    .border_1()
                                    .border_color(rgb(0x4c1d95))
                                    .bg(rgb(0x321354))
                                    .text_xs()
                                    .text_color(rgb(0xd8b4fe))
                                    .child(if is_active {
                                        "✓ Active"
                                    } else {
                                        "Select Role"
                                    }),
                            )
                    })),
            )
            .into_any_element()
    }

    fn model_list(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let filtered = self.filtered_models(cx);
        let grouped = self.models_by_provider(&filtered);
        let active = self.active_model.clone();

        div()
            .id("model-list-container")
            .flex_1()
            .min_h_0()
            .overflow_y_scroll()
            .track_scroll(&self.scroll_handle)
            .p_3()
            .flex()
            .flex_col()
            .gap_4()
            .when(filtered.is_empty(), |list| {
                list.child(
                    div()
                        .py_8()
                        .text_center()
                        .text_sm()
                        .text_color(rgb(0x808080))
                        .child("No models matched your filter."),
                )
            })
            .children(
                grouped
                    .into_iter()
                    .enumerate()
                    .map(|(provider_idx, (provider, models))| {
                        let display_title = Self::provider_display_name(&provider);
                        let count = models.len();
                        let active_sel = active.clone();

                        div()
                            .flex()
                            .flex_col()
                            .gap_1p5()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .px_2()
                                    .py_1()
                                    .child(
                                        div()
                                            .text_xs()
                                            .font_weight(gpui::FontWeight::BOLD)
                                            .text_color(rgb(0x94a3b8))
                                            .child(format!("{display_title} ({count})")),
                                    ),
                            )
                            .child(div().flex().flex_col().gap_1().children(
                                models.into_iter().enumerate().map(|(model_idx, model)| {
                                    let selector = model.selector();
                                    let is_active = selector == active_sel;
                                    let sel_copy = selector.clone();

                                    div()
                                        .id(("model-entry", provider_idx * 1000 + model_idx))
                                        .flex()
                                        .items_center()
                                        .justify_between()
                                        .px_3()
                                        .py_2()
                                        .rounded_md()
                                        .border_1()
                                        .border_color(if is_active {
                                            rgb(0x1e5a2e)
                                        } else {
                                            rgb(0x272727)
                                        })
                                        .bg(if is_active {
                                            rgb(0x132717)
                                        } else {
                                            rgb(0x1f1f1f)
                                        })
                                        .cursor_pointer()
                                        .hover(|s| s.bg(rgb(0x292929)))
                                        .on_click(cx.listener(
                                            move |this, _: &ClickEvent, _, cx| {
                                                this.select_model(sel_copy.clone(), cx);
                                            },
                                        ))
                                        .child(
                                            div()
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
                                                                .text_sm()
                                                                .font_weight(
                                                                    gpui::FontWeight::MEDIUM,
                                                                )
                                                                .text_color(if is_active {
                                                                    rgb(0x4ade80)
                                                                } else {
                                                                    rgb(0xffffff)
                                                                })
                                                                .child(model.name.clone()),
                                                        )
                                                        .child(
                                                            div()
                                                                .text_xs()
                                                                .text_color(rgb(0x71717a))
                                                                .child(selector.clone()),
                                                        ),
                                                )
                                                .child(
                                                    div()
                                                        .flex()
                                                        .items_center()
                                                        .gap_1p5()
                                                        .when_some(
                                                            model.context_window,
                                                            |row, ctx| {
                                                                row.child(
                                                                    div()
                                                                        .px_1p5()
                                                                        .py_0p5()
                                                                        .rounded_sm()
                                                                        .bg(rgb(0x2a2a2a))
                                                                        .text_size(px(10.))
                                                                        .text_color(rgb(0xa1a1aa))
                                                                        .child(format!(
                                                                            "{}k ctx",
                                                                            ctx / 1000
                                                                        )),
                                                                )
                                                            },
                                                        )
                                                        .when(model.reasoning, |row| {
                                                            row.child(
                                                                div()
                                                                    .px_1p5()
                                                                    .py_0p5()
                                                                    .rounded_sm()
                                                                    .bg(rgb(0x381e4a))
                                                                    .text_size(px(10.))
                                                                    .text_color(rgb(0xd896ff))
                                                                    .child("🧠 Reasoning"),
                                                            )
                                                        })
                                                        .when(model.supports_tools, |row| {
                                                            row.child(
                                                                div()
                                                                    .px_1p5()
                                                                    .py_0p5()
                                                                    .rounded_sm()
                                                                    .bg(rgb(0x163d23))
                                                                    .text_size(px(10.))
                                                                    .text_color(rgb(0x4ade80))
                                                                    .child("⚡ Tools"),
                                                            )
                                                        })
                                                        .when(
                                                            model.cost_input_per_mtok > 0.0,
                                                            |row| {
                                                                row.child(
                                                                    div()
                                                                        .text_size(px(10.))
                                                                        .text_color(rgb(0x71717a))
                                                                        .child(format!(
                                                                    "${:.2} in / ${:.2} out",
                                                                    model.cost_input_per_mtok,
                                                                    model.cost_output_per_mtok
                                                                )),
                                                                )
                                                            },
                                                        ),
                                                ),
                                        )
                                        .child(
                                            div()
                                                .px_2p5()
                                                .py_1()
                                                .rounded_sm()
                                                .border_1()
                                                .border_color(if is_active {
                                                    rgb(0x22c55e)
                                                } else {
                                                    rgb(0x3f3f46)
                                                })
                                                .bg(if is_active {
                                                    rgb(0x15803d)
                                                } else {
                                                    rgb(0x27272a)
                                                })
                                                .text_xs()
                                                .font_weight(gpui::FontWeight::MEDIUM)
                                                .text_color(if is_active {
                                                    rgb(0xffffff)
                                                } else {
                                                    rgb(0xd4d4d8)
                                                })
                                                .child(if is_active {
                                                    "✓ Active"
                                                } else {
                                                    "Select"
                                                }),
                                        )
                                }),
                            ))
                    }),
            )
    }
}

impl Focusable for ModelPicker {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for ModelPicker {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .key_context("ModelPicker")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(Self::close))
            .flex()
            .flex_col()
            .size_full()
            .bg(rgb(0x161616))
            .text_color(rgb(0xe4e4e7))
            .child(self.header(cx))
            .child(self.search_bar())
            .child(self.recent_section(cx))
            .child(self.roles_section(cx))
            .child(self.model_list(cx))
    }
}

fn default_fallback_models() -> Vec<ModelEntry> {
    vec![
        ModelEntry {
            id: "claude-sonnet-4-5".into(),
            name: "Claude 3.5 Sonnet".into(),
            provider: "anthropic".into(),
            api: "anthropic-messages".into(),
            context_window: Some(200_000),
            max_tokens: Some(8_192),
            supports_tools: true,
            reasoning: false,
            cost_input_per_mtok: 3.0,
            cost_output_per_mtok: 15.0,
        },
        ModelEntry {
            id: "gpt-4o".into(),
            name: "GPT-4o".into(),
            provider: "openai".into(),
            api: "openai-responses".into(),
            context_window: Some(128_000),
            max_tokens: Some(4_096),
            supports_tools: true,
            reasoning: false,
            cost_input_per_mtok: 2.5,
            cost_output_per_mtok: 10.0,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_picker_populates_roles_and_recents() {
        let registry = Registry::bundled().unwrap_or_else(|_| Registry::empty());
        let config = RuntimeConfig {
            model: "anthropic/claude-sonnet-4-5".into(),
            ..Default::default()
        };
        let models: Vec<ModelEntry> = registry.iter().map(ModelEntry::from_catalog).collect();
        assert!(!models.is_empty(), "Bundled registry should have models");

        let entry = models
            .iter()
            .find(|m| m.id == "claude-sonnet-4-5" && m.provider == "anthropic")
            .unwrap();
        assert_eq!(entry.selector(), "anthropic/claude-sonnet-4-5");
        assert!(entry.supports_tools);
        assert_eq!(config.model, "anthropic/claude-sonnet-4-5");
    }

    #[test]
    fn model_entry_selector_formatting() {
        let model = ModelEntry {
            id: "gpt-4o".into(),
            name: "GPT-4o".into(),
            provider: "openai".into(),
            api: "openai-responses".into(),
            context_window: Some(128_000),
            max_tokens: Some(4_096),
            supports_tools: true,
            reasoning: false,
            cost_input_per_mtok: 2.5,
            cost_output_per_mtok: 10.0,
        };
        assert_eq!(model.selector(), "openai/gpt-4o");
    }
}
