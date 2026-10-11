//! Connected-provider model and role picker. Filtering is cached; only visible
//! rows are rendered, even when an authenticated gateway has thousands of models.

use std::collections::BTreeSet;

use gpui::{
    App, ClickEvent, Context, Entity, FocusHandle, Focusable, KeyBinding, ScrollStrategy,
    SharedString, Subscription, Task, UniformListScrollHandle, Window, actions, div, prelude::*,
    px, rgb, uniform_list,
};
use juto_ai::HttpProvider;
use juto_catalog::{Model, Registry};
use juto_runtime::{Runtime, RuntimeConfig};

use crate::text_input::{SubmitText, TextInput};

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

struct ModelEntry {
    selector: String,
    provider: String,
    name: SharedString,
    details: SharedString,
    search: String,
}

impl ModelEntry {
    fn from_catalog(model: &Model) -> Self {
        let selector = format!("{}/{}", model.provider, model.id);
        let name = if model.name.is_empty() {
            model.id.clone()
        } else {
            model.name.clone()
        };
        let mut badges = Vec::new();
        if let Some(context) = model.context_window {
            badges.push(format!("{}k context", context / 1000));
        }
        if model.reasoning {
            badges.push("Reasoning".into());
        }
        if model.supports_tools {
            badges.push("Tools".into());
        }
        if model.cost.input > 0.0 || model.cost.output > 0.0 {
            badges.push(format!(
                "${:.2} in / ${:.2} out per Mtok",
                model.cost.input, model.cost.output
            ));
        }
        let search = format!("{selector} {name} {}", model.api).to_lowercase();
        Self {
            selector,
            provider: model.provider.clone(),
            name: name.into(),
            details: badges.join(" · ").into(),
            search,
        }
    }
}

struct RoleEntry {
    role: String,
    target: String,
    resolved: String,
}

struct PickerData {
    models: Vec<ModelEntry>,
    roles: Vec<RoleEntry>,
    connected: BTreeSet<String>,
    available: BTreeSet<String>,
    filtered: Vec<usize>,
    query: Option<String>,
}

impl PickerData {
    fn new(config: &RuntimeConfig, registry: &Registry) -> Self {
        let models = registry
            .iter()
            .filter(|model| {
                matches!(model.kind.as_str(), "chat" | "tiny")
                    && HttpProvider::supports_model(model)
            })
            .map(ModelEntry::from_catalog)
            .collect();
        let mut role_names = BTreeSet::from(["default".to_string()]);
        role_names.extend(config.model_roles.keys().cloned());
        let roles = role_names
            .into_iter()
            .filter_map(|name| {
                let role = format!("@{name}");
                let (target, _) = config.resolve_selector(&role).ok()?;
                let model = registry.resolve(&target).ok()?;
                Some(RoleEntry {
                    role,
                    target,
                    resolved: format!("{}/{}", model.provider, model.id),
                })
            })
            .collect();
        Self {
            models,
            roles,
            connected: BTreeSet::new(),
            available: BTreeSet::new(),
            filtered: Vec::new(),
            query: None,
        }
    }

    fn connect(&mut self, providers: BTreeSet<String>, query: &str) {
        self.connected = providers;
        self.available = self
            .models
            .iter()
            .filter(|model| self.connected.contains(&model.provider))
            .map(|model| model.selector.clone())
            .collect();
        self.query = None;
        self.filter(query);
    }

    fn filter(&mut self, query: &str) -> bool {
        let query = query.trim().to_lowercase();
        if self.query.as_ref() == Some(&query) {
            return false;
        }
        self.filtered = self
            .models
            .iter()
            .enumerate()
            .filter(|(_, model)| {
                self.connected.contains(&model.provider) && model.search.contains(&query)
            })
            .map(|(index, _)| index)
            .collect();
        self.query = Some(query);
        true
    }

    fn visible_roles<'a>(&'a self, query: &str) -> impl Iterator<Item = &'a RoleEntry> {
        let query = query.trim().to_lowercase();
        self.roles.iter().filter(move |role| {
            self.available.contains(&role.resolved)
                && (role.role.to_lowercase().contains(&query)
                    || role.target.to_lowercase().contains(&query))
        })
    }
}

#[derive(Clone, Debug)]
pub enum ModelPickerEvent {
    SelectModel(String),
    Close,
    ManageProviders,
}

pub struct ModelPicker {
    focus_handle: FocusHandle,
    search_input: Entity<TextInput>,
    active_model: String,
    recent_models: Vec<String>,
    data: PickerData,
    selected: usize,
    scroll_handle: UniformListScrollHandle,
    loading: bool,
    error: Option<String>,
    refresh_task: Option<Task<()>>,
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
            TextInput::new(cx).with_placeholder("Search connected models or configured roles…")
        });
        let subscription = cx.observe(&search_input, |this, input, cx| {
            if this.data.filter(input.read(cx).text()) {
                this.selected = 0;
                this.scroll_handle.scroll_to_item(0, ScrollStrategy::Top);
                cx.notify();
            }
        });
        let submitted = cx.subscribe(&search_input, |this, _, _: &SubmitText, cx| {
            this.select_current(cx);
        });
        Self {
            focus_handle: cx.focus_handle(),
            search_input,
            recent_models: vec![active_model.clone()],
            active_model,
            data: PickerData::new(config, registry),
            selected: 0,
            scroll_handle: UniformListScrollHandle::new(),
            loading: true,
            error: None,
            refresh_task: None,
            _subscriptions: vec![subscription, submitted],
        }
    }

    pub fn refresh_credentials(
        &mut self,
        runtime: Runtime,
        handle: tokio::runtime::Handle,
        cx: &mut Context<Self>,
    ) {
        self.loading = true;
        self.error = None;
        // Fail closed while a refresh is pending; removed providers cannot remain selectable.
        self.data
            .connect(BTreeSet::new(), self.search_input.read(cx).text());
        let providers: BTreeSet<String> = self
            .data
            .models
            .iter()
            .map(|model| model.provider.clone())
            .collect();
        let task = handle.spawn_blocking(move || {
            runtime
                .credentials()
                .available_providers(providers.iter().map(String::as_str))
        });
        self.refresh_task = Some(cx.spawn(async move |this, cx| {
            let result = task.await;
            let _ = this.update(cx, |this, cx| {
                this.loading = false;
                match result {
                    Ok(Ok(providers)) => this
                        .data
                        .connect(providers, this.search_input.read(cx).text()),
                    Ok(Err(error)) => {
                        this.error = Some(format!("Could not read provider credentials: {error}"))
                    }
                    Err(error) => this.error = Some(format!("Credential lookup failed: {error}")),
                }
                this.selected = 0;
                this.scroll_handle.scroll_to_item(0, ScrollStrategy::Top);
                this.refresh_task = None;
                cx.notify();
            });
        }));
        cx.notify();
    }

    pub fn focus(&self, window: &mut Window, cx: &App) {
        self.search_input.read(cx).focus_handle(cx).focus(window);
    }

    pub fn set_active_model(&mut self, model: impl Into<String>, cx: &mut Context<Self>) {
        let model = model.into();
        self.active_model = model.clone();
        self.recent_models.retain(|recent| recent != &model);
        self.recent_models.insert(0, model);
        self.recent_models.truncate(6);
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

    fn previous(&mut self, _: &SelectPreviousModel, _: &mut Window, cx: &mut Context<Self>) {
        self.selected = self.selected.saturating_sub(1);
        self.scroll_handle
            .scroll_to_item(self.selected, ScrollStrategy::Top);
        cx.notify();
    }

    fn next(&mut self, _: &SelectNextModel, _: &mut Window, cx: &mut Context<Self>) {
        self.selected = (self.selected + 1).min(self.data.filtered.len().saturating_sub(1));
        self.scroll_handle
            .scroll_to_item(self.selected, ScrollStrategy::Bottom);
        cx.notify();
    }

    fn confirm(&mut self, _: &ConfirmModelSelection, _: &mut Window, cx: &mut Context<Self>) {
        self.select_current(cx);
    }

    fn select_current(&mut self, cx: &mut Context<Self>) {
        if let Some(index) = self.data.filtered.get(self.selected) {
            self.select_model(self.data.models[*index].selector.clone(), cx);
        }
    }

    fn model_row(
        &self,
        row: usize,
        index: usize,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + use<> {
        let model = &self.data.models[index];
        let selector = model.selector.clone();
        let active = model.selector == self.active_model;
        div()
            .id(("model-entry", index))
            .h(px(72.))
            .w_full()
            .px_3()
            .py_2()
            .flex()
            .items_center()
            .justify_between()
            .gap_3()
            .rounded_md()
            .border_1()
            .border_color(if row == self.selected {
                rgb(0x606060)
            } else {
                rgb(0x272727)
            })
            .bg(if active { rgb(0x132717) } else { rgb(0x1f1f1f) })
            .cursor_pointer()
            .hover(|style| style.bg(rgb(0x292929)))
            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                this.select_model(selector.clone(), cx)
            }))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .min_w_0()
                    .gap_1()
                    .child(
                        div()
                            .text_sm()
                            .text_color(if active { rgb(0x4ade80) } else { rgb(0xffffff) })
                            .truncate()
                            .child(model.name.clone()),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0xaaaaaa))
                            .truncate()
                            .child(model.selector.clone()),
                    )
                    .child(
                        div()
                            .text_size(px(10.))
                            .text_color(rgb(0x888888))
                            .truncate()
                            .child(model.details.clone()),
                    ),
            )
            .child(
                div()
                    .flex_none()
                    .text_xs()
                    .text_color(rgb(0xaaaaaa))
                    .child(if active { "✓ Active" } else { "Select" }),
            )
    }

    fn shortcuts(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let query = self.search_input.read(cx).text();
        let recents = self
            .recent_models
            .iter()
            .filter(|selector| self.data.available.contains(*selector));
        div()
            .flex()
            .flex_col()
            .gap_2()
            .px_4()
            .py_2()
            .when(query.trim().is_empty(), |view| {
                view.child(
                    div()
                        .flex()
                        .flex_wrap()
                        .gap_2()
                        .children(recents.enumerate().map(|(index, selector)| {
                            let selection = selector.clone();
                            div()
                                .id(("recent-model", index))
                                .px_2()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(0x242424))
                                .text_xs()
                                .cursor_pointer()
                                .hover(|style| style.bg(rgb(0x333333)))
                                .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                    this.select_model(selection.clone(), cx)
                                }))
                                .child(format!("Recent: {selector}"))
                        })),
                )
            })
            .child(
                div().flex().flex_wrap().gap_2().children(
                    self.data
                        .visible_roles(query)
                        .enumerate()
                        .map(|(index, role)| {
                            let selection = role.role.clone();
                            div()
                                .id(("role-row", index))
                                .px_2()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(0x261440))
                                .text_xs()
                                .cursor_pointer()
                                .hover(|style| style.bg(rgb(0x3b1c60)))
                                .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                    this.select_model(selection.clone(), cx)
                                }))
                                .child(format!("{} → {}", role.role, role.target))
                        }),
                ),
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
            .on_action(cx.listener(Self::previous))
            .on_action(cx.listener(Self::next))
            .on_action(cx.listener(Self::confirm))
            .flex()
            .flex_col()
            .size_full()
            .bg(rgb(0x161616))
            .text_color(rgb(0xe4e4e7))
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .px_4()
                    .py_3()
                    .border_b_1()
                    .border_color(rgb(0x333333))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child("Select Model or Role")
                            .child(div().text_xs().text_color(rgb(0x888888)).child(format!(
                                "{} connected models · {} matching",
                                self.data.available.len(),
                                self.data.filtered.len()
                            ))),
                    )
                    .child(
                        div()
                            .id("close-picker-btn")
                            .px_2()
                            .py_1()
                            .text_xs()
                            .cursor_pointer()
                            .on_click(cx.listener(|_, _: &ClickEvent, _, cx| {
                                cx.emit(ModelPickerEvent::Close)
                            }))
                            .child("Close (Esc)"),
                    ),
            )
            .child(
                div()
                    .p_3()
                    .border_b_1()
                    .border_color(rgb(0x2a2a2a))
                    .child(self.search_input.clone()),
            )
            .child(self.shortcuts(cx))
            .when(
                self.loading || self.error.is_some() || self.data.filtered.is_empty(),
                |view| {
                    let message = if self.loading {
                        "Loading connected providers…".to_string()
                    } else if let Some(error) = &self.error {
                        error.clone()
                    } else if self.data.available.is_empty() {
                        "No connected models. Sign in to a provider or add an API key.".into()
                    } else {
                        "No connected models match your search.".into()
                    };
                    view.child(
                        div()
                            .p_4()
                            .text_sm()
                            .text_color(rgb(0xaaaaaa))
                            .child(message),
                    )
                },
            )
            .child(
                uniform_list(
                    "model-list",
                    self.data.filtered.len(),
                    cx.processor(|this, range: std::ops::Range<usize>, _, cx| {
                        range
                            .map(|row| this.model_row(row, this.data.filtered[row], cx))
                            .collect()
                    }),
                )
                .track_scroll(self.scroll_handle.clone())
                .w_full()
                .flex_1()
                .min_h_0(),
            )
            .child(
                div()
                    .px_4()
                    .py_3()
                    .border_t_1()
                    .border_color(rgb(0x333333))
                    .flex()
                    .justify_between()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x888888))
                            .child("↑ ↓ Navigate · Enter Select"),
                    )
                    .child(
                        div()
                            .id("picker-manage-providers")
                            .text_xs()
                            .cursor_pointer()
                            .on_click(cx.listener(|_, _: &ClickEvent, _, cx| {
                                cx.emit(ModelPickerEvent::ManageProviders)
                            }))
                            .child("Manage providers…"),
                    ),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disconnected_models_and_roles_are_hidden_and_revoked() {
        let registry = Registry::bundled().unwrap();
        let config = RuntimeConfig {
            model: "anthropic/claude-sonnet-4-5".into(),
            model_roles: [("fast".into(), "openai/gpt-4o:high".into())].into(),
            ..Default::default()
        };
        let mut data = PickerData::new(&config, &registry);
        assert!(data.filtered.is_empty());
        assert_eq!(data.visible_roles("").count(), 0);
        data.connect(BTreeSet::from(["openai".into()]), "");
        assert!(data.available.contains("openai/gpt-4o"));
        assert!(!data.available.contains("anthropic/claude-sonnet-4-5"));
        assert!(
            data.filtered
                .iter()
                .all(|index| data.models[*index].provider == "openai")
        );
        assert_eq!(
            data.visible_roles("")
                .map(|role| role.role.as_str())
                .collect::<Vec<_>>(),
            ["@fast"]
        );
        data.filter(" GPT-4O ");
        assert!(
            data.filtered
                .iter()
                .any(|index| data.models[*index].selector == "openai/gpt-4o")
        );
        assert!(
            data.filtered
                .iter()
                .all(|index| data.models[*index].search.contains("gpt-4o"))
        );
        data.connect(BTreeSet::new(), "");
        assert!(data.filtered.is_empty());
        assert_eq!(data.visible_roles("").count(), 0);
    }

    #[test]
    fn unsupported_transports_and_non_chat_models_are_not_offered() {
        let mut registry = Registry::empty();
        let mut model = Registry::bundled()
            .unwrap()
            .get("openai", "gpt-4o")
            .unwrap()
            .clone();
        model.id = "chat".into();
        registry.upsert(model.clone());
        model.id = "embedding".into();
        model.kind = "embedding".into();
        registry.upsert(model.clone());
        model.id = "unsupported".into();
        model.kind = "chat".into();
        model.api = "future-transport".into();
        registry.upsert(model);
        let mut data = PickerData::new(&RuntimeConfig::default(), &registry);
        data.connect(BTreeSet::from(["openai".into()]), "");
        assert_eq!(data.available, BTreeSet::from(["openai/chat".into()]));
    }
}
