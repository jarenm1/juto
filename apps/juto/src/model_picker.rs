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
    context: SharedString,
    roles: Vec<String>,
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
        let context = model
            .context_window
            .map(format_context_window)
            .unwrap_or_else(|| "n/a".into());
        let search = format!("{selector} {name} {}", model.api).to_lowercase();
        Self {
            selector,
            provider: model.provider.clone(),
            name: name.into(),
            context: context.into(),
            roles: Vec::new(),
            search,
        }
    }
}

fn format_context_window(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        let millions = tokens as f64 / 1_000_000.0;
        if millions.fract() == 0.0 {
            format!("{}M", millions as u64)
        } else {
            format!("{millions:.1}M")
        }
    } else {
        format!("{}k", tokens / 1_000)
    }
}

fn provider_label(provider: &str) -> String {
    match provider {
        "openai" => "OpenAI".into(),
        "meta" => "Meta".into(),
        _ => {
            let mut chars = provider.chars();
            chars
                .next()
                .map(|first| first.to_uppercase().collect::<String>() + chars.as_str())
                .unwrap_or_default()
        }
    }
}

struct RoleEntry {
    role: String,
    resolved: Option<String>,
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
        let mut role_names = BTreeSet::from([
            "default".to_string(),
            "task".to_string(),
            "judge".to_string(),
            "advisor".to_string(),
            "review".to_string(),
        ]);
        role_names.extend(config.model_roles.keys().cloned());
        let mut roles = role_names
            .into_iter()
            .map(|name| {
                let resolution = config.resolve_selector(&format!("@{name}")).ok();
                let (_target, resolved) = resolution
                    .and_then(|(target, _)| {
                        let model = registry.resolve(&target).ok()?;
                        Some((
                            Some(target),
                            Some(format!("{}/{}", model.provider, model.id)),
                        ))
                    })
                    .unwrap_or((None, None));
                RoleEntry {
                    role: name,
                    resolved,
                }
            })
            .collect::<Vec<_>>();
        let role_order = ["default", "task", "judge", "advisor", "review"];
        roles.sort_by_key(|role| {
            role_order
                .iter()
                .position(|name| *name == role.role)
                .unwrap_or(role_order.len())
        });
        let mut models = registry
            .iter()
            .filter(|model| {
                matches!(model.kind.as_str(), "chat" | "tiny")
                    && HttpProvider::supports_model(model)
            })
            .map(ModelEntry::from_catalog)
            .collect::<Vec<_>>();
        models.sort_by(|left, right| {
            left.provider
                .cmp(&right.provider)
                .then_with(|| left.name.to_lowercase().cmp(&right.name.to_lowercase()))
        });
        for model in &mut models {
            model.roles = roles
                .iter()
                .filter(|role| role.resolved.as_deref() == Some(model.selector.as_str()))
                .map(|role| role.role.clone())
                .collect();
        }
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
        let search_input =
            cx.new(|cx| TextInput::new(cx).with_placeholder("Search connected models…"));
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
        for entry in &mut self.data.models {
            entry.roles.retain(|role| role != "default");
        }
        if let Some(entry) = self
            .data
            .models
            .iter_mut()
            .find(|entry| entry.selector == model)
        {
            entry.roles.push("default".into());
        }
        if let Some(role) = self
            .data
            .roles
            .iter_mut()
            .find(|role| role.role == "default")
        {
            role.resolved = Some(model);
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
        compact: bool,
        name_width: gpui::Pixels,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + use<> {
        let model = &self.data.models[index];
        let selector = model.selector.clone();
        let provider_start =
            row == 0 || self.data.models[self.data.filtered[row - 1]].provider != model.provider;
        let provider_count = self
            .data
            .filtered
            .iter()
            .filter(|index| self.data.models[**index].provider == model.provider)
            .count();
        div()
            .id(("model-entry", index))
            .h(px(74.))
            .w_full()
            .px_3()
            .py_1()
            .flex()
            .flex_col()
            .border_b_1()
            .border_color(if row == self.selected {
                rgb(0x606060)
            } else {
                rgb(0x272727)
            })
            .bg(if row == self.selected {
                rgb(0x252525)
            } else {
                rgb(0x1b1b1b)
            })
            .cursor_pointer()
            .hover(|style| style.bg(rgb(0x292929)))
            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                this.select_model(selector.clone(), cx)
            }))
            .child(
                div()
                    .h(px(17.))
                    .flex()
                    .items_center()
                    .justify_between()
                    .text_xs()
                    .text_color(if provider_start {
                        rgb(0xe4e4e7)
                    } else {
                        rgb(0x454545)
                    })
                    .child(if provider_start {
                        provider_label(&model.provider)
                    } else {
                        String::new()
                    })
                    .child(if provider_start {
                        format!("{provider_count} models")
                    } else {
                        String::new()
                    }),
            )
            .child(
                div()
                    .flex()
                    .flex_1()
                    .min_w_0()
                    .items_center()
                    .gap_3()
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .min_w_0()
                            .when(compact, |name| {
                                name.w(name_width).flex_none().flex_col().gap_1()
                            })
                            .when(!compact, |name| name.w(px(270.)).flex_none().items_center())
                            .child(
                                div()
                                    .min_w_0()
                                    .when(compact, |name| name.w(name_width))
                                    .text_sm()
                                    .text_color(rgb(0xffffff))
                                    .truncate()
                                    .child(model.name.clone()),
                            )
                            .when(!model.roles.is_empty(), |name| {
                                name.child(div().flex().gap_2().children(model.roles.iter().map(
                                    |role| {
                                        div()
                                            .px_2()
                                            .py_1()
                                            .rounded_sm()
                                            .bg(rgb(0x343434))
                                            .text_size(px(10.))
                                            .text_color(rgb(0xe4e4e7))
                                            .child(provider_label(role))
                                    },
                                )))
                            }),
                    )
                    .when(!compact, |row| {
                        row.child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .text_xs()
                                .text_color(rgb(0xaaaaaa))
                                .truncate()
                                .child(model.selector.clone()),
                        )
                    })
                    .child(
                        div()
                            .w(px(90.))
                            .flex_none()
                            .text_right()
                            .text_xs()
                            .text_color(rgb(0xd4d4d8))
                            .child(model.context.clone()),
                    )
                    .child(
                        div()
                            .w(px(60.))
                            .flex_none()
                            .text_right()
                            .text_xs()
                            .text_color(rgb(0xd4d4d8))
                            .child("n/a"),
                    )
                    .child(
                        div()
                            .w(px(64.))
                            .flex_none()
                            .text_right()
                            .text_xs()
                            .text_color(rgb(0xaaaaaa))
                            .child("n/a"),
                    ),
            )
    }

    fn columns_header(&self, compact: bool, name_width: gpui::Pixels) -> impl IntoElement {
        div()
            .flex_none()
            .flex()
            .items_center()
            .gap_3()
            .px_3()
            .py_1()
            .text_xs()
            .text_color(rgb(0xaaaaaa))
            .child(div().min_w_0().w(name_width).flex_none().child("Model"))
            .when(!compact, |row| row.child(div().flex_1()))
            .child(div().w(px(90.)).flex_none().text_right().child("Context"))
            .child(div().w(px(60.)).flex_none().text_right().child("tok/s"))
            .child(div().w(px(64.)).flex_none().text_right().child("DeepSWE"))
    }

    fn assignments(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let midpoint = self.data.roles.len().div_ceil(2);
        let (left, right) = self.data.roles.split_at(midpoint);
        let column = |items: &[RoleEntry], cx: &mut Context<Self>| {
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_col()
                .children(items.iter().map(|role| {
                    let target = if role.role == "default" {
                        Some(self.active_model.clone())
                    } else {
                        role.resolved.clone()
                    };
                    let model_name = target
                        .as_ref()
                        .and_then(|target| {
                            self.data
                                .models
                                .iter()
                                .find(|model| &model.selector == target)
                        })
                        .map(|model| model.name.to_string())
                        .unwrap_or_else(|| target.clone().unwrap_or_else(|| "Unassigned".into()));
                    let row = div()
                        .id(SharedString::from(role.role.clone()))
                        .flex()
                        .items_center()
                        .justify_between()
                        .gap_2()
                        .px_2()
                        .py_1()
                        .border_t_1()
                        .border_color(rgb(0x303030))
                        .text_xs()
                        .child(
                            div()
                                .w(px(88.))
                                .flex_none()
                                .text_color(rgb(0xd4d4d8))
                                .child(provider_label(&role.role)),
                        )
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .text_color(if target.is_some() {
                                    rgb(0xe4e4e7)
                                } else {
                                    rgb(0x888888)
                                })
                                .truncate()
                                .child(model_name),
                        );
                    if let Some(target) =
                        target.filter(|target| self.data.available.contains(target))
                    {
                        row.cursor_pointer()
                            .hover(|style| style.bg(rgb(0x292929)))
                            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                this.select_model(target.clone(), cx)
                            }))
                    } else {
                        row
                    }
                }))
        };
        div()
            .flex()
            .flex_col()
            .gap_1()
            .p_3()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x333333))
            .child(
                div()
                    .pb_1()
                    .text_sm()
                    .text_color(rgb(0xe4e4e7))
                    .child("Role assignments"),
            )
            .child(
                div()
                    .flex()
                    .gap_3()
                    .child(column(left, cx))
                    .child(div().w(px(1.)).bg(rgb(0x3a3a3a)))
                    .child(column(right, cx)),
            )
    }
}

impl Focusable for ModelPicker {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for ModelPicker {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let compact = window.viewport_size().width < px(800.);
        // Compact names share the remaining width with the header. Reserve the
        // dialog padding (24), border (2), row padding (24), gaps (36), and
        // metric columns (90 + 60 + 64) before measuring truncated text.
        let name_width = if compact {
            (window.viewport_size().width - px(300.)).max(px(0.))
        } else {
            px(270.)
        };
        let message = if self.loading {
            Some("Loading connected providers…".to_string())
        } else if let Some(error) = &self.error {
            Some(error.clone())
        } else if self.data.available.is_empty() {
            Some("No connected models. Sign in to a provider or add an API key.".into())
        } else if self.data.filtered.is_empty() {
            Some("No connected models match your search.".into())
        } else {
            None
        };
        div()
            .key_context("ModelPicker")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(Self::close))
            .on_action(cx.listener(Self::previous))
            .on_action(cx.listener(Self::next))
            .on_action(cx.listener(Self::confirm))
            .flex()
            .items_center()
            .justify_center()
            .size_full()
            .p_6()
            .when(compact, |view| view.p_3())
            .bg(rgb(0x0d0d0d))
            .text_color(rgb(0xe4e4e7))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .w_full()
                    .max_w(px(900.))
                    .h_full()
                    .max_h(px(660.))
                    .min_w_0()
                    .min_h_0()
                    .rounded_lg()
                    .border_1()
                    .border_color(rgb(0x4a4a4a))
                    .bg(rgb(0x191919))
                    .child(
                        div()
                            .flex_none()
                            .flex()
                            .items_center()
                            .justify_between()
                            .px_4()
                            .py_3()
                            .border_b_1()
                            .border_color(rgb(0x333333))
                            .child(
                                div()
                                    .text_size(px(20.))
                                    .text_color(rgb(0xffffff))
                                    .child("Select Model or Role"),
                            )
                            .child(
                                div()
                                    .id("close-picker-btn")
                                    .px_2()
                                    .py_1()
                                    .text_size(px(20.))
                                    .text_color(rgb(0xaaaaaa))
                                    .cursor_pointer()
                                    .hover(|style| style.text_color(rgb(0xffffff)))
                                    .on_click(cx.listener(|_, _: &ClickEvent, _, cx| {
                                        cx.emit(ModelPickerEvent::Close)
                                    }))
                                    .child("×"),
                            ),
                    )
                    .child(div().flex_none().p_3().child(self.search_input.clone()))
                    .child(self.columns_header(compact, name_width))
                    .child(
                        div()
                            .pb_2()
                            .flex_1()
                            .min_h_0()
                            .when(message.is_some(), |list| {
                                list.child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .justify_center()
                                        .h_full()
                                        .text_sm()
                                        .text_color(rgb(0xaaaaaa))
                                        .child(message.clone().unwrap_or_default()),
                                )
                            })
                            .when(!self.data.filtered.is_empty(), |list| {
                                list.child(
                                    uniform_list(
                                        "model-list",
                                        self.data.filtered.len(),
                                        cx.processor(
                                            move |this, range: std::ops::Range<usize>, _, cx| {
                                                range
                                                    .map(|row| {
                                                        this.model_row(
                                                            row,
                                                            this.data.filtered[row],
                                                            compact,
                                                            name_width,
                                                            cx,
                                                        )
                                                    })
                                                    .collect()
                                            },
                                        ),
                                    )
                                    .track_scroll(self.scroll_handle.clone())
                                    .w_full()
                                    .h_full(),
                                )
                            }),
                    )
                    .child(self.assignments(cx))
                    .child(
                        div()
                            .flex_none()
                            .px_4()
                            .py_2()
                            .border_t_1()
                            .border_color(rgb(0x333333))
                            .flex()
                            .items_center()
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
                                    .text_color(rgb(0xd4d4d8))
                                    .cursor_pointer()
                                    .hover(|style| style.text_color(rgb(0xffffff)))
                                    .on_click(cx.listener(|_, _: &ClickEvent, _, cx| {
                                        cx.emit(ModelPickerEvent::ManageProviders)
                                    }))
                                    .child("Manage providers…"),
                            ),
                    ),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disconnected_models_and_role_targets_are_unavailable_after_revocation() {
        let registry = Registry::bundled().unwrap();
        let config = RuntimeConfig {
            model: "anthropic/claude-sonnet-4-5".into(),
            model_roles: [("fast".into(), "openai/gpt-4o:high".into())].into(),
            ..Default::default()
        };
        let mut data = PickerData::new(&config, &registry);
        assert!(data.filtered.is_empty());
        assert_eq!(
            data.roles
                .iter()
                .filter(|role| role.resolved.is_some())
                .count(),
            2
        );
        data.connect(BTreeSet::from(["openai".into()]), "");
        assert!(data.available.contains("openai/gpt-4o"));
        assert!(!data.available.contains("anthropic/claude-sonnet-4-5"));
        assert!(
            data.filtered
                .iter()
                .all(|index| data.models[*index].provider == "openai")
        );
        assert_eq!(
            data.roles
                .iter()
                .filter(|role| role
                    .resolved
                    .as_ref()
                    .is_some_and(|resolved| data.available.contains(resolved)))
                .map(|role| role.role.as_str())
                .collect::<Vec<_>>(),
            ["fast"]
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
        assert!(!data.roles.iter().any(|role| {
            role.resolved
                .as_ref()
                .is_some_and(|target| data.available.contains(target))
        }));
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
