//! Connected-provider model and role picker. Filtering is cached; only visible
//! rows are rendered, even when an authenticated gateway has thousands of models.

use std::collections::BTreeSet;

use gpui::{
    App, ClickEvent, Context, Entity, FocusHandle, Focusable, KeyBinding, ScrollStrategy,
    SharedString, Subscription, Task, UniformListScrollHandle, Window, actions, div, prelude::*,
    px, rgb, uniform_list,
};
use juto_ai::judge_role_accepts;
use juto_catalog::Model;
use juto_daemon::client::{Client, Notification};
use juto_daemon::protocol::{Reply, Request, ServerEvent};
use juto_runtime::RuntimeConfig;
use tokio::sync::broadcast::error::RecvError;

use crate::chat_request::call;
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
    judge: bool,
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
            judge: model.kind == "judge",
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

#[derive(Default)]
struct PickerData {
    config: RuntimeConfig,
    catalog: Vec<Model>,
    models: Vec<ModelEntry>,
    roles: Vec<RoleEntry>,
    connected: BTreeSet<String>,
    available: BTreeSet<String>,
    filtered: Vec<usize>,
    query: Option<String>,
    judge_role: bool,
}

/// Role rows and the selector each resolves to. Resolution uses the full
/// catalog, so a role may target a model the picker does not offer.
fn role_entries(config: &RuntimeConfig, catalog: &[Model]) -> Vec<RoleEntry> {
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
            let resolved = config
                .resolve_selector(&format!("@{name}"))
                .ok()
                .and_then(|(target, _)| resolve_model(catalog, &target))
                .map(|model| format!("{}/{}", model.provider, model.id));
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
    roles
}

/// Selector lookup with `juto_catalog::Registry::resolve` semantics: a
/// `provider/id` selector naming a known provider is locked to it, and a bare
/// id must match exactly one row.
fn resolve_model<'a>(catalog: &'a [Model], selector: &str) -> Option<&'a Model> {
    if let Some((provider, id)) = selector.split_once('/')
        && catalog
            .iter()
            .any(|model| model.provider.eq_ignore_ascii_case(provider))
    {
        return catalog.iter().find(|model| {
            model.provider.eq_ignore_ascii_case(provider) && model.id.eq_ignore_ascii_case(id)
        });
    }
    let mut found = catalog
        .iter()
        .filter(|model| model.id.eq_ignore_ascii_case(selector));
    let first = found.next()?;
    found.next().is_none().then_some(first)
}

impl PickerData {
    /// Replaces the catalog and config, then rebuilds rows and roles against `query`.
    fn load(&mut self, config: RuntimeConfig, catalog: Vec<Model>, query: &str) {
        self.config = config;
        self.catalog = catalog;
        self.rebuild(query);
    }

    /// Applies a role-binding change reported by the daemon.
    fn set_config(&mut self, config: RuntimeConfig, query: &str) {
        self.config = config;
        self.rebuild(query);
    }

    fn rebuild(&mut self, query: &str) {
        let mut models = self
            .catalog
            .iter()
            .filter(|model| judge_role_accepts(model))
            .map(ModelEntry::from_catalog)
            .collect::<Vec<_>>();
        models.sort_by(|left, right| {
            left.provider
                .cmp(&right.provider)
                .then_with(|| left.name.to_lowercase().cmp(&right.name.to_lowercase()))
        });
        self.models = models;
        self.roles = role_entries(&self.config, &self.catalog);
        for model in &mut self.models {
            model.roles = self
                .roles
                .iter()
                .filter(|role| role.resolved.as_deref() == Some(model.selector.as_str()))
                .map(|role| role.role.clone())
                .collect();
        }
        let connected = std::mem::take(&mut self.connected);
        self.connect(connected, query);
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
                (self.judge_role || !model.judge)
                    && self.connected.contains(&model.provider)
                    && model.search.contains(&query)
            })
            .map(|(index, _)| index)
            .collect();
        self.query = Some(query);
        true
    }

    /// Switch between chat and judge scopes and re-filter with the current query.
    fn set_judge_role(&mut self, judge_role: bool, query: &str) {
        self.judge_role = judge_role;
        self.query = None;
        self.filter(query);
    }
}

#[derive(Clone, Debug)]
pub enum ModelPickerEvent {
    AssignRole { role: String, selector: String },
    Close,
    ManageProviders,
}

/// Picker state fetched from the daemon in one pass.
struct PickerLoad {
    config: RuntimeConfig,
    catalog: Vec<Model>,
    available: BTreeSet<String>,
}

async fn load_picker(
    client: Client,
    tokio_handle: tokio::runtime::Handle,
) -> Result<PickerLoad, String> {
    let config = match call(client.clone(), tokio_handle.clone(), Request::GetConfig).await {
        Ok(Reply::Config(config)) => config,
        Ok(_) => return Err("Unexpected daemon reply to config request".into()),
        Err(error) => return Err(crate::chat_request::describe_error(&error)),
    };
    let catalog = match call(client.clone(), tokio_handle.clone(), Request::ListModels).await {
        Ok(Reply::Models(catalog)) => catalog,
        Ok(_) => return Err("Unexpected daemon reply to model list".into()),
        Err(error) => return Err(crate::chat_request::describe_error(&error)),
    };
    let providers: Vec<String> = catalog
        .iter()
        .map(|model| model.provider.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let available: BTreeSet<String> = match call(
        client,
        tokio_handle,
        Request::CredentialStatus { providers },
    )
    .await
    {
        Ok(Reply::Credentials(status)) => status.available.into_iter().collect(),
        Ok(_) => return Err("Unexpected daemon reply to credential status".into()),
        Err(error) => return Err(crate::chat_request::describe_error(&error)),
    };
    Ok(PickerLoad {
        config,
        catalog,
        available,
    })
}

pub struct ModelPicker {
    client: Client,
    tokio_handle: tokio::runtime::Handle,
    focus_handle: FocusHandle,
    search_input: Entity<TextInput>,
    active_model: String,
    assign_role: String,
    data: PickerData,
    selected: usize,
    scroll_handle: UniformListScrollHandle,
    loading: bool,
    error: Option<String>,
    refresh_task: Option<Task<()>>,
    assign_task: Option<Task<()>>,
    _notifications_task: Task<()>,
    _subscriptions: Vec<Subscription>,
}

impl gpui::EventEmitter<ModelPickerEvent> for ModelPicker {}

impl ModelPicker {
    pub fn new(
        client: Client,
        tokio_handle: tokio::runtime::Handle,
        active_model: impl Into<String>,
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
        let mut notifications = client.subscribe();
        let notifications_task = cx.spawn(async move |this, cx| {
            loop {
                let update = match notifications.recv().await {
                    Ok(Notification::Event(ServerEvent::Config { config })) => {
                        this.update(cx, |this, cx| {
                            this.data
                                .set_config(config, this.search_input.read(cx).text());
                            cx.notify();
                        })
                    }
                    Ok(Notification::Connected(_)) | Err(RecvError::Lagged(_)) => {
                        this.update(cx, |this, cx| this.refresh_credentials(cx))
                    }
                    Ok(Notification::Disconnected) => this.update(cx, |this, cx| {
                        this.data
                            .connect(BTreeSet::new(), this.search_input.read(cx).text());
                        this.error = Some("Daemon disconnected".into());
                        cx.notify();
                    }),
                    Ok(_) => continue,
                    Err(RecvError::Closed) => break,
                };
                if update.is_err() {
                    break;
                }
            }
        });
        Self {
            client,
            tokio_handle,
            focus_handle: cx.focus_handle(),
            search_input,
            active_model,
            assign_role: "default".into(),
            data: PickerData::default(),
            selected: 0,
            scroll_handle: UniformListScrollHandle::new(),
            loading: true,
            error: None,
            refresh_task: None,
            assign_task: None,
            _notifications_task: notifications_task,
            _subscriptions: vec![subscription, submitted],
        }
    }

    /// Reloads config, the model catalog, and which providers hold usable
    /// credentials. Models stay hidden until the daemon confirms them.
    pub fn refresh_credentials(&mut self, cx: &mut Context<Self>) {
        self.loading = true;
        self.error = None;
        // Fail closed while a refresh is pending; removed providers cannot remain selectable.
        self.data
            .connect(BTreeSet::new(), self.search_input.read(cx).text());
        let client = self.client.clone();
        let tokio_handle = self.tokio_handle.clone();
        self.refresh_task = Some(cx.spawn(async move |this, cx| {
            let result = load_picker(client, tokio_handle).await;
            let _ = this.update(cx, |this, cx| {
                this.loading = false;
                match result {
                    Ok(loaded) => {
                        let query = this.search_input.read(cx).text();
                        this.data.load(loaded.config, loaded.catalog, query);
                        this.data.connect(loaded.available, query);
                    }
                    Err(error) => this.error = Some(format!("Could not load models: {error}")),
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

    fn select_model(&mut self, selector: String, cx: &mut Context<Self>) {
        if self.assign_task.is_some() {
            return;
        }
        let role = self.assign_role.clone();
        let request = Request::SetModelRole {
            role: role.clone(),
            selector: selector.clone(),
        };
        let client = self.client.clone();
        let tokio_handle = self.tokio_handle.clone();
        self.error = None;
        self.assign_task = Some(cx.spawn(async move |this, cx| {
            let result = call(client, tokio_handle, request).await;
            let _ = this.update(cx, |this, cx| {
                this.assign_task = None;
                match result {
                    Ok(Reply::Config(config)) => {
                        this.data
                            .set_config(config, this.search_input.read(cx).text());
                        if role == "default" {
                            this.active_model = selector.clone();
                        }
                        cx.emit(ModelPickerEvent::AssignRole { role, selector });
                    }
                    Ok(_) => {
                        this.error = Some("Unexpected daemon reply to role assignment".into());
                    }
                    Err(error) => {
                        this.error = Some(format!("Could not assign @{role}: {error}"));
                    }
                }
                cx.notify();
            });
        }));
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
        narrow: bool,
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
                            .when(narrow, |name| name.flex_1().flex_col().gap_1())
                            .when(!narrow, |name| name.w(px(270.)).flex_none().items_center())
                            .child(
                                div()
                                    .min_w_0()
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
                    .when(!narrow, |row| {
                        row.child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .text_xs()
                                .text_color(rgb(0xaaaaaa))
                                .truncate()
                                .child(model.selector.clone()),
                        )
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
                        )
                    }),
            )
    }

    fn columns_header(&self, narrow: bool) -> impl IntoElement {
        let header = div()
            .flex_none()
            .flex()
            .items_center()
            .gap_3()
            .px_3()
            .py_1()
            .text_xs()
            .text_color(rgb(0xaaaaaa))
            .child(
                div()
                    .min_w_0()
                    .when(narrow, |model| model.flex_1())
                    .when(!narrow, |model| model.w(px(270.)).flex_none())
                    .child("Model"),
            );
        if narrow {
            header
        } else {
            header
                .child(div().flex_1())
                .child(div().w(px(90.)).flex_none().text_right().child("Context"))
                .child(div().w(px(60.)).flex_none().text_right().child("tok/s"))
                .child(div().w(px(64.)).flex_none().text_right().child("DeepSWE"))
        }
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
                    let is_target = self.assign_role == role.role;
                    let role_name = role.role.clone();
                    div()
                        .id(SharedString::from(role.role.clone()))
                        .flex()
                        .items_center()
                        .justify_between()
                        .gap_2()
                        .px_2()
                        .py_1()
                        .border_t_1()
                        .border_color(if is_target {
                            rgb(0x4c5f9e)
                        } else {
                            rgb(0x303030)
                        })
                        .bg(if is_target {
                            rgb(0x1d2438)
                        } else {
                            rgb(0x191919)
                        })
                        .text_xs()
                        .cursor_pointer()
                        .hover(|style| style.bg(rgb(0x232c44)))
                        .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                            this.assign_role = role_name.clone();
                            let judge = this.assign_role == "judge";
                            this.data
                                .set_judge_role(judge, this.search_input.read(cx).text());
                            this.selected = 0;
                            this.scroll_handle.scroll_to_item(0, ScrollStrategy::Top);
                            this.error = None;
                            cx.notify();
                        }))
                        .child(
                            div()
                                .w(px(88.))
                                .flex_none()
                                .text_color(if is_target {
                                    rgb(0xa5b8f3)
                                } else {
                                    rgb(0xd4d4d8)
                                })
                                .font_weight(if is_target {
                                    gpui::FontWeight::SEMIBOLD
                                } else {
                                    gpui::FontWeight::NORMAL
                                })
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
                        )
                        .when(is_target, |row| {
                            row.child(
                                div()
                                    .px_1p5()
                                    .py_0p5()
                                    .rounded_sm()
                                    .bg(rgb(0x2c3a63))
                                    .text_size(px(10.))
                                    .text_color(rgb(0xc7d4ff))
                                    .child("target"),
                            )
                        })
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
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .text_sm()
                            .text_color(rgb(0xe4e4e7))
                            .child("Role assignments"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x888888))
                            .child(format!("Assigning to @{}", self.assign_role)),
                    ),
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
        let panel_width = (window.viewport_size().width - px(48.)).min(px(900.));
        // Below 640px the selector and metric columns are hidden so the model
        // name fills the row instead of pushing content past the dialog edge.
        let narrow = panel_width < px(640.);
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
            .flex_col()
            .w(px(900.))
            .max_w(window.viewport_size().width - px(48.))
            .h(px(660.))
            .max_h(window.viewport_size().height - px(48.))
            .min_w_0()
            .min_h_0()
            .rounded_lg()
            .border_1()
            .border_color(rgb(0x4a4a4a))
            .bg(rgb(0x191919))
            .shadow_xl()
            .text_color(rgb(0xe4e4e7))
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
            .child(self.columns_header(narrow))
            .child(
                div()
                    .pb_2()
                    .flex_1()
                    .min_h_0()
                    .when(message.is_some(), |list| {
                        list.child(
                            div()
                                .w_full()
                                .h_full()
                                .flex()
                                .items_center()
                                .justify_center()
                                .px_4()
                                .child(
                                    div()
                                        .w_full()
                                        .text_sm()
                                        .text_center()
                                        .text_color(rgb(0xaaaaaa))
                                        .child(message.clone().unwrap_or_default()),
                                ),
                        )
                    })
                    .when(!self.data.filtered.is_empty(), |list| {
                        list.child(
                            uniform_list(
                                "model-list",
                                self.data.filtered.len(),
                                cx.processor(move |this, range: std::ops::Range<usize>, _, cx| {
                                    range
                                        .map(|row| {
                                            this.model_row(row, this.data.filtered[row], narrow, cx)
                                        })
                                        .collect()
                                }),
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
                    .flex_wrap()
                    .gap_2()
                    .child(div().text_xs().text_color(rgb(0x888888)).child(format!(
                        "↑ ↓ Navigate · Enter assigns to @{}",
                        self.assign_role
                    )))
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
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(provider: &str, id: &str, api: &str, kind: &str) -> Model {
        Model {
            id: id.into(),
            name: id.into(),
            api: api.into(),
            provider: provider.into(),
            base_url: String::new(),
            reasoning: false,
            input: vec!["text".into()],
            cost: Default::default(),
            context_window: None,
            max_tokens: None,
            kind: kind.into(),
            supports_tools: true,
            compat: Default::default(),
        }
    }

    #[test]
    fn daemon_catalog_selectors_preserve_ambiguity_and_provider_locking() {
        let catalog = vec![
            model("openai", "shared", "openai-responses", "chat"),
            model("anthropic", "shared", "anthropic-messages", "chat"),
            model("openrouter", "vendor/model", "openai-responses", "chat"),
        ];
        assert!(resolve_model(&catalog, "shared").is_none());
        assert_eq!(
            resolve_model(&catalog, "OPENAI/SHARED").map(|model| model.provider.as_str()),
            Some("openai")
        );
        assert!(resolve_model(&catalog, "openai/vendor/model").is_none());
        assert_eq!(
            resolve_model(&catalog, "vendor/model").map(|model| model.provider.as_str()),
            Some("openrouter")
        );
    }

    #[test]
    fn disconnected_models_and_role_targets_are_unavailable_after_revocation() {
        let catalog = vec![
            model(
                "anthropic",
                "claude-sonnet-4-5",
                "anthropic-messages",
                "chat",
            ),
            model("openai", "gpt-4o", "openai-responses", "chat"),
        ];
        let config = RuntimeConfig {
            model: "anthropic/claude-sonnet-4-5".into(),
            model_roles: [("fast".into(), "openai/gpt-4o:high".into())].into(),
            ..Default::default()
        };
        let mut data = PickerData::default();
        data.load(config, catalog, "");
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
        let catalog = vec![
            model("openai", "chat", "openai-responses", "chat"),
            model("openai", "embedding", "openai-responses", "embedding"),
            model("openai", "unsupported", "future-transport", "chat"),
        ];
        let mut data = PickerData::default();
        data.load(RuntimeConfig::default(), catalog, "");
        data.connect(BTreeSet::from(["openai".into()]), "");
        assert_eq!(data.available, BTreeSet::from(["openai/chat".into()]));
    }

    #[test]
    fn devin_models_are_offered_once_connected() {
        let catalog = vec![model("devin", "swe-1-6", "devin-agent", "chat")];
        let mut data = PickerData::default();
        data.load(RuntimeConfig::default(), catalog, "");
        data.connect(BTreeSet::new(), "");
        assert!(data.available.is_empty());
        data.connect(BTreeSet::from(["devin".into()]), "");
        assert_eq!(data.available, BTreeSet::from(["devin/swe-1-6".into()]));
    }

    #[test]
    fn typesafe_judge_models_are_offered_only_for_the_judge_role() {
        let catalog = vec![model("typesafe", "jev-preview", "typesafe", "judge")];
        let mut data = PickerData::default();
        data.load(RuntimeConfig::default(), catalog, "");
        data.connect(BTreeSet::from(["typesafe".into()]), "");
        assert!(data.filtered.is_empty());
        data.set_judge_role(true, "");
        assert_eq!(data.filtered.len(), 1);
        assert_eq!(
            data.models[data.filtered[0]].selector,
            "typesafe/jev-preview"
        );
    }

    #[test]
    fn judge_scope_offers_chat_models_and_chat_scope_hides_judge_models() {
        let catalog = vec![
            model("typesafe", "jev-preview", "typesafe", "judge"),
            model("openai", "gpt-4o", "openai-responses", "chat"),
        ];
        let mut data = PickerData::default();
        data.load(RuntimeConfig::default(), catalog, "");
        data.connect(BTreeSet::from(["typesafe".into(), "openai".into()]), "");
        let selectors = |data: &PickerData| -> Vec<String> {
            data.filtered
                .iter()
                .map(|&index| data.models[index].selector.clone())
                .collect()
        };
        assert_eq!(selectors(&data), ["openai/gpt-4o"]);
        data.set_judge_role(true, "");
        assert_eq!(selectors(&data), ["openai/gpt-4o", "typesafe/jev-preview"]);
    }
}
