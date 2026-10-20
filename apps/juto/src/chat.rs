//! Chat UI consuming workspace-daemon snapshots and events.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::Quit;
use crate::auth_modal::{AuthModal, AuthModalEvent};
use crate::chat_input::{ChatInput, SubmitMessage};
use crate::focus_view::{FocusEvent, FocusView};
use crate::model_picker::{ModelPicker, ModelPickerEvent};
use crate::sidebar::{Sidebar, SidebarEvent};
use crate::subagent_card;
use crate::theme;
use crate::view_element::{ViewActions, render_view};
use gpui::{
    AnyElement, App, ClickEvent, Context, Div, Entity, Focusable, IntoElement, KeyBinding,
    MouseButton, MouseDownEvent, PathPromptOptions, Render, ScrollHandle, SharedString, Stateful,
    Subscription, Task, Window, actions, deferred, div, prelude::*, px, rgb, rgba,
};
use juto_agent::{AgentEvent, ToolTier, ViewNode, view_from_details};
use juto_ai::{Message as AiMessage, ProviderEvent, StopReason, ToolCall};
use juto_daemon::client::{Client, Notification};
use juto_daemon::protocol::{
    PendingApproval, Reply, Request, ServerEvent, SessionEvent, SessionView, SubagentCommand,
};
use juto_runtime::{DesktopSessionArtifact, EntryKind, SubagentSummary};
use tokio::sync::broadcast::{self, error::RecvError};

actions!(
    chat,
    [
        CloseAttachmentMenu,
        ToggleModelPicker,
        ToggleAuthModal,
        ToggleSidebar
    ]
);

pub fn init(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("escape", CloseAttachmentMenu, Some("ChatView")),
        KeyBinding::new("ctrl-m", ToggleModelPicker, None),
        KeyBinding::new("cmd-m", ToggleModelPicker, None),
        KeyBinding::new("ctrl-p", ToggleAuthModal, None),
        KeyBinding::new("cmd-p", ToggleAuthModal, None),
        KeyBinding::new("ctrl-b", ToggleSidebar, None),
        KeyBinding::new("cmd-b", ToggleSidebar, None),
    ]);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActiveModal {
    None,
    ModelPicker,
    AuthModal,
    DesktopViewer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageRole {
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attachment {
    pub path: PathBuf,
    pub label: SharedString,
}

impl Attachment {
    pub fn new(path: PathBuf) -> Self {
        let label = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        Self {
            path,
            label: label.into(),
        }
    }
}

/// Appends newly picked paths, keeping the first entry for any path already attached.
pub fn add_attachments(attachments: &mut Vec<Attachment>, paths: Vec<PathBuf>) {
    for path in paths {
        if !attachments.iter().any(|attachment| attachment.path == path) {
            attachments.push(Attachment::new(path));
        }
    }
}

pub fn format_prompt(text: &str, attachments: &[Attachment]) -> String {
    let mut prompt = String::new();
    if !attachments.is_empty() {
        prompt.push_str("Attached files:\n");
        for attachment in attachments {
            prompt.push_str(&format!("- {}\n", attachment.path.display()));
        }
        if !text.trim().is_empty() {
            prompt.push('\n');
        }
    }
    if !text.trim().is_empty() {
        prompt.push_str(text);
    }
    prompt
}

pub fn extract_desktop_artifacts(message: &AiMessage) -> Vec<DesktopSessionArtifact> {
    let mut artifacts = Vec::new();
    match message {
        AiMessage::ToolResult { details, .. } => {
            if details.get("type").and_then(|t| t.as_str()) == Some("desktop_session_artifact")
                && let Some(art) = details.get("artifact")
                && let Ok(artifact) = serde_json::from_value::<DesktopSessionArtifact>(art.clone())
            {
                artifacts.push(artifact);
            }
        }
        AiMessage::Assistant(assistant) => {
            // In case artifacts are mirrored in assistant metadata/tool calls
            for call in assistant.tool_calls() {
                if call.name == "desktop_artifact"
                    && let Ok(artifact) =
                        serde_json::from_value::<DesktopSessionArtifact>(call.arguments.clone())
                {
                    artifacts.push(artifact);
                }
            }
        }
        _ => {}
    }
    artifacts
}

pub fn load_messages(entries: &[juto_runtime::SessionEntry]) -> Vec<Message> {
    view_messages(entries.iter().filter_map(|entry| match &entry.kind {
        EntryKind::Message { message } => Some(message),
        _ => None,
    }))
}

/// Converts agent messages into transcript rows. Shared by the main transcript
/// and focus views.
pub fn view_messages<'a>(source: impl IntoIterator<Item = &'a AiMessage>) -> Vec<Message> {
    let mut messages = Vec::new();
    for message in source {
        match message {
            AiMessage::User { .. } => {
                messages.push(Message {
                    role: MessageRole::User,
                    text: message.text().into(),
                    attachments: Vec::new(),
                    desktop_artifacts: Vec::new(),
                    approval: None,
                    views: Vec::new(),
                });
            }
            AiMessage::Assistant(assistant) => {
                let text = assistant.text();
                let desktop_artifacts = extract_desktop_artifacts(message);
                if !text.is_empty() || !desktop_artifacts.is_empty() {
                    messages.push(Message {
                        role: MessageRole::Assistant,
                        text: text.into(),
                        attachments: Vec::new(),
                        desktop_artifacts,
                        approval: None,
                        views: Vec::new(),
                    });
                }
            }
            AiMessage::ToolResult { details, .. } => {
                let desktop_artifacts = extract_desktop_artifacts(message);
                if !desktop_artifacts.is_empty() {
                    if let Some(last) = messages.last_mut()
                        && last.role == MessageRole::Assistant
                    {
                        last.desktop_artifacts.extend(desktop_artifacts);
                    } else {
                        messages.push(Message {
                            role: MessageRole::Assistant,
                            text: SharedString::default(),
                            attachments: Vec::new(),
                            desktop_artifacts,
                            approval: None,
                            views: Vec::new(),
                        });
                    }
                }
                if let Some(view) = view_from_details(details) {
                    messages.push(view_row(view));
                }
            }
            _ => {}
        }
    }
    messages
}

/// Transcript row that shows a tool's view tree in place of text output.
pub fn view_row(view: ViewNode) -> Message {
    Message {
        role: MessageRole::Assistant,
        text: SharedString::default(),
        attachments: Vec::new(),
        desktop_artifacts: Vec::new(),
        approval: None,
        views: vec![view],
    }
}

/// Renders a view. A subagent card is clickable and opens its focus thread.
fn card_element(view: &ViewNode, actions: &ViewActions, cx: &Context<ChatView>) -> AnyElement {
    let Some(id) = subagent_card::card_id(view) else {
        return render_view(view, actions);
    };
    let id = id.to_owned();
    div()
        .id(SharedString::from(format!("subagent-card-{id}")))
        .cursor_pointer()
        .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
            this.enter_focus(&id, cx);
        }))
        .child(render_view(view, actions))
        .into_any_element()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: MessageRole,
    pub text: SharedString,
    pub attachments: Vec<Attachment>,
    pub desktop_artifacts: Vec<DesktopSessionArtifact>,
    pub approval: Option<ApprovalCard>,
    pub views: Vec<ViewNode>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApprovalState {
    Pending,
    Allowed,
    Denied,
    Cancelled,
}

/// Inline permission prompt for one tool call. The responder that answers it
/// lives in `ChatView::pending_approvals`; this is only the rendered record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApprovalCard {
    pub id: u64,
    pub tool: SharedString,
    pub tier: ToolTier,
    /// Exact bash command, or pretty-printed arguments for other tools.
    pub detail: SharedString,
    pub cwd: Option<SharedString>,
    pub state: ApprovalState,
}

fn bash_command(call: &ToolCall) -> Option<&str> {
    if call.name == "bash" {
        call.arguments.get("command")?.as_str()
    } else {
        None
    }
}

fn approval_cwd(call: &ToolCall) -> Option<&str> {
    if call.name == "bash" {
        call.arguments.get("cwd")?.as_str()
    } else {
        None
    }
}

/// What the operator is approving: the exact bash command, or the tool's
/// arguments verbatim as pretty-printed JSON.
fn approval_detail(call: &ToolCall) -> String {
    if let Some(command) = bash_command(call) {
        return command.to_owned();
    }
    serde_json::to_string_pretty(&call.arguments).unwrap_or_else(|_| call.arguments.to_string())
}

pub struct ChatView {
    input: Entity<ChatInput>,
    model_picker: Entity<ModelPicker>,
    auth_modal: Entity<AuthModal>,
    sidebar: Entity<Sidebar>,
    focus: Option<Entity<FocusView>>,
    _focus_subscription: Option<Subscription>,
    subagents: Vec<SubagentSummary>,
    show_sidebar: bool,
    active_modal: ActiveModal,
    active_model: SharedString,
    messages: Vec<Message>,
    attachments: Vec<Attachment>,
    attachment_menu_open: bool,
    menu_just_dismissed: bool,
    attachment_error: Option<SharedString>,
    picker: Option<Task<()>>,
    scroll: ScrollHandle,
    client: Client,
    session_id: String,
    committed: u64,
    streaming_row: Option<usize>,
    tokio_handle: tokio::runtime::Handle,
    is_running: bool,
    connected: bool,
    artifact_notice: Option<SharedString>,
    desktop_viewer: Option<Entity<crate::desktop_viewer::DesktopViewer>>,
    pending_approvals: HashMap<u64, PendingApproval>,
    _subscriptions: Vec<Subscription>,
    _daemon_events: Task<()>,
}

impl ChatView {
    pub fn new(
        client: Client,
        tokio_handle: tokio::runtime::Handle,
        session: SessionView,
        default_model: String,
        mut notifications: broadcast::Receiver<Notification>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let input = cx.new(ChatInput::new);
        input.read(cx).focus_handle(cx).focus(window);
        let initial_model = session.info.model.clone().unwrap_or(default_model);
        let model_picker =
            cx.new(|cx| ModelPicker::new(client.clone(), tokio_handle.clone(), &initial_model, cx));
        let auth_modal = cx.new(|cx| AuthModal::new(client.clone(), tokio_handle.clone(), cx));
        let sidebar = cx.new(Sidebar::new);
        let submitted =
            cx.subscribe_in(&input, window, |this, _, _: &SubmitMessage, window, cx| {
                this.send_message(window, cx);
            });
        let changed = cx.observe(&input, |_, _, cx| cx.notify());

        let model_picker_sub = cx.subscribe_in(
            &model_picker,
            window,
            |this, _, event: &ModelPickerEvent, window, cx| match event {
                ModelPickerEvent::AssignRole { role, selector } => {
                    this.assign_role(role.clone(), selector.clone(), window, cx);
                }
                ModelPickerEvent::Close => {
                    this.close_modal(window, cx);
                }
                ModelPickerEvent::ManageProviders => {
                    this.open_auth_modal(window, cx);
                }
            },
        );
        let auth_modal_sub = cx.subscribe_in(
            &auth_modal,
            window,
            |this, _, event: &AuthModalEvent, window, cx| match event {
                AuthModalEvent::Close => {
                    this.close_modal(window, cx);
                }
                AuthModalEvent::CredentialsChanged => {
                    this.model_picker.update(cx, |picker, cx| {
                        picker.refresh_credentials(cx);
                    });
                    cx.notify();
                }
            },
        );

        let sidebar_sub = cx.subscribe(&sidebar, |this, _, event: &SidebarEvent, cx| match event {
            SidebarEvent::Close => {
                this.show_sidebar = false;
                cx.notify();
            }
            SidebarEvent::SelectTask(id) => this.enter_focus(id, cx),
            SidebarEvent::PauseTask(id) => {
                this.subagent_command(id.clone(), SubagentCommand::Pause, cx)
            }
            SidebarEvent::AbortTask(id) => {
                this.subagent_command(id.clone(), SubagentCommand::Cancel, cx)
            }
            SidebarEvent::ResumeTask { id, resolution } => this.subagent_command(
                id.clone(),
                SubagentCommand::Unblock {
                    resolution: resolution.clone(),
                },
                cx,
            ),
            SidebarEvent::SteerTask { id, guidance } => this.subagent_command(
                id.clone(),
                SubagentCommand::Send {
                    text: guidance.clone(),
                },
                cx,
            ),
        });

        let messages = load_messages(&session.entries);
        let event_client = client.clone();
        let event_handle = tokio_handle.clone();
        let session_id = session.info.id.clone();
        let reattach_id = session_id.clone();
        let daemon_events = cx.spawn_in(window, async move |this, cx| {
            // Subscribe before taking snapshots. Journal indexes make replay idempotent.
            let mut reattach = true;
            loop {
                if reattach {
                    let result = crate::chat_request::call(
                        event_client.clone(),
                        event_handle.clone(),
                        Request::OpenSession {
                            session_id: reattach_id.clone(),
                        },
                    )
                    .await;
                    if this
                        .update_in(cx, |this, _, cx| {
                            match result {
                                Ok(Reply::Session(view)) => this.apply_snapshot(view, cx),
                                Ok(_) => this.report_error("Unexpected session reply".into(), cx),
                                Err(error) => this.report_error(error.to_string(), cx),
                            }
                            this.sync_subagents(cx);
                        })
                        .is_err()
                    {
                        break;
                    }
                    reattach = false;
                }
                match notifications.recv().await {
                    Ok(Notification::Connected(_)) | Err(RecvError::Lagged(_)) => reattach = true,
                    Ok(Notification::Disconnected) => {
                        if this
                            .update_in(cx, |this, _, cx| {
                                this.connected = false;
                                this.artifact_notice =
                                    Some("Daemon disconnected; reconnecting…".into());
                                cx.notify();
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                    Ok(Notification::Event(event)) => {
                        if this
                            .update_in(cx, |this, window, cx| {
                                this.handle_server_event(event, window, cx)
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(RecvError::Closed) => break,
                }
            }
        });
        Self {
            input,
            model_picker,
            auth_modal,
            sidebar,
            focus: None,
            _focus_subscription: None,
            subagents: Vec::new(),
            show_sidebar: true,
            active_modal: ActiveModal::None,
            active_model: initial_model.into(),
            messages,
            attachments: Vec::new(),
            attachment_menu_open: false,
            menu_just_dismissed: false,
            attachment_error: None,
            artifact_notice: None,
            desktop_viewer: None,
            picker: None,
            scroll: ScrollHandle::new(),
            client,
            session_id,
            committed: session.committed,
            streaming_row: None,
            tokio_handle,
            is_running: session.info.run.is_some(),
            connected: true,
            pending_approvals: HashMap::new(),
            _subscriptions: vec![
                submitted,
                changed,
                model_picker_sub,
                auth_modal_sub,
                sidebar_sub,
            ],
            _daemon_events: daemon_events,
        }
    }

    fn assign_role(
        &mut self,
        role: String,
        selector: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // The picker persists role assignments before emitting this event.
        if role == "default" {
            self.set_model(selector, window, cx);
        }
    }

    pub fn set_model(&mut self, selector: String, window: &mut Window, cx: &mut Context<Self>) {
        let request = Request::SetSessionModel {
            session_id: self.session_id.clone(),
            selector: selector.clone(),
        };
        let client = self.client.clone();
        let handle = self.tokio_handle.clone();
        cx.spawn_in(window, async move |this, cx| {
            let result = crate::chat_request::call(client, handle, request).await;
            let _ = this.update_in(cx, |this, window, cx| match result {
                Ok(Reply::Ok) => {
                    this.active_model = selector.into();
                    this.close_modal(window, cx);
                }
                Ok(_) => this.report_error("Unexpected model reply".into(), cx),
                Err(error) => this.report_error(error.to_string(), cx),
            });
        })
        .detach();
    }

    pub fn open_model_picker(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.active_modal == ActiveModal::AuthModal {
            self.auth_modal.update(cx, |modal, cx| modal.dismiss(cx));
        }
        self.active_modal = ActiveModal::ModelPicker;
        self.model_picker.update(cx, |picker, cx| {
            picker.refresh_credentials(cx);
        });
        self.model_picker.read(cx).focus(window, cx);
        cx.notify();
    }

    pub fn open_auth_modal(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.active_modal = ActiveModal::AuthModal;
        self.auth_modal
            .update(cx, |modal, cx| modal.refresh_credentials(cx));
        self.auth_modal.read(cx).focus(window, cx);
        cx.notify();
    }

    /// Resolves a desktop artifact and opens the viewer it authorizes.
    ///
    /// Resolution is explicit: an unknown host, environment, or grant reports
    /// `Unsupported`, `Unavailable`, `Unauthorized`, `Expired`, or `Stopped`
    /// rather than fabricating a live stream.
    pub fn open_desktop_artifact(
        &mut self,
        artifact: &DesktopSessionArtifact,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let request = Request::ResolveDesktopArtifact {
            artifact: serde_json::to_value(artifact).expect("serializable desktop artifact"),
        };
        let client = self.client.clone();
        let handle = self.tokio_handle.clone();
        cx.spawn_in(window, async move |this, cx| {
            let result = crate::chat_request::call(client, handle, request).await;
            let _ = this.update_in(cx, |this, window, cx| {
                match result {
                    Ok(Reply::Route(Some(juto_runtime::ViewerRoute::Local { env_id, .. }))) => {
                        this.open_desktop_viewer(&env_id, window, cx)
                    }
                    Ok(Reply::Route(Some(juto_runtime::ViewerRoute::Remote { url, .. }))) => {
                        this.artifact_notice =
                            Some(format!("Remote desktop: {url} (one-time token, 60s)").into())
                    }
                    Ok(Reply::Route(None)) => {
                        this.artifact_notice =
                            Some("Desktop artifact cannot be routed by this daemon".into())
                    }
                    Ok(_) => this.report_error("Unexpected artifact reply".into(), cx),
                    Err(error) => this.report_error(error.to_string(), cx),
                }
                cx.notify();
            });
        })
        .detach();
    }

    pub fn open_desktop_viewer(
        &mut self,
        env_id: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let client = self.client.clone();
        let title = format!("Desktop {env_id}");
        self.desktop_viewer = Some(cx.new(|cx| {
            crate::desktop_viewer::DesktopViewer::new(client, env_id.to_owned(), title, window, cx)
        }));
        self.active_modal = ActiveModal::DesktopViewer;
        self.artifact_notice = None;
        cx.notify();
    }

    pub fn open_desktop_command(
        &mut self,
        argv: Vec<String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let client = self.client.clone();
        let handle = self.tokio_handle.clone();
        cx.spawn_in(window, async move |this, cx| {
            let result =
                crate::chat_request::call(client, handle, Request::DesktopOpen { argv }).await;
            let _ = this.update_in(cx, |this, window, cx| {
                match result {
                    Ok(Reply::DesktopEnvironment(env)) => {
                        this.open_desktop_viewer(&env.env_id, window, cx)
                    }
                    Ok(_) => this.report_error("Unexpected desktop reply".into(), cx),
                    Err(error) => this.report_error(error.to_string(), cx),
                }
                cx.notify();
            });
        })
        .detach();
    }

    pub fn close_modal(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.active_modal == ActiveModal::AuthModal {
            self.auth_modal.update(cx, |modal, cx| modal.dismiss(cx));
        }
        if self.active_modal == ActiveModal::DesktopViewer {
            self.desktop_viewer = None;
        }
        self.active_modal = ActiveModal::None;
        self.focus_input(window, cx);
        cx.notify();
    }

    pub fn toggle_sidebar(&mut self, cx: &mut Context<Self>) {
        self.show_sidebar = !self.show_sidebar;
        cx.notify();
    }

    /// Opens focus mode for thread `id`: a focus thread, or a worker card's child.
    fn enter_focus(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some(summary) = self
            .subagents
            .iter()
            .find(|summary| summary.id == id)
            .cloned()
        else {
            return;
        };
        let client = self.client.clone();
        let tokio_handle = self.tokio_handle.clone();
        let focus = cx.new(|cx| FocusView::new(summary, client, tokio_handle, cx));
        let subscription = cx.subscribe(&focus, |this, _, _: &FocusEvent, cx| {
            this.exit_focus(cx);
        });
        self.focus = Some(focus);
        self._focus_subscription = Some(subscription);
        cx.notify();
    }

    /// Leaves focus mode and drops the focus view, including its VCS panel.
    fn exit_focus(&mut self, cx: &mut Context<Self>) {
        self.focus = None;
        self._focus_subscription = None;
        self.sidebar
            .update(cx, |sidebar, cx| sidebar.clear_selection(cx));
        cx.notify();
    }

    pub fn sync_subagents(&mut self, cx: &mut Context<Self>) {
        let client = self.client.clone();
        let handle = self.tokio_handle.clone();
        cx.spawn(async move |this, cx| {
            let result = crate::chat_request::call(client, handle, Request::ListSubagents).await;
            let _ = this.update(cx, |this, cx| match result {
                Ok(Reply::Subagents(summaries)) => {
                    this.sidebar.update(cx, |sidebar, cx| {
                        sidebar.sync_from_summaries(&summaries, cx)
                    });
                    for summary in &summaries {
                        subagent_card::upsert_subagent_card(
                            &mut this.messages,
                            &summary.id,
                            summary.name.as_deref(),
                            &summary.status,
                            this.is_running,
                        );
                    }
                    this.subagents = summaries;
                    cx.notify();
                }
                Ok(_) => this.report_error("Unexpected subagent reply".into(), cx),
                Err(error) => this.report_error(error.to_string(), cx),
            });
        })
        .detach();
    }

    fn subagent_command(&mut self, id: String, command: SubagentCommand, cx: &mut Context<Self>) {
        self.send_request(Request::SubagentCommand { id, command }, cx);
    }

    fn send_request(&mut self, request: Request, cx: &mut Context<Self>) {
        let client = self.client.clone();
        let handle = self.tokio_handle.clone();
        cx.spawn(async move |this, cx| {
            let result = crate::chat_request::call(client, handle, request).await;
            let _ = this.update(cx, |this, cx| {
                if let Err(error) = result {
                    this.report_error(error.to_string(), cx);
                }
            });
        })
        .detach();
    }

    fn report_error(&mut self, error: String, cx: &mut Context<Self>) {
        self.artifact_notice = Some(error.into());
        cx.notify();
    }

    fn apply_snapshot(&mut self, view: SessionView, cx: &mut Context<Self>) {
        self.connected = true;
        self.artifact_notice = None;
        self.is_running = view.info.run.is_some();
        if let Some(model) = view.info.model {
            self.active_model = model.into();
        }
        self.committed = view.committed;
        self.messages = load_messages(&view.entries);
        self.streaming_row = None;
        self.pending_approvals.clear();
        for approval in view.pending_approvals {
            self.handle_approval_request(approval, cx);
        }
        cx.notify();
    }

    fn reattach(&mut self, cx: &mut Context<Self>) {
        self.connected = false;
        let client = self.client.clone();
        let handle = self.tokio_handle.clone();
        let session_id = self.session_id.clone();
        cx.spawn(async move |this, cx| {
            let result =
                crate::chat_request::call(client, handle, Request::OpenSession { session_id })
                    .await;
            let _ = this.update(cx, |this, cx| match result {
                Ok(Reply::Session(view)) => this.apply_snapshot(view, cx),
                Ok(_) => this.report_error("Unexpected session reply".into(), cx),
                Err(error) => this.report_error(error.to_string(), cx),
            });
        })
        .detach();
    }

    fn handle_server_event(
        &mut self,
        event: ServerEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            ServerEvent::Session { session_id, event } if session_id == self.session_id => {
                match event {
                    SessionEvent::RunStarted { .. } => {
                        self.is_running = true;
                    }
                    SessionEvent::Agent { event } => self.handle_agent_event(event, cx),
                    SessionEvent::MessageCommitted { index, message } => {
                        if index < self.committed {
                            return;
                        }
                        if index != self.committed {
                            self.reattach(cx);
                            return;
                        }
                        self.committed = index + 1;
                        if matches!(&message, AiMessage::Assistant(_))
                            && let Some(row) = self.streaming_row.take()
                        {
                            self.messages.remove(row);
                        }
                        self.messages
                            .extend(view_messages(std::iter::once(&message)));
                        self.sync_subagents(cx);
                    }
                    SessionEvent::ApprovalRequested { approval } => {
                        self.handle_approval_request(approval, cx)
                    }
                    SessionEvent::ApprovalResolved { approval_id } => {
                        self.pending_approvals.remove(&approval_id);
                        self.set_approval_state(approval_id, ApprovalState::Cancelled);
                    }
                    SessionEvent::DecisionBlocked { decision } => {
                        self.report_error(
                            format!(
                                "Blocked: {} needs an operator decision ({}). The run has stopped.",
                                decision.capability, decision.id
                            ),
                            cx,
                        );
                    }
                    SessionEvent::DecisionResolved { .. } => {}
                    SessionEvent::RunFinished {
                        stop_reason, error, ..
                    } => {
                        self.is_running = false;
                        self.cancel_pending_approvals();
                        if let Some(error) = error {
                            self.report_error(error, cx);
                        } else if stop_reason == Some(StopReason::Aborted) {
                            self.report_error("Run cancelled".into(), cx);
                        }
                        self.focus_input(window, cx);
                        self.sync_subagents(cx);
                    }
                }
            }
            ServerEvent::Subagent { .. } => self.sync_subagents(cx),
            _ => {}
        }
        self.scroll.scroll_to_bottom();
        cx.notify();
    }
    /// Whether a new run may start, ignoring what the composer holds.
    fn can_submit(&self, cx: &App) -> bool {
        self.connected && !self.is_running && !self.input.read(cx).is_composing()
    }

    fn can_send(&self, cx: &App) -> bool {
        let input = self.input.read(cx);
        self.connected
            && !input.is_composing()
            && (input.has_text() || !self.attachments.is_empty())
    }

    /// Actions for rendered views. A `Button` submits a turn only while one can start.
    fn view_actions(&self, cx: &Context<Self>) -> ViewActions {
        if !self.can_submit(cx) {
            return ViewActions::none();
        }
        let chat = cx.weak_entity();
        ViewActions::submit(move |prompt, window, cx| {
            let _ = chat.update(cx, |this, cx| this.submit_prompt(prompt, window, cx));
        })
    }

    fn focus_input(&self, window: &mut Window, cx: &App) {
        self.input.read(cx).focus_handle(cx).focus(window);
    }

    fn send_message(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.can_send(cx) {
            return;
        }
        let text = self.input.update(cx, |input, cx| input.take_text(cx));
        let attachments = std::mem::take(&mut self.attachments);
        let prompt = format_prompt(&text, &attachments);
        if self.is_running {
            self.steer(prompt, cx);
            self.focus_input(window, cx);
        } else {
            self.submit_turn(text, attachments, prompt, window, cx);
        }
    }

    /// Submits a `Button` click: the prompt is shown and sent unchanged.
    fn submit_prompt(&mut self, prompt: String, window: &mut Window, cx: &mut Context<Self>) {
        if !self.can_submit(cx) {
            return;
        }
        let text = SharedString::from(prompt.clone());
        self.submit_turn(text, Vec::new(), prompt, window, cx);
    }

    /// Records a user turn, then starts a run for `prompt`. `text` and
    /// `attachments` are what the transcript shows; `prompt` is what the model gets.
    fn submit_turn(
        &mut self,
        _text: SharedString,
        _attachments: Vec<Attachment>,
        prompt: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.attachment_error = None;
        self.artifact_notice = None;
        self.scroll.scroll_to_bottom();
        self.focus_input(window, cx);
        self.start_run(prompt, window, cx);
    }

    /// Requests a foreground run. Only committed daemon events create transcript rows.
    fn start_run(&mut self, prompt: String, window: &mut Window, cx: &mut Context<Self>) {
        self.is_running = true;
        let client = self.client.clone();
        let handle = self.tokio_handle.clone();
        let request = Request::Prompt {
            session_id: self.session_id.clone(),
            text: prompt,
            unattended: false,
        };
        cx.spawn_in(window, async move |this, cx| {
            let result = crate::chat_request::call(client, handle, request).await;
            let _ = this.update_in(cx, |this, _, cx| {
                match result {
                    Ok(Reply::Started { .. }) => {}
                    Ok(_) => {
                        this.is_running = false;
                        this.report_error("Unexpected prompt reply".into(), cx);
                    }
                    Err(error) => {
                        this.is_running = false;
                        this.report_error(error.to_string(), cx);
                    }
                }
                cx.notify();
            });
        })
        .detach();
        cx.notify();
    }

    pub fn steer(&mut self, text: String, cx: &mut Context<Self>) {
        self.send_request(
            Request::Steer {
                session_id: self.session_id.clone(),
                text,
            },
            cx,
        );
    }

    pub fn cancel(&mut self, cx: &mut Context<Self>) {
        self.send_request(
            Request::Cancel {
                session_id: self.session_id.clone(),
            },
            cx,
        );
    }

    fn handle_agent_event(&mut self, event: AgentEvent, cx: &mut Context<Self>) {
        match event {
            AgentEvent::MessageStart => {
                self.streaming_row = Some(self.messages.len());
                self.messages.push(Message {
                    role: MessageRole::Assistant,
                    text: SharedString::default(),
                    attachments: Vec::new(),
                    desktop_artifacts: Vec::new(),
                    approval: None,
                    views: Vec::new(),
                });
                self.scroll.scroll_to_bottom();
                cx.notify();
            }
            AgentEvent::MessageUpdate {
                event: ProviderEvent::TextDelta { delta, .. },
            } => {
                if let Some(last) = self.messages.last_mut()
                    && last.role == MessageRole::Assistant
                {
                    let mut text = last.text.to_string();
                    text.push_str(&delta);
                    last.text = text.into();
                    self.scroll.scroll_to_bottom();
                    cx.notify();
                }
            }
            AgentEvent::Error { error } => {
                let error_text = format!("Error: {error}");
                if let Some(last) = self.messages.last_mut() {
                    if last.role == MessageRole::Assistant && last.text.is_empty() {
                        last.text = error_text.into();
                    } else if last.role == MessageRole::Assistant {
                        let mut text = last.text.to_string();
                        text.push_str(&format!("\n\n{error_text}"));
                        last.text = text.into();
                    } else {
                        self.messages.push(Message {
                            role: MessageRole::Assistant,
                            text: error_text.into(),
                            attachments: Vec::new(),
                            desktop_artifacts: Vec::new(),
                            approval: None,
                            views: Vec::new(),
                        });
                    }
                } else {
                    self.messages.push(Message {
                        role: MessageRole::Assistant,
                        text: error_text.into(),
                        attachments: Vec::new(),
                        desktop_artifacts: Vec::new(),
                        approval: None,
                        views: Vec::new(),
                    });
                }
                self.scroll.scroll_to_bottom();
                cx.notify();
            }
            AgentEvent::ToolExecutionStart { .. } => {
                self.sync_subagents(cx);
            }
            // Tool results enter the transcript only through MessageCommitted.
            AgentEvent::ToolExecutionEnd { .. } => self.sync_subagents(cx),
            _ => {}
        }
    }

    fn handle_approval_request(&mut self, request: PendingApproval, cx: &mut Context<Self>) {
        let id = request.approval_id;
        if self.pending_approvals.contains_key(&id) {
            return;
        }
        let call = &request.call;
        let card = ApprovalCard {
            id,
            tool: call.name.clone().into(),
            tier: request.tier,
            detail: approval_detail(call).into(),
            cwd: approval_cwd(call).map(|cwd| SharedString::from(cwd.to_owned())),
            state: ApprovalState::Pending,
        };
        self.pending_approvals.insert(id, request);
        self.messages.push(Message {
            role: MessageRole::Assistant,
            text: SharedString::default(),
            attachments: Vec::new(),
            desktop_artifacts: Vec::new(),
            approval: Some(card),
            views: Vec::new(),
        });
        self.scroll.scroll_to_bottom();
        cx.notify();
    }

    fn resolve_approval(&mut self, id: u64, allow: bool, cx: &mut Context<Self>) {
        if !self.pending_approvals.contains_key(&id) {
            return;
        }
        let client = self.client.clone();
        let handle = self.tokio_handle.clone();
        cx.spawn(async move |this, cx| {
            let result = crate::chat_request::call(
                client,
                handle,
                Request::ApprovalRespond {
                    approval_id: id,
                    allow,
                },
            )
            .await;
            let _ = this.update(cx, |this, cx| {
                match result {
                    Ok(Reply::Ok) => {
                        this.pending_approvals.remove(&id);
                        this.set_approval_state(
                            id,
                            if allow {
                                ApprovalState::Allowed
                            } else {
                                ApprovalState::Denied
                            },
                        );
                    }
                    Ok(_) => this.report_error("Unexpected approval reply".into(), cx),
                    Err(error) => this.report_error(error.to_string(), cx),
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn set_approval_state(&mut self, id: u64, state: ApprovalState) {
        for card in self
            .messages
            .iter_mut()
            .filter_map(|message| message.approval.as_mut())
        {
            if card.id == id {
                card.state = state;
            }
        }
    }

    /// Clears the display state; the daemon owns approval responder lifetimes.
    fn cancel_pending_approvals(&mut self) {
        self.pending_approvals.clear();
        for card in self
            .messages
            .iter_mut()
            .filter_map(|message| message.approval.as_mut())
        {
            if card.state == ApprovalState::Pending {
                card.state = ApprovalState::Cancelled;
            }
        }
    }

    fn close_attachment_menu(
        &mut self,
        _: &CloseAttachmentMenu,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.attachment_menu_open {
            self.attachment_menu_open = false;
            cx.notify();
        } else {
            cx.propagate();
        }
    }

    fn open_picker(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.attachment_menu_open = false;
        cx.notify();
        if self.picker.is_some() {
            return;
        }
        self.attachment_error = None;
        let paths = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: Some("Attach".into()),
        });
        self.picker = Some(cx.spawn_in(window, async move |this, cx| {
            let result = paths.await;
            this.update_in(cx, |this, window, cx| {
                this.picker = None;
                match result {
                    Ok(Ok(Some(paths))) => add_attachments(&mut this.attachments, paths),
                    Ok(Ok(None)) => {}
                    Ok(Err(error)) => {
                        this.attachment_error =
                            Some(format!("Couldn't open the file picker: {error}").into());
                    }
                    Err(_) => {
                        this.attachment_error =
                            Some("The file picker closed without a response.".into());
                    }
                }
                this.focus_input(window, cx);
                cx.notify();
            })
            .ok();
        }));
    }

    fn header(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .items_center()
            .justify_between()
            .flex_none()
            .h(px(theme::HEADER_HEIGHT))
            .px_6()
            .border_b_1()
            .border_color(rgb(theme::BORDER))
            .child(div().when(!self.show_sidebar, |title| title.child(app_title())))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        header_button("header-model-btn", false)
                            .on_click(cx.listener(|this, _: &ClickEvent, window, cx| {
                                if this.active_modal == ActiveModal::ModelPicker {
                                    this.close_modal(window, cx);
                                } else {
                                    this.open_model_picker(window, cx);
                                }
                            }))
                            .child(format!("🤖 {} ▾", self.active_model)),
                    )
                    .child(
                        header_button("header-auth-btn", false)
                            .on_click(cx.listener(|this, _: &ClickEvent, window, cx| {
                                if this.active_modal == ActiveModal::AuthModal {
                                    this.close_modal(window, cx);
                                } else {
                                    this.open_auth_modal(window, cx);
                                }
                            }))
                            .child("Providers"),
                    )
                    .child(
                        header_button("header-sidebar-btn", self.show_sidebar)
                            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                this.toggle_sidebar(cx);
                            }))
                            .child("Threads"),
                    ),
            )
    }

    /// Full-height left column: app title aligned with the chat header, then
    /// the focus-thread list.
    fn sidebar_column(&self) -> impl IntoElement {
        div()
            .w(px(300.))
            .flex_none()
            .h_full()
            .flex()
            .flex_col()
            .border_r_1()
            .border_color(rgb(theme::BORDER))
            .child(
                div()
                    .flex()
                    .items_center()
                    .flex_none()
                    .h(px(theme::HEADER_HEIGHT))
                    .px_5()
                    .border_b_1()
                    .border_color(rgb(theme::BORDER))
                    .child(app_title()),
            )
            .child(div().flex_1().min_h_0().child(self.sidebar.clone()))
    }

    /// The main transcript and composer.
    fn main_body(&self, window: &Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .flex_1()
            .min_h_0()
            .w_full()
            .child(self.message_list(cx))
            .child(self.composer(window, cx))
    }

    fn message_list(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let messages = self.messages.clone();
        let actions = self.view_actions(cx);
        div()
            .id("messages")
            .flex_1()
            .min_h_0()
            .w_full()
            .min_w_0()
            .overflow_y_scroll()
            .track_scroll(&self.scroll)
            .px_6()
            .py_6()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_4()
                    .w_full()
                    .min_w_0()
                    .max_w(px(800.))
                    .mx_auto()
                    .when(messages.is_empty(), |view| {
                        view.child(
                            div()
                                .py_12()
                                .flex()
                                .flex_col()
                                .items_center()
                                .gap_3()
                                .child(div().text_xl().child("Start a conversation"))
                                .child(
                                    div()
                                        .text_sm()
                                        .text_color(rgb(0xa0a0a0))
                                        .child("Type a message below to try the chat view."),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(rgb(0x808080))
                                        .child("Agent runtime connected."),
                                ),
                        )
                    })
                    .children(messages.iter().enumerate().map(|(index, message)| {
                        let author = match message.role {
                            MessageRole::User => "You",
                            MessageRole::Assistant => "Assistant",
                        };
                        let artifacts = message.desktop_artifacts.clone();
                        let views = message.views.clone();
                        div()
                            .id(("message", index))
                            .flex_none()
                            .w_full()
                            .min_w_0()
                            .child(
                                div()
                                    .mb_2()
                                    .text_xs()
                                    .text_color(rgb(0xa0a0a0))
                                    .child(author),
                            )
                            .when(!message.text.is_empty(), |row| {
                                row.child(
                                    div()
                                        .w_full()
                                        .text_size(px(15.))
                                        .child(message.text.clone()),
                                )
                            })
                            .when_some(message.approval.clone(), |row, card| {
                                row.child(self.approval_card(&card, cx))
                            })
                            .when(!message.attachments.is_empty(), |row| {
                                row.child(div().mt_2().flex().flex_wrap().gap_2().children(
                                    message.attachments.iter().map(|attachment| {
                                        attachment_chip(attachment.label.clone()).pr_2()
                                    }),
                                ))
                            })
                            .when(!artifacts.is_empty(), |row| {
                                row.child(
                                    div().mt_3().flex().flex_col().gap_2().children(
                                        artifacts.into_iter().enumerate().map(
                                            |(art_idx, artifact)| {
                                                let card_artifact = artifact.clone();
                                                let click_artifact = artifact.clone();
                                                let btn = div()
                                                    .id((
                                                        "open-desktop-btn",
                                                        (index * 1000 + art_idx) as u32,
                                                    ))
                                                    .px_2p5()
                                                    .py_1()
                                                    .rounded_md()
                                                    .bg(rgb(0x2563eb))
                                                    .hover(|s| s.bg(rgb(0x1d4ed8)))
                                                    .text_xs()
                                                    .font_weight(gpui::FontWeight::MEDIUM)
                                                    .text_color(rgb(0xffffff))
                                                    .cursor_pointer()
                                                    .on_click(cx.listener(
                                                        move |this, _: &ClickEvent, window, cx| {
                                                            this.open_desktop_artifact(
                                                                &click_artifact,
                                                                window,
                                                                cx,
                                                            );
                                                        },
                                                    ))
                                                    .child("Open Desktop ↗");
                                                desktop_artifact_card(&card_artifact, btn)
                                            },
                                        ),
                                    ),
                                )
                            })
                            .when(!views.is_empty(), |row| {
                                row.child(div().mt_2().flex().flex_col().gap_2().children(
                                    views.iter().map(|view| card_element(view, &actions, cx)),
                                ))
                            })
                    })),
            )
    }

    fn approval_card(&self, card: &ApprovalCard, cx: &Context<Self>) -> impl IntoElement {
        let id = card.id;
        let tier = match card.tier {
            ToolTier::Read => "read",
            ToolTier::Write => "write",
            ToolTier::Exec => "exec",
        };
        let footer = match card.state {
            ApprovalState::Pending => div()
                .flex()
                .gap_2()
                .child(
                    div()
                        .id(("approval-allow", id as u32))
                        .px_3()
                        .py_1()
                        .rounded_md()
                        .bg(rgb(0x16a34a))
                        .hover(|s| s.bg(rgb(0x15803d)))
                        .text_xs()
                        .font_weight(gpui::FontWeight::MEDIUM)
                        .text_color(rgb(0xffffff))
                        .cursor_pointer()
                        .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                            this.resolve_approval(id, true, cx);
                        }))
                        .child("Allow"),
                )
                .child(
                    div()
                        .id(("approval-deny", id as u32))
                        .px_3()
                        .py_1()
                        .rounded_md()
                        .bg(rgb(0xdc2626))
                        .hover(|s| s.bg(rgb(0xb91c1c)))
                        .text_xs()
                        .font_weight(gpui::FontWeight::MEDIUM)
                        .text_color(rgb(0xffffff))
                        .cursor_pointer()
                        .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                            this.resolve_approval(id, false, cx);
                        }))
                        .child("Deny"),
                ),
            ApprovalState::Allowed => div().text_xs().text_color(rgb(0x86efac)).child("Allowed"),
            ApprovalState::Denied => div().text_xs().text_color(rgb(0xfca5a5)).child("Denied"),
            ApprovalState::Cancelled => {
                div().text_xs().text_color(rgb(0xa0a0a0)).child("Cancelled")
            }
        };
        let detail_lines = card.detail.lines().map(|line| {
            let text = if line.is_empty() { " " } else { line };
            div().child(SharedString::from(text.to_owned()))
        });
        div()
            .w_full()
            .p_3()
            .rounded_lg()
            .border_1()
            .border_color(rgb(0xf59e0b))
            .bg(rgb(0x1e293b))
            .flex()
            .flex_col()
            .gap_2()
            .child(
                div()
                    .text_xs()
                    .font_weight(gpui::FontWeight::MEDIUM)
                    .text_color(rgb(0xfcd34d))
                    .child(SharedString::from(format!(
                        "Permission requested: {} ({tier})",
                        card.tool
                    ))),
            )
            .when_some(card.cwd.clone(), |card_div, cwd| {
                card_div.child(
                    div()
                        .text_xs()
                        .text_color(rgb(0xa0a0a0))
                        .child(SharedString::from(format!("cwd: {cwd}"))),
                )
            })
            .child(
                div()
                    .p_2()
                    .rounded_md()
                    .bg(rgb(0x0f172a))
                    .font_family("monospace")
                    .text_size(px(13.))
                    .text_color(rgb(0xe2e8f0))
                    .flex()
                    .flex_col()
                    .children(detail_lines),
            )
            .child(footer)
    }

    fn attachment_menu(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("attachment-menu")
            .occlude()
            .on_mouse_down_out(cx.listener(|this, _: &MouseDownEvent, _, cx| {
                this.attachment_menu_open = false;
                this.menu_just_dismissed = true;
                cx.notify();
            }))
            .w(px(240.))
            .p_1()
            .rounded_lg()
            .border_1()
            .border_color(rgb(0x3a3a3a))
            .bg(rgb(0x262626))
            .shadow_lg()
            .child(menu_item(
                "attach-files",
                "Files…",
                "Attach one or more files",
                cx.listener(|this, _: &ClickEvent, window, cx| this.open_picker(window, cx)),
            ))
    }

    fn composer(&self, window: &Window, cx: &mut Context<Self>) -> impl IntoElement {
        let can_send = self.can_send(cx);
        let focused = self.input.read(cx).focus_handle(cx).is_focused(window);
        let menu_open = self.attachment_menu_open;

        let attach_button = div()
            .id("attach")
            .flex()
            .items_center()
            .justify_center()
            .size_8()
            .rounded_full()
            .border_1()
            .border_color(rgb(0x444444))
            .text_size(px(20.))
            .text_color(rgb(0xcccccc))
            .cursor_pointer()
            .when(menu_open, |button| button.bg(rgb(0x333333)))
            .hover(|style| style.bg(rgb(0x2e2e2e)))
            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                if this.menu_just_dismissed {
                    this.menu_just_dismissed = false;
                    return;
                }
                this.attachment_menu_open = !this.attachment_menu_open;
                cx.notify();
            }))
            .child("+");

        let send_button = div()
            .id("send-message")
            .flex()
            .items_center()
            .justify_center()
            .size_8()
            .rounded_full()
            .text_size(px(17.))
            .bg(if can_send {
                rgb(0xd4d4d4)
            } else {
                rgb(0x333333)
            })
            .text_color(if can_send {
                rgb(0x181818)
            } else {
                rgb(0x707070)
            })
            .when(can_send, |button| {
                button
                    .cursor_pointer()
                    .hover(|style| style.bg(rgb(0xeeeeee)))
            })
            .on_click(cx.listener(|this, _: &ClickEvent, window, cx| this.send_message(window, cx)))
            .child("↑");

        div().flex_none().px_6().pt_2().pb_4().child(
            div()
                .flex()
                .flex_col()
                .gap_2()
                .w_full()
                .max_w(px(800.))
                .mx_auto()
                .when_some(self.attachment_error.clone(), |column, error| {
                    column.child(div().text_xs().text_color(rgb(0xb0b0b0)).child(error))
                })
                .when_some(self.artifact_notice.clone(), |column, notice| {
                    column.child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .px_3()
                            .py_2()
                            .rounded_md()
                            .border_1()
                            .border_color(rgb(0x3b82f6))
                            .bg(rgb(0x1e293b))
                            .text_xs()
                            .text_color(rgb(0x93c5fd))
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_1p5()
                                    .child("ℹ️")
                                    .child(notice),
                            )
                            .child(
                                div()
                                    .id("dismiss-artifact-notice")
                                    .px_1p5()
                                    .py_0p5()
                                    .rounded_sm()
                                    .cursor_pointer()
                                    .hover(|s| s.bg(rgb(0x334155)))
                                    .text_color(rgb(0xe2e8f0))
                                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                        this.artifact_notice = None;
                                        cx.notify();
                                    }))
                                    .child("×"),
                            ),
                    )
                })
                .child(
                    div()
                        .id("composer")
                        .flex()
                        .flex_col()
                        .gap_2()
                        .p_2()
                        .rounded_xl()
                        .border_1()
                        .border_color(if focused {
                            rgb(0x6a6a6a)
                        } else {
                            rgb(0x3a3a3a)
                        })
                        .bg(rgb(0x222222))
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(|this, _: &MouseDownEvent, window, cx| {
                                this.focus_input(window, cx)
                            }),
                        )
                        .when(!self.attachments.is_empty(), |card| {
                            card.child(
                                div().flex().flex_wrap().gap_2().px_1().pt_1().children(
                                    self.attachments.iter().enumerate().map(
                                        |(index, attachment)| {
                                            attachment_chip(attachment.label.clone()).child(
                                                div()
                                                    .id(("remove-attachment", index))
                                                    .flex_none()
                                                    .px_1()
                                                    .rounded_sm()
                                                    .text_color(rgb(0x909090))
                                                    .cursor_pointer()
                                                    .hover(|style| {
                                                        style
                                                            .bg(rgb(0x333333))
                                                            .text_color(rgb(0xe7e7e7))
                                                    })
                                                    .on_mouse_down(
                                                        MouseButton::Left,
                                                        cx.listener(
                                                            move |this, _: &MouseDownEvent, window, cx| {
                                                                cx.stop_propagation();
                                                                if index < this.attachments.len() {
                                                                    this.attachments.remove(index);
                                                                    this.focus_input(window, cx);
                                                                    cx.notify();
                                                                }
                                                            },
                                                        ),
                                                    )
                                                    .child("×"),
                                            )
                                        },
                                    ),
                                ),
                            )
                        })
                        .child(self.input.clone())
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .child(div().relative().child(attach_button).when(
                                    menu_open,
                                    |anchor| {
                                        anchor.child(
                                            deferred(
                                                div()
                                                    .absolute()
                                                    .bottom(px(40.))
                                                    .left_0()
                                                    .child(self.attachment_menu(cx)),
                                            )
                                            .with_priority(1),
                                        )
                                    },
                                ))
                                .when(self.is_running, |row| row.child(
                                    header_button("cancel-run", false)
                                        .on_click(cx.listener(|this, _: &ClickEvent, _, cx| this.cancel(cx)))
                                        .child("Stop")
                                ))
                                .child(send_button),
                        ),
                )
                .child(
                    div()
                        .flex()
                        .justify_end()
                        .text_xs()
                        .text_color(rgb(0x808080))
                        .child("Enter to send"),
                ),
        )
    }

    /// The active modal's content, if any. Rendered by `modal_overlay` as a
    /// deferred layer so the chat view stays visible behind it.
    fn modal_content(&self) -> Option<gpui::AnyView> {
        match self.active_modal {
            ActiveModal::ModelPicker => Some(self.model_picker.clone().into()),
            ActiveModal::AuthModal => Some(self.auth_modal.clone().into()),
            ActiveModal::DesktopViewer => self.desktop_viewer.clone().map(|viewer| viewer.into()),
            ActiveModal::None => None,
        }
    }

    /// Translucent scrim covering the chat view (gpui has no backdrop blur);
    /// clicks outside the centered panel close the modal.
    fn modal_overlay(&self, content: gpui::AnyView, cx: &mut Context<Self>) -> impl IntoElement {
        deferred(
            div()
                .id("modal-scrim")
                .absolute()
                .inset_0()
                .occlude()
                .flex()
                .items_center()
                .justify_center()
                .bg(rgba(0x00000080))
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(|this, _: &MouseDownEvent, window, cx| {
                        this.close_modal(window, cx);
                    }),
                )
                .child(
                    div()
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(|_, _: &MouseDownEvent, _, cx| {
                                cx.stop_propagation();
                            }),
                        )
                        .child(content),
                ),
        )
        .with_priority(10)
    }
}

fn attachment_chip(label: SharedString) -> Div {
    div()
        .flex()
        .items_center()
        .gap_1()
        .max_w(px(260.))
        .pl_2()
        .pr_1()
        .py_1()
        .rounded_md()
        .border_1()
        .border_color(rgb(0x3a3a3a))
        .bg(rgb(0x2a2a2a))
        .text_sm()
        .child(
            div()
                .min_w_0()
                .overflow_hidden()
                .text_ellipsis()
                .whitespace_nowrap()
                .child(label),
        )
}

fn menu_item(
    id: &'static str,
    title: &'static str,
    detail: &'static str,
    on_click: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> impl IntoElement {
    div()
        .id(id)
        .flex()
        .flex_col()
        .gap_0p5()
        .px_3()
        .py_2()
        .rounded_md()
        .cursor_pointer()
        .hover(|style| style.bg(rgb(0x333333)))
        .on_click(on_click)
        .child(div().text_sm().child(title))
        .child(div().text_xs().text_color(rgb(0x909090)).child(detail))
}

fn desktop_artifact_card(artifact: &DesktopSessionArtifact, open_button: impl IntoElement) -> Div {
    div()
        .flex()
        .flex_col()
        .gap_2()
        .p_3()
        .max_w(px(460.))
        .rounded_lg()
        .border_1()
        .border_color(rgb(0x3b82f6))
        .bg(rgb(0x1e293b))
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .gap_2()
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap_1p5()
                        .text_sm()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .text_color(rgb(0xf1f5f9))
                        .child("🖥️")
                        .child(artifact.title.clone()),
                )
                .child(
                    div()
                        .px_2()
                        .py_0p5()
                        .rounded_sm()
                        .bg(rgb(0x0f172a))
                        .border_1()
                        .border_color(rgb(0x334155))
                        .text_xs()
                        .text_color(rgb(0x94a3b8))
                        .child(format!("v{}", artifact.version)),
                ),
        )
        .child(
            div()
                .flex()
                .flex_col()
                .gap_1()
                .text_xs()
                .text_color(rgb(0xcbd5e1))
                .child(div().child(format!("Artifact ID: {}", artifact.artifact_id)))
                .child(div().child(format!("Environment ID: {}", artifact.environment_id)))
                .child(div().child(format!("Host ID: {}", artifact.host_id)))
                .child(div().child(format!("Grant ID: {}", artifact.grant_id))),
        )
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .pt_1()
                .border_t_1()
                .border_color(rgb(0x334155))
                .child(
                    div()
                        .text_xs()
                        .text_color(rgb(0x94a3b8))
                        .child("Remote Sharing Session (IDs-only)"),
                )
                .child(open_button),
        )
}

fn app_title() -> impl IntoElement {
    div()
        .text_size(px(20.))
        .font_weight(gpui::FontWeight::MEDIUM)
        .child("Juto")
}

/// Outlined header control; `active` marks a toggled-on state with the accent.
fn header_button(id: &'static str, active: bool) -> Stateful<Div> {
    div()
        .id(id)
        .flex()
        .items_center()
        .gap_1p5()
        .px_3()
        .py_1p5()
        .rounded_md()
        .border_1()
        .border_color(rgb(theme::BORDER_STRONG))
        .when(active, |button| button.bg(rgb(theme::BG_RAISED)))
        .text_xs()
        .text_color(if active {
            rgb(theme::ACCENT)
        } else {
            rgb(theme::TEXT)
        })
        .cursor_pointer()
        .hover(|s| s.bg(rgb(theme::BG_HOVER)))
}

impl Render for ChatView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .relative()
            .flex()
            .size_full()
            .bg(rgb(theme::BG))
            .text_color(rgb(theme::TEXT))
            .key_context("ChatView")
            .on_action(|_: &Quit, _, cx| cx.quit())
            .on_action(cx.listener(Self::close_attachment_menu))
            .on_action(cx.listener(|this, _: &ToggleModelPicker, window, cx| {
                if this.active_modal == ActiveModal::ModelPicker {
                    this.close_modal(window, cx);
                } else {
                    this.open_model_picker(window, cx);
                }
            }))
            .on_action(cx.listener(|this, _: &ToggleAuthModal, window, cx| {
                if this.active_modal == ActiveModal::AuthModal {
                    this.close_modal(window, cx);
                } else {
                    this.open_auth_modal(window, cx);
                }
            }))
            .on_action(cx.listener(|this, _: &ToggleSidebar, _, cx| {
                this.toggle_sidebar(cx);
            }))
            .when(self.show_sidebar, |root| root.child(self.sidebar_column()))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .child(self.header(cx))
                    .child(match self.focus.clone() {
                        Some(focus) => focus.into_any_element(),
                        None => self.main_body(window, cx).into_any_element(),
                    }),
            )
            .when_some(self.modal_content(), |view, content| {
                view.child(self.modal_overlay(content, cx))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use juto_runtime::Session;
    #[test]
    fn bash_approval_shows_exact_command_and_cwd() {
        let call = ToolCall {
            id: "call-1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({
                "command": "rm -rf build\n  echo done",
                "cwd": "/tmp/work",
            }),
            thought_signature: None,
        };
        assert_eq!(approval_detail(&call), "rm -rf build\n  echo done");
        assert_eq!(approval_cwd(&call), Some("/tmp/work"));
    }

    #[test]
    fn non_bash_approval_shows_arguments_and_no_cwd() {
        let call = ToolCall {
            id: "call-2".into(),
            name: "write".into(),
            arguments: serde_json::json!({ "path": "a.txt", "cwd": "/tmp" }),
            thought_signature: None,
        };
        assert!(approval_detail(&call).contains("\"path\": \"a.txt\""));
        assert_eq!(approval_cwd(&call), None);
    }

    fn labels(attachments: &[Attachment]) -> Vec<&str> {
        attachments
            .iter()
            .map(|attachment| attachment.label.as_ref())
            .collect()
    }

    #[test]
    fn picking_an_already_attached_path_keeps_one_chip_in_original_order() {
        let mut attachments = Vec::new();
        add_attachments(
            &mut attachments,
            vec!["/work/notes.md".into(), "/work/plan.txt".into()],
        );
        add_attachments(
            &mut attachments,
            vec!["/work/plan.txt".into(), "/work/diagram.png".into()],
        );
        assert_eq!(
            labels(&attachments),
            ["notes.md", "plan.txt", "diagram.png"]
        );
    }

    #[test]
    fn prompt_formatting_combines_attachments_and_text() {
        let attachments = vec![
            Attachment::new("/tmp/alpha.rs".into()),
            Attachment::new("/tmp/beta.txt".into()),
        ];
        let formatted = format_prompt("Explain this code", &attachments);
        assert_eq!(
            formatted,
            "Attached files:\n- /tmp/alpha.rs\n- /tmp/beta.txt\n\nExplain this code"
        );

        let attachments_only = format_prompt("", &attachments);
        assert_eq!(
            attachments_only,
            "Attached files:\n- /tmp/alpha.rs\n- /tmp/beta.txt\n"
        );

        let text_only = format_prompt("Hello world", &[]);
        assert_eq!(text_only, "Hello world");
    }

    #[test]
    fn load_messages_extracts_user_and_assistant_history() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cwd = std::env::current_dir().unwrap();
        let mut session = Session::create(temp_dir.path(), &cwd).unwrap();
        session
            .append(EntryKind::Message {
                message: AiMessage::user("Hello from user"),
            })
            .unwrap();
        let assistant = juto_ai::AssistantMessage {
            content: vec![juto_ai::ContentBlock::Text {
                text: "Hello from assistant".into(),
            }],
            api: "test".into(),
            provider: "test".into(),
            model: "test-model".into(),
            usage: Default::default(),
            stop_reason: StopReason::Stop,
            timestamp: 0,
            response_id: None,
            error: None,
        };
        session
            .append(EntryKind::Message {
                message: AiMessage::Assistant(assistant),
            })
            .unwrap();

        let entries: Vec<juto_runtime::SessionEntry> = session
            .active_entries()
            .unwrap()
            .into_iter()
            .cloned()
            .collect();
        let messages = load_messages(&entries);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::User);
        assert_eq!(messages[0].text.as_ref(), "Hello from user");
        assert_eq!(messages[1].role, MessageRole::Assistant);
        assert_eq!(messages[1].text.as_ref(), "Hello from assistant");
    }

    #[test]
    fn load_messages_extracts_and_mounts_desktop_artifacts_from_tool_results() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cwd = std::env::current_dir().unwrap();
        let mut session = Session::create(temp_dir.path(), &cwd).unwrap();

        let artifact = DesktopSessionArtifact {
            version: 1,
            artifact_id: "desk-test-01".into(),
            title: "Simulated Game Desktop".into(),
            host_id: "host-west-01".into(),
            environment_id: "env-sandbox-42".into(),
            grant_id: "grant-view-99".into(),
        };

        let tool_result = AiMessage::ToolResult {
            tool_call_id: "call-1".into(),
            tool_name: "desktop_artifact".into(),
            content: vec![juto_ai::ContentBlock::Text {
                text: "Desktop sharing artifact mounted".into(),
            }],
            is_error: false,
            details: serde_json::json!({
                "type": "desktop_session_artifact",
                "artifact": artifact.clone()
            }),
            timestamp: 0,
        };

        session
            .append(EntryKind::Message {
                message: tool_result,
            })
            .unwrap();

        let entries: Vec<juto_runtime::SessionEntry> = session
            .active_entries()
            .unwrap()
            .into_iter()
            .cloned()
            .collect();
        let messages = load_messages(&entries);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, MessageRole::Assistant);
        assert_eq!(messages[0].desktop_artifacts.len(), 1);
        assert_eq!(messages[0].desktop_artifacts[0], artifact);
    }

    #[test]
    fn load_messages_places_view_result_in_its_own_row_after_text() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cwd = std::env::current_dir().unwrap();
        let mut session = Session::create(temp_dir.path(), &cwd).unwrap();

        let assistant = juto_ai::AssistantMessage {
            content: vec![juto_ai::ContentBlock::Text {
                text: "Spawned a worker".into(),
            }],
            api: "test".into(),
            provider: "test".into(),
            model: "test-model".into(),
            usage: Default::default(),
            stop_reason: StopReason::Stop,
            timestamp: 0,
            response_id: None,
            error: None,
        };
        session
            .append(EntryKind::Message {
                message: AiMessage::Assistant(assistant),
            })
            .unwrap();

        let view = juto_agent::ViewNode {
            tag: "Badge".into(),
            props: serde_json::json!({"label": "done"})
                .as_object()
                .unwrap()
                .clone(),
            children: Vec::new(),
        };
        let tool_result = AiMessage::ToolResult {
            tool_call_id: "call-2".into(),
            tool_name: "view_test".into(),
            content: vec![juto_ai::ContentBlock::Text { text: "ok".into() }],
            is_error: false,
            details: juto_agent::view_details(view.clone()),
            timestamp: 0,
        };
        session
            .append(EntryKind::Message {
                message: tool_result,
            })
            .unwrap();

        let entries: Vec<juto_runtime::SessionEntry> = session
            .active_entries()
            .unwrap()
            .into_iter()
            .cloned()
            .collect();
        let messages = load_messages(&entries);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].text.as_ref(), "Spawned a worker");
        assert!(messages[0].views.is_empty());
        assert!(messages[1].text.is_empty());
        assert_eq!(messages[1].views, vec![view]);
    }
}
