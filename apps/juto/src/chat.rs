//! Chat UI wired to the agent runtime and session persistence.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::Quit;
use crate::auth_modal::{AuthModal, AuthModalEvent};
use crate::chat_input::{ChatInput, SubmitMessage};
use crate::focus_view::{FocusEvent, FocusView};
use crate::model_picker::{ModelPicker, ModelPickerEvent};
use crate::sidebar::{Sidebar, SidebarEvent};
use gpui::{
    App, ClickEvent, Context, Div, Entity, Focusable, IntoElement, KeyBinding, MouseButton,
    MouseDownEvent, PathPromptOptions, Render, ScrollHandle, SharedString, Subscription, Task,
    Window, actions, deferred, div, prelude::*, px, rgb, rgba,
};
use juto_agent::{AgentEvent, RunControl, RunOutcome, ToolTier};
use juto_ai::{Message as AiMessage, ProviderEvent, StopReason, ToolCall};
use juto_runtime::{ApprovalRequest, DesktopSessionArtifact, EntryKind, Runtime, Session};
use tokio::sync::Mutex;

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

pub fn load_messages(session: &Session) -> Vec<Message> {
    let Ok(entries) = session.active_entries() else {
        return Vec::new();
    };
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
                    });
                }
            }
            AiMessage::ToolResult { .. } => {
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
                        });
                    }
                }
            }
            _ => {}
        }
    }
    messages
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: MessageRole,
    pub text: SharedString,
    pub attachments: Vec<Attachment>,
    pub desktop_artifacts: Vec<DesktopSessionArtifact>,
    pub approval: Option<ApprovalCard>,
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
    subagents: Arc<juto_runtime::subagents::SubagentManager>,
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
    runtime: Runtime,
    session: Arc<Mutex<Session>>,
    tokio_handle: tokio::runtime::Handle,
    is_running: bool,
    active_control: Option<RunControl>,
    run_task: Option<Task<()>>,
    artifact_notice: Option<SharedString>,
    desktop_viewer: Option<Entity<crate::desktop_viewer::DesktopViewer>>,
    approval_requests: Arc<Mutex<tokio::sync::mpsc::UnboundedReceiver<ApprovalRequest>>>,
    pending_approvals: HashMap<u64, ApprovalRequest>,
    next_approval_id: u64,
    _subscriptions: Vec<Subscription>,
}
impl Drop for ChatView {
    fn drop(&mut self) {
        if let Some(control) = self.active_control.take() {
            control.cancel();
        }
    }
}

impl ChatView {
    pub fn new(
        runtime: Runtime,
        session: Session,
        tokio_handle: tokio::runtime::Handle,
        approval_requests: tokio::sync::mpsc::UnboundedReceiver<ApprovalRequest>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let input = cx.new(ChatInput::new);
        input.read(cx).focus_handle(cx).focus(window);
        let initial_model = if runtime.config().model.is_empty() {
            "anthropic/claude-sonnet-4-5".to_string()
        } else {
            runtime.config().model.clone()
        };
        let model_picker = cx.new(|cx| {
            let mut picker =
                ModelPicker::new(&initial_model, &runtime.config(), runtime.catalog(), cx);
            picker.refresh_credentials(runtime.clone(), tokio_handle.clone(), cx);
            picker
        });
        let auth_modal = cx.new(|cx| AuthModal::new(runtime.clone(), tokio_handle.clone(), cx));
        let subagent_control = RunControl::new();
        let subagents = runtime.subagent_manager(&subagent_control);
        let sidebar = cx.new(Sidebar::new);
        let initial_summaries = subagents.list();
        sidebar.update(cx, |s, cx| {
            s.sync_from_summaries(&initial_summaries, cx);
        });
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
                        picker.refresh_credentials(
                            this.runtime.clone(),
                            this.tokio_handle.clone(),
                            cx,
                        );
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
                if let Err(err) = this.subagents.pause(id) {
                    this.sidebar.update(cx, |s, cx| {
                        s.set_error(Some(format!("Pause rejected: {err}").into()), cx);
                    });
                } else {
                    this.sync_subagents(cx);
                }
            }
            SidebarEvent::AbortTask(id) => {
                if let Err(err) = this.subagents.cancel(id) {
                    this.sidebar.update(cx, |s, cx| {
                        s.set_error(Some(format!("Abort rejected: {err}").into()), cx);
                    });
                } else {
                    this.sync_subagents(cx);
                }
            }
            SidebarEvent::ResumeTask { id, resolution } => {
                if let Err(err) = this.subagents.unblock(id, resolution) {
                    this.sidebar.update(cx, |s, cx| {
                        s.set_error(Some(format!("Unblock rejected: {err}").into()), cx);
                    });
                } else {
                    this.sync_subagents(cx);
                }
            }
            SidebarEvent::SteerTask { id, guidance } => {
                if let Err(err) = this.subagents.send(id, guidance.clone()) {
                    this.sidebar.update(cx, |s, cx| {
                        s.set_error(Some(format!("Steer rejected: {err}").into()), cx);
                    });
                } else {
                    this.sync_subagents(cx);
                }
            }
        });

        let messages = load_messages(&session);
        Self {
            input,
            model_picker,
            auth_modal,
            sidebar,
            focus: None,
            _focus_subscription: None,
            subagents,
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
            runtime,
            session: Arc::new(Mutex::new(session)),
            tokio_handle,
            is_running: false,
            active_control: None,
            run_task: None,
            approval_requests: Arc::new(Mutex::new(approval_requests)),
            pending_approvals: HashMap::new(),
            next_approval_id: 0,
            _subscriptions: vec![
                submitted,
                changed,
                model_picker_sub,
                auth_modal_sub,
                sidebar_sub,
            ],
        }
    }

    /// Persist `selector` to `role` (`modelRoles` in the global config layer,
    /// `model` for `default`) and update the picker. Assigning `default` also
    /// switches the session model and closes the picker, matching the OMP
    /// model hub; other roles keep the picker open so several roles can be
    /// assigned in one pass.
    fn assign_role(
        &mut self,
        role: String,
        selector: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Err(error) = self.runtime.set_model_role(&role, &selector) {
            self.model_picker.update(cx, |picker, cx| {
                picker.set_error(Some(format!("Could not assign @{role}: {error}")), cx);
            });
            return;
        }
        if role == "default" {
            self.set_model(selector, window, cx);
        } else {
            self.model_picker.update(cx, |picker, cx| {
                picker.set_error(None, cx);
                picker.apply_assignment(&role, &selector, cx);
            });
            cx.notify();
        }
    }

    pub fn set_model(&mut self, selector: String, window: &mut Window, cx: &mut Context<Self>) {
        self.active_model = selector.clone().into();
        self.model_picker.update(cx, |picker, cx| {
            picker.set_error(None, cx);
            picker.apply_assignment("default", &selector, cx);
        });
        let runtime = self.runtime.clone();
        let session = self.session.clone();
        let tokio_handle = self.tokio_handle.clone();
        let sel = selector.clone();
        tokio_handle.spawn(async move {
            let mut session_guard = session.lock().await;
            let _ = runtime.switch_model(&mut session_guard, &sel);
        });
        self.close_modal(window, cx);
    }

    pub fn open_model_picker(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.active_modal == ActiveModal::AuthModal {
            self.auth_modal.update(cx, |modal, cx| modal.dismiss(cx));
        }
        self.active_modal = ActiveModal::ModelPicker;
        self.model_picker.update(cx, |picker, cx| {
            picker.refresh_credentials(self.runtime.clone(), self.tokio_handle.clone(), cx);
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
        let resolver = self.runtime.desktop_artifact_resolver();
        // Resolution is bound to this session's authenticated principal, not
        // to whoever issued the grant: a grant issued to another principal
        // must be rejected, and one issued to this session must be honored.
        match resolver.resolve(crate::desktop_viewer::SESSION_PRINCIPAL, artifact) {
            Ok(juto_runtime::ViewerRoute::Local { env_id, .. }) => {
                self.open_desktop_viewer(&env_id, window, cx);
            }
            Ok(juto_runtime::ViewerRoute::Remote { url, .. }) => {
                self.artifact_notice = Some(
                    format!(
                        "Desktop '{}' is available remotely at {url} (one-time token, 60s). \
                         External-browser play is not inline GPUI interaction.",
                        artifact.title
                    )
                    .into(),
                );
                cx.notify();
            }
            Err(err) => {
                self.artifact_notice =
                    Some(format!("Desktop '{}' cannot be opened: {err}", artifact.title).into());
                cx.notify();
            }
        }
    }

    /// Opens the embedded viewer for an existing desktop environment.
    pub fn open_desktop_viewer(
        &mut self,
        env_id: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let supervisor = self.runtime.desktop_supervisor();
        match supervisor.get_environment(env_id) {
            Ok(env) => {
                let title = format!("Desktop {env_id}");
                let viewer =
                    cx.new(|cx| crate::desktop_viewer::DesktopViewer::new(env, title, window, cx));
                self.desktop_viewer = Some(viewer);
                self.active_modal = ActiveModal::DesktopViewer;
                self.artifact_notice = None;
            }
            Err(err) => {
                self.artifact_notice =
                    Some(format!("Desktop environment '{env_id}' is unavailable: {err}").into());
            }
        }
        cx.notify();
    }

    /// Creates a desktop environment, launches `argv` into it, and opens the
    /// embedded viewer. Used by the `--desktop` developer mode.
    pub fn open_desktop_command(
        &mut self,
        argv: Vec<String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let supervisor = self.runtime.desktop_supervisor();
        let env_id = format!("desk-{}", uuid::Uuid::now_v7());
        let config = juto_runtime::EnvironmentConfig {
            env_id: env_id.clone(),
            owner_principal: crate::desktop_viewer::SESSION_PRINCIPAL.to_string(),
            output: juto_runtime::OutputConfig {
                width: 1280,
                height: 720,
                scale: 1,
                fps: 30,
            },
            render_profile: juto_runtime::RenderProfile::Software,
            limits: juto_runtime::SystemLimits::default(),
            workspace_dir: self.runtime.cwd().to_path_buf(),
            lease_ttl_secs: 6,
            compositor_mode: juto_runtime::CompositorMode::Worker,
            owner_heartbeat_timeout_secs: 60,
        };
        let request_id = uuid::Uuid::now_v7().to_string();
        let env = match supervisor.create_environment(config, &request_id) {
            Ok(env) => env,
            Err(err) => {
                self.artifact_notice = Some(format!("cannot create desktop: {err}").into());
                cx.notify();
                return;
            }
        };
        if let Err(err) = env.launch("desktop-app", &argv, None, &[], &request_id) {
            self.artifact_notice = Some(format!("cannot launch desktop application: {err}").into());
            env.destroy();
            cx.notify();
            return;
        }
        self.open_desktop_viewer(&env_id, window, cx);
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

    /// Opens focus mode for the focus thread `id`. Ids that are not focus threads are ignored.
    fn enter_focus(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some(summary) = self
            .subagents
            .list()
            .into_iter()
            .find(|summary| summary.focus && summary.id == id)
        else {
            return;
        };
        let updates = match self.subagents.subscribe(id) {
            Ok(updates) => updates,
            Err(err) => {
                self.sidebar.update(cx, |s, cx| {
                    s.set_error(Some(format!("Focus rejected: {err}").into()), cx);
                });
                return;
            }
        };
        let subagents = Arc::clone(&self.subagents);
        let tokio_handle = self.tokio_handle.clone();
        let focus = cx.new(|cx| FocusView::new(summary, updates, subagents, tokio_handle, cx));
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
        cx.notify();
    }

    pub fn sync_subagents(&mut self, cx: &mut Context<Self>) {
        let summaries = self.subagents.list();
        self.sidebar.update(cx, |sidebar, cx| {
            sidebar.sync_from_summaries(&summaries, cx);
        });
    }

    #[allow(dead_code)]
    pub fn subagents(&self) -> &Arc<juto_runtime::subagents::SubagentManager> {
        &self.subagents
    }
    fn can_send(&self, cx: &App) -> bool {
        let input = self.input.read(cx);
        !self.is_running
            && !input.is_composing()
            && (input.has_text() || !self.attachments.is_empty())
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

        self.messages.push(Message {
            role: MessageRole::User,
            text: if text.trim().is_empty() {
                SharedString::default()
            } else {
                text
            },
            attachments,
            desktop_artifacts: Vec::new(),
            approval: None,
        });
        self.attachment_error = None;
        self.artifact_notice = None;
        self.scroll.scroll_to_bottom();
        self.focus_input(window, cx);
        self.is_running = true;

        let control = RunControl::new();
        self.active_control = Some(control.clone());
        let runtime = self.runtime.clone();
        let session = self.session.clone();
        let tokio_handle = self.tokio_handle.clone();
        let (events_tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel();

        let run_future = tokio_handle.spawn(async move {
            let mut session_guard = session.lock().await;
            runtime
                .run(&mut session_guard, prompt, events_tx, control)
                .await
        });

        let approval_requests = self.approval_requests.clone();
        self.run_task = Some(cx.spawn_in(window, async move |this, cx| {
            let mut approvals = approval_requests.lock_owned().await;
            loop {
                tokio::select! {
                    biased;
                    event = events_rx.recv() => {
                        let Some(event) = event else { break };
                        let updated = this.update_in(cx, |this, _window, cx| {
                            this.handle_agent_event(event, cx);
                        });
                        if updated.is_err() {
                            break;
                        }
                    }
                    Some(request) = approvals.recv() => {
                        let updated = this.update_in(cx, |this, _window, cx| {
                            this.handle_approval_request(request, cx);
                        });
                        if updated.is_err() {
                            break;
                        }
                    }
                }
            }
            drop(approvals);

            let outcome = run_future.await;
            this.update_in(cx, |this, window, cx| {
                this.finish_run(outcome, window, cx);
            })
            .ok();
        }));

        cx.notify();
    }

    fn handle_agent_event(&mut self, event: AgentEvent, cx: &mut Context<Self>) {
        match event {
            AgentEvent::MessageStart => {
                self.messages.push(Message {
                    role: MessageRole::Assistant,
                    text: SharedString::default(),
                    attachments: Vec::new(),
                    desktop_artifacts: Vec::new(),
                    approval: None,
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
            AgentEvent::MessageEnd { message } => {
                let text = message.text();
                if let Some(last) = self.messages.last_mut()
                    && last.role == MessageRole::Assistant
                {
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
                        });
                    }
                } else {
                    self.messages.push(Message {
                        role: MessageRole::Assistant,
                        text: error_text.into(),
                        attachments: Vec::new(),
                        desktop_artifacts: Vec::new(),
                        approval: None,
                    });
                }
                self.scroll.scroll_to_bottom();
                cx.notify();
            }
            AgentEvent::ToolExecutionStart { .. } => {
                self.sync_subagents(cx);
            }
            AgentEvent::ToolExecutionEnd { result, .. } => {
                let artifacts = extract_desktop_artifacts(&result);
                if !artifacts.is_empty() {
                    if let Some(last) = self.messages.last_mut()
                        && last.role == MessageRole::Assistant
                    {
                        last.desktop_artifacts.extend(artifacts);
                    } else {
                        self.messages.push(Message {
                            role: MessageRole::Assistant,
                            text: SharedString::default(),
                            attachments: Vec::new(),
                            desktop_artifacts: artifacts,
                            approval: None,
                        });
                    }
                    self.scroll.scroll_to_bottom();
                    cx.notify();
                }
                self.sync_subagents(cx);
            }
            _ => {}
        }
    }

    fn handle_approval_request(&mut self, request: ApprovalRequest, cx: &mut Context<Self>) {
        if request.is_abandoned() {
            return;
        }
        let id = self.next_approval_id;
        self.next_approval_id += 1;
        let call = request.call();
        let card = ApprovalCard {
            id,
            tool: call.name.clone().into(),
            tier: request.tier(),
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
        });
        self.scroll.scroll_to_bottom();
        cx.notify();
    }

    fn resolve_approval(&mut self, id: u64, allow: bool, cx: &mut Context<Self>) {
        let Some(request) = self.pending_approvals.remove(&id) else {
            return;
        };
        request.respond(allow);
        let state = if allow {
            ApprovalState::Allowed
        } else {
            ApprovalState::Denied
        };
        self.set_approval_state(id, state);
        cx.notify();
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

    /// Ends outstanding approvals when a run stops. Dropping the responders
    /// fails any tool call still waiting on them instead of leaving it hung.
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

    fn finish_run(
        &mut self,
        outcome: Result<Result<RunOutcome, anyhow::Error>, tokio::task::JoinError>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.is_running = false;
        self.cancel_pending_approvals();
        self.active_control = None;
        self.run_task = None;
        match outcome {
            Ok(Ok(outcome)) => {
                if outcome.stop_reason == StopReason::Aborted
                    && let Some(last) = self.messages.last_mut()
                    && last.role == MessageRole::Assistant
                {
                    let mut text = last.text.to_string();
                    if !text.is_empty() {
                        text.push_str("\n\n[Run cancelled]");
                    } else {
                        text.push_str("[Run cancelled]");
                    }
                    last.text = text.into();
                } else if outcome.stop_reason == StopReason::Error
                    && let Some(last) = self.messages.last_mut()
                    && last.role == MessageRole::Assistant
                    && last.text.is_empty()
                {
                    last.text = "Error: Run failed".into();
                }
            }
            Ok(Err(err)) => {
                let error_msg = format!("Error: {err}");
                if let Some(last) = self.messages.last_mut() {
                    if last.role == MessageRole::Assistant && last.text.is_empty() {
                        last.text = error_msg.into();
                    } else if last.role == MessageRole::Assistant {
                        let mut text = last.text.to_string();
                        text.push_str(&format!("\n\n{error_msg}"));
                        last.text = text.into();
                    } else {
                        self.messages.push(Message {
                            role: MessageRole::Assistant,
                            text: error_msg.into(),
                            attachments: Vec::new(),
                            desktop_artifacts: Vec::new(),
                            approval: None,
                        });
                    }
                } else {
                    self.messages.push(Message {
                        role: MessageRole::Assistant,
                        text: error_msg.into(),
                        attachments: Vec::new(),
                        desktop_artifacts: Vec::new(),
                        approval: None,
                    });
                }
            }
            Err(join_err) => {
                let error_msg = format!("Error: Task panicked or was cancelled: {join_err}");
                self.messages.push(Message {
                    role: MessageRole::Assistant,
                    text: error_msg.into(),
                    attachments: Vec::new(),
                    desktop_artifacts: Vec::new(),
                    approval: None,
                });
            }
        }
        self.sync_subagents(cx);
        self.scroll.scroll_to_bottom();
        self.focus_input(window, cx);
        cx.notify();
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
            .px_6()
            .py_3p5()
            .border_b_1()
            .border_color(rgb(0x333333))
            .bg(rgb(0x1a1a1a))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(
                        div()
                            .text_size(px(20.))
                            .font_weight(gpui::FontWeight::BOLD)
                            .child("Juto"),
                    )
                    .child(div().text_sm().text_color(rgb(0xa0a0a0)).child("Chat")),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .id("header-model-btn")
                            .flex()
                            .items_center()
                            .gap_1p5()
                            .px_3()
                            .py_1p5()
                            .rounded_md()
                            .border_1()
                            .border_color(rgb(0x3f3f46))
                            .bg(rgb(0x27272a))
                            .text_xs()
                            .text_color(rgb(0xe4e4e7))
                            .cursor_pointer()
                            .hover(|s| s.bg(rgb(0x3f3f46)).border_color(rgb(0x52525b)))
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
                        div()
                            .id("header-auth-btn")
                            .flex()
                            .items_center()
                            .gap_1()
                            .px_3()
                            .py_1p5()
                            .rounded_md()
                            .border_1()
                            .border_color(rgb(0x3f3f46))
                            .bg(rgb(0x27272a))
                            .text_xs()
                            .text_color(rgb(0xd4d4d8))
                            .cursor_pointer()
                            .hover(|s| s.bg(rgb(0x3f3f46)).text_color(rgb(0xffffff)))
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
                        div()
                            .id("header-sidebar-btn")
                            .flex()
                            .items_center()
                            .gap_1p5()
                            .px_3()
                            .py_1p5()
                            .rounded_md()
                            .border_1()
                            .border_color(if self.show_sidebar {
                                rgb(0x2563eb)
                            } else {
                                rgb(0x3f3f46)
                            })
                            .bg(if self.show_sidebar {
                                rgb(0x1e3a5f)
                            } else {
                                rgb(0x27272a)
                            })
                            .text_xs()
                            .text_color(if self.show_sidebar {
                                rgb(0x60a5fa)
                            } else {
                                rgb(0xd4d4d8)
                            })
                            .cursor_pointer()
                            .hover(|s| s.bg(rgb(0x3f3f46)).text_color(rgb(0xffffff)))
                            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                this.toggle_sidebar(cx);
                            }))
                            .child("⚡ Mission Control"),
                    ),
            )
    }

    /// The main transcript and composer, with the Mission Control sidebar when shown.
    fn main_body(&self, window: &Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .flex_1()
            .min_h_0()
            .w_full()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .child(self.message_list(cx))
                    .child(self.composer(window, cx)),
            )
            .when(self.show_sidebar, |row| {
                row.child(
                    div()
                        .w(px(380.0))
                        .flex_none()
                        .h_full()
                        .flex()
                        .flex_col()
                        .bg(rgb(0x18181b))
                        .border_l_1()
                        .border_color(rgb(0x27272a))
                        .child(self.sidebar.clone()),
                )
            })
    }

    fn message_list(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let messages = self.messages.clone();
        div()
            .id("messages")
            .flex_1()
            .min_h_0()
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

impl Render for ChatView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .relative()
            .flex()
            .flex_col()
            .size_full()
            .bg(rgb(0x181818))
            .text_color(rgb(0xe7e7e7))
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
            .child(self.header(cx))
            .child(match self.focus.clone() {
                Some(focus) => focus.into_any_element(),
                None => self.main_body(window, cx).into_any_element(),
            })
            .when_some(self.modal_content(), |view, content| {
                view.child(self.modal_overlay(content, cx))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let messages = load_messages(&session);
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

        let messages = load_messages(&session);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, MessageRole::Assistant);
        assert_eq!(messages[0].desktop_artifacts.len(), 1);
        assert_eq!(messages[0].desktop_artifacts[0], artifact);
    }
}
