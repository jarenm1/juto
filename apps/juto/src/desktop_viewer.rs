//! Embedded live desktop viewer.
//!
//! Renders the composed frames of a desktop environment inside the harness
//! window and forwards pointer/keyboard input to the environment's virtual
//! seat. The viewer speaks the desktop control socket: frames come from the
//! environment's sealed shared-memory pool and input goes over the same socket
//! (`docs/agent-compositor.md` section 7). No video codec is involved.

use std::cell::Cell;
use std::collections::HashSet;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, mpsc};
use std::time::Duration;

use gpui::{
    AnyElement, Bounds, Context, CursorStyle, FocusHandle, KeyDownEvent, KeyUpEvent, Modifiers,
    ModifiersChangedEvent, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ObjectFit,
    Pixels, Point, Render, RenderImage, ScrollWheelEvent, Subscription, Task, Window, canvas, div,
    img, prelude::*, px, rgb, rgba,
};
use juto_daemon::client::Client;
use juto_desktop::protocol::{DesktopEvent, GrantRole, InputEvent};
use juto_desktop::supervisor::OutputConfig;
use juto_desktop::{DesktopError, ViewerClient, viewer::EventStream};

use crate::desktop_stream;
use crate::side_pane::SidePaneItem;

/// Control lease renewal interval (spec section 8: heartbeat 2s, expiry 6s).
const LEASE_HEARTBEAT: Duration = Duration::from_secs(2);
/// Keep-awake deadline: a live canvas renews it every heartbeat, so it lapses
/// within one lease expiry of the canvas going away.
const AWAKE_DEADLINE_SECS: u64 = 6;
/// How often the viewer drains cursor events from the listener thread.
const CURSOR_POLL: Duration = Duration::from_millis(33);
/// How long the side pane's size must hold before the desktop is resized to
/// it, so dragging the divider or the window does not reconfigure every frame.
const RESIZE_DEBOUNCE: Duration = Duration::from_millis(250);
/// Height of the canvas row while the viewer sits inline in a chat card.
const INLINE_HEIGHT: f32 = 405.;
/// Evdev codes for the modifier keys forwarded to the seat.
const KEY_LEFTSHIFT: u32 = 42;
const KEY_LEFTCTRL: u32 = 29;
const KEY_LEFTALT: u32 = 56;
const KEY_LEFTMETA: u32 = 125;

/// Seat key events for modifiers whose state differs between two GPUI snapshots.
fn modifier_events(previous: Modifiers, current: Modifiers) -> Vec<InputEvent> {
    [
        (previous.shift, current.shift, KEY_LEFTSHIFT),
        (previous.control, current.control, KEY_LEFTCTRL),
        (previous.alt, current.alt, KEY_LEFTALT),
        (previous.platform, current.platform, KEY_LEFTMETA),
    ]
    .into_iter()
    .filter(|(before, after, _)| before != after)
    .map(|(_, pressed, keycode)| InputEvent::Key { keycode, pressed })
    .collect()
}

/// Maps a CSS cursor name sent by the compositor to a GPUI cursor. Names GPUI has
/// no cursor for fall back to the arrow.
fn cursor_style_for_shape(shape: &str) -> CursorStyle {
    match shape {
        "none" => CursorStyle::None,
        "pointer" => CursorStyle::PointingHand,
        "text" => CursorStyle::IBeam,
        "vertical-text" => CursorStyle::IBeamCursorForVerticalLayout,
        "crosshair" => CursorStyle::Crosshair,
        "grab" => CursorStyle::OpenHand,
        "grabbing" => CursorStyle::ClosedHand,
        "not-allowed" | "no-drop" => CursorStyle::OperationNotAllowed,
        "copy" => CursorStyle::DragCopy,
        "alias" => CursorStyle::DragLink,
        "context-menu" => CursorStyle::ContextualMenu,
        "e-resize" => CursorStyle::ResizeRight,
        "w-resize" => CursorStyle::ResizeLeft,
        "n-resize" => CursorStyle::ResizeUp,
        "s-resize" => CursorStyle::ResizeDown,
        "ew-resize" => CursorStyle::ResizeLeftRight,
        "ns-resize" => CursorStyle::ResizeUpDown,
        "nesw-resize" => CursorStyle::ResizeUpRightDownLeft,
        "nwse-resize" => CursorStyle::ResizeUpLeftDownRight,
        "col-resize" => CursorStyle::ResizeColumn,
        "row-resize" => CursorStyle::ResizeRow,
        _ => CursorStyle::Arrow,
    }
}

/// The authenticated principal this interactive session acts as: the local
/// human viewing the desktop. Grants and artifact resolution are bound to it.
pub(crate) const SESSION_PRINCIPAL: &str = juto_runtime::SESSION_VIEWER_PRINCIPAL;

/// How a viewer may act on its environment, derived from its resolved grant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlAccess {
    /// The grant carries no Control role: the canvas is read-only.
    ViewOnly,
    /// The grant carries Control and ends at `expires_at` (unix seconds); the
    /// viewer stops offering control once it lapses.
    Granted { expires_at: u64 },
    /// Local developer session that created the environment. It issues and
    /// renews its own grant, since the host caps grant lifetime.
    LocalOwner,
}

/// Requested lifetime of a local owner grant; the host clamps it to its cap.
const LOCAL_GRANT_SECS: u64 = 1800;
/// Re-issue a local owner grant once less than this much lifetime remains.
const LOCAL_GRANT_RENEW_SECS: u64 = 60;

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

/// Roles of the local owner's grant: Owner, View, and Control.
fn local_grant_roles() -> Vec<GrantRole> {
    vec![GrantRole::Owner, GrantRole::View, GrantRole::Control]
}

/// The viewer's control connection, shared by the canvas and its heartbeat task.
type SharedClient = Arc<Mutex<ViewerClient>>;

fn lock_client(client: &SharedClient) -> MutexGuard<'_, ViewerClient> {
    client.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Where the viewer is shown. Moving between placements keeps the frame feed,
/// pause state and control lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// In its chat card, at a fixed height.
    Inline,
    /// In the window-sized overlay.
    Expanded,
    /// In its thread's side pane. The desktop's resolution follows the pane
    /// while the viewer is live and may control the desktop.
    Docked,
}

impl SidePaneItem for DesktopViewer {
    fn set_docked(&mut self, docked: bool, cx: &mut Context<Self>) {
        let placement = if docked {
            Placement::Docked
        } else {
            Placement::Inline
        };
        self.set_placement(placement, cx);
    }
}

/// Client cursor bitmap drawn over the canvas at the local pointer.
struct CursorOverlay {
    image: Arc<RenderImage>,
    /// Drawn size in environment pixels.
    display_size: (u32, u32),
    /// Hotspot in environment pixels.
    hotspot: (i32, i32),
}

/// An embedded viewer for one agent-owned desktop environment.
pub struct DesktopViewer {
    /// Control connection; `None` when the environment could not be attached.
    client: Option<SharedClient>,
    /// Socket and id needed to open the frame and event streams.
    socket: PathBuf,
    env_id: String,
    /// Heading shown above the canvas (the artifact title).
    title: String,
    frame: Option<Arc<RenderImage>>,
    frame_size: (u32, u32),
    /// Geometry generation last reported by the environment; input carries it.
    generation: u64,
    surface_bounds: Rc<Cell<Bounds<Pixels>>>,
    focus_handle: FocusHandle,
    status: String,
    error: Option<String>,
    has_control: bool,
    /// Whether the viewer's grant permits control. A view-only grant gets a
    /// read-only canvas: no control buttons and no input forwarding.
    can_control: bool,
    /// How control was granted; only a local owner grant is renewed here.
    access: ControlAccess,
    /// Expiry of the grant that authorizes control: renewed for a local owner,
    /// fixed for a resolved grant, None when view-only.
    grant_expires_at: Option<u64>,
    paused: bool,
    placement: Placement,
    /// Newest geometry generation this viewer's connection acknowledged. After
    /// a resize the desktop gates input until every viewer acknowledges.
    acknowledged_generation: Option<u64>,
    /// Debounced resize to the side pane's size.
    resize_task: Option<Task<()>>,
    frame_task: Option<Task<()>>,
    heartbeat_task: Option<Task<()>>,
    snapshot_task: Option<Task<()>>,
    /// Set when a left press starts while paused; its release resumes the canvas.
    press_started_paused: bool,
    /// Modifier state last reported by GPUI; only changes are sent to the seat.
    modifiers: Modifiers,
    /// Keys currently down on the seat; auto-repeat presses are not re-sent.
    held_keys: HashSet<u32>,
    /// Releases held input on focus or window-activation loss.
    _input_guards: Vec<Subscription>,
    /// Application cursor shape, mirrored onto the canvas while live.
    cursor: CursorStyle,
    /// Listener that keeps `cursor` current for the viewer's whole life.
    cursor_task: Option<Task<()>>,
    /// Client cursor bitmap drawn at the local pointer while live; `None` means the
    /// named shape above applies.
    cursor_overlay: Option<CursorOverlay>,
    /// Pointer position in window coordinates while over the image.
    cursor_pointer: Option<Point<Pixels>>,
    /// Whether a live canvas holds the desktop awake. The heartbeat renews the
    /// hold while this is set. Live frames are read from shared memory, which is
    /// not activity on its own, so without this a watched desktop would freeze at
    /// the idle limit. A paused canvas holds nothing, so the desktop can still go
    /// idle.
    awake: bool,
}

impl DesktopViewer {
    /// Creates a paused viewer for environment `env_id`, reached over the
    /// workspace desktop socket named in the daemon hello. It captures one still
    /// of the newest frame and runs no frame loop until resumed. An environment
    /// that cannot be attached yields a stopped viewer that shows the reason
    /// instead of panicking.
    pub fn new(
        client: Client,
        env_id: String,
        access: ControlAccess,
        title: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let hello = client.hello();
        let socket = hello
            .as_ref()
            .map(|hello| hello.desktop_socket.clone())
            .unwrap_or_default();
        let attached = match &hello {
            Some(hello) => {
                ViewerClient::attach(&hello.desktop_socket, &env_id).map_err(|err| err.to_string())
            }
            None => Err("daemon disconnected".to_string()),
        };
        let mut error = None;
        let (viewer_client, frame_size) = match attached {
            Ok(viewer_client) => {
                let info = viewer_client.info();
                let size = (info.width, info.height);
                (Some(Arc::new(Mutex::new(viewer_client))), size)
            }
            Err(err) => {
                error = Some(format!("desktop unavailable: {err}"));
                (None, (0, 0))
            }
        };
        let can_control = access != ControlAccess::ViewOnly;
        let mut viewer = Self {
            client: viewer_client,
            socket,
            env_id,
            title,
            frame: None,
            frame_size,
            generation: 0,
            surface_bounds: Rc::new(Cell::new(Bounds::default())),
            focus_handle: cx.focus_handle(),
            status: "waiting for first frame".to_string(),
            error,
            has_control: false,
            can_control,
            access,
            grant_expires_at: None,
            paused: true,
            placement: Placement::Inline,
            acknowledged_generation: None,
            resize_task: None,
            frame_task: None,
            snapshot_task: None,
            press_started_paused: false,
            modifiers: Modifiers::default(),
            heartbeat_task: None,
            held_keys: HashSet::new(),
            _input_guards: Vec::new(),
            cursor: CursorStyle::Arrow,
            cursor_task: None,
            cursor_overlay: None,
            cursor_pointer: None,
            awake: false,
        };
        viewer.grant_expires_at = match access {
            ControlAccess::ViewOnly => None,
            ControlAccess::Granted { expires_at } => Some(expires_at),
            ControlAccess::LocalOwner => match viewer.issue_local_grant() {
                Ok(expires_at) => Some(expires_at),
                Err(err) => {
                    viewer.error = Some(format!("control grant failed: {err}"));
                    viewer.can_control = false;
                    None
                }
            },
        };
        viewer._input_guards = vec![
            cx.on_focus_out(&viewer.focus_handle, window, |this, _event, _window, cx| {
                this.release_held_input(cx);
            }),
            cx.observe_window_activation(window, |this, window, cx| {
                if !window.is_window_active() {
                    this.release_held_input(cx);
                }
            }),
        ];
        if viewer.client.is_some() {
            viewer.start_heartbeat(cx);
            viewer.start_cursor_listener(window, cx);
            viewer.capture_snapshot(0, window, cx);
        }
        viewer
    }

    pub fn placement(&self) -> Placement {
        self.placement
    }

    /// Moves the viewer; the frame feed, pause state and lease are kept. The
    /// side pane resizes the desktop to fit it; leaving it keeps the size.
    pub fn set_placement(&mut self, placement: Placement, cx: &mut Context<Self>) {
        self.placement = placement;
        self.schedule_resize(cx);
        cx.notify();
    }

    /// Whether the desktop's resolution should follow the canvas now: in the
    /// side pane, live, attached, and allowed to control the desktop.
    fn follows_canvas(&self) -> bool {
        self.placement == Placement::Docked
            && !self.paused
            && self.can_control
            && self.client.is_some()
    }

    /// Resizes the desktop to the canvas once its size has held for
    /// [`RESIZE_DEBOUNCE`]. A newer call restarts the wait.
    fn schedule_resize(&mut self, cx: &mut Context<Self>) {
        if !self.follows_canvas() {
            self.resize_task = None;
            return;
        }
        self.resize_task = Some(cx.spawn(async move |this, cx| {
            cx.background_executor().timer(RESIZE_DEBOUNCE).await;
            let _ = this.update(cx, |this, cx| this.resize_to_canvas(cx));
        }));
    }

    /// Reconfigures the desktop to the canvas size in logical pixels, fitted to
    /// what the environment admits at its current frame rate.
    fn resize_to_canvas(&mut self, cx: &mut Context<Self>) {
        self.resize_task = None;
        if !self.follows_canvas() {
            return;
        }
        let size = self.surface_bounds.get().size;
        let (width, height) = (f32::from(size.width), f32::from(size.height));
        if width < 1.0 || height < 1.0 {
            return;
        }
        let status = match self.with_client(|client| client.describe()) {
            Ok(status) => status,
            Err(err) => {
                self.error = Some(format!("resize failed: {err}"));
                cx.notify();
                return;
            }
        };
        let current = status.output;
        let (width, height) = status
            .output_profile
            .fit(width as u32, height as u32, current.fps);
        if (width, height) == (current.width, current.height) {
            return;
        }
        // Keys held across a geometry change would be released against a
        // different layout; release them first.
        self.release_held_input(cx);
        let mode = OutputConfig {
            width,
            height,
            ..current
        };
        match self.with_client(|client| client.configure_output(status.generation, mode)) {
            Ok((generation, output)) => {
                self.generation = generation;
                self.error = None;
                self.status = format!("resized to {}x{}", output.width, output.height);
            }
            Err(err) => self.error = Some(format!("resize failed: {err}")),
        }
        cx.notify();
    }

    /// Reports a presented frame's generation over this viewer's connection,
    /// once per generation. Input after a resize waits for it.
    fn acknowledge_geometry(&mut self, generation: u64) {
        if self.client.is_none() || self.acknowledged_generation == Some(generation) {
            return;
        }
        match self.with_client(|client| client.geometry_ready(generation)) {
            Ok(()) => self.acknowledged_generation = Some(generation),
            // A newer resize already landed; its frame acknowledges instead.
            Err(DesktopError::Rejected { code, .. }) if code == "StaleGeneration" => {}
            Err(err) => self.error = Some(format!("geometry acknowledgement failed: {err}")),
        }
    }

    /// Runs `f` on the control connection. Fails when the environment was never
    /// attached.
    fn with_client<T>(
        &self,
        f: impl FnOnce(&mut ViewerClient) -> Result<T, DesktopError>,
    ) -> Result<T, DesktopError> {
        let Some(client) = &self.client else {
            return Err(DesktopError::Protocol("desktop not attached".to_string()));
        };
        f(&mut lock_client(client))
    }

    /// Whether the environment still exists. A destroyed environment fails its
    /// describe call.
    fn is_alive(&self) -> bool {
        self.with_client(|client| client.describe()).is_ok()
    }

    /// Issues the local owner's grant and returns its expiry.
    fn issue_local_grant(&self) -> Result<u64, DesktopError> {
        self.with_client(|client| {
            client.grant(
                SESSION_PRINCIPAL,
                local_grant_roles(),
                "session",
                LOCAL_GRANT_SECS,
            )
        })
        .map(|grant| grant.expires_at)
    }

    /// Follows the environment's shared frame feed. Only a live viewer does this.
    fn start_frames(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let mut feed = desktop_stream::subscribe(&self.socket, &self.env_id);
        self.frame_task = Some(cx.spawn_in(window, async move |this, cx| {
            while let Some(frame) = feed.next().await {
                let updated = this.update_in(cx, |this, window, cx| {
                    this.set_frame(frame, window, cx);
                });
                if updated.is_err() {
                    return;
                }
            }
            // The feed ends only when the environment does.
            let _ = this.update_in(cx, |this, _window, cx| this.mark_stopped(cx));
        }));
    }

    /// Mirrors the application's cursor shape and client cursor bitmap onto the
    /// canvas. Runs for the viewer's whole life, so a resume shows the current cursor.
    ///
    /// The event stream blocks, so a listener thread reads it and forwards each
    /// event; the UI drains the channel on a short timer.
    fn start_cursor_listener(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let (sender, receiver) = mpsc::channel::<DesktopEvent>();
        let socket = self.socket.clone();
        let env_id = self.env_id.clone();
        std::thread::Builder::new()
            .name("juto-desktop-cursor".to_string())
            .spawn(move || {
                let Ok(events) = EventStream::subscribe(&socket, &env_id) else {
                    return;
                };
                while let Ok(event) = events.next_event() {
                    if sender.send(event).is_err() {
                        break;
                    }
                }
            })
            .expect("failed to spawn desktop cursor listener");
        self.cursor_task = Some(cx.spawn_in(window, async move |this, cx| {
            loop {
                cx.background_executor().timer(CURSOR_POLL).await;
                let mut pending = Vec::new();
                let mut disconnected = false;
                loop {
                    match receiver.try_recv() {
                        Ok(event) => pending.push(event),
                        Err(mpsc::TryRecvError::Empty) => break,
                        Err(mpsc::TryRecvError::Disconnected) => {
                            disconnected = true;
                            break;
                        }
                    }
                }
                if !pending.is_empty() {
                    let update = this.update_in(cx, |this, window, cx| {
                        for event in pending {
                            this.apply_desktop_event(event, window, cx);
                        }
                    });
                    if update.is_err() {
                        break;
                    }
                }
                if disconnected {
                    break;
                }
            }
        }));
    }

    fn apply_desktop_event(
        &mut self,
        event: DesktopEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            DesktopEvent::CursorShapeChanged { shape, .. } => self.set_cursor_shape(&shape, cx),
            DesktopEvent::CursorImageChanged {
                width,
                height,
                display_width,
                display_height,
                hotspot_x,
                hotspot_y,
                pixels,
                ..
            } => {
                let overlay = cursor_overlay_from_event(
                    width,
                    height,
                    display_width,
                    display_height,
                    hotspot_x,
                    hotspot_y,
                    pixels,
                );
                self.set_cursor_overlay(overlay, cx);
            }
            // A paused canvas receives no frames, yet input after a resize waits
            // until it presents the new geometry. Capture a still of it.
            DesktopEvent::OutputChanged { generation, .. } if self.paused => {
                self.capture_snapshot(generation, window, cx);
            }
            // Missed shape changes leave the last known shape in place: the
            // compositor cannot re-send it, and guessing a default would be wrong too.
            _ => {}
        }
    }

    fn set_cursor_shape(&mut self, shape: &str, cx: &mut Context<Self>) {
        let style = cursor_style_for_shape(shape);
        if style != self.cursor {
            self.cursor = style;
            cx.notify();
        }
    }

    fn set_cursor_overlay(&mut self, overlay: Option<CursorOverlay>, cx: &mut Context<Self>) {
        self.cursor_overlay = overlay;
        cx.notify();
    }

    /// Records the pointer while it is over the image; `None` once it leaves.
    fn track_cursor_pointer(&mut self, position: Option<Point<Pixels>>, cx: &mut Context<Self>) {
        if self.cursor_pointer != position {
            self.cursor_pointer = position;
            cx.notify();
        }
    }

    /// The client cursor bitmap at the pointer, positioned in the canvas container.
    /// `None` while paused, with no bitmap, or with the pointer off the image.
    fn cursor_overlay_element(&self) -> Option<AnyElement> {
        if self.paused {
            return None;
        }
        let overlay = self.cursor_overlay.as_ref()?;
        let pointer = self.cursor_pointer?;
        let (_, _, scale) = self.fitted_geometry()?;
        let bounds = self.surface_bounds.get();
        let (display_width, display_height) = overlay.display_size;
        let (hotspot_x, hotspot_y) = overlay.hotspot;
        let left = f32::from(pointer.x) - f32::from(bounds.origin.x) - hotspot_x as f32 * scale;
        let top = f32::from(pointer.y) - f32::from(bounds.origin.y) - hotspot_y as f32 * scale;
        Some(
            img(overlay.image.clone())
                .absolute()
                .left(px(left))
                .top(px(top))
                .w(px(display_width as f32 * scale))
                .h(px(display_height as f32 * scale))
                .object_fit(ObjectFit::Fill)
                .into_any_element(),
        )
    }

    /// Captures one still of the newest frame of at least `min_generation`
    /// without starting the live feed.
    pub fn capture_snapshot(
        &mut self,
        min_generation: u64,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let mut feed = desktop_stream::subscribe(&self.socket, &self.env_id);
        self.snapshot_task = Some(cx.spawn_in(window, async move |this, cx| {
            // The worker may not have composed its first frame, or its first
            // frame after a resize, yet; the feed waits for one and ends only
            // when the environment does.
            while let Some(frame) = feed.next().await {
                if frame.generation < min_generation {
                    continue;
                }
                let _ = this.update_in(cx, |this, window, cx| {
                    this.set_frame(frame, window, cx);
                });
                return;
            }
        }));
    }

    fn set_frame(
        &mut self,
        frame: Arc<desktop_stream::StreamFrame>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let previous = self.frame.replace(frame.image.clone());
        if let Some(previous) = previous {
            cx.drop_image(previous, Some(window));
        }
        // Input is mapped in environment pixels, not the downscaled image.
        let (width, height) = frame.source;
        self.generation = frame.generation;
        self.acknowledge_geometry(frame.generation);
        self.frame_size = (width, height);
        if self.error.is_none() {
            self.status = if self.paused {
                "paused".to_string()
            } else {
                format!("live {width}x{height}")
            };
        }
        cx.notify();
    }

    /// Stops the frame loop and releases control. The last frame stays on
    /// screen as a still; input is not forwarded until the next resume.
    fn pause(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.paused {
            return;
        }
        self.release_held_input(cx);
        self.paused = true;
        self.frame_task = None;
        self.resize_task = None;
        // A paused canvas no longer keeps the desktop awake; the idle clock starts.
        self.awake = false;
        if self.has_control {
            self.release_control(cx);
        }
        self.status = "paused".to_string();
        self.capture_snapshot(0, window, cx);
        cx.notify();
    }

    /// Restarts the frame loop; the next frame arrives within one interval.
    fn resume(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.paused {
            return;
        }
        if !self.is_alive() {
            self.mark_stopped(cx);
            return;
        }
        // Resume a suspended desktop before the first live frame, and keep it
        // awake for as long as the canvas stays live.
        match self.with_client(|client| client.retain(SESSION_PRINCIPAL, AWAKE_DEADLINE_SECS)) {
            Ok(_) => self.awake = true,
            Err(err) => {
                self.error = Some(format!("desktop did not resume: {err}"));
                cx.notify();
                return;
            }
        }
        self.paused = false;
        self.status = "connecting".to_string();
        // A pending still must not overwrite live frames.
        self.snapshot_task = None;
        self.start_frames(window, cx);
        self.schedule_resize(cx);
        cx.notify();
    }

    /// Acquires (or renews) the human control lease.
    pub fn take_control(&mut self, cx: &mut Context<Self>) {
        if self.paused || !self.can_control {
            return;
        }
        match self.with_client(|client| client.take_control()) {
            Ok(epoch) => {
                self.has_control = true;
                self.error = None;
                self.status = format!("control held (epoch {epoch})");
            }
            Err(err) => {
                self.has_control = false;
                self.error = Some(err.to_string());
            }
        }
        cx.notify();
    }

    /// Releases the human control lease.
    pub fn release_control(&mut self, cx: &mut Context<Self>) {
        self.release_held_input(cx);
        let result = self.with_client(|client| client.release_control());
        // Drop the local flag even when the release call fails: the lease may
        // already be gone, and keeping `has_control` would render a stale state.
        self.has_control = false;
        match result {
            Ok(()) => {
                self.status = "control released".to_string();
                self.error = None;
            }
            Err(err) => self.error = Some(err.to_string()),
        }
        cx.notify();
    }

    /// Ends the session after the environment was destroyed underneath the
    /// viewer. The last frame stays on screen; input is no longer forwarded.
    fn mark_stopped(&mut self, cx: &mut Context<Self>) {
        self.frame_task = None;
        self.awake = false;
        self.has_control = false;
        self.status = "stopped".to_string();
        self.error = Some("desktop stopped".to_string());
        cx.notify();
    }

    /// Runs for the viewer's whole lifetime. Each tick it notices a destroyed
    /// environment, keeps a local owner's grant fresh, holds the desktop awake
    /// while the canvas is live, and renews the control lease while control is
    /// held. It sends no owner heartbeat: desktops the harness creates set no
    /// owner timeout, and a viewer alone must never keep an abandoned desktop
    /// alive.
    fn start_heartbeat(&mut self, cx: &mut Context<Self>) {
        let Some(client) = self.client.clone() else {
            return;
        };
        let access = self.access;
        let mut grant_expires_at = self.grant_expires_at;
        self.heartbeat_task = Some(cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(LEASE_HEARTBEAT).await;
                let status = match lock_client(&client).describe() {
                    Ok(status) => status,
                    Err(_) => {
                        let _ = this.update(cx, |this, cx| this.mark_stopped(cx));
                        break;
                    }
                };
                let Ok(awake) = this.update(cx, |this, _| {
                    this.generation = status.generation;
                    this.awake
                }) else {
                    break;
                };
                // A local owner session keeps its own grant alive before the host
                // cap expires it; otherwise control could not be renewed.
                if access == ControlAccess::LocalOwner
                    && grant_expires_at
                        .is_some_and(|expires_at| expires_at <= unix_now() + LOCAL_GRANT_RENEW_SECS)
                {
                    let renewed = lock_client(&client).grant(
                        SESSION_PRINCIPAL,
                        local_grant_roles(),
                        "session",
                        LOCAL_GRANT_SECS,
                    );
                    match renewed {
                        Ok(grant) => grant_expires_at = Some(grant.expires_at),
                        Err(err) => {
                            let _ = this.update(cx, |this, cx| {
                                this.error = Some(format!("control grant renewal failed: {err}"));
                                cx.notify();
                            });
                        }
                    }
                }
                // A live canvas keeps the desktop awake; a paused one does not.
                if awake {
                    let held = lock_client(&client).retain(SESSION_PRINCIPAL, AWAKE_DEADLINE_SECS);
                    if let Err(err) = held {
                        let _ = this.update(cx, |this, cx| {
                            this.error = Some(format!("keep-awake failed: {err}"));
                            cx.notify();
                        });
                    }
                }
                // A resolved grant ends at its own expiry, which the host enforces
                // too. Stop offering control and release the lease it authorized.
                if matches!(access, ControlAccess::Granted { expires_at } if expires_at <= unix_now()) {
                    let Ok(ended) = this.update(cx, |this, cx| this.end_control_grant(cx)) else {
                        break;
                    };
                    if ended {
                        let _ = lock_client(&client).release_control();
                    }
                }
                let Ok(holding) = this.update(cx, |this, _| this.has_control) else {
                    break;
                };
                if !holding {
                    continue;
                }
                // Renew only a lease this viewer already holds. Acquiring while
                // an agent holds control would take that lease over.
                let ours = matches!(
                    status.active_control_lease,
                    Some(lease) if lease.holder == SESSION_PRINCIPAL
                );
                let renewed = ours && lock_client(&client).take_control().is_ok();
                if this
                    .update(cx, |this, cx| {
                        this.has_control = renewed;
                        if !renewed {
                            this.error = Some("control lease lost".to_string());
                        }
                        cx.notify();
                    })
                    .is_err()
                {
                    break;
                }
            }
        }));
    }

    /// Ends control when its resolved grant lapses. Returns whether control was
    /// still offered, so the caller releases the lease only once.
    fn end_control_grant(&mut self, cx: &mut Context<Self>) -> bool {
        if !self.can_control {
            return false;
        }
        self.release_held_input(cx);
        self.can_control = false;
        self.has_control = false;
        self.error = Some("control grant expired".to_string());
        cx.notify();
        true
    }

    /// Sends one batch to the seat, taking control first when this viewer does
    /// not hold the lease yet.
    fn dispatch(&self, events: &[InputEvent]) -> Result<(), DesktopError> {
        let generation = self.generation;
        self.with_client(|client| {
            if client.control_epoch().is_none() {
                client.take_control()?;
            }
            client.send_input(generation, events.to_vec())
        })
    }

    fn send_events(&mut self, events: Vec<InputEvent>, cx: &mut Context<Self>) {
        // A paused or view-only viewer forwards no input; a paused click resumes instead.
        if self.paused || !self.can_control || events.is_empty() || !self.is_alive() {
            return;
        }
        match self.dispatch(&events) {
            Ok(()) => {
                self.has_control = true;
                self.error = None;
            }
            Err(err) => {
                // A geometry change makes the seat reject input with the old
                // generation. Refresh it so the next event goes through.
                if let Ok(status) = self.with_client(|client| client.describe()) {
                    self.generation = status.generation;
                }
                self.has_control = false;
                self.error = Some(err.to_string());
            }
        }
        cx.notify();
    }

    /// Forwards modifier changes as seat key events. The modifier state is
    /// updated even while paused, so it stays accurate for the next event.
    fn on_modifiers_changed(&mut self, event: &ModifiersChangedEvent, cx: &mut Context<Self>) {
        let events = modifier_events(self.modifiers, event.modifiers);
        self.modifiers = event.modifiers;
        self.send_events(events, cx);
    }

    /// Releases every key and modifier still down on the seat before input stops
    /// being forwarded: pause, control release, grant expiry, or focus loss.
    fn release_held_input(&mut self, cx: &mut Context<Self>) {
        let previous = std::mem::take(&mut self.modifiers);
        let mut events: Vec<InputEvent> = self
            .held_keys
            .drain()
            .map(|keycode| InputEvent::Key {
                keycode,
                pressed: false,
            })
            .collect();
        events.extend(modifier_events(previous, Modifiers::default()));
        if self.has_control {
            self.send_events(events, cx);
        }
    }

    /// Geometry of the letterboxed image inside the surface container:
    /// `(origin_x, origin_y, scale)` in window coordinates.
    fn fitted_geometry(&self) -> Option<(f32, f32, f32)> {
        let bounds = self.surface_bounds.get();
        let (image_width, image_height) = self.frame_size;
        if image_width == 0 || image_height == 0 {
            return None;
        }
        let box_width = f32::from(bounds.size.width);
        let box_height = f32::from(bounds.size.height);
        if box_width <= 0.0 || box_height <= 0.0 {
            return None;
        }
        let scale = (box_width / image_width as f32).min(box_height / image_height as f32);
        let drawn_width = image_width as f32 * scale;
        let drawn_height = image_height as f32 * scale;
        let origin_x = f32::from(bounds.origin.x) + (box_width - drawn_width) / 2.0;
        let origin_y = f32::from(bounds.origin.y) + (box_height - drawn_height) / 2.0;
        Some((origin_x, origin_y, scale))
    }

    fn pointer_position(&self, position: &gpui::Point<Pixels>) -> Option<(u32, u32)> {
        let (origin_x, origin_y, scale) = self.fitted_geometry()?;
        let (image_width, image_height) = self.frame_size;
        let x = f32::from(position.x) - origin_x;
        let y = f32::from(position.y) - origin_y;
        if x < 0.0 || y < 0.0 || x >= image_width as f32 * scale || y >= image_height as f32 * scale
        {
            return None;
        }
        Some((
            (x / scale).clamp(0.0, image_width as f32 - 1.0) as u32,
            (y / scale).clamp(0.0, image_height as f32 - 1.0) as u32,
        ))
    }

    fn on_key(&mut self, event: &KeyDownEvent, pressed: bool, cx: &mut Context<Self>) {
        // Only a live controlling viewer forwards keys. Held keys are tracked and
        // released on pause, so a forwarded press always gets its release.
        if self.paused || !self.has_control {
            return;
        }
        let Some(keycode) = evdev_keycode(&event.keystroke.key) else {
            return;
        };
        if pressed {
            // Repeats arrive as further key-downs: the X server's repeat on X11,
            // GPUI's is_held repeat on Wayland. The seat's repeat governs them.
            if !self.held_keys.insert(keycode) {
                return;
            }
        } else if !self.held_keys.remove(&keycode) {
            return;
        }
        self.send_events(vec![InputEvent::Key { keycode, pressed }], cx);
    }

    fn on_key_up(&mut self, event: &KeyUpEvent, cx: &mut Context<Self>) {
        self.on_key(
            &KeyDownEvent {
                keystroke: event.keystroke.clone(),
                is_held: false,
            },
            false,
            cx,
        );
    }
}

impl Render for DesktopViewer {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl gpui::IntoElement {
        let frame = self.frame.clone();
        let bounds_cell = self.surface_bounds.clone();
        let this = cx.weak_entity();
        let status = self.status.clone();
        let error = self.error.clone();
        let has_control = self.has_control;
        let paused = self.paused;
        let title = self.title.clone();
        let can_control = self.can_control;
        let cursor_overlay = self.cursor_overlay_element();
        let canvas_cursor = if paused {
            CursorStyle::Arrow
        } else if cursor_overlay.is_some() {
            CursorStyle::None
        } else {
            self.cursor
        };

        let surface = match frame {
            Some(frame) => img(frame)
                .absolute()
                .top_0()
                .left_0()
                .size_full()
                .object_fit(ObjectFit::Contain)
                .into_any_element(),
            None => div()
                .absolute()
                .top_0()
                .left_0()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .text_color(rgb(0x9aa0a6))
                .child("waiting for the first frame…")
                .into_any_element(),
        };

        div()
            .id("desktop-viewer")
            .track_focus(&self.focus_handle)
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, _window, cx| {
                this.on_key(event, true, cx);
            }))
            .on_key_up(cx.listener(|this, event: &KeyUpEvent, _window, cx| {
                this.on_key_up(event, cx);
            }))
            .on_modifiers_changed(cx.listener(
                |this, event: &ModifiersChangedEvent, _window, cx| {
                    this.on_modifiers_changed(event, cx);
                },
            ))
            .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _window, cx| {
                let pointer = this.pointer_position(&event.position);
                if let Some((x, y)) = pointer {
                    this.send_events(vec![InputEvent::PointerMoveAbs { x, y }], cx);
                }
                this.track_cursor_pointer(pointer.map(|_| event.position), cx);
            }))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, event: &MouseDownEvent, window, cx| {
                    this.press_started_paused = this.paused;
                    window.focus(&this.focus_handle);
                    if let Some((x, y)) = this.pointer_position(&event.position) {
                        this.send_events(
                            vec![
                                InputEvent::PointerMoveAbs { x, y },
                                InputEvent::PointerButton {
                                    button: 272,
                                    pressed: true,
                                },
                            ],
                            cx,
                        );
                    }
                }),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, event: &MouseUpEvent, window, cx| {
                    // Resume only when this press began while paused, so the
                    // Pause button's own click does not resume the canvas.
                    let started_paused = std::mem::take(&mut this.press_started_paused);
                    if this.paused {
                        if started_paused {
                            this.resume(window, cx);
                        }
                        return;
                    }
                    if let Some((x, y)) = this.pointer_position(&event.position) {
                        this.send_events(
                            vec![
                                InputEvent::PointerMoveAbs { x, y },
                                InputEvent::PointerButton {
                                    button: 272,
                                    pressed: false,
                                },
                            ],
                            cx,
                        );
                    }
                }),
            )
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(|this, event: &MouseDownEvent, window, cx| {
                    window.focus(&this.focus_handle);
                    if let Some((x, y)) = this.pointer_position(&event.position) {
                        this.send_events(
                            vec![
                                InputEvent::PointerMoveAbs { x, y },
                                InputEvent::PointerButton {
                                    button: 273,
                                    pressed: true,
                                },
                            ],
                            cx,
                        );
                    }
                }),
            )
            .on_mouse_up(
                MouseButton::Right,
                cx.listener(|this, event: &MouseUpEvent, _window, cx| {
                    if let Some((x, y)) = this.pointer_position(&event.position) {
                        this.send_events(
                            vec![
                                InputEvent::PointerMoveAbs { x, y },
                                InputEvent::PointerButton {
                                    button: 273,
                                    pressed: false,
                                },
                            ],
                            cx,
                        );
                    }
                }),
            )
            .on_scroll_wheel(cx.listener(|this, event: &ScrollWheelEvent, window, cx| {
                if this.paused || this.pointer_position(&event.position).is_none() {
                    return;
                }
                cx.stop_propagation();
                let delta = event.delta.pixel_delta(window.line_height());
                let dy = f32::from(delta.y);
                if dy != 0.0 {
                    this.send_events(
                        vec![InputEvent::PointerScroll {
                            axis: 1,
                            value: -dy as i32,
                        }],
                        cx,
                    );
                }
            }))
            .w_full()
            .map(|view| {
                if self.placement == Placement::Inline {
                    view.h(px(INLINE_HEIGHT))
                } else {
                    view.h_full()
                }
            })
            .flex()
            .flex_col()
            .gap_2()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .h(px(30.))
                    .child(div().text_color(rgb(0x9ca3af)).child(title))
                    .when(paused, |row| {
                        row.child(
                            div()
                                .id("desktop-resume")
                                .px_2()
                                .py_1()
                                .rounded_md()
                                .bg(rgb(0x2d2f34))
                                .text_color(rgb(0xe8eaed))
                                .cursor_pointer()
                                .child("Resume")
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.resume(window, cx);
                                })),
                        )
                    })
                    .when(!paused, |row| {
                        row.child(
                            div()
                                .flex()
                                .gap_2()
                                .child(
                                    div()
                                        .id("desktop-pause")
                                        .px_2()
                                        .py_1()
                                        .rounded_md()
                                        .bg(rgb(0x2d2f34))
                                        .text_color(rgb(0xe8eaed))
                                        .cursor_pointer()
                                        .child("Pause")
                                        .on_click(cx.listener(|this, _, window, cx| {
                                            this.pause(window, cx);
                                        })),
                                )
                                .when(can_control, |row| {
                                    row.child(
                                        div()
                                            .id("desktop-take-control")
                                            .px_2()
                                            .py_1()
                                            .rounded_md()
                                            .bg(rgb(0x2d2f34))
                                            .text_color(rgb(0xe8eaed))
                                            .cursor_pointer()
                                            .child(if has_control {
                                                "Renew control"
                                            } else {
                                                "Take control"
                                            })
                                            .on_click(cx.listener(|this, _, _window, cx| {
                                                this.take_control(cx);
                                            })),
                                    )
                                    .when(has_control, |row| {
                                        row.child(
                                            div()
                                                .id("desktop-release-control")
                                                .px_2()
                                                .py_1()
                                                .rounded_md()
                                                .bg(rgb(0x2d2f34))
                                                .text_color(rgb(0xe8eaed))
                                                .cursor_pointer()
                                                .child("Release control")
                                                .on_click(cx.listener(|this, _, _window, cx| {
                                                    this.release_control(cx);
                                                })),
                                        )
                                    })
                                }),
                        )
                    }),
            )
            .child(
                div()
                    .id("desktop-canvas")
                    .relative()
                    .flex_1()
                    .min_h_0()
                    .overflow_hidden()
                    .bg(rgb(0x000000))
                    .cursor(canvas_cursor)
                    .on_hover(cx.listener(|this, hovered: &bool, _window, cx| {
                        if !*hovered {
                            this.track_cursor_pointer(None, cx);
                        }
                    }))
                    .child(
                        canvas(
                            move |bounds, _window, cx| {
                                // Layout runs after render, so a new size is
                                // only known here; resize once it settles.
                                if bounds_cell.replace(bounds).size != bounds.size {
                                    let this = this.clone();
                                    cx.defer(move |cx| {
                                        let _ =
                                            this.update(cx, |this, cx| this.schedule_resize(cx));
                                    });
                                }
                            },
                            move |_bounds, _state, _window, _cx| {},
                        )
                        .absolute()
                        .size_full(),
                    )
                    .child(surface)
                    .when_some(cursor_overlay, |container, overlay| {
                        container.child(overlay)
                    })
                    .when(paused, |container| {
                        container.child(
                            div()
                                .absolute()
                                .inset_0()
                                .flex()
                                .items_center()
                                .justify_center()
                                .bg(rgba(0x00000080))
                                .text_color(rgb(0xe8eaed))
                                .child("Paused — click to resume"),
                        )
                    }),
            )
            .child(div().text_color(rgb(0x9aa0a6)).child(match &error {
                Some(error) => format!("{status} — {error}"),
                None => status,
            }))
    }
}

/// Builds the cursor overlay from a `CursorImageChanged` payload. Empty or
/// inconsistent payloads yield `None`, so the viewer shows the named cursor.
fn cursor_overlay_from_event(
    width: u32,
    height: u32,
    display_width: u32,
    display_height: u32,
    hotspot_x: i32,
    hotspot_y: i32,
    pixels: Vec<u8>,
) -> Option<CursorOverlay> {
    let expected = width as usize * height as usize * 4;
    if width == 0
        || height == 0
        || display_width == 0
        || display_height == 0
        || pixels.len() != expected
    {
        return None;
    }
    Some(CursorOverlay {
        image: desktop_stream::render_image_from_bgra(pixels, width, height),
        display_size: (display_width, display_height),
        hotspot: (hotspot_x, hotspot_y),
    })
}

/// Maps a gpui key name to a Linux evdev keycode.
///
/// Only the keys a desktop session needs for ordinary interaction are mapped;
/// unmapped keys are dropped rather than synthesized as something else.
fn evdev_keycode(key: &str) -> Option<u32> {
    let code = match key {
        "escape" => 1,
        "1" => 2,
        "2" => 3,
        "3" => 4,
        "4" => 5,
        "5" => 6,
        "6" => 7,
        "7" => 8,
        "8" => 9,
        "9" => 10,
        "0" => 11,
        "minus" => 12,
        "equal" => 13,
        "backspace" => 14,
        "tab" => 15,
        "q" => 16,
        "w" => 17,
        "e" => 18,
        "r" => 19,
        "t" => 20,
        "y" => 21,
        "u" => 22,
        "i" => 23,
        "o" => 24,
        "p" => 25,
        "[" => 26,
        "]" => 27,
        "enter" => 28,
        "control" => 29,
        "a" => 30,
        "s" => 31,
        "d" => 32,
        "f" => 33,
        "g" => 34,
        "h" => 35,
        "j" => 36,
        "k" => 37,
        "l" => 38,
        ";" => 39,
        "'" => 40,
        "`" => 41,
        "shift" => 42,
        "\\" => 43,
        "z" => 44,
        "x" => 45,
        "c" => 46,
        "v" => 47,
        "b" => 48,
        "n" => 49,
        "m" => 50,
        "," => 51,
        "." => 52,
        "/" => 53,
        "alt" => 56,
        "space" => 57,
        "capslock" => 58,
        "f1" => 59,
        "f2" => 60,
        "f3" => 61,
        "f4" => 62,
        "f5" => 63,
        "f6" => 64,
        "f7" => 65,
        "f8" => 66,
        "f9" => 67,
        "f10" => 68,
        "f11" => 87,
        "f12" => 88,
        "home" => 102,
        "up" => 103,
        "pageup" => 104,
        "left" => 105,
        "right" => 106,
        "end" => 107,
        "down" => 108,
        "pagedown" => 109,
        "insert" => 110,
        "delete" => 111,
        "super" | "meta" | "cmd" => 125,
        _ => return None,
    };
    Some(code)
}
