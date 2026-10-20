//! Embedded live desktop viewer.
//!
//! Renders the composed frames of a [`DesktopEnvironment`] inside the harness
//! window and forwards pointer/keyboard input to the environment's virtual
//! seat. Frames are read from the environment's sealed shared-memory pool; no
//! video codec is involved on this path (`docs/agent-compositor.md` section 7).

use std::cell::Cell;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use gpui::{
    AnyElement, Bounds, Context, CursorStyle, FocusHandle, KeyDownEvent, KeyUpEvent, Modifiers,
    ModifiersChangedEvent, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ObjectFit,
    Pixels, Point, Render, RenderImage, ScrollWheelEvent, Subscription, Task, Window, canvas, div,
    img, prelude::*, px, rgb, rgba,
};
use juto_runtime::protocol::{DesktopEvent, GrantRole, InputBatch, InputEvent};
use juto_runtime::{AwakeHold, DesktopEnvironment};
use tokio::sync::broadcast::error::RecvError;

use crate::desktop_stream;

/// Control lease renewal interval (spec section 8: heartbeat 2s, expiry 6s).
const LEASE_HEARTBEAT: Duration = Duration::from_secs(2);
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

/// Issues the local owner's Owner/View/Control grant and returns its expiry.
fn issue_local_grant(env: &DesktopEnvironment) -> u64 {
    env.issue_grant(
        SESSION_PRINCIPAL.to_string(),
        vec![GrantRole::Owner, GrantRole::View, GrantRole::Control],
        "session".to_string(),
        LOCAL_GRANT_SECS,
    )
    .expires_at
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
    env: DesktopEnvironment,
    frame: Option<Arc<RenderImage>>,
    frame_size: (u32, u32),
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
    expanded: bool,
    input_seq: u64,
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
    /// Resumes a suspended desktop and keeps it awake while the canvas is live.
    /// Live frames are read from shared memory, which is not activity on its
    /// own, so without this a watched desktop would freeze at the idle limit. A
    /// paused canvas holds nothing, so the desktop can still go idle.
    awake: Option<AwakeHold>,
}

impl DesktopViewer {
    /// Creates a paused viewer for `env`. It captures one still of the newest
    /// frame and runs no frame loop until resumed.
    pub fn new(
        env: DesktopEnvironment,
        access: ControlAccess,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let output = env.output();
        let can_control = access != ControlAccess::ViewOnly;
        let grant_expires_at = match access {
            ControlAccess::ViewOnly => None,
            ControlAccess::Granted { expires_at } => Some(expires_at),
            ControlAccess::LocalOwner => Some(issue_local_grant(&env)),
        };
        let mut viewer = Self {
            env,
            frame: None,
            frame_size: (output.width, output.height),
            surface_bounds: Rc::new(Cell::new(Bounds::default())),
            focus_handle: cx.focus_handle(),
            status: "waiting for first frame".to_string(),
            error: None,
            has_control: false,
            can_control,
            access,
            grant_expires_at,
            paused: true,
            expanded: false,
            input_seq: 0,
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
            awake: None,
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
        viewer.start_heartbeat(cx);
        viewer.start_cursor_listener(window, cx);
        viewer.capture_snapshot(window, cx);
        viewer
    }

    pub fn is_expanded(&self) -> bool {
        self.expanded
    }

    /// Presentation changes keep the same frame feed, pause state and lease.
    pub fn set_expanded(&mut self, expanded: bool, cx: &mut Context<Self>) {
        self.expanded = expanded;
        cx.notify();
    }

    /// Follows the environment's shared frame feed. Only a live viewer does this.
    fn start_frames(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let mut feed = desktop_stream::subscribe(&self.env);
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
    fn start_cursor_listener(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let mut events = self.env.subscribe();
        self.cursor_task = Some(cx.spawn_in(window, async move |this, cx| {
            loop {
                let update = match events.recv().await {
                    Ok(DesktopEvent::CursorShapeChanged { shape, .. }) => {
                        this.update_in(cx, |this, _window, cx| this.set_cursor_shape(&shape, cx))
                    }
                    Ok(DesktopEvent::CursorImageChanged {
                        width,
                        height,
                        display_width,
                        display_height,
                        hotspot_x,
                        hotspot_y,
                        pixels,
                        ..
                    }) => {
                        let overlay = cursor_overlay_from_event(
                            width,
                            height,
                            display_width,
                            display_height,
                            hotspot_x,
                            hotspot_y,
                            pixels,
                        );
                        this.update_in(cx, |this, _window, cx| this.set_cursor_overlay(overlay, cx))
                    }
                    Ok(_) => continue,
                    // Missed shape changes leave the last known shape in place: the
                    // compositor cannot re-send it, and guessing a default would be wrong too.
                    Err(RecvError::Lagged(_)) => continue,
                    Err(RecvError::Closed) => break,
                };
                if update.is_err() {
                    break;
                }
            }
        }));
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

    /// Captures one still of the newest frame without starting the live feed.
    pub fn capture_snapshot(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let mut feed = desktop_stream::subscribe(&self.env);
        self.snapshot_task = Some(cx.spawn_in(window, async move |this, cx| {
            // The worker may not have composed its first frame yet; the feed waits
            // for one and ends only when the environment does.
            if let Some(frame) = feed.next().await {
                let _ = this.update_in(cx, |this, window, cx| {
                    this.set_frame(frame, window, cx);
                });
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
        // A paused canvas no longer keeps the desktop awake; the idle clock starts.
        self.awake = None;
        if self.has_control {
            self.release_control(cx);
        }
        self.status = "paused".to_string();
        self.capture_snapshot(window, cx);
        cx.notify();
    }

    /// Restarts the frame loop; the next frame arrives within one interval.
    fn resume(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.paused {
            return;
        }
        if !self.env.is_alive() {
            self.mark_stopped(cx);
            return;
        }
        // Resume a suspended desktop before the first live frame, and keep it
        // awake for as long as the canvas stays live.
        match self.env.hold_awake() {
            Ok(hold) => self.awake = Some(hold),
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
        cx.notify();
    }

    /// Acquires (or renews) the human control lease.
    pub fn take_control(&mut self, cx: &mut Context<Self>) {
        if self.paused || !self.can_control {
            return;
        }
        match self.env.set_control(SESSION_PRINCIPAL, "acquire", true) {
            Ok(Some(lease)) => {
                self.has_control = true;
                self.error = None;
                self.status = format!("control held (epoch {})", lease.epoch);
            }
            Ok(None) => {
                self.has_control = false;
                self.status = "no control".to_string();
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
        let result = self.env.set_control(SESSION_PRINCIPAL, "release", true);
        // Drop the local flag even when the release call fails: the lease may
        // already be gone, and keeping `has_control` would render a stale state.
        self.has_control = false;
        match result {
            Ok(_) => {
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
        self.awake = None;
        self.has_control = false;
        self.status = "stopped".to_string();
        self.error = Some("desktop stopped".to_string());
        cx.notify();
    }

    /// Runs for the viewer's whole lifetime. Each tick it notices a destroyed
    /// environment, keeps a local owner's grant fresh, and renews the control
    /// lease while control is held. It sends no owner heartbeat: desktops the
    /// harness creates set no owner timeout, and a viewer alone must never keep
    /// an abandoned desktop alive.
    fn start_heartbeat(&mut self, cx: &mut Context<Self>) {
        let env = self.env.clone();
        let access = self.access;
        let mut grant_expires_at = self.grant_expires_at;
        self.heartbeat_task = Some(cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(LEASE_HEARTBEAT).await;
                if !env.is_alive() {
                    let _ = this.update(cx, |this, cx| this.mark_stopped(cx));
                    break;
                }
                // A local owner session keeps its own grant alive before the host
                // cap expires it; otherwise control could not be renewed.
                if access == ControlAccess::LocalOwner
                    && grant_expires_at
                        .is_some_and(|expires_at| expires_at <= unix_now() + LOCAL_GRANT_RENEW_SECS)
                {
                    grant_expires_at = Some(issue_local_grant(&env));
                }
                // A resolved grant ends at its own expiry, which the host enforces
                // too. Stop offering control and release the lease it authorized.
                if matches!(access, ControlAccess::Granted { expires_at } if expires_at <= unix_now()) {
                    let Ok(ended) = this.update(cx, |this, cx| this.end_control_grant(cx)) else {
                        break;
                    };
                    if ended {
                        let _ = env.set_control(SESSION_PRINCIPAL, "release", true);
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
                    env.get_status().active_control_lease,
                    Some(lease) if lease.holder == SESSION_PRINCIPAL
                );
                let renewed = ours
                    && env
                        .set_control(SESSION_PRINCIPAL, "acquire", true)
                        .ok()
                        .flatten()
                        .is_some();
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

    fn send_events(&mut self, events: Vec<InputEvent>, cx: &mut Context<Self>) {
        // A paused or view-only viewer forwards no input; a paused click resumes instead.
        if self.paused || !self.can_control || events.is_empty() || !self.env.is_alive() {
            return;
        }
        let lease = match self.env.get_status().active_control_lease {
            Some(lease) if lease.holder == SESSION_PRINCIPAL => lease,
            _ => match self.env.set_control(SESSION_PRINCIPAL, "acquire", true) {
                Ok(Some(lease)) => {
                    self.has_control = true;
                    lease
                }
                Ok(None) => return,
                Err(err) => {
                    self.error = Some(err.to_string());
                    cx.notify();
                    return;
                }
            },
        };
        self.input_seq = self.input_seq.wrapping_add(1);
        let batch = InputBatch {
            seq: self.input_seq,
            client_timestamp_us: 0,
            events,
        };
        match self
            .env
            .dispatch_input(SESSION_PRINCIPAL, lease.epoch, self.env.generation(), batch)
        {
            Ok(_) => self.error = None,
            Err(err) => self.error = Some(err.to_string()),
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
        let status = self.status.clone();
        let error = self.error.clone();
        let has_control = self.has_control;
        let paused = self.paused;
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
                if self.expanded {
                    view.h_full()
                } else {
                    view.h(px(405.))
                }
            })
            .flex()
            .flex_col()
            .gap_2()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_end()
                    .h(px(30.))
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
                            move |bounds, _window, _cx| {
                                bounds_cell.set(bounds);
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
