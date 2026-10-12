//! Headless Smithay compositor for a single agent-owned desktop environment.
//!
//! One virtual output, one virtual seat, no host display and no physical input.
//! Toplevels are composed with Smithay's software (pixman) renderer into a
//! memory image; the composed frame is copied into the environment's sealed
//! shared-memory pool and authorized input is injected into the virtual seat.
//! See `docs/agent-compositor.md` sections 2, 5, and 8.

use std::collections::HashMap;
use std::os::fd::{AsRawFd, BorrowedFd};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::text_input::{TextInputManagerState, TextInputUserData};
use smithay::backend::input::{Axis, ButtonState, KeyState};
use smithay::backend::renderer::pixman::PixmanRenderer;
use smithay::backend::renderer::utils::{draw_render_elements, on_commit_buffer_handler};
use smithay::backend::renderer::{Bind, Color32F, Frame, Renderer};
use smithay::desktop::Window;
use smithay::desktop::space::Space;
use smithay::input::keyboard::{FilterResult, KeyboardHandle, Keycode, XkbConfig};
use smithay::input::pointer::{AxisFrame, ButtonEvent, MotionEvent, PointerHandle};
use smithay::input::{Seat, SeatHandler, SeatState};
use smithay::output::{Mode, Output, PhysicalProperties, Scale, Subpixel};
use smithay::reexports::calloop::channel::{
    Channel, Event as ChannelEvent, Sender as ChannelSender, channel,
};
use smithay::reexports::calloop::generic::Generic;
use smithay::reexports::calloop::timer::{TimeoutAction, Timer};
use smithay::reexports::calloop::{
    EventLoop, Interest, LoopSignal, Mode as CalloopMode, PostAction,
};
use smithay::reexports::wayland_protocols::wp::text_input::zv3::server::{
    zwp_text_input_manager_v3::{self, ZwpTextInputManagerV3},
    zwp_text_input_v3::{self, ZwpTextInputV3},
};
use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel;
use smithay::reexports::wayland_server::backend::{ClientData, ClientId, DisconnectReason};
use smithay::reexports::wayland_server::protocol::wl_buffer;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{
    Client, DataInit, Dispatch, Display, GlobalDispatch, ListeningSocket, New, Resource,
};
use smithay::utils::{
    Logical, Physical, Point, Rectangle, SERIAL_COUNTER, Serial, Size, Transform,
};
use smithay::wayland::buffer::BufferHandler;
use smithay::wayland::compositor::{
    CompositorClientState, CompositorHandler, CompositorState, SurfaceAttributes, TraversalAction,
    with_states, with_surface_tree_downward,
};
use smithay::wayland::output::{OutputHandler, OutputManagerState};
use smithay::wayland::seat::WaylandFocus;
use smithay::wayland::selection::SelectionHandler;
use smithay::wayland::selection::data_device::{
    ClientDndGrabHandler, DataDeviceHandler, DataDeviceState, ServerDndGrabHandler,
};
use smithay::wayland::shell::xdg::{
    PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState,
    XdgToplevelSurfaceData,
};
use smithay::wayland::shm::{ShmHandler, ShmState};

use crate::error::DesktopError;
use crate::protocol::{DesktopEvent, InputEvent, WindowInfo};
use crate::shm::ShmFramePool;

/// Static configuration for one headless compositor instance.
#[derive(Debug, Clone)]
pub struct CompositorConfig {
    pub env_id: String,
    pub width: u32,
    pub height: u32,
    pub scale: u32,
    pub fps: u32,
    /// Absolute path of the Wayland listening socket.
    pub socket_path: PathBuf,
}

/// Commands accepted by the compositor thread.
pub enum CompositorCommand {
    /// Inject one authorized input event into the virtual seat.
    Input(InputEvent),
    /// Atomically change the virtual output mode.
    Configure {
        width: u32,
        height: u32,
        scale: u32,
        fps: u32,
        generation: u64,
        reply: std_mpsc::SyncSender<Result<(), String>>,
    },
    /// Force a fresh composition and report the resulting frame.
    Snapshot {
        reply: std_mpsc::SyncSender<Result<SnapshotInfo, String>>,
    },
    /// Stop the compositor and release the socket.
    Shutdown,
}

/// Metadata describing a composed frame written into the environment pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotInfo {
    pub frame_id: u64,
    pub generation: u64,
    pub timestamp_us: u64,
    pub width: u32,
    pub height: u32,
    pub slot: usize,
    /// Virtual-clock ticks since the compositor started.
    pub ticks: u64,
    /// Frames composed since the compositor started.
    pub frames_composed: u64,
    /// Frames dropped because every pool slot was held by a reader.
    pub frames_dropped: u64,
    /// Duration of the last composition, in microseconds.
    pub compose_micros: u64,
}

/// Cloneable handle for sending commands to a running compositor thread.
#[derive(Clone)]
pub struct CompositorSender {
    cmd_tx: ChannelSender<CompositorCommand>,
}

impl CompositorSender {
    /// Queue a command for the compositor thread.
    pub fn send(&self, command: CompositorCommand) -> Result<(), DesktopError> {
        self.cmd_tx
            .send(command)
            .map_err(|_| DesktopError::Compositor("compositor thread is not running".to_string()))
    }
}

/// Handle to a running compositor thread.
pub struct Compositor {
    cmd_tx: ChannelSender<CompositorCommand>,
    socket_path: PathBuf,
    join: Option<std::thread::JoinHandle<()>>,
}

impl Compositor {
    /// Start a compositor thread bound to `config.socket_path`.
    ///
    /// Returns only after the Wayland socket is listening, so a caller may
    /// launch clients immediately.
    pub fn spawn(
        config: CompositorConfig,
        pool: Arc<ShmFramePool>,
        events: std_mpsc::Sender<DesktopEvent>,
    ) -> Result<Self, DesktopError> {
        let (cmd_tx, cmd_rx) = channel::<CompositorCommand>();
        let (ready_tx, ready_rx) = std_mpsc::sync_channel::<Result<(), String>>(1);
        let socket_path = config.socket_path.clone();

        let join = std::thread::Builder::new()
            .name(format!("juto-compositor-{}", config.env_id))
            .spawn(move || {
                if let Err(err) = run_compositor(config, pool, events, cmd_rx, &ready_tx) {
                    let _ = ready_tx.try_send(Err(err.to_string()));
                    tracing::error!(error = %err, "compositor thread exited with error");
                }
            })
            .map_err(|e| {
                DesktopError::Compositor(format!("failed to spawn compositor thread: {e}"))
            })?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                cmd_tx,
                socket_path,
                join: Some(join),
            }),
            Ok(Err(msg)) => {
                let _ = join.join();
                Err(DesktopError::Compositor(msg))
            }
            Err(_) => {
                let _ = join.join();
                Err(DesktopError::Compositor(
                    "compositor thread exited before signalling readiness".to_string(),
                ))
            }
        }
    }

    /// Absolute path of the Wayland socket clients must connect to.
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// A cloneable command sender for this compositor.
    pub fn sender(&self) -> CompositorSender {
        CompositorSender {
            cmd_tx: self.cmd_tx.clone(),
        }
    }

    /// Queue a command for the compositor thread.
    pub fn send(&self, command: CompositorCommand) -> Result<(), DesktopError> {
        self.cmd_tx
            .send(command)
            .map_err(|_| DesktopError::Compositor("compositor thread is not running".to_string()))
    }

    /// Stop the compositor, join its thread, and remove the socket.
    pub fn shutdown(mut self) {
        let _ = self.cmd_tx.send(CompositorCommand::Shutdown);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

impl Drop for Compositor {
    fn drop(&mut self) {
        if let Some(join) = self.join.take() {
            let _ = self.cmd_tx.send(CompositorCommand::Shutdown);
            let _ = join.join();
            let _ = std::fs::remove_file(&self.socket_path);
        }
    }
}

/// Per-client Wayland state.
#[derive(Default)]
struct ClientState {
    compositor_state: CompositorClientState,
}

impl ClientData for ClientState {
    fn initialized(&self, _client_id: ClientId) {}
    fn disconnected(&self, _client_id: ClientId, _reason: DisconnectReason) {}
}

/// A mapped toplevel tracked by the compositor.
struct MappedWindow {
    id: u32,
    window: Window,
    info: WindowInfo,
}

/// The single-threaded Smithay state owned by the compositor thread.
struct DesktopState {
    display: Option<Display<DesktopState>>,
    signal: LoopSignal,

    compositor_state: CompositorState,
    xdg_shell_state: XdgShellState,
    shm_state: ShmState,
    seat_state: SeatState<Self>,
    data_device_state: DataDeviceState,
    output_manager_state: OutputManagerState,
    text_input_manager_state: TextInputManagerState,

    space: Space<Window>,
    output: Output,
    /// Kept alive so the virtual seat stays registered for the environment.
    _seat: Seat<Self>,
    keyboard: KeyboardHandle<Self>,
    pointer: PointerHandle<Self>,

    renderer: PixmanRenderer,
    image: smithay::reexports::pixman::Image<'static, 'static>,
    pool: Arc<ShmFramePool>,
    events: std_mpsc::Sender<DesktopEvent>,

    env_id: String,
    width: u32,
    height: u32,
    scale: u32,
    fps: u32,
    generation: u64,
    frame_counter: u64,
    ticks: u64,
    frames_composed: u64,
    frames_dropped: u64,
    compose_micros: u64,
    dirty: bool,
    start_time: Instant,

    windows: HashMap<u32, MappedWindow>,
    next_window_id: u32,
    cascade_index: i32,

    pending_snapshots: Vec<std_mpsc::SyncSender<Result<SnapshotInfo, String>>>,
}

impl DesktopState {
    fn now_us() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0)
    }

    fn logical_size(&self) -> Size<i32, Logical> {
        Size::from((
            (self.width / self.scale.max(1)) as i32,
            (self.height / self.scale.max(1)) as i32,
        ))
    }

    fn next_cascade(&mut self) -> Point<i32, Logical> {
        let step = 32;
        let slots = 8;
        let offset = (self.cascade_index % slots) * step;
        self.cascade_index = self.cascade_index.wrapping_add(1);
        Point::from((offset, offset))
    }

    fn window_info(&self, mapped: &MappedWindow) -> WindowInfo {
        let geometry = self
            .space
            .element_geometry(&mapped.window)
            .unwrap_or_else(|| Rectangle::from_size(self.logical_size()));
        let (title, app_id) = mapped
            .window
            .toplevel()
            .map(toplevel_identity)
            .unwrap_or((None, None));
        let is_focused = self
            .keyboard
            .current_focus()
            .map(|focus| {
                mapped
                    .window
                    .wl_surface()
                    .map(|s| s.as_ref() == &focus)
                    .unwrap_or(false)
            })
            .unwrap_or(false);
        WindowInfo {
            window_id: mapped.id,
            title: title.unwrap_or_default(),
            app_id: app_id.unwrap_or_default(),
            x: geometry.loc.x,
            y: geometry.loc.y,
            width: geometry.size.w.max(0) as u32,
            height: geometry.size.h.max(0) as u32,
            is_focused,
        }
    }

    fn focus_window(&mut self, window: &Window) {
        if let Some(surface) = window.wl_surface() {
            let keyboard = self.keyboard.clone();
            keyboard.set_focus(
                self,
                Some(surface.as_ref().clone()),
                SERIAL_COUNTER.next_serial(),
            );
        }
    }

    fn map_toplevel(&mut self, toplevel: ToplevelSurface) {
        let window = Window::new_wayland_window(toplevel.clone());
        if self.windows.values().any(|m| m.window == window) {
            return;
        }
        let location = self.next_cascade();
        self.space.map_element(window.clone(), location, true);
        self.focus_window(&window);
        let id = self.next_window_id;
        self.next_window_id = self.next_window_id.wrapping_add(1);
        let mut mapped = MappedWindow {
            id,
            window,
            info: WindowInfo {
                window_id: id,
                title: String::new(),
                app_id: String::new(),
                x: 0,
                y: 0,
                width: 0,
                height: 0,
                is_focused: true,
            },
        };
        mapped.info = self.window_info(&mapped);
        let info = mapped.info.clone();
        self.windows.insert(id, mapped);
        self.dirty = true;
        let _ = self.events.send(DesktopEvent::WindowMapped {
            env_id: self.env_id.clone(),
            window: info,
        });
    }

    fn unmap_window(&mut self, id: u32) {
        if let Some(mapped) = self.windows.remove(&id) {
            self.space.unmap_elem(&mapped.window);
            self.dirty = true;
            let _ = self.events.send(DesktopEvent::WindowUnmapped {
                env_id: self.env_id.clone(),
                window_id: id,
            });
        }
    }

    /// Recomputes and stores the reported metadata for a mapped window.
    fn refresh_window_info(&mut self, id: u32) {
        let info = match self.windows.get(&id) {
            Some(mapped) => self.window_info(mapped),
            None => return,
        };
        let changed = match self.windows.get_mut(&id) {
            Some(mapped) => {
                if mapped.info == info {
                    false
                } else {
                    mapped.info = info.clone();
                    true
                }
            }
            None => false,
        };
        if changed {
            let _ = self.events.send(DesktopEvent::WindowChanged {
                env_id: self.env_id.clone(),
                window: info,
            });
        }
    }

    fn window_id_for_surface(&self, surface: &WlSurface) -> Option<u32> {
        self.windows.iter().find_map(|(id, mapped)| {
            mapped
                .window
                .wl_surface()
                .map(|s| s.as_ref() == surface)
                .unwrap_or(false)
                .then_some(*id)
        })
    }

    /// Compose the desktop into the memory image and copy it into the pool.
    fn render_frame(&mut self) -> Result<SnapshotInfo, DesktopError> {
        let started = Instant::now();
        let physical = Size::<i32, Physical>::from((self.width as i32, self.height as i32));
        let logical = Rectangle::from_size(self.logical_size());
        let damage = [Rectangle::from_size(physical)];

        let elements = self.space.render_elements_for_region(
            &mut self.renderer,
            &logical,
            self.scale as f64,
            1.0,
        );

        let mut target = self
            .renderer
            .bind(&mut self.image)
            .map_err(|e| DesktopError::Compositor(format!("bind render target failed: {e}")))?;
        let mut frame = self
            .renderer
            .render(&mut target, physical, Transform::Normal)
            .map_err(|e| DesktopError::Compositor(format!("renderer failed: {e}")))?;
        frame
            .clear(Color32F::new(0.05, 0.06, 0.09, 1.0), &damage)
            .map_err(|e| DesktopError::Compositor(format!("clear failed: {e}")))?;
        draw_render_elements(&mut frame, 1.0, &elements, &damage)
            .map_err(|e| DesktopError::Compositor(format!("draw failed: {e}")))?;
        let _ = frame
            .finish()
            .map_err(|e| DesktopError::Compositor(format!("finish failed: {e}")))?;
        drop(target);

        let composed_micros = started.elapsed().as_micros() as u64;
        let stride = self.image.stride();
        let row_bytes = self.width as usize * 4;
        let mut pixels = vec![0u8; row_bytes * self.height as usize];
        let base = unsafe { self.image.data() };
        if base.is_null() {
            return Err(DesktopError::Compositor(
                "pixman image has no backing store".to_string(),
            ));
        }
        for y in 0..self.height as usize {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (base as *const u8).add(y * stride),
                    pixels.as_mut_ptr().add(y * row_bytes),
                    row_bytes,
                );
            }
        }

        self.frame_counter = self.frame_counter.wrapping_add(1);
        let frame_id = self.frame_counter;
        let timestamp_us = Self::now_us();
        let frame = match self.pool.write_frame(
            frame_id,
            self.generation,
            timestamp_us,
            self.width,
            self.height,
            &pixels,
        ) {
            Ok(frame) => frame,
            Err(err) => {
                // Every slot is held by a slow reader: drop this frame rather
                // than blocking the seat or growing the pool.
                self.frames_dropped = self.frames_dropped.wrapping_add(1);
                return Err(err);
            }
        };
        self.frames_composed = self.frames_composed.wrapping_add(1);
        self.compose_micros = composed_micros;

        Ok(SnapshotInfo {
            frame_id,
            generation: self.generation,
            timestamp_us,
            width: self.width,
            height: self.height,
            slot: frame.slot,
            ticks: self.ticks,
            frames_composed: self.frames_composed,
            frames_dropped: self.frames_dropped,
            compose_micros: self.compose_micros,
        })
    }

    fn flush_snapshots(&mut self) {
        if self.pending_snapshots.is_empty() {
            return;
        }
        let result = self.render_frame();
        let payload = match &result {
            Ok(info) => Ok(*info),
            Err(err) => Err(err.to_string()),
        };
        for reply in self.pending_snapshots.drain(..) {
            let _ = reply.send(payload.clone());
        }
    }

    fn send_frame_callbacks(&self) {
        let time = self.start_time.elapsed().as_millis() as u32;
        for mapped in self.windows.values() {
            if let Some(toplevel) = mapped.window.toplevel() {
                send_frames_surface_tree(toplevel.wl_surface(), time);
            }
        }
    }

    fn tick(&mut self) {
        self.ticks = self.ticks.wrapping_add(1);
        if self.dirty {
            if let Err(err) = self.render_frame() {
                tracing::warn!(error = %err, "composition failed");
            }
            self.dirty = false;
        }
        self.flush_snapshots();
        self.send_frame_callbacks();
        self.flush();
    }

    /// Pushes queued events to clients.
    ///
    /// The display socket only becomes readable when a client sends something,
    /// so events addressed to an idle client (frame callbacks, text input) must
    /// be flushed explicitly rather than waiting for the next request.
    fn flush(&mut self) {
        if let Some(mut display) = self.display.take() {
            if let Err(err) = display.flush_clients() {
                tracing::warn!(error = %err, "wayland flush failed");
            }
            self.display = Some(display);
        }
    }

    fn apply_input(&mut self, event: InputEvent) {
        let time = self.start_time.elapsed().as_millis() as u32;
        let serial = SERIAL_COUNTER.next_serial();
        let pointer = self.pointer.clone();
        let keyboard = self.keyboard.clone();
        match event {
            InputEvent::PointerMoveAbs { x, y } => {
                let point = Point::<f64, Logical>::from((
                    x as f64 / self.scale.max(1) as f64,
                    y as f64 / self.scale.max(1) as f64,
                ));
                let focus = self.space.element_under(point).and_then(|(window, loc)| {
                    window
                        .wl_surface()
                        .map(|s| (s.as_ref().clone(), loc.to_f64()))
                });
                pointer.motion(
                    self,
                    focus,
                    &MotionEvent {
                        location: point,
                        serial,
                        time,
                    },
                );
                pointer.frame(self);
                self.dirty = true;
            }
            InputEvent::PointerDelta { dx, dy } => {
                let current = self.pointer.current_location();
                let point =
                    Point::<f64, Logical>::from((current.x + dx as f64, current.y + dy as f64));
                let focus = self.space.element_under(point).and_then(|(window, loc)| {
                    window
                        .wl_surface()
                        .map(|s| (s.as_ref().clone(), loc.to_f64()))
                });
                pointer.motion(
                    self,
                    focus,
                    &MotionEvent {
                        location: point,
                        serial,
                        time,
                    },
                );
                pointer.frame(self);
                self.dirty = true;
            }
            InputEvent::PointerButton { button, pressed } => {
                let state = if pressed {
                    ButtonState::Pressed
                } else {
                    ButtonState::Released
                };
                pointer.button(
                    self,
                    &ButtonEvent {
                        serial,
                        time,
                        button,
                        state,
                    },
                );
                pointer.frame(self);
                self.dirty = true;
            }
            InputEvent::PointerScroll { axis, value } => {
                let axis = if axis == 0 {
                    Axis::Horizontal
                } else {
                    Axis::Vertical
                };
                let frame = AxisFrame::new(time).value(axis, value as f64);
                pointer.axis(self, frame);
                pointer.frame(self);
                self.dirty = true;
            }
            InputEvent::Key { keycode, pressed } => {
                let state = if pressed {
                    KeyState::Pressed
                } else {
                    KeyState::Released
                };
                keyboard.input::<(), _>(
                    self,
                    Keycode::new(keycode),
                    state,
                    serial,
                    time,
                    |_, _, _| FilterResult::Forward,
                );
                self.dirty = true;
            }
            InputEvent::TextCommit { text, .. } => {
                self.text_input_manager_state.commit_string(&text);
                self.dirty = true;
            }
            InputEvent::TextPreedit {
                text,
                cursor_begin,
                cursor_end,
                ..
            } => {
                self.text_input_manager_state
                    .preedit_string(&text, cursor_begin, cursor_end);
                self.dirty = true;
            }
        }
    }
}

fn toplevel_identity(toplevel: &ToplevelSurface) -> (Option<String>, Option<String>) {
    with_states(toplevel.wl_surface(), |states| match states
        .data_map
        .get::<XdgToplevelSurfaceData>()
    {
        Some(data) => {
            let data = data.lock().unwrap();
            (data.title.clone(), data.app_id.clone())
        }
        None => (None, None),
    })
}

fn send_frames_surface_tree(surface: &WlSurface, time: u32) {
    with_surface_tree_downward(
        surface,
        (),
        |_, _, &()| TraversalAction::DoChildren(()),
        |_surf, states, &()| {
            for callback in states
                .cached_state
                .get::<SurfaceAttributes>()
                .current()
                .frame_callbacks
                .drain(..)
            {
                callback.done(time);
            }
        },
        |_, _, &()| true,
    );
}

impl BufferHandler for DesktopState {
    fn buffer_destroyed(&mut self, _buffer: &wl_buffer::WlBuffer) {}
}

impl CompositorHandler for DesktopState {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }

    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client.get_data::<ClientState>().unwrap().compositor_state
    }

    fn commit(&mut self, surface: &WlSurface) {
        on_commit_buffer_handler::<Self>(surface);
        let mut updated: Vec<u32> = Vec::new();
        for (id, mapped) in self.windows.iter() {
            let mut belongs = false;
            mapped.window.with_surfaces(|s, _| {
                if s == surface {
                    belongs = true;
                }
            });
            if belongs {
                mapped.window.on_commit();
                updated.push(*id);
            }
        }
        for id in updated {
            self.refresh_window_info(id);
        }
        self.dirty = true;
    }
}

impl ShmHandler for DesktopState {
    fn shm_state(&self) -> &ShmState {
        &self.shm_state
    }
}

impl SeatHandler for DesktopState {
    type KeyboardFocus = WlSurface;
    type PointerFocus = WlSurface;
    type TouchFocus = WlSurface;

    fn seat_state(&mut self) -> &mut SeatState<Self> {
        &mut self.seat_state
    }

    fn focus_changed(&mut self, _seat: &Seat<Self>, focused: Option<&WlSurface>) {
        self.text_input_manager_state.set_focus(focused.cloned());
    }
    fn cursor_image(
        &mut self,
        _seat: &Seat<Self>,
        _image: smithay::input::pointer::CursorImageStatus,
    ) {
    }
}

impl OutputHandler for DesktopState {
    fn output_bound(
        &mut self,
        _output: Output,
        _wl_output: smithay::reexports::wayland_server::protocol::wl_output::WlOutput,
    ) {
    }
}

impl SelectionHandler for DesktopState {
    type SelectionUserData = ();
}

impl DataDeviceHandler for DesktopState {
    fn data_device_state(&self) -> &DataDeviceState {
        &self.data_device_state
    }
}

impl ClientDndGrabHandler for DesktopState {}
impl ServerDndGrabHandler for DesktopState {
    fn send(&mut self, _mime_type: String, _fd: std::os::unix::io::OwnedFd, _seat: Seat<Self>) {}
}

impl XdgShellHandler for DesktopState {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.xdg_shell_state
    }

    fn new_toplevel(&mut self, surface: ToplevelSurface) {
        let size = self.logical_size();
        surface.with_pending_state(|state| {
            state.states.set(xdg_toplevel::State::Activated);
            state.size = Some(size);
        });
        surface.send_configure();
        self.map_toplevel(surface);
    }

    fn new_popup(&mut self, _surface: PopupSurface, _positioner: PositionerState) {}

    fn grab(
        &mut self,
        _surface: PopupSurface,
        _seat: smithay::reexports::wayland_server::protocol::wl_seat::WlSeat,
        _serial: Serial,
    ) {
    }

    fn reposition_request(
        &mut self,
        _surface: PopupSurface,
        _positioner: PositionerState,
        _token: u32,
    ) {
    }

    fn toplevel_destroyed(&mut self, surface: ToplevelSurface) {
        if let Some(id) = self.window_id_for_surface(surface.wl_surface()) {
            self.unmap_window(id);
        }
    }

    fn maximize_request(&mut self, surface: ToplevelSurface) {
        let size = self.logical_size();
        surface.with_pending_state(|state| {
            state.states.set(xdg_toplevel::State::Maximized);
            state.size = Some(size);
        });
        surface.send_configure();
        if let Some(id) = self.window_id_for_surface(surface.wl_surface())
            && let Some(mapped) = self.windows.get(&id)
        {
            let window = mapped.window.clone();
            self.space.map_element(window, Point::from((0, 0)), true);
        }
        self.dirty = true;
    }

    fn unmaximize_request(&mut self, surface: ToplevelSurface) {
        surface.with_pending_state(|state| {
            state.states.unset(xdg_toplevel::State::Maximized);
        });
        surface.send_configure();
        self.dirty = true;
    }

    fn fullscreen_request(
        &mut self,
        surface: ToplevelSurface,
        _output: Option<smithay::reexports::wayland_server::protocol::wl_output::WlOutput>,
    ) {
        let size = self.logical_size();
        surface.with_pending_state(|state| {
            state.states.set(xdg_toplevel::State::Fullscreen);
            state.size = Some(size);
        });
        surface.send_configure();
        if let Some(id) = self.window_id_for_surface(surface.wl_surface())
            && let Some(mapped) = self.windows.get(&id)
        {
            let window = mapped.window.clone();
            self.space.map_element(window, Point::from((0, 0)), true);
        }
        self.dirty = true;
    }

    fn unfullscreen_request(&mut self, surface: ToplevelSurface) {
        surface.with_pending_state(|state| {
            state.states.unset(xdg_toplevel::State::Fullscreen);
        });
        surface.send_configure();
        self.dirty = true;
    }

    fn title_changed(&mut self, surface: ToplevelSurface) {
        if let Some(id) = self.window_id_for_surface(surface.wl_surface()) {
            self.refresh_window_info(id);
        }
    }
}

smithay::delegate_compositor!(DesktopState);
smithay::delegate_shm!(DesktopState);
smithay::delegate_seat!(DesktopState);
smithay::delegate_xdg_shell!(DesktopState);
smithay::delegate_data_device!(DesktopState);
smithay::delegate_output!(DesktopState);
impl GlobalDispatch<ZwpTextInputManagerV3, (), DesktopState> for DesktopState {
    fn bind(
        _state: &mut DesktopState,
        _handle: &smithay::reexports::wayland_server::DisplayHandle,
        _client: &Client,
        resource: New<ZwpTextInputManagerV3>,
        _global_data: &(),
        data_init: &mut DataInit<'_, DesktopState>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<ZwpTextInputManagerV3, (), DesktopState> for DesktopState {
    fn request(
        state: &mut DesktopState,
        _client: &Client,
        _resource: &ZwpTextInputManagerV3,
        request: zwp_text_input_manager_v3::Request,
        _data: &(),
        _handle: &smithay::reexports::wayland_server::DisplayHandle,
        data_init: &mut DataInit<'_, DesktopState>,
    ) {
        if let zwp_text_input_manager_v3::Request::GetTextInput { id, .. } = request {
            let instance = data_init.init(id, TextInputUserData::default());
            state.text_input_manager_state.add_instance(instance);
        }
    }
}

impl Dispatch<ZwpTextInputV3, TextInputUserData, DesktopState> for DesktopState {
    fn request(
        _state: &mut DesktopState,
        _client: &Client,
        resource: &ZwpTextInputV3,
        request: zwp_text_input_v3::Request,
        _data: &TextInputUserData,
        _handle: &smithay::reexports::wayland_server::DisplayHandle,
        _data_init: &mut DataInit<'_, DesktopState>,
    ) {
        let Some(data) = resource.data::<TextInputUserData>() else {
            return;
        };
        match request {
            zwp_text_input_v3::Request::Enable => data.set_enabled(true),
            zwp_text_input_v3::Request::Disable => data.set_enabled(false),
            zwp_text_input_v3::Request::Commit => {
                data.next_serial();
            }
            _ => {}
        }
    }
}

impl AsMut<CompositorState> for DesktopState {
    fn as_mut(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }
}

impl AsMut<ShmState> for DesktopState {
    fn as_mut(&mut self) -> &mut ShmState {
        &mut self.shm_state
    }
}

impl AsMut<XdgShellState> for DesktopState {
    fn as_mut(&mut self) -> &mut XdgShellState {
        &mut self.xdg_shell_state
    }
}

impl AsMut<DataDeviceState> for DesktopState {
    fn as_mut(&mut self) -> &mut DataDeviceState {
        &mut self.data_device_state
    }
}

impl AsMut<OutputManagerState> for DesktopState {
    fn as_mut(&mut self) -> &mut OutputManagerState {
        &mut self.output_manager_state
    }
}

impl AsMut<TextInputManagerState> for DesktopState {
    fn as_mut(&mut self) -> &mut TextInputManagerState {
        &mut self.text_input_manager_state
    }
}

impl AsMut<SeatState<DesktopState>> for DesktopState {
    fn as_mut(&mut self) -> &mut SeatState<DesktopState> {
        &mut self.seat_state
    }
}

fn run_compositor(
    config: CompositorConfig,
    pool: Arc<ShmFramePool>,
    events: std_mpsc::Sender<DesktopEvent>,
    cmd_rx: Channel<CompositorCommand>,
    ready_tx: &std_mpsc::SyncSender<Result<(), String>>,
) -> Result<(), DesktopError> {
    let mut display: Display<DesktopState> = Display::new()
        .map_err(|e| DesktopError::Compositor(format!("failed to create wayland display: {e}")))?;
    let dh = display.handle();

    let compositor_state = CompositorState::new::<DesktopState>(&dh);
    let xdg_shell_state = XdgShellState::new::<DesktopState>(&dh);
    let shm_state = ShmState::new::<DesktopState>(&dh, vec![]);
    let data_device_state = DataDeviceState::new::<DesktopState>(&dh);
    let output_manager_state = OutputManagerState::new_with_xdg_output::<DesktopState>(&dh);
    let text_input_manager_state = TextInputManagerState::default();
    dh.create_global::<DesktopState, ZwpTextInputManagerV3, _>(1, ());

    let mut seat_state = SeatState::new();
    let mut seat = seat_state.new_wl_seat(&dh, "juto-seat");
    let keyboard = seat
        .add_keyboard(XkbConfig::default(), 200, 25)
        .map_err(|e| DesktopError::Compositor(format!("failed to add keyboard: {e}")))?;
    let pointer = seat.add_pointer();

    let output = Output::new(
        "juto-virtual-0".to_string(),
        PhysicalProperties {
            size: (0, 0).into(),
            subpixel: Subpixel::Unknown,
            make: "Juto".to_string(),
            model: "Virtual Output".to_string(),
        },
    );
    let mode = Mode {
        size: (config.width as i32, config.height as i32).into(),
        refresh: (config.fps.max(1) as i32) * 1000,
    };
    output.set_preferred(mode);
    output.change_current_state(
        Some(mode),
        Some(Transform::Normal),
        Some(Scale::Integer(config.scale.max(1) as i32)),
        Some(Point::from((0, 0))),
    );
    output.create_global::<DesktopState>(&dh);

    let mut space = Space::<Window>::default();
    space.map_output(&output, Point::from((0, 0)));

    let renderer = PixmanRenderer::new()
        .map_err(|e| DesktopError::Compositor(format!("failed to create pixman renderer: {e}")))?;
    let image = smithay::reexports::pixman::Image::new(
        smithay::reexports::pixman::FormatCode::A8R8G8B8,
        config.width as usize,
        config.height as usize,
        false,
    )
    .map_err(|_| DesktopError::Compositor("failed to allocate render image".to_string()))?;

    let listener = ListeningSocket::bind_absolute(config.socket_path.clone())
        .map_err(|e| DesktopError::Compositor(format!("failed to bind wayland socket: {e}")))?;

    let mut event_loop: EventLoop<'static, DesktopState> = EventLoop::try_new()
        .map_err(|e| DesktopError::Compositor(format!("failed to create event loop: {e}")))?;
    let signal = event_loop.get_signal();

    let mut state = DesktopState {
        display: None,
        signal: signal.clone(),
        compositor_state,
        xdg_shell_state,
        shm_state,
        seat_state,
        data_device_state,
        output_manager_state,
        text_input_manager_state,
        space,
        output,
        _seat: seat,
        keyboard,
        pointer,
        renderer,
        image,
        pool,
        events,
        env_id: config.env_id.clone(),
        width: config.width,
        height: config.height,
        scale: config.scale.max(1),
        fps: config.fps.max(1),
        generation: 1,
        frame_counter: 0,
        ticks: 0,
        frames_composed: 0,
        frames_dropped: 0,
        compose_micros: 0,
        dirty: true,
        start_time: Instant::now(),
        windows: HashMap::new(),
        next_window_id: 1,
        cascade_index: 0,
        pending_snapshots: Vec::new(),
    };

    let handle = event_loop.handle();

    // Wayland display socket: accept clients and dispatch their requests.
    let display_fd = display.backend().poll_fd().as_raw_fd();
    let display_source = Generic::new(
        unsafe { BorrowedFd::borrow_raw(display_fd) },
        Interest::READ,
        CalloopMode::Level,
    );
    handle
        .insert_source(display_source, move |_, _, state: &mut DesktopState| {
            let mut display = state
                .display
                .take()
                .expect("wayland display present during dispatch");
            if let Err(err) = display.dispatch_clients(state) {
                tracing::warn!(error = %err, "wayland dispatch failed");
            }
            if let Err(err) = display.flush_clients() {
                tracing::warn!(error = %err, "wayland flush failed");
            }
            state.display = Some(display);
            Ok(PostAction::Continue)
        })
        .map_err(|e| DesktopError::Compositor(format!("failed to register display source: {e}")))?;

    // Client connections.
    let listener_source = Generic::new(listener, Interest::READ, CalloopMode::Level);
    handle
        .insert_source(
            listener_source,
            move |_, listener, state: &mut DesktopState| {
                loop {
                    match listener.accept() {
                        Ok(Some(stream)) => {
                            let display = state
                                .display
                                .take()
                                .expect("wayland display present during accept");
                            match display
                                .handle()
                                .insert_client(stream, Arc::new(ClientState::default()))
                            {
                                Ok(_) => {}
                                Err(err) => {
                                    tracing::warn!(error = %err, "failed to insert wayland client")
                                }
                            }
                            state.display = Some(display);
                        }
                        Ok(None) => break,
                        Err(err) => {
                            tracing::warn!(error = %err, "failed to accept wayland client");
                            break;
                        }
                    }
                }
                Ok(PostAction::Continue)
            },
        )
        .map_err(|e| {
            DesktopError::Compositor(format!("failed to register listener source: {e}"))
        })?;

    // Control commands.
    handle
        .insert_source(cmd_rx, |event, _, state: &mut DesktopState| {
            if let ChannelEvent::Msg(command) = event {
                match command {
                    CompositorCommand::Input(input) => {
                        state.apply_input(input);
                        state.flush();
                    }
                    CompositorCommand::Configure {
                        width,
                        height,
                        scale,
                        fps,
                        generation,
                        reply,
                    } => {
                        let result = reconfigure(state, width, height, scale, fps, generation)
                            .map_err(|err| err.to_string());
                        let _ = reply.send(result);
                    }
                    CompositorCommand::Snapshot { reply } => {
                        state.pending_snapshots.push(reply);
                    }
                    CompositorCommand::Shutdown => state.signal.stop(),
                }
            }
        })
        .map_err(|e| DesktopError::Compositor(format!("failed to register command source: {e}")))?;

    // Virtual clock.
    let interval = Duration::from_micros(1_000_000 / state.fps.max(1) as u64);
    let timer = Timer::from_duration(interval);
    handle
        .insert_source(timer, |_, _, state: &mut DesktopState| {
            state.tick();
            TimeoutAction::ToDuration(Duration::from_micros(1_000_000 / state.fps.max(1) as u64))
        })
        .map_err(|e| DesktopError::Compositor(format!("failed to register frame timer: {e}")))?;

    state.display = Some(display);

    let _ = ready_tx.try_send(Ok(()));

    event_loop
        .run(None, &mut state, |_state| {})
        .map_err(|e| DesktopError::Compositor(format!("event loop failed: {e}")))?;

    Ok(())
}

fn reconfigure(
    state: &mut DesktopState,
    width: u32,
    height: u32,
    scale: u32,
    fps: u32,
    generation: u64,
) -> Result<(), DesktopError> {
    let scale = scale.max(1);
    let image = smithay::reexports::pixman::Image::new(
        smithay::reexports::pixman::FormatCode::A8R8G8B8,
        width as usize,
        height as usize,
        false,
    )
    .map_err(|_| DesktopError::Compositor("failed to allocate render image".to_string()))?;

    state.image = image;
    state.width = width;
    state.height = height;
    state.scale = scale;
    state.fps = fps.max(1);
    state.generation = generation;

    let mode = Mode {
        size: (width as i32, height as i32).into(),
        refresh: (state.fps as i32) * 1000,
    };
    state.output.set_preferred(mode);
    state.output.change_current_state(
        Some(mode),
        Some(Transform::Normal),
        Some(Scale::Integer(scale as i32)),
        None,
    );

    let size = state.logical_size();
    for mapped in state.windows.values() {
        if let Some(toplevel) = mapped.window.toplevel() {
            toplevel.with_pending_state(|pending| {
                pending.size = Some(size);
            });
            toplevel.send_configure();
        }
    }

    state.dirty = true;
    let _ = state.events.send(DesktopEvent::OutputChanged {
        env_id: state.env_id.clone(),
        generation: state.generation,
        output: crate::supervisor::OutputConfig {
            width,
            height,
            scale,
            fps: state.fps,
        },
    });
    Ok(())
}
