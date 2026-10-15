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
use smithay::backend::egl::native::EGLSurfacelessDisplay;
use smithay::backend::egl::{EGLContext, EGLDisplay};
use smithay::backend::input::{Axis, ButtonState, KeyState};
use smithay::backend::renderer::element::Kind;
use smithay::backend::renderer::element::surface::{
    WaylandSurfaceRenderElement, render_elements_from_surface_tree,
};
use smithay::backend::renderer::gles::{GlesRenderer, GlesTexture};
use smithay::backend::renderer::pixman::PixmanRenderer;
use smithay::backend::renderer::utils::{draw_render_elements, on_commit_buffer_handler};
use smithay::backend::renderer::{
    Bind, Color32F, ExportMem, Frame, ImportAll, Offscreen, Renderer,
};
use smithay::desktop::space::Space;
use smithay::desktop::{PopupKind, PopupManager, Window};
use smithay::input::keyboard::{
    FilterResult, KeyboardHandle, KeyboardTarget, Keycode, KeysymHandle, ModifiersState, XkbConfig,
};
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
    Buffer, IsAlive, Logical, Physical, Point, Rectangle, SERIAL_COUNTER, Serial, Size, Transform,
};
use smithay::wayland::buffer::BufferHandler;
use smithay::wayland::compositor::{
    CompositorClientState, CompositorHandler, CompositorState, SurfaceAttributes, TraversalAction,
    with_states, with_surface_tree_downward,
};
use smithay::wayland::output::{OutputHandler, OutputManagerState};
use smithay::wayland::seat::WaylandFocus;
use smithay::wayland::selection::SelectionHandler;
use smithay::wayland::selection::SelectionTarget;
use smithay::wayland::selection::data_device::{
    ClientDndGrabHandler, DataDeviceHandler, DataDeviceState, ServerDndGrabHandler,
};
use smithay::wayland::shell::xdg::{
    PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState,
    XdgToplevelSurfaceData,
};
use smithay::wayland::shm::{ShmHandler, ShmState};
use smithay::wayland::xwayland_shell::{XWaylandShellHandler, XWaylandShellState};
use smithay::xwayland::xwm::{
    Reorder, ResizeEdge, WmWindowProperty, X11Surface, X11Wm, XwmHandler, XwmId,
};
use smithay::xwayland::{XWayland, XWaylandClientData, XWaylandEvent};

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
    /// Renderer profile requested by the environment configuration.
    pub render_profile: crate::supervisor::RenderProfile,
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
    /// Reserve a pool slot so a reader in another process is not overwritten.
    AcquireSlot { slot: usize },
    /// Release a previously reserved pool slot.
    ReleaseSlot { slot: usize },
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

/// Writes an `MIT-MAGIC-COOKIE-1` Xauthority file for the environment's
/// Xwayland instance and returns its path.
///
/// The cookie is generated here and handed to Xwayland and to launched
/// applications, so the X11 socket is not usable by other local users
/// (spec section 5).
fn write_xauthority(runtime_dir: &Path) -> Result<PathBuf, DesktopError> {
    use rand::RngCore;

    let mut cookie = [0u8; 16];
    rand::rng().fill_bytes(&mut cookie);
    let hostname = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|name| name.trim().to_string())
        .unwrap_or_else(|_| "localhost".to_string());

    fn push_field(entry: &mut Vec<u8>, bytes: &[u8]) {
        entry.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
        entry.extend_from_slice(bytes);
    }

    let mut entry: Vec<u8> = Vec::new();
    // FamilyLocal (256), hostname address, any display number, cookie auth.
    entry.extend_from_slice(&256u16.to_be_bytes());
    push_field(&mut entry, hostname.as_bytes());
    push_field(&mut entry, b"");
    push_field(&mut entry, b"MIT-MAGIC-COOKIE-1");
    push_field(&mut entry, &cookie);

    let path = runtime_dir.join("Xauthority");
    std::fs::write(&path, &entry).map_err(|source| DesktopError::Io {
        path: path.clone(),
        source,
    })?;
    Ok(path)
}

/// True when the host exposes at least one DRM render node.
fn has_render_node() -> bool {
    std::fs::read_dir("/dev/dri")
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .any(|entry| entry.file_name().to_string_lossy().starts_with("renderD"))
        })
        .unwrap_or(false)
}

/// True when the deployment declares that EGL must use Mesa software GL.
fn software_gl_declared() -> bool {
    [
        "LIBGL_ALWAYS_SOFTWARE",
        "GALLIUM_DRIVER",
        "MESA_LOADER_DRIVER_OVERRIDE",
    ]
    .iter()
    .any(|key| std::env::var_os(key).is_some())
}

/// The renderer backing the virtual output.
///
/// The software profile uses surfaceless EGL with Mesa software GL, falling back
/// to pixman when EGL is unavailable; the hardware profile requires a render-node
/// EGL device and fails explicitly rather than silently rendering on the CPU.
#[allow(clippy::large_enum_variant)]
enum FrameRenderer {
    Pixman {
        renderer: PixmanRenderer,
        image: smithay::reexports::pixman::Image<'static, 'static>,
    },
    Gl {
        renderer: GlesRenderer,
        texture: GlesTexture,
        buffer_size: Size<i32, Buffer>,
    },
}

impl FrameRenderer {
    fn new(
        profile: crate::supervisor::RenderProfile,
        width: u32,
        height: u32,
    ) -> Result<Self, DesktopError> {
        use crate::supervisor::RenderProfile;
        match profile {
            RenderProfile::Hardware => {
                // The supervisor holds the shared advisory GPU lock for this
                // environment's lifetime, so the default EGL device may be used
                // directly. A render-node/GBM display is not reachable through
                // Smithay's `EGLDisplay` on platforms without
                // `EGL_EXT_device_enumeration`/`EGL_EXT_device_query`.
                let display = unsafe { EGLDisplay::new(EGLSurfacelessDisplay) }.map_err(|e| {
                    DesktopError::RendererUnavailable(format!("hardware EGL display: {e}"))
                })?;
                Self::gl(display, width, height)
            }
            RenderProfile::Software => {
                // The software profile must never touch the host GPU. When a
                // render node exists and the deployment has not declared
                // software GL, render with pixman instead of probing the device.
                if has_render_node() && !software_gl_declared() {
                    tracing::info!(
                        "a render node is present and software GL is not declared; \
                         the software profile renders with pixman"
                    );
                    return Self::pixman(width, height);
                }
                match unsafe { EGLDisplay::new(EGLSurfacelessDisplay) } {
                    Ok(display) => match Self::gl(display, width, height) {
                        Ok(renderer) if renderer.is_software_gl() => Ok(renderer),
                        Ok(renderer) => {
                            tracing::info!(
                                "default EGL device is not a software rasterizer; \
                                 the software profile renders with pixman"
                            );
                            drop(renderer);
                            Self::pixman(width, height)
                        }
                        Err(err) => {
                            tracing::warn!(
                                error = %err,
                                "software GL unavailable; falling back to the pixman renderer"
                            );
                            Self::pixman(width, height)
                        }
                    },
                    Err(err) => {
                        tracing::warn!(
                            error = %err,
                            "surfaceless EGL unavailable; falling back to the pixman renderer"
                        );
                        Self::pixman(width, height)
                    }
                }
            }
        }
    }

    fn gl(display: EGLDisplay, width: u32, height: u32) -> Result<Self, DesktopError> {
        let context = EGLContext::new(&display)
            .map_err(|e| DesktopError::RendererUnavailable(format!("EGL context: {e}")))?;
        let mut renderer = unsafe { GlesRenderer::new(context) }
            .map_err(|e| DesktopError::RendererUnavailable(format!("GL renderer: {e}")))?;
        let buffer_size: Size<i32, Buffer> = Size::from((width as i32, height as i32));
        let texture: GlesTexture = renderer
            .create_buffer(smithay::backend::allocator::Fourcc::Argb8888, buffer_size)
            .map_err(|e| DesktopError::RendererUnavailable(format!("GL target: {e}")))?;
        tracing::info!(width, height, "compositor using the GL renderer");
        Ok(Self::Gl {
            renderer,
            texture,
            buffer_size,
        })
    }

    fn pixman(width: u32, height: u32) -> Result<Self, DesktopError> {
        let renderer = PixmanRenderer::new()
            .map_err(|e| DesktopError::Compositor(format!("pixman renderer: {e}")))?;
        let image = smithay::reexports::pixman::Image::new(
            smithay::reexports::pixman::FormatCode::A8R8G8B8,
            width as usize,
            height as usize,
            false,
        )
        .map_err(|_| DesktopError::Compositor("failed to allocate render image".to_string()))?;
        tracing::info!(width, height, "compositor using the pixman renderer");
        Ok(Self::Pixman { renderer, image })
    }

    /// True when the active EGL device is a software rasterizer.
    fn is_software_gl(&self) -> bool {
        let Self::Gl { renderer, .. } = self else {
            return false;
        };
        use smithay::backend::renderer::gles::ffi;
        unsafe {
            if renderer.egl_context().make_current().is_err() {
                return false;
            }
            let gl = ffi::Gles2::load_with(|symbol| {
                smithay::backend::egl::get_proc_address(symbol) as *const std::os::raw::c_void
            });
            let ptr = gl.GetString(ffi::RENDERER);
            if ptr.is_null() {
                return false;
            }
            let name = std::ffi::CStr::from_ptr(ptr as *const std::os::raw::c_char)
                .to_string_lossy()
                .to_lowercase();
            ["llvmpipe", "softpipe", "swrast", "software rasterizer"]
                .iter()
                .any(|needle| name.contains(needle))
        }
    }

    fn resize(&mut self, width: u32, height: u32) -> Result<(), DesktopError> {
        match self {
            Self::Pixman { image, .. } => {
                *image = smithay::reexports::pixman::Image::new(
                    smithay::reexports::pixman::FormatCode::A8R8G8B8,
                    width as usize,
                    height as usize,
                    false,
                )
                .map_err(|_| {
                    DesktopError::Compositor("failed to allocate render image".to_string())
                })?;
                Ok(())
            }
            Self::Gl {
                renderer,
                texture,
                buffer_size,
            } => {
                let size: Size<i32, Buffer> = Size::from((width as i32, height as i32));
                *texture = renderer
                    .create_buffer(smithay::backend::allocator::Fourcc::Argb8888, size)
                    .map_err(|e| DesktopError::Compositor(format!("GL target: {e}")))?;
                *buffer_size = size;
                Ok(())
            }
        }
    }

    /// Composes the space and returns tightly packed ARGB8888 pixels.
    fn compose(
        &mut self,
        space: &Space<Window>,
        popups: &[(WlSurface, Point<i32, Logical>)],
        logical: &Rectangle<i32, Logical>,
        physical: Size<i32, Physical>,
        scale: u32,
    ) -> Result<Vec<u8>, DesktopError> {
        let damage = [Rectangle::from_size(physical)];
        let clear = Color32F::new(0.05, 0.06, 0.09, 1.0);
        match self {
            Self::Pixman { renderer, image } => {
                let elements =
                    space.render_elements_for_region(renderer, logical, scale as f64, 1.0);
                let popup_elements = popup_render_elements(renderer, popups, scale);
                let mut target = renderer
                    .bind(image)
                    .map_err(|e| DesktopError::Compositor(format!("bind target: {e}")))?;
                let mut frame = renderer
                    .render(&mut target, physical, Transform::Normal)
                    .map_err(|e| DesktopError::Compositor(format!("render: {e}")))?;
                frame
                    .clear(clear, &damage)
                    .map_err(|e| DesktopError::Compositor(format!("clear: {e}")))?;
                draw_render_elements(&mut frame, 1.0, &elements, &damage)
                    .map_err(|e| DesktopError::Compositor(format!("draw: {e}")))?;
                draw_render_elements(&mut frame, 1.0, &popup_elements, &damage)
                    .map_err(|e| DesktopError::Compositor(format!("draw popups: {e}")))?;
                let _ = frame
                    .finish()
                    .map_err(|e| DesktopError::Compositor(format!("finish: {e}")))?;
                drop(target);

                let stride = image.stride();
                let row_bytes = physical.w as usize * 4;
                let mut pixels = vec![0u8; row_bytes * physical.h as usize];
                let base = unsafe { image.data() };
                if base.is_null() {
                    return Err(DesktopError::Compositor(
                        "pixman image has no backing store".to_string(),
                    ));
                }
                for y in 0..physical.h as usize {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (base as *const u8).add(y * stride),
                            pixels.as_mut_ptr().add(y * row_bytes),
                            row_bytes,
                        );
                    }
                }
                Ok(pixels)
            }
            Self::Gl {
                renderer,
                texture,
                buffer_size,
            } => {
                let elements =
                    space.render_elements_for_region(renderer, logical, scale as f64, 1.0);
                let popup_elements = popup_render_elements(renderer, popups, scale);
                {
                    let mut target = renderer
                        .bind(texture)
                        .map_err(|e| DesktopError::Compositor(format!("bind target: {e}")))?;
                    let mut frame = renderer
                        .render(&mut target, physical, Transform::Normal)
                        .map_err(|e| DesktopError::Compositor(format!("render: {e}")))?;
                    frame
                        .clear(clear, &damage)
                        .map_err(|e| DesktopError::Compositor(format!("clear: {e}")))?;
                    draw_render_elements(&mut frame, 1.0, &elements, &damage)
                        .map_err(|e| DesktopError::Compositor(format!("draw: {e}")))?;
                    draw_render_elements(&mut frame, 1.0, &popup_elements, &damage)
                        .map_err(|e| DesktopError::Compositor(format!("draw popups: {e}")))?;
                    let _ = frame
                        .finish()
                        .map_err(|e| DesktopError::Compositor(format!("finish: {e}")))?;
                }
                let target = renderer
                    .bind(texture)
                    .map_err(|e| DesktopError::Compositor(format!("bind target: {e}")))?;
                let mapping = renderer
                    .copy_framebuffer(
                        &target,
                        Rectangle::from_size(*buffer_size),
                        smithay::backend::allocator::Fourcc::Argb8888,
                    )
                    .map_err(|e| DesktopError::Compositor(format!("readback: {e}")))?;
                let bytes = renderer
                    .map_texture(&mapping)
                    .map_err(|e| DesktopError::Compositor(format!("map readback: {e}")))?;
                Ok(bytes.to_vec())
            }
        }
    }
}

/// Render elements for popup surface trees at region-relative logical positions.
fn popup_render_elements<R>(
    renderer: &mut R,
    popups: &[(WlSurface, Point<i32, Logical>)],
    scale: u32,
) -> Vec<WaylandSurfaceRenderElement<R>>
where
    R: Renderer + ImportAll,
    R::TextureId: Clone + 'static,
{
    let mut elements = Vec::new();
    for (surface, loc) in popups {
        let location = Point::<i32, Physical>::from((loc.x * scale as i32, loc.y * scale as i32));
        elements.extend(render_elements_from_surface_tree::<
            R,
            WaylandSurfaceRenderElement<R>,
        >(
            renderer,
            surface,
            location,
            scale as f64,
            1.0,
            Kind::Unspecified,
        ));
    }
    elements
}

smithay::delegate_xwayland_shell!(DesktopState);

impl XWaylandShellHandler for DesktopState {
    fn xwayland_shell_state(&mut self) -> &mut XWaylandShellState {
        &mut self.xwayland_shell_state
    }

    fn surface_associated(&mut self, _xwm: XwmId, _wl_surface: WlSurface, surface: X11Surface) {
        // XWayland binds the wl_surface after mapping, so map-time focus can be a no-op.
        if let Some(window) = self.x11_windows.get(&surface.window_id()).cloned() {
            self.focus_window(&window);
        }
    }
}

impl XwmHandler for DesktopState {
    fn xwm_state(&mut self, _xwm: XwmId) -> &mut X11Wm {
        self.xwm.as_mut().expect("X11 window manager is running")
    }

    fn new_window(&mut self, _xwm: XwmId, _window: X11Surface) {}

    fn new_override_redirect_window(&mut self, _xwm: XwmId, _window: X11Surface) {}

    fn map_window_request(&mut self, _xwm: XwmId, window: X11Surface) {
        if let Err(err) = window.set_mapped(true) {
            tracing::warn!(error = %err, "failed to map an X11 window");
        }
    }

    fn map_window_notify(&mut self, _xwm: XwmId, window: X11Surface) {
        self.map_x11_window(window);
    }

    fn mapped_override_redirect_window(&mut self, _xwm: XwmId, window: X11Surface) {
        self.map_x11_window(window);
    }

    fn unmapped_window(&mut self, _xwm: XwmId, window: X11Surface) {
        self.unmap_x11_window(&window);
    }

    fn destroyed_window(&mut self, _xwm: XwmId, window: X11Surface) {
        self.unmap_x11_window(&window);
    }

    fn configure_request(
        &mut self,
        _xwm: XwmId,
        window: X11Surface,
        x: Option<i32>,
        y: Option<i32>,
        w: Option<u32>,
        h: Option<u32>,
        _reorder: Option<Reorder>,
    ) {
        let mut geometry = window.geometry();
        geometry.loc.x = x.unwrap_or(geometry.loc.x);
        geometry.loc.y = y.unwrap_or(geometry.loc.y);
        geometry.size.w = w.map(|w| w as i32).unwrap_or(geometry.size.w);
        geometry.size.h = h.map(|h| h as i32).unwrap_or(geometry.size.h);
        if let Err(err) = window.configure(geometry) {
            tracing::warn!(error = %err, "failed to configure an X11 window");
        }
    }

    fn configure_notify(
        &mut self,
        _xwm: XwmId,
        window: X11Surface,
        geometry: Rectangle<i32, Logical>,
        _above: Option<u32>,
    ) {
        if let Some(mapped) = self.x11_windows.get(&window.window_id()) {
            self.space.map_element(mapped.clone(), geometry.loc, false);
            self.dirty = true;
        }
    }

    fn property_notify(&mut self, _xwm: XwmId, window: X11Surface, _property: WmWindowProperty) {
        if let Some(mapped) = self.x11_windows.get(&window.window_id())
            && let Some(id) = self
                .windows
                .iter()
                .find_map(|(id, m)| (m.window == *mapped).then_some(*id))
        {
            self.refresh_window_info(id);
        }
    }

    fn resize_request(
        &mut self,
        _xwm: XwmId,
        window: X11Surface,
        button: u32,
        resize_edge: ResizeEdge,
    ) {
        self.start_x11_grab(window, button, X11GrabKind::Resize(resize_edge));
    }

    fn move_request(&mut self, _xwm: XwmId, window: X11Surface, button: u32) {
        self.start_x11_grab(window, button, X11GrabKind::Move);
    }

    fn maximize_request(&mut self, _xwm: XwmId, window: X11Surface) {
        self.set_x11_output_geometry(&window, |surface| {
            if let Err(err) = surface.set_maximized(true) {
                tracing::warn!(error = %err, "failed to mark an X11 window maximized");
            }
        });
    }

    fn unmaximize_request(&mut self, _xwm: XwmId, window: X11Surface) {
        self.restore_x11_geometry(&window, |surface| {
            if let Err(err) = surface.set_maximized(false) {
                tracing::warn!(error = %err, "failed to clear X11 maximized state");
            }
        });
    }

    fn fullscreen_request(&mut self, _xwm: XwmId, window: X11Surface) {
        self.set_x11_output_geometry(&window, |surface| {
            if let Err(err) = surface.set_fullscreen(true) {
                tracing::warn!(error = %err, "failed to mark an X11 window fullscreen");
            }
        });
    }

    fn unfullscreen_request(&mut self, _xwm: XwmId, window: X11Surface) {
        self.restore_x11_geometry(&window, |surface| {
            if let Err(err) = surface.set_fullscreen(false) {
                tracing::warn!(error = %err, "failed to clear X11 fullscreen state");
            }
        });
    }

    fn allow_selection_access(&mut self, _xwm: XwmId, _selection: SelectionTarget) -> bool {
        // Clipboard stays environment-local; X11 selections are not bridged.
        false
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

/// How an X11 interactive move/resize grab changes the window geometry.
enum X11GrabKind {
    Move,
    Resize(ResizeEdge),
}

/// State of an in-progress X11 interactive move or resize grab.
///
/// The X client holds the pointer grab it requested; the compositor tracks the
/// drag in `apply_input` until the grab button is released.
struct X11Grab {
    surface: X11Surface,
    button: u32,
    kind: X11GrabKind,
    /// Pointer position where the grab began.
    start_location: Point<f64, Logical>,
    /// Pointer position at the last delivered absolute pointer event.
    last_location: Point<f64, Logical>,
    /// Window geometry when the grab began.
    start_geometry: Rectangle<i32, Logical>,
}

/// The single-threaded Smithay state owned by the compositor thread.
struct DesktopState {
    display: Option<Display<DesktopState>>,
    signal: LoopSignal,

    compositor_state: CompositorState,
    xdg_shell_state: XdgShellState,
    xwayland_shell_state: XWaylandShellState,
    shm_state: ShmState,
    seat_state: SeatState<Self>,
    data_device_state: DataDeviceState,
    output_manager_state: OutputManagerState,
    text_input_manager_state: TextInputManagerState,

    space: Space<Window>,
    popups: PopupManager,
    output: Output,
    /// Kept alive so the virtual seat stays registered for the environment.
    _seat: Seat<Self>,
    keyboard: KeyboardHandle<Self>,
    pointer: PointerHandle<Self>,

    renderer: FrameRenderer,
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
    /// X11 surfaces mapped through Xwayland, keyed by X11 window id.
    x11_windows: HashMap<u32, Window>,
    /// The X11 window manager driving Xwayland, once it is ready.
    xwm: Option<X11Wm>,
    /// X11 surface geometries saved when a window is maximized or fullscreened,
    /// so unmaximize/unfullscreen restore the previous layout.
    x11_saved_geometries: HashMap<u32, Rectangle<i32, Logical>>,
    /// In-progress X11 interactive move/resize grab, if any.
    x11_grab: Option<X11Grab>,
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
        let (title, app_id) = match mapped.window.x11_surface() {
            Some(x11) => (Some(x11.title()), Some(x11.class())),
            None => mapped
                .window
                .toplevel()
                .map(toplevel_identity)
                .unwrap_or((None, None)),
        };
        let is_focused = match (
            self.keyboard.current_focus(),
            keyboard_target(&mapped.window),
        ) {
            (Some(focus), Some(target)) => focus == target,
            _ => false,
        };
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
        if let Some(target) = keyboard_target(window) {
            let keyboard = self.keyboard.clone();
            keyboard.set_focus(self, Some(target), SERIAL_COUNTER.next_serial());
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

    /// Maps an X11 surface into the composited space and reports it.
    fn map_x11_window(&mut self, surface: X11Surface) {
        let x11_id = surface.window_id();
        if self.x11_windows.contains_key(&x11_id) {
            return;
        }
        let window = Window::new_x11_window(surface.clone());
        let geometry = surface.geometry();
        self.space.map_element(window.clone(), geometry.loc, true);
        self.x11_windows.insert(x11_id, window.clone());
        self.focus_window(&window);
        let id = self.next_window_id;
        self.next_window_id = self.next_window_id.wrapping_add(1);
        let mut mapped = MappedWindow {
            id,
            window,
            info: WindowInfo {
                window_id: id,
                title: surface.title(),
                app_id: surface.class(),
                x: geometry.loc.x,
                y: geometry.loc.y,
                width: geometry.size.w.max(0) as u32,
                height: geometry.size.h.max(0) as u32,
                is_focused: false,
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

    /// Removes an X11 surface from the composited space.
    fn unmap_x11_window(&mut self, surface: &X11Surface) {
        let x11_id = surface.window_id();
        if self
            .x11_grab
            .as_ref()
            .is_some_and(|grab| grab.surface.window_id() == x11_id)
        {
            self.x11_grab = None;
        }
        self.x11_saved_geometries.remove(&x11_id);
        if let Some(window) = self.x11_windows.remove(&x11_id)
            && let Some(id) = self
                .windows
                .iter()
                .find_map(|(id, mapped)| (mapped.window == window).then_some(*id))
        {
            self.unmap_window(id);
        }
    }

    /// Begins an X11 interactive move or resize grab.
    ///
    /// The grabbed window is raised and focused, and subsequent pointer motion
    /// updates its geometry until the grab button is released.
    fn start_x11_grab(&mut self, surface: X11Surface, button: u32, kind: X11GrabKind) {
        let x11_id = surface.window_id();
        let Some(window) = self.x11_windows.get(&x11_id).cloned() else {
            return;
        };
        self.space.raise_element(&window, true);
        self.focus_window(&window);
        let location = self.pointer.current_location();
        let geometry = surface.geometry();
        self.x11_grab = Some(X11Grab {
            surface,
            button,
            kind,
            start_location: location,
            last_location: location,
            start_geometry: geometry,
        });
    }

    /// Applies pointer motion to an in-progress X11 move/resize grab.
    /// Returns `true` when the event was consumed by the grab.
    fn update_x11_grab(&mut self, event: &InputEvent) -> bool {
        let Some(grab) = self.x11_grab.as_mut() else {
            return false;
        };
        let point = match event {
            InputEvent::PointerMoveAbs { x, y } => Point::<f64, Logical>::from((
                *x as f64 / self.scale.max(1) as f64,
                *y as f64 / self.scale.max(1) as f64,
            )),
            InputEvent::PointerDelta { dx, dy } => Point::<f64, Logical>::from((
                grab.last_location.x + *dx as f64,
                grab.last_location.y + *dy as f64,
            )),
            InputEvent::PointerButton { button, pressed } => {
                if *button == grab.button && !*pressed {
                    self.x11_grab = None;
                }
                return true;
            }
            _ => return true,
        };
        grab.last_location = point;
        let delta_x = (point.x - grab.start_location.x) as i32;
        let delta_y = (point.y - grab.start_location.y) as i32;
        let mut geometry = grab.start_geometry;
        match grab.kind {
            X11GrabKind::Move => {
                geometry.loc.x += delta_x;
                geometry.loc.y += delta_y;
            }
            X11GrabKind::Resize(edge) => {
                match edge {
                    ResizeEdge::TopLeft | ResizeEdge::Left | ResizeEdge::BottomLeft => {
                        let applied = delta_x.min(geometry.size.w - 1);
                        geometry.loc.x += applied;
                        geometry.size.w -= applied;
                    }
                    ResizeEdge::TopRight | ResizeEdge::Right | ResizeEdge::BottomRight => {
                        geometry.size.w = (geometry.size.w + delta_x).max(1);
                    }
                    _ => {}
                }
                match edge {
                    ResizeEdge::TopLeft | ResizeEdge::Top | ResizeEdge::TopRight => {
                        let applied = delta_y.min(geometry.size.h - 1);
                        geometry.loc.y += applied;
                        geometry.size.h -= applied;
                    }
                    ResizeEdge::BottomLeft | ResizeEdge::Bottom | ResizeEdge::BottomRight => {
                        geometry.size.h = (geometry.size.h + delta_y).max(1);
                    }
                    _ => {}
                }
            }
        }
        let x11_id = grab.surface.window_id();
        if self.x11_windows.contains_key(&x11_id) {
            if let Err(err) = grab.surface.configure(geometry) {
                tracing::warn!(error = %err, "X11 interactive grab configure failed");
                self.x11_grab = None;
            }
        } else {
            self.x11_grab = None;
        }
        self.dirty = true;
        true
    }

    /// Moves `surface` onto the full output and remembers its layout for
    /// `restore_x11_geometry`. `mark` updates the X11 window state.
    fn set_x11_output_geometry(&mut self, surface: &X11Surface, mark: impl FnOnce(&X11Surface)) {
        let x11_id = surface.window_id();
        self.x11_saved_geometries
            .entry(x11_id)
            .or_insert_with(|| surface.geometry());
        mark(surface);
        let geometry = Rectangle::new(Point::from((0, 0)), self.logical_size());
        if let Err(err) = surface.configure(geometry) {
            tracing::warn!(error = %err, "failed to fullscreen an X11 window");
        }
        if let Some(window) = self.x11_windows.get(&x11_id).cloned() {
            self.space.raise_element(&window, true);
            self.focus_window(&window);
        }
        self.dirty = true;
    }

    /// Restores an X11 window's saved pre-maximize/pre-fullscreen geometry.
    fn restore_x11_geometry(&mut self, surface: &X11Surface, mark: impl FnOnce(&X11Surface)) {
        let x11_id = surface.window_id();
        mark(surface);
        if let Some(geometry) = self.x11_saved_geometries.remove(&x11_id)
            && let Err(err) = surface.configure(geometry)
        {
            tracing::warn!(error = %err, "failed to restore X11 window geometry");
        }
        self.dirty = true;
    }

    fn unmap_window(&mut self, id: u32) {
        if let Some(mapped) = self.windows.remove(&id) {
            let was_focused = match (
                self.keyboard.current_focus(),
                keyboard_target(&mapped.window),
            ) {
                (Some(focus), Some(target)) => focus == target,
                _ => false,
            };
            self.space.unmap_elem(&mapped.window);
            if was_focused {
                self.focus_topmost_window();
            }
            self.dirty = true;
            let _ = self.events.send(DesktopEvent::WindowUnmapped {
                env_id: self.env_id.clone(),
                window_id: id,
            });
        }
    }

    /// Moves keyboard focus to the topmost remaining window, or clears it.
    ///
    /// Smithay keeps keyboard focus on a destroyed surface and drops input
    /// sent to it, so a closed window must hand focus off explicitly.
    fn focus_topmost_window(&mut self) {
        let next = self
            .space
            .elements()
            .rev()
            .find(|window| keyboard_target(window).is_some())
            .cloned();
        match next {
            Some(window) => self.focus_window(&window),
            None => {
                let keyboard = self.keyboard.clone();
                keyboard.set_focus(self, None, SERIAL_COUNTER.next_serial());
            }
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

    /// Surface origins of every mapped window's popups, relative to `region`, in the
    /// same coordinate space as `Space::render_elements_for_region`.
    fn popup_placements(
        &self,
        region: &Rectangle<i32, Logical>,
    ) -> Vec<(WlSurface, Point<i32, Logical>)> {
        let mut placements = Vec::new();
        for window in self.space.elements() {
            let Some(root) = window.wl_surface() else {
                continue;
            };
            let Some(location) = self.space.element_location(window) else {
                continue;
            };
            let origin = location - window.geometry().loc;
            for (popup, offset) in PopupManager::popups_for_surface(&root) {
                let loc = origin + offset - popup.geometry().loc - region.loc;
                placements.push((popup.wl_surface().clone(), loc));
            }
        }
        placements
    }

    /// Compose the desktop into the memory image and copy it into the pool.
    fn render_frame(&mut self) -> Result<SnapshotInfo, DesktopError> {
        let started = Instant::now();
        let physical = Size::<i32, Physical>::from((self.width as i32, self.height as i32));
        let logical = Rectangle::from_size(self.logical_size());

        self.popups.cleanup();
        let popups = self.popup_placements(&logical);
        let pixels = self
            .renderer
            .compose(&self.space, &popups, &logical, physical, self.scale)?;
        let composed_micros = started.elapsed().as_micros() as u64;

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
            // X11 windows have no xdg toplevel, but Xwayland paces its commits
            // on wl_surface frame callbacks, so they need them too.
            if let Some(surface) = mapped.window.wl_surface() {
                send_frames_surface_tree(surface.as_ref(), time);
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
        // An interactive X11 move/resize grab owns pointer traffic until the
        // grab button is released.
        if self.x11_grab.is_some() && self.update_x11_grab(&event) {
            return;
        }
        let time = self.start_time.elapsed().as_millis() as u32;
        let serial = SERIAL_COUNTER.next_serial();
        let pointer = self.pointer.clone();
        let keyboard = self.keyboard.clone();
        tracing::debug!(?event, "input dispatched to virtual seat");
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
                    // Wire keycodes are evdev; the seat expects XKB codes (evdev + 8).
                    Keycode::new(keycode.saturating_add(8)),
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
        // Xwayland's own Wayland client carries `XWaylandClientData`; every
        // other client carries our `ClientState`.
        if let Some(state) = client.get_data::<ClientState>() {
            &state.compositor_state
        } else if let Some(xwayland) = client.get_data::<XWaylandClientData>() {
            &xwayland.compositor_state
        } else {
            panic!("client has no compositor state");
        }
    }

    fn commit(&mut self, surface: &WlSurface) {
        on_commit_buffer_handler::<Self>(surface);
        self.popups.commit(surface);
        if let Some(PopupKind::Xdg(popup)) = self.popups.find_popup(surface)
            && !popup.is_initial_configure_sent()
            && let Err(err) = popup.send_configure()
        {
            tracing::warn!(error = %err, "initial popup configure failed");
        }
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

/// Keyboard focus target. X11 windows focus through `X11Surface`, so its
/// `KeyboardTarget` impl sets X input focus with the seat. A bare `WlSurface`
/// leaves X input focus on PointerRoot, so keys follow the pointer instead.
#[derive(Debug, Clone, PartialEq)]
pub enum KeyboardFocusTarget {
    Wayland(WlSurface),
    X11(X11Surface),
}

impl WaylandFocus for KeyboardFocusTarget {
    fn wl_surface(&self) -> Option<std::borrow::Cow<'_, WlSurface>> {
        match self {
            Self::Wayland(surface) => Some(std::borrow::Cow::Borrowed(surface)),
            Self::X11(surface) => surface.wl_surface().map(std::borrow::Cow::Owned),
        }
    }
}

impl IsAlive for KeyboardFocusTarget {
    fn alive(&self) -> bool {
        match self {
            Self::Wayland(surface) => surface.alive(),
            Self::X11(surface) => surface.alive(),
        }
    }
}

impl KeyboardTarget<DesktopState> for KeyboardFocusTarget {
    fn enter(
        &self,
        seat: &Seat<DesktopState>,
        data: &mut DesktopState,
        keys: Vec<KeysymHandle<'_>>,
        serial: Serial,
    ) {
        match self {
            Self::Wayland(surface) => KeyboardTarget::enter(surface, seat, data, keys, serial),
            Self::X11(surface) => KeyboardTarget::enter(surface, seat, data, keys, serial),
        }
    }

    fn leave(&self, seat: &Seat<DesktopState>, data: &mut DesktopState, serial: Serial) {
        match self {
            Self::Wayland(surface) => KeyboardTarget::leave(surface, seat, data, serial),
            Self::X11(surface) => KeyboardTarget::leave(surface, seat, data, serial),
        }
    }

    fn key(
        &self,
        seat: &Seat<DesktopState>,
        data: &mut DesktopState,
        key: KeysymHandle<'_>,
        state: KeyState,
        serial: Serial,
        time: u32,
    ) {
        match self {
            Self::Wayland(surface) => {
                KeyboardTarget::key(surface, seat, data, key, state, serial, time)
            }
            Self::X11(surface) => {
                KeyboardTarget::key(surface, seat, data, key, state, serial, time)
            }
        }
    }

    fn modifiers(
        &self,
        seat: &Seat<DesktopState>,
        data: &mut DesktopState,
        modifiers: ModifiersState,
        serial: Serial,
    ) {
        match self {
            Self::Wayland(surface) => {
                KeyboardTarget::modifiers(surface, seat, data, modifiers, serial)
            }
            Self::X11(surface) => KeyboardTarget::modifiers(surface, seat, data, modifiers, serial),
        }
    }
}

/// The keyboard target for a window: its X11 surface when it is one.
fn keyboard_target(window: &Window) -> Option<KeyboardFocusTarget> {
    match window.x11_surface() {
        Some(surface) => Some(KeyboardFocusTarget::X11(surface.clone())),
        None => window
            .wl_surface()
            .map(|surface| KeyboardFocusTarget::Wayland(surface.into_owned())),
    }
}

impl SeatHandler for DesktopState {
    type KeyboardFocus = KeyboardFocusTarget;
    type PointerFocus = WlSurface;
    type TouchFocus = WlSurface;

    fn seat_state(&mut self) -> &mut SeatState<Self> {
        &mut self.seat_state
    }

    fn focus_changed(&mut self, _seat: &Seat<Self>, focused: Option<&KeyboardFocusTarget>) {
        self.text_input_manager_state.set_focus(
            focused
                .and_then(WaylandFocus::wl_surface)
                .map(|surface| surface.into_owned()),
        );
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

    fn new_popup(&mut self, surface: PopupSurface, positioner: PositionerState) {
        surface.with_pending_state(|state| {
            state.geometry = positioner.get_geometry();
            state.positioner = positioner;
        });
        if let Err(err) = self.popups.track_popup(PopupKind::from(surface)) {
            tracing::warn!(error = %err, "failed to track popup");
        }
        self.dirty = true;
    }

    fn grab(
        &mut self,
        _surface: PopupSurface,
        _seat: smithay::reexports::wayland_server::protocol::wl_seat::WlSeat,
        _serial: Serial,
    ) {
    }

    fn reposition_request(
        &mut self,
        surface: PopupSurface,
        positioner: PositionerState,
        token: u32,
    ) {
        surface.with_pending_state(|state| {
            state.geometry = positioner.get_geometry();
            state.positioner = positioner;
        });
        surface.send_repositioned(token);
        if let Err(err) = surface.send_configure() {
            tracing::warn!(error = %err, "failed to configure repositioned popup");
        }
        self.dirty = true;
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

pub(crate) fn run_compositor(
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
    let xwayland_shell_state = XWaylandShellState::new::<DesktopState>(&dh);
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

    let renderer = FrameRenderer::new(config.render_profile, config.width, config.height)?;

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
        popups: PopupManager::default(),
        output,
        _seat: seat,
        keyboard,
        pointer,
        renderer,
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
        x11_windows: HashMap::new(),
        xwm: None,
        x11_saved_geometries: HashMap::new(),
        x11_grab: None,
        xwayland_shell_state,
        next_window_id: 1,
        cascade_index: 0,
        pending_snapshots: Vec::new(),
    };

    let handle = event_loop.handle();

    // Xwayland: one private X11 server per environment, driven by our X11 WM.
    // Sockets stay private (no abstract socket, no TCP listener) and Xwayland
    // is supervised by this compositor process.
    let runtime_dir = config
        .socket_path
        .parent()
        .ok_or_else(|| DesktopError::Compositor("socket path has no parent".to_string()))?;
    let xauthority = write_xauthority(runtime_dir)?;
    let (xwayland, xwayland_client) = XWayland::spawn(
        &dh,
        None,
        [("XAUTHORITY", xauthority.as_os_str())],
        false,
        std::process::Stdio::null(),
        std::process::Stdio::null(),
        |_| {},
    )
    .map_err(|e| DesktopError::Compositor(format!("failed to spawn Xwayland: {e}")))?;
    let xwm_handle = handle.clone();
    handle
        .insert_source(
            xwayland,
            move |event, _, state: &mut DesktopState| match event {
                XWaylandEvent::Ready {
                    x11_socket,
                    display_number,
                } => {
                    match X11Wm::start_wm(xwm_handle.clone(), x11_socket, xwayland_client.clone()) {
                        Ok(wm) => {
                            state.xwm = Some(wm);
                            tracing::info!(display_number, "Xwayland ready");
                            let _ = state.events.send(DesktopEvent::XwaylandReady {
                                env_id: state.env_id.clone(),
                                display_number,
                            });
                        }
                        Err(err) => {
                            tracing::warn!(error = %err, "failed to start the X11 window manager")
                        }
                    }
                }
                XWaylandEvent::Error => {
                    tracing::warn!("Xwayland exited during startup");
                }
            },
        )
        .map_err(|e| DesktopError::Compositor(format!("failed to register Xwayland: {e}")))?;

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
                    CompositorCommand::AcquireSlot { slot } => state.pool.acquire_slot(slot),
                    CompositorCommand::ReleaseSlot { slot } => state.pool.release_slot(slot),
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
    state.renderer.resize(width, height)?;
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
