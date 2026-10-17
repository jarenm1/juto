//! Shared test support: a deterministic Wayland client used by both the
//! integration tests and the `test_client` example.
//!
//! Each consumer uses a subset of this module, so unused items are expected.
#![allow(dead_code)]

use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant};

use wayland_client::globals::{GlobalListContents, registry_queue_init};
use wayland_client::protocol::{
    wl_buffer, wl_callback, wl_compositor, wl_pointer, wl_registry, wl_seat, wl_shm, wl_shm_pool,
    wl_surface,
};
use wayland_client::{Connection, Dispatch, EventQueue, Proxy, QueueHandle, WEnum};
use wayland_protocols::wp::cursor_shape::v1::client::{
    wp_cursor_shape_device_v1, wp_cursor_shape_manager_v1,
};
use wayland_protocols::wp::text_input::zv3::client::{
    zwp_text_input_manager_v3, zwp_text_input_v3,
};
use wayland_protocols::xdg::shell::client::{
    xdg_popup, xdg_positioner, xdg_surface, xdg_toplevel, xdg_wm_base,
};

/// Width of the test client's toplevel in pixels.
pub const WIDTH: u32 = 320;
/// Height of the test client's toplevel in pixels.
pub const HEIGHT: u32 = 240;

/// Opaque red, ARGB8888 little-endian (`[B, G, R, A]` in memory).
#[allow(dead_code)]
pub const RED: [u8; 4] = [0x00, 0x00, 0xFF, 0xFF];
/// Opaque blue, ARGB8888 little-endian (`[B, G, R, A]` in memory).
#[allow(dead_code)]
pub const BLUE: [u8; 4] = [0xFF, 0x00, 0x00, 0xFF];
/// Popup edge length in pixels.
pub const POPUP_SIZE: u32 = 64;
/// Popup anchor offset from the toplevel origin, in pixels.
pub const POPUP_OFFSET: i32 = 100;
/// Opaque green, ARGB8888 little-endian (`[B, G, R, A]` in memory).
#[allow(dead_code)]
pub const GREEN: [u8; 4] = [0x00, 0xFF, 0x00, 0xFF];
/// Client cursor surface edge length in pixels.
pub const CURSOR_SIZE: u32 = 16;
/// Client cursor hotspot offset from its top-left corner, in pixels.
pub const CURSOR_HOTSPOT: i32 = 3;

/// Which popup the test client creates.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PopupMode {
    /// No popup.
    None,
    /// A popup mapped at startup with no grab.
    Mapped,
    /// A menu: created on a press over the toplevel and grabbed with that press's
    /// serial before its first buffer commit, as xdg-shell requires.
    OnPress,
}

/// Cursor the test client requests on every pointer enter over the toplevel.
#[derive(Clone, Copy)]
enum CursorRequest {
    /// A named `wp_cursor_shape_v1` shape.
    Shape(wp_cursor_shape_device_v1::Shape),
    /// A client-drawn `CURSOR_SIZE` square cursor surface.
    Bitmap,
}

#[derive(Default)]
struct State {
    configured: bool,
    frames: u32,
    shm_format: Option<wl_shm::Format>,
    /// Text committed to the focused text-input instance.
    committed_text: String,
    text_enabled: bool,
    /// Pointer created once the seat advertises the pointer capability.
    pointer: Option<wl_pointer::WlPointer>,
    /// Surface the pointer is over, from the latest enter/leave pair.
    hovered: Option<wl_surface::WlSurface>,
    /// Serial and surface of the latest button press.
    press: Option<(u32, wl_surface::WlSurface)>,
    /// Set when the compositor dismisses a popup with `popup_done`.
    popup_dismissed: bool,
    /// Cursor shape manager, bound at startup.
    cursor_manager: Option<wp_cursor_shape_manager_v1::WpCursorShapeManagerV1>,
    /// Shape device for the pointer, created on the first enter.
    cursor_device: Option<wp_cursor_shape_device_v1::WpCursorShapeDeviceV1>,
    /// Cursor requested on every pointer enter, when the test asks for one.
    cursor_request: Option<CursorRequest>,
    /// Red cursor surface attached on enter when the request is `Bitmap`.
    cursor_surface: Option<wl_surface::WlSurface>,
}

macro_rules! ignore_events {
    ($($iface:ty => $udata:ty),* $(,)?) => {
        $(
            impl Dispatch<$iface, $udata> for State {
                fn event(
                    _state: &mut Self,
                    _proxy: &$iface,
                    _event: <$iface as Proxy>::Event,
                    _data: &$udata,
                    _conn: &Connection,
                    _qh: &QueueHandle<Self>,
                ) {
                }
            }
        )*
    };
}

ignore_events! {
    wl_compositor::WlCompositor => (),
    wl_shm_pool::WlShmPool => (),
    wl_buffer::WlBuffer => (),
    wl_surface::WlSurface => (),
    xdg_wm_base::XdgWmBase => (),
    xdg_toplevel::XdgToplevel => (),
    wl_callback::WlCallback => (),
    zwp_text_input_manager_v3::ZwpTextInputManagerV3 => (),
    xdg_positioner::XdgPositioner => (),
    wp_cursor_shape_manager_v1::WpCursorShapeManagerV1 => (),
    wp_cursor_shape_device_v1::WpCursorShapeDeviceV1 => (),
}

impl Dispatch<wl_seat::WlSeat, ()> for State {
    fn event(
        state: &mut Self,
        seat: &wl_seat::WlSeat,
        event: wl_seat::Event,
        _data: &(),
        _conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_seat::Event::Capabilities {
            capabilities: WEnum::Value(capabilities),
        } = event
            && capabilities.contains(wl_seat::Capability::Pointer)
            && state.pointer.is_none()
        {
            state.pointer = Some(seat.get_pointer(qh, ()));
        }
    }
}

impl Dispatch<wl_pointer::WlPointer, ()> for State {
    fn event(
        state: &mut Self,
        proxy: &wl_pointer::WlPointer,
        event: wl_pointer::Event,
        _data: &(),
        _conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        match event {
            wl_pointer::Event::Enter {
                serial, surface, ..
            } => {
                state.hovered = Some(surface);
                match (
                    state.cursor_request,
                    &state.cursor_manager,
                    &state.cursor_surface,
                ) {
                    (Some(CursorRequest::Bitmap), _, Some(cursor)) => {
                        proxy.set_cursor(serial, Some(cursor), CURSOR_HOTSPOT, CURSOR_HOTSPOT);
                    }
                    (Some(CursorRequest::Shape(shape)), Some(manager), _) => {
                        let device = state
                            .cursor_device
                            .get_or_insert_with(|| manager.get_pointer(proxy, qh, ()));
                        device.set_shape(serial, shape);
                    }
                    _ => {}
                }
            }
            wl_pointer::Event::Leave { surface, .. } => {
                if state.hovered.as_ref() == Some(&surface) {
                    state.hovered = None;
                }
            }
            wl_pointer::Event::Button {
                serial,
                state: WEnum::Value(wl_pointer::ButtonState::Pressed),
                ..
            } => {
                state.press = state.hovered.clone().map(|surface| (serial, surface));
            }
            _ => {}
        }
    }
}

impl Dispatch<xdg_popup::XdgPopup, ()> for State {
    fn event(
        state: &mut Self,
        _proxy: &xdg_popup::XdgPopup,
        event: xdg_popup::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let xdg_popup::Event::PopupDone = event {
            state.popup_dismissed = true;
        }
    }
}

impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for State {
    fn event(
        _state: &mut Self,
        _proxy: &wl_registry::WlRegistry,
        _event: wl_registry::Event,
        _data: &GlobalListContents,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<zwp_text_input_v3::ZwpTextInputV3, ()> for State {
    fn event(
        state: &mut Self,
        proxy: &zwp_text_input_v3::ZwpTextInputV3,
        event: zwp_text_input_v3::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            zwp_text_input_v3::Event::Enter { .. } => {
                // Enable the instance and commit the state so the compositor
                // may send preedit/commit events.
                proxy.enable();
                proxy.set_content_type(
                    zwp_text_input_v3::ContentHint::None,
                    zwp_text_input_v3::ContentPurpose::Normal,
                );
                proxy.commit();
                state.text_enabled = true;
                if let Ok(path) = std::env::var("JUTO_TEST_CLIENT_TEXT_FILE") {
                    let _ = std::fs::write(&path, "ENTER");
                }
            }
            zwp_text_input_v3::Event::Leave { .. } => {
                proxy.disable();
                proxy.commit();
                state.text_enabled = false;
            }
            zwp_text_input_v3::Event::CommitString { text: Some(text) } => {
                state.committed_text.push_str(&text);
                if let Ok(path) = std::env::var("JUTO_TEST_CLIENT_TEXT_FILE") {
                    let _ = std::fs::write(&path, &state.committed_text);
                }
            }
            zwp_text_input_v3::Event::PreeditString { .. }
            | zwp_text_input_v3::Event::DeleteSurroundingText { .. }
            | zwp_text_input_v3::Event::Done { .. } => {}
            _ => {}
        }
    }
}

impl Dispatch<wl_shm::WlShm, ()> for State {
    fn event(
        state: &mut Self,
        _proxy: &wl_shm::WlShm,
        event: wl_shm::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let wl_shm::Event::Format { format } = event {
            state.shm_format = format.into_result().ok();
        }
    }
}

impl Dispatch<xdg_surface::XdgSurface, ()> for State {
    fn event(
        state: &mut Self,
        proxy: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            proxy.ack_configure(serial);
            state.configured = true;
        }
    }
}

fn create_shm_fd(size: usize) -> std::io::Result<OwnedFd> {
    let name = std::ffi::CString::new("juto-test-client-shm").unwrap();
    let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    if unsafe { libc::ftruncate(fd, size as libc::off_t) } != 0 {
        let err = std::io::Error::last_os_error();
        unsafe { libc::close(fd) };
        return Err(err);
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

fn paint(fd: &OwnedFd, width: u32, height: u32) {
    let stride = width * 4;
    let size = (stride * height) as usize;
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd.as_fd().as_raw_fd(),
            0,
        )
    };
    assert!(ptr != libc::MAP_FAILED, "mmap failed");
    let pixels = ptr as *mut u32;
    for y in 0..height {
        for x in 0..width {
            let color: u32 = if x < width / 2 {
                0xFFFF0000 // opaque red
            } else {
                0xFF0000FF // opaque blue
            };
            unsafe { *pixels.add((y * width + x) as usize) = color };
        }
    }
    unsafe { libc::munmap(ptr, size) };
}

/// A child xdg popup: green at rest, blue while hovered, and unmapped once the
/// compositor dismisses it.
struct TestPopup {
    surface: wl_surface::WlSurface,
    xdg_popup: xdg_popup::XdgPopup,
    green: wl_buffer::WlBuffer,
    blue: wl_buffer::WlBuffer,
    shown_blue: bool,
    dismissed: bool,
}

impl TestPopup {
    fn show(&self) {
        let buffer = if self.shown_blue {
            &self.blue
        } else {
            &self.green
        };
        self.surface.attach(Some(buffer), 0, 0);
        self.surface
            .damage(0, 0, POPUP_SIZE as i32, POPUP_SIZE as i32);
        self.surface.commit();
    }

    /// Applies compositor events: dismissal, then the hover color.
    fn sync(&mut self, state: &State) {
        if self.dismissed {
            return;
        }
        if state.popup_dismissed {
            self.dismissed = true;
            self.surface.attach(None, 0, 0);
            self.surface.commit();
            return;
        }
        let hovered = state.hovered.as_ref() == Some(&self.surface);
        if hovered != self.shown_blue {
            self.shown_blue = hovered;
            self.show();
        }
    }
}

/// Creates a child popup of `parent` anchored at (`POPUP_OFFSET`, `POPUP_OFFSET`).
///
/// With `grab`, the popup requests a pointer grab with that serial before its first
/// buffer commit, since xdg-shell rejects grabs on mapped popups. The popup maps
/// with a `GREEN` buffer after the compositor's initial configure.
fn create_popup(
    compositor: &wl_compositor::WlCompositor,
    shm: &wl_shm::WlShm,
    wm_base: &xdg_wm_base::XdgWmBase,
    parent: &xdg_surface::XdgSurface,
    event_queue: &mut EventQueue<State>,
    state: &mut State,
    grab: Option<(&wl_seat::WlSeat, u32)>,
) -> Result<TestPopup, Box<dyn std::error::Error + Send + Sync>> {
    let qh = event_queue.handle();
    let positioner = wm_base.create_positioner(&qh, ());
    positioner.set_size(POPUP_SIZE as i32, POPUP_SIZE as i32);
    positioner.set_anchor_rect(POPUP_OFFSET, POPUP_OFFSET, 1, 1);
    positioner.set_anchor(xdg_positioner::Anchor::TopLeft);
    positioner.set_gravity(xdg_positioner::Gravity::BottomRight);
    let surface = compositor.create_surface(&qh, ());
    let xdg = wm_base.get_xdg_surface(&surface, &qh, ());
    let xdg_popup = xdg.get_popup(Some(parent), &positioner, &qh, ());
    // xdg-shell: a grab must precede the popup's first commit, or it is rejected
    // as already mapped. The initial commit then yields the configure to ack.
    if let Some((seat, serial)) = grab {
        xdg_popup.grab(seat, serial);
    }
    surface.commit();
    event_queue.roundtrip(state)?;

    let popup = TestPopup {
        surface,
        xdg_popup,
        green: popup_buffer(shm, &qh, 0xFF00_FF00)?,
        blue: popup_buffer(shm, &qh, 0xFF00_00FF)?,
        shown_blue: false,
        dismissed: false,
    };
    popup.show();
    Ok(popup)
}

/// A `POPUP_SIZE` square ARGB8888 buffer filled with `color`.
fn popup_buffer(
    shm: &wl_shm::WlShm,
    qh: &QueueHandle<State>,
    color: u32,
) -> Result<wl_buffer::WlBuffer, Box<dyn std::error::Error + Send + Sync>> {
    let stride = POPUP_SIZE * 4;
    let size = (stride * POPUP_SIZE) as usize;
    let fd = create_shm_fd(size)?;
    fill(&fd, POPUP_SIZE, POPUP_SIZE, color);
    let pool = shm.create_pool(fd.as_fd(), size as i32, qh, ());
    Ok(pool.create_buffer(
        0,
        POPUP_SIZE as i32,
        POPUP_SIZE as i32,
        stride as i32,
        wl_shm::Format::Argb8888,
        qh,
        (),
    ))
}

fn fill(fd: &OwnedFd, width: u32, height: u32, color: u32) {
    let size = (width * height * 4) as usize;
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd.as_fd().as_raw_fd(),
            0,
        )
    };
    assert!(ptr != libc::MAP_FAILED, "mmap failed");
    let pixels = ptr as *mut u32;
    for i in 0..(width * height) as usize {
        unsafe { *pixels.add(i) = color };
    }
    unsafe { libc::munmap(ptr, size) };
}

/// Connects to `socket`, maps one toplevel, paints the two-tone pattern, and
/// dispatches events for `seconds`. Returns the number of frames presented.
pub fn run(socket: &str, seconds: u64) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
    run_client(socket, seconds, PopupMode::None, false, None)
}

/// Like [`run`], plus a child xdg popup mapped at startup, anchored at
/// (`POPUP_OFFSET`, `POPUP_OFFSET`). It is green, and blue while hovered.
pub fn run_with_popup(
    socket: &str,
    seconds: u64,
) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
    run_client(socket, seconds, PopupMode::Mapped, false, None)
}

/// Like [`run`], plus a menu popup. The first press over the toplevel creates the
/// popup at (`POPUP_OFFSET`, `POPUP_OFFSET`) with a grab on that press. The popup is
/// green, and blue while hovered. The compositor dismisses it with `popup_done`.
pub fn run_with_menu(
    socket: &str,
    seconds: u64,
) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
    run_client(socket, seconds, PopupMode::OnPress, false, None)
}

/// Like [`run`], plus a toplevel that maximizes about 1.5 s after its first frame
/// and unmaximizes 2 s later. Used to check that unmaximize restores the layout the
/// window had before maximize.
pub fn run_with_maximize_cycle(
    socket: &str,
    seconds: u64,
) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
    run_client(socket, seconds, PopupMode::None, true, None)
}

/// Like [`run`], plus a cursor shape requested on every pointer enter over the toplevel.
pub fn run_with_cursor(
    socket: &str,
    seconds: u64,
    shape: wp_cursor_shape_device_v1::Shape,
) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
    run_client(
        socket,
        seconds,
        PopupMode::None,
        false,
        Some(CursorRequest::Shape(shape)),
    )
}

/// Like [`run`], plus a `CURSOR_SIZE` square red client cursor with hotspot
/// (`CURSOR_HOTSPOT`, `CURSOR_HOTSPOT`) set on every pointer enter over the toplevel.
pub fn run_with_cursor_bitmap(
    socket: &str,
    seconds: u64,
) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
    run_client(
        socket,
        seconds,
        PopupMode::None,
        false,
        Some(CursorRequest::Bitmap),
    )
}

fn run_client(
    socket: &str,
    seconds: u64,
    popup_mode: PopupMode,
    cycle_maximize: bool,
    cursor_request: Option<CursorRequest>,
) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
    let stream = UnixStream::connect(socket)?;
    let conn = Connection::from_socket(stream)?;

    let (globals, mut event_queue): (_, EventQueue<State>) = registry_queue_init(&conn)?;
    let qh = event_queue.handle();

    let compositor: wl_compositor::WlCompositor = globals.bind(&qh, 4..=6, ())?;
    let shm: wl_shm::WlShm = globals.bind(&qh, 1..=1, ())?;
    let wm_base: xdg_wm_base::XdgWmBase = globals.bind(&qh, 1..=6, ())?;
    let seat: wl_seat::WlSeat = globals.bind(&qh, 1..=9, ())?;
    let text_input_manager: zwp_text_input_manager_v3::ZwpTextInputManagerV3 =
        globals.bind(&qh, 1..=1, ())?;
    let cursor_manager: wp_cursor_shape_manager_v1::WpCursorShapeManagerV1 =
        globals.bind(&qh, 1..=2, ())?;
    let text_input = text_input_manager.get_text_input(&seat, &qh, ());

    let surface = compositor.create_surface(&qh, ());
    let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
    let toplevel = xdg_surface.get_toplevel(&qh, ());
    toplevel.set_title("juto-test-client".to_string());
    toplevel.set_app_id("dev.juto.test-client".to_string());
    surface.commit();

    let mut state = State {
        cursor_manager: Some(cursor_manager),
        cursor_request,
        ..State::default()
    };
    event_queue.roundtrip(&mut state)?;

    if matches!(state.cursor_request, Some(CursorRequest::Bitmap)) {
        let cursor = compositor.create_surface(&qh, ());
        let stride = CURSOR_SIZE * 4;
        let size = (stride * CURSOR_SIZE) as usize;
        let fd = create_shm_fd(size)?;
        fill(&fd, CURSOR_SIZE, CURSOR_SIZE, 0xFFFF_0000);
        let pool = shm.create_pool(fd.as_fd(), size as i32, &qh, ());
        let buffer = pool.create_buffer(
            0,
            CURSOR_SIZE as i32,
            CURSOR_SIZE as i32,
            stride as i32,
            wl_shm::Format::Argb8888,
            &qh,
            (),
        );
        cursor.attach(Some(&buffer), 0, 0);
        cursor.damage(0, 0, CURSOR_SIZE as i32, CURSOR_SIZE as i32);
        cursor.commit();
        state.cursor_surface = Some(cursor);
    }
    // Request text input for the surface we are about to map.
    text_input.set_cursor_rectangle(0, 0, 1, 1);
    text_input.commit();

    let stride = WIDTH * 4;
    let size = (stride * HEIGHT) as usize;
    let fd = create_shm_fd(size)?;
    paint(&fd, WIDTH, HEIGHT);
    let pool = shm.create_pool(fd.as_fd(), size as i32, &qh, ());
    let buffer = pool.create_buffer(
        0,
        WIDTH as i32,
        HEIGHT as i32,
        stride as i32,
        wl_shm::Format::Argb8888,
        &qh,
        (),
    );

    surface.attach(Some(&buffer), 0, 0);
    surface.damage(0, 0, WIDTH as i32, HEIGHT as i32);
    surface.commit();

    let mut popup = if popup_mode == PopupMode::Mapped {
        Some(create_popup(
            &compositor,
            &shm,
            &wm_base,
            &xdg_surface,
            &mut event_queue,
            &mut state,
            None,
        )?)
    } else {
        None
    };

    // Maximize/restore script for `cycle_maximize`, timed from the first frame.
    let mut cycle_started: Option<Instant> = None;
    let mut cycle_step = 0u8;

    let deadline = Instant::now() + Duration::from_secs(seconds);
    while Instant::now() < deadline {
        if let Some(guard) = event_queue.prepare_read() {
            let fd = guard.connection_fd().as_raw_fd();
            let mut pollfd = libc::pollfd {
                fd,
                events: libc::POLLIN,
                revents: 0,
            };
            let timeout_ms = 50;
            let ready = unsafe { libc::poll(&mut pollfd, 1, timeout_ms) };
            if ready > 0 {
                let _ = guard.read();
            }
        }
        event_queue.dispatch_pending(&mut state)?;
        if popup.is_none()
            && popup_mode == PopupMode::OnPress
            && state
                .press
                .as_ref()
                .is_some_and(|(_, pressed)| *pressed == surface)
        {
            let (serial, _) = state.press.take().expect("press checked above");
            popup = Some(create_popup(
                &compositor,
                &shm,
                &wm_base,
                &xdg_surface,
                &mut event_queue,
                &mut state,
                Some((&seat, serial)),
            )?);
        }
        if cycle_maximize && state.frames > 0 {
            let started = *cycle_started.get_or_insert_with(Instant::now);
            match cycle_step {
                0 if started.elapsed() >= Duration::from_millis(1500) => {
                    toplevel.set_maximized();
                    cycle_step = 1;
                }
                1 if started.elapsed() >= Duration::from_millis(3500) => {
                    toplevel.unset_maximized();
                    cycle_step = 2;
                }
                _ => {}
            }
        }
        if state.configured {
            state.configured = false;
            surface.attach(Some(&buffer), 0, 0);
            surface.damage(0, 0, WIDTH as i32, HEIGHT as i32);
            surface.commit();
            if let Some(popup) = popup.as_ref().filter(|popup| !popup.dismissed) {
                popup.show();
            }
            state.frames += 1;
        }
        if let Some(popup) = popup.as_mut() {
            popup.sync(&state);
        }
        event_queue.flush()?;
    }

    Ok(state.frames)
}
