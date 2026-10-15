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
    wl_buffer, wl_callback, wl_compositor, wl_registry, wl_seat, wl_shm, wl_shm_pool, wl_surface,
};
use wayland_client::{Connection, Dispatch, EventQueue, Proxy, QueueHandle};
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

#[derive(Default)]
struct State {
    configured: bool,
    frames: u32,
    shm_format: Option<wl_shm::Format>,
    /// Text committed to the focused text-input instance.
    committed_text: String,
    text_enabled: bool,
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
    wl_seat::WlSeat => (),
    zwp_text_input_manager_v3::ZwpTextInputManagerV3 => (),
    xdg_popup::XdgPopup => (),
    xdg_positioner::XdgPositioner => (),
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

/// Creates a child popup of `parent` anchored at (`POPUP_OFFSET`, `POPUP_OFFSET`)
/// and attaches a `GREEN` buffer after the compositor's initial configure.
fn create_popup(
    compositor: &wl_compositor::WlCompositor,
    shm: &wl_shm::WlShm,
    wm_base: &xdg_wm_base::XdgWmBase,
    parent: &xdg_surface::XdgSurface,
    event_queue: &mut EventQueue<State>,
    state: &mut State,
) -> Result<(wl_surface::WlSurface, wl_buffer::WlBuffer), Box<dyn std::error::Error + Send + Sync>>
{
    let qh = event_queue.handle();
    let positioner = wm_base.create_positioner(&qh, ());
    positioner.set_size(POPUP_SIZE as i32, POPUP_SIZE as i32);
    positioner.set_anchor_rect(POPUP_OFFSET, POPUP_OFFSET, 1, 1);
    positioner.set_anchor(xdg_positioner::Anchor::TopLeft);
    positioner.set_gravity(xdg_positioner::Gravity::BottomRight);
    let surface = compositor.create_surface(&qh, ());
    let xdg = wm_base.get_xdg_surface(&surface, &qh, ());
    let _popup = xdg.get_popup(Some(parent), &positioner, &qh, ());
    surface.commit();
    // The compositor configures the popup on creation; attach only after that.
    event_queue.roundtrip(state)?;

    let stride = POPUP_SIZE * 4;
    let size = (stride * POPUP_SIZE) as usize;
    let fd = create_shm_fd(size)?;
    fill(&fd, POPUP_SIZE, POPUP_SIZE, 0xFF00_FF00);
    let pool = shm.create_pool(fd.as_fd(), size as i32, &qh, ());
    let buffer = pool.create_buffer(
        0,
        POPUP_SIZE as i32,
        POPUP_SIZE as i32,
        stride as i32,
        wl_shm::Format::Argb8888,
        &qh,
        (),
    );
    surface.attach(Some(&buffer), 0, 0);
    surface.damage(0, 0, POPUP_SIZE as i32, POPUP_SIZE as i32);
    surface.commit();
    Ok((surface, buffer))
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
    run_client(socket, seconds, false)
}

/// Like [`run`], plus a child xdg popup anchored at (`POPUP_OFFSET`, `POPUP_OFFSET`)
/// filled with `GREEN`.
pub fn run_with_popup(
    socket: &str,
    seconds: u64,
) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
    run_client(socket, seconds, true)
}

fn run_client(
    socket: &str,
    seconds: u64,
    with_popup: bool,
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
    let text_input = text_input_manager.get_text_input(&seat, &qh, ());

    let surface = compositor.create_surface(&qh, ());
    let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
    let toplevel = xdg_surface.get_toplevel(&qh, ());
    toplevel.set_title("juto-test-client".to_string());
    toplevel.set_app_id("dev.juto.test-client".to_string());
    surface.commit();

    let mut state = State::default();
    event_queue.roundtrip(&mut state)?;
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

    let popup = if with_popup {
        Some(create_popup(
            &compositor,
            &shm,
            &wm_base,
            &xdg_surface,
            &mut event_queue,
            &mut state,
        )?)
    } else {
        None
    };

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
        if state.configured {
            state.configured = false;
            surface.attach(Some(&buffer), 0, 0);
            surface.damage(0, 0, WIDTH as i32, HEIGHT as i32);
            surface.commit();
            if let Some((popup_surface, popup_buffer)) = &popup {
                popup_surface.attach(Some(popup_buffer), 0, 0);
                popup_surface.damage(0, 0, POPUP_SIZE as i32, POPUP_SIZE as i32);
                popup_surface.commit();
            }
            state.frames += 1;
        }
        event_queue.flush()?;
    }

    Ok(state.frames)
}
