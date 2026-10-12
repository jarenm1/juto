//! Shared test support: a deterministic Wayland client used by both the
//! integration tests and the `test_client` example.

use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant};

use wayland_client::globals::{GlobalListContents, registry_queue_init};
use wayland_client::protocol::{
    wl_buffer, wl_callback, wl_compositor, wl_registry, wl_shm, wl_shm_pool, wl_surface,
};
use wayland_client::{Connection, Dispatch, EventQueue, Proxy, QueueHandle};
use wayland_protocols::xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base};

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

#[derive(Default)]
struct State {
    configured: bool,
    frames: u32,
    shm_format: Option<wl_shm::Format>,
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

/// Connects to `socket`, maps one toplevel, paints the two-tone pattern, and
/// dispatches events for `seconds`. Returns the number of frames presented.
pub fn run(socket: &str, seconds: u64) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
    let stream = UnixStream::connect(socket)?;
    let conn = Connection::from_socket(stream)?;

    let (globals, mut event_queue): (_, EventQueue<State>) = registry_queue_init(&conn)?;
    let qh = event_queue.handle();

    let compositor: wl_compositor::WlCompositor = globals.bind(&qh, 4..=6, ())?;
    let shm: wl_shm::WlShm = globals.bind(&qh, 1..=1, ())?;
    let wm_base: xdg_wm_base::XdgWmBase = globals.bind(&qh, 1..=6, ())?;

    let surface = compositor.create_surface(&qh, ());
    let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
    let toplevel = xdg_surface.get_toplevel(&qh, ());
    toplevel.set_title("juto-test-client".to_string());
    toplevel.set_app_id("dev.juto.test-client".to_string());
    surface.commit();

    let mut state = State::default();
    event_queue.roundtrip(&mut state)?;

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
            state.frames += 1;
        }
        event_queue.flush()?;
    }

    Ok(state.frames)
}
