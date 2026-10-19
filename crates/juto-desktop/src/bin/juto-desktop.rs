//! `juto-desktop` daemon: hosts the desktop supervisor and serves the
//! length-prefixed JSON control protocol over a private Unix socket.
//!
//! Usage:
//!   juto-desktop --socket <path>
//!
//! The socket defaults to `$XDG_RUNTIME_DIR/juto-desktop/control.sock`.

use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::sync::Arc;

use juto_desktop::supervisor::DesktopSupervisor;

fn main() {
    tracing_subscriber_init();

    // `juto-desktop worker ...` runs the compositor for one environment inside
    // that environment's slice.
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("worker") {
        if let Err(err) = run_worker_command(&args[1..]) {
            eprintln!("juto-desktop worker: {err}");
            std::process::exit(1);
        }
        return;
    }

    let socket_path = parse_socket_path();
    if let Some(parent) = socket_path.parent()
        && let Err(err) = std::fs::create_dir_all(parent)
    {
        eprintln!("juto-desktop: cannot create {}: {err}", parent.display());
        std::process::exit(1);
    }
    let _ = std::fs::remove_file(&socket_path);

    let listener = match UnixListener::bind(&socket_path) {
        Ok(listener) => listener,
        Err(err) => {
            eprintln!("juto-desktop: cannot bind {}: {err}", socket_path.display());
            std::process::exit(1);
        }
    };
    eprintln!("juto-desktop: listening on {}", socket_path.display());

    let supervisor = Arc::new(DesktopSupervisor::new());
    let report = supervisor.reconcile();
    if !report.stopped_units.is_empty() || !report.removed_directories.is_empty() {
        eprintln!(
            "juto-desktop: reconciled {} orphan scope(s) and {} stale director(ies)",
            report.stopped_units.len(),
            report.removed_directories.len()
        );
    }
    let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));

    {
        let supervisor = supervisor.clone();
        let shutdown = shutdown.clone();
        let socket_path = socket_path.clone();
        ctrlc_handler(move || {
            if shutdown.swap(true, std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            eprintln!("juto-desktop: shutting down");
            supervisor.destroy_all();
            let _ = std::fs::remove_file(&socket_path);
            std::process::exit(0);
        });
    }

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let supervisor = supervisor.clone();
                std::thread::Builder::new()
                    .name("juto-desktop-conn".to_string())
                    .spawn(move || {
                        if let Err(err) = juto_desktop::control::serve(stream, supervisor) {
                            eprintln!("juto-desktop: connection error: {err}");
                        }
                    })
                    .ok();
            }
            Err(err) => eprintln!("juto-desktop: accept failed: {err}"),
        }
    }
}

fn run_worker_command(args: &[String]) -> Result<(), juto_desktop::DesktopError> {
    let mut env_id = None;
    let mut width = 1280u32;
    let mut height = 720u32;
    let mut scale = 1u32;
    let mut fps = 30u32;
    let mut render_profile = juto_desktop::RenderProfile::Software;
    let mut wayland_socket = None;
    let mut control_socket = None;
    let mut index = 0;
    while index < args.len() {
        let value = args.get(index + 1).cloned().unwrap_or_default();
        match args[index].as_str() {
            "--env" => env_id = Some(value),
            "--width" => width = value.parse().unwrap_or(width),
            "--height" => height = value.parse().unwrap_or(height),
            "--scale" => scale = value.parse().unwrap_or(scale),
            "--fps" => fps = value.parse().unwrap_or(fps),
            "--render-profile" => {
                render_profile = value.parse().unwrap_or(render_profile);
            }
            "--wayland-socket" => wayland_socket = Some(PathBuf::from(value)),
            "--control-socket" => control_socket = Some(PathBuf::from(value)),
            _ => {}
        }
        index += 2;
    }
    let (Some(env_id), Some(wayland_socket), Some(control_socket)) =
        (env_id, wayland_socket, control_socket)
    else {
        return Err(juto_desktop::DesktopError::Protocol(
            "worker requires --env, --wayland-socket, and --control-socket".to_string(),
        ));
    };
    juto_desktop::run_worker(juto_desktop::WorkerArgs {
        env_id,
        width,
        height,
        scale,
        fps,
        render_profile,
        wayland_socket,
        control_socket,
    })
}

fn tracing_subscriber_init() {
    // Keep the daemon quiet by default; RUST_LOG enables tracing output.
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();
}

fn parse_socket_path() -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--socket" {
            if let Some(path) = args.next() {
                return PathBuf::from(path);
            }
        } else if let Some(path) = arg.strip_prefix("--socket=") {
            return PathBuf::from(path);
        }
    }
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    base.join("juto-desktop").join("control.sock")
}

/// Minimal SIGINT/SIGTERM handling: the signal handler only writes one byte to
/// a self-pipe (async-signal-safe); a dedicated thread performs the shutdown.
fn ctrlc_handler<F: Fn() + Send + 'static>(handler: F) {
    use std::os::fd::{FromRawFd, OwnedFd};

    let mut fds = [0 as libc::c_int; 2];
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        eprintln!("juto-desktop: cannot create signal pipe; Ctrl-C will not clean up");
        return;
    }
    let read_fd = unsafe { OwnedFd::from_raw_fd(fds[0]) };
    let write_fd = fds[1];

    unsafe extern "C" fn trampoline(sig: libc::c_int) {
        let byte = sig as u8;
        let fd = SIGNAL_PIPE_WRITE.load(std::sync::atomic::Ordering::Relaxed);
        if fd >= 0 {
            unsafe {
                libc::write(fd, &byte as *const u8 as *const libc::c_void, 1);
            }
        }
    }
    unsafe {
        SIGNAL_PIPE_WRITE.store(write_fd, std::sync::atomic::Ordering::Relaxed);
        libc::signal(libc::SIGINT, trampoline as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, trampoline as *const () as libc::sighandler_t);
    }

    std::thread::Builder::new()
        .name("juto-desktop-signal".to_string())
        .spawn(move || {
            use std::io::Read;
            let mut file = std::fs::File::from(read_fd);
            let mut buf = [0u8; 1];
            if file.read_exact(&mut buf).is_ok() {
                handler();
            }
        })
        .ok();
}

/// Write end of the signal self-pipe, set once before the handler is installed.
static SIGNAL_PIPE_WRITE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);
