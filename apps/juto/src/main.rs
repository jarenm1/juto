//! Juto: build-environment bootstrap binary.
//!
//! Opens a single native GPUI window to prove the toolchain, Nix-provided
//! native libraries, and GPU/display stack work end to end. The agent runtime
//! is intentionally absent at this stage.

use std::env;
use std::process::ExitCode;

use gpui::{
    App, Application, Bounds, Context, FocusHandle, KeyBinding, TitlebarOptions, Window,
    WindowBounds, WindowOptions, actions, div, prelude::*, px, rgb, size,
};

actions!(juto, [Quit]);

struct JutoWindow {
    focus_handle: FocusHandle,
}

impl Render for JutoWindow {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .track_focus(&self.focus_handle)
            .on_action(|_: &Quit, _window, cx| cx.quit())
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap_2()
            .size_full()
            .bg(rgb(0x1a1a1a))
            .text_color(rgb(0xe6e6e6))
            .child(div().text_xl().child("Juto"))
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x999999))
                    .child("GPUI build environment OK. Press Ctrl-Q or close the window to quit."),
            )
    }
}

const USAGE: &str = "Usage: juto [--version] [--help]
Opens the Juto build-foundation window.

Options:
  --version    Print version and exit
  --help       Print this help and exit";

fn print_version() {
    println!("juto {}", env!("CARGO_PKG_VERSION"));
}

fn run() -> ExitCode {
    match env::args().nth(1).as_deref() {
        // Handle informational flags without initializing the graphical runtime
        // so the binary can smoke-test on headless hosts.
        Some("--version" | "-V") => {
            print_version();
            return ExitCode::SUCCESS;
        }
        Some("--help" | "-h") => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Some(arg) => {
            eprintln!("error: unrecognized argument `{arg}`\n\n{USAGE}");
            return ExitCode::from(2);
        }
        None => {}
    }

    Application::new().run(|cx: &mut App| {
        cx.bind_keys([
            KeyBinding::new("ctrl-q", Quit, None),
            KeyBinding::new("cmd-q", Quit, None),
        ]);
        cx.on_window_closed(|cx| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();

        let bounds = Bounds::centered(None, size(px(640.0), px(400.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(TitlebarOptions {
                    title: Some("Juto".into()),
                    ..Default::default()
                }),
                app_id: Some("dev.juto".into()),
                ..Default::default()
            },
            |window, cx| {
                cx.new(|cx| {
                    let focus_handle = cx.focus_handle();
                    focus_handle.focus(window);
                    JutoWindow { focus_handle }
                })
            },
        )
        .unwrap();
        cx.activate(true);
    });

    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    run()
}
