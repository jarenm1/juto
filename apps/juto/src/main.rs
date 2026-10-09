//! Juto's native, local-only chat interface.
//! Agent execution and persistence are not connected to this UI.

mod chat;
mod chat_input;

use std::env;
use std::process::ExitCode;

use gpui::{
    App, Application, Bounds, KeyBinding, TitlebarOptions, WindowBounds, WindowOptions, actions,
    prelude::*, px, size,
};

actions!(juto, [Quit]);

const USAGE: &str = "Usage: juto [--version] [--help]
Opens the Juto local chat window (agent runtime is not connected).

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
        chat::init(cx);
        chat_input::init(cx);
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

        let bounds = Bounds::centered(None, size(px(960.0), px(720.0)), cx);
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
            |window, cx| cx.new(|cx| chat::ChatView::new(window, cx)),
        )
        .unwrap();
        cx.activate(true);
    });

    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    run()
}
