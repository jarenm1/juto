//! Juto's native chat interface wired to the agent runtime.

mod chat;
mod chat_input;

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use gpui::{
    App, Application, Bounds, KeyBinding, TitlebarOptions, WindowBounds, WindowOptions, actions,
    prelude::*, px, size,
};
use juto_runtime::{Runtime, RuntimeConfig, Session, default_data_dir};

actions!(juto, [Quit]);

const USAGE: &str = "Usage: juto [--version] [--help] [--session <path>] [--model <model>]
Opens the Juto local chat window connected to the agent runtime.

Options:
  --session <path>  Open an existing session journal
  --model <model>   Select a model (e.g. anthropic/claude-sonnet-4-5)
  --version         Print version and exit
  --help            Print this help and exit";

fn print_version() {
    println!("juto {}", env!("CARGO_PKG_VERSION"));
}

fn run() -> ExitCode {
    let mut session_path = None;
    let mut model_arg = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--version" | "-V" => {
                print_version();
                return ExitCode::SUCCESS;
            }
            "--help" | "-h" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "--session" => match args.next() {
                Some(path) => session_path = Some(PathBuf::from(path)),
                None => {
                    eprintln!("error: `--session` requires a path argument\n\n{USAGE}");
                    return ExitCode::from(2);
                }
            },
            "--model" => match args.next() {
                Some(model) => model_arg = Some(model),
                None => {
                    eprintln!("error: `--model` requires a model argument\n\n{USAGE}");
                    return ExitCode::from(2);
                }
            },
            arg => {
                eprintln!("error: unrecognized argument `{arg}`\n\n{USAGE}");
                return ExitCode::from(2);
            }
        }
    }

    let tokio_runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(err) => {
            eprintln!("error: failed to start Tokio runtime: {err}");
            return ExitCode::from(1);
        }
    };
    let _guard = tokio_runtime.enter();
    let tokio_handle = tokio_runtime.handle().clone();

    let cwd = match env::current_dir() {
        Ok(cwd) => cwd,
        Err(err) => {
            eprintln!("error: failed to determine current working directory: {err}");
            return ExitCode::from(1);
        }
    };
    let data_dir = match default_data_dir() {
        Ok(dir) => dir,
        Err(err) => {
            eprintln!("error: failed to determine default data directory: {err}");
            return ExitCode::from(1);
        }
    };
    let mut config =
        match RuntimeConfig::load(&data_dir.join("config.yml"), &cwd.join(".juto/config.yml")) {
            Ok(config) => config,
            Err(err) => {
                eprintln!("error: failed to load runtime config: {err}");
                return ExitCode::from(1);
            }
        };
    if let Some(model) = model_arg {
        config.model = model;
    }
    if config.model.trim().is_empty() {
        config.model = "anthropic/claude-sonnet-4-5".into();
    }
    let runtime = match Runtime::from_config(&cwd, &data_dir, config) {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("error: failed to initialize agent runtime: {err}");
            return ExitCode::from(1);
        }
    };
    let session = match session_path {
        Some(path) => match Session::open(&path) {
            Ok(session) => session,
            Err(err) => {
                eprintln!(
                    "error: failed to open session at `{}`: {err}",
                    path.display()
                );
                return ExitCode::from(1);
            }
        },
        None => match runtime.new_session() {
            Ok(session) => session,
            Err(err) => {
                eprintln!("error: failed to create new session: {err}");
                return ExitCode::from(1);
            }
        },
    };

    Application::new().run(move |cx: &mut App| {
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
            |window, cx| {
                cx.new(|cx| chat::ChatView::new(runtime, session, tokio_handle, window, cx))
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
