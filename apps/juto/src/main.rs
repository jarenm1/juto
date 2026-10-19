//! Juto's native chat interface, served by the workspace daemon.

mod auth_modal;
mod chat;
mod chat_error;
mod chat_input;
mod chat_request;
mod desktop_viewer;
mod focus_view;
mod model_picker;
mod sidebar;
mod subagent_card;
mod task_card;
mod text_input;
mod theme;
mod vcs_sidebar;
mod view_element;
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context as _, bail};
use gpui::{
    App, Application, Bounds, KeyBinding, TitlebarOptions, WindowBounds, WindowOptions, actions,
    prelude::*, px, size,
};
use juto_daemon::client::{Client, Notification};
use juto_daemon::protocol::{Reply, Request, SessionView};
use tokio::sync::broadcast;

actions!(juto, [Quit]);

const DEFAULT_MODEL: &str = "anthropic/claude-sonnet-4-5";

const USAGE: &str = "Usage: juto [--version] [--help] [--session <path>] [--model <model>]
            juto --desktop <command> [args...]
Opens the Juto chat window, connected to this workspace's daemon.

Options:
  --session <path>  Open an existing session journal
  --model <model>   Select a model (e.g. anthropic/claude-sonnet-4-5)
  --desktop <cmd>   Create a headless desktop, launch <cmd> into it, and open
                    the embedded viewer (developer mode)
  --version         Print version and exit
  --help            Print this help and exit";

fn print_version() {
    println!("juto {}", env!("CARGO_PKG_VERSION"));
}

/// Everything the chat window needs from the daemon before it can open.
struct Startup {
    client: Client,
    notifications: broadcast::Receiver<Notification>,
    session: SessionView,
    default_model: String,
}

/// Connects to the workspace daemon and opens (or creates) the session the
/// window shows. Failures are returned so the window can report them.
async fn start_workspace(
    cwd: &Path,
    session_path: Option<PathBuf>,
    model: Option<String>,
) -> anyhow::Result<Startup> {
    let client = Client::connect(cwd)
        .await
        .context("failed to connect to the workspace daemon")?;
    let notifications = client.subscribe();

    let Reply::Config(config) = client
        .request(Request::GetConfig)
        .await
        .context("failed to load runtime config")?
    else {
        bail!("daemon sent an unexpected reply to GetConfig");
    };
    let model = model.filter(|model| !model.trim().is_empty());
    let default_model = model.clone().unwrap_or(config.model);
    let default_model = if default_model.trim().is_empty() {
        DEFAULT_MODEL.into()
    } else {
        default_model
    };

    let request = match session_path {
        Some(path) => Request::OpenSessionPath { path },
        None => Request::CreateSession,
    };
    let Reply::Session(mut session) = client
        .request(request)
        .await
        .context("failed to open or create a session")?
    else {
        bail!("daemon sent an unexpected reply to session open");
    };

    if let Some(model) = model {
        client
            .request(Request::SetSessionModel {
                session_id: session.info.id.clone(),
                selector: model.clone(),
            })
            .await
            .context("failed to set session model")?;
        session.info.model = Some(model);
    }

    Ok(Startup {
        client,
        notifications,
        session,
        default_model,
    })
}

fn window_options(cx: &mut App) -> WindowOptions {
    let bounds = Bounds::centered(None, size(px(960.0), px(720.0)), cx);
    WindowOptions {
        window_bounds: Some(WindowBounds::Windowed(bounds)),
        titlebar: Some(TitlebarOptions {
            title: Some("Juto".into()),
            ..Default::default()
        }),
        app_id: Some("dev.juto".into()),
        ..Default::default()
    }
}

fn run() -> ExitCode {
    let mut session_path = None;
    let mut model_arg = None;
    let mut desktop_argv: Option<Vec<String>> = None;
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
            "--desktop" => {
                let command: Vec<String> = args.by_ref().collect();
                if command.is_empty() {
                    eprintln!("error: `--desktop` requires a command\n\n{USAGE}");
                    return ExitCode::from(2);
                }
                desktop_argv = Some(command);
                break;
            }
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
    let tokio_handle = tokio_runtime.handle().clone();

    let cwd = match env::current_dir() {
        Ok(cwd) => cwd,
        Err(err) => {
            eprintln!("error: failed to determine current working directory: {err}");
            return ExitCode::from(1);
        }
    };
    let startup = tokio_runtime.block_on(start_workspace(&cwd, session_path, model_arg));

    Application::new().run(move |cx: &mut App| {
        chat::init(cx);
        chat_input::init(cx);
        text_input::init(cx);
        model_picker::init(cx);
        auth_modal::init(cx);
        sidebar::init(cx);
        task_card::init(cx);
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

        match startup {
            Ok(startup) => {
                let options = window_options(cx);
                cx.open_window(options, |window, cx| {
                    let chat = cx.new(|cx| {
                        chat::ChatView::new(
                            startup.client,
                            tokio_handle,
                            startup.session,
                            startup.default_model,
                            startup.notifications,
                            window,
                            cx,
                        )
                    });
                    if let Some(argv) = desktop_argv.clone() {
                        chat.update(cx, |chat, cx| {
                            chat.open_desktop_command(argv, window, cx);
                        });
                    }
                    chat
                })
                .unwrap();
            }
            Err(err) => {
                let message = format!("{err:#}");
                let options = window_options(cx);
                cx.open_window(options, |_, cx| {
                    cx.new(|_| chat_error::StartupError::new(message))
                })
                .unwrap();
            }
        }
        cx.activate(true);
    });

    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    run()
}
