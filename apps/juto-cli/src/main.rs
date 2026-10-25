mod automations;
mod changes;
mod jobs;
mod resources;

use std::{
    collections::BTreeSet,
    io::{self, BufRead, IsTerminal, Read, Write},
    path::PathBuf,
};

use anyhow::{Context as _, Result, bail};
use clap::{Parser, Subcommand};
use juto_agent::{AgentEvent, RunControl};
use juto_ai::{HttpProvider, ProviderEvent, StopReason};
use juto_runtime::{ApprovalMode, MainSession, Runtime, RuntimeConfig, Session, default_data_dir};
use tokio::sync::mpsc;

#[derive(Parser)]
#[command(
    version,
    about = "Juto headless agent runtime (the GPUI app is a separate host)"
)]
struct Arguments {
    #[arg(long, global = true, default_value = ".")]
    cwd: PathBuf,
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,
    #[arg(long, global = true)]
    model: Option<String>,
    #[arg(long, global = true)]
    thinking: Option<String>,
    #[arg(long, global = true)]
    max_tokens: Option<u64>,
    /// Explicitly permit write/process/subagent tools without interactive approval.
    #[arg(long, global = true, conflicts_with = "read_only")]
    yes: bool,
    #[arg(long, global = true)]
    read_only: bool,
    /// Emit one runtime event per JSON line instead of text deltas.
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Inspect live edits, mark exact snapshots, and transfer marked points to Git.
    Changes {
        #[command(subcommand)]
        command: changes::ChangesCommand,
    },
    /// Show measured resource use per command class, or run a command under governance.
    Resources {
        /// Summarize only commands recorded for this repository root.
        #[arg(long)]
        repo: Option<PathBuf>,
        #[command(subcommand)]
        command: Option<resources::ResourcesCommand>,
    },
    /// Start, inspect, and manage background jobs in this workspace's daemon.
    Jobs {
        #[command(subcommand)]
        command: jobs::JobsCommand,
    },
    /// Create, list, run, pause, enable, and delete scheduled automations in this
    /// workspace's daemon.
    Automations {
        #[command(subcommand)]
        command: automations::AutomationsCommand,
    },
    /// Stream a turn in a new or existing main session: one conversation, seen
    /// through its memory view. A prompt of '-' reads stdin.
    Run {
        prompt: String,
        #[arg(long)]
        session: Option<PathBuf>,
    },
    /// List runnable chat models for providers with configured credentials.
    Models {
        #[arg(long)]
        provider: Option<String>,
    },
    /// Perform OAuth login, or securely read an API key from stdin/terminal.
    Login {
        provider: String,
        #[arg(long)]
        api_key: bool,
    },
    Logout {
        provider: String,
    },
    Credentials,
    /// Print a main session's conversation, oldest first.
    History {
        session: PathBuf,
    },
    /// Persist a model/role/thinking selector for the next turn.
    Model {
        session: PathBuf,
        selector: String,
    },
    /// Fork the active branch into a separate session journal.
    Fork {
        session: PathBuf,
    },
    /// Select a journal entry as the active branch; omission selects the empty branch.
    Branch {
        session: PathBuf,
        entry: Option<String>,
    },
    /// Summarize a session journal with the selected provider and preserve recent
    /// complete tool groups. Main sessions keep a memory tree instead.
    Compact {
        session: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut raw = std::env::args_os().skip(1);
    if raw
        .next()
        .is_some_and(|arg| arg == juto_runtime::network::FORWARDER_ARGUMENT)
    {
        let code = juto_runtime::network::run_forwarder(raw)
            .context("sandbox network forwarder failed")?;
        std::process::exit(code);
    }
    let arguments = Arguments::parse();
    let cwd = arguments.cwd.canonicalize().context("invalid --cwd")?;
    if let Command::Changes { command } = &arguments.command {
        return changes::run(&cwd, command);
    }
    if let Command::Resources { repo, command } = &arguments.command {
        let data_dir = match &arguments.data_dir {
            Some(path) => path.clone(),
            None => default_data_dir()?,
        };
        let code = resources::run(
            &cwd,
            &data_dir,
            repo.as_deref(),
            command.as_ref(),
            arguments.json,
        )
        .await?;
        if code != 0 {
            std::process::exit(code);
        }
        return Ok(());
    }
    if let Command::Jobs { command } = &arguments.command {
        if arguments.model.is_some()
            || arguments.thinking.is_some()
            || arguments.max_tokens.is_some()
            || arguments.yes
            || arguments.read_only
        {
            bail!(
                "--model, --thinking, --max-tokens, --yes and --read-only do not apply to jobs: \
                 the daemon runs a job with this workspace's .juto/config.yml and its own approval rules"
            );
        }
        return jobs::run(&cwd, command, arguments.json).await;
    }
    if let Command::Automations { command } = &arguments.command {
        if arguments.model.is_some()
            || arguments.thinking.is_some()
            || arguments.max_tokens.is_some()
            || arguments.yes
            || arguments.read_only
        {
            bail!(
                "--model, --thinking, --max-tokens, --yes and --read-only do not apply to automations: \
                 pin a model with --model-pin; occurrences run under the daemon's unattended rules"
            );
        }
        return automations::run(&cwd, command, arguments.json).await;
    }
    let data_dir = match arguments.data_dir {
        Some(path) => path,
        None => default_data_dir()?,
    };
    let mut config =
        RuntimeConfig::load(&data_dir.join("config.yml"), &cwd.join(".juto/config.yml"))?;
    let model_override = arguments.model.is_some();
    if let Some(model) = arguments.model {
        config.model = model;
    }
    if let Some(thinking) = arguments.thinking {
        config.thinking = Some(thinking);
    }
    if let Some(max_tokens) = arguments.max_tokens {
        config.max_tokens = Some(max_tokens);
    }
    if arguments.yes {
        config.approval_mode = ApprovalMode::Allow;
    }
    if arguments.read_only {
        config.approval_mode = ApprovalMode::ReadOnly;
    }
    let runtime = Runtime::from_config(cwd, data_dir, config)?;
    match arguments.command {
        Command::Changes { .. } => unreachable!("handled before runtime initialization"),
        Command::Resources { .. } => unreachable!("handled before runtime initialization"),
        Command::Jobs { .. } => unreachable!("handled before runtime initialization"),
        Command::Automations { .. } => unreachable!("handled before runtime initialization"),
        Command::Run { prompt, session } => {
            let resumed = session.is_some();
            let prompt = if prompt == "-" {
                let mut value = String::new();
                io::stdin().read_to_string(&mut value)?;
                value
            } else {
                prompt
            };
            if prompt.trim().is_empty() {
                bail!("prompt must not be empty");
            }
            let mut session = match session {
                Some(path) => MainSession::open(&path)?,
                None => runtime.new_main_session()?,
            };
            runtime.open_main_inbox(&session)?;
            if model_override && resumed {
                runtime.switch_main_model(&mut session, &runtime.config().model)?;
            }
            eprintln!("session: {}", session.path().display());
            let control = RunControl::new();
            let signals = install_interrupt(control.clone());
            let (events, receiver) = mpsc::unbounded_channel();
            let printer = tokio::spawn(print_events(receiver, arguments.json));
            let result = runtime.send(&mut session, prompt, events, control).await;
            signals.abort();
            printer.await??;
            let outcome = result?;
            eprintln!(
                "tokens: {} (input {}, output {}, cache read {}, cache write {}), cost: ${:.6}",
                outcome.usage.total_tokens,
                outcome.usage.input,
                outcome.usage.output,
                outcome.usage.cache_read,
                outcome.usage.cache_write,
                outcome.usage.cost
            );
            if outcome.stop_reason == StopReason::Aborted {
                bail!("run cancelled");
            }
            if outcome.stop_reason == StopReason::Error {
                bail!("run failed");
            }
            if outcome.stop_reason != StopReason::Stop {
                bail!("run did not complete: {:?}", outcome.stop_reason);
            }
        }
        Command::Models { provider } => {
            let providers: BTreeSet<&str> = runtime
                .catalog()
                .iter()
                .map(|model| model.provider.as_str())
                .collect();
            let available = runtime.credentials().available_providers(providers)?;
            let stdout = io::stdout();
            let mut writer = io::BufWriter::new(stdout.lock());
            for model in runtime.catalog().iter().filter(|model| {
                available.contains(&model.provider)
                    && matches!(model.kind.as_str(), "chat" | "tiny")
                    && HttpProvider::supports_model(model)
                    && provider
                        .as_deref()
                        .is_none_or(|provider| provider == model.provider)
            }) {
                if arguments.json {
                    serde_json::to_writer(&mut writer, model)?;
                    writeln!(writer)?;
                } else {
                    writeln!(
                        writer,
                        "{}/{}\t{}\t{}",
                        model.provider, model.id, model.name, model.api
                    )?;
                }
            }
            writer.flush()?;
        }
        Command::Login { provider, api_key } => {
            let credentials = runtime.credentials();
            if api_key {
                let key = if io::stdin().is_terminal() {
                    rpassword::prompt_password(format!("{provider} API key: "))?
                } else {
                    let mut key = String::new();
                    io::stdin().lock().read_line(&mut key)?;
                    key.trim_end_matches(['\r', '\n']).to_owned()
                };
                if key.trim().is_empty() {
                    bail!("API key must not be empty");
                }
                credentials.set_api_key(&provider, &key)?;
                eprintln!("API key stored for {provider}.");
            } else {
                let control = RunControl::new();
                let signals = install_interrupt(control.clone());
                let result = credentials
                    .login(
                        &provider,
                        control.token(),
                        std::sync::Arc::new(|url| {
                            eprintln!("Open this URL in your browser:\n{url}")
                        }),
                    )
                    .await;
                signals.abort();
                result?;
                eprintln!("Logged in to {provider}.");
            }
        }
        Command::Logout { provider } => {
            runtime.credentials().remove(&provider)?;
            eprintln!("Credentials removed for {provider}.");
        }
        Command::Credentials => {
            println!(
                "{}",
                serde_json::to_string_pretty(&runtime.credentials().list()?)?
            );
        }
        Command::History { session } => {
            for message in MainSession::open(&session)?.timeline()? {
                if arguments.json {
                    println!("{}", serde_json::to_string(&message)?);
                } else {
                    println!("{}", message.text());
                }
            }
        }
        Command::Model { session, selector } => {
            let mut session = Session::open(&session)?;
            runtime.switch_model(&mut session, &selector)?;
            println!("{selector}");
        }
        Command::Fork { session } => {
            let session = Session::open(&session)?;
            let fork = session.fork(&runtime.data_dir().join("sessions"))?;
            println!("{}", fork.path().display());
        }
        Command::Branch { session, entry } => {
            let mut session = Session::open(&session)?;
            session.branch(entry.as_deref())?;
            println!("{}", session.path().display());
        }
        Command::Compact { session } => {
            let mut session = Session::open(&session)?;
            let control = RunControl::new();
            let signals = install_interrupt(control.clone());
            let result = runtime.compact(&mut session, &control).await;
            signals.abort();
            let usage = result?;
            eprintln!(
                "Compacted {} ({} tokens, ${:.6}).",
                session.path().display(),
                usage.total_tokens,
                usage.cost
            );
        }
    }
    Ok(())
}

fn install_interrupt(control: RunControl) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            control.cancel();
        }
    })
}

async fn print_events(mut receiver: mpsc::UnboundedReceiver<AgentEvent>, json: bool) -> Result<()> {
    while let Some(event) = receiver.recv().await {
        let stdout = io::stdout();
        let mut writer = stdout.lock();
        if json {
            serde_json::to_writer(&mut writer, &event)?;
            writeln!(writer)?;
        } else {
            match event {
                AgentEvent::MessageUpdate {
                    event: ProviderEvent::TextDelta { delta, .. },
                } => {
                    write!(writer, "{delta}")?;
                    writer.flush()?;
                }
                AgentEvent::ToolExecutionStart { call } => {
                    eprintln!("tool: {} ({})", call.name, call.id)
                }
                AgentEvent::Error { error } => eprintln!("error: {error}"),
                _ => {}
            }
        }
    }
    if !json {
        println!();
    }
    Ok(())
}
