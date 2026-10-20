use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use clap::{Subcommand, ValueEnum};
use juto_daemon::{
    client::Client,
    protocol::{Reply, Request},
};
use juto_runtime::automations::{
    Automation, AutomationEvent, AutomationHistory, AutomationSpec, Overlap,
};
use juto_runtime::broker::Priority;
use juto_runtime::jobs::{Job, JobBudget, JobContract};

#[derive(Subcommand)]
pub enum AutomationsCommand {
    /// Create a recurring automation in this workspace's daemon. Each occurrence starts one
    /// job, the same as `jobs run-now`. Occurrences fall every N seconds after creation,
    /// in UTC; see docs/unattended-runs.md for the recurrence and restart rules.
    Create {
        prompt: String,
        /// Seconds between occurrences: 1 to 31622400 (366 days).
        #[arg(long)]
        every_secs: u64,
        /// Wall-clock seconds each occurrence's job may run in total, across attempts.
        #[arg(long)]
        wall_secs: u64,
        /// Attempts each occurrence's job may start, counting the first.
        #[arg(long)]
        max_attempts: u32,
        /// Model turns one attempt may take. Unset leaves only the runtime's own turn limit.
        #[arg(long)]
        max_turns: Option<u32>,
        /// USD each occurrence's job may spend across attempts. Unset means no spend limit.
        #[arg(long)]
        max_spend: Option<f64>,
        /// Completion contract: a workspace-relative file that must exist and be non-empty.
        #[arg(long)]
        artifact: Option<String>,
        /// Completion contract: a JSON Schema file the final answer must satisfy.
        #[arg(long)]
        output_schema: Option<PathBuf>,
        /// Completion contract: the workspace must have no changes when the job stops.
        #[arg(long)]
        no_change: bool,
        /// Run occurrences at background priority instead of scheduled priority.
        #[arg(long)]
        background: bool,
        /// What an occurrence does while the previous occurrence's job is still open.
        #[arg(long, value_enum, default_value = "skip")]
        overlap: OverlapArg,
        /// Pin this model selector on each occurrence's session. Unset uses the daemon's model.
        #[arg(long)]
        model_pin: Option<String>,
    },
    /// List automations, oldest first.
    List,
    /// Show one automation's definition and schedule.
    Show { automation_id: String },
    /// Show the jobs an automation started, and its event log.
    History { automation_id: String },
    /// Start one job for the automation now. Its schedule does not move. Refused while a
    /// job of this automation is still open.
    RunNow { automation_id: String },
    /// Pause an automation. No occurrence fires until it is enabled again.
    Pause { automation_id: String },
    /// Enable a paused automation. Occurrences that passed while paused are recorded as
    /// missed and are not run.
    Enable { automation_id: String },
    /// Delete an automation. The jobs it started are kept.
    Delete { automation_id: String },
}

#[derive(Clone, Copy, ValueEnum)]
pub enum OverlapArg {
    Skip,
    Queue,
}

pub async fn run(cwd: &Path, command: &AutomationsCommand, json: bool) -> Result<()> {
    let client = Client::connect(cwd)
        .await
        .context("connecting to this workspace's daemon")?;
    match command {
        AutomationsCommand::Create {
            prompt,
            every_secs,
            wall_secs,
            max_attempts,
            max_turns,
            max_spend,
            artifact,
            output_schema,
            no_change,
            background,
            overlap,
            model_pin,
        } => {
            let spec = AutomationSpec {
                prompt: prompt.clone(),
                model: model_pin.clone(),
                priority: background.then_some(Priority::Background),
                budget: JobBudget {
                    wall_secs: *wall_secs,
                    max_attempts: *max_attempts,
                    max_turns: *max_turns,
                    max_spend: *max_spend,
                },
                contract: JobContract {
                    artifact: artifact.clone(),
                    output_schema: read_output_schema(output_schema.as_ref())?,
                    no_change: *no_change,
                },
                every_secs: *every_secs,
                overlap: Some(match overlap {
                    OverlapArg::Skip => Overlap::Skip,
                    OverlapArg::Queue => Overlap::Queue,
                }),
            };
            print_automation(
                &expect_automation(client.request(Request::CreateAutomation { spec }).await?)?,
                json,
            )
        }
        AutomationsCommand::List => match client.request(Request::ListAutomations).await? {
            Reply::Automations(automations) => print_automations(&automations, json),
            other => unexpected(other),
        },
        AutomationsCommand::Show { automation_id } => print_automation(
            &expect_automation(
                client
                    .request(Request::GetAutomation {
                        automation_id: automation_id.clone(),
                    })
                    .await?,
            )?,
            json,
        ),
        AutomationsCommand::History { automation_id } => match client
            .request(Request::AutomationHistory {
                automation_id: automation_id.clone(),
            })
            .await?
        {
            Reply::AutomationHistory(history) => print_history(&history, json),
            other => unexpected(other),
        },
        AutomationsCommand::RunNow { automation_id } => print_job(
            &expect_job(
                client
                    .request(Request::RunAutomationNow {
                        automation_id: automation_id.clone(),
                    })
                    .await?,
            )?,
            json,
        ),
        AutomationsCommand::Pause { automation_id } => print_automation(
            &expect_automation(
                client
                    .request(Request::SetAutomationEnabled {
                        automation_id: automation_id.clone(),
                        enabled: false,
                    })
                    .await?,
            )?,
            json,
        ),
        AutomationsCommand::Enable { automation_id } => print_automation(
            &expect_automation(
                client
                    .request(Request::SetAutomationEnabled {
                        automation_id: automation_id.clone(),
                        enabled: true,
                    })
                    .await?,
            )?,
            json,
        ),
        AutomationsCommand::Delete { automation_id } => match client
            .request(Request::DeleteAutomation {
                automation_id: automation_id.clone(),
            })
            .await?
        {
            Reply::Ok => {
                if json {
                    println!("{}", serde_json::json!({ "deleted": automation_id }));
                } else {
                    println!("deleted {automation_id}");
                }
                Ok(())
            }
            other => unexpected(other),
        },
    }
}

fn read_output_schema(path: Option<&PathBuf>) -> Result<Option<serde_json::Value>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let schema = serde_json::from_str(&text)
        .with_context(|| format!("parsing {} as JSON", path.display()))?;
    Ok(Some(schema))
}

fn expect_automation(reply: Reply) -> Result<Automation> {
    match reply {
        Reply::Automation(automation) => Ok(automation),
        other => unexpected(other),
    }
}

fn expect_job(reply: Reply) -> Result<Job> {
    match reply {
        Reply::Job(job) => Ok(job),
        other => unexpected(other),
    }
}

fn unexpected<T>(reply: Reply) -> Result<T> {
    bail!("unexpected daemon reply: {reply:?}")
}

fn print_automation(automation: &Automation, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string(automation)?);
    } else {
        println!("{}", summary(automation));
    }
    Ok(())
}

fn print_automations(automations: &[Automation], json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string(automations)?);
    } else {
        for automation in automations {
            println!("{}", summary(automation));
        }
    }
    Ok(())
}

fn print_job(job: &Job, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string(job)?);
    } else {
        println!("{}  {:?}  attempts:{}", job.id, job.status, job.attempts);
    }
    Ok(())
}

fn print_history(history: &AutomationHistory, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string(history)?);
        return Ok(());
    }
    for job in &history.jobs {
        println!(
            "job {}  {:?}  attempts:{}",
            job.id, job.status, job.attempts
        );
    }
    for event in &history.events {
        println!("{}", event_line(event));
    }
    Ok(())
}

fn summary(automation: &Automation) -> String {
    let state = if automation.enabled {
        "enabled"
    } else {
        "paused"
    };
    let overlap = match automation.overlap {
        Overlap::Skip => "skip",
        Overlap::Queue => "queue",
    };
    format!(
        "{}  every:{}s  {state}  overlap:{overlap}  next:{}  {}",
        automation.id,
        automation.every_secs,
        automation.next_instant_ms,
        automation.prompt.lines().next().unwrap_or("")
    )
}

fn event_line(event: &AutomationEvent) -> String {
    match event {
        AutomationEvent::Fired {
            at_ms,
            instant_ms,
            job_id,
        } => format!("{at_ms}  fired  instant:{instant_ms}  job:{job_id}"),
        AutomationEvent::RunNow { at_ms, job_id } => format!("{at_ms}  run-now  job:{job_id}"),
        AutomationEvent::Missed {
            at_ms,
            from_ms,
            count,
        } => format!("{at_ms}  missed  from:{from_ms}  count:{count}"),
        AutomationEvent::Skipped {
            at_ms,
            instant_ms,
            count,
        } => format!("{at_ms}  skipped  instant:{instant_ms}  count:{count}"),
        AutomationEvent::Failed {
            at_ms,
            instant_ms,
            error,
        } => format!("{at_ms}  failed  instant:{instant_ms}  {error}"),
    }
}
