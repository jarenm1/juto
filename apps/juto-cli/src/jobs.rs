use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use clap::Subcommand;
use juto_daemon::{
    client::Client,
    protocol::{Reply, Request},
};
use juto_runtime::jobs::{InboxState, Job, JobBudget, JobContract};

#[derive(Subcommand)]
pub enum JobsCommand {
    /// Start a background job in this workspace's daemon and print its id. The job runs
    /// to completion without a client attached; its result lands in the inbox.
    RunNow {
        prompt: String,
        /// Wall-clock seconds the job may run in total, across attempts.
        #[arg(long)]
        wall_secs: u64,
        /// Attempts the job may start, counting the first.
        #[arg(long)]
        max_attempts: u32,
        /// Model turns one attempt may take. Unset leaves only the runtime's own turn limit.
        #[arg(long)]
        max_turns: Option<u32>,
        /// USD the job may spend across attempts, priced by the model catalog. Unset means no spend limit.
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
    },
    /// List jobs, oldest first.
    List,
    /// Show one job, including its result or error.
    Status { job_id: String },
    /// Cancel a queued, running, or blocked job.
    Cancel { job_id: String },
    /// Start a new attempt for a failed or interrupted job, within its budget.
    Retry { job_id: String },
    /// Archive a finished job and reclaim its isolated workspaces.
    Archive { job_id: String },
    /// Mark a job read in the inbox.
    MarkRead { job_id: String },
    /// Return finished jobs the main agent has not yet seen. Each job is returned once.
    Drain,
    /// Resolve the decision a blocked job is waiting on. Rejects unless --approve is given.
    Decide {
        decision_id: String,
        #[arg(long)]
        approve: bool,
    },
}

pub async fn run(cwd: &Path, command: &JobsCommand, json: bool) -> Result<()> {
    let client = Client::connect(cwd)
        .await
        .context("connecting to this workspace's daemon")?;
    match command {
        JobsCommand::RunNow {
            prompt,
            wall_secs,
            max_attempts,
            max_turns,
            max_spend,
            artifact,
            output_schema,
            no_change,
        } => {
            let output_schema = match output_schema {
                Some(path) => {
                    let text = std::fs::read_to_string(path)
                        .with_context(|| format!("reading {}", path.display()))?;
                    Some(
                        serde_json::from_str(&text)
                            .with_context(|| format!("parsing {} as JSON", path.display()))?,
                    )
                }
                None => None,
            };
            let request = Request::StartJob {
                prompt: prompt.clone(),
                budget: JobBudget {
                    wall_secs: *wall_secs,
                    max_attempts: *max_attempts,
                    max_turns: *max_turns,
                    max_spend: *max_spend,
                },
                contract: JobContract {
                    artifact: artifact.clone(),
                    output_schema,
                    no_change: *no_change,
                },
            };
            print_job(&expect_job(client.request(request).await?)?, json)
        }
        JobsCommand::List => match client.request(Request::ListJobs).await? {
            Reply::Jobs(jobs) => print_jobs(&jobs, json),
            other => unexpected(other),
        },
        JobsCommand::Status { job_id } => {
            let job = expect_job(
                client
                    .request(Request::GetJob {
                        job_id: job_id.clone(),
                    })
                    .await?,
            )?;
            print_detail(&job, json)
        }
        JobsCommand::Cancel { job_id } => print_job(
            &expect_job(
                client
                    .request(Request::CancelJob {
                        job_id: job_id.clone(),
                    })
                    .await?,
            )?,
            json,
        ),
        JobsCommand::Retry { job_id } => print_job(
            &expect_job(
                client
                    .request(Request::RetryJob {
                        job_id: job_id.clone(),
                    })
                    .await?,
            )?,
            json,
        ),
        JobsCommand::Archive { job_id } => print_job(
            &expect_job(
                client
                    .request(Request::SetJobInbox {
                        job_id: job_id.clone(),
                        state: InboxState::Archived,
                    })
                    .await?,
            )?,
            json,
        ),
        JobsCommand::MarkRead { job_id } => print_job(
            &expect_job(
                client
                    .request(Request::SetJobInbox {
                        job_id: job_id.clone(),
                        state: InboxState::Read,
                    })
                    .await?,
            )?,
            json,
        ),
        JobsCommand::Drain => match client.request(Request::DrainJobs).await? {
            Reply::Jobs(jobs) => print_jobs(&jobs, json),
            other => unexpected(other),
        },
        JobsCommand::Decide {
            decision_id,
            approve,
        } => match client
            .request(Request::ResolveDecision {
                decision_id: decision_id.clone(),
                approve: *approve,
            })
            .await?
        {
            Reply::Decision { decision, .. } => {
                if json {
                    println!("{}", serde_json::to_string(&decision)?);
                } else {
                    let outcome = if *approve { "approved" } else { "rejected" };
                    println!("{} {outcome}", decision.id);
                }
                Ok(())
            }
            other => unexpected(other),
        },
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

fn print_job(job: &Job, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string(job)?);
    } else {
        println!("{}", summary(job));
    }
    Ok(())
}

fn print_jobs(jobs: &[Job], json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string(jobs)?);
    } else {
        for job in jobs {
            println!("{}", summary(job));
        }
    }
    Ok(())
}

fn print_detail(job: &Job, json: bool) -> Result<()> {
    if json {
        return print_job(job, true);
    }
    println!("{}", summary(job));
    println!("  prompt: {}", job.prompt);
    println!(
        "  budget: {}s, {} attempt(s)",
        job.budget.wall_secs, job.budget.max_attempts
    );
    if let Some(result) = &job.result {
        println!("  result: {result}");
    }
    if let Some(error) = &job.error {
        println!("  error: {error}");
    }
    if let Some(decision) = &job.decision_id {
        println!("  decision: {decision}");
    }
    Ok(())
}

fn summary(job: &Job) -> String {
    format!(
        "{}  {:?}  inbox:{}  attempts:{}  {}",
        job.id,
        job.status,
        inbox_name(job.inbox),
        job.attempts,
        job.prompt.lines().next().unwrap_or("")
    )
}

fn inbox_name(state: InboxState) -> &'static str {
    match state {
        InboxState::Unread => "unread",
        InboxState::Read => "read",
        InboxState::Archived => "archived",
    }
}
