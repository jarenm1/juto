//! End-to-end multi-agent orchestrator and peer review DAG.
//!
//! Orchestrates high-level user intents through a disciplined multi-agent lifecycle:
//! 1. **Intent Triage**: Uses a discrete [`Judge`] to classify prompts into direct replies,
//!    read-only exploration ([`scout`]), small edits ([`sonic`]), or full multi-step pipelines
//!    ([`builder`] + verification quality gates + [`reviewer`]).
//! 2. **Builder Implementation**: Dispatches to a builder agent to formulate or modify code.
//! 3. **Verification Quality Gates**: Automates fail-fast sequential quality gates
//!    ([`Format`], [`Build`], [`Test`], [`Smoke`]).
//! 4. **Peer Review Sign-off**: Dispatches to an adversarial reviewer subagent only after all
//!    verification quality gates pass.
//! 5. **Remediation Loop**: Automatically routes verification failures or review critiques
//!    back to the builder up to a bounded iteration ceiling.
//! 6. **Real-time Observability & Steering**: Emits [`OrchestratorEvent`] broadcasts for
//!    UI sidebar tracking and respects cooperative [`CancellationToken`].

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use juto_ai::{Choice, Decision, Judge, JudgeError, Usage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::change_graph::ChangeGraph;
use crate::subagents::{SubagentError, SubagentManager, SubagentResult, TaskSpec};
use crate::workspace::{ResourceLimits, ScopedChild, ScopedStdio, SubagentWorkspace};

/// Intent route classified by discrete triage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntentRoute {
    /// Conversational answer or clarification; no subagent or code edit needed.
    DirectReply,
    /// Read-only discovery or file search using the `scout` agent persona.
    ExploreScout,
    /// Small, localized, single-turn refactor using the `sonic` agent persona.
    MechanicalEdit,
    /// Multi-step implementation pipeline requiring builder, verification gates, and reviewer.
    EndToEndTask,
}

impl IntentRoute {
    /// Returns the target agent persona identifier, or `None` if no subagent is required.
    pub fn persona(&self) -> Option<&'static str> {
        match self {
            Self::DirectReply => None,
            Self::ExploreScout => Some("scout"),
            Self::MechanicalEdit => Some("sonic"),
            Self::EndToEndTask => Some("builder"),
        }
    }
}

/// Deterministic heuristic/rule-based judge for intent triage when external API keys are absent.
#[derive(Debug, Default, Clone)]
pub struct RuleBasedJudge;

#[async_trait]
impl Judge for RuleBasedJudge {
    async fn choose(&self, context: &str, options: &[Choice]) -> Result<Decision, JudgeError> {
        if options.is_empty() {
            return Err(JudgeError::InvalidOptions("options cannot be empty".into()));
        }

        let lower = context.to_lowercase();

        // 1. Check for explicit question / scout patterns
        let is_scout = lower.starts_with("what ")
            || lower.starts_with("where ")
            || lower.starts_with("find ")
            || lower.starts_with("search ")
            || lower.starts_with("locate ")
            || lower.starts_with("list ")
            || lower.contains("how does")
            || lower.contains("where is")
            || lower.contains("scout");

        // 2. Check for small / mechanical edit patterns
        let is_sonic = lower.starts_with("rename ")
            || lower.starts_with("format ")
            || lower.starts_with("fix typo")
            || lower.starts_with("typo:")
            || lower.contains("quick fix")
            || lower.contains("sonic");

        // 3. Check for simple conversational greetings or direct answers
        let is_direct = lower == "hi"
            || lower == "hello"
            || lower == "hey"
            || lower.starts_with("thanks")
            || lower.starts_with("thank you")
            || lower == "ok"
            || lower == "yes"
            || lower == "no";

        let selected_key = if is_direct && options.iter().any(|c| c.key == "direct_reply") {
            "direct_reply"
        } else if is_scout && options.iter().any(|c| c.key == "explore_scout") {
            "explore_scout"
        } else if is_sonic && options.iter().any(|c| c.key == "mechanical_edit") {
            "mechanical_edit"
        } else if options.iter().any(|c| c.key == "end_to_end_task") {
            "end_to_end_task"
        } else {
            &options[0].key
        };

        Ok(Decision {
            key: selected_key.to_string(),
            confidence: Some(0.95),
            rationale: Some(format!(
                "RuleBasedJudge classified intent as '{selected_key}'"
            )),
            usage: Usage::default(),
            latency_ms: Some(1),
        })
    }
}

/// Scripted mock judge for unit testing and deterministic scenario simulation.
#[derive(Debug, Clone)]
pub struct ScriptedJudge {
    decision_queue: Arc<tokio::sync::Mutex<Vec<Decision>>>,
}

impl ScriptedJudge {
    pub fn new(decisions: Vec<Decision>) -> Self {
        Self {
            decision_queue: Arc::new(tokio::sync::Mutex::new(decisions)),
        }
    }

    pub fn single(key: impl Into<String>) -> Self {
        Self::new(vec![Decision {
            key: key.into(),
            confidence: Some(1.0),
            rationale: Some("Scripted decision".into()),
            usage: Usage::default(),
            latency_ms: Some(0),
        }])
    }
}

#[async_trait]
impl Judge for ScriptedJudge {
    async fn choose(&self, _context: &str, options: &[Choice]) -> Result<Decision, JudgeError> {
        let mut queue = self.decision_queue.lock().await;
        if queue.is_empty() {
            if let Some(first) = options.first() {
                return Ok(Decision {
                    key: first.key.clone(),
                    confidence: Some(1.0),
                    rationale: Some("Fallback first option".into()),
                    usage: Usage::default(),
                    latency_ms: Some(0),
                });
            }
            return Err(JudgeError::NoMatch(
                "scripted decision queue exhausted".into(),
            ));
        }
        Ok(queue.remove(0))
    }
}

/// Category of verification quality gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityGateKind {
    /// Source code formatting check.
    Format,
    /// Compilation and type checking pass.
    Build,
    /// Automated test suite execution.
    Test,
    /// Observable smoke check (e.g. running application or visual smoke command).
    Smoke,
}

impl fmt::Display for QualityGateKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format => write!(f, "format"),
            Self::Build => write!(f, "build"),
            Self::Test => write!(f, "test"),
            Self::Smoke => write!(f, "smoke"),
        }
    }
}

/// Specification for a single quality gate verification step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QualityGateSpec {
    pub kind: QualityGateKind,
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    #[serde(default = "default_gate_timeout")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub optional: bool,
}

fn default_gate_timeout() -> u64 {
    120
}

impl QualityGateSpec {
    pub fn new(kind: QualityGateKind, name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            kind,
            name: name.into(),
            command: command.into(),
            args: Vec::new(),
            cwd: None,
            timeout_secs: default_gate_timeout(),
            optional: false,
        }
    }

    pub fn with_args(mut self, args: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    pub fn with_timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    pub fn optional(mut self, optional: bool) -> Self {
        self.optional = optional;
        self
    }
}

/// Execution outcome of an individual quality gate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GateResult {
    pub kind: QualityGateKind,
    pub name: String,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u64,
}

/// Aggregate report of all executed quality gates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerificationReport {
    /// True if all non-optional quality gates passed.
    pub passed: bool,
    /// Individual results for each executed gate.
    pub gate_results: Vec<GateResult>,
    /// Human-readable summary of the verification outcome.
    pub summary: String,
}

impl VerificationReport {
    pub fn new(gate_results: Vec<GateResult>) -> Self {
        let passed = gate_results.iter().all(|r| r.passed);
        let total = gate_results.len();
        let passed_count = gate_results.iter().filter(|r| r.passed).count();
        let summary = format!("{passed_count}/{total} quality gates passed");
        Self {
            passed,
            gate_results,
            summary,
        }
    }

    /// Formats failure details into an actionable prompt for the builder remediation loop.
    pub fn format_remediation_feedback(&self) -> String {
        let mut feedback = String::from("Verification quality gates failed:\n\n");
        for gate in &self.gate_results {
            if !gate.passed {
                feedback.push_str(&format!(
                    "--- [FAILED] Gate: {} ({}) ---\n",
                    gate.name, gate.kind
                ));
                if let Some(code) = gate.exit_code {
                    feedback.push_str(&format!("Exit code: {code}\n"));
                }
                if !gate.stderr.trim().is_empty() {
                    feedback.push_str(&format!("STDERR:\n{}\n", gate.stderr.trim()));
                }
                if !gate.stdout.trim().is_empty() {
                    feedback.push_str(&format!("STDOUT:\n{}\n", gate.stdout.trim()));
                }
                feedback.push('\n');
            }
        }
        feedback.push_str("Please fix the issues above and ensure all verification gates pass.");
        feedback
    }
}

/// Trait for executing quality gate verification commands.
#[async_trait]
pub trait QualityGateRunner: Send + Sync + fmt::Debug {
    async fn execute_gate(
        &self,
        spec: &QualityGateSpec,
        cancel: &CancellationToken,
    ) -> Result<GateResult, OrchestratorError>;
}

/// Real process runner executing quality gates via `tokio::process::Command` with cancellation safety.
///
/// When `limits` is set, a gate runs inside a systemd user scope so the CPU and
/// memory caps apply to the whole process tree — gates are the heaviest commands
/// the orchestrator runs. `enforce_scope` is left to the caller: with it off, a
/// host without `systemd-run` degrades to a bare spawn rather than failing the
/// gate.
#[derive(Debug, Default, Clone)]
pub struct CommandGateRunner {
    pub limits: Option<ResourceLimits>,
}

#[async_trait]
impl QualityGateRunner for CommandGateRunner {
    async fn execute_gate(
        &self,
        spec: &QualityGateSpec,
        cancel: &CancellationToken,
    ) -> Result<GateResult, OrchestratorError> {
        let start = Instant::now();
        let mut cmd = tokio::process::Command::new(&spec.command);
        cmd.args(&spec.args);
        if let Some(cwd) = &spec.cwd {
            cmd.current_dir(cwd);
        }
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        cmd.kill_on_drop(true);

        let child = match &self.limits {
            Some(limits) => {
                // A gate without an explicit cwd runs in the process's own
                // directory, and the scope builder needs a real workspace path.
                let cwd = match &spec.cwd {
                    Some(cwd) => cwd.clone(),
                    None => std::env::current_dir().map_err(|err| {
                        OrchestratorError::Verification(format!(
                            "gate '{}' has no cwd and the process directory is unavailable: {err}",
                            spec.name
                        ))
                    })?,
                };
                let workspace = SubagentWorkspace {
                    path: cwd.clone(),
                    node_id: String::new(),
                    thread: String::new(),
                    repo_root: cwd,
                    subagent_id: String::new(),
                };
                workspace
                    .spawn_scoped(
                        &mut cmd,
                        limits.clone(),
                        ScopedStdio {
                            stdin: std::process::Stdio::null(),
                            stdout: std::process::Stdio::piped(),
                            stderr: std::process::Stdio::piped(),
                        },
                    )
                    .await
                    .map_err(|err| {
                        OrchestratorError::Verification(format!(
                            "failed to spawn gate '{}' ({}): {err}",
                            spec.name, spec.command
                        ))
                    })?
            }
            None => ScopedChild::unscoped(cmd.spawn().map_err(|err| {
                OrchestratorError::Verification(format!(
                    "failed to spawn gate '{}' ({}): {err}",
                    spec.name, spec.command
                ))
            })?),
        };

        let timeout_duration = Duration::from_secs(spec.timeout_secs);
        let run_future = child.wait_with_output();

        tokio::select! {
            _ = cancel.cancelled() => {
                Err(OrchestratorError::Cancelled)
            }
            res = tokio::time::timeout(timeout_duration, run_future) => {
                match res {
                    Ok(Ok(output)) => {
                        let duration_ms = start.elapsed().as_millis() as u64;
                        let passed = output.status.success() || spec.optional;
                        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                        Ok(GateResult {
                            kind: spec.kind,
                            name: spec.name.clone(),
                            passed,
                            exit_code: output.status.code(),
                            stdout,
                            stderr,
                            duration_ms,
                        })
                    }
                    Ok(Err(err)) => {
                        Err(OrchestratorError::Verification(format!("gate '{}' failed: {err}", spec.name)))
                    }
                    Err(_) => {
                        let duration_ms = start.elapsed().as_millis() as u64;
                        Ok(GateResult {
                            kind: spec.kind,
                            name: spec.name.clone(),
                            passed: spec.optional,
                            exit_code: None,
                            stdout: String::new(),
                            stderr: format!("Gate timed out after {} seconds", spec.timeout_secs),
                            duration_ms,
                        })
                    }
                }
            }
        }
    }
}

/// Mock gate runner for deterministic testing.
#[derive(Debug, Clone)]
pub struct MockGateRunner {
    results: Arc<tokio::sync::Mutex<Vec<GateResult>>>,
}

impl MockGateRunner {
    pub fn new(results: Vec<GateResult>) -> Self {
        Self {
            results: Arc::new(tokio::sync::Mutex::new(results)),
        }
    }

    pub fn passing() -> Self {
        Self::new(vec![
            GateResult {
                kind: QualityGateKind::Format,
                name: "format".into(),
                passed: true,
                exit_code: Some(0),
                stdout: "ok".into(),
                stderr: String::new(),
                duration_ms: 10,
            },
            GateResult {
                kind: QualityGateKind::Build,
                name: "build".into(),
                passed: true,
                exit_code: Some(0),
                stdout: "ok".into(),
                stderr: String::new(),
                duration_ms: 20,
            },
            GateResult {
                kind: QualityGateKind::Test,
                name: "test".into(),
                passed: true,
                exit_code: Some(0),
                stdout: "ok".into(),
                stderr: String::new(),
                duration_ms: 30,
            },
            GateResult {
                kind: QualityGateKind::Smoke,
                name: "smoke".into(),
                passed: true,
                exit_code: Some(0),
                stdout: "ok".into(),
                stderr: String::new(),
                duration_ms: 15,
            },
        ])
    }
}

#[async_trait]
impl QualityGateRunner for MockGateRunner {
    async fn execute_gate(
        &self,
        spec: &QualityGateSpec,
        cancel: &CancellationToken,
    ) -> Result<GateResult, OrchestratorError> {
        if cancel.is_cancelled() {
            return Err(OrchestratorError::Cancelled);
        }
        let mut lock = self.results.lock().await;
        if let Some(pos) = lock.iter().position(|r| r.name == spec.name) {
            Ok(lock.remove(pos))
        } else {
            Ok(GateResult {
                kind: spec.kind,
                name: spec.name.clone(),
                passed: true,
                exit_code: Some(0),
                stdout: "default mock pass".into(),
                stderr: String::new(),
                duration_ms: 5,
            })
        }
    }
}

/// Verdict returned by the peer reviewer subagent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewVerdict {
    /// Reviewer approves the implementation.
    Approved { summary: String },
    /// Reviewer requests modifications with critique.
    ChangesRequested {
        critique: String,
        remediation_prompt: String,
    },
}

/// Real-time lifecycle phase of the orchestration pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelinePhase {
    /// Prompt ingestion and intent triage.
    Triaging,
    /// Builder subagent actively generating or modifying code.
    Building { attempt: usize },
    /// Running automated verification quality gates.
    Verifying {
        attempt: usize,
        current_gate: Option<String>,
    },
    /// Adversarial peer review evaluating verified artifacts.
    Reviewing { attempt: usize },
    /// Subagent or pipeline suspended on a blocker awaiting resolution.
    Blocked { reason: String, prompt: String },
    /// Orchestration pipeline successfully completed all quality gates and review.
    Completed,
    /// Orchestration pipeline failed permanently.
    Failed { reason: String },
    /// Cancelled cooperatively by user or parent token.
    Cancelled,
}

impl PipelinePhase {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed { .. } | Self::Cancelled
        )
    }
}

/// Real-time broadcast event emitted as the orchestrator progresses.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrchestratorEvent {
    pub mission_id: String,
    pub timestamp_ms: u64,
    pub phase: PipelinePhase,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

/// The terminal outcome of an orchestrated task.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskOutcome {
    pub mission_id: String,
    pub route: IntentRoute,
    pub final_phase: PipelinePhase,
    pub builder_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification_report: Option<VerificationReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub review_verdict: Option<ReviewVerdict>,
    pub attempts: usize,
    pub usage: Usage,
    pub duration_ms: u64,
}

/// Configuration options for the [`Orchestrator`].
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Maximum builder remediation attempts (e.g. 3).
    pub max_remediation_attempts: usize,
    /// Overall mission timeout in seconds.
    pub total_timeout_secs: u64,
    /// Verification quality gates executed sequentially.
    pub gates: Vec<QualityGateSpec>,
    /// Model selector override for builder subagents.
    pub default_model: Option<String>,
    /// Model selector override for reviewer subagents.
    pub reviewer_model: Option<String>,
    /// Repository whose change graph records a verified checkpoint when a task
    /// is approved. `None` disables the checkpoint.
    pub repo_root: Option<PathBuf>,
    /// Resource governance for the quality gates. `None` runs them as bare
    /// spawns; gates are the heaviest commands the orchestrator runs.
    pub gate_limits: Option<ResourceLimits>,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_remediation_attempts: 3,
            total_timeout_secs: 600,
            gates: vec![
                QualityGateSpec::new(QualityGateKind::Format, "format-check", "cargo")
                    .with_args(["fmt", "--", "--check"]),
                QualityGateSpec::new(QualityGateKind::Build, "build-check", "cargo")
                    .with_args(["check", "--all-targets"]),
                QualityGateSpec::new(QualityGateKind::Test, "test-suite", "cargo")
                    .with_args(["test"]),
            ],
            default_model: None,
            reviewer_model: None,
            repo_root: None,
            gate_limits: None,
        }
    }
}

/// Errors surfaced during orchestration.
#[derive(Debug, thiserror::Error)]
pub enum OrchestratorError {
    #[error("intent triage error: {0}")]
    Triage(String),
    #[error("subagent error: {0}")]
    Subagent(#[from] SubagentError),
    #[error("verification error: {0}")]
    Verification(String),
    #[error("peer review error: {0}")]
    Review(String),
    #[error("orchestration cancelled")]
    Cancelled,
    #[error("orchestration timed out: {0}")]
    Timeout(String),
    #[error("configuration error: {0}")]
    Config(String),
}

/// Fluent builder for [`Orchestrator`].
pub struct OrchestratorBuilder {
    judge: Option<Arc<dyn Judge>>,
    subagents: Option<Arc<SubagentManager>>,
    verifier: Option<Arc<dyn QualityGateRunner>>,
    config: OrchestratorConfig,
}

impl OrchestratorBuilder {
    pub fn new() -> Self {
        Self {
            judge: None,
            subagents: None,
            verifier: None,
            config: OrchestratorConfig::default(),
        }
    }

    pub fn with_judge(mut self, judge: Arc<dyn Judge>) -> Self {
        self.judge = Some(judge);
        self
    }

    pub fn with_subagents(mut self, subagents: Arc<SubagentManager>) -> Self {
        self.subagents = Some(subagents);
        self
    }

    pub fn with_verifier(mut self, verifier: Arc<dyn QualityGateRunner>) -> Self {
        self.verifier = Some(verifier);
        self
    }

    pub fn with_config(mut self, config: OrchestratorConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_max_remediation_attempts(mut self, attempts: usize) -> Self {
        self.config.max_remediation_attempts = attempts.max(1);
        self
    }

    pub fn with_gate(mut self, gate: QualityGateSpec) -> Self {
        self.config.gates.push(gate);
        self
    }

    pub fn with_gates(mut self, gates: Vec<QualityGateSpec>) -> Self {
        self.config.gates = gates;
        self
    }

    pub fn build(self) -> Result<Orchestrator, OrchestratorError> {
        let judge = self.judge.unwrap_or_else(|| Arc::new(RuleBasedJudge));
        let verifier = self.verifier.unwrap_or_else(|| {
            Arc::new(CommandGateRunner {
                limits: self.config.gate_limits.clone(),
            })
        });
        let subagents = self
            .subagents
            .ok_or_else(|| OrchestratorError::Config("subagents manager is required".into()))?;

        let (events_tx, _events_rx) = broadcast::channel(128);

        Ok(Orchestrator {
            judge,
            subagents,
            verifier,
            config: self.config,
            events_tx,
            cancellation: CancellationToken::new(),
        })
    }
}

impl Default for OrchestratorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The multi-agent orchestrator managing the DAG pipeline, quality gates, and peer review.
pub struct Orchestrator {
    judge: Arc<dyn Judge>,
    subagents: Arc<SubagentManager>,
    verifier: Arc<dyn QualityGateRunner>,
    config: OrchestratorConfig,
    events_tx: broadcast::Sender<OrchestratorEvent>,
    cancellation: CancellationToken,
}

impl Orchestrator {
    pub fn builder() -> OrchestratorBuilder {
        OrchestratorBuilder::new()
    }

    pub fn new(
        judge: Arc<dyn Judge>,
        subagents: Arc<SubagentManager>,
        verifier: Arc<dyn QualityGateRunner>,
        config: OrchestratorConfig,
    ) -> Self {
        let (events_tx, _events_rx) = broadcast::channel(128);
        Self {
            judge,
            subagents,
            verifier,
            config,
            events_tx,
            cancellation: CancellationToken::new(),
        }
    }

    /// Subscribes to real-time orchestrator lifecycle events.
    pub fn subscribe(&self) -> broadcast::Receiver<OrchestratorEvent> {
        self.events_tx.subscribe()
    }

    /// Triggers cooperative cancellation of all running tasks managed by this orchestrator.
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    /// Classifies an incoming intent into an [`IntentRoute`] using the configured [`Judge`].
    pub async fn triage_intent(&self, intent: &str) -> Result<IntentRoute, OrchestratorError> {
        let options = [
            Choice::new(
                "direct_reply",
                "Answer directly with explanation or conversational clarification; no subagent needed",
            ),
            Choice::new(
                "explore_scout",
                "Read-only exploration, file search, or answering codebase questions using scout agent",
            ),
            Choice::new(
                "mechanical_edit",
                "Small localized refactor, single-file typo or syntax fix using sonic agent",
            ),
            Choice::new(
                "end_to_end_task",
                "Full multi-step implementation pipeline requiring builder, quality gates, and peer review",
            ),
        ];

        let decision = self
            .judge
            .choose(intent, &options)
            .await
            .map_err(|err| OrchestratorError::Triage(err.to_string()))?;

        match decision.key.as_str() {
            "direct_reply" => Ok(IntentRoute::DirectReply),
            "explore_scout" => Ok(IntentRoute::ExploreScout),
            "mechanical_edit" => Ok(IntentRoute::MechanicalEdit),
            "end_to_end_task" => Ok(IntentRoute::EndToEndTask),
            other => Err(OrchestratorError::Triage(format!(
                "unrecognized triage choice key: {other}"
            ))),
        }
    }

    /// Executes the full orchestrated pipeline for an incoming user intent.
    pub async fn execute_task(&self, intent: &str) -> Result<TaskOutcome, OrchestratorError> {
        self.execute_task_with_cancel(intent, self.cancellation.child_token())
            .await
    }

    /// Executes the orchestrated pipeline with a specific cancellation token.
    pub async fn execute_task_with_cancel(
        &self,
        intent: &str,
        cancel: CancellationToken,
    ) -> Result<TaskOutcome, OrchestratorError> {
        let start = Instant::now();
        let mission_id = Uuid::now_v7().to_string();

        // 1. Triage Intent
        self.emit_event(
            &mission_id,
            PipelinePhase::Triaging,
            "Triaging user intent",
            None,
        );

        if cancel.is_cancelled() {
            return self.cancelled_outcome(&mission_id, IntentRoute::DirectReply, start);
        }

        let route = self.triage_intent(intent).await?;
        self.emit_event(
            &mission_id,
            PipelinePhase::Triaging,
            format!("Triage complete: routed to {route:?}"),
            Some(serde_json::json!({ "route": route })),
        );

        match route {
            IntentRoute::DirectReply => {
                let outcome = TaskOutcome {
                    mission_id: mission_id.clone(),
                    route,
                    final_phase: PipelinePhase::Completed,
                    builder_output: Some(format!("Direct reply for: {intent}")),
                    verification_report: None,
                    review_verdict: None,
                    attempts: 1,
                    usage: Usage::default(),
                    duration_ms: start.elapsed().as_millis() as u64,
                };
                self.emit_event(
                    &mission_id,
                    PipelinePhase::Completed,
                    "Direct reply completed",
                    None,
                );
                Ok(outcome)
            }

            IntentRoute::ExploreScout => {
                let mut spec = TaskSpec::new(intent);
                spec.name = Some("scout".into());
                spec.model = self.config.default_model.clone();
                spec.read_only = true;

                let child_id = self.subagents.spawn(spec)?;
                let subagent_res = self.wait_child_with_cancel(&child_id, &cancel).await?;

                let outcome = TaskOutcome {
                    mission_id: mission_id.clone(),
                    route,
                    final_phase: PipelinePhase::Completed,
                    builder_output: Some(subagent_res.output),
                    verification_report: None,
                    review_verdict: None,
                    attempts: 1,
                    usage: subagent_res.usage,
                    duration_ms: start.elapsed().as_millis() as u64,
                };
                self.emit_event(
                    &mission_id,
                    PipelinePhase::Completed,
                    "Scout exploration completed",
                    None,
                );
                Ok(outcome)
            }

            IntentRoute::MechanicalEdit => {
                let mut spec = TaskSpec::new(intent);
                spec.name = Some("sonic".into());
                spec.model = self.config.default_model.clone();
                spec.read_only = false;

                let child_id = self.subagents.spawn(spec)?;
                let subagent_res = self.wait_child_with_cancel(&child_id, &cancel).await?;

                // Run verification gates once for mechanical edits
                let report = self.run_verification(&mission_id, 1, &cancel).await?;
                let final_phase = if report.passed {
                    PipelinePhase::Completed
                } else {
                    PipelinePhase::Failed {
                        reason: "Verification failed for mechanical edit".into(),
                    }
                };

                let outcome = TaskOutcome {
                    mission_id: mission_id.clone(),
                    route,
                    final_phase: final_phase.clone(),
                    builder_output: Some(subagent_res.output),
                    verification_report: Some(report),
                    review_verdict: None,
                    attempts: 1,
                    usage: subagent_res.usage,
                    duration_ms: start.elapsed().as_millis() as u64,
                };
                self.emit_event(&mission_id, final_phase, "Mechanical edit finished", None);
                Ok(outcome)
            }

            IntentRoute::EndToEndTask => {
                self.execute_e2e_dag(&mission_id, intent, start, cancel)
                    .await
            }
        }
    }

    /// Executes the full Builder -> Verifier -> Reviewer Manager DAG pipeline.
    async fn execute_e2e_dag(
        &self,
        mission_id: &str,
        intent: &str,
        start: Instant,
        cancel: CancellationToken,
    ) -> Result<TaskOutcome, OrchestratorError> {
        let mut total_usage = Usage::default();
        let mut last_builder_output = String::new();
        let mut last_report: Option<VerificationReport> = None;
        let mut last_verdict: Option<ReviewVerdict> = None;
        let mut remediation_prompt: Option<String> = None;

        for attempt in 1..=self.config.max_remediation_attempts {
            if cancel.is_cancelled() {
                return self.cancelled_outcome(mission_id, IntentRoute::EndToEndTask, start);
            }

            // --- Step A: Builder Subagent ---
            self.emit_event(
                mission_id,
                PipelinePhase::Building { attempt },
                format!(
                    "Builder running (attempt {attempt}/{})",
                    self.config.max_remediation_attempts
                ),
                None,
            );

            let builder_task = match &remediation_prompt {
                Some(remediation) => format!("{intent}\n\n[REMEDIATION FEEDBACK]:\n{remediation}"),
                None => intent.to_string(),
            };

            let mut builder_spec = TaskSpec::new(builder_task);
            builder_spec.name = Some("builder".into());
            builder_spec.model = self.config.default_model.clone();
            builder_spec.read_only = false;
            builder_spec.isolated_workspace = true;

            let child_id = self.subagents.spawn(builder_spec)?;
            let builder_res = match self.wait_child_with_cancel(&child_id, &cancel).await {
                Ok(res) => res,
                Err(OrchestratorError::Cancelled) => {
                    return self.cancelled_outcome(mission_id, IntentRoute::EndToEndTask, start);
                }
                Err(err) => {
                    if attempt < self.config.max_remediation_attempts {
                        remediation_prompt = Some(format!("Builder execution failed: {err}"));
                        continue;
                    }
                    let phase = PipelinePhase::Failed {
                        reason: format!("Builder execution failed: {err}"),
                    };
                    self.emit_event(mission_id, phase.clone(), "Builder run failed", None);
                    return Ok(TaskOutcome {
                        mission_id: mission_id.to_string(),
                        route: IntentRoute::EndToEndTask,
                        final_phase: phase,
                        builder_output: None,
                        verification_report: None,
                        review_verdict: None,
                        attempts: attempt,
                        usage: total_usage,
                        duration_ms: start.elapsed().as_millis() as u64,
                    });
                }
            };

            accumulate_usage(&mut total_usage, &builder_res.usage);
            last_builder_output = builder_res.output;

            if cancel.is_cancelled() {
                return self.cancelled_outcome(mission_id, IntentRoute::EndToEndTask, start);
            }

            // --- Step B: Sequential Verification Quality Gates ---
            let report = self.run_verification(mission_id, attempt, &cancel).await?;
            let gates_passed = report.passed;
            last_report = Some(report.clone());

            if !gates_passed {
                if attempt < self.config.max_remediation_attempts {
                    self.emit_event(
                        mission_id,
                        PipelinePhase::Verifying {
                            attempt,
                            current_gate: None,
                        },
                        format!("Quality gates failed; requesting builder remediation (attempt {attempt})"),
                        Some(serde_json::to_value(&report).unwrap_or_default()),
                    );
                    remediation_prompt = Some(report.format_remediation_feedback());
                    continue;
                }

                // Exhausted remediation attempts
                let phase = PipelinePhase::Failed {
                    reason: "Quality gate verification exhausted all remediation attempts".into(),
                };
                self.emit_event(mission_id, phase.clone(), "Verification failed", None);
                return Ok(TaskOutcome {
                    mission_id: mission_id.to_string(),
                    route: IntentRoute::EndToEndTask,
                    final_phase: phase,
                    builder_output: Some(last_builder_output),
                    verification_report: last_report,
                    review_verdict: None,
                    attempts: attempt,
                    usage: total_usage,
                    duration_ms: start.elapsed().as_millis() as u64,
                });
            }

            // --- Step C: Peer Review Handoff ---
            self.emit_event(
                mission_id,
                PipelinePhase::Reviewing { attempt },
                format!("Peer review starting (attempt {attempt})"),
                None,
            );

            if cancel.is_cancelled() {
                return self.cancelled_outcome(mission_id, IntentRoute::EndToEndTask, start);
            }

            let review_prompt = format!(
                "Review the implementation against the original intent:\n\
                INTENT: {intent}\n\n\
                BUILDER SUMMARY:\n{last_builder_output}\n\n\
                VERIFICATION PASSED: All quality gates passed cleanly.\n\n\
                Evaluate code quality, test coverage, safety, and correctness. Approve or request changes."
            );

            let mut reviewer_spec = TaskSpec::new(review_prompt);
            reviewer_spec.name = Some("reviewer".into());
            reviewer_spec.model = self
                .config
                .reviewer_model
                .clone()
                .or_else(|| self.config.default_model.clone());
            reviewer_spec.read_only = true;

            let reviewer_id = self.subagents.spawn(reviewer_spec)?;
            let reviewer_res = match self.wait_child_with_cancel(&reviewer_id, &cancel).await {
                Ok(res) => res,
                Err(OrchestratorError::Cancelled) => {
                    return self.cancelled_outcome(mission_id, IntentRoute::EndToEndTask, start);
                }
                Err(err) => {
                    return Err(OrchestratorError::Review(format!(
                        "reviewer child execution failed: {err}"
                    )));
                }
            };

            accumulate_usage(&mut total_usage, &reviewer_res.usage);

            // Parse review verdict
            let verdict = parse_review_verdict(&reviewer_res.output);
            last_verdict = Some(verdict.clone());

            match verdict {
                ReviewVerdict::Approved { summary } => {
                    self.emit_event(
                        mission_id,
                        PipelinePhase::Completed,
                        format!("Task approved by reviewer: {summary}"),
                        None,
                    );
                    // The builder's isolated workspace names its thread after the
                    // subagent id, so its head is exactly the state the gates and
                    // the reviewer just approved.
                    self.mark_verified_checkpoint(&child_id, mission_id);
                    return Ok(TaskOutcome {
                        mission_id: mission_id.to_string(),
                        route: IntentRoute::EndToEndTask,
                        final_phase: PipelinePhase::Completed,
                        builder_output: Some(last_builder_output),
                        verification_report: last_report,
                        review_verdict: last_verdict,
                        attempts: attempt,
                        usage: total_usage,
                        duration_ms: start.elapsed().as_millis() as u64,
                    });
                }
                ReviewVerdict::ChangesRequested {
                    critique,
                    remediation_prompt: fix_req,
                } => {
                    if attempt < self.config.max_remediation_attempts {
                        self.emit_event(
                            mission_id,
                            PipelinePhase::Reviewing { attempt },
                            format!("Reviewer requested changes; re-routing to builder (attempt {attempt})"),
                            Some(serde_json::json!({ "critique": critique })),
                        );
                        remediation_prompt = Some(format!(
                            "Peer reviewer requested modifications:\n{critique}\n\nSpecific instructions: {fix_req}"
                        ));
                        continue;
                    }

                    // Remediation exhausted
                    let phase = PipelinePhase::Failed {
                        reason:
                            "Reviewer requested changes but remediation attempts were exhausted"
                                .into(),
                    };
                    self.emit_event(mission_id, phase.clone(), "Review changes exhausted", None);
                    return Ok(TaskOutcome {
                        mission_id: mission_id.to_string(),
                        route: IntentRoute::EndToEndTask,
                        final_phase: phase,
                        builder_output: Some(last_builder_output),
                        verification_report: last_report,
                        review_verdict: last_verdict,
                        attempts: attempt,
                        usage: total_usage,
                        duration_ms: start.elapsed().as_millis() as u64,
                    });
                }
            }
        }

        // Catch-all failure if loop terminated
        let phase = PipelinePhase::Failed {
            reason: "Maximum remediation attempts reached without passing gates and review".into(),
        };
        self.emit_event(mission_id, phase.clone(), "Orchestration failed", None);
        Ok(TaskOutcome {
            mission_id: mission_id.to_string(),
            route: IntentRoute::EndToEndTask,
            final_phase: phase,
            builder_output: Some(last_builder_output),
            verification_report: last_report,
            review_verdict: last_verdict,
            attempts: self.config.max_remediation_attempts,
            usage: total_usage,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Runs all configured verification quality gates sequentially.
    pub async fn run_verification(
        &self,
        mission_id: &str,
        attempt: usize,
        cancel: &CancellationToken,
    ) -> Result<VerificationReport, OrchestratorError> {
        let mut gate_results = Vec::new();

        for gate in &self.config.gates {
            if cancel.is_cancelled() {
                return Err(OrchestratorError::Cancelled);
            }

            self.emit_event(
                mission_id,
                PipelinePhase::Verifying {
                    attempt,
                    current_gate: Some(gate.name.clone()),
                },
                format!("Running gate '{}' ({})", gate.name, gate.kind),
                None,
            );

            let result = self.verifier.execute_gate(gate, cancel).await?;
            let passed = result.passed;
            gate_results.push(result);

            if !passed && !gate.optional {
                // Fail-fast on required gates
                break;
            }
        }

        Ok(VerificationReport::new(gate_results))
    }

    /// Awaits child completion while respecting cooperative cancellation.
    async fn wait_child_with_cancel(
        &self,
        child_id: &str,
        cancel: &CancellationToken,
    ) -> Result<SubagentResult, OrchestratorError> {
        tokio::select! {
            _ = cancel.cancelled() => {
                let _ = self.subagents.cancel(child_id);
                Err(OrchestratorError::Cancelled)
            }
            res = self.subagents.wait(child_id) => {
                match res {
                    Ok(result) => Ok(result),
                    Err(SubagentError::Cancelled) => Err(OrchestratorError::Cancelled),
                    Err(err) => Err(OrchestratorError::Subagent(err)),
                }
            }
        }
    }

    fn cancelled_outcome(
        &self,
        mission_id: &str,
        route: IntentRoute,
        start: Instant,
    ) -> Result<TaskOutcome, OrchestratorError> {
        self.emit_event(
            mission_id,
            PipelinePhase::Cancelled,
            "Mission cancelled",
            None,
        );
        Ok(TaskOutcome {
            mission_id: mission_id.to_string(),
            route,
            final_phase: PipelinePhase::Cancelled,
            builder_output: None,
            verification_report: None,
            review_verdict: None,
            attempts: 0,
            usage: Usage::default(),
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Freeze the builder's thread head as a verified checkpoint.
    ///
    /// The builder runs in an isolated workspace whose thread is named after its
    /// subagent id, so that thread's head is the state the quality gates and the
    /// reviewer just approved. Best-effort: a missing graph or thread must never
    /// fail an approved task, so a failure is reported and the task still
    /// completes.
    fn mark_verified_checkpoint(&self, thread: &str, mission_id: &str) {
        let Some(repo_root) = &self.config.repo_root else {
            return;
        };
        let result = (|| -> anyhow::Result<String> {
            let graph = ChangeGraph::open(repo_root)?;
            let head = graph
                .state()?
                .heads
                .get(thread)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("thread {thread:?} has no head"))?;
            let node = graph.mark(
                &head,
                &format!("verified-{mission_id}"),
                &format!("verified: {mission_id}"),
                true,
            )?;
            Ok(node.id)
        })();
        match result {
            Ok(id) => self.emit_event(
                mission_id,
                PipelinePhase::Completed,
                format!("Recorded verified checkpoint for node {id}"),
                None,
            ),
            Err(error) => {
                eprintln!("juto: failed to record a verified checkpoint: {error:#}");
            }
        }
    }

    fn emit_event(
        &self,
        mission_id: &str,
        phase: PipelinePhase,
        message: impl Into<String>,
        details: Option<Value>,
    ) {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let event = OrchestratorEvent {
            mission_id: mission_id.to_string(),
            timestamp_ms,
            phase,
            message: message.into(),
            details,
        };

        let _ = self.events_tx.send(event);
    }
}

fn accumulate_usage(dest: &mut Usage, src: &Usage) {
    dest.input = dest.input.saturating_add(src.input);
    dest.output = dest.output.saturating_add(src.output);
    dest.cache_read = dest.cache_read.saturating_add(src.cache_read);
    dest.cache_write = dest.cache_write.saturating_add(src.cache_write);
    dest.total_tokens = dest.total_tokens.saturating_add(src.total_tokens);
    dest.cost += src.cost;
}

fn parse_review_verdict(text: &str) -> ReviewVerdict {
    let lower = text.to_lowercase();
    if lower.contains("reject")
        || lower.contains("changes requested")
        || lower.contains("request changes")
        || lower.contains("issue found")
        || lower.contains("fails review")
    {
        ReviewVerdict::ChangesRequested {
            critique: text.to_string(),
            remediation_prompt: "Address reviewer issues and re-verify".into(),
        }
    } else {
        ReviewVerdict::Approved {
            summary: text.trim().to_string(),
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::subagents::SubagentFactory;
    use juto_agent::{AgentEvent, RunControl};
    use juto_ai::StopReason;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::mpsc;

    /// Mock subagent factory simulating different agent personas.
    #[derive(Clone)]
    struct MockSubagentFactory {
        builder_outputs: Arc<tokio::sync::Mutex<Vec<Result<String, String>>>>,
        reviewer_outputs: Arc<tokio::sync::Mutex<Vec<Result<String, String>>>>,
        runs: Arc<AtomicUsize>,
    }

    impl MockSubagentFactory {
        fn new(
            builder_outputs: Vec<Result<String, String>>,
            reviewer_outputs: Vec<Result<String, String>>,
        ) -> Self {
            Self {
                builder_outputs: Arc::new(tokio::sync::Mutex::new(builder_outputs)),
                reviewer_outputs: Arc::new(tokio::sync::Mutex::new(reviewer_outputs)),
                runs: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl SubagentFactory for MockSubagentFactory {
        async fn run(
            &self,
            id: &str,
            spec: TaskSpec,
            _control: RunControl,
            _events: mpsc::UnboundedSender<AgentEvent>,
        ) -> Result<SubagentResult, String> {
            self.runs.fetch_add(1, Ordering::SeqCst);
            let persona = spec.name.as_deref().unwrap_or("unknown");
            let output_res = if persona == "reviewer" {
                let mut queue = self.reviewer_outputs.lock().await;
                if queue.is_empty() {
                    Ok("Looks good! Approved.".into())
                } else {
                    queue.remove(0)
                }
            } else {
                let mut queue = self.builder_outputs.lock().await;
                if queue.is_empty() {
                    Ok(format!("Completed task: {}", spec.task))
                } else {
                    queue.remove(0)
                }
            };

            match output_res {
                Ok(output) => Ok(SubagentResult {
                    id: id.to_string(),
                    output,
                    data: None,
                    usage: Usage {
                        input: 100,
                        output: 50,
                        cache_read: 0,
                        cache_write: 0,
                        total_tokens: 150,
                        cost: 0.001,
                    },
                    stop_reason: StopReason::Stop,
                    session_path: PathBuf::from("/tmp/session"),
                }),
                Err(err) => Err(err),
            }
        }
    }

    fn test_manager(factory: MockSubagentFactory) -> Arc<SubagentManager> {
        Arc::new(SubagentManager::new(
            Arc::new(factory),
            4,
            CancellationToken::new(),
        ))
    }

    #[tokio::test]
    async fn test_rule_based_judge_triage() {
        let judge = RuleBasedJudge;

        let options = [
            Choice::new("direct_reply", "direct"),
            Choice::new("explore_scout", "scout"),
            Choice::new("mechanical_edit", "sonic"),
            Choice::new("end_to_end_task", "builder"),
        ];

        let d1 = judge.choose("hello", &options).await.unwrap();
        assert_eq!(d1.key, "direct_reply");

        let d2 = judge
            .choose("find all occurrences of foo", &options)
            .await
            .unwrap();
        assert_eq!(d2.key, "explore_scout");

        let d3 = judge
            .choose("rename variable foo to bar", &options)
            .await
            .unwrap();
        assert_eq!(d3.key, "mechanical_edit");

        let d4 = judge
            .choose(
                "implement OAuth login provider with refresh token rotation",
                &options,
            )
            .await
            .unwrap();
        assert_eq!(d4.key, "end_to_end_task");
    }

    #[tokio::test]
    async fn test_direct_reply_workflow() {
        let factory = MockSubagentFactory::new(vec![], vec![]);
        let runs = factory.runs.clone();
        let subagents = test_manager(factory);

        let orchestrator = Orchestrator::builder()
            .with_judge(Arc::new(ScriptedJudge::single("direct_reply")))
            .with_subagents(subagents)
            .build()
            .unwrap();

        let outcome = orchestrator.execute_task("hello there").await.unwrap();
        assert_eq!(outcome.route, IntentRoute::DirectReply);
        assert_eq!(outcome.final_phase, PipelinePhase::Completed);
        assert!(outcome.builder_output.is_some());
        // Direct reply does not spawn subagent runs
        assert_eq!(runs.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn test_explore_scout_workflow() {
        let factory = MockSubagentFactory::new(vec![Ok("Found 3 files".into())], vec![]);
        let runs = factory.runs.clone();
        let subagents = test_manager(factory);

        let orchestrator = Orchestrator::builder()
            .with_judge(Arc::new(ScriptedJudge::single("explore_scout")))
            .with_subagents(subagents)
            .build()
            .unwrap();

        let outcome = orchestrator
            .execute_task("search for authentication")
            .await
            .unwrap();
        assert_eq!(outcome.route, IntentRoute::ExploreScout);
        assert_eq!(outcome.final_phase, PipelinePhase::Completed);
        assert_eq!(outcome.builder_output.as_deref(), Some("Found 3 files"));
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_mechanical_edit_workflow() {
        let factory = MockSubagentFactory::new(vec![Ok("Fixed typo".into())], vec![]);
        let subagents = test_manager(factory);

        let orchestrator = Orchestrator::builder()
            .with_judge(Arc::new(ScriptedJudge::single("mechanical_edit")))
            .with_subagents(subagents)
            .with_verifier(Arc::new(MockGateRunner::passing()))
            .build()
            .unwrap();

        let outcome = orchestrator
            .execute_task("fix typo in readme")
            .await
            .unwrap();
        assert_eq!(outcome.route, IntentRoute::MechanicalEdit);
        assert_eq!(outcome.final_phase, PipelinePhase::Completed);
        assert!(outcome.verification_report.as_ref().unwrap().passed);
    }

    #[tokio::test]
    async fn test_e2e_dag_happy_path() {
        let factory = MockSubagentFactory::new(
            vec![Ok("Implemented auth feature".into())],
            vec![Ok("Code looks clean and tests pass. Approved.".into())],
        );
        let runs = factory.runs.clone();
        let subagents = test_manager(factory);

        let orchestrator = Orchestrator::builder()
            .with_judge(Arc::new(ScriptedJudge::single("end_to_end_task")))
            .with_subagents(subagents)
            .with_verifier(Arc::new(MockGateRunner::passing()))
            .build()
            .unwrap();

        let mut events_rx = orchestrator.subscribe();

        let outcome = orchestrator
            .execute_task("build auth feature")
            .await
            .unwrap();
        assert_eq!(outcome.route, IntentRoute::EndToEndTask);
        assert_eq!(outcome.final_phase, PipelinePhase::Completed);
        assert_eq!(outcome.attempts, 1);
        assert!(outcome.verification_report.as_ref().unwrap().passed);
        assert!(matches!(
            outcome.review_verdict,
            Some(ReviewVerdict::Approved { .. })
        ));
        // Builder (1) + Reviewer (1) = 2 runs
        assert_eq!(runs.load(Ordering::SeqCst), 2);

        // Verify events were emitted
        let mut phases = Vec::new();
        while let Ok(event) = events_rx.try_recv() {
            phases.push(event.phase);
        }
        assert!(phases.contains(&PipelinePhase::Triaging));
        assert!(phases.contains(&PipelinePhase::Building { attempt: 1 }));
        assert!(phases.contains(&PipelinePhase::Reviewing { attempt: 1 }));
        assert!(phases.contains(&PipelinePhase::Completed));
    }

    #[tokio::test]
    async fn test_e2e_dag_verification_remediation_loop() {
        // Attempt 1: Builder writes code, verifier fails test gate
        // Attempt 2: Builder fixes code, verifier passes, reviewer approves
        let factory = MockSubagentFactory::new(
            vec![
                Ok("Initial implementation".into()),
                Ok("Fixed test failures in auth".into()),
            ],
            vec![Ok("All checks pass now. Approved.".into())],
        );
        let subagents = test_manager(factory);

        // Verifier: first call fails, second call passes
        let verifier = Arc::new(MockGateRunner::new(vec![
            GateResult {
                kind: QualityGateKind::Test,
                name: "test-suite".into(),
                passed: false,
                exit_code: Some(101),
                stdout: String::new(),
                stderr: "test failed: assertion violated".into(),
                duration_ms: 10,
            },
            GateResult {
                kind: QualityGateKind::Test,
                name: "test-suite".into(),
                passed: true,
                exit_code: Some(0),
                stdout: "all tests passed".into(),
                stderr: String::new(),
                duration_ms: 10,
            },
        ]));

        let orchestrator = Orchestrator::builder()
            .with_judge(Arc::new(ScriptedJudge::single("end_to_end_task")))
            .with_subagents(subagents)
            .with_verifier(verifier)
            .with_gates(vec![QualityGateSpec::new(
                QualityGateKind::Test,
                "test-suite",
                "cargo",
            )])
            .build()
            .unwrap();

        let outcome = orchestrator
            .execute_task("implement feature with test")
            .await
            .unwrap();
        assert_eq!(outcome.final_phase, PipelinePhase::Completed);
        assert_eq!(outcome.attempts, 2);
        assert!(outcome.verification_report.as_ref().unwrap().passed);
        assert!(matches!(
            outcome.review_verdict,
            Some(ReviewVerdict::Approved { .. })
        ));
    }

    #[tokio::test]
    async fn test_e2e_dag_review_remediation_loop() {
        // Attempt 1: Builder writes code, verifier passes, Reviewer requests changes
        // Attempt 2: Builder addresses critique, verifier passes, Reviewer approves
        let factory = MockSubagentFactory::new(
            vec![
                Ok("Initial implementation".into()),
                Ok("Refactored error handling as requested".into()),
            ],
            vec![
                Ok("Changes requested: please handle error edge case".into()),
                Ok("Edge cases handled properly. Approved.".into()),
            ],
        );
        let subagents = test_manager(factory);

        let orchestrator = Orchestrator::builder()
            .with_judge(Arc::new(ScriptedJudge::single("end_to_end_task")))
            .with_subagents(subagents)
            .with_verifier(Arc::new(MockGateRunner::passing()))
            .build()
            .unwrap();

        let outcome = orchestrator
            .execute_task("implement edge case handling")
            .await
            .unwrap();
        assert_eq!(outcome.final_phase, PipelinePhase::Completed);
        assert_eq!(outcome.attempts, 2);
        assert!(matches!(
            outcome.review_verdict,
            Some(ReviewVerdict::Approved { .. })
        ));
    }

    #[tokio::test]
    async fn test_e2e_dag_retry_exhaustion() {
        // Repeatedly failing verification exhausts retries
        let factory =
            MockSubagentFactory::new(vec![Ok("Attempt 1".into()), Ok("Attempt 2".into())], vec![]);
        let subagents = test_manager(factory);

        let verifier = Arc::new(MockGateRunner::new(vec![
            GateResult {
                kind: QualityGateKind::Test,
                name: "test-suite".into(),
                passed: false,
                exit_code: Some(1),
                stdout: String::new(),
                stderr: "fatal error".into(),
                duration_ms: 5,
            },
            GateResult {
                kind: QualityGateKind::Test,
                name: "test-suite".into(),
                passed: false,
                exit_code: Some(1),
                stdout: String::new(),
                stderr: "fatal error again".into(),
                duration_ms: 5,
            },
        ]));

        let orchestrator = Orchestrator::builder()
            .with_judge(Arc::new(ScriptedJudge::single("end_to_end_task")))
            .with_subagents(subagents)
            .with_verifier(verifier)
            .with_max_remediation_attempts(2)
            .with_gates(vec![QualityGateSpec::new(
                QualityGateKind::Test,
                "test-suite",
                "cargo",
            )])
            .build()
            .unwrap();

        let outcome = orchestrator.execute_task("impossible task").await.unwrap();
        assert!(matches!(outcome.final_phase, PipelinePhase::Failed { .. }));
        assert_eq!(outcome.attempts, 2);
        assert!(!outcome.verification_report.as_ref().unwrap().passed);
    }

    #[tokio::test]
    async fn test_cancellation_during_execution() {
        let factory = MockSubagentFactory::new(vec![], vec![]);
        let subagents = test_manager(factory);

        let cancel = CancellationToken::new();
        cancel.cancel(); // Pre-cancelled

        let orchestrator = Orchestrator::builder()
            .with_judge(Arc::new(ScriptedJudge::single("end_to_end_task")))
            .with_subagents(subagents)
            .build()
            .unwrap();

        let outcome = orchestrator
            .execute_task_with_cancel("cancelled task", cancel)
            .await
            .unwrap();
        assert_eq!(outcome.final_phase, PipelinePhase::Cancelled);
    }

    #[tokio::test]
    async fn test_command_gate_runner_real_process() {
        let runner = CommandGateRunner { limits: None };
        let cancel = CancellationToken::new();

        // Successful command
        let spec_pass = QualityGateSpec::new(QualityGateKind::Smoke, "echo-test", "echo")
            .with_args(["juto_smoke_proof"]);
        let res_pass = runner.execute_gate(&spec_pass, &cancel).await.unwrap();
        assert!(res_pass.passed);
        assert!(res_pass.stdout.contains("juto_smoke_proof"));

        // Failing command
        let spec_fail = QualityGateSpec::new(QualityGateKind::Test, "false-test", "false");
        let res_fail = runner.execute_gate(&spec_fail, &cancel).await.unwrap();
        assert!(!res_fail.passed);
    }

    /// Live proof that a governed gate runs inside a systemd user scope.
    ///
    /// Gates are the heaviest commands the orchestrator runs, so an uncapped gate
    /// can exhaust the host.
    #[tokio::test]
    async fn a_governed_gate_runs_under_a_systemd_scope() {
        if !crate::workspace::is_systemd_run_available() {
            eprintln!("skipping governed-gate test: systemd-run unavailable");
            return;
        }
        let runner = CommandGateRunner {
            limits: Some(ResourceLimits::default().with_enforce_scope(false)),
        };
        let cancel = CancellationToken::new();
        let spec = QualityGateSpec::new(QualityGateKind::Build, "cgroup", "cat")
            .with_args(["/proc/self/cgroup"]);
        let result = runner.execute_gate(&spec, &cancel).await.expect("gate");
        assert!(result.passed, "{result:?}");
        assert!(
            result.stdout.contains(".scope"),
            "expected a systemd scope cgroup, got: {}",
            result.stdout
        );
    }

    #[tokio::test]
    async fn test_verification_report_formatting() {
        let report = VerificationReport::new(vec![
            GateResult {
                kind: QualityGateKind::Format,
                name: "format-check".into(),
                passed: true,
                exit_code: Some(0),
                stdout: String::new(),
                stderr: String::new(),
                duration_ms: 10,
            },
            GateResult {
                kind: QualityGateKind::Test,
                name: "test-suite".into(),
                passed: false,
                exit_code: Some(101),
                stdout: "run 5 passed".into(),
                stderr: "FAILED: test_auth".into(),
                duration_ms: 25,
            },
        ]);

        assert!(!report.passed);
        assert_eq!(report.summary, "1/2 quality gates passed");
        let feedback = report.format_remediation_feedback();
        assert!(feedback.contains("FAILED: test_auth"));
        assert!(feedback.contains("[FAILED] Gate: test-suite (test)"));
    }
}
