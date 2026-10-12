//! Observable smoke test for the End-to-End Orchestrator and Peer Review DAG (Slice E).

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use juto_agent::{AgentEvent, RunControl};
use juto_ai::{StopReason, Usage};
use juto_runtime::change_graph::{ChangeGraph, EditProvenance};
use juto_runtime::orchestrator::{
    CommandGateRunner, IntentRoute, Orchestrator, OrchestratorConfig, PipelinePhase,
    QualityGateKind, QualityGateSpec, ReviewVerdict, RuleBasedJudge, ScriptedJudge,
};
use juto_runtime::subagents::{SubagentFactory, SubagentManager, SubagentResult, TaskSpec};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Clone)]
struct SmokeSubagentFactory {
    runs: Arc<AtomicUsize>,
}

#[async_trait]
impl SubagentFactory for SmokeSubagentFactory {
    async fn run(
        &self,
        id: &str,
        spec: TaskSpec,
        _control: RunControl,
        _events: mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<SubagentResult, String> {
        let count = self.runs.fetch_add(1, Ordering::SeqCst);
        let persona = spec.name.as_deref().unwrap_or("unknown");
        println!("[SMOKE-FACTORY] Run #{count} spawned child '{id}' for persona '{persona}'");

        let output = if persona == "reviewer" {
            "Adversarial review complete: code meets all quality standards and invariants. Approved.".to_string()
        } else if persona == "scout" {
            "Found 4 relevant files in workspace.".to_string()
        } else if persona == "sonic" {
            "Mechanical edit applied cleanly.".to_string()
        } else {
            format!("Builder implemented task: {}", spec.task)
        };

        Ok(SubagentResult {
            id: id.to_string(),
            output,
            data: None,
            usage: Usage {
                input: 120,
                output: 45,
                cache_read: 0,
                cache_write: 0,
                total_tokens: 165,
                cost: 0.0012,
            },
            stop_reason: StopReason::Stop,
            session_path: PathBuf::from(format!("/tmp/juto-smoke-session-{id}")),
        })
    }
}

#[tokio::test]
async fn smoke_e2e_dag_with_real_command_gates_and_subagents() {
    println!("\n=== STARTING SMOKE TEST: Orchestrator E2E DAG with Real Command Gates ===");

    let factory = SmokeSubagentFactory {
        runs: Arc::new(AtomicUsize::new(0)),
    };
    let runs_counter = factory.runs.clone();
    let subagents = Arc::new(SubagentManager::new(
        Arc::new(factory),
        4,
        CancellationToken::new(),
    ));

    // Configure real command gates using basic echo commands to prove real process spawning
    let gates = vec![
        QualityGateSpec::new(QualityGateKind::Format, "smoke-fmt", "echo")
            .with_args(["format_check_passed"]),
        QualityGateSpec::new(QualityGateKind::Build, "smoke-build", "echo")
            .with_args(["cargo_build_passed"]),
        QualityGateSpec::new(QualityGateKind::Test, "smoke-test", "echo")
            .with_args(["cargo_test_passed"]),
        QualityGateSpec::new(QualityGateKind::Smoke, "smoke-proof", "echo")
            .with_args(["observable_smoke_proof_captured"]),
    ];

    let orchestrator = Orchestrator::builder()
        .with_judge(Arc::new(ScriptedJudge::single("end_to_end_task")))
        .with_subagents(subagents)
        .with_verifier(Arc::new(CommandGateRunner { limits: None }))
        .with_gates(gates)
        .with_max_remediation_attempts(3)
        .build()
        .expect("orchestrator should build");

    let mut event_rx = orchestrator.subscribe();

    let task_intent = "Implement OAuth token refresh with rotation and tests";
    let outcome = orchestrator
        .execute_task(task_intent)
        .await
        .expect("orchestrator task execution should succeed");

    println!("[SMOKE-PROOF] Mission ID: {}", outcome.mission_id);
    println!("[SMOKE-PROOF] Route: {:?}", outcome.route);
    println!("[SMOKE-PROOF] Final Phase: {:?}", outcome.final_phase);
    println!("[SMOKE-PROOF] Total Attempts: {}", outcome.attempts);
    println!("[SMOKE-PROOF] Duration: {}ms", outcome.duration_ms);
    println!("[SMOKE-PROOF] Total Tokens: {}", outcome.usage.total_tokens);

    // Verify Invariants:
    assert_eq!(outcome.route, IntentRoute::EndToEndTask);
    assert_eq!(outcome.final_phase, PipelinePhase::Completed);
    assert_eq!(outcome.attempts, 1);
    assert!(outcome.builder_output.is_some());

    // Verify verification quality gates:
    let report = outcome
        .verification_report
        .as_ref()
        .expect("verification report must be present");
    assert!(report.passed);
    assert_eq!(report.gate_results.len(), 4);
    for gate in &report.gate_results {
        assert!(gate.passed);
        println!(
            "[SMOKE-GATE] {} ({}): exit_code={:?}, duration={}ms, stdout={}",
            gate.name,
            gate.kind,
            gate.exit_code,
            gate.duration_ms,
            gate.stdout.trim()
        );
    }

    // Verify peer review verdict:
    let review = outcome
        .review_verdict
        .as_ref()
        .expect("review verdict must be present");
    assert!(matches!(review, ReviewVerdict::Approved { .. }));
    println!("[SMOKE-REVIEW] Verdict: {:?}", review);

    // Verify 2 subagent runs executed (Builder + Reviewer)
    assert_eq!(runs_counter.load(Ordering::SeqCst), 2);

    // Verify event stream captured all chronological lifecycle phases
    let mut recorded_phases = Vec::new();
    while let Ok(event) = event_rx.try_recv() {
        println!("[SMOKE-EVENT] Phase={:?}: {}", event.phase, event.message);
        recorded_phases.push(event.phase);
    }

    assert!(recorded_phases.contains(&PipelinePhase::Triaging));
    assert!(recorded_phases.contains(&PipelinePhase::Building { attempt: 1 }));
    assert!(
        recorded_phases
            .iter()
            .any(|p| matches!(p, PipelinePhase::Verifying { attempt: 1, .. }))
    );
    assert!(recorded_phases.contains(&PipelinePhase::Reviewing { attempt: 1 }));
    assert!(recorded_phases.contains(&PipelinePhase::Completed));

    println!("=== SMOKE TEST PASSED: ALL INVARIANTS OBSERVED ===\n");
}

#[tokio::test]
async fn smoke_triage_routing_matrix() {
    let judge = RuleBasedJudge;

    let routes = [
        ("what is the project structure?", IntentRoute::ExploreScout),
        (
            "find all usages of SubagentManager",
            IntentRoute::ExploreScout,
        ),
        (
            "rename auth_token to access_token",
            IntentRoute::MechanicalEdit,
        ),
        ("format the imports", IntentRoute::MechanicalEdit),
        ("hello", IntentRoute::DirectReply),
        (
            "build a distributed lock manager with tests",
            IntentRoute::EndToEndTask,
        ),
    ];

    let factory = SmokeSubagentFactory {
        runs: Arc::new(AtomicUsize::new(0)),
    };
    let subagents = Arc::new(SubagentManager::new(
        Arc::new(factory),
        2,
        CancellationToken::new(),
    ));

    let orchestrator = Orchestrator::builder()
        .with_judge(Arc::new(judge))
        .with_subagents(subagents)
        .build()
        .unwrap();

    for (prompt, expected) in routes {
        let route = orchestrator.triage_intent(prompt).await.unwrap();
        assert_eq!(
            route, expected,
            "Prompt '{prompt}' expected {expected:?}, got {route:?}"
        );
        println!("[SMOKE-TRIAGE] '{prompt}' => {:?}", route);
    }
}

/// A factory that records a real change-graph node on the child's thread, the
/// way an isolated builder workspace does.
#[derive(Clone)]
struct GraphSubagentFactory {
    repo: PathBuf,
}

#[async_trait]
impl SubagentFactory for GraphSubagentFactory {
    async fn run(
        &self,
        id: &str,
        spec: TaskSpec,
        _control: RunControl,
        _events: mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<SubagentResult, String> {
        let persona = spec.name.as_deref().unwrap_or("unknown");
        if persona == "builder" {
            let repo = self.repo.clone();
            let thread = id.to_string();
            tokio::task::spawn_blocking(move || -> Result<(), String> {
                let graph = ChangeGraph::open(&repo).map_err(|e| e.to_string())?;
                graph
                    .capture(
                        &thread,
                        EditProvenance {
                            session_id: "smoke".into(),
                            tool: "builder".into(),
                            selection_mode: None,
                            path: None,
                        },
                        None,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(())
            })
            .await
            .map_err(|e| e.to_string())??;
        }

        let output = if persona == "reviewer" {
            "Adversarial review complete: approved.".to_string()
        } else {
            format!("Builder implemented task: {}", spec.task)
        };
        Ok(SubagentResult {
            id: id.to_string(),
            output,
            data: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            session_path: PathBuf::from(format!("/tmp/juto-smoke-session-{id}")),
        })
    }
}

#[tokio::test]
async fn an_approved_task_records_a_verified_checkpoint() {
    let temp = tempfile::tempdir().expect("tempdir");
    let repo = temp.path().join("repo");
    std::fs::create_dir_all(&repo).expect("create repo");
    let git = |args: &[&str]| {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(&repo)
            .status()
            .expect("run git");
        assert!(status.success(), "git {args:?}");
    };
    git(&["init", "-q", "-b", "main"]);
    git(&["config", "user.name", "Smoke"]);
    git(&["config", "user.email", "smoke@example.test"]);
    std::fs::write(repo.join("f.txt"), "base\n").expect("write");
    git(&["add", "-A"]);
    git(&["commit", "-q", "-m", "base"]);
    ChangeGraph::open(&repo).expect("init graph");

    let subagents = Arc::new(SubagentManager::new(
        Arc::new(GraphSubagentFactory { repo: repo.clone() }),
        4,
        CancellationToken::new(),
    ));

    let config = OrchestratorConfig {
        repo_root: Some(repo.clone()),
        gates: vec![QualityGateSpec::new(QualityGateKind::Build, "ok", "echo").with_args(["ok"])],
        ..OrchestratorConfig::default()
    };

    let orchestrator = Orchestrator::builder()
        .with_judge(Arc::new(ScriptedJudge::single("end_to_end_task")))
        .with_subagents(subagents)
        .with_verifier(Arc::new(CommandGateRunner { limits: None }))
        .with_config(config)
        .build()
        .expect("orchestrator should build");

    let outcome = orchestrator
        .execute_task("implement the thing")
        .await
        .expect("task execution");
    assert_eq!(outcome.final_phase, PipelinePhase::Completed);

    // The builder's thread head is frozen as a verified checkpoint.
    let graph = ChangeGraph::open(&repo).expect("reopen graph");
    let state = graph.state().expect("state");
    let expected = format!("verified-{}", outcome.mission_id);
    assert!(
        state.checkpoints.contains_key(&expected),
        "expected checkpoint {expected}, got {:?}",
        state.checkpoints
    );

    // The approved work is folded back onto the parent thread, so it is not
    // stranded on a thread nobody reads.
    assert!(
        state.heads.contains_key("main"),
        "the builder's work must reach the parent thread: {:?}",
        state.heads
    );
}
