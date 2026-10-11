# Juto Orchestration & Multi-Agent Architecture Plan

Full detailed specification: [`docs/orchestration-plan.md`](docs/orchestration-plan.md).

## Quick Overview

1. **UX Hub**: Single human chat for strategic intent and direction; GPUI Sidebar
   "Mission Control" tracking subagent states (`Queued`, `Running`, `Blocked`,
   `Verifying`, `Reviewing`, `Done`), live diffs, and steering controls.
2. **Decision Layer (TypeSafe Jev / System One)**: Sub-50ms calibrated selection over
   discrete choices for intent triage, subagent persona routing, blocker taxonomy, and
   safety gating.
3. **Subagent Contract (OMP Style)**: Typed `TaskSpec` and `SubagentResult`, structured
   JSON schemas, in-process Tokio sessions, peer communication bus, and coworker roles
   (Manager, Builder, Reviewer, Scout, Sonic).
4. **Isolated Workspaces & Resource Governance**:
   * Isolated Jujutsu workspaces (`jj workspace add ~/workspaces/juto-<subagent-id>`).
   * Process and tool execution inside systemd user scopes (`agents.slice`), bounded
     CPU/RAM quotas, and GPU lock serialization (`scripts/agent-scope.sh` model).
   * Repo initialization step (`juto init`) to automatically adapt to repository rules
     and build systems.
5. **End-to-End Quality Gate & Zed Delta**:
   * E2E completion requires: code edits + formatting + build pass + deterministic test
     pass + observable smoke proof + automated peer review pass.
   * Zed Delta model: Every diff hunk is anchored to the conversational turns, tools,
     and reasoning that generated it.
6. **Future Extensibility**:
   * Headless runtime daemon + GPUI as an RPC client.
   * Mobile companion app for remote notifications, 1-off prompt fixes, and approvals.
   * Autonomous event triggers (GitHub issues, PR webhooks).

## Concurrent Workstream Splits (Non-Overlapping Ownership)

- [x] **Slice A: Discrete Decision Layer (TypeSafe Jev & Judge Trait)**
  - *Files*: `crates/juto-ai/src/judge.rs`, `crates/juto-ai/src/types.rs`
  - *Scope*: `Judge` trait, `TypeSafeJudge` (`/v1/systemone` / OpenRouter decisions), `Choice`/`Decision` types, mock tests.
- [x] **Slice B: Workspace Isolation & Resource Scopes (`jj` + systemd)**
  - *Files*: `crates/juto-runtime/src/workspace.rs`
  - *Scope*: `SubagentWorkspace` (`jj workspace add/forget`), `systemd-run` user scope wrapper (`agents.slice`), flock GPU lock.
- [x] **Slice C: Subagent State Machine & Blocker Protocol**
  - *Files*: `crates/juto-runtime/src/subagents.rs`
  - *Scope*: Add `Blocked`, `Verifying`, `Reviewing` to `SubagentStatus`, pause/resume escalation channel, `TaskSpec` extensions.
- [ ] **Slice D: GPUI Sidebar Mission Control & Diff Cards**
  - *Files*: `apps/juto/src/sidebar.rs`, `apps/juto/src/task_card.rs`
  - *Scope*: Sidebar view, task status badges, expandable duration/token cards, live diff viewer, steering buttons.
- [ ] **Slice E: End-to-End Orchestrator & Peer Review DAG**
  - *Files*: `crates/juto-runtime/src/orchestrator.rs`
  - *Scope*: Manager DAG pipeline, Jev routing hook, Builder $\to$ Reviewer handoff, verification quality gates.
