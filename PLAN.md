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
   * Bundled `juto-repo-setup` skill guides agents to inspect and document each
     target repository's supported development workflow.
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
  - *Scope*: `SubagentWorkspace` (native change-graph fork), `systemd-run` user scope wrapper (`agents.slice`), flock GPU lock.
- [x] **Slice C: Subagent State Machine & Blocker Protocol**
  - *Files*: `crates/juto-runtime/src/subagents.rs`
  - *Scope*: Add `Blocked`, `Verifying`, `Reviewing` to `SubagentStatus`, pause/resume escalation channel, `TaskSpec` extensions including strict/permissive structured-output schema validation.
- [x] **Slice D: GPUI Sidebar Mission Control & Diff Cards**
  - *Files*: `apps/juto/src/sidebar.rs`, `apps/juto/src/task_card.rs`
  - *Scope*: Sidebar view, task status badges, expandable duration/token cards, live diff viewer, steering buttons.
- [x] **Slice E: Role-Based Delegation (replaces the Orchestrator)**
  - *Files*: `crates/juto-runtime/src/roles.rs`, `crates/juto-runtime/src/roles/*.md`
  - *Scope*: `main` delegates through `task` and cannot edit; built-in `builder`/`scout`/`reviewer`/`task` roles from markdown with `tools` frontmatter; `.juto/agents` and data-dir overrides; `max_recursion_depth` defaults to 2.
- [x] **Slice F: Peer Messaging & Session History URIs**
  - *Files*: `crates/juto-runtime/src/peers.rs`
  - *Scope*: `agent://<id>` status/output reads and steering, `agent://all` live-child broadcast, and read-only `history://<id>` session journals.
