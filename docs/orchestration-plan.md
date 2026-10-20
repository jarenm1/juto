# Juto End-to-End Multi-Agent Orchestration Architecture & Plan

This document specifies the architecture, contracts, and concurrent workstream slices for
Juto's end-to-end agent orchestration system.
The core design centers on a **Single Human Chat** where the user directs high-level
intent, supported by a **GPUI Sidebar Mission Control** displaying real-time task
progress, an ultra-fast **Discrete Decision Layer (TypeSafe Jev)** for semantic routing
and state classification, **Isolated `jj` Workspaces** with systemd resource
governance, and a **Zed Delta-inspired model** connecting code modifications directly to
agent conversations.

---

## 1. Executive Architecture & UX Vision

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       GPUI Desktop & Remote Clients                         │
│                                                                             │
│  ┌──────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │     Main Chat (Human Hub)        │  │   Sidebar: Mission Control      │  │
│  │                                  │  │                                 │  │
│  │  User: "Let's implement the auth │  │  [#1 Auth Session]   RUNNING   │  │
│  │         token refresh flow"      │  │    ├─ Worker: builder (jj-ws-1) │  │
│  │                                  │  │    ├─ Step: running cargo test  │  │
│  │  Manager: "Triage complete.      │  │    └─ Diff: +42 -12 in auth.rs  │  │
│  │            Spawned implementation│  │                                 │  │
│  │            pipeline in workspace │  │  [#2 Security Audit] QUEUED     │  │
│  │            juto-auth-refresh."   │  │    └─ Worker: reviewer          │  │
│  └──────────────────────────────────┘  └─────────────────────────────────┘  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ RPC / IPC Events
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                    Headless Juto Agent Runtime Daemon                       │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     Manager Agent (The Director)                      │  │
│  │   - Holds overall task context, acceptance criteria, and plan.        │  │
│  │   - Communicates with the user; breaks down milestones.               │  │
│  └───────────────────┬───────────────────────────────┬───────────────────┘  │
│                      │                               │                      │
│     Route / Triage   ▼                               ▼  Classify State      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │        Discrete Decision Engine (TypeSafe Jev / System One)         │   │
│   │   - <50ms calibrated selection over finite choices                  │   │
│   │   - Routes agent persona, tool approvals, and blocker taxonomy      │   │
│   └──────────────────┬───────────────────────────────┬──────────────────┘   │
│                      │                               │                      │
│     Isolated Spawn   ▼                               ▼  Isolated Spawn      │
│   ┌───────────────────────────────┐   ┌─────────────────────────────────┐   │
│   │ Subagent: Builder (Task)      │   │ Subagent: Reviewer (Coworker)   │   │
│   │   - Isolated `jj` workspace   │   │   - Read-only audit & tests     │   │
│   │   - cgroup CPU / Mem capped   │   │   - cgroup CPU / Mem capped     │   │
│   └───────────────┬───────────────┘   └───────────────┬─────────────────┘   │
│                   │                                   │                     │
│                   └─────────────────┬─────────────────┘                     │
│                                     ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │           Repository & Workspace Governance Boundary                │   │
│   │   - `jj workspace add ~/workspaces/juto-<task>`                     │   │
│   │   - Resource wrapper: `scripts/agent-scope.sh` (systemd user scope) │   │
│   │   - Zed Delta store: Patches anchored to conversation turns         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 The UX Principles
1. **The Human Stays at the Strategy Level**: The main chat is not polluted with
   hundreds of intermediate tool calls, compiler errors, or shell loops. The user
   discusses goals, provides guidance, and receives concise milestone updates.
2. **Sidebar Mission Control**:
   * Shows active, queued, and completed subagents with explicit status badges:
     `Queued`, `Running`, `Blocked (Action Required)`, `Verifying`, `Reviewing`, `Done`.
   * Expanding a task card reveals:
     - Real-time step progress (current tool, compilation status, test pass/fail counts).
     - Live Delta preview (colored unified diff attached to the subagent).
     - Live steering controls (Pause, Cancel, Inject Guidance).
3. **Two-Way Steering**:
   * If a human types a correction into the main chat (*"Don't change the public trait
     signature"*), the Manager agent injects a live steering message directly into the
     active subagent's run loop without tearing down its execution state.
   * If a subagent encounters a true blocker (e.g., missing API key, ambiguous product
     choice), it transitions to `Blocked` and escalates a structured decision card
     directly to the user.

---

## 2. Discrete Decision & Routing Layer (TypeSafe Jev)

### 2.1 Why Jev?
Generative LLMs (e.g. Claude 3.7 Sonnet, GPT-4o) are powerful for synthesis and coding,
but suffer high latency (1–4 seconds) and token cost when deciding simple, discrete
branching points. 

**TypeSafe Jev** (via TypeSafe's System One `/v1/systemone` endpoint or OpenRouter
`openrouter/~typesafe/jev-latest`) is an ultra-fast, calibrated classification model
designed specifically for picking from discrete candidate sets in under 50ms.

### 2.2 Integration Seams in Juto
1. **Intent Triage at Prompt Ingestion**:
   When the user submits a message in the main chat, Jev evaluates whether it is:
   * `DIRECT_REPLY`: Clarification or conversational answer (no subagent needed).
   * `EXPLORE_SCOUT`: Read-only question needing fast code search (`scout`).
   * `MECHANICAL_EDIT`: Small, localized, deterministic refactor (`builder`).
   * `END_TO_END_TASK`: Multi-step implementation needing an isolated workspace and manager
     orchestration (`builder`, then `reviewer`, delegated through the `task` tool).
2. **Agent Persona & Model Tier Routing**:
   Selecting the best worker profile for a delegated task based on complexity, context
   depth, and tool capability.
3. **State Machine Transition & Blocker Classification**:
   Evaluating subagent turn outputs to classify execution status:
   * `CONTINUE`: Worker is executing normally.
   * `BLOCKED_USER_INPUT`: Ambiguity requiring human confirmation.
   * `BLOCKED_CREDENTIAL`: Missing provider token or environment permission.
   * `READY_FOR_VERIFY`: Code edits completed; ready for build/test verification.
   * `COMPLETED`: Invariants satisfied and verified.
4. **Action & Tool Safety Gating**:
   Replacing heavy regex heuristics with calibrated semantic classification on shell
   commands and destructive file operations before execution.

---

## 3. Subagent Contract & Coworker Model (OMP Heritage)

Juto adopts and refines the proven subagent lifecycle from OMP (`oh-my-pi`), adapting
it into native Rust.

### 3.1 The Contract (`TaskSpec` and `SubagentResult`)
Subagents are not opaque shell processes; they are structured in-process `AgentSession`
instances running on the Tokio runtime:

```rust
pub struct TaskSpec {
    /// Human-readable title displayed in the sidebar
    pub name: Option<String>,
    /// High-level goal and contract
    pub task: String,
    /// Role name: "builder" | "scout" | "reviewer" | "task" (default)
    pub agent: Option<String>,
    /// Model selector override (e.g., "anthropic/claude-sonnet-4-5")
    pub model: Option<String>,
    /// Strict JSON schema the subagent must conform to upon completion
    pub output_schema: Option<serde_json::Value>,
    /// Schema validation mode: "strict" (fails on mismatch) or "permissive"
    pub schema_mode: SchemaMode,
    /// Read-only sandbox (denies write/exec tools)
    pub read_only: bool,
    /// Dedicated native change-graph fork requested
    pub isolated_workspace: bool,
}

pub struct SubagentResult {
    pub id: String,
    /// Explanatory summary text
    pub output: String,
    /// Structured payload validating against output_schema
    pub data: Option<serde_json::Value>,
    /// Total tokens and cost breakdown
    pub usage: Usage,
    /// Final status: Completed | Failed | Cancelled
    pub status: SubagentStatus,
    /// Working copy changeset ID or patch reference
    pub change_id: Option<String>,
    /// Path to persistent session journal
    pub session_path: PathBuf,
}
```

### 3.2 Coworker Collaboration Topology
Subagents coordinate as specialized coworkers rather than an uncontrolled group chat:
* **The Manager**: Maintains the plan DAG, tracks dependency order, spawns tasks,
  monitors status, and integrates results.
* **The Builder**: Focuses strictly on writing code, reading documentation, and running
  local smoke checks inside its isolated workspace.
* **The Reviewer**: Spawned automatically upon Builder completion to perform an
  adversarial audit: checking regressions, test coverage, code style, and security.
* **Structured Yield & Peer Messaging**:
  * Workers complete their lifecycle with a structured `yield` tool call providing
    schema-checked artifacts.
  * Peer communication uses an internal URI/bus pattern (`agent://<id>`, `history://<id>`).
    A reviewer can directly inspect `history://<builder-id>` to see why a decision was
    made without re-running the builder's analysis.
  * `agent://<id>` reads status/output and sends steering messages; `agent://{a, b, c}`
    addresses an explicit child list and reports per-child failures.
  * `history://<id>` reads the active journal entries for a successfully completed child.
    The URI resolves through the subagent result; it is never treated as a filesystem path.
  * `recall {}` reads the main journal's active-path index, including compacted
    originals. `from`/`count` page the list; `recall {"id": N, "at": 0}` reads a
    message with byte-offset continuation hints. Pages are about 16 KB, and
    images appear as MIME placeholders. Branch switches select a new index path.
  * A `task` call captures one main-view snapshot for all siblings: a developer
    message containing the latest compaction summary and newest visible index
    lines within 12 KB. Explicit shared `context` still works. Nested children
    snapshot and recall the main session too; follow-ups do not reinsert snapshots.

### 3.3 Juto Main-Thread Compaction (Snapcompact)
Juto auto-compacts when the estimated prompt exceeds `compactionThreshold` of the
context window, preserving the newest `keepRecentMessages` messages and whole tool-call/
result groups. This Rust port follows oh-my-pi snapcompact at
`579da1d661c5cb8d43bc2ddd429ab72e67165ad8` (MIT, `licenses/OMP-MIT.txt`), without
claiming complete OMP feature parity.

* `compactionMethod: summary` is the default and always uses the LLM summarizer.
  Bitmap compaction is experimental and explicitly opt-in: `auto` uses deterministic
  images when the active model accepts images and the complete archive fits the frame
  budget, otherwise it summarizes. `snapcompact` requires vision and sufficient budget,
  returning an error without changing the journal otherwise. No real-model reading
  accuracy or token savings have been established for this port.
* `compactor` (optional `provider/model[:effort]` or `@role`, the `modelRoles` grammar)
  names the model that writes compaction summaries. Unset, compaction uses the session's
  current model. Invalid selectors fail config load.
* The public-domain X.org misc-fixed 8x13 Latin-1 glyph table is embedded in `font.rs`
  with its source and public-domain notice; no unused BDF is vendored. Anthropic uses
  11px advance/16px pitch; OpenAI/default uses 8px advance/22px pitch. Frames are
  1568px wide, or 1932px for known high-resolution Claude lines, 2048px for Google;
  height follows printed rows, with a 64px minimum.
* ANSI is stripped, whitespace collapsed, and newline runs become `█`. Common symbols
  fold to ASCII. Unlike OMP, this port omits Silver: unsupported Unicode, including
  combining marks and non-Latin identifiers, becomes visible `\u{HEX}` escapes rather
  than silently becoming `?` or disappearing. Symbol folding and whitespace
  normalization remain lossy; original messages stay in the journal.
* Tool arguments/results have character caps with explicit elision markers. A FILES
  section records paths from read/write/edit tool calls (not proof of success).
* Frame budget is `min(provider limit, 80, window/4/4096)` (80 without a known window).
  Provider limits: Anthropic/Bedrock/OpenRouter 90, OpenAI/Google 200, otherwise 5.
  Overflow never drops the oldest archive: auto falls back to summarization.
* Compaction entries durably store normalized source and base64 PNGs; old entries
  without an archive remain readable. Recompaction combines previous source with new
  discarded messages. Context rebuilds attach frames to a **user** message because
  developer/system-role images are not portable across providers.
* Summary priority: near-verbatim user orders, decisions and corrections; lasting
  effects and failures; findings/open questions; tool steps last. Names, paths, IDs,
  and errors must be exact, without overstating completion. Both paths are cancellable;
  cancelled work does not append a compaction.

Known limits: summarizing an accumulated archive can exceed the summarizer's context;
that provider error surfaces without changing the journal. Switching an already
image-compacted session to a non-vision model requires explicit summary compaction
before continuing, because providers drop unsupported images. Image token accounting
uses a conservative 4096-token estimate, not model-specific billing. Provider OCR
accuracy and token savings versus summaries are not established by local tests.

---

## 4. Workspace Isolation & Resource Governance

### 4.1 Workspace Isolation
Concurrent agents editing the same files cause corrupted working copies, broken compiler
caches, and merge races.

Juto establishes strict workspace isolation:
1. **Creation**: When a task requiring code edits is spawned, the runtime forks a dedicated
   directory from an immutable change-graph point:
   ```rust
   SubagentWorkspace::provision_in(repo_root, path, base_change, subagent_id)
   ```
   The fork shares the parent's Git object store through a marker file; no Git branch,
   worktree, or index is created, and no `jj` workspace is involved. The original plan called
   for `jj workspace add`; the implementation went with native forks so the runtime does not
   depend on `jj` being installed.
2. **Execution**: The subagent executes all filesystem mutations, formatting, builds,
   and tests strictly inside that directory.
3. **Integration**: When an isolated run ends, the runtime keeps the fork's net changes as
   the checkpoint `subagent/<id>` and only then removes the fork directory. Changes are
   captured from the fork's files, so edits made by shell commands are kept too:
   `ChangeGraph::capture_thread_changes` records them and `mark` names the node. A completed
   turn sets `SubagentResult.change_id` to that node when the fork holds net changes. A failed
   or cancelled run keeps its checkpoint only when it has changes, and its error names the
   checkpoint. gc keeps every checkpoint, so kept changes survive until they are joined or
   dropped. The main thread (depth 0 only) folds a checkpoint into its workspace with
   `subagent_join`, which calls `ChangeGraph::join`: a clean merge writes a two-parent node, and
   a conflict changes nothing and reports the paths. Uncaptured main-workspace edits block the
   join. `subagent_drop` removes a checkpoint, after which gc may collect its changes.

### 4.2 Resource Governance (`agent-scope.sh` Model)
To prevent parallel agents from exhausting host memory, CPU, or lock contention:
1. **systemd units under `agents.slice`**:
   * Developer heavy commands (Cargo builds, tests, language servers) run through
     `scripts/agent-scope.sh` in a systemd scope under a shared slice (`agents.slice`).
   * Runtime bash in a `workspace` or `untrusted` tier (or `attended` with `bashScoped`)
     runs as a systemd unit under the same slice. Sandboxed tiers also confine its filesystem
     and environment ([docs/sandbox.md](sandbox.md)). A governed command fails when
     `systemd-run` is unavailable rather than running without its caps.
   * Limits derive from measured capacity, not fixed percentages: the agent memory
     budget is total memory minus the session reserve; each scope's `MemoryMax`
     comes from recorded history (see `docs/resources.md`). CPU and IO are shared
     by `CPUWeight`/`IOWeight`, and `CPUQuota` is set only when requested.
   * Cargo build jobs derive from the granted memory (`CARGO_BUILD_JOBS`).
2. **GPU Serialization**:
   * Hardware GPU access is serialized via a shared file lock (`flock`).
   * Automated visual tests use headless software Vulkan (Mesa / Xvfb) to avoid blocking
     the host GPU.
3. **Measured usage and capacity** (implemented): every scoped command run through the runtime
   records peak memory, CPU, OOM kills, and IO per command class in
   `<data-dir>/resources/usage.jsonl`, read from inside the unit's cgroup. Grants and capacity
  are derived from these records and the measured host. Admission is a shared ledger
  (`resources/broker`): a command waits until its reserved memory fits the budget, and
  bash commands that wait return a queued job id whose result is steered in later.
  Agent desks, the provider pool, and the review limit are implemented; see
  `docs/resources.md` (Agent admission).

### 4.3 Repository Setup Skill
The harness installs the `juto-repo-setup` skill in its global skill directory.
When onboarding a target repository, the skill directs the agent to inspect repository
instructions, identify its supported VCS, development environment, and build/test
commands, and document the Juto workspace, resource-scope, and verification workflow.
The skill reuses repository conventions and requests approval before changing project
files or installing dependencies; it does not synthesize an execution profile or
transparently wrap commands.

---

## 5. End-to-End Semantics & The Zed Delta Model

### 5.1 Defining "End-to-End"
A task is not considered done when the model stops generating code. E2E completion
requires passing a verified quality gate:
```
[ Code Modified ] ──► [ Formatting & Lint ] ──► [ Automated Build & Tests ]
                                                         │
[ Merged / Published ] ◄── [ Peer Review Passed ] ◄── [ Observed Smoke Proof ]
```
* **Observable Proof**: A compilation pass is not proof of streaming, UI rendering,
  or persistence. The subagent must run the specific smoke test or launch verification
  and capture observable proof.
* **Peer Review**: A separate reviewer subagent must sign off on the change.
* **Publication**: Depending on configuration:
  - Local commit/rebase into the primary jj branch.
  - Creation of a published Git branch and Pull Request with structured summary.

### 5.2 Zed Delta Model: Anchoring Reasoning to Code
Traditional PRs lose the reasoning behind changes. Inspired by Zed Delta:
1. **Conversation-Anchored Diffs** *(shipped, partial)*:
   Every edit records `EditProvenance` — the session ID, the tool, the selection
   mode, and the path — so a diff chunk points back to the session and tool call
   that produced it. The conversational turn ID and the motivating prompt are
   **not** recorded.
2. **Live Multiplayer / Collaborative View** *(proposed)*:
   In the sidebar or diff viewer, diffs are live-updated as the subagent edits files.
   Reviewers (human or AI) can attach comments directly to hunks while the agent is
   still working. The sidebar renders live diffs today; hunk comments are not built.
3. **Replay & Auditability** *(shipped, partial)*:
   Every node carries its provenance, so a change traces back to the session and
   tool call that created it. Replaying the reasoning that produced it is not built.
4. **Native Live DAG to Git Transfer** *(shipped)*:
   Instead of generating premature Git commits or depending on external VCS wrappers
   during live tool loops, Juto tracks atomic edit trees and thread branches natively.
   Named checkpoints (such as points verified working) freeze immutable snapshots,
   and `juto-cli changes export/push` projects only named points onto Git feature branches
   using compare-and-swap ref updates and raw byte tree capture.

### 5.3 Cross-Machine Replication *(shipped)*

Agents may run locally or on different machines at the same time. The DAG
replicates over ordinary Git transport — no custom server — as a grow-only set of
content-addressed refs:

```
refs/juto/nodes/<node-id>          -> blob(ChangeNode JSON)          create-only
refs/juto/snapshots/<id>           -> tree | commit                  create-only
refs/juto/heads/<owner>/<thread>   -> blob(head node id)             single writer per owner
refs/juto/checkpoints/<name>       -> blob(node id)                  single writer per name
```

Node ids are UUIDv7 and node content is never rewritten, so union-by-id is
idempotent and every ref update is a create or an identical-value no-op. A
rejected push therefore means a real conflict, never a lost update.

* **Publish** (`push_graph`): pin every node's metadata and tree, then push the
  four namespaces. Only populated namespaces are pushed, because a wildcard
  refspec that matches nothing is a hard `git push` error.
* **Import** (`fetch_graph` → `import_refs`): fetch into a staging namespace,
  union by node id with a hard content-equality assertion (a node present on both
  sides with different content aborts the whole import and changes nothing),
  verify every node's tree ref resolves, topologically sort with Kahn's algorithm,
  and save the journal once under the graph lock. Staging refs are dropped on
  every exit path. Re-running imports nothing and saves nothing.
* **Thread identity**: head refs are owner-qualified at the wire level only, so
  local thread names are unchanged. Two machines advancing the same thread
  produce two heads rather than a conflict; `reconcile` folds the remote head into
  the local thread — joining cleanly, adopting it when the thread has no local
  head, or reporting the conflicting paths and changing nothing.
* **Join** (`join`): merge base from the DAG (falling back to the graph base tree
  when two threads share no node ancestor), then `git merge-tree --write-tree
  --merge-base`, producing a two-parent node. A conflicted merge creates no node.
* **Resolve** (`resolve`): record a conflict resolved by hand. The node's tree is
  the merge result with the *conflicted* paths taken from the workspace, so a
  sparse workspace or a stale copy cannot revert the other side's changes; the
  resolved tree is then materialized and the folded-in remote head retired, so
  `needs_reconcile` clears rather than staying true after the agent acted on it.
* **Dirty** (`dirty`) and **Capture** (`capture`): the edit tools capture
  automatically, but a `bash` command does not, so `dirty` lists the paths where
  the workspace differs from its head and `capture` records them as a new point.
* **Rewind** (`rewind`): materialize a tree delta into the workspace and append a
  *forward* node whose tree equals the target, so history stays append-only.
* **Reclaim** (`gc`): mark-sweep from heads, workspace heads, and checkpoints;
  prune unreachable nodes and release both their snapshot and node refs.

`workspace_heads` is keyed by absolute workspace path and is machine-local: it is
never replicated.

### 5.4 Focus Threads *(shipped)*

A focus thread is a subagent that owns exactly one orchestration-started task.
`TaskSpec.focus` marks it; only the main thread (`depth == 0`) may request it —
the `task` tool rejects `focus: true` from child threads with "focus threads may
only be spawned by the main thread". Spawning a focus spec forces
`isolated_workspace`, since a per-thread VCS view needs its own workspace, and
`SubagentManager::list` reports `focus` plus the child's `workspace` path so the
UI can surface the thread's working directory.

A follow-up the user types makes any subagent a focus thread: the runtime sets its
`focus` flag, so the sidebar lists it. Main-started follow-ups do not promote.

Follow-up turns: `SubagentManager::follow_up(id, prompt)` resumes a `Completed`
child by reopening its recorded session and reusing its isolated workspace without
re-provisioning. Queued, running, blocked, failed, and cancelled children reject
follow-ups. A successful turn keeps its workspace so later turns share one fork; a
failed or cancelled turn is terminal and its fork is removed after its changes are kept as
the checkpoint `subagent/<id>` (section 4.1). Retained workspaces stay on disk until the main
agent releases them with `subagent_release` (PR6, below).

PR3 adds a main mailbox. Children send messages to the main agent with
`agent://main`, which lands in one in-memory inbox shared by the whole run tree.
When a child reaches a terminal state (`Completed`, `Failed`, or `Cancelled`), a
completion notice is submitted to the same inbox before terminal status is
published. Oversized notices digest asynchronously, so `wait` and terminal status
never wait for the digest. A `MainInboxReady` broadcast after publication wakes an
idle main UI without stealing the runtime forwarder's notification. The main thread
drains the inbox with `agent_inbox`, returning `from`, `name`, `kind`, `digest_of`,
and `body`, and clearing the inbox. Reading `agent://main` fails with
`PeerError::WriteOnly`, so model acknowledgment happens only through `agent_inbox`.
Children cannot author arbitrary user UI; their messages and notices are displayed
as host-rendered sidebar inbox entries. The
inbox is persisted under `data_dir/inbox/<main-session-id>.json` once `Runtime::open_main_inbox`
binds it to the main session (the desktop app and CLI do this after opening the session). Each
undrained message keeps its `origin`, the top-level session it is addressed to, so each thread
drains only its own. Reports still waiting for a digest are not persisted; a restart before the
digest finishes drops them. A file that cannot be read fails the bind instead of being replaced.
Main-started turns (the first run and an `agent_send` to a completed child) push a notice on every terminal publication. A
turn the user types in any subagent pushes none; its result reaches the user through
the focus view.

The desktop sidebar's Inbox section lists the notices the user has not dismissed. Reading it never drains the model's queue. Its Dismiss button removes only the user's copy, so the model still receives that message through `agent_inbox`. The user's copies and the model's queue persist in the same file, and dismissals survive a restart.

Run ownership: the registry, concurrency limit, and inbox are shared by the process, but
`Runtime::subagent_manager(&control)` returns a handle that owns only the children it spawns, scoped
to that run's control. Cancelling the run cancels its children; ending the run cancels the children
it still has open (`ChildCancellation`). Other runs' children are left alone.

Juto bounds all child-to-main report bodies, including peer messages and failures,
to `mainReportBytes` (default 1500 UTF-8 bytes; minimum 512). Small text-only bodies
pass through byte-exactly. Larger bodies use the `@memory` model role, prioritizing
status and verified results, exact paths/ids/errors, then decisions and blockers.
An overlong digest is requested once more with a smaller target; failure, cancellation,
timeout (60 seconds per attempt), or repeated overflow produces a clearly marked
UTF-8-safe clip of the original. Reports above 128 KiB skip the model and clip.
Every transformed body ends with exact `wait` arguments for an in-memory report id;
`wait` returns the original body and structured data under `report`. Small structured
JSON is included inline; larger data is deferred to that pointer. The header retains
the child id, optional name, kind, and original byte count. Per-sender lanes preserve
message order even while a digest is pending, and keep its originating topic when the
active thread changes. Each digest uses its turn's owner token, not an earlier run's.
Weak event registrations are pruned as managers come and go; an inbox-ready event wakes
the idle GUI after delayed publication without retaining dropped managers.
Archives persist with the inbox, so a digest pointer returned before a restart still resolves
after it. Explicit `wait` on a
child still returns its complete terminal `SubagentResult`, and nested synchronous
`task` results are unchanged.

PR5 adds focus chat mode. Clicking a focus card opens the thread's view: its
transcript, a follow-up composer, the VCS panel rooted at its workspace, and a
`← Main` button that returns to the main transcript and sidebar. The transcript
is read from the child's session file through `SubagentManager::transcript(id)`,
which accepts any completed subagent. The read runs on the background executor, and
the view reloads it each time the thread reaches a terminal state. Limitations:
live streaming is not implemented, so a running thread shows only its status
(a blocked thread shows its prompt); the transcript refreshes only on terminal
state, and no transcript is available while a follow-up turn is in progress. A
`Failed` or `Cancelled` subagent has no transcript and shows the read error.
The composer is enabled only for `Completed` threads, and rejected follow-ups
show an error and keep the draft. The main sidebar no longer has a Changes tab;
VCS is visible only in focus mode.

PR6 adds workspace release, which belongs to the main thread. `subagent_release` takes
`{ "id" }` and is registered only for the main thread, so children never receive it.
It calls `SubagentManager::release(id)`, which accepts only a terminal child: a running
child is rejected with `NotTerminal` and its workspace is left alone. Otherwise its changes are
kept as the checkpoint `subagent/<id>` (section 4.1), and the fork is removed
through `SubagentWorkspace::cleanup`, which deletes only a directory carrying the
`.juto-workspace.json` ownership marker. The removal runs on the blocking pool, so the
async runtime and the GPUI foreground do not wait on disk I/O. On success the summary
reports `workspace: None` and `released: true`, and a second release is a no-op. A failed
cleanup keeps the workspace recorded and returns `WorkspaceRelease`, so the release can
be retried. After release, `follow_up` and `transcript` both return `WorkspaceReleased`.
The focus view treats a missing workspace as no VCS panel and a released thread as
follow-up disabled. A focus view already open at release time is not updated; its next
send is refused and shows the error. The sidebar Inbox renders child messages and
completion notices independently of model delivery. Reading is non-destructive;
Dismiss removes only the user's copy, while `agent_inbox` drains only the model's
copy. Both queues persist in the session inbox file, so model wake-ups cannot hide
notices before the user sees them, and dismissals survive restart. Blocked workers
also appear as task cards with their blocker prompt and resume controls.

Limitation: a failed cleanup restores the recorded workspace, so follow-ups are accepted
again until a release succeeds, even if the directory was partly removed.

---

## 6. Future Extensibility: Headless Daemon, Mobile & Automations

### 6.1 Headless Runtime vs. Presentation Layer
* **`juto-runtime` as a Background Daemon**:
  Manages credentials, sessions, subagent pools, jj workspaces, and tool execution.
  Exposes an asynchronous RPC API (Unix socket or TLS).
* **GPUI Desktop as Client**:
  GPUI connects as an RPC client. Closing the UI window does not kill background builds
  or running subagents.

### 6.2 Mobile Companion
* Lightweight companion app (or web interface) connects to the desktop daemon over
  secure local network or Tailscale.
* **Capabilities**:
  - Live notification when a subagent is `Blocked` or completes.
  - Instant mobile approval for sensitive actions or auth prompts.
  - Sending quick 1-off bug fixes or prompts on the go.
  - Inspecting sidebar task cards and live diffs.

### 6.3 Autonomous Triggers & Event Handlers
* **GitHub Issue / PR Webhooks**:
  When an issue is opened or assigned to Juto, the runtime triggers an orchestration flow:
  1. Jev classifies the issue (bug report, feature request, question).
  2. Spawns an isolated `scout` subagent to locate the relevant code.
  3. Spawns a `builder` subagent in a dedicated `jj` workspace to reproduce and fix.
  4. Spawns a `reviewer` subagent to verify tests pass.
  5. Opens a draft PR or reports findings back to the issue tracker.

---

## 7. Concurrent Workstream Slices for Parallel Agent Execution

To build this rapidly, the architecture is decoupled into six independent slices with
**zero overlapping file ownership**. Each slice has an explicit contract, can be
implemented by a dedicated subagent concurrently, and validates against its own tests.

### Slice A: Discrete Decision Layer (TypeSafe Jev & Judge Trait)
* **File Ownership**: `crates/juto-ai/src/judge.rs`, `crates/juto-ai/src/types.rs`
* **Interface Contract**:
  ```rust
  #[async_trait]
  pub trait Judge: Send + Sync {
      /// Classifies an input prompt or turn context into one of N discrete options.
      async fn choose(&self, context: &str, options: &[Choice]) -> Result<Decision, JudgeError>;
  }
  ```
* **Implementation Details**:
  - Implements `TypeSafeJudge` targeting `POST https://api.typesafe.ai/v1/systemone`
    and OpenRouter `/decisions`.
  - Calibrated selection returning choice key, confidence score, and token usage.
  - Fallback `TextJudge` for standard completion models when no System One key is present.
* **Acceptance**: Unit tests with mock responses verifying discrete option parsing,
  header authentication, and error handling without depending on runtime or GPUI.

### Slice B: Workspace Isolation & Resource Scopes (`jj` + systemd)
* **File Ownership**: `crates/juto-runtime/src/workspace.rs`
* **Interface Contract**:
  ```rust
  pub struct SubagentWorkspace {
      pub path: PathBuf,
      pub change_id: String,
  }
  impl SubagentWorkspace {
      pub async fn provision(base_change: &str, subagent_id: &str) -> Result<Self, WorkspaceError>;
      pub async fn run_scoped(&self, cmd: &mut Command, limits: ResourceLimits) -> Result<ExitStatus, WorkspaceError>;
      pub async fn rebase_into(&self, target_dest: &str) -> Result<String, WorkspaceError>;
      pub async fn cleanup(self) -> Result<(), WorkspaceError>;
  }
  ```
* **Implementation Details**:
  - Manages `jj workspace add/forget` per subagent task.
  - Wraps shell process invocation in `scripts/agent-scope.sh` style systemd user scopes
    (`agents.slice`) with configurable CPU quota and memory limits.
  - Enforces flock-based GPU access when required.
* **Acceptance**: Integration test provisioning a temporary `jj` workspace, executing a
  command inside a memory-capped scope, and cleaning up cleanly.

### Slice C: Subagent State Machine & Blocker Protocol
* **File Ownership**: `crates/juto-runtime/src/subagents.rs`
* **Interface Contract**:
  ```rust
  pub enum SubagentStatus {
      Queued,
      Running,
      Blocked { reason: BlockerReason, prompt: String },
      Verifying,
      Reviewing,
      Completed,
      Failed,
      Cancelled,
  }
  ```
* **Implementation Details**:
  - Adds `Blocked`, `Verifying`, and `Reviewing` lifecycle states to `SubagentManager`.
  - Implements unblock/resume signaling channel: runtime suspends child loop without
    killing process until parent injects resolution.
  - Extends `TaskSpec` with `isolated_workspace` and `solution_space`.
* **Acceptance**: Unit tests verifying state transitions, watcher updates, blocker
  escalation and resumption, and timeout handling.

### Slice D: GPUI Sidebar Mission Control & Diff Cards
* **File Ownership**: `apps/juto/src/sidebar.rs`, `apps/juto/src/task_card.rs`
* **Interface Contract**:
  ```rust
  pub struct Sidebar {
      tasks: Vec<TaskCardModel>,
      selected_task: Option<String>,
  }
  impl Render for Sidebar { ... }
  ```
* **Implementation Details**:
  - Visual task list displaying badge per `SubagentStatus` (color-coded badges).
  - Expandable task cards: execution duration, token count, active tool log.
  - Unified diff preview pane rendering hunks attached to the subagent session.
  - Interactive action buttons: Pause, Abort, and Resume/Unblock prompt input.
* **Acceptance**: Visual smoke test under Xvfb/Mesa displaying updating task cards
  and responsive state transitions without panics.

### Slice E: Role-Based Delegation (replaces the Orchestrator)
* **File Ownership**: `crates/juto-runtime/src/roles.rs`, `crates/juto-runtime/src/roles/*.md`,
  and the `run_depth` and `TaskTool` code in `crates/juto-runtime/src/lib.rs`
* **Implementation Details**:
  - `main` is top-level only. It has `read`, `glob`, `grep`, `task`, `wait`, `agent_send`,
    `agent_cancel`, `agent_inbox`, `subagent_release`, `subagent_join`, `subagent_drop`, and `view`. `parse_task` rejects `main` as a child agent.
  - Built-in roles: `builder` (all tools), `scout` and `reviewer` (read-only), and `task` (all
    tools, the default). Frontmatter `tools` takes `all`, `read-only`, or a list; missing means
    read-only.
  - Overrides: `.juto/agents/<role>.md` in the working directory or an ancestor, then
    `<data_dir>/agents/<role>.md`. The nearest file wins.
  - Delegation tools are offered only below `max_recursion_depth` (default 2).
  - No intent triage, quality-gate pipeline, or remediation loop. Verification is left to the
    caller; `reviewer` is read-only and cannot run tests.
  - Main-thread `task` is non-blocking: `TaskTool` returns `queued` ids and
    `forward_main_inbox` steers each child's terminal `MainMessage` into the active run.
    Child-level `task` still uses `collect_agents` and blocks.
* **Acceptance**: `roles.rs` tests cover built-in grants, nearest-file override, and read-only
  default. `lib.rs` tests observe the advertised tool set for the main thread, read-only
  approval, a scout child, and the recursion limit, and check that `task` rejects unknown and
  `main` roles before spawning anything.

### Slice F: Peer Messaging & Session History URIs
* **File Ownership**: `crates/juto-runtime/src/peers.rs`, `crates/juto-runtime/src/lib.rs`
* **Interface Contract**:
  ```rust
  pub struct PeerBus { /* shared SubagentManager */ }
  impl PeerBus {
      pub fn send(&self, uri: &str, message: impl Into<String>)
          -> Result<PeerSendReport, PeerError>;
      pub fn read(&self, uri: &str) -> Result<PeerResource, PeerError>;
  }
  ```
* **Implementation Details**:
  - `agent://<id>` exposes a child status/result snapshot and accepts steering messages.
  - `agent://{a, b, c}` sends to each listed child, including completed ones (which start a new turn). Duplicates collapse, `main` is rejected, and a per-child failure does not stop delivery to the rest. `agent_send` takes the same forms in its `to` field.
  - `agent://main` is write-only: child messages land in the main inbox with the
    child id as `from` (`unknown` when sent from the main thread). Reads fail with
    `PeerError::WriteOnly`. A child list never reaches main.
  - `history://<id>` reads a completed child's active session journal via the manager's
    result path; unsupported, malformed, unknown, and unavailable resources fail explicitly.
* **Acceptance**: Tests cover completed output and transcript reads, child-list sends with per-child failures, malformed and read-only URIs (including `{}`, `{main}`, and a list with a suffix), and history availability before completion.
