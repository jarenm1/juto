# Juto

A Rust agent harness with a native GPUI interface, targeting the runtime
capabilities of [oh-my-pi](https://github.com/can1357/oh-my-pi).

## Current state

The build includes a pinned Nix development environment, a Cargo workspace,
the modular Rust agent runtime crates, and a native GPUI chat view. You can type
in a modern composer card with an attachment dropdown (**+**), submit with Enter
or a circular Send button (**↑**), and view plain, left-aligned messages in a scrollable list.
Messages and turns are durably journaled to append-only JSONL sessions in the background.

**The UI is wired into the agent runtime.** User prompts and attachments are submitted
to the streaming agent runtime on a background Tokio thread, streaming assistant deltas
into the native UI without blocking GPUI's foreground thread. Existing sessions can be loaded
with `--session <path>`, and models can be selected with `--model <provider/model>`.
See [the full OMP runtime inventory and port plan](docs/omp-runtime.md) for the
capabilities to preserve, primary-source references, and acceptance criteria.
Agent contributors should read [AGENTS.md](AGENTS.md).

**Workspace isolation and resource governance.** `juto-runtime` provides `SubagentWorkspace`
for provisioning a dedicated fork directory per subagent task: a native change-graph fork that
shares the parent's Git object store, not a `jj` workspace and not a Git worktree. Commands run
under systemd user scopes (`agents.slice`) with configurable CPU, memory, and swap ceilings, and
GPU access is serialized with `flock`. `ChangeGraph::join` and `reconcile` merge trees into a
two-parent node rather than rebasing. They are explicit operations (the `vcs` tool and
`juto-cli changes`); subagent results are not folded back automatically.

`jj` is the *human's* interface to this repository (see [AGENTS.md](AGENTS.md)); the runtime does
not invoke it. The only place `jj` appears in the runtime is the `.jj` metadata-path exclusion, so
a capture never records Jujutsu's own directory.

**Sandboxed commands (opt-in).** `ResourceLimits::sandbox` adds a real filesystem boundary:
`ProtectSystem=strict`, `ProtectHome=read-only`, `PrivateTmp=yes`, and `ReadWritePaths` covering the
workspace plus the toolchain directories a build must write (`CARGO_HOME`, `RUSTUP_HOME`,
`XDG_CACHE_HOME`). Enable it with `AGENT_SANDBOX=1` or `ResourceLimits::with_sandbox`.

Two details matter in practice. Transient *scopes* reject namespace properties, so a sandboxed
command runs as a transient *service* instead — which does not inherit the caller's environment, so
the caller's variables are forwarded explicitly with `--setenv`. And a service outlives its
`systemd-run` client, so the unit is named and stopped when the child is dropped; otherwise a
cancelled command would keep running.

It is **not the default**: `ResourceLimits::default()` leaves `sandbox` unset. Enabling it turns a
missing `systemd-run` into a hard failure rather than an uncapped run, and it changes how a command
is supervised — a service rather than a scope, which is why the environment and the unit lifecycle
are handled explicitly above. The mechanism is tested against a real `cargo check` inside the
sandbox.

**Delegating role model.** The top-level `main` role has `read`, `glob`, `grep`, `task`, `wait`,
`agent_send`, `agent_cancel`, `agent_inbox`, `subagent_release`, and `view`; it cannot edit files. Work runs in child agents defined as
markdown roles in `crates/juto-runtime/src/roles/`: `builder` (all tools), `scout` and `reviewer`
(read-only), and `task` (all tools, the default). A role's `tools` frontmatter takes `all`,
`read-only`, or a list of tool names; a missing field means read-only. A `<role>.md` file in
`.juto/agents/` in the working directory or an ancestor, or in the runtime data directory's
`agents/`, replaces the built-in of the same name, and the nearest file wins. Children receive
delegation tools only below `max_recursion_depth` (default 2).

Delegation runs through the Exec-tier `task` tool, so it requires `juto-cli --yes`
(`ApprovalMode::Allow`) or an installed `ApprovalHandler`; in Ask mode with neither, `task` fails.
Builders write to the shared working tree unless a call sets `isolated_workspace: true`, and
concurrent builders on one tree can conflict. An isolated child's fork is deleted when its run
finishes and its file edits are not folded back, so they never reach the parent; its returned
result and session transcript are kept. Nothing folds subagent work back automatically.

On the top-level thread, `task` does not block. It returns `queued` ids immediately, and each
child's terminal result (completed, failed, or cancelled) reaches the running main turn as a
steered user message headed `[subagent <id> <kind>]`. A completion that lands while no main run
is active is delivered at the start of the next run. Waking an idle session when a child finishes
is the host's job; the runtime does not start a run on its own. Lifecycle events (`spawned`,
`status_changed`, `finished`) are available from `SubagentManager::subscribe_events`. Child agents
that call `task` still wait for their own children.

**Native live-change DAG and Git transfer.** Juto tracks live agent edits in its own
version control DAG rather than synthesizing premature Git commits or depending on
jj at runtime. Edits capture immutable trees with session/thread provenance,
independent threads fork isolated workspaces without Git worktrees, and named
checkpoints (e.g. verified-working states) project into deterministic Git commits
that can be safely published to a feature branch.

**Cross-machine replication.** The DAG replicates over ordinary Git transport, so
agents can work locally or on different machines at the same time. It is a
grow-only set of content-addressed refs — `refs/juto/nodes/<id>` (node metadata),
`refs/juto/snapshots/<id>` (trees), `refs/juto/heads/<owner>/<thread>`, and
`refs/juto/checkpoints/<name>` — so every update is a create or an identical-value
no-op and a rejected push is a real conflict, never a lost update. `push_graph`
publishes, `fetch_graph` imports into a staging namespace and unions by node id
with a hard content-equality assertion, and `reconcile` folds another machine's
head into the local thread (joining cleanly, adopting when the thread has no local
head, or reporting conflicting paths). `join` merges two threads into a two-parent
node via `git merge-tree`; `resolve` records a conflict resolved by hand, taking
the conflicted paths from the workspace so a sparse or stale copy cannot revert
the other side's changes; `rewind` reverts a thread by materializing a tree delta
and appending a forward node, so history stays append-only. `dirty` lists the
paths a `bash` command left uncaptured and `capture` records them, since only the
edit tools capture automatically. `gc` mark-sweeps
unreachable nodes, releases their refs, and packs the survivors. All of it is
exposed both as `juto-cli changes` subcommands and as the runtime's `vcs` tool,
so an agent can run the whole workflow — including handing a verified checkpoint
to a reviewer as a Git branch — without shelling out.

Fork provisioning materializes a tree with a single `git cat-file --batch` child,
recovers forks interrupted mid-provision, and skips gitlink (submodule) entries —
a baseline gitlink is preserved by capture, so a fork that omits the submodule
still captures a tree identical to the base.

**Sparse forks.** `fork_workspace_sparse` (CLI: `changes fork --sparse <path>`)
materializes only the requested subtrees, so an agent scoped to one crate in a
monorepo does not pay for the whole tree. The workspace marker records the spec,
and every path that produces a node honours it: capture and `restore_tree`
preserve baseline paths outside the spec instead of reading their absence as a
deletion, `join` and `rewind` record the derived tree directly rather than
snapshotting the partial workspace, and the edit tool refuses an out-of-spec
path before touching the file. A sparse fork therefore still records a complete
tree.

The `edit` tool selects a unique tree-sitter
AST selection for Rust, Python, JavaScript, TypeScript, and TSX; unsupported
languages, invalid queries, parse errors, and ambiguous captures fail explicitly
rather than silently switching modes. The default capture is `@edit`.

```json
{"path":"src/lib.rs","selection":{"mode":"ast","query":"(function_item name: (identifier) @name (#eq? @name \"run\")) @edit"},"replacement":"fn run() { work(); }"}
{"path":"notes.txt","selection":{"mode":"range","start_line":2,"end_line":4},"replacement":"replacement chunk\n"}
{"path":"config.toml","selection":{"mode":"text","old_string":"enabled = false"},"replacement":"enabled = true"}
```

Ranges are 1-based and inclusive, replacing whole lines **including their
terminators**. Replacement text is verbatim; include a newline when needed.
Exact-text selection requires a unique match unless `replace_all: true` is set
inside `selection`. This replaces the previous top-level
`old_string` / `new_string` edit contract; there is no legacy alias.

Both `edit` and whole-file `write` now require a path inside the session's
workspace. A shared per-workspace advisory lock covers before/after snapshots
and atomic writes, preserving existing file permissions. Receipts contain
immutable `before_tree` / `after_tree` tree ids, the `graph_node` the edit
recorded, an operation-only Git-format diff (capped at 64 KiB with
`diff_truncated`), no-op/application status, and session/tool/selection
provenance. An edit appends a node to the change graph; it never creates a Git
commit. Existing unrelated working edits are snapshotted before the operation,
not included in its diff.

Workspace escapes, metadata paths, symlinks, and existing ignored/untracked files
are rejected. A newly created ignored file can be applied
before Git reports it untracked; this returns an explicit applied-but-untracked
error, not success. Post-write bookkeeping errors report that the file was
applied; no rollback is claimed. Cancellation before writing leaves the file
unchanged; once applied, snapshot/diff bookkeeping finishes. The lock coordinates
Juto callers, not arbitrary external writers; `bash` is not intercepted.

Observed verification: runtime tests and a real Agent command/event smoke using
a deterministic local model driver, actual file tools, and an actual scratch Git
repository. The smoke exercised write, AST/range/text edits, ambiguous AST
rejection, paired tool results, revision/provenance receipts, and no-op reporting.
No live-provider or native-window behavior was exercised by that smoke.

**Repository setup skill.** The runtime installs the bundled `juto-repo-setup`
skill in its data directory on first startup. It guides agents through identifying
a target repository's supported development workflow and documenting Juto-specific
resource, workspace, and verification requirements without overwriting user edits.

Subagent `output_schema` validation is strict by default. Task requests may set
`schema_mode: "permissive"` to retain a completed result even when its structured
data is absent or does not match the declared schema; malformed schema definitions
remain errors in either mode.

### Agent desktop compositor

The [agent-owned headless compositor design](docs/agent-compositor.md) specifies
isolated application environments, configurable output/FPS/resource limits,
computer-use control, and live human interaction with running games. Same-host
viewing uses shared memory or negotiated DMA-BUF; remote laptop/phone viewing
uses WebRTC.

**Shipped: the headless compositor, environment supervisor, control daemon,
runtime tool, and embedded viewer.**
`juto-desktop` runs a Smithay Wayland compositor with one virtual output and one
virtual seat per environment — no host display, no physical input. An environment
owns a private `XDG_RUNTIME_DIR` and Wayland socket, launches applications by
argv into a per-environment systemd scope (`CPUQuota`, `MemoryMax`,
`MemorySwapMax`, `TasksMax`), composes toplevels into a sealed shared-memory
frame pool with a GLES renderer over surfaceless EGL (the `hardware` profile,
which holds the shared advisory GPU lock for the environment lifetime) or with
Smithay's software pixman renderer (the `software` profile, which never touches
the host GPU), injects authorized pointer and
keyboard input into the virtual seat, and reports mapped windows, geometry
generations, grants, and control leases. `Destroy` terminates the application
tree, stops the compositor, and removes the private runtime directory; it is
idempotent. Applications inherit only an approved environment allowlist plus
environment-private paths — never the host display, session bus, SSH agent, or
provider secrets.

The runtime owns a process-wide `DesktopSupervisor` and exposes a `desktop` tool
with `create`, `launch`, `observe`, `input`, `describe`, and `destroy` actions.
`create` returns the environment id, host id, and an owner grant id, which feed
the `desktop_artifact` chat descriptor. `observe` writes a lossless PNG snapshot
of the composed desktop and returns frame metadata plus the mapped window list,
so an agent can attach the running application as evidence.

```sh
scripts/agent-scope.sh -- nix develop . -c cargo run -p juto-desktop -- --socket /tmp/juto-desktop.sock
scripts/agent-scope.sh -- nix develop . -c cargo test -p juto-desktop
```

The daemon serves a length-prefixed JSON control protocol over a private Unix
socket (`Create`, `Launch`, `Describe`, `Observe`, `ConfigureOutput`, `Grant`,
`Revoke`, `Control`, `Input`, `AutomationQuiesced`, `Destroy`, `Subscribe`).

**Embedded viewer.** A `desktop_artifact` card mounts the environment's canvas inline in
the chat message. The canvas renders composed frames from the environment's sealed
shared-memory pool, with no video codec on this path. It starts **paused**: it shows one
still captured at mount and forwards no input. Clicking the canvas resumes live
rendering; **Pause** returns to the still and releases any control lease. While live,
pointer, scroll, and keyboard input goes to the virtual seat, and **Take control** /
**Release control** manage the human control lease. The viewer letterboxes the frame,
maps pointer coordinates to desktop pixels, and reports connection state and errors
instead of showing a stale frame as live. `juto --desktop <command> [args...]` mounts
the same canvas directly for development.

**Resize gating.** A resize publishes a new geometry generation and clears every
viewer's acknowledgement; pointer input is rejected with `StaleGeneration` until a
viewer reports `GeometryReady` for the new generation, so a click cannot land on a
stale frame even when the encoded size is unchanged.

**Artifact resolution.** Opening a `desktop_artifact` card resolves the
descriptor through the runtime: the host must be this host, the environment must
exist here, and the grant must be issued to the requesting principal and
unexpired. The resolver returns a same-host viewer route or a one-time remote
attach token, and otherwise reports `Unsupported`, `Unauthorized`, `Expired`,
`Stopped`, or `Unavailable` explicitly. A resolved route is not proof of a live
stream — only received frames establish that.

**Owner lifetime and retention.** An environment records an owner heartbeat;
`Heartbeat` renews either the owner deadline or the control lease (matching the
current epoch), and input traffic never renews either implicitly. A reaper retires
an environment whose heartbeat has lapsed, so an abandoned desktop does not
outlive its task. `Retain` transfers lifetime responsibility to an authenticated
human session with a host-bounded deadline (default maximum 30 minutes).
The runtime's `desktop` tool heartbeats on every action. The embedded canvas renews the
owner heartbeat every 2 seconds for as long as it is mounted, paused or live, and renews
the human control lease only while the human holds control.

**Compositor worker process.** The compositor runs in its own process inside the
environment slice (`juto-desktop-<env>-compositor.scope`), so its CPU and memory
are charged to the environment and a compositor failure cannot take the
supervisor down. The worker owns the writable frame-pool mapping and applies the
seals; the supervisor and every viewer map the pool read-only, and slot
reservations travel back over the control socket so the producer honours a reader
in another process. `CompositorMode::InProcess` keeps a supervisor-thread
fallback for hosts without the worker binary.

**Aggregate environment cgroup.** Each environment owns a transient
`juto-desktop-<env>.slice` carrying the configured `CPUQuota`, `MemoryMax`,
`MemorySwapMax`, and `TasksMax`; applications run as scopes inside that slice, so
the limits bound the whole environment rather than each process. `Destroy` stops
the slice, retiring every scope inside it. A monitor samples the slice's cgroup
and emits `LimitPressure` when memory crosses 90% of the limit or the CPU
throttle counter advances.

**Pacing and statistics.** The compositor runs a virtual clock at the effective
FPS and never renders twice inside one interval or bursts to catch up. `Observe`
reports the tick count, frames composed, frames dropped because every pool slot
was held by a slow reader, and the last composition duration, so a caller can
measure pacing and backpressure instead of inferring them.

**Text input.** The compositor advertises `zwp_text_input_manager_v3` and acts as
the input method for its own clients: focus changes send `enter`/`leave`, and
`TextCommit`/`TextPreedit` input events are delivered as `commit_string` /
`preedit_string` only to instances that enabled themselves for the focused
surface. Events addressed to an idle client are flushed explicitly, so text and
frame callbacks are not delayed until the client happens to send a request.

**Same-host viewer transport.** The daemon's `Attach` command returns the
environment's sealed frame pool as an `SCM_RIGHTS` descriptor alongside
negotiated metadata (pool id, dimensions, stride, format, slot count, slot
size); `ViewerClient` maps it read-only and reads composed frames with no codec
on the path. The control socket authenticates callers by `SO_PEERCRED`, so only
the daemon's own user may drive its environments. A viewer cannot alter desktop
content: the pool is sealed against future writable mappings.

**Remote across-device viewing.** `desktop attach` issues a one-time attach
token (60-second expiry, consumed on use) and returns a loopback viewer URL. The
signaling server serves the browser viewer page and relays SDP/ICE over a
WebSocket; the presentation worker encodes composed frames with GStreamer
`webrtcbin` (VP8 by default, H.264 available) and carries pointer, scroll, and
keyboard input back over a reliable ordered data channel. A native receiver
(`NativeReceiver`) decodes the same stream for non-browser clients. TLS
termination and public reachability belong to the deployment gateway; the
shipped server binds loopback and never creates a world-readable attach URL.

**Hardening.** After launching an application, the supervisor reads the scope's
effective `CPUQuotaPerSecUSec`, `MemoryMax`, and `TasksMax` and fails closed with
`IsolationUnavailable` if they do not match the requested limits, so a silently
uncapped application is never reported as isolated. Control leases and grants
carry deadlines: a lease held by the same caller is renewed without changing its
epoch, an expired lease is cleared by a reaper and rejected, and expired grants
are not reported. `DesktopSupervisor::reconcile` stops orphan
`juto-desktop-*.scope` units and removes stale runtime directories left by a
previous process; the daemon and the runtime call it at startup.

**Not yet shipped: audio and the remaining profiles.** Audio (private PipeWire
sink plus Opus), Xwayland, the hardware render-node profile, DMA-BUF
export/import, and TURN relay configuration are not implemented.
The embedded GPUI viewer still reads the pool in-process; wiring it to the
daemon's `SCM_RIGHTS` attach path is a follow-up. A compile pass is not evidence of
streaming; the shipped behavior is verified by the integration tests and the
smokes described below.

**Remote sharing desktop artifacts in chat.** Agents mount remote desktop sharing
sessions in chat conversations via the `desktop_artifact` tool. Artifact descriptors persist
as IDs-only `DesktopSessionArtifact` records (`version`, `artifact_id`, `title`, `host_id`,
`environment_id`, `grant_id`) without embedding bearer tokens, SDP payloads, or local paths.
The chat view mounts each same-host artifact as an inline canvas (see **Embedded viewer**)
that starts paused. A remote-host artifact shows an explicit notice instead of a canvas,
and a resolver failure shows `Desktop '<title>' cannot be opened: <reason>` in its place.

**Element trees in tool results.** A tool result may attach a declarative tree under
`ToolOutput.details` as `{"type":"view","view":{tag, props, children}}`. The model-facing
`content` is unchanged. The chat and focus views render the tree in its own row after the
result, using the primitives `Column`, `Row`, `Text`, `Badge`, `Stats` with `Stat` cells, `Timeline` with `Lane` rows of `Bar` segments, `Chart` with `Series` lines drawn as bars, a line, dots (`scatter`), or filled areas (`area`), and `Table` with `TableRow` rows of text cells. Hovering a chart point or a timeline bar shows its values in a tooltip. Any unknown tag renders a
visible `Unknown element <tag>` box, never silently dropped. Trees are decoded with serde; no
markup is parsed. The `view` tool lets agents emit a tree. The runtime validates it before the host renders it: allowed tags and props only, `Badge` `tone` one of success, warning, error, or info, `Stat` only inside `Stats`, `Lane` only inside `Timeline`, `Bar` only inside `Lane`, numeric `Timeline` `max` and `tick` in seconds (at most 48 ticks), and bars with `0 <= start < end <= max`. Also at most 8 levels of nesting, 200 elements, and 4096 bytes per string.
`Chart` `kind` is bar, line, scatter, or area; it takes 1 to 4 `Series` children with equal-length comma-separated finite values (at most 48 points), and `labels`, when present, must match that count.
`Table` `columns` lists 1 to 8 comma-separated headers, and it takes at most 48 `TableRow` children, each with exactly one text cell per column.
`Button` takes `label` and `prompt`, both required and non-empty, and an optional `tone` of primary, neutral, or danger; it takes no children. Clicking it submits `prompt` as the next user message. It is disabled while a run is active and inside subagent focus threads.

## Development

Requires Nix with flakes enabled. The supported development system is
`x86_64-linux`; the flake supplies Rust, Cargo, rustfmt, Clippy, rust-analyzer,
Clang, mold, native GPUI libraries, and jj. The resource wrapper requires a
working systemd user manager.

```sh
nix develop
# Alternatively, when using direnv:
direnv allow

scripts/agent-scope.sh -- nix develop . -c cargo build --workspace --locked
scripts/agent-scope.sh --gpu -- nix develop . -c cargo run -p juto --locked
```

The flake also packages the GUI app for `x86_64-linux`. Install the latest
published GitHub revision into your user profile:

```sh
nix profile install github:jarenm1/juto
juto
```

For a flake-based NixOS system configuration, add the flake as an input and
include its package:

```nix
{
  inputs.juto.url = "github:jarenm1/juto";

  # In the host's configuration:
  environment.systemPackages = [ inputs.juto.packages.${pkgs.system}.default ];
}
```

The same package can be added to `home.packages` in a Home Manager
configuration. The app needs a graphical session and working Vulkan driver;
the package supplies Juto's native libraries and uses `/run/opengl-driver/lib`
for the host GPU driver.

**Updates without recompiling.** The flake advertises `juto.cachix.org` via
`nixConfig`; accept the substituter prompt (or run as a `trusted-user`) and
`nix profile upgrade` downloads the CI-built binary instead of compiling.
Without the cache, the crane-based build compiles the dependency graph once
per `Cargo.lock`/toolchain change and caches it in the Nix store — updates
that only touch Juto source recompile a few workspace crates, not the whole
graph, and doc/script-only changes reuse the existing store path entirely.
The cache is populated by `.github/workflows/cache.yml` on pushes to `main`.

Headless informational commands do not initialize GPUI:

```sh
scripts/agent-scope.sh -- nix develop . -c cargo run -p juto --locked -- --version
scripts/agent-scope.sh -- nix develop . -c cargo run -p juto --locked -- --help
```


Ctrl-Q exits the window. Closing the last window also requests application exit.

### Providers and model selection

Open **Providers** (Ctrl-P) to search supported providers and choose a login
method. OpenAI has one entry with **ChatGPT / Codex OAuth** and **OpenAI API Key**
tabs: subscription credentials enable `openai-codex` models, while a Platform
API key enables `openai` models. These credentials can coexist and are removed
independently. Anthropic offers browser login and API-key entry under one
provider; those methods replace the same stored Anthropic credential. Devin
follows the same shape — browser OAuth sign-in against app.devin.ai and a
`DEVIN_API_KEY` service credential — and its `devin-agent` wire transport is
ported: swe-1-6 models appear in the picker once connected and stream over
the Codeium Cascade Connect protocol.

API keys are masked by default. Saving/removing credentials and reading
connection status run in the background, not during GPUI rendering. Browser
login can be cancelled; leaving the login screen also cancels a pending OAuth
flow and clears the key input. Removing a stored credential does not unset an
environment variable, so an environment credential can keep a provider connected.
The provider list and login details scroll independently in compact windows.

TypeSafe (`TYPESAFE_API_KEY`) is a judge provider and appears in the Providers login list as an API-key method. `typesafe/jev-latest` and
`typesafe/jev-preview` (wire ids `jev-latest` and `jev-preview`) call
`POST /v1/systemone`, a classifier that picks one of the caller's discrete options.
They are judge-only: `set_model_role` rejects them for chat roles. The `judge` role
also accepts chat and tiny models served by a connected HTTP provider; those judge
through the chat provider with `TextJudge`. TypeSafe judge models build through
`judge_for_model`, which requires TypeSafe credentials. Verified by the wire contract
in TypeSafe's OpenAPI example, runtime role tests (`model_roles`), and picker unit
tests. Not verified: a live TypeSafe call (no key was available), and the picker's
native-window rendering.

The model picker (Ctrl-M) shows chat models for chat roles. The `judge` role lists
the same chat and tiny models plus TypeSafe judge models, and each scope is limited to
providers with a stored or nonempty environment credential. `juto-cli models` was not
changed by this work.
local models are not included in this connected-only view; explicit CLI/session
model selectors remain available. With no connections, the picker offers
**Manage providers…** rather than fabricated fallback models.

The picker opens as a centered dialog over a dimmed chat view and groups
models by provider, showing context windows, role badges, and tok/s and
DeepSWE columns (`n/a` when data is unavailable). The role assignment panel
lists every configured or built-in role; click a role to choose the
assignment target, then pick a model (or press Enter on a filtered row) to
persist `modelRoles.<role>` into the global `config.yml`. Assigning `default`
also switches the session model and closes the dialog; other roles stay open
so several bindings can be set in one pass. Clicking the scrim, Escape, or
the close button returns to chat; **Manage providers…** opens login.

The centered dialog fits the window up to 900×660 over the translucent scrim.
In narrow windows it hides secondary model IDs and stacks role badges,
keeping the metrics, role summary, and controls visible. Native-window
screenshots at
[960×720](docs/screenshots/model-picker-960x720.png) and
[560×480](docs/screenshots/model-picker-560x480.png) were captured under
Xvfb/Mesa software Vulkan with isolated placeholder credentials for list
rendering, not live provider authentication.

The picker renders only visible rows and caches searchable model metadata.
Credential changes refresh availability immediately; reopening the picker
also picks up credentials changed outside the window.

The bundled catalog is refreshed from
[OMP revision `0affc77e`](https://github.com/can1357/oh-my-pi/blob/0affc77e41c357bb33f8bf056c3ea3cf50fd408b/packages/catalog/src/models.json).
[Snapshot provenance](crates/juto-catalog/data/source.json) records the source,
encoding, SHA-256, and retained MIT license. Catalog availability is a fast
configuration check, not network validation or proof of account entitlement:
an invalid key, expired non-refreshable login, or restricted model can still
fail at inference.

### UI-only iteration

`apps/juto/src/main.rs` launches the native window. `chat.rs` owns the chat view,
plain message rows, attachment dropdown, and Send control; `chat_input.rs` owns
native text editing. UI work stays in this application without changing the agent runtime.

The composer features a modern card with an input field and bottom toolbar row:
a circular **+** button opens a menu to attach **Files…** through the native file
dialog. Selected attachments appear as removable chips (`×`). Messages can be sent
with text, attachments, or attachments alone. The composer supports cursor movement,
selection, Backspace/Delete, clipboard copy/cut/paste. Long drafts wrap onto new rows; the
composer grows to eight rows, then scrolls vertically. Sent messages wrap naturally.

Surfaces are neutral grays—backgrounds, borders, hover states, focus borders,
caret, selection, and disabled states—with one blue accent (`#7aa2f7`) for
selection, active toggles, and primary actions, amber for blocked threads, and
red for failures and destructive actions. `apps/juto/src/theme.rs` holds the
shell and sidebar palette. Messages have no bubble backgrounds or borders.

The window is a full-height left sidebar beside the chat column. The sidebar
(`sidebar.rs`, rows in `task_card.rs`) lists **focus threads** only; the main
chat is not a row. Each row shows the thread title, its status, and a thin bar
marking its lifecycle stage (queued → running → verifying → reviewing → done).
The bar is a stage indicator, not a completion percentage: the runtime reports
no progress measure. Clicking a row selects it and opens focus mode, with the
sidebar still visible; leaving focus clears the selection. Blocked rows always
show the blocker prompt with a resume field. A row's chevron reveals its agent,
duration, workspace, current step, Pause/Steer/Abort, token usage, step log,
and diffs. Filter tabs (All, Active, Blocked, Done) sit under the heading. The
chat header keeps the model picker, **Providers**, and the **Threads** sidebar
toggle at the top right; with the sidebar hidden, the header shows the app
title. The transcript and composer stay capped at 800 px wide.

The input component adapts [GPUI 0.2.2's input example](https://docs.rs/crate/gpui/0.2.2/source/examples/input.rs);
its upstream Apache-2.0 license is retained in
`apps/juto/licenses/GPUI-APACHE-2.0.txt`. Unicode grapheme navigation uses
`unicode-segmentation`, already present in GPUI's dependency graph.

Start UI work in a dedicated jj workspace from the published base:

```sh
jj workspace add ~/workspaces/juto-ui -r main@origin
```

From that workspace, use the default checkout's Git-aware flake while keeping
Cargo's working directory and build cache local to the UI workspace:

```sh
export JUTO_FLAKE="$(dirname "$(jj git root)")"
scripts/agent-scope.sh --gpu -- nix develop "$JUTO_FLAKE" -c cargo run -p juto --locked
# Software Vulkan on a virtual display:
scripts/agent-scope.sh -- nix develop "$JUTO_FLAKE#smoke" -c xvfb-run -a -s "-screen 0 1280x1024x24" cargo run -p juto --locked
```

For remote review, put screenshots in the draft PR's description or comments
using a remotely accessible image URL. Keep capture scripts and screenshots
outside the project source; a remote `/tmp` path does not render for reviewers.

### Fast-build choices

- Reuse `~/jtech`'s exact nixpkgs and flake-parts pins, not its CUDA environment.
- Rust 2024 workspace, resolver 3; workspace-wide dependency versions.
- Unoptimized dev/test builds, line-table debug information, incremental
  compilation. Dependencies remain unoptimized too: no `opt-level = 3` override.
- Clang + mold on `x86_64-unknown-linux-gnu`. No dev LTO, single-codegen-unit
  override, compiler-cache wrapper, or forced host-specific CPU flags.
- Cargo retains its per-workspace `target/` cache. Agent builds use jtech's
  adapted resource wrapper: four jobs by default, with CPU/memory caps and a
  shared hardware GPU lock. See AGENTS.md for overrides and concurrency rules.
- Pin published [GPUI 0.2.2](https://docs.rs/crate/gpui/0.2.2), enabling X11 and
  Wayland only. Avoid a moving Git dependency on the full Zed workspace.
- Track `Cargo.lock` and `flake.lock`; use `--locked` in verification.

These choices favor iteration time over optimized application performance.
No comparative build-speed benchmark has been run.

### Nix and jj

Use Git-aware flake references (`.` and `.#smoke`), **not `path:.`**. A path
flake copies ignored files, including mutable `target/` contents, into the Nix
store; it can fail while a build removes intermediate files. This workspace is
Git-colocated under jj. jj remains the version-control interface; its snapshots
make new files available to Git-aware Nix evaluation.

```sh
jj describe -m "feat: <change>"
jj commit -m "feat: <change>"
```

Use jj to snapshot new source/configuration files before Nix evaluates them.
Do not use Git staging or commits.

## Verification

```sh
nix flake check . --no-build
nix develop . -c cargo fmt --all --check
nix fmt .
scripts/agent-scope.sh -- nix develop . -c cargo build --workspace --locked
scripts/agent-scope.sh -- nix develop . -c cargo test -p juto --locked
scripts/agent-scope.sh -- nix develop . -c cargo clippy --workspace --all-targets --locked -- -D warnings
```

`flake check --no-build` verifies flake evaluation, not an application build.
Build or run the packaged application with `nix build .#default` or
`nix run .#default`. For a headless native-window smoke, see the
`devShells.smoke` instructions below; that exercises the development binary,
not the packaged artifact.

### Agent desktop compositor verification

```sh
scripts/agent-scope.sh -- nix develop . -c cargo test -p juto-desktop --locked
```

The integration tests create real environments and verify admission and
geometry-generation rules, sealed shared-memory round-trip, control-lease
preemption, input validation, a real Wayland client's composed frame (pixel
checked), systemd-scope launch, and destroy cleanup (application scope stopped,
runtime directory removed).

The daemon smoke drives the control protocol end to end with the bundled
deterministic Wayland client:

```sh
scripts/agent-scope.sh -- nix develop . -c cargo build -p juto-desktop --bin juto-desktop --example test_client
./target/debug/juto-desktop --socket /tmp/juto-desktop.sock   # in the dev shell
# then Create -> Launch (target/debug/examples/test_client) -> Describe -> Observe -> Destroy
```

Observed: `Create` returned effective output and limits; `Launch` started the
client in a `juto-desktop-<env>-<app>.scope` systemd scope; `Describe` reported
the mapped window (`juto-test-client`, `dev.juto.test-client`, 320×240);
`Observe` returned a 640×480 frame; `Destroy` removed the runtime directory and
left no active scope. The integration test pixel-verified the composed frame as
the client's two-tone pattern (left red, right blue).

The runtime tool and the embedded viewer are covered by:

```sh
scripts/agent-scope.sh -- nix develop . -c cargo test -p juto-runtime --lib tools::tests::desktop
scripts/agent-scope.sh -- nix develop .#smoke -c xvfb-run -a -s "-screen 0 1280x1024x24" \
  ./target/debug/juto --desktop ./target/debug/examples/test_client 60
```

Observed: the `desktop` tool created an environment, launched `sleep` in a
systemd scope, wrote a lossless PNG snapshot that begins with the PNG signature,
reported one running application, issued a loopback viewer URL, and destroyed the
environment. Under Xvfb/Mesa software Vulkan the `--desktop` mode mounted the canvas
paused: one still captured at mount, `paused` status, no frame loop. A click resumed it
to `live 1280x720` and the client's two-tone frame appeared letterboxed in the harness
window. A 75-second paused hold kept the environment alive (the click still reached
`live 1280x720`); the test client exits after its 60-second argument, so that run's
post-resume canvas was empty. Process-tree CPU over 5-second windows: 1% paused,
187–230% live at 1280×720 (live copies every frame with no change detection). No
Wayland, hardware GPU, or provider access was exercised.

The remote path is covered by an end-to-end WebRTC test and a browser smoke:

```sh
scripts/agent-scope.sh -- nix develop . -c cargo test -p juto-desktop --test remote_tests
# Browser viewer: run the remote_smoke harness and point headless Chromium at the URL it prints.
```

Observed: the native receiver decoded the composed desktop over DTLS/SRTP and
the two-tone pattern survived VP8 (left red, right blue); a headless Chromium
viewer reported `connected: true`, a 640×480 video, a sampled red pixel, and
`inputSent: true`, and the data-channel input reached the environment's virtual
seat. No audio, hardware renderer, Xwayland, TURN relay, or real remote-network
behavior was exercised.

### Release binary cache

`.github/workflows/cache.yml` builds `packages.default` on every `main` push
and uploads the store closure to `juto.cachix.org`; the flake's `nixConfig`
advertises that substituter to installers. Provisioning is manual and lives
with the repo owner: create the public `juto` cache at
<https://app.cachix.org> with Cachix-managed signing, copy its public key into
`flake.nix` `nixConfig`, create a personal access token at
<https://app.cachix.org/personal-access-tokens> and store it as the
`CACHIX_AUTH_TOKEN` Actions secret, then set the `CACHIX_ENABLED=1` Actions
variable. Until then the workflow is inert and installs fall back to local
builds, still split by crane so unchanged dependency graphs are not
recompiled.

For a headless **native-window** check, `devShells.smoke` supplies Xvfb, xdotool,
ImageMagick, Python, a DejaVu font configuration, and a pinned Mesa software
Vulkan driver. It does not require the hardware GPU lock:

```sh
scripts/agent-scope.sh -- nix develop .#smoke -c env -u WAYLAND_DISPLAY xvfb-run -a -s "-screen 0 1280x1024x24" cargo run -p juto --locked
```

`WAYLAND_DISPLAY` must be unset for this command. GPUI picks Wayland whenever it is set, and the window then appears on the host session instead of Xvfb.

This opens a real window on the virtual display and remains running until
closed or sent Ctrl-Q. The explicit display resolution keeps the 960×720
window fully visible; the smoke shell's default 640×480 display clips it.
Automations can find the window with `xdotool search --sync --onlyvisible
--name '^Juto$'`. Focus and resize it to trigger the initial Xvfb redraw:

```sh
xdotool windowfocus --sync <id>
xdotool windowsize --sync <id> 961 721
sleep 1
xdotool windowsize --sync <id> 960 720
sleep 3
magick import -descend -window <id> /tmp/juto.png
xdotool key ctrl+q
```

Run these commands in the same Xvfb session as the application. Synthetic typing
can outpace software rendering: wait for the input queue and display to settle
before capturing or asserting the visible result.
Observed for the agent-wired chat view: locked application build; 15 unit tests
passing in `juto` (116 across workspace); Clippy with warnings denied; native typing,
Enter and Send submission, empty/whitespace rejection, attachment dropdown (+),
attached file chips, message with attachment submission, attachment-only submission,
chip removal (×), copy/cut/paste, multi-line draft wrapping with vertical scrolling past eight rows, wrapped messages, history
scrolling, streaming assistant responses, durable append-only JSONL session creation
and persistence, existing session loading via `--session`, model selection via `--model`,
mounting `DesktopSessionArtifact` cards as inline paused canvases with click-to-resume,
560×480 resizing, and Ctrl-Q exit
status 0 under Xvfb/Mesa software Vulkan.
Host Wayland, hardware acceleration, live provider network transport, and live video streaming backend
were not exercised.

## Dependency constraints

GPUI 0.2.2's `gpui_http_client → zed-async-tar → xattr 0.2.3` dependency references
`libc::ENOATTR`. Building with libc 0.2.190 failed on Linux because that symbol
is absent. `Cargo.lock` pins libc 0.2.186, which still provides the
[deprecated alias](https://docs.rs/libc/0.2.186/libc/constant.ENOATTR.html).
Preserve that resolution until GPUI's dependency chain is updated; a blanket
`cargo update` can break the build.

The pinned Rust toolchain also reports a future-incompatibility warning in
`proc-macro-error2 2.0.1`: a private `proc_macro` extern crate is re-exported
([Rust issue #127909](https://github.com/rust-lang/rust/issues/127909)). It builds
with the current toolchain; compiler upgrades need to recheck that dependency.
The warning is not suppressed.
