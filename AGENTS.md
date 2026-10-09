# Agent instructions

Juto: a Rust agent runtime with a native
GPUI interface. The runtime port is the product goal, not a claim of current
implementation. Read `README.md` for the current build and launch commands.

## Non-negotiables

1. **Use jj.** Inspect the working-copy changes before editing; preserve the
   human's work. Use jj for commits and workspaces, not git staging or checkout.
2. **Wrap heavy commands in `scripts/agent-scope.sh`.** Builds, tests, clippy,
   and application runs need resource governance, including commands launched
   through Nix or direnv. No automatic enforcement hook is installed here.
3. **Serialize hardware GPU access with `--gpu`.** Prefer the `smoke` Nix shell
   for automated visual checks: Xvfb and Mesa software Vulkan, not the host GPU.
4. **Ship observable behavior.** A compile pass does not prove streaming,
   cancellation, persistence, authentication, or visible UI behavior. Exercise
   the changed path and report precisely what was observed.

## Workspace isolation

The default workspace is the human's checkout. Agent edits, builds, tests, and
application runs belong in a dedicated jj workspace under `~/workspaces/`,
not in default. `origin` is configured; use `main@origin` as the base for new
independent tasks after fetching. Do not assume PR or branch-protection policy.

Use Git-aware flake references (`.` / `.#smoke`), not `path:.`: the latter
copies ignored build outputs into the Nix store and can race active builds.
Keep new files snapshotted with jj so Nix includes them.

Before starting a task, inspect `jj root` and create a dedicated workspace from
the published base:

```sh
jj workspace add ~/workspaces/juto-<slug> -r main@origin
```

When explicitly moving an existing unfinished change, preserve that change id
and edit it in the new workspace; leave default on a clean published base.

Use one integration owner. Delegated slices may edit the integration workspace
only with explicit, nonoverlapping file ownership; the owner alone mutates jj
state and runs final verification. Otherwise use separate workspaces. Keep
build outputs per workspace; do not override `CARGO_TARGET_DIR` to a shared
writable directory. Long-lived processes use unique names and ephemeral ports.

## Resource governance

Wrap the entire process tree, with the scope outermost:

```sh
scripts/agent-scope.sh -- nix develop . -c cargo build -p juto
scripts/agent-scope.sh -- nix develop . -c cargo test -p <touched-crate>
scripts/agent-scope.sh -- nix develop . -c cargo clippy --workspace --all-targets -- -D warnings
scripts/agent-scope.sh --gpu -- nix develop . -c cargo run -p juto
scripts/agent-scope.sh -- nix develop .#smoke -c <visual-smoke-command>
```

The wrapper inherits jtech's defaults: 400% CPU and 8 GiB memory per scope,
no swap; the shared `agents.slice` defaults to 600% CPU and 16 GiB total. Cargo
jobs derive from the CPU quota. Existing slice limits are preserved. Override
with `--cpu-quota`, `--memory-max`, `--slice`, or the documented `AGENT_*`
variables when required; do not silently bypass caps. It requires a working
systemd user manager and fails rather than run uncapped if systemd-run is absent.

The default GPU lock is shared with jtech because both projects use the same
physical device. The `smoke` shell uses software Vulkan and needs no hardware
GPU lock. It still requires a display server: use Xvfb, not an invented
application `--headless` flag. Light commands (`cargo fmt`, `cargo metadata`,
jj, source searches) need no scope.

## Commits

- Name each logical change with a conventional prefix (`chore:`, `feat:`,
  `fix:`, `refactor:`, `docs:`).
- `jj describe -m "<type>: <what>"` names the current change;
  `jj commit -m "<type>: <what>"` finalizes it and opens a new working copy.
  Snapshotting is automatic; no staging step is needed.
- Amend only changes owned by this task. Preserve unrelated edits and leave
  workspaces in place for review. Fetch, rebase, push, or create a PR only when
  the task calls for it and the relevant remote/bookmarks actually exist.
- Track `Cargo.lock` and `flake.lock`. Change dependencies only as the task
  requires; regenerate Cargo locks with Cargo rather than hand-editing them.

## Runtime and GPUI design

Before implementing provider login, model selection, agent execution, tools,
sessions, subagents, or extensions, read `docs/omp-runtime.md`: it identifies
upstream sources, port scope, compatibility hazards, and acceptance criteria.
Cite and retain the upstream license when copying source, prompts, or assets.

- Keep agent execution independent of GPUI. The UI consumes runtime events and
  submits commands; it does not own provider connections, credentials, tools,
  persistence, or subagent scheduling.
- Keep blocking filesystem/process/network work and the agent loop off GPUI's
  foreground thread. Use GPUI contexts for UI state updates.
- Model streaming, terminal errors, cancellation, tool-call/result pairing,
  and subagent lifetimes explicitly. Render partial output as partial output;
  never manufacture successful responses or silently drop execution errors.
- Keep credentials outside transcripts, source, screenshots, and logs. Use
  provider-supported login and refresh flows; session data is not a token store.
- Port behavior deliberately. Tool implementations may invoke external
  interpreters or language servers; running the TypeScript OMP agent underneath
  a Rust window is not a Rust runtime port.
- Add crates when they own real behavior, not empty public interfaces or
  speculative abstraction. Reuse existing conventions and migrate callers
  together when an interface changes.

## Environment and verification

Use `nix develop . -c <command>` in unhooked shells, or allow direnv once
and use `direnv exec <workspace> <command>`. Keep the scope outermost for heavy
commands. The flake's compiler is authoritative; a second rustup toolchain can
change artifacts and undo reproducibility.

Optimize for build iteration: preserve unoptimized incremental dev/test
profiles, reduced debug info, and Clang + mold linking. Do not copy jtech's
optimized dependency profiles, CUDA toolchain, or release LTO into dev builds
without measured justification. Leave Cargo's job count to resource governance.

Before reporting a code change complete:

1. Format changed Rust/Nix files with the configured toolchain.
2. Compile the touched crates (whole workspace for cross-crate changes), then
   run existing relevant behavioral tests. Add deterministic regression tests
   for consumer-visible invariants, not tests of source text or trivial wiring.
3. Smoke the real changed path. For GPUI, launch the native window, interact
   with the changed controls, capture a screenshot, and inspect it. For runtime
   changes, exercise the real command/event flow, including applicable failures
   and cancellation. State which credential-dependent flows could not be run.
4. Update behavior/build documentation after verification. Distinguish shipped
   capabilities from proposed parity work.

## Handoff and visual evidence

Report the workspace, jj change id(s), actual verification commands and results,
and any concrete manual steps still required. Save transient screenshots,
recordings, and smoke scripts outside tracked source (for example `/tmp`);
keep them available for review. Do not claim Wayland, hardware acceleration,
provider access, or OAuth was verified from a CLI-only or Xvfb smoke.

The human owns publication and merging. When asked for a PR, follow the
repository's template if one exists and include evidence for behavioral or UI
changes; otherwise hand off the local jj change directly.
