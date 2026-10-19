# Agent workflow details

Reference for `AGENTS.md`. Read the section you need.

## Flake source

Evaluate the committed flake at `main@origin`, pinned by full 40-character
commit id:

```sh
export JUTO_FLAKE="git+file://$(dirname "$(jj git root)")?rev=$(jj log -r main@origin --no-graph -T commit_id)"
```

- Short hashes fail with "wrong length".
- The bare checkout path failed for one agent with a missing `glib-2.0`.
- Secondary workspaces are not git-colocated. Evaluating their `.` flake copies
  ignored build outputs into the Nix store.
- The pinned rev excludes uncommitted `flake.nix` edits. If a task changes the
  flake, evaluate a filtered source snapshot instead.
- If the command errors or prints more than one id, `main@origin` is divergent.
  Report it and stop; do not guess a commit id.

## Resource governance

```sh
scripts/agent-scope.sh -- nix develop "$JUTO_FLAKE" -c cargo build -p juto
scripts/agent-scope.sh -- nix develop "$JUTO_FLAKE" -c cargo test -p <touched-crate>
scripts/agent-scope.sh -- nix develop "$JUTO_FLAKE" -c cargo clippy --workspace --all-targets -- -D warnings
scripts/agent-scope.sh --gpu -- nix develop "$JUTO_FLAKE" -c cargo run -p juto
scripts/agent-scope.sh -- nix develop "$JUTO_FLAKE#smoke" -c <visual-smoke-command>
```

- Defaults: 400% CPU and 8 GiB per scope, no swap; shared `agents.slice` at
  600% CPU and 16 GiB.
- Override with `--cpu-quota`, `--memory-max`, `--slice`, or `AGENT_*`
  variables. Do not bypass caps.
- Requires a working systemd user manager. Fails rather than running uncapped
  if `systemd-run` is absent.
- This script governs developer commands only. The runtime's own bash sandbox and execution
  tiers are documented in [docs/sandbox.md](sandbox.md).
- The GPU lock is shared with jtech. `smoke` uses software Vulkan and needs no
  GPU lock, but still needs a display: use Xvfb, not an invented `--headless`
  flag.
- Light commands (`cargo fmt`, `cargo metadata`, jj, source search) need no
  scope.
- Runs through `juto-cli resources run` are recorded (see `docs/resources.md`);
  `scripts/agent-scope.sh` runs are not, because they bypass the runtime.

## Build profile

- Keep unoptimized incremental dev and test profiles, reduced debug info, and
  Clang + mold linking.
- Do not copy jtech's optimized dependency profiles, CUDA toolchain, or release
  LTO into dev builds without measured justification.
- Leave Cargo's job count to resource governance.
- Do not override `CARGO_TARGET_DIR` to a shared writable directory. Keep build
  outputs per workspace.
- Track `Cargo.lock` and `flake.lock`. Regenerate Cargo locks with Cargo, not by
  hand.

## Known build gap

`flake.nix`'s `nativeLibraries` is incomplete for the desktop crates, so a clean
`cargo build -p juto` can fail at link time. Report the first error line. Do not
patch `flake.nix` inside an unrelated task.

## Pull requests

1. `jj git fetch && jj rebase -d main@origin`
2. `jj bookmark set <type>/<slug> -r <change-id>`, then
   `jj git push --bookmark <type>/<slug>`
3. From the default checkout, since `gh` needs git and secondary workspaces
   lack it:
   `gh pr create --repo jarenm1/juto --base main --head <type>/<slug>`.
   Use `.github/pull_request_template.md` if it exists.
4. The PR body lists change ids, verification commands and results, observed
   behavior for runtime or UI changes (screenshot paths outside the repo), and
   any manual steps still required.
5. Report the PR URL. Do not merge.

## Evidence files

Save screenshots, recordings, and smoke scripts outside tracked source, for
example in `/tmp`, and keep them available for review.
