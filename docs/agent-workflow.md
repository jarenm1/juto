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
- The GPU lock is shared with jtech. `smoke` uses software Vulkan and needs no
  GPU lock, but still needs a display: use Xvfb, not an invented `--headless`
  flag.
- Light commands (`cargo fmt`, `cargo metadata`, jj, source search) need no
  scope.

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

### Shared compile cache

- Dev shells (`default` and `smoke`) set `RUSTC_WRAPPER` to the flake's pinned
  sccache, export `SCCACHE_DIR` (default `${XDG_CACHE_HOME:-~/.cache}/sccache`)
  and `SCCACHE_CACHE_SIZE` (default `10G`, LRU eviction), and unset
  `CARGO_INCREMENTAL`. Override the two directories in the environment.
- Identical rustc invocations (same toolchain, flags and registry source) hit
  one entry from any workspace's `target/`. Workspace members are compiled
  incrementally; sccache passes those through and reports them as
  "incremental". Cargo already builds registry dependencies non-incremental,
  so no dependency-only profile override is needed.
- `CARGO_INCREMENTAL=1` in the environment forces incremental compilation for
  every crate, dependencies included. sccache then refuses to run
  (`incremental compilation is prohibited: Unset CARGO_INCREMENTAL`). Shells
  inherit it from older sessions, so the dev shell unsets it; keep it unset
  elsewhere.
- sccache runs one server per user on `127.0.0.1:4226`, shared by every shell
  on the host. `SCCACHE_DIR` and `SCCACHE_CACHE_SIZE` are read when the server
  starts; the first shell to start it wins. Run `sccache --stop-server` after
  changing either.
- Check hits with `sccache --show-stats` inside a dev shell. Build scripts,
  proc-macros and binaries are not Rust cache hits. The `cc` crate also detects
  `RUSTC_WRAPPER=sccache` and caches supported C/C++/assembler invocations;
  other native build systems are not automatically wrapped.
- Trust boundary: every agent on the host shares this cache. Content-triggered
  (untrusted) work MUST NOT read or write it. Run its cargo commands with
  `env RUSTC_WRAPPER= cargo ...`, which bypasses sccache entirely. Clearing
  `RUSTC_WRAPPER` in the parent shell is insufficient: the dev shell sets it.
- Measured (one serial run per row, `cargo build -p juto`, host loadavg 12-14
  on 24 cores, isolated `SCCACHE_DIR`):
  - Cold, empty cache, workspace A: 2m44s wall; 0 Rust hits, 635 misses; cache
    520 MiB after.
  - Cold, workspace B at the same `main` commit, cache populated by A:
    2m02s wall; 478 hits / 635 requests (75.3%), Rust 436/593 (73.5%), 0
    cache errors.
  - Cold, workspace B, `RUSTC_WRAPPER=` (no cache): 2m24s wall.
  - Edit loop (touch `crates/juto-ai/src/lib.rs`, rebuild): 4.9 s with the
    wrapper, 4.0 s without.
  Caveats: one run per row under shared load, so the 15% cold-build gain is
  close to load noise; cargo's `user` time excludes the sccache server, so
  wall time is the comparable figure. The benefit grows with each workspace
  that builds the same dependency set; a fully cold cache gains nothing.

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
