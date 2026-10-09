# Juto

A Rust agent harness with a native GPUI interface, targeting the runtime
capabilities of [oh-my-pi](https://github.com/can1357/oh-my-pi).

## Current state

The first step is implemented: a pinned Nix development environment, a Cargo
workspace, a real GPUI window, and jj-based agent workflow instructions.
**The agent runtime is not implemented yet.** There are no provider connections,
login flows, chat messages, model switching, tools, or subagents in this build.
The window identifies itself as the build-foundation surface rather than
simulating those features.

See [the full OMP runtime inventory and port plan](docs/omp-runtime.md) for the
capabilities to preserve, primary-source references, and acceptance criteria.
Agent contributors should read [AGENTS.md](AGENTS.md).

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

The window uses Vulkan and either X11 or Wayland; graphical launch needs a
working display and GPU driver. The default shell adds `/run/opengl-driver/lib`
and the native libraries to `LD_LIBRARY_PATH`. The binary is not yet packaged
for standalone installation: run it inside the development shell.

Headless informational commands do not initialize GPUI:

```sh
scripts/agent-scope.sh -- nix develop . -c cargo run -p juto --locked -- --version
scripts/agent-scope.sh -- nix develop . -c cargo run -p juto --locked -- --help
```

Ctrl-Q exits the window. Closing the last window also requests application exit.

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
scripts/agent-scope.sh -- nix develop . -c cargo clippy --workspace --all-targets --locked -- -D warnings
```

`flake check --no-build` verifies flake evaluation, not an application build.
There is no packaged `nix build`/`nix run` target yet, and no permanent behavior
tests for this native-window bootstrap.

For a headless **native-window** check, `devShells.smoke` supplies Xvfb, xdotool,
ImageMagick, Python, a DejaVu font configuration, and a pinned Mesa software
Vulkan driver. It does not require the hardware GPU lock:

```sh
scripts/agent-scope.sh -- nix develop .#smoke -c xvfb-run -a cargo run -p juto --locked
```

This opens a real window on the virtual display and remains running until
closed or sent Ctrl-Q. Automations can find it with `xdotool search --sync
--onlyvisible --name '^Juto$'`, capture it with `magick import -window <id>
/tmp/juto.png`, focus it, and send `xdotool key ctrl+q` from the same Xvfb session.

Observed during setup: successful locked workspace build; a rendered 640×400
native window under Xvfb/Mesa; Ctrl-Q exit status 0; correct headless version/help
output and invalid-argument exit status 2; mold 2.42.0 recorded in the binary.
Host Wayland and hardware acceleration were not exercised.

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
