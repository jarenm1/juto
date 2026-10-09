# Juto

A Rust agent harness with a native GPUI interface, targeting the runtime
capabilities of [oh-my-pi](https://github.com/can1357/oh-my-pi).

## Current state

The build includes a pinned Nix development environment, a Cargo workspace,
and a native GPUI chat view. You can type in a modern composer card with an
attachment dropdown (**+**), submit with Enter or a circular Send button (**↑**),
and view plain, left-aligned messages in a scrollable list. Empty or whitespace-only
submissions are ignored; messages stay in memory until the window closes.

**The UI is not connected to an agent runtime.** It does not generate assistant
replies, save conversations, connect to providers, or offer login, model
switching, tools, or subagents.

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

### UI-only iteration

`apps/juto/src/main.rs` launches the native window. `chat.rs` owns the chat view,
plain message rows, attachment dropdown, and Send control; `chat_input.rs` owns
native text editing. UI work stays in this application without changing the agent runtime.

The composer features a modern card with an input field and bottom toolbar row:
a circular **+** button opens a menu to attach **Files…** through the native file
dialog. Selected attachments appear as removable chips (`×`). Messages can be sent
with text, attachments, or attachments alone. The composer supports cursor movement,
selection, Backspace/Delete, clipboard copy/cut/paste, and horizontal scrolling for
long drafts. Sent messages wrap naturally.

The UI is styled with neutral grays throughout—including backgrounds, borders,
hover states, focus borders, caret, selection, and active/disabled button states.
Messages have no bubble backgrounds or borders.
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
There is no packaged `nix build`/`nix run` target yet. Unit tests cover Unicode
platform text offsets and IME selection positioning; native interaction still
needs a real-window smoke.

For a headless **native-window** check, `devShells.smoke` supplies Xvfb, xdotool,
ImageMagick, Python, a DejaVu font configuration, and a pinned Mesa software
Vulkan driver. It does not require the hardware GPU lock:

```sh
scripts/agent-scope.sh -- nix develop .#smoke -c xvfb-run -a -s "-screen 0 1280x1024x24" cargo run -p juto --locked
```

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
Observed for the attachment chat view: locked application build; 3 unit tests
passing; Clippy with warnings denied; native typing, Enter and Send submission,
empty/whitespace rejection, attachment dropdown (+), native file chooser dialog,
attached file chips, message with attachment submission, attachment-only submission,
chip removal (×), copy/cut/paste, long-input scrolling, wrapped messages, history
scrolling, 560×480 resizing, and Ctrl-Q exit status 0 under Xvfb/Mesa software Vulkan.
Host Wayland, hardware acceleration, provider access, and an actual platform IME
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
