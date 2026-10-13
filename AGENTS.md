# Agent instructions

Juto is a Rust agent runtime with a native GPUI interface. Build and launch
commands: `README.md`. Detailed workflow: `docs/agent-workflow.md`.

## Rules

1. **Use jj** for commits and workspaces. Never stage or check out with git.
   Exception: `gh`, run from the default checkout.
2. **Work in your own workspace**, never in the default checkout:
   `jj workspace add ~/workspaces/juto-<slug> -r main@origin`.
   Re-check `main@origin` before starting and before opening the PR.
3. **Wrap heavy commands** (build, test, clippy, run) in
   `scripts/agent-scope.sh -- nix develop "$JUTO_FLAKE" -c ...`. GPU runs add
   `--gpu`. Visual smoke uses the `smoke` shell under Xvfb.
4. **Ship observable behavior.** Compiling is not proof. Exercise the changed
   path and report exactly what you saw.
5. **Commit as you go.** Each commit is one logical change that compiles and
   passes its relevant tests, with a conventional prefix (`feat:`, `fix:`,
   `refactor:`, `docs:`, `test:`, `chore:`). Review `jj diff` before each
   commit. Keep history linear: `jj rebase`, never merge commits.
6. **Leave the default checkout alone.** Do not resolve, delete, or move its
   divergent or conflicted bookmarks. Refer to divergent changes by commit id.
7. **One owner per workspace.** Parallel agents use separate workspaces. A
   shared workspace has one owner, who alone mutates jj state.

## Runtime design

- Read `docs/omp-runtime.md` before work on provider login, model selection,
  agent execution, tools, sessions, subagents, or extensions.
- Keep agent execution independent of GPUI. The UI consumes events and submits
  commands.
- Keep credentials out of transcripts, source, screenshots, and logs.
- Model streaming, errors, cancellation, and tool-call/result pairing
  explicitly. Never fabricate successful output.

## Verification and delivery

- Before reporting complete: format changed files, compile touched crates, run
  relevant tests, and smoke the changed path.
- Do not claim Wayland, hardware acceleration, provider access, or OAuth from a
  CLI or Xvfb smoke.
- Tasks that change tracked files end with a PR. Read-only investigations end
  with a written report instead.
- PR steps and body requirements: `docs/agent-workflow.md#pull-requests`.
- Report the workspace, change ids, verification commands and results, and the
  PR URL. Do not merge; the human merges.
