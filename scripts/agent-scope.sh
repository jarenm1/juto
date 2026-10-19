#!/usr/bin/env bash
# Run an agent command inside a resource-governed systemd user scope.
#
# The limits come from `juto-cli resources grant`, the same derivation the
# runtime uses: CPU and IO are weighted (work-conserving, no CPUQuota unless
# asked), memory is capped at a grant from measured capacity and the command's
# recorded usage, and the scope joins the shared agents slice. If the grant
# cannot be computed or applied the script fails closed and never runs the
# command unscoped. See docs/resources.md.
#
#   scripts/agent-scope.sh -- nix develop . -c cargo build -p juto
#   scripts/agent-scope.sh --memory-max 4G -- cargo build -p juto-runtime
#   scripts/agent-scope.sh --gpu -- nix develop . -c cargo run -p juto
#
# Options: --cpu-quota, --memory-max, --slice, --no-slice, --gpu.
# Environment: AGENT_CPU_QUOTA, AGENT_MEMORY_MAX, AGENT_SLICE,
# AGENT_SLICE_CPU_QUOTA, AGENT_SLICE_MEMORY_MAX, AGENT_GPU_LOCK, JUTO_CLI.
# juto-cli is taken from $JUTO_CLI, then PATH, then this checkout's target/debug.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
# Share jtech's lock: both projects use the same physical GPU.
gpu_lock="${AGENT_GPU_LOCK:-${XDG_RUNTIME_DIR:-/tmp}/jtech-agent-gpu.lock}"
gpu=0
cli_args=()

usage() {
    cat <<'EOF'
usage: scripts/agent-scope.sh [--cpu-quota N%] [--memory-max SIZE]
                              [--slice NAME | --no-slice] [--gpu] -- COMMAND...

Runs COMMAND in a systemd user scope whose limits come from `juto-cli resources
grant. Refuses to run when the grant cannot be computed or applied.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu-quota) cli_args+=(--cpu-quota "$2"); shift 2 ;;
        --memory-max) cli_args+=(--memory-max "$2"); shift 2 ;;
        --slice) cli_args+=(--slice "$2"); shift 2 ;;
        --no-slice) cli_args+=(--no-slice); shift ;;
        --gpu) gpu=1; shift ;;
        -h | --help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *) break ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "agent-scope: no command given" >&2
    usage >&2
    exit 2
fi

if ! command -v systemd-run >/dev/null 2>&1; then
    echo "agent-scope: systemd-run unavailable; refusing an uncapped run" >&2
    exit 1
fi

if [[ -n "${JUTO_CLI:-}" ]]; then
    cli="$JUTO_CLI"
elif command -v juto-cli >/dev/null 2>&1; then
    cli="$(command -v juto-cli)"
elif [[ -x "$repo/target/debug/juto-cli" ]]; then
    cli="$repo/target/debug/juto-cli"
else
    echo "agent-scope: juto-cli not found (set JUTO_CLI, put it on PATH, or run" >&2
    echo "agent-scope: cargo build -p juto-cli); refusing an unscoped run" >&2
    exit 1
fi
if [[ ! -x "$cli" ]]; then
    echo "agent-scope: $cli is not executable; refusing an unscoped run" >&2
    exit 1
fi

# The grant is shell assignments (GRANT_CARGO_JOBS, GRANT_SLICE,
# GRANT_PROPERTIES). `--apply-slice` installs the shared slice's transient caps
# first, and a failure there stops the run.
if ! grant="$("$cli" resources grant --shell --apply-slice \
    ${cli_args[@]+"${cli_args[@]}"} -- "$@")"; then
    echo "agent-scope: resource grant failed; refusing an unscoped run" >&2
    exit 1
fi
eval "$grant"

# Cargo parallelism follows the grant's memory budget unless the caller set it.
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-$GRANT_CARGO_JOBS}"

scope=(systemd-run --user --scope --quiet)
for property in "${GRANT_PROPERTIES[@]}"; do
    scope+=(-p "$property")
done
if [[ -n "${GRANT_SLICE:-}" ]]; then
    scope+=("--slice=$GRANT_SLICE")
fi
scope+=(--)

# The GPU is a single shared device: `--gpu` serializes jobs so concurrent
# agents cannot each open a Vulkan context on it. The smoke shell's Mesa
# software renderer does not need the lock.
if ((gpu)); then
    # Hold the lock on fd 9 in this shell and close fd 9 for the command. An
    # exec'd `flock` would pass the lock to every daemon the command starts
    # (pipewire keeps it after the build exits).
    exec 9>"$gpu_lock"
    flock 9
    "${scope[@]}" "$@" 9>&-
else
    exec "${scope[@]}" "$@"
fi
