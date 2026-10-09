#!/usr/bin/env bash
# Run an agent command inside a resource-capped systemd user scope.
#
# Caps CPU and memory so parallel agents cannot saturate the workstation, and
# optionally serializes GPU access behind a lock. Every scope also joins one
# shared slice whose aggregate cap bounds the total across all agents, not just
# each scope. See AGENTS.md, "Resource governance".
#
#   scripts/agent-scope.sh -- nix develop . -c cargo build -p juto
#   scripts/agent-scope.sh --cpu-quota 200% -- cargo build
#   scripts/agent-scope.sh --gpu -- nix develop . -c cargo run -p juto
#
# Environment overrides: AGENT_CPU_QUOTA, AGENT_MEMORY_MAX, AGENT_GPU_LOCK,
# AGENT_SLICE, AGENT_SLICE_CPU_QUOTA, AGENT_SLICE_MEMORY_MAX.
set -euo pipefail

cpu_quota="${AGENT_CPU_QUOTA:-400%}"
memory_max="${AGENT_MEMORY_MAX:-8G}"
# Share jtech's lock: both projects use the same physical GPU.
gpu_lock="${AGENT_GPU_LOCK:-${XDG_RUNTIME_DIR:-/tmp}/jtech-agent-gpu.lock}"
slice="${AGENT_SLICE:-agents.slice}"
slice_cpu_quota="${AGENT_SLICE_CPU_QUOTA:-600%}"
slice_memory_max="${AGENT_SLICE_MEMORY_MAX:-16G}"
gpu=0

usage() {
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu-quota) cpu_quota="$2"; shift 2 ;;
        --memory-max) memory_max="$2"; shift 2 ;;
        --slice) slice="$2"; shift 2 ;;
        --no-slice) slice=""; shift ;;
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

# Cargo defaults to one job per core; scale it to the quota so a capped scope
# does not spawn more compilers than it can run.
jobs=$((${cpu_quota%\%} / 100))
((jobs < 1)) && jobs=1
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-$jobs}"

if ! command -v systemd-run >/dev/null 2>&1; then
    echo "agent-scope: systemd-run unavailable; refusing an uncapped run" >&2
    exit 1
fi

scope=(
    systemd-run --user --scope --quiet
    -p "CPUQuota=$cpu_quota"
    -p "MemoryMax=$memory_max"
    -p "MemorySwapMax=0"
)

# Every agent scope joins one shared slice so the aggregate across parallel
# agents is bounded, not just each scope. Fill in only the caps the human has
# not already set (a persistent drop-in, or a non-infinity runtime value).
if [[ -n "$slice" ]]; then
    if command -v systemctl >/dev/null 2>&1; then
        props=()
        cpu_now=$(systemctl --user show "$slice" -p CPUQuotaPerSecUSec --value 2>/dev/null || true)
        if [[ -z "$cpu_now" || "$cpu_now" == "infinity" ]]; then
            props+=("CPUQuota=$slice_cpu_quota")
        fi
        mem_now=$(systemctl --user show "$slice" -p MemoryMax --value 2>/dev/null || true)
        if [[ -z "$mem_now" || "$mem_now" == "infinity" ]]; then
            props+=("MemoryMax=$slice_memory_max")
        fi
        if ((${#props[@]} > 0)); then
            systemctl --user set-property "$slice" "${props[@]}" >/dev/null 2>&1 || true
        fi
    fi
    scope+=("--slice=$slice")
fi

scope+=(--)

# The GPU is a single shared device: `--gpu` serializes jobs so concurrent
# agents cannot each open a Vulkan context on it. The smoke shell's Mesa
# software renderer does not need the lock.
if ((gpu)); then
    exec flock "$gpu_lock" "${scope[@]}" "$@"
fi
exec "${scope[@]}" "$@"
