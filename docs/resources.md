# Resource usage records

Juto records what each governed command actually used, and derives the limits
each command runs under from the host's measured capacity. This document covers
recording, inspection, capacity, grants, admission queueing, and agent admission
(see Admission and queueing and Agent admission).

## Where records live

Records live under the data directory (`JUTO_DATA_DIR`, else
`$XDG_DATA_HOME/juto`, else `~/.local/share/juto`):

- `resources/usage.jsonl`: one JSON object per finished command, appended.
- `resources/pending/<uuid>.usage`: a per-command report file while the
  command runs. It is removed when the command finishes.

Malformed lines in `usage.jsonl` are skipped when reading.

## What is recorded

| field | meaning |
| --- | --- |
| `recorded_at_ms` | Unix milliseconds when the record was written |
| `repo` | repository root the command ran for (the workspace `repo_root`) |
| `class` | command class, see below |
| `outcome` | `completed`: the wrapper reported the payload's exit status. `aborted`: no report arrived, so `exit_code` is the child's own status (128+signal when killed by a signal) |
| `exit_code` | exit status of the payload |
| `wall_ms` | wall-clock time of the run |
| `memory_peak_bytes` | peak memory of the unit's cgroup (`memory.peak`) |
| `cpu_usec` | CPU time of the unit's cgroup (`cpu.stat` `usage_usec`) |
| `oom_kills` | OOM kills in the unit's cgroup (`memory.events` `oom_kill`) |
| `io_read_bytes`, `io_write_bytes` | cgroup IO totals (`io.stat`); `null` when the `io` controller is not delegated to the unit |

A field is `null` when it could not be read. Recording never changes a
command's result.

## How usage is measured

systemd drops a unit's cgroup accounting when the unit exits, so it cannot be
read afterwards. Instead, a governed payload runs as
`sh -c <wrapper> juto-usage <pending-file> <payload…>`. The wrapper runs the
payload unchanged, reads the cgroup it is in (from `/proc/self/cgroup`), appends
one report line to the pending file, and exits with the payload's status.

When the sandbox is enabled, the pending directory is the only extra writable
path granted to the unit. The rest of the data directory, including
credentials, stays hidden.

## Command class

A class is the program's basename followed by its plain tokens, in order.
Tokens containing `/`, `$`, quotes, or other characters outside
`[A-Za-z0-9._:=+@-]` are dropped, so checkout paths do not split one command
into many classes. At most 8 tokens, each at most 64 bytes.

Shell strings (`sh -c`, `bash -c`, `zsh -c`, `dash -c`) are classified by their
first simple command. Leading `cd`, `export`, `set`, `true`, and `NAME=value`
assignments are skipped. For example, `sh -c 'i=0; while …'` is classified as
`while -lt 2000000`.

## Which commands are recorded

- `juto-cli resources run [--priority CLASS] -- <argv>`: always, under default limits
  and through the admission queue. Scope enforcement stays on, so it fails closed
  when systemd is unavailable.
- Scoped bash commands: when `bash_scoped` is enabled in the runtime
  configuration (off by default).
- Any other `run_scoped`, `spawn_scoped`, or `run_scoped_blocking` call on a
  workspace whose limits set a usage store. `spawn_scoped_blocking` never records.

Unscoped and bare fallback commands are not recorded. Command builders
(`build_scoped_std_command_named`) return a command without running it, so they never record.

## Inspecting records

```sh
juto-cli [--data-dir DIR] resources                 # table per command class
juto-cli [--data-dir DIR] --json resources          # JSON array
juto-cli [--data-dir DIR] resources --repo PATH     # one repository root
juto-cli [--data-dir DIR] resources run [--cwd DIR] -- <argv…>
```

Each class shows runs, aborted runs, OOM kills, and nearest-rank p50/p95 of
memory peak, CPU time, and wall time. Missing values print as `-`.

`--json` is a global flag, so its help text describes runtime events. For
`resources` it selects the JSON listing.

## Capacity and grants

`juto-capacity` measures the host once per decision (`MachineCapacity::measure`):

| field | derivation |
| --- | --- |
| `mem_total_bytes` | `MemTotal` from `/proc/meminfo` |
| `cores` | `available_parallelism()` |
| `reserved_bytes` | working set of the user's session outside `agents.slice`: anonymous memory, kernel stacks and page tables from each cgroup's `memory.stat`, with the slice's working set subtracted. Page cache is excluded because the kernel reclaims it, so cached files never reserve memory. Plus one tenth of total memory as host headroom. Without a `user-<uid>.slice`, this is `MemTotal - MemAvailable` |
| `budget_bytes` | `mem_total_bytes - reserved_bytes`; measurement fails when this is not positive |
| `slice_memory_high_bytes` | `budget_bytes` × 9/10 |
| `tasks_max` | `cores` × 128 |

A command's grant is its memory cap (`MemoryMax`), with `MemoryHigh` at 9/10 of
it. Its cargo parallelism is `min(cores, memory / 1 GiB)`, at least one job.

The memory cap for a command class comes from its recorded history:

- With history: the p95 of recorded memory peaks plus a quarter of that peak
  (nearest-rank p95 over `usage.jsonl`).
- If any recorded run for the class was OOM-killed: the larger of the grant
  computed above and twice the largest recorded peak.
- First run (no history): half the budget.

Every grant is clamped to `[1 byte, budget_bytes]`. CPU and IO use
`CPUWeight`/`IOWeight` of 100, which shares the machine with other scopes and
uses idle cores. `CPUQuota` is set only when requested explicitly.

`TasksMax` is `tasks_max`. `MemorySwapMax` is `0`.

The shared `agents.slice` is configured the same way. Properties already set on
it, by an administrator or a drop-in, are kept and not overwritten. Only unset
properties are applied, with `systemctl --user --runtime set-property`, so no
transient limit is written into persistent user configuration. If a shared
property cannot be applied, the command fails instead of running uncapped.

Inspect and apply the same derivation:

```sh
juto-cli resources capacity                       # measured capacity, JSON with --json
juto-cli resources grant -- cargo build -p juto   # grant for one command
juto-cli resources grant --shell -- cargo build   # shell assignments for scripts
```

`scripts/agent-scope.sh` evaluates `resources grant --shell --apply-slice`, so
it uses the same numbers. It fails closed when `juto-cli` cannot be found. Build
it first with `cargo build -p juto-cli` when it is not on `PATH`.

Without systemd, no limits are applied and `CARGO_BUILD_JOBS` is left to cargo.
Callers that need enforcement fail closed before running.

## Admission and queueing

A governed command reserves its memory ceiling in a shared ledger before it
starts. The ledger is `broker/broker.json` under the resources directory, guarded
by `broker/broker.lock`. The runtime, `juto-cli`, and detached jobs share it, so
the budget holds across processes.

- A command starts when its reservation fits the agent memory budget
  (`juto-capacity`), no other GPU command holds the exclusive GPU slot (for GPU
  commands), and memory stall is not rising (`full avg10` above `full avg60` in
  `/proc/pressure/memory`). A steady stall from other work on the host does not block
  admission, and without PSI the ledger alone gates admission.
- A request larger than the budget fails at once; it never queues.
- The queue is ordered by class: `attended`, then `verification`, `background`,
  and `scheduled`. Each 60 seconds of waiting raises a request one class, capped
  at `attended`, so an old scheduled command eventually runs ahead of fresh
  attended ones. Equal requests run in arrival order.
- A queued request that is cancelled or dropped leaves the queue at once and
  never blocks the requests behind it.
- Ledger entries of dead processes are pruned on every update. Liveness is the
  pid together with its start time, so a reused pid does not keep a reservation.
- The reservation is the command's resolved memory ceiling. Recorded usage does
  not yet change it.

Runtime: when `bash_scoped` is enabled (off by default), a bash command in the main
session is `attended`; a delegated one is `background`. A bash command that must wait returns at once with a job id
(`status: "queued"`, its queue position, and the reserved MiB). Its result is
steered into the run when it finishes, and the run does not stop while such a job
is outstanding. `bash_job` reports a job's status; with `wait: true` it blocks
until the job has a result.

CLI: `juto-cli resources run` waits in the same queue and prints its position on
stderr while it waits. Its class defaults to `attended` and is set with
`--priority`.

Automations: a scheduled occurrence's main run is admitted at the automation's
priority, `scheduled` by default or `background`. See unattended-runs.md, Automations.

## Agent admission

Agents are not capped by a fixed count. Each run tree shares three pools
(`crates/juto-runtime/src/pools.rs`):

- Desks: each agent turn reserves `budget_bytes / 512` bytes in the same broker
  ledger, as an `attended` request. A full budget admits 512 desks; later turns
  queue for memory like any command. A turn holds its desk while it runs, and gives
  it back while it waits on children (the `task` sync wait and the `wait` tool), then
  queues for a desk again before it continues.
- Provider pool: model requests. The limit starts at `max_concurrency`, halves on
  HTTP 429 (never below one), and grows by one per accepted stream, up to
  `max_concurrency`. A request holds its slot until its stream ends or is dropped.
- Review limit: isolated children (`isolated_workspace: true`) hold one review slot each,
  at most `max_concurrency` at a time. Other children take no review slot.

A child waiting for a desk or review slot is `queued`. Cancelling it, or cancelling
its parent, removes it from the queue at once and it never runs.

A parent that waits on children never holds a desk they need: its desk is released
for the wait and reacquired afterward, so a full budget of waiting parents cannot
starve their children.

Limitation: without systemd, desks are not reserved, so agents run unmetered, the
same as the bare command fallback. A turn cancelled while it waits for its desk back
ends with a cancelled error and runs no further work.

## Limitations

- A command whose handle is dropped without `wait()` records nothing.
- If the wrapper shell itself is killed (for example, when the whole unit is
  OOM-killed), no report is written and the run is not recorded.
- IO fields are `null` when the `io` cgroup controller is not delegated to the
  unit. On the reference host the user manager delegates only `memory` and
  `pids`, so IO is `null` there.
- The bare fallback (no systemd) runs unwrapped and is not recorded.
- A run that ends by error or by the turn limit does not wait for outstanding
  background bash jobs. Their results are dropped, because steering no longer
  reaches a run that has stopped.
- Reservations use the memory ceiling, not learned usage, so a command that uses
  less than its ceiling holds memory it does not use.
