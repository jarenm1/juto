# Resource usage records

Juto records what each governed command actually used, so scheduling can later
rely on measured history instead of guesses. This document covers recording and
inspection. It does not cover admission, capacity derivation, or queuing.

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

- `juto-cli resources run -- <argv>`: always, under default limits. Scope
  enforcement stays on, so it fails closed when systemd is unavailable.
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

## Limitations

- A command whose handle is dropped without `wait()` records nothing.
- If the wrapper shell itself is killed (for example, when the whole unit is
  OOM-killed), no report is written and the run is not recorded.
- IO fields are `null` when the `io` cgroup controller is not delegated to the
  unit. On the reference host the user manager delegates only `memory` and
  `pids`, so IO is `null` there.
- The bare fallback (no systemd) runs unwrapped and is not recorded.
