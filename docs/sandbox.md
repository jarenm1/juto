# Execution tiers and sandbox

Every run has an execution tier. The tier decides how the run's shell commands are
supervised and confined. Approval is a separate axis: approval asks a person before a
tool runs; the tier limits what the command can reach once it runs.

## Tiers

| Tier | Bash | Who uses it |
| --- | --- | --- |
| `attended` | Ungoverned by default. With `bashScoped: true`, it runs under resource governance (CPU/memory caps in a systemd scope). Not sandboxed. | Interactive chat the user watches. Default for the top-level run (`executionTier`). |
| `workspace` | Sandboxed systemd service with resource caps (see below). | Default for background and isolated children, and for a child whose `task` entry names no tier. |
| `untrusted` | Same sandbox as `workspace`. | Content-triggered work. Its stricter network and output rules are not implemented yet (see Limits). |
| `full` | Ungoverned and unsandboxed. | Explicit opt-in only, set in `executionTier`. Never offered to a child. |

Order: `untrusted < workspace < attended < full`.

### Child tiers

A child's tier comes from the `task` entry's `tier` field (`untrusted`, `workspace`, or
`attended`). A child can never get a looser tier than its parent:

- A requested tier above the parent's is an error (`tier X exceeds this run's tier Y`).
  It is not clamped.
- `full` is only for the top-level run. A child asking for it is an error.
- With no requested tier, the child gets `workspace` when it is isolated or a focus
  thread, otherwise `attended`, capped at the parent's tier.

## Sandbox (`untrusted` and `workspace` bash)

A sandboxed command runs as a transient systemd **service** (`systemd-run --user
--pipe --collect`). Scopes reject namespace properties, so a service is the only
supervisor that can apply them. Its properties:

- `ProtectSystem=strict`: the system hierarchy is read-only.
- `ProtectHome=tmpfs`: `$HOME` is an empty tmpfs. `~/.ssh`, `~/.gnupg`, `~/.config/gh`,
  `~/.netrc`, `~/.aws`, `~/.docker`, and juto's data directory are absent.
- `PrivateTmp=yes`: `/tmp` is private.
- `BindPaths=` (write): the run's workspace, plus `CARGO_HOME`, `RUSTUP_HOME`, and
  `XDG_CACHE_HOME` (or `~/.cargo`, `~/.rustup`, `~/.cache`) when they exist.
- `BindReadOnlyPaths=` (read): the flake checkout named by `JUTO_FLAKE`, so that
  `nix develop "$JUTO_FLAKE"` works.
- Resource caps: the same `CPUQuota`/`MemoryMax` limits as a governed command.

Grants are bind mounts, not `ReadWritePaths`: `ReadWritePaths` fails with status 226
under a hidden home or a private `/tmp`. Grants that do not exist on the host are
skipped.

The command's working directory is the run's workspace root. A model-chosen `cwd` cannot
widen the writable set, because it must stay inside the root.

### Environment

The command's environment is cleared (`/usr/bin/env -i`). Only these are forwarded from
the runtime's process:

- Names: `PATH`, `HOME`, `USER`, `LOGNAME`, `LANG`, `LANGUAGE`, `TERM`, `TZ`,
  `JUTO_FLAKE`, `IN_NIX_SHELL`, `NIX_PATH`, `NIX_PROFILES`, `NIX_USER_PROFILE_DIR`,
  `NIX_SSL_CERT_FILE`, `SSL_CERT_FILE`, `LOCALE_ARCHIVE`, `PKG_CONFIG_PATH`,
  `LD_LIBRARY_PATH`, `LIBCLANG_PATH`, `CC`, `CXX`, `AR`, `CFLAGS`, `CXXFLAGS`,
  `CPPFLAGS`, `LDFLAGS`, `XDG_CACHE_HOME`.
- Prefixes: `LC_`, `CARGO_`, `RUST_`, `RUSTUP_`, `CMAKE_`, excluding any name containing
  `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `AUTH`, `COOKIE`, or `API_KEY`.

Variables set explicitly on the command pass through. Everything else, including
`SSH_AUTH_SOCK`, `DBUS_SESSION_BUS_ADDRESS`, and `XDG_RUNTIME_DIR`, is dropped.

## Fail closed

Governed commands require `systemd-run`. If it is missing, the command returns
`systemd-run is unavailable` and does not run. There is no bare-spawn fallback. The
`AGENT_SANDBOX` environment variable and the `enforce_scope` option are removed. A
sandboxed tier cannot run unsandboxed.

## Verified on this host

- `cargo build` and `cargo test` of a small crate run inside the sandbox.
- `nix develop "$JUTO_FLAKE" -c ...` runs inside the sandbox with the flake checkout
  bound read-only. It prints a non-fatal warning that it cannot read `HEAD`.
- From inside the sandbox, `~/.ssh`, `~/.config/gh`, and juto's data directory are not
  visible, `SSH_AUTH_SOCK` is unset, and `/run/user` is empty.

## File tools

`read`, `glob`, and `grep` run in-process. When a run has a sandbox, each path is
canonicalized (symlinks followed) and must fall under the run's workspace root or a
granted root (`read_paths` or `write_paths`). Anything else is refused with
`file access denied: outside workspace and granted roots`. Credential paths are refused
with `file access denied: credential path`, even when a grant contains them.

Runs without a sandbox (`attended`) keep the existing behavior: these tools read any
path the process can read.

`write` and `edit` always require the target to lie inside the workspace root. Before
touching the file they refuse symlinked components, `.git`/`.jj`/native metadata paths,
and paths outside a partial fork's sparse spec (`vcs::mutate`).

## Limits

This is containment against accidents and careless commands, not a guarantee against
hostile code.

- **Network is not restricted yet.** Sandboxed commands can reach the network. A domain
  allowlist and a `PrivateNetwork` proxy are planned for the next change.
- **The nix daemon is reachable.** `/nix/var/nix/daemon-socket/socket` is available, so
  builds and substitutes run in the daemon, outside the sandbox.
- **`untrusted` is not yet distinct from `workspace`.** It gets the same filesystem
  sandbox. Package-mirror-only network access and "output only a proposed diff" are not
  implemented.
- **No separate unattended mode.** A top-level run takes its tier from `executionTier`.
  Setting `full` there is an explicit opt-in, and nothing stops an unattended run from using it.
