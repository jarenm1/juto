# Unattended runs and operator decisions

An unattended run is a prompt sent with `unattended: true` (protocol v2
`Request::Prompt`). Nobody is watching it, so it never waits for an approval
prompt. When it reaches an action outside its profile, the run records a
decision, stops, and tells clients. An operator then approves or rejects that
exact action.

## Execution tier

Unattended runs map the configured tier to `workspace`, never `full`:

- `untrusted` stays `untrusted`.
- Every other tier becomes `workspace`.
- Bash always runs sandboxed for `untrusted` and `workspace`. If the sandbox or
  its network helper is unavailable, the command fails closed. It never runs
  unsandboxed.
- Browser and desktop tools are not offered on sandboxed tiers, so an unattended run never has them.

## Decisions

A decision is a blocked action. Its `capability` names what it needs, for
example `publish:feat/x`. Its status moves through:

- `pending`: recorded, waiting for an operator.
- `approved`: an exact grant for (session, tool, argument digest). The grant
  is single-use.
- `consumed`: the approved action was run by the continuation run.
- `rejected`: refused. It can never be approved later.

Records are JSON files in `<data_dir>/decisions/<id>.json`, written atomically,
so a pending decision survives a daemon restart.

A call that needs a decision after its unattended run has ended is refused with an error and records no decision.

Protocol:

- `ListDecisions` returns all decisions, oldest first.
- `ResolveDecision { decision_id, approve }`. Approving starts a continuation
  run that retries the blocked action once. Rejecting starts nothing. The
  session must not be busy, or the request fails.
- Events: `DecisionBlocked` (the run has stopped) and `DecisionResolved`.

A blocked run ends with `stop_reason: null`. Clients see `DecisionBlocked`
before `RunFinished`.

Notification is best effort. On Linux the daemon runs `notify-send` with title
`Juto needs a decision: <capability>` and the decision id as the body. The body
never contains arguments or credentials.

## Publish

`publish` is the only privileged in-process tool for unattended runs. It runs
`jj git push --bookmark <bookmark>` and then `gh pr create`, without a shell.
Credentials stay in the daemon. They are never mounted into a sandbox.

- Granted to the `main` role.
- Not available to `untrusted` runs. Those produce a proposed checkpoint or
  diff, not a published branch.
- Unattended calls need a decision for `publish:<bookmark>`. Attended calls use
  the normal approval prompt.

An unattended grant pins the commit the bookmark pointed at when the decision was recorded. An approved `publish` pushes only that commit. If the bookmark has moved since, the call is refused and nothing is pushed. A bookmark that does not exist is an error, not a decision.

## Operator flow

1. Client sends `prompt` with `unattended: true`.
2. Run hits `publish`. The journal records
   `Tool call rejected: blocked: operator decision <id> is required for publish:<bookmark>`.
3. Client receives `DecisionBlocked`, then `RunFinished` with no stop reason.
4. Client calls `ResolveDecision { decision_id, approve: true }`.
5. Daemon persists `approved`, starts the continuation run, and sends
   `DecisionResolved`. The continuation retries `publish` once. The grant is
   consumed, so a second attempt is blocked again.

## Children

Children an unattended run starts belong to that run. When the root run finishes, every child still working is cancelled and settled before the run returns. Their reports reach the session's main inbox.

## Jobs

A job is a durable background run started with `juto-cli jobs run-now` (or
`JobRegistry` in `juto-runtime::jobs`). It runs in a fresh fork of the
repository, so it never writes into the user's working copy. Its result lands
in the inbox: `jobs drain` returns each finished job once.

A job needs a completion contract. Pass at least one flag:

- `--artifact <path>`: a workspace-relative file that must exist and be
  non-empty.
- `--output-schema <file>`: the final answer must parse as JSON and satisfy the
  schema.
- `--no-change`: the workspace must have no changes when the job stops.

The model stopping is not enough. A job whose contract is not met fails with the
reason, for example `completion contract not met: artifact out/x.txt was not
written`.

`--no-change` measures changes from the tree the job's fork started from, not
from the repository's initial commit. Work committed after the graph was
created does not count. A no-change job that writes a file fails with
`the job changed 1 path(s) but its contract requires no change`.

Budgets and states:

- `--wall-secs` bounds total running time across attempts and decision
  resumptions. When it runs out the job fails with
  `wall-clock budget of <n>s exhausted`.
- `--max-turns` bounds the model turns each attempt may take. When an attempt
  reaches the cap the job fails with `turn budget exhausted: ...`. The
  runtime's own turn limit still applies; the lower of the two wins.
- `--max-spend` bounds the USD the job may spend across all attempts, priced by
  the model catalog. The cap is checked before each model turn of the job's own
  run, so no turn starts once it is reached, and the job fails with
  `spend budget of <n> USD exhausted`. Subagent spend is added when each
  subagent settles, so a subagent still running is not yet counted.
- `--max-attempts` bounds the attempts the job may start, counting the first.
  `jobs retry` starts a new attempt within the budget.
- A runtime that stops while a job is queued or running leaves it
  `interrupted`. Nothing replays on restart. `jobs retry` continues the saved
  session. A tool call that was in flight is recorded as interrupted and is not
  run again.
- A tool call that needs an operator decision moves the job to `blocked`.
  `jobs decide <decision-id> --approve` resumes it. The GUI jobs panel offers
  Approve and Deny for the same decision.

Unattended runs use the `workspace` tier (see Execution tier), so writes happen
only in the job's fork. The `main` role cannot write. A job that must change
files therefore delegates the writes: the main agent starts a `task` child and
calls `wait` for it. If the main agent answers without calling `wait`, the job
ends and the child is cancelled.

## Automations

An automation starts a job on a schedule. Create one with
`juto-cli automations create <prompt> --every-secs <n>`, and pass the same budget
and contract flags as `jobs run-now` (`--wall-secs`, `--max-attempts`,
`--max-turns`, `--max-spend`, `--artifact`, `--output-schema`, `--no-change`).
Each automation stores its prompt, project, model pin (`--model-pin`), priority,
budget, contract, schedule, overlap policy, enabled state, tier (`--tier`), and
permission profile (`--permission-profile`). Its project is the daemon's
workspace. Each occurrence produces the same durable job as `jobs run-now`: same
budget, contract, inbox, and `jobs` commands.

### Tier and permission profile

- Tier: `untrusted` or `workspace`. The default is `workspace`. `attended` and
  `full` are refused at create. A stored tier that is not allowed is also refused
  when an occurrence starts, and that occurrence is recorded as `failed`. The
  rule is `check_unattended_tier` in `crates/juto-runtime/src/tier.rs`, and the
  error names it.
- Effective run tier: `min(stored tier, unattended_tier(configured tier))`. On a
  daemon configured `untrusted`, a `workspace` automation runs `untrusted`. A
  definition can narrow the unattended boundary but cannot widen it.
- Permission profile: `unattended` (the default) or `read-only`. `read-only`
  restricts the run to the read, glob, and grep tools and sets approval to
  read-only. `unattended` is the widest profile that exists, so the create check
  (`check_unattended_profile`) currently admits both values. It is the place where
  a wider profile would be refused.

### Recurrence and time zones

The schedule is a fixed interval in UTC. `--every-secs` takes 1 to 31622400 (366
days). Occurrence `k` (`k >= 1`) falls at `created_ms + k * every_secs * 1000`
milliseconds since the Unix epoch. The anchor is the creation time.

There are no time zones and no cron expressions. No DST rule applies: an interval
of 86400 seconds fires at the same UTC instant every day. A local wall-clock
schedule is not supported. A new schedule kind would need its own spec.

The daemon checks for due occurrences once per second. A job starts within about
one second of its occurrence instant.

### Restart and missed occurrences

On each tick the runtime looks at the occurrences due since `next_instant_ms`. It
starts one job for the latest due occurrence and never replays the others. The
earlier undecided occurrences coalesce into one `missed` event with a count.

- A daemon that was down across several occurrences starts one job at its first
  tick and records the rest as missed. This runs once, not once per missed tick.
- The job starts before the cursor (`next_instant_ms`) is saved. If the daemon
  stops between the two, the restart evaluates the same occurrence again. Its job
  id already exists, so nothing new starts.
- Interrupted jobs are not replayed, as for any job (see Jobs).

### Overlap

A job is open while it is queued, running, or blocked. When an occurrence is due
and the automation's previous job is open:

- `skip` (the default): the occurrence is recorded as `skipped` and the schedule
  moves on.
- `queue`: the cursor holds. When the open job settles, one job starts for the
  latest due occurrence. Earlier due occurrences are recorded as missed. At most
  one occurrence waits.

`run-now` is refused while a job of the automation is open, under either policy.

### Pause, enable, delete, run now

- A paused automation (`enabled: false`) never fires.
- Enabling a paused automation records the paused stretch as one `missed` event
  and moves the cursor past now. It does not start a catch-up job.
- `run-now` starts one job immediately, records a `run-now` event, and does not
  move the schedule.
- Delete removes the definition and its event log (`automations/<id>.json` and
  `automations/<id>.events.jsonl`). The jobs it started are kept in the job store.

### Dedupe key

One occurrence is the pair (automation id, scheduled instant in ms). Its job id is
`auto-<automation id>-<instant ms>`. Starting a second job with that id fails with
`JobError::Exists`, and the runtime records that as already fired. An occurrence
therefore produces at most one job, including across restarts. Jobs started by
`run-now` have no occurrence key.

### Priority

An automation's priority is `scheduled` (the default) or `background`. `attended`
is refused. The job's main run takes that class through the resource broker (see
resources.md, Admission and queueing). Its subagent runs stay `background`.

### Storage and protocol

Each automation is stored under the data directory as `automations/<id>.json`
(definition and cursor, written atomically) and `automations/<id>.events.jsonl`
(event log). Protocol version 5 adds `tier` and `permission` to the automation
spec and record. Both are optional on create and default to `workspace` and
`unattended`. Records written by protocol 4 builds lack these fields and do not
load; re-create those automations. Protocol version 4 adds the automation
requests: `CreateAutomation`, `ListAutomations`, `GetAutomation`,
`AutomationHistory`, `RunAutomationNow`, `SetAutomationEnabled`, and
`DeleteAutomation`. Protocol version 3 has no automation requests.

## GitHub events

A repository watch turns GitHub issue and pull request activity into the same
durable job as `jobs run-now`. The primary trigger is authenticated REST polling:
the daemon makes outbound requests only and opens no listener. A webhook path is
not implemented.

Watches are managed through the protocol (`CreateGithubWatch`,
`ListGithubWatches`, `DeleteGithubWatch`, `ListGithubEvents`, `PollGithub`). There
is no CLI command for them yet. The daemon polls every 60 seconds, and
`PollGithub` polls on demand.

- Endpoints: `GET {base}/repos/{owner}/{repo}/issues` (pull requests excluded)
  and `GET {base}/repos/{owner}/{repo}/pulls`. Both list open items, sorted by
  `updated`, 100 per page, at most 10 pages. A poll that reaches the page cap
  fails rather than truncating. `base` is `https://api.github.com`, overridden by
  `JUTO_GITHUB_API_BASE`. Plain `http` is accepted only for loopback. Redirects
  are not followed.
- Credential: the token is stored under provider `github` in the daemon's
  credential store (`<data_dir>/credentials.json`), set with
  `juto-cli login github --api-key`. It is sent only in an `Authorization: Bearer`
  header. It is never mounted into a sandbox, placed in a prompt, or written to
  the ledger or logs. With no token stored, a poll fails before any request is
  made.
- Allowlist: one watch per repository. Repository names compare
  case-insensitively. An item from another repository is rejected and recorded.
  For a pull request, the repository is its base repository. No job starts.
- Labels and assignees: an item must carry every configured label and at least
  one configured assignee. An empty list imposes no constraint. Only open items
  are considered. An item that fails these rules is skipped and not recorded.
- Checkout: the job runs in the daemon's checkout, so a watch is accepted only
  when that checkout's `origin` remote names the watched repository on
  github.com. This is checked when the watch is created and before each poll.
  A watch that fails the check is reported as a poll error and starts no jobs.
  There is no per-watch checkout path.
- Author gate: an item is accepted only when its `author_association` is
  `OWNER`, `MEMBER`, or `COLLABORATOR`. Any other item is rejected and recorded,
  and starts no job.
- Dedupe key: an issue is `issue:<repo>#<number>:<id>:<sha256>`, where the hash
  covers its title and body. An edit that keeps the title and body (a label, an
  assignee, or a comment) starts no job. A pull request is
  `pull:<repo>#<number>:<head sha>`. The job id is `gh-` plus the first 32 hex
  digits of SHA-256 of the key. A repeat finds the job already stored
  (`JobError::Exists`) and starts nothing, including across restarts. A newer
  revision has a new key and starts the next job.
- Baseline: a watch's first poll records what already exists and starts no jobs.
- Job: `untrusted` tier, `unattended` profile, 1 attempt, 1800 s wall clock, 60
  turns, a 2.0 USD spend cap that includes child runs, and a required
  `PROPOSAL.md` artifact. The item text appears in the prompt only inside
  `<untrusted_github_event>` tags, and the event cannot change the tier or
  profile. Publishing a proposal is not part of the event: it needs an operator
  decision.
- Watch ids: `gh-` plus 16 lowercase hex digits. Deleting a watch and listing
  watches reject any other id, and a watch file whose id does not match its file
  name is refused.
- Storage: the ledger is `<data_dir>/github/events.jsonl`, one line per accepted
  or rejected item. Each watch is `<data_dir>/github/watches/<id>.json`, which
  holds the cursor (`since`) and baseline flag.
- Limitations: pull requests have no server-side `since`, so every poll fetches
  all open pull requests. Duplicates are not written to the ledger. While watches
  exist and no token is stored, the daemon logs an error on every poll.

## Limitations

- `net:<host>` denials and out-of-workspace file access are hard errors. They
  produce no decision card.
- Redaction in decision summaries is heuristic. Bash command text is not
  redacted.
- Decision records are never pruned.
- Unattended writes need the `task`/`wait` delegation above. Whether unattended
  runs should get a writable role is an open product question.
- If the continuation run fails to start after approval, the decision stays
  `approved` with no run. The grant is not lost: a new unattended prompt that
  repeats the exact action consumes it.
