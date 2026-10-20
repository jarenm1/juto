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

## Operator flow

1. Client sends `prompt` with `unattended: true`.
2. Run hits `publish`. The journal records
   `Tool call rejected: blocked: operator decision <id> is required for publish:<bookmark>`.
3. Client receives `DecisionBlocked`, then `RunFinished` with no stop reason.
4. Client calls `ResolveDecision { decision_id, approve: true }`.
5. Daemon persists `approved`, starts the continuation run, and sends
   `DecisionResolved`. The continuation retries `publish` once. The grant is
   consumed, so a second attempt is blocked again.

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
