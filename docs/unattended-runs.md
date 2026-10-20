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

## Limitations

- `net:<host>` denials and out-of-workspace file access are hard errors. They
  produce no decision card.
- Redaction in decision summaries is heuristic. Bash command text is not
  redacted.
- Decision records are never pruned.
- The GUI shows a notice for a blocked run but has no decision card. Decisions
  can be resolved through the daemon protocol only; `juto-cli` has no decision
  commands.
- If the continuation run fails to start after approval, the decision stays
  `approved` with no run. The grant is not lost: a new unattended prompt that
  repeats the exact action consumes it.
