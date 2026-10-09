# OMP runtime capability inventory — port basis for Juto

This document inventories the runtime capabilities of
[oh-my-pi](https://github.com/can1357/oh-my-pi) ("omp") for the Rust/GPUI port.
Source paths below are relative to the pinned revision; the linked source index
provides entry points for each subsystem. Specialized subsystems surveyed only
at the capability level are identified in the limitations.

## Source revision and method

- Repository: `github.com/can1357/oh-my-pi`
- Pinned revision: **`579da1d661c5cb8d43bc2ddd429ab72e67165ad8`** (main, 2026-10-09)
- Scoped source: `packages/agent`, `packages/ai`, `packages/coding-agent`,
  `packages/catalog`, `packages/natives`, `docs/`, plus `crates/`, `packages/tui`,
  `packages/utils`, `packages/wire`, `packages/omptype`, `sdk/` where they define the
  runtime boundary.
- Version at pin: `@oh-my-pi/*` **18.8.6** across the workspace
  (`packages/*/package.json`).
- Counts cited here were verified against source at the pin (e.g. bundled catalog =
  75 provider keys / 5,671 model rows in `packages/catalog/src/models.json`; 30 builtin
  tool names in `packages/coding-agent/src/tools/builtin-names.ts`). The README's
  "60+ providers / 31 built-in tools / ~80k lines of Rust" badges are approximate and are
  **not** used as evidence anywhere below.

### Pinned primary-source index

- [License](https://github.com/can1357/oh-my-pi/blob/579da1d661c5cb8d43bc2ddd429ab72e67165ad8/LICENSE)
  and [third-party notices](https://github.com/can1357/oh-my-pi/blob/579da1d661c5cb8d43bc2ddd429ab72e67165ad8/THIRD-PARTY-NOTICES.txt).
- Provider transport, streaming, authentication, usage:
  [AI source](https://github.com/can1357/oh-my-pi/tree/579da1d661c5cb8d43bc2ddd429ab72e67165ad8/packages/ai/src).
- Catalog, compatibility rules, provider IDs:
  [catalog source](https://github.com/can1357/oh-my-pi/tree/579da1d661c5cb8d43bc2ddd429ab72e67165ad8/packages/catalog/src).
- Agent execution, events, context, compaction:
  [agent source](https://github.com/can1357/oh-my-pi/tree/579da1d661c5cb8d43bc2ddd429ab72e67165ad8/packages/agent/src).
- Session orchestration, persistence, model selection, recovery, approvals:
  [coding-agent source](https://github.com/can1357/oh-my-pi/tree/579da1d661c5cb8d43bc2ddd429ab72e67165ad8/packages/coding-agent/src).
- Subagent scheduling and lifecycle:
  [task source](https://github.com/can1357/oh-my-pi/tree/579da1d661c5cb8d43bc2ddd429ab72e67165ad8/packages/coding-agent/src/task).
- Tool registry and implementations:
  [tools source](https://github.com/can1357/oh-my-pi/tree/579da1d661c5cb8d43bc2ddd429ab72e67165ad8/packages/coding-agent/src/tools),
  [built-in names](https://github.com/can1357/oh-my-pi/blob/579da1d661c5cb8d43bc2ddd429ab72e67165ad8/packages/coding-agent/src/tools/builtin-names.ts).
- MCP:
  [client source](https://github.com/can1357/oh-my-pi/tree/579da1d661c5cb8d43bc2ddd429ab72e67165ad8/packages/coding-agent/src/mcp).
- Extension and discovery contracts:
  [official extension documentation](https://github.com/can1357/oh-my-pi/blob/579da1d661c5cb8d43bc2ddd429ab72e67165ad8/docs/extensions.md).
- Session journal and external protocol:
  [session documentation](https://github.com/can1357/oh-my-pi/blob/579da1d661c5cb8d43bc2ddd429ab72e67165ad8/docs/session.md),
  [RPC documentation](https://github.com/can1357/oh-my-pi/blob/579da1d661c5cb8d43bc2ddd429ab72e67165ad8/docs/rpc.md).
- Remaining configuration and subsystem docs:
  [official docs](https://github.com/can1357/oh-my-pi/tree/579da1d661c5cb8d43bc2ddd429ab72e67165ad8/docs).
- Native Rust reuse:
  [Cargo workspace](https://github.com/can1357/oh-my-pi/blob/579da1d661c5cb8d43bc2ddd429ab72e67165ad8/Cargo.toml),
  [crates](https://github.com/can1357/oh-my-pi/tree/579da1d661c5cb8d43bc2ddd429ab72e67165ad8/crates).
- Terminal renderer and native-surface vocabulary:
  [TUI source](https://github.com/can1357/oh-my-pi/tree/579da1d661c5cb8d43bc2ddd429ab72e67165ad8/packages/tui/src).

## Lineage and licensing

- omp is a **fork of [pi-mono](https://github.com/badlogic/pi-mono)** (Pi, by Mario
  Zechner), extended by Stencil Labs (`README.md`).
- License: **MIT**. `LICENSE` carries three copyright lines: `2025 Mario Zechner`,
  `2025-2026 Can Bölük`, `2026 Stencil Labs, Inc.` Any Juto code copied or closely
  derived from omp/pi-mono must retain the MIT license text and those notices.
- `THIRD-PARTY-NOTICES.txt` (release-artifact aggregate) flags attribution traps for
  porting:
  - `crates/vendor/brush-core`, `crates/vendor/brush-parser` — vendored, locally
    patched brush shell (MIT, © reuben olinsky); selected via `[patch.crates-io]` in the
    root `Cargo.toml`.
  - `crates/pi-builtins` — in-process ports of **uutils coreutils/findutils/sed** and
    **jaq**; the crate ships its own `LICENSE`/`LICENSE-MIT` that must be preserved.
  - `crates/pi-natives/data/LICENSE.ctok` (Claude tokenizer data), syntect syntax/theme
    packs, vendored `napi`/`cfg_aliases`/`tree-sitter-go` — own licenses each.
  - `packages/omptype/test/ark/LICENSE` (arktype test data).
- If Juto links or copies any `crates/pi-*` code or data, the corresponding LICENSE files
  and `THIRD-PARTY-NOTICES` must ship with Juto artifacts.

## Implementation languages and runtimes

| Layer | Language / runtime | Notes |
|---|---|---|
| Provider client, agent loop, catalog, coding-agent runtime | TypeScript on **Bun ≥ 1.3.14** | Pervasive `Bun.*`/`bun:*` APIs: `bun:sqlite` (auth store, model cache, memory, stats), `Bun.serve`, `Bun.spawn`, `Bun.$`, `Bun.file`, `import .md with {type:"text"}` (`bunfig.toml` loaders). |
| Native core | **Rust**, 12 workspace crates + `crates/vendor` | Single N-API `cdylib` (`crates/pi-natives`), consumed in-process by Bun. Root `Cargo.toml`, edition 2024, `panic="unwind"` release profile so panics become rejected JS promises. |
| Bench/automation extras | Python (`python/robomp`), Go | Not part of the session runtime. |
| Protocol SDKs | Rust, Go, Python | `sdk/{rust,go,python}/omp-rpc` — clients for the `omp --mode rpc` JSONL protocol. `sdk/rust/omp-rpc` is a standalone MIT crate. |
| Build | bun workspaces + Cargo workspace + Bazel (release addon builds) | `packages/natives` has **no `src/`**: it is the loader/packaging shell for the `.node` addon (`native/loader-state.js`, `scripts/gen-npm-packages.ts`). |

**Port consequence:** the Rust port does not start from zero. Shell, editing, search,
AST, diff, tokenization, VCS, text layout, and desktop automation are *already Rust* —
the TS side is the agent orchestration, provider transport, session journal, tool
policy, and all extensibility. Bun-specific seams that must be replaced: `bun:sqlite`
(→ `rusqlite`), `Bun.spawn`/`$` (→ `tokio::process` / `std::process`), `Bun.serve`
(→ `axum`/equivalent), `.md`-as-text imports (→ `include_str!`), `Bun.env`/`bun:ffi`.

## Implemented status (first step — foundation only)

What exists in Juto today:

- Nix flake + Cargo workspace skeleton with fast-build-oriented configuration.
- `.envrc` dev environment; `jj` version control; adapted `AGENTS.md`; inherited
  scripts/agent-scope resource wrapper.
- A minimal **GPUI 0.2.2 application shell** targeting X11 + Wayland — a launchable
  native window, nothing more.

What does **not** exist yet:

- **No OMP runtime behavior is ported.** There is no runtime crate yet; the module
  layout in §"Rust module seam proposal" is a *prospective* sketch, not existing code.
- No provider clients, no auth, no agent loop, no tools, no sessions, no extensions.
- GPUI currently stands in for `packages/tui` only; every runtime subsystem below is
  unimplemented in Juto.

Full parity with omp is a long-horizon program (§"Ordered implementation plan"), not the
initial milestone. The first shippable slice should be judged against its own acceptance
criteria, not against this inventory.

---

# Capability inventory

## 1. Provider layer (`packages/ai` — `@oh-my-pi/pi-ai`)

### 1.1 Provider/API split and registration

Two-level dispatch: `model.provider` (account/backend namespace) selects credentials +
provider hooks; `model.api` (wire protocol) selects the transport
(`packages/catalog/src/types.ts:9-26`, `packages/ai/src/api-registry.ts:19-36`).

- **Built-in wire APIs (`KnownApi`)**: `openai-completions`, `openai-responses`,
  `openrouter` (pseudo-API routed to Responses unless `PI_OPENROUTER_RESPONSES=0`),
  `openai-codex-responses`, `azure-openai-responses`, `anthropic-messages`,
  `bedrock-converse-stream`, `google-generative-ai`, `google-gemini-cli`,
  `google-vertex`, `ollama-chat`, `cursor-agent`, `factory-droid-agent`,
  `gitlab-duo-agent`, `devin-agent`, `apple-foundation-models`.
- **Built-in provider ids (`KnownProvider`)**: ~85 generated values
  (`packages/catalog/src/compat/provider-ids.ts`) — first-party (anthropic, openai,
  openai-codex, google, google-gemini-cli, google-vertex, amazon-bedrock, bedrock-mantle,
  azure, deepseek, mistral, groq, cerebras, xai/xai-oauth, moonshot/kimi-code,
  zhipu-coding-plan, minimax*, qianfan, stepfun, zai, nvidia, together, fireworks,
  deepinfra, baseten, coreweave, novita, huggingface, vllm, openrouter,
  vercel-ai-gateway, cloudflare-ai-gateway, litellm, ollama/ollama-cloud, lm-studio,
  apple, local, github-copilot, cursor, devin, factory-droid, gitlab-duo(-agent),
  google-antigravity, cline-pass, kilo, opencode-go/zen, alibaba-*-plan, xiaomi
  token-plans, siliconflow(-cn), gmi-cloud, charm-hyper, muse-code, commandcode,
  helmcode, wafer-serverless, singularityapi-dev/tech, umans, venice, sakana, snowflake,
  synthetic, typesafe, yolo-auto, zenmux, abliteration, aiand, aimlapi, nanogpt,
  qwen-portal, web) — enumerated in `provider-ids.ts`, not a marketing figure.
- Registration is **data-driven**: `PROVIDER_REGISTRY` maps compiled KDL auth rules
  (`packages/catalog/src/compat/rules/auth/*.kdl` → `rules.json` via `bun run
  gen:compat`) through `buildProviderDefinition()` plus a `TRANSPORTS` table
  (`packages/ai/src/registry/registry.ts:17-48`, `registry/build.ts:52-74`). Compile-time
  check forces every catalog provider to have an auth policy. A port should treat the
  **KDL rules as the schema**, not the generated TS unions.
- Extension registration: `registerCustomApi()` adds/override wire APIs, checked before
  built-in dispatch (`api-registry.ts:74-111`, `stream.ts:931`); `registerProvider` in
  the extension API can even shadow built-in provider ids (docs/extensions.md).
- Non-HTTP transports worth noting: **Codex SSE + WebSocket** with fallback
  (`openai-codex-transport.ts`), **Cursor** HTTP/2 Connect-RPC + bidi protobuf,
  **Devin** protobuf Cascade protocol (`devin/proto/` vendors ~100 `.proto` files),
  **GitLab Duo Workflow** WebSocket action bridge, **Apple Foundation Models** native
  bridge (no HTTP), **`pi-native`** gateway stream relay
  (`pi-native-{client,server}.ts`).

Public entry points: `stream()`, `streamSimple()`, `completeSimple()`
(`packages/ai/src/stream.ts`).

### 1.2 Auth: API keys, OAuth, token refresh, storage

Three layers:

1. **`src/auth/` — local credential store and policy.** `AuthStorage` facade over
   `AuthCredentialStore` composing ~19 modules (`auth-storage.ts`): `KeyCascade`/
   `KeyOverrides` precedence (runtime override > config `apiKey` > OAuth > stored key >
   env), `CredentialPool` + `CredentialSelector`/`rank.ts` (usage-aware multi-account
   ranking), `OAuthAccounts`, `OAuthRefresher` (**60s expiry skew, durable cross-process
   refresh leases, 5-min re-mint cooldown**), `RateLimits`/`CredentialBlocks`
   (persisted per-credential blocks with `chat|spark|shared` scopes), `SessionAffinity`
   (per-session account pins), `ResetCredits` (banked usage-reset redemption),
   `CredentialHealth`, `UsageService`/`UsageCache`.
   **Storage is SQLite, not an OS keyring**: `agent.db` at `~/.omp/agent/agent.db`
   (`auth/sqlite-credential-store.ts`), tables `auth_credentials`,
   `auth_credential_blocks`, `auth_credential_refresh_leases`, `usage_history`,
   `clients`, `client_usage`; soft-delete tombstones, `identity_key` per credential.
2. **`src/auth-broker/` — credential vault service** (`docs/auth-broker-gateway.md`).
   HTTP server that is the sole writer of OAuth refresh tokens; clients hold a
   `RemoteAuthCredentialStore` receiving redacted snapshots (refresh tokens replaced by
   `REMOTE_REFRESH_SENTINEL`) over SSE/long-poll, delegating refreshes via
   `POST /v1/credential/:id/refresh`. Encrypted local snapshot cache
   (AES-256-GCM keyed on broker token) at `~/.omp/cache/auth-broker-snapshot.enc`.
   Bearer token from `<config-dir>/auth-broker.token`; env `OMP_AUTH_BROKER_URL/TOKEN`.
   Account pools route by `identityKey` — routing, not authorization.
3. **`src/auth-gateway/` — HTTP forward proxy.** Clients speak foreign wire formats;
   the gateway resolves broker-backed credentials and calls `streamSimple()`. Routes:
   `/v1/chat/completions`, `/v1/messages`, `/v1/responses`, `/v1/pi/stream` (canonical
   `AssistantMessageEvent` SSE), `/v1/systemone`, `/v1/images/generations|edits`,
   `/v1/audio/{speech,transcriptions}`, `/v1/embeddings`, `/v1/rerank`,
   `/v1/videos[...]`, `/v1/usage`, `/v1/models`, `/v1/credentials/check`. `stdio` mode =
   JSON-lines variant for a parent process (`omp auth-gateway serve|stdio`).

**OAuth flows** (declarative engines driven by KDL `login` nodes,
`registry/engine/*`):

- `oauth-code` (auth code + optional PKCE; loopback callback server
  `NativeOAuthCallback` in pi-natives, or paste-code): anthropic, openai-codex,
  google-gemini-cli, google-antigravity, gitlab-duo(+agent), devin, openrouter,
  stencil, zai-coding-plan.
- `device-code` (RFC 8628): factory-droid, kimi-code, muse-code, xai-oauth.
- `custom`: github-copilot (token exchange), cursor, perplexity, snowflake,
  alibaba plans, cloudflare-ai-gateway, kilo, xiaomi, openai-codex-device.
- `api-key`: prompt + per-provider validation.
- Refresh: per-provider `refresh` KDL nodes + `CUSTOM_REFRESH_HOOKS`
  (copilot/cursor/snowflake), `registry/engine/refresh.ts`.

**Credential sources**: env vars per provider (`docs/providers.md` table),
`models.yml` `apiKey` (incl. **`!command` secret execution** with 10s timeout, process
cache, 30s failure backoff), stored auth, `--api-key` override. `.env` discovery:
cwd → agent-dir → config-root → `~/.env`.

**Auth retry a/b/c**: `ApiKeyResolver` — initial key → refresh same account →
`lastChance` rotate sibling (`auth-retry.ts`); `streamSimple` buffers events until
replay-unsafe; `withAuth()` wraps non-stream callers.

### 1.3 Streaming normalization

Every provider emits unified `AssistantMessageEvent`s: `start` → block triplets
`text|thinking|toolcall` × `_start/_delta/_end`, `image_end` → `done`
(`stop|length|toolUse`) or `error` (`aborted|error`) (`types.ts`, contract in
`docs/provider-streaming-internals.md`). `AssistantMessageEventStream` is an async
iterator with `result()/fail()/push()` and `trackLocalWork()` so server-requested local
work doesn't trip the stall watchdog.

- Watchdogs: `streamFirstEventTimeoutMs` (100s shared / 300s idle-iter) and
  `streamIdleTimeoutMs` (120s), per-provider ownership flags, env `PI_STREAM_*`,
  `PI_OPENAI_STREAM_*` (`register-builtins.ts`, `types.ts`).
- Tool-arg streaming: `kStreamingPartialJson` buffer + throttled partial-JSON parse with
  repair fallback (`utils/tool-call-arguments.ts`).
- Thinking: `ThinkingContent`/`RedactedThinkingContent` with signatures; Anthropic
  signature/byte-identical replay policy; demotion of foreign thinking to text
  (`dialect/demotion.ts`, `transform-messages.ts`); leaked-thinking healing for
  unofficial endpoints (`utils/leaked-thinking-stream.ts`); thinking-loop guard (≤3
  resamples).
- Glyph codec for `model.requiresGlyphTokenization` (`utils/glyph-codec.ts`).

### 1.4 Message model and multimodal

- Roles: `user`, `developer`, `assistant`, `toolResult`; `Context.systemPrompt:
  string[]`, `inactiveTools` for Anthropic tool_removal (`types.ts`).
- Content: user/toolResult = `(TextContent | ImageContent)[]` — **no document/PDF/
  audio/video input blocks**; images carry base64 `data`+`mimeType`, `detail` hint,
  `providerFile` (uploaded-file refs: OpenAI/Anthropic/Google), `url` mirror.
  Assistant content adds thinking/redacted/Anthropic-fallback/server-tool/ToolCall.
- `ToolCall {id, name, arguments, providerMetadata}`; 64-char id cap, Responses
  `call_id|item_id` composites, per-origin canonicalization (`transform-messages.ts`).
- `transformMessages()` — shared cross-provider history transform: malformed tool-call
  sanitization, id normalization/dedup, thinking-signature strip per endpoint class,
  **credential redaction** (`configureCredentialRedaction`, `SENSITIVE_TOKEN_RE`).
- **Dialects** — in-band text tool-calling for models without native tools: `glm`,
  `hermes`, `kimi`, `xml`, `anthropic`, `deepseek`, `minimax`, `harmony`, `qwen3`,
  `gemini`, `gemma` (`catalog/src/identity/dialect.ts`, `ai/src/dialect/*`,
  `docs/toolconv/*.md`). `PI_DIALECT` env override. Each dialect = prompt + scanner +
  renderers; owned-stream projector handles fabricated-result cutoffs.

### 1.5 Errors and retries

- Error taxonomy: `errorId` bitfield — ThinkingLoop, Transient, Timeout, UsageLimit,
  StaleResponsesItem, MalformedFunctionCall, ProviderFinishError, EmptyResponse,
  ContentBlocked, AccountPolicy, ContextOverflow, AuthFailed, SilentAbort,
  UserInterrupt, Abort, Grammar, FastModeUnsupported, OAuthExpiry, PayloadRejected
  (`error/flags.ts`); exception classes per provider family (`error/classes.ts`).
- Rate-limit classification: `RateLimitReason` with per-reason backoffs (30min / 30s /
  5s / 45s±15s / 20s) and a large message-pattern bank **including Simplified-Chinese
  quota/throttle phrasing** (`error/rate-limit.ts`); `retry-after` /
  `x-ratelimit-reset` extraction (`utils/retry-after.ts`).
- Retry layers (stacked): in-provider replay-safe retries (empty-completion, strict-tool
  fallback); stream-level auth a/b/c rotation; `completeOneshot` oneshot retry (≤3,
  500ms→30s, honors retry-after); durable credential blocks + sibling rotation;
  session-layer `TurnRecovery` (coding-agent, §3.4); per-model `retry.fallbackChains`;
  Anthropic server-side `fallbacks` beta.
- Concurrency: per-provider in-flight caps via **filesystem lock dirs** (cross-process)
  (`stream.ts`, `types.ts`).

### 1.6 Usage and cost tracking

- `Usage`: input/output/cacheRead/cacheWrite/total/context/orchestration tokens,
  reasoningTokens, cache-TTL breakdown (5m/1h), server tool counts, credits
  `{cost,committedCost,acuCost}`, computed `cost.*` (`catalog/src/types.ts`).
- `calculateCost(model, usage, timestamp)` from catalog pricing with long-context
  tiers, **time-based peak/off-peak schedules and effective-from rate cards**
  (DeepSeek documented in `docs/models.md`); provider-reported costs (OpenRouter,
  TypeSafe) override estimates.
- Provider quota reporting: normalized `UsageReport`/`UsageLimit`/`UsageWindow` for 22
  providers (`usage/registry.ts`); 5-min TTL ±jitter cache, 15s single-flight;
  `omp usage` CLI.

### 1.7 Auxiliary APIs

| API | Entry | Wire apis |
|---|---|---|
| Embeddings | `embed()` | `openai-embeddings` (OpenAI, OpenRouter) |
| Rerank | `rerank()` | `openrouter-rerank` |
| Transcription/STT | `transcribeAudio()` | `openai-transcriptions` |
| TTS | `synthesizeSpeech()` | `xai-tts`, `openai-speech` |
| Image generation | `generateImage()` | `openai-images`, `openrouter-images`, `google-generative-ai`, `google-gemini-cli`, hosted via `openai-responses`/`openai-codex-responses` |
| Video | `submitVideo/pollVideo/downloadVideo` | `openrouter-video` |
| Judgment | `Judge` iface; `TypeSafeJudge`, `TextJudge` | `typesafe` (`/v1/systemone`), `openrouter-decisions` (`/decisions`), any chat model |

### 1.8 Tool declaration and schema normalization

`Tool{name, description, parameters (omptype Type | JSON Schema), strict, deferLoading,
customFormat{lark|regex}, customWireName, native:{computer}, examples[]}`
(`types.ts`). Per-endpoint schema sanitizers: `normalizeSchemaFor{Google,CCA,MCP,
Moonshot}`, `sanitizeSchemaFor{OpenAIResponses,StrictMode,Ollama,Grammar,Cursor}`,
`enforceStrictSchema`, `toFoundationModelsSchema` (`utils/schema/`,
`docs/ai-schema-normalize.md`); tool-choice mappers per family (`stream.ts`);
`PI_NO_STRICT` escape hatch; Anthropic `strict` field allowlist with auto-retry on
grammar-reject (`disableStrictTools` opt-out for proxies).

## 2. Agent core (`packages/agent` — `@oh-my-pi/pi-agent-core`)

### 2.1 Agent loop

`agent-loop.ts` (~4k lines) + `agent.ts` (event-sourced `Agent` wrapper, ~60
config fields):

- Entries: `agentLoop(prompts, context, config, signal, streamFn)`,
  `agentLoopContinue`, `*Detailed` variants with telemetry. Event stream:
  `agent_start/end` (end carries telemetry+coverage), `turn_start/end`,
  `message_start/update/end`, `tool_execution_start/update/end`,
  `tool_stream_update`.
- **No max-turns cap**: the loop runs until tools stop, deadline, abort, or a terminal
  yield. Bounded continuation classes only: `pause_turn` re-sample ≤8, DeepSeek DSML
  leak nudges ≤2, soft-tool-requirement escalations ≤3, Harmony leak retries ≤2+2.
- Per iteration: yield check → pause gate → fold pending/aside messages →
  `syncContextBeforeModelCall` → resolve tool-choice directive (hard `ToolChoice` or
  **soft requirement** `{toolName, satisfies, reminder}` that skips non-compliant calls
  and escalates to `forcedToolChoice`) → `beforeModelCall` gate (can halt before the
  request) → stream.
- Stop reasons: `error`/`aborted` → placeholder tool results + end; `toolUse`/`stop`
  with tool calls → dispatch; `length` → calls abandoned w/ skipped results;
  `pause_turn` stopDetails → re-sample.
- Tool execution: per-tool `concurrency` `"shared"|"exclusive"|fn`; per-tool
  `interruptible`; cooperative `steeringSignal`; ordered result emission regardless of
  completion order; `beforeToolCall` hook can block/replace args/inject
  `additionalContext`; `afterToolCall` post-hook; intent `i` field extraction;
  `resolveFallbackTool` for unadvertised names; `TERMINAL_TOOL_RESULT_ABORT_REASON`
  for subagent `yield` (commits batch, stops loop gracefully).
- Live mid-run steering: `LiveSteeringChannel` claim/accept/reject delivered
  provider-side (Codex `response.steer`, `live-steering.ts` +
  `openai-codex-responses.ts`) plus boundary-injected `steer()`/`followUp()` queues
  (`"all"|"one-at-a-time"` dequeue modes) and interruptible-tool aborts.
- Leak mitigations owned by the loop: GPT-5 **Harmony protocol leak** (detect → recover
  tool call → truncate+resume → escalate, +0.05 temperature on retry,
  `ERRATA-GPT5-HARMONY.md`), DeepSeek DSML markup strip+nudge.
- Global `agentPauseGate` (`pause.ts`): parks all loops in the process at safe
  boundaries.
- `proxy.ts` `streamProxy`: streamFn over a server relay (partial-message rebuild).
- `speculative-execution.ts`: opt-in coordinator admitting tool calls at
  `toolcall_start`/`_end`, dependency graph, reconcile against finalized args
  (`canonicalJson`), commit/discard, `maxInFlight=2`, nested children.

### 2.2 Context/token machinery

- `append-only-context.ts`: `StablePrefix` (systemPrompt+tools fingerprint) +
  `AppendOnlyLog` + digest-trimmed sync — maximizes provider **prompt-cache hits**
  on in-place history rewrites.
- `tokenizer.ts`: catalog-driven local tokenizers resolved through **pi-natives Rust
  BPE** (claude-v3/v47/v5/v5-sonnet, qwen3, deepseek-v3, kimi-k2, glm5); modes
  strict|approximate|upperbound; `PI_TOKENIZER_ACCURATE`; fragment LRU cache.
- `image-tokens.ts`: image cost estimation per catalog `image-tokenization` rule.
- `output-budget.ts`: shrink output cap so prompt+output ≤ window, anchored on last
  provider usage.
- `sent-tool-definitions.ts`: byte-identical re-declare of withdrawn tools for
  Anthropic tool_removal.
- `replay-policy.ts`: drops provider refusal/sensitive messages from replay.
- `run-collector.ts`: per-run summary (stop reasons, tool counters, usage, cost) +
  coverage (tools available/invoked/unused, models/providers used).
- `telemetry.ts` (~2.2k lines): OTEL GenAI semconv spans `invoke_agent`→`chat`/
  `execute_tool`/`handoff`/`judgment`; content-capture levels; gateway detection.

### 2.3 Compaction engine (`src/compaction/`)

Pure functions over a `SessionEntry[]` **tree** (id/parentId — the journal is
branchable by construction). Orchestration lives in coding-agent (§3.3).

- Settings: `enabled`, strategy `context-full|handoff|shake|snapcompact|off`,
  thresholds (percent/tokens), `midTurnEnabled`, `reserveTokens` (16k default, ≥15%
  window), `keepRecentTokens` (20k), `autoContinue`, remote-compaction flags
  (`compaction.ts`).
- Triggers (`docs/compaction.md`): manual `/compact`, overflow recovery,
  incomplete-output, post-turn threshold, mid-turn (forced on for subagents), idle.
- Ordered lanes in `compact()`: **V2 streaming Responses compaction** → OpenAI native
  `/responses/compact` → Anthropic beta `compact-2026-09-04` → local `generateSummary`
  (provider-aware error ordering: protocol failures outrank auth failures for
  cross-provider fallback).
- `prepareCompaction`/`findCutPoint`: never cuts at toolResult; honors
  `reset_boundary`, reusable native payloads, adaptive `keepRecentTokens`, turn-prefix
  split summarization.
- Handoff: `generateHandoff`/`generateHandoffFromContext` → handoff document committed
  as a `compaction` entry (`docs/handoff-generation-pipeline.md`).
- `shake.ts`: mechanical elision of tool results/fences to placeholders (default /
  aggressive / 0-protect rescue presets). `pruning.ts`: superseded/useless result
  blanking only when prompt-cache-safe. `branch-summarization.ts`: old-leaf→ancestor
  summaries for tree navigation/subagents.
- Thinking interplay: `resolveCompactionEffort`, per-model clamp, `ThinkingLevel`
  adds `Inherit|Off` over catalog `Effort`.
- Snapcompact: `@oh-my-pi/snapcompact` — deterministic serialization + **PNG raster of
  discarded history** for vision-model compaction (native `renderSnapcompactPng`).

### 2.4 Hooks seam

`Agent` exposes ~30 host hooks — `getModel`, `getApiKey`, `getReasoning`,
`getServiceTier`, `getDialect`, `getToolChoice`, `getToolContext`,
`beforeToolCall`/`afterToolCall`, `transformContext`, `transformProviderContext`,
`transformAssistantMessage`, `convertToLlm`, `beforeModelCall`,
`syncContextBeforeModelCall`, `onPayload`/`onResponse`/`onSseEvent`,
`onAssistantMessageEvent`, `onTurnEnd`, `onBeforeYield`,
`prepareQueuedMessages`, steering/aside/background peek hooks, Cursor exec handlers
(`agent.ts`, `types.ts`). **This is the port's loop contract.** Notably, system prompt,
tool set, approvals, retry orchestration, and magic keywords are all *outside*
agent-core.

## 3. Coding agent (`packages/coding-agent` — `@oh-my-pi/pi-coding-agent`)

The CLI+runtime. `agent-session.ts` (~13.4k lines) is the god-object where sessions,
steering, advisors, TTSR, async jobs, memory, vibe/goal/plan state, approvals, and
provider glue converge — the largest single porting seam.

### 3.1 Sessions: storage, resume, fork, tree

- **JSONL session files** at `~/.omp/agent/sessions/<encoded-cwd>/<ts>_<id>.jsonl`
  (`docs/session.md`). Format: fixed-width **256-byte title slot** first line (mutated
  in place so listing avoids full scans), `SessionHeader` (`version:3`, UUIDv7 id, cwd,
  title, `additionalDirectories`, `parentSession` (opaque lineage string),
  `previousSessionFiles`, `providerPromptCacheKey`), then `SessionEntry` records.
- Entry types: `message`, `model_usage`, `thinking_level_change`, `model_change`,
  `service_tier_change`, `compaction`, `branch_summary`, `reset_boundary`, `custom`,
  `custom_message`, `label`, `title_change`, `ttsr_injection`, `credential_pin`,
  `session_init`, `mode_change` — all `id`/`parentId` → **in-file tree**.
- `session_init` is the cold subagent-revival contract (systemPrompt, task, tools,
  outputSchema, spawns, readOnly, advisor, modelRole, workPoolYieldItems, isolated).
- Subagent transcripts nest beside the session file (`<session>/<AgentId>.jsonl`,
  recursively; `__advisor*.jsonl`; tombstone sidecars).
- Blobs: images/large strings externalized to content-addressed
  `~/.omp/agent/blobs/<sha256>` (`blob:sha256:` refs, ≥1KiB base64 threshold; 500K-char
  generic truncation).
- **Not session storage**: `agent.db` (SQLite: settings, model_usage, model_perf,
  auth credentials); `history.db` (prompt history FTS + session_titles/recaps index).
- `SessionStorage` abstraction: `FileSessionStorage`, `MemorySessionStorage`,
  `IndexedSessionStorage` over a backend → `SqlSessionStorage` (postgres/mysql/sqlite
  via `Bun.SQL`), `RedisSessionStorage` (`bun:redis`). Write durability: append-only,
  optimistic `expectedSize` CAS + commit guard, native `FileLock`, `.bak` orphan
  repair.
- `SessionManager` (~4.4k lines): create/open/fork/relocate/continueRecent/inMemory/
  list variants; `fork()` rewrites header (new id + `parentSession`, inherited
  `providerPromptCacheKey`); `createBranchedSession`, `branch`/`resetLeaf`/
  `branchWithSummary`; `continueRecent` uses terminal-scoped breadcrumbs
  (`~/.omp/agent/terminal-sessions/`, keyed by TTY/tmux/zellij/kitty/wezterm ids).
- `AgentSession`: `switchSession` (12-step rollback-protected), `newSession`, `fork`,
  `branch`, `navigateTree` (in-file leaf move + optional branch summary), `freshSession`
  (provider-state reset), `resetSessionContext` (`/clear` → `reset_boundary`),
  `exportToHtml`, `handoff`, `compact`/`abortCompaction`.
- Import/export/share: `--from-claude`/`--from-codex` session import; HTML export with
  embedded subagent transcripts; `/share` = AES-256-GCM sealed gzip to share server
  (key in URL fragment) or secret gist; optional secret redaction; user
  `~/.omp/agent/share.{ts,js}` hook.
- Rewind/checkpoint: model-facing `checkpoint`/`rewind` tool pair — marks state +
  `branchWithSummary` cuts exploration into a report (`rewind-report` entry); yield
  blocked until rewind completes; rehydrated on resume.

### 3.2 Model selection, roles, catalog, fallbacks

- Catalog (`packages/catalog`): generated `models.json` (75 provider keys, 5,671 rows
  at pin) + **KDL rule tree** `compat/rules/{taxonomy,classes,providers,runtime,auth}/`
  → compiled `rules.json`. `Model` fields: api, reasoning, `thinking` config
  (effort|budget|google-level|anthropic-adaptive|anthropic-budget-effort, efforts,
  effortRouting, effortBudgets, prefixBinding), `input [text,image]`, supportsTools/
  ComputerUse, cost (incl. time-based), promptCache lifetimes, contextWindow/
  maxContextWindow/maxTokens, `compat` (resolved), identity {class,family,revision},
  `kind` (chat/tiny/image/tts/stt/search/judge/embedding/rerank/video), requestModelId,
  `contextPromotionTarget`, `compactionModel`, `webSearch`, `remoteCompaction`,
  `accountAccess`, `transport:"pi-native"`, `preferWebsockets`, provider-specific
  fields.
- `model-manager.ts` merge precedence: bundled/static → SQLite cache (`models.db`,
  schema v13, materialization-policy stamp) → models.dev fallback → dynamic endpoint
  fetch; 2h default TTL; headers never persisted.
- Shared catalog refresh: background fetch
  `https://catalog.stencil.so/models.json.zstd` merged additively (new models appear
  without binary updates); not authoritative for removals.
- `models.yml` (`~/.omp/agent/`): custom providers/models/overrides — `baseUrl`,
  `apiKey` (env or `!command`), `api`, `headers`, `auth` (`apiKey|none|oauth`),
  `authHeader`, `disableStrictTools`, `discovery` (`ollama|llama.cpp|lm-studio|
  openai-models-list|proxy|litellm|apple-foundation-models`), `transport: pi-native`,
  `modelOverrides` (~25 overridable fields incl. compat blocks, tokenizer,
  promptCache, contextPromotionTarget), `compat` sparse per-API overrides,
  `remoteCompaction`, Bedrock guardrails/metadata. Implicit discovery for
  ollama/llama.cpp/lm-studio/apple when unconfigured (`docs/models.md`,
  `docs/local-models.md`).
- Selector grammar: `provider/modelId`, bare id (multi-provider tie-break:
  recent → `modelProviderOrder` → registry), retired effort-variant aliases,
  fuzzy/substring, globs (`--models`, `enabledModels`), `:thinkingLevel` suffix
  (`off|minimal|low|medium|high|xhigh|max|auto|inherit`), `@upstream` routing suffix
  (OpenRouter/Vercel).
- **Roles** (`modelRoles` in config.yml): chat roles `default, smol, slow, vision,
  plan, commit, tiny, memory, task, advisor`; model-kind roles `image, web, speech,
  dictation, judge`; `@role` aliases, comma-separated first-available selectors,
  thinking suffixes; role inheritance (tiny→smol→default; memory→tiny); custom roles;
  `modelPresets` snapshots; path-scoped `enabledModels`/`enabledProviders`/
  `disabledProviders`.
- Fallbacks: `contextPromotionTarget` (explicit larger-window switch *before*
  compaction, recorded as ephemeral `model_change` role `fallback`);
  `retry.fallbackChains` (role/model/provider-wildcard keys); credential/model
  rotation via `TurnRecovery`; server-side Anthropic `fallbacks`.
- Initial selection: explicit CLI → scoped list → saved default → provider defaults →
  first available (credentialed providers preferred).
- Per-model/per-provider `tier` (OpenAI service tiers), `providerSessionId`/
  `promptCacheKey` continuity, cache warming (`providers.cacheWarming`,
  `promptCache` lifetimes, `PI_CACHE_RETENTION`).

### 3.3 Compaction orchestration (session layer)

`SessionMaintenance` + compaction-methods: ordered methods `[remote, snapcompact,
handoff, shake, soft]` → engine strategies; async/idle/pre-prompt/mid-run compaction;
overflow and incomplete-output recovery with rollback; per-agent threshold overrides;
`/compact`, `/handoff`, RPC `compact`/`handoff`/`set_auto_compaction`; hooks
`session_before_compact`, `session.compacting`, `session_compact`,
`auto_compaction_start/end`.

### 3.4 Steering, queues, interrupt

- `Agent` queues: `steer()`/`followUp()` (`"all"|"one-at-a-time"`), `interruptMode`
  (`immediate|wait`), asides (non-interrupting delivery at boundaries), queue
  claim/restore on abort, `AgentBusyError`.
- `AgentSession.prompt` pipeline: extension-command expansion → custom TS commands →
  file slash commands → prompt templates → `^model` mentions → magic keywords →
  streaming requires `streamingBehavior: steer|followUp`; `promptGeneration` guards
  drop stale prompts after branch/fork; `PromptDroppedError`.
- `sendCustomMessage` `deliverAs: steer|followUp|aside|nextTurn`; persisted queued
  messages survive resume; `YieldQueue` batches async-job/IRC deliveries.
- RPC mirrors: `steer`, `follow_up`, `abort`, `abort_and_prompt`, queue ops
  (`remove_queued_message`, `promote_queued_message`, mode setters), `steer_subagent`.
- `TurnRecovery` (non-compaction retry policy, `docs/non-compaction-retry-policy.md`):
  error classification → exponential backoff 500ms×2ⁿ cap 8s, 75–100% jitter,
  `retry.maxRetries=10`, `retry.maxDelayMs`, credential rotation, model fallback chains,
  `waitForUsageReset`, `usageReservePct`.
- **TTSR** (time-traveling stream rules): regex/ast-grep/judged rules matched
  incrementally against streamed text/thinking/toolcall deltas → abort stream → inject
  rule reminder → retry; `ttsr_injection` journal entries; `repeatMode/repeatGap/
  interruptMode/contextMode` (`docs/ttsr-injection-lifecycle.md`).

### 3.5 Subagents (`src/task`)

- Spawn: model-facing `task` tool — single or batch `{context, tasks:[{agent, task,
  solutionSpace, outputSchema?, schemaMode?, effort?, blocking?, name?, isolated?}]}`
  (`task/types.ts`); shared pipeline `resolveEffectiveSubagentPolicy` →
  `runStructuredSubagent`; `runSubprocess` executor also serves eval `agent()`/
  `workpool()` and vibe workers.
- **In-process**: subagents are full child `AgentSession`s on the main loop — not
  threads/subprocesses. Isolation is *filesystem-level*: optional CoW checkout via
  pi-iso (`apfs|btrfs|zfs|reflink|overlayfs|projfs|block-clone|rcopy|worktree|copy`),
  merge modes `patch|branch`, nested-patch persistence.
- Concurrency/limits: semaphore `task.maxConcurrency=32` (live-resizable),
  `task.maxRecursionDepth=2`, `task.maxRuntimeMs`, `task.softRequestBudget`,
  `task.enableLsp` (off for children), `task.speculativeLaunch`, `task.eager`,
  `task.completionProbe`; per-agent `spawns` policy, `task.disabledAgents`,
  `before_subagent_spawn` hook (block/re-route model).
- Agent definitions: bundled `scout`, `reviewer`, `security-reviewer`, `task`, `sonic`
  (embedded text) + discovery merge: project `.omp/agents` > `~/.omp/agent/agents` >
  extension `agents/` > Claude marketplace > bundled (first wins); frontmatter: model
  (list/roles), thinking, tools, spawns, output, blocking, autoloadSkills,
  readSummarize, prewalk, advisor, effort (`docs/task-agent-discovery.md`).
- Structured output: per-item `outputSchema` > agent `output` > parent schema;
  `schemaMode strict|permissive`; JTD→JSON-Schema normalization + validator;
  `yield` tool = terminal/sectioned result protocol (`{data}` / `{error}` /
  `type:["section"]` / `type:"result"`, ≤500KB/5000 lines).
- Messaging: **IRC bus** (`irc/bus.ts`) — process-global mailboxes; `send` never
  blocks; delivery revives parked agents, wakes idle, asides busy ones;
  `agent://<id>` write / `agent://all` broadcast; cap 100; gated by `isIrcEnabled`.
- Lifecycle: `AgentRegistry` (running|idle|parked|aborted + metrics, `MAIN_AGENT_ID`);
  `AgentLifecycleManager` parks after `task.agentIdleTtlMs`, revives from
  `session_init` + transcript; tombstones; `AgentActivity` tail reader;
  `TASK_SUBAGENT_*` event channels.
- Async: `AsyncJobManager` detached/background spawns (`async.enabled`,
  `async.maxJobs≤100`); `blocking:true` forces sync wait; results self-deliver into
  parent conversation; `agent://<id>` resolves output/status, `history://<id>`
  transcript.
- Workpool: keep-alive pool for eval `workpool()` — push items, bounded by
  maxConcurrency, follow-up turns per batch.
- Subagents run headless-yolo (approval boundary = the parent `task` call);
  mid-turn compaction forced on; MCP tools proxied over the parent's connections.

### 3.6 Execution modes

CLI `Mode`: `text | json | rpc | rpc-ui | acp` (`--mode`), plus:

- **Interactive TUI** (default on TTY) and **print/headless** (`-p`/`--print`,
  auto-selected on non-TTY/piped stdin; `--mode text|json` also run print-mode output).
- **RPC**: bespoke line-delimited JSON protocol over stdio (not JSON-RPC 2.0;
  `docs/rpc.md`). ~70 command types: prompt/steer/follow_up/abort, session ops
  (new/open/switch/fork/branch/tree/entries/messages), compaction, model/thinking/
  service-tier, queue ops, subagent ops, host tools + URI schemes
  (`host_tool_call`/`host_uri_request`), extension UI frames (`rpc-ui`), voice
  (`live_*`), `btw` side questions, predict. Protocol v2 = chunked base64 transport.
  SDKs: `sdk/{rust,go,python}/omp-rpc`.
- **ACP**: Agent Client Protocol server over stdio (`session/new|load|resume|fork|
  set_mode|prompt|cancel`, `session/request_permission` for approval-elicitation).
- **SDK embedding**: `createAgentSession()` (`src/sdk.ts`) — full in-process surface
  (SessionManager, Settings, AuthStorage, ModelRegistry, AgentRegistry, tools,
  discovery) with subagent options and extension control.
- ~50 `omp <subcommand>`s registered in `cli-commands.ts` (`docs/cli-reference.md`):
  `models`, `login`, `plugin`/`marketplace`, `skill`, `stats`, `usage`, `commit`,
  `collab`, `join`, `acp`, `auth-broker`, `auth-gateway`, `browser-relay`, `ps`,
  `worktree`/`wt`, `shell`, `cleanse`, `git`, `stream`, `clip`, `play`, `ttsr`,
  `tiny-models`, `token`, `toks`, `web-search`, `read`, `find`, `grep`, `say`,
  `predict`, `dry-balance`, `grievances`, `if-bench`, `bench`, `compress`, `gc`,
  `ssh`, `images`, `setup`, `update`, `completions`, `share`, `export`.
- Approval modes: per-tool tiers `read|write|exec`, `policy allow|deny|prompt`,
  `tools.approvalMode always-ask|write|yolo` (default yolo), per-tool overrides,
  `bash.patterns` ordered rules, `bash.allowCompoundCommands`, critical-command
  prompts, provider `pendingSafetyChecks` always prompt; headless fails closed.
  **No sandbox** — approval ≠ containment (`docs/approval-mode.md`).
- Plan mode: read-only mode persisted via `mode_change` entries; plan proposal via
  `xd://propose`; model transition on approval (`--plan-yolo`, `plan.defaultOnStartup`).
- Goal mode: token-budget tracked objective; `/goal`, `--goal`, RPC `goal` ops;
  `mode_change` persistence.
- Vibe mode: director mode — parent reduced to read/todo + `vibe_{spawn,send,wait,
  kill,list}` worker tools driving keep-alive subagents (`docs/vibe-mode.md`).
- Prewalk, `--max-time`, ephemeral `--no-session`, multi-root `--add-dir`, profiles
  (`--profile` → `~/.omp/profiles/<name>/agent`).

### 3.7 Tools (`src/tools`) — three exposure channels

1. **Top-level tool schemas** — 30 builtin names (`builtin-names.ts`; README's "31" is
   stale): `read, bash, edit, ast_grep, ast_edit, ask, debug, ida, eval, github, glob,
   grep, find, lsp, checkpoint, rewind, context_notes, new_context, security_scan,
   task, wait, todo, web_search, write, memory_edit, retain, recall, reflect, learn,
   manage_skill` + hidden `yield, goal, think`; legacy alias `search`→`grep`.
   Settings-gated extras: `generate_image`, `tts`; mode-scoped `vibe_*`; MCP
   `mcp__<server>_<tool>`; extension-registered tools.
2. **`xd://` virtual devices** (`tools/xdev`, default on): `loadMode:"discoverable"`
   tools mount as `xd://<name>` devices driven via `read`/`write` with JSON args —
   most builtins are actually invoked this way; only read/write/bash/edit/glob/grep/
   find/eval/task/wait/ask/todo/yield/web_search/learn/manage_skill/context_notes/
   new_context are guaranteed top-level. Special devices: `xd://resolve|reject|propose`
   (plan/ast-edit gates), `xd://report_issue`, `xd://<tool>/<topic>` docs.
3. **Eval prelude globals**: `browser`, `computer`, `ratchet`, `archive` — not tools,
   but globals injected into eval kernels.

Tool inventory by category:

- **FS/edit**: `read` (path/URL/internal-URI + `:line`, `:raw`, `:img`, `:conflicts`,
  `?q=` vision selectors; dirs, archives, SQLite, PDFs, images/video, .ipynb),
  `write` (files, archive members, SQLite rows, `conflict://`, xd:// transport),
  `edit` (modes `apply_patch|hashline|patch|replace|sloppy`; hashline ops native in
  pi-edit), `ast_grep`, `ast_edit` (native tree-sitter), `glob` (native), `find`
  (semantic jfind cascade), `checkpoint`/`rewind`, `context_notes`/`new_context`,
  `security_scan`.
- **Shell**: `bash` — embedded **Rust brush shell** (pi-shell) with in-process uutils
  coreutils/findutils/sed + jaq; persistent shell sessions; `{command, timeout, cwd,
  pty, async?, name?, ready{log,port}}`; PTY interactive sessions (TUI);
  auto-background after 60s → managed jobs; named **services** under a project daemon
  broker with readiness probes (`proc://` inspect/stdin/kill, persist modes); direnv;
  internal-URL filesystem inside the shell (`cat artifact://...` works); output sink
  with head+tail budget, artifact spill, sixel/kitty frame extraction; `PI_NO_PTY`,
  `PI_DISABLE_UUTILS_BUILTINS`. No sandbox; approval+patterns only.
- **Eval/REPL**: `eval` — one call = one cell, retained kernels. Python: subprocess
  `runner.py` NDJSON, magics (`%pip %load %time`…, `%%bash %%writefile`), top-level
  await, MIME display, env allowlist, tool bridge via loopback HTTP +
  `__omp_tools__`. JS: retained Bun subprocess/Worker, `Bun.*` API (not a sandbox),
  `%bun add`, `%environment`. Bridges (both): `tool.<name>()`, `agent()`,
  `completion()`, `workpool()`, `judge`/`judge_batch`, `budget`, `wait/status/cancel`,
  `@tool`/`tool(fn)` kernel-defined tools, bridge-timeout pause. Speculation:
  shadow-cell snapshot/prefetch.
- **LSP**: `lsp` tool — 14 actions: diagnostics, definition, references, hover,
  symbols, rename, rename_file, code_actions, type_definition, implementation,
  status, reload, capabilities, raw `request`; per-(server,cwd) clients,
  init-failure backoff, **writethrough** (push unsaved buffers + formatOnWrite +
  deferred diagnostics), workspace-diagnostics checkers, lspmux sharing, ~58 bundled
  server configs, JSON/YAML config merge across home/plugin/`.omp`/`lsp.*`.
- **DAP**: `debug` tool — 29 actions (launch/attach, breakpoints incl.
  instruction/data, stepping, evaluate contexts, stack/threads/scopes/variables,
  disassemble, read/write_memory, modules, loaded_sources, custom_request, output,
  terminate, multi-session); adapters gdb, lldb-dap, codelldb, debugpy, dlv,
  js-debug, netcoredbg, kotlin-debug, rdbg; user `dap.{json,yaml}` config.
- **Browser**: eval `browser` prelude — per-tab workers; ~100 helpers (nav, observe,
  ariaSnapshot, screenshot/pdf, interactions, waits, frames, dialogs, emulate,
  cookies/storage, downloads, console/trace/profile, record, vitals, React hooks,
  network routes/HAR, `webmcp*` page tools, `tab.run` in-tab code); backends:
  puppeteer-core headless Chromium (+stealth init), `app.cdp_url` connect,
  **browser-relay** (Chrome MV3 extension → loopback WS relay driving the user's real
  tabs, daemon-broker leases), tern, cmux; Electron attach.
- **Computer use**: `computer` prelude — native `DesktopSession`: displays/windows,
  screenshots (3840×2400 capture → 1280×896 model cap), pixel input
  (background|takeover), AX trees with `eN` refs, clipboard, menus; macOS
  ScreenCaptureKit/AX, X11 XI2/AT-SPI/uinput, Wayland RemoteDesktop/LIBEI portals,
  Windows UIA; `computer.control.acquire` human grant (`docs/computer-use.md`).
- **Web**: `web_search` — 26 engine providers (parallel, perplexity, gemini,
  anthropic, codex, openai, xai, openrouter, zai, exa, tinyfish, jina, kagi, tavily,
  firecrawl, brave, kimi, synthetic, ollama, searxng, startpage, duckduckgo, ecosia,
  google, mojeek, public) + query directives; `read` URL path with ~70 site scrapers
  (github, arxiv, wikipedia, youtube, crates.io, pypi, npm, mdn, stackoverflow,
  sec-edgar, rfc…); native Exa MCP bridge; `github` tool shells out to `gh` CLI.
- **Memory/learning**: `retain|recall|reflect|memory_edit` (backend-selected), `learn`,
  `manage_skill`.
- **Agent meta**: `task`, `wait`, `yield`, `ask` (interactive pickers), `todo`,
  `think`, `goal`.
- **Media**: `generate_image`, `tts` (Kokoro ONNX or xAI voice).
- **IDA**: `ida` — shared IDA Pro DBs via project daemon (NDJSON).

### 3.8 Processes, services, MCP

- Process management: `ps` CLI + `proc://` internal scheme for daemon-supervised
  services; async job manager; `FileLock`; internal-URL filesystem makes shell/tool
  args scheme-aware.
- **MCP** (`src/mcp`): transports `stdio` (JSONL, detached process groups), `http`
  (Streamable HTTP + SSE resume, `Mcp-Session-Id`, negotiated protocol version), `sse`
  (legacy). Config `.omp/mcp.json` + `~/.omp/agent/mcp.json` + root fallbacks; imports
  translated from Claude/Codex/Gemini/OpenCode/Cursor/Windsurf/VSCode; `${VAR}`
  expansion + `!command` secrets; `disabledServers`/`enabledServers`. Lifecycle:
  parallel connect + initialize (protocol 2025-11-25, `roots` capability), 250ms
  fast-startup gate then `DeferredMCPTool`s, background re-registration on
  `list_changed`, reconnect backoff + crash-storm breaker, per-call one reconnect+retry;
  resources/templates/prompts + subscriptions best-effort. Namespacing
  `mcp__<server>_<tool>` (sanitized, hash-suffixed >64 chars, deterministic collision
  winner); arg normalization; `structuredContent` preserved; `www_authenticate` →
  reauth+retry. **Full MCP OAuth** (discovery, loopback callback, profile-scoped
  stored creds, refresh, 401/403 retry); Smithery registry integration. Browser-MCP
  servers auto-dropped when the browser prelude is available.
- **Blob-broker** (distinct from session blob store): gives outgoing images fetchable
  URLs — ~65 destinations (image hosts, cloud drives, S3/R2/B2, tunnels
  cloudflared/ngrok/tailscale…, `provider-files-{anthropic,gemini,openai}` native file
  APIs, `direct` serve), daemon-shared, unguessable tokens, fail-to-inline.
- **`ssh://`** internal scheme: read/write/grep over ControlMaster reuse, UTF-8
  ≤1MiB/dir-listing, sshfs mount option, `~/.ssh/config` aliases.
- **Internal URL router**: ~18 schemes — `agent, artifact, attachment, cfg, conflict,
  history, issue, pr, local, mcp, memory, omp, proc, rule, security, skill, ssh,
  vault, xd` — each a spec'd handler; the real VFS abstraction for a port.

### 3.9 Extensions, skills, hooks, settings, prompts

- **Extensions**: TS/JS module default-exporting `function(pi: ExtensionAPI)`; loaded
  in-process, **unsandboxed** (managed `ctx.setInterval/setTimeout` the safe path).
  Discovery order: `.omp/extensions/*` (cwd) → `<agentDir>/extensions` → legacy
  settings lists → `hooks/pre|post` factories → installed-plugin `omp.extensions` →
  `-e/--extension/--hook` → settings `extensions`; first-wins dedup;
  `--trusted-extension` allowlist; `disabledExtensions` by `<capability>:<name>` id.
  `ExtensionAPI`: `on(event)`, `registerTool` (full `ToolDefinition` incl. approval,
  loadMode, renderers), `registerCommand/Shortcut/Flag`, `registerProvider` (can
  shadow builtins + supply usage reporting), `registerFileWriteFallback`/
  `FileDeleteFallback` (EPERM seams), message send/`appendEntry`, `exec`,
  active-tools/model/thinking/service-tier setters, `ctx` (ui/mode/hasUI/cwd/
  sessionManager/modelRegistry/compact/abort/shutdown/systemPrompt/`runEphemeralTurn`/
  memory handle), schema builders. Handler timeout 30s; `tool_call` fails closed.
- **Plugins/marketplaces**: two ecosystems — npm/git packages under
  `~/.omp/plugins/node_modules` (`PluginManager`, `bun install`, lockfile,
  `pkg[feat]` spec grammar) and Claude-Code-format marketplace catalogs
  (`name@marketplace`, `.omp-plugin/marketplace.json`, user/project scopes);
  capability dirs per root (`skills/ commands/ rules/ prompts/ hooks/ tools/ agents/
  mcp.json`); Agent Plugins 1.0.0 `plugin.json`; install-time validation + rollback.
  Gemini `gemini-extension.json` manifests = metadata only.
  **Skillshare**: separate SRI-integrity package registry for skills
  (`omp skill install/publish`, `skills.lock.json`).
- **Skills**: `<root>/<name>/SKILL.md` + YAML frontmatter (`name, description, globs,
  alwaysApply, hide, disableModelInvocation, enabled`); providers by priority: native
  `.omp/skills` (ancestor walk) + `~/.omp/agent/skills` > skillshare > omp-plugins >
  claude > agent-plugins > claude-plugins/agents/codex > opencode > github >
  omp-managed; filters `disabledExtensions`/source toggles/`ignoredSkills`/
  `includeSkills`; name collisions → `<ns>/<name>`; system prompt lists skills only
  when a tool declares `readsSkillUris`; `skill://` URI reads; `/skill:<name>`
  invocation; smol-role description compression.
- **Hooks**: two layers — capability items (`hooks/pre|post/<tool>.ts` filename is
  metadata; dispatch still via `pi.on("tool_call")`) and the extension event bus.
  Events: session lifecycle (start/switch/branch/compact/tree/shutdown/stop),
  `input` interception, `before_agent_start` (**can replace the whole system
  prompt**), `before_provider_request`/`after_provider_response`, `context`,
  `agent_start/end`, `turn_start/end`, `assistant_message` rewrite, message events,
  `cache_warming_decision`, `tool_call` (block/replace args/additionalContext) /
  `tool_result` (rewrite/isError), `tool_execution_*`, `tool_approval_*`,
  `before_subagent_spawn` (block/model-route), `auto_compaction_*`, `auto_retry_*`,
  `retry_fallback_*`, `ttsr_triggered`, `todo_reminder`, `goal_updated`,
  `credential_disabled`, `mcp_notification`, `user_bash`/`user_python` override.
  (`resources_discover` exists but is dead — no emitters.)
- **Settings**: layered `config.yml` (global `<agentDir>` + project `.omp/config.yml`
  + `--config` overlays + env + CLI; deep merge, arrays replace, `null` tombstones;
  typed `cfgX` handles declared via `config/registry.ts`; live reload keep-last-good;
  broken YAML quarantined; profiles; XDG routing). Key runtime settings: `modelRoles`,
  `modelPresets`, `enabledModels`/`enabledProviders`/`disabledProviders`
  (path-scoped), `defaultThinkingLevel`, `thinkingBudgets.*`, sampling knobs,
  `tier.{openai,anthropic,google,subagent,advisor}`, `retry.*`, `tools.approvalMode` +
  `tools.approval.<tool>`, `bash.patterns`, `bashInterceptor`,
  `compaction.{enabled,asyncEnabled,midTurnEnabled,methodOrder,threshold*,
  reserveTokens,keepRecentTokens,autoContinue}`, `task.*` (~20 keys), `async.*`,
  `memory.backend`, `autolearn.*`, `extensionHandlers.toolCallTimeoutMs`,
  `tools.{xdev,xdevDocs,xdevInlineDevices,maxTimeout,artifact*,intentTracing}`,
  `lsp.*`, `mcp.*`, `secrets.enabled`, `speech.*`, `stt.*`, `magicKeywords.*`,
  `skills.*`, `ttsr.*`, `collab.*`, `advisor.*`.
- **Prompts**: `buildSystemPrompt` → ordered `string[]` blocks (Handlebars `.md`
  templates imported as text; ~60 templates under `src/prompts/`): instruction
  template → eval-prelude blocks → `<project-context>` (workstation, `<repo-rules>`
  context files, dir-context, workspace tree, completion requirements). Anthropic
  cache breakpoint placed before the context block. Customization precedence: CLI
  `--system-prompt*` > discovered `SYSTEM.md`/`SYSTEM_TEMPLATE.md` >
  `--append-system-prompt`/`APPEND_SYSTEM.md` > `TITLE_SYSTEM.md`/`PERSONALITY.md`.
  **Context files**: multi-provider merge — `.omp/AGENTS.md` (ancestor walk), user
  AGENTS.md, `.claude/CLAUDE.md`, `.codex/AGENTS.md`, `.gemini/GEMINI.md`,
  `~/.config/opencode/AGENTS.md`, `.github/copilot-instructions.md`, `.agents/`; `@`
  imports (≤5 depth); sticky `RULES.md`. **Magic keywords**: `ultrathink`,
  `orchestrate`, `workflowz`, `jevify` (tool-gated hidden directives).
  **Rulebook**: `rules/*.{md,mdc}` from native/plugins/cursor/windsurf/cline/github →
  buckets: TTSR registration, always-apply, advisory `rule://` reads; `agents:` glob
  scoping. Prompt templates `~/.omp/agent/prompts/*.md` → `/name` commands.
- **Secrets obfuscation** (opt-in): env + `secrets.yml` + credential-pattern
  collectors → reversible `$$HASH$$` placeholders (HMAC key) or one-way replace;
  restored in tool args pre-exec.
- **Usage/stats**: `agent.db` live (`model_usage`, `model_perf` decaying tok/s+TTFT,
  `usage_history`, `client_usage`); `stats.db` derived offline from session JSONL
  (`omp stats` dashboard, frustration judging via `judge` role); OTel export opt-in
  (`OTEL_*`); `install-id` file.
- **Memory backends**: `off|local|hindsight|mnemopi|sharpshooter` — local two-phase
  extract→consolidate to `~/.omp/agent/memories/<cwd>/` (MEMORY.md, playbooks,
  `learned.md`); mnemopi = `@oh-my-pi/pi-mnemopi` SQLite banks + ONNX embeddings
  (bge-base/e5-large) + own MCP server; hindsight = remote server; sharpshooter =
  friction-gated decision files. `/memory` subcommands route via `MemoryBackend`.
- **Slash commands**: unified registry; `handle` (host-agnostic, works in
  RPC/ACP/print) vs `handleTui` (interactive only); ACP advertises only `handle`
  specs. Runtime-capable: `/advisor /export /share /compact /handoff /fresh /retry
  /pin /rename /move /model /switch /effort /fast /slow /extended-context /ratchet
  /prewalk /modelpreset /todo /session /jobs /usage /stats /context /tools /mcp
  /memory /ssh /force /images /shake /elide /security /marketplace /plugins
  /reload-plugins /skillful /computer /thinking /trace /dump /browser /help`.
  TUI-only: `/new /clear /delete /resume /exit /restart /tree /branch /fork /collab
  /join /record /live /pause /quit /btw /tan /omfg /cleanse /debug /plan /vibe /goal
  /guided-goal /loop /queue /settings /setup /hotkeys /extensions /agents /git /hub
  /login /logout /skills`. File commands (`.md`) and TS custom commands
  (`commands/<name>/index.ts`) expand in the same pipeline; unknown `/foo` falls
  through to the model.

### 3.10 Collab, advisor, goals, misc

- **Collab** (`src/collab`): live session replication over a relay (`wss://my.omp.sh`
  default) with AES-256-GCM sealed frames; guests render stream + tool cards +
  subagent views and can prompt/interrupt (control link = key + write token); local
  host registry on unix socket (`omp collab`); `collab-web` browser SPA; 4MiB
  transcript cap.
- **Advisor** (`src/advisor`): reviewer-model sideband — transcript deltas at
  `turn|agent-end` boundaries, notes injected via `advise` tool; isolated ToolSession
  (read/grep/glob/recall grants); `WATCHDOG.yml` roster (multiple advisors, per-entry
  model/cadence/tools); `__advisor*.jsonl` transcripts; `advisor` model role +
  fallback chain; excluded from peer surfaces (`agent://`, hub).
- **Goals** (`src/goals`): token-budgeted objective mode (above).
- **Security scans** (`src/security`): model-driven scans + SARIF + Codex Security
  cloud + `security://` namespace; `security_scan` tool + `/security`.
- **Stream/record**: `omp stream/clip/play`, `/record`, `/live` — screen/chat
  broadcast (`pi-wire` stream proto) — TUI-adjacent, not runtime.
- **TTS/STT**: Kokoro-82M ONNX TTS (`say`, `tts` tool, speech-enhancer); whisper/
  sherpa-onnx STT push-to-talk; tiny local models (`tiny` kind) for titles/memory/
  prediction (`docs/local-models.md`).
- **Auto-subsystems** (larger feature surfaces, runtime-owned): `autoresearch`,
  `auto-graph`, `cleanse`, `commit`, `predict`/`pi-predict`, `stencil`, `hindsight`,
  `sharpshooter`, `autolearn`, `if-bench`, `tiny` workers, `eval` speculation —
  enumerated for completeness; each is its own subdir with settings + prompts.

## 4. Native Rust layer (`crates/pi-*` + `packages/natives`)

All first-party Rust links into **one N-API cdylib** (`crates/pi-natives`), loaded
in-process by Bun (`packages/natives` = loader + npm packaging only, no `src/`).
Consumers import e.g. `import { Shell } from "@oh-my-pi/pi-natives"` — 78 files in
coding-agent alone.

| Crate | Surface (verified via `native/index.d.ts` exports / crate src) |
|---|---|
| `pi-shell` | Persistent embedded **brush shell**: `Shell`, `execute_shell`, output minimizer, git fast paths, process/cancel. The `bash` tool's executor. |
| `pi-builtins` | In-process bash builtins + **uutils coreutils/findutils/sed + jaq ports** (~80 utilities). Third-party LICENSE. |
| `pi-vfs` | Injectable async fs trait + blocking facade + native host fast paths (all shell/walker I/O). |
| `pi-walker` | Parallel fs walker (ignore/globset/rayon) + shared fs scan cache (`PI_WALK_WORKERS`). |
| `pi-vcs` | In-process git (gix) + **jj-lib 0.44**; network ops shell out to git CLI deliberately. `vcs*` + repo watch. |
| `pi-diff` | Myers/line/word diffs, structured patches (`diff*` exports). |
| `pi-edit` | Edit engine: replace/patch/apply_patch, fuzzy match, **hashline ops**, `EditSession`/`EditStore` snapshots, notebook, streaming. |
| `pi-ast` | Tree-sitter registry (~60 grammars), ast-grep ops, block analysis, code summarization. |
| `pi-iso` | Workspace isolation backends: APFS, btrfs, ZFS, reflink, overlayfs, ProjFS, block-clone, worktree/copy. |
| `pi-predict` | Ghost-text completion: ngram + SmolLM2 GGUF (candle, Metal) + Apple spelling. |
| `pi-voice` | Mic/playback (CoreAudio/WASAPI/Linux), Opus, WebRTC peer. |
| `pi-natives` (top) | Plus `pty` (portable-pty `PtySession`), `Process`, `FileLock`, `TtyWriter`, grep/search (`grep-pcre2`, `OMP_PCRE2_JIT`), `countTokens` (hand-rolled BPE: cl100k/o200k/deepseek/kimi/qwen/Claude), text-width/`wrapTextWithAnsi`, syntect highlight, `htmlToMarkdown`, `pdfToMarkdown`, `rasterizeSvg`, `renderMermaidAscii`, sixel codec, `renderSnapcompactPng`, vector ops (topK/MMR), `DesktopSession` (screen capture + input injection + AX across macOS/Wayland/X11/Windows), `NativeOAuthCallback`, devicecheck, power assertions, crash handler, appleFm bridge. |

Loader mechanics (`native/loader-state.js`): platform leaf packages
`@oh-my-pi/pi-natives-<tag>` (6 tags × `-modern/-baseline` x86-64 v3/v2 variants),
`PI_NATIVE_VARIANT`, compiled-binary embedded addon (`embedded-addons.<tag>.tar.gz`)
extracted to a versioned cache dir (`PI_NATIVES_DIR`/`~/.omp/natives`), post-load
`__piNativesBuildVersion` check (version **stamped post-link**, not compiled in),
`__ompInstallTokioRuntime` deferred init. Releases build via Bazel
(`//:natives-<target>`).

**For Juto:** these crates are candidates for direct Rust reuse. Inspect their
public interfaces, N-API dependencies, runtime initialization, and license/data
requirements before extracting or linking them. Reuse can preserve behavior,
but it is not a drop-in replacement for the TypeScript orchestration.

## 5. Runtime vs terminal-UI boundary

Terminal-specific (`packages/tui` + `src/modes/controllers/*`), which GPUI replaces:

- Differential renderer (`tui.ts`: `TUI`/`Container`/`Component`), `ProcessTerminal`,
  alt-screen, ConPTY, SGR/mouse/bracketed-paste, Kitty keyboard/graphics, sixel,
  `TtyWriter`, tmux, terminal-capability detection.
- ~73 overlays/pickers, ~62 per-tool ANSI renderers, transcript/chat components,
  composer/editor, status line, theme runtime, charts, fullscreen apps (git TUI,
  ps-top, gallery), keybindings, kill-ring, vim mode.
- `modes/controllers/*`: TUI glue between `AgentSession` and pi-tui (input/selector/
  event/btw/todo/session-focus controllers).
- TUI-only slash commands listed in §3.9; `src/debug/*` diagnostics menu; interactive
  PTY overlay; downloads tracker; stream/recording UI.

**Not terminal-only** (pi-tui doubles as shared toolkit): tool *types/render
contracts* used by RPC/collab/export, `OutputSink` truncation policy, `fuzzy`,
`autocomplete`, `latex-to-unicode`, shared key registry.

**Ready-made seam:** `packages/tui/src/native/` — **Tern Surface Protocol (TSP)**:
components implement `describe()` → semantic `NativeNode` trees → `reconcile` diffs →
`TspOp` frames with credit flow control + blob dedup over pi-wire `tsp.ts`. Designed
for a native host; a GPUI app can consume the same describe-tree vocabulary instead of
terminal APC bytes.

## 6. Rust module seam proposal (prospective — nothing implemented)

Mirroring omp's layering; dependency direction only downward. All names provisional.

| Juto crate | Ports | Notes |
|---|---|---|
| `juto` (app, GPUI) | packages/tui + modes/controllers | Window, transcript surface, composer, pickers; consumes `juto-session` events. The existing package is `apps/juto`; runtime modules below are prospective. TSP's vocabulary may inform rendering, but the app will own native GPUI views rather than run OMP beneath a window. |
| `juto-provider` | packages/ai | `Provider` interface per wire API; `stream()`/`streamSimple()`; `AssistantMessageEvent` stream; error taxonomy; usage/cost; auth cascade (SQLite via rusqlite); OAuth engines (oauth-code/device-code/custom); auth-broker client and broker/gateway servers; dialect scanners; schema normalization. Start: anthropic-messages + openai-responses/-completions; Codex WS, Bedrock SigV4, Vertex, Cursor/Devin protobuf are per-API follow-ons. |
| `juto-catalog` | packages/catalog | `Model` type; bundled models.json (embed or regenerate); KDL→rules compat cascade (or ship compiled rules.json); identity/dialect classification; model-manager merge + cache; pricing. |
| `juto-loop` | packages/agent | `agentLoop` core: turn loop, tool dispatch (concurrency/interruptible), steering queues + live-steering channel, pause gate, append-only context, tokenizer hook, output budget, sent-tool-defs, replay policy, compaction engine (shake/prune/branch-summary/native+local), speculation coordinator, telemetry hooks. `Agent` wrapper = session-side concern here too. |
| `juto-session` | coding-agent session/modes | SessionEntry journal (byte-exact JSONL incl. 256-byte title slot), SessionManager (fork/branch/tree/navigate/continueRecent), SessionStorage trait (file/memory/SQL), blob store, artifacts, `AgentSession` facade (the 13.4k-line god-object decomposed: prompt pipeline, queues, TurnRecovery, compaction orchestration, TTSR, goals/plan/vibe state, approval resolver). |
| `juto-tools` | coding-agent tools | Tool trait + registry; fs/edit/search tools (thin over native crates); `bash` over pi-shell; eval kernels (py runner NDJSON + a JS option); lsp/dap clients; browser/computer preludes; MCP client (3 transports + OAuth); internal-URL router/VFS; blob-broker; approvals. |
| `juto-agents` | coding-agent task/irc/registry | Subagent executor (in-process AgentSession children), semaphore/workpool, isolation via pi-iso, structured yield, IRC bus, lifecycle/park-revive, async jobs, agent definitions/discovery. |
| `juto-ext` | extensibility | OMP extensions execute unsandboxed TS/JS with Bun/Node APIs and UI callbacks. Full compatibility requires an explicit JS execution and host-interface design; a Rust trait or WASM interface alone would change that contract. Skills/hooks/rules/prompts/context-file discovery can port independently. Preserve this requirement in the parity matrix; do not silently drop extensions. |
| `juto-proto` | wire + rpc | Session/event wire types; omp-compatible JSONL RPC protocol (v2 chunking); ACP server; internal-URL scheme specs. |
| Native reuse | crates/pi-* | Link `pi-shell`, `pi-builtins`, `pi-vfs`, `pi-walker`, `pi-vcs`, `pi-edit`, `pi-diff`, `pi-ast`, `pi-iso`, `pi-natives` internals (tokenizer, grep, highlight, desktop) as path/git deps — keep their LICENSEs (MIT + third-party notices). |

This is a module map, not a requirement to create nine crates immediately.
Introduce crates only when they own real behavior and have a useful interface.

## 7. Ordered implementation plan toward parity

Acceptance per stage is observable behavior, not structure. Stages sequence
delivery without removing any runtime capability from the requested port scope.
The initial usable harness must include the user's named features: provider login,
model switching, messages, and subagents; full runtime parity remains the target.

1. **Foundation (done, this step)** — workspace, Nix, GPUI window, jj.
   Observed: locked Cargo build; rendered native X11 window under Xvfb/Mesa;
   Ctrl-Q exit status 0. Wayland is enabled but has not been exercised.
2. **`juto-catalog` + `juto-provider` minimal vertical** — Model type + bundled catalog
   embed; anthropic-messages + openai-completions streams (SSE parse → unified event
   stream); API-key + env auth; Usage/cost. Acceptance: a CLI test streams a real
   response from each wire API through the unified event type with token counts.
3. **`juto-loop` core** — turn loop, tool dispatch with before/after hooks, steering
   queues, abort, deadline; append-only context; tokenizer via pi-natives crate;
   output budget. Acceptance: scripted two-tool session against a recording/mock
   provider AND a live provider; steering mid-run alters the next request.
4. **`juto-tools` fs+shell slice** — read/write/edit (hashline via pi-edit)/glob/grep
   (pi-walker/grep)/bash (pi-shell); approval resolver (tiers + patterns + modes).
   Acceptance: model can fix a file end-to-end via tools; yolo vs always-ask observed.
5. **`juto-session` journal** — SessionEntry tree, JSONL byte-compat format, blobs,
   resume/fork/branch/navigateTree; compaction engine (local summarize + shake first;
   provider-native lanes later). Acceptance: resume byte-for-byte compatible with an
   omp-written session file; fork/branch tree semantics match.
6. **GPUI app slice** — transcript/composer/picker surface bound to session events
   (TSP describe-model where useful); print + interactive modes. Acceptance: full
   interactive session: prompt → streaming render → tool cards → approval prompt →
   resume next launch.
7. **Auth depth** — OAuth code+PKCE + device-code flows, SQLite store, refresh
   leases, rotation a/b/c; models.yml custom providers/models/overrides + discovery
   (ollama/lm-studio/openai-models-list); roles + selectors + fallbacks +
   contextPromotion; service tiers; cache warming. Acceptance: OAuth login round-trip
   with refresh; `provider/model:level` selection incl. roles; documented failover.
8. **`juto-agents`** — task tool, subagent sessions, concurrency/recursion limits,
   structured yield, IRC bus, parked/revive, async jobs, agent definitions+discovery.
   Acceptance: parallel subagents with schema-checked results and peer messaging.
9. **Tool breadth** — eval (py first), lsp, debug/DAP, task extras, memory tools,
   web_search providers, github; MCP client (stdio→http/sse, namespacing, OAuth,
   list_changed). Acceptance: each tool round-trips against its real backend.
10. **Modes + protocols** — RPC JSONL (v2), ACP, print/json modes, plan/goal/vibe
    state machines, collab, advisor, TTSR, checkpoints/rewind. Acceptance:
    `sdk/rust/omp-rpc`-equivalent client drives a session; ACP handshake with a real
    editor client.
11. **Extensibility + ecosystem** — skills/rules/context-files/prompt-templates/hooks
    discovery & injection (plain-file part); implement the TS/JS host compatibility
    contract; plugins/marketplace/skillshare; secrets obfuscation;
    memory backends; TTS/STT/browser/computer-use. Acceptance: documented parity
    matrix vs the pinned omp revision.

### Initial usable harness acceptance

The foundation above is not the agent-harness MVP. That milestone covers stages
2–8 and requires an end-to-end native GPUI session with:

- A real provider-supported login flow, credential persistence and refresh,
  plus API-key authentication where appropriate.
- Model switching that changes subsequent requests while preserving valid history.
- Streaming messages, tool calls/results, cancellation, and session resume.
- Concurrent subagents with visible lifecycle, structured results, and cancellation.

Stages 9–11 and advanced transports, broker/gateway, browser/computer use,
extensions, collaboration, advisor, TTSR, voice, and tiny-model inference remain
in the full runtime-port scope. They are sequenced later, not deleted from scope.
Do not claim parity until the pinned-source capability matrix is exercised.

## 8. Limitations of this inventory

- Inventoried at pin `579da1d6`; upstream moves fast (CHANGELOG shows near-daily
  feature commits). Re-pin before each port stage.
- Docs in `docs/` are authoritative but occasionally lag source; where they conflicted
  the source won (e.g. 30 vs 31 tool names).
- The `coding-agent` package contains additional subsystems only enumerated by name
  (§3.10); their internals were not exhaustively surveyed.
- Bun-API surface needed by a port was surveyed by usage, not exhaustively grepped;
  expect stragglers (`Bun.*`, `bun:*`, `node:*` compat).
