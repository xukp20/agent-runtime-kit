# Provider startup failure diagnostics

Unexpected AgentStep exceptions remain suspended. They do not become business failures and do not automatically retry.

Inspect the persisted Step `error.details.diagnostics` through the existing Step read API. It contains:

- `exception_type` and up to twelve traceback frames (`file` basename, `function`, `line`). No source text, locals, or arbitrary exception message is exported.
- `stage`, when supplied by the Grok adapter: execution context, workdir validation, resume identity, MCP preflight, ACP initialization, session load/create, Home session-start commit, tool catalog, or prompt.
- `code` for recognized fixed adapter errors (auth reference, manifest, managed files, config, or missing native session identity).
- `fingerprint`, a grouping key based on exception type, stage, known code, and final frame. Compare within a code version; line changes can change the fingerprint.

Monitors should report the project, Agent role/id, Step/Flow/lease ids, code identity, diagnostics, recovery actions, and paused/active state. Group repeated fingerprints across roles and projects. A shared fingerprint is evidence of a common failure location, not proof of its underlying cause. Do not retry a deterministic preflight failure before resolving its cause.

Grok emits the complete session locator immediately after session creation/load, before Home commit and tool catalog checks. AgentService persists this native identity even if startup subsequently fails. Historical partial locators are not repaired automatically; do not invent native paths or blindly resume those sessions.

HomeService serializes context validation, explicit sealing, and lifecycle manifest commits per Home within one service instance. The lock covers manifest and database updates, not model turns or different Homes. It does not coordinate separate production processes sharing a runtime, nor prevent native CLI writes outside ARK. It does not automatically accept changed auth references.

This change adds evidence for future failures; it cannot reconstruct exception details discarded by older processes. Deploy at a settled pause boundary. Diagnose the first resumed attempt before broad recovery, and keep checkpoint restore separate from ordinary Step recovery.

A session-load event preserves an existing complete locator, including backend identity resolved by the previous turn. Only partial locators are upgraded by startup events; the completed turn and its session remain consistent.

Both Grok session creation and loading invoke the trusted Home lifecycle commit before the prompt. Only the existing verified marketplace metadata exception is allowed. Previously unsealed changes require an operator-verified lifecycle commit at a settled boundary; arbitrary config or credential changes remain rejected.

Scheduler admission failures now pause the runtime before terminating a semantic lease. The lease exposes `failure_diagnostics` with safe exception frames/fingerprint, so monitors should capture it before a process restart. Existing running work may drain, but new admission remains paused. Flow/Step scope and global index rebuilds parse records first and publish the replacement in one SQLite transaction; readers do not observe a cleared intermediate index.

### Required MCP initialization failures

Codex thread creation and resume now identify an explicit required-MCP initialization failure without exposing its raw message. Diagnostics include a fixed category, failure count, and booleans for the LC application and submission servers. Unknown RPC failures remain unclassified.

Only an explicit aggregate containing exclusively `MCP client startup timed out` failures may be retried once before any turn starts. The retry delay is at most 30 seconds and the existing retry policy may disable it. Authentication errors, mixed failures, transport disconnects and generic `-32603` errors are not replayed by this gate. Exhaustion preserves the exception and attempt count for suspended-Step recovery. This does not retroactively classify historical errors or prove that MCP endpoint latency has been resolved.

### Unclassified Codex RPC message indicators

RPC failures additionally expose message length and fixed boolean indicators for the resume wrapper, required-MCP aggregate, startup timeout, event-stream timeout, handshake, generic timeout, and the two LC server names. No raw message or RPC data is exported. Messages longer than 16 KiB only expose length. These indicators do not change retry eligibility: an unclassified error remains suspended for operator review.
