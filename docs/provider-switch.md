# Paused Agent identity migration

`agent_runtime_kit.flow.provider_switch` implements an explicit migration boundary.
`plan_provider_switch` inventories source Agent identities and future references;
`apply_provider_switch` validates the CAS plan and preflights destination registry Homes.
Call planning under `switch_boundary`; apply acquires that boundary itself.

Each source identity gets at most one candidate per committed group. Live Flow bindings,
CREATED Steps and unconsumed dispatch requests migrate in one store edit session.
Current suspended Steps must pass existing recovery assessment and receive new replacement
Steps; their original records remain immutable. Agent creation/closing is outside the
transaction and reported separately. Retrying uses the original source IDs. The operation
never unpauses or enqueues; an application must verify completion and rebuild scheduler
candidates before explicitly continuing. This is not a crash-atomic transaction.

Applications may name specific Flow types whose consumed `input.agent_id` is historical;
other source identity references in Flow inputs fail closed. LC's integration and deployment
procedure are documented in its `docs/provider-switch.md`.
