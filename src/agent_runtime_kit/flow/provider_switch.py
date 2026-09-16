"""Paused, explicit identity migration; completed execution evidence is immutable."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict

from .models import FlowStatus, FlowStepValidationError, StepStatus
from .standard_steps.agent_step import AgentStep
from .standard_steps.dispatch_step import DispatchStepState


def _in_scope(scope: str, root: str) -> bool:
    return scope == root or scope.startswith(root + ":")


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def _contains(value, ids: set[str]) -> bool:
    if isinstance(value, str):
        return value in ids
    if isinstance(value, dict):
        return any(_contains(v, ids) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains(v, ids) for v in value)
    return False


@contextmanager
def switch_boundary(service):
    ark = service.ark
    if ark.pause_controller is None or ark.agent_service is None:
        raise FlowStepValidationError(
            "Provider switch requires pause and Agent services"
        )
    with (
        ark.pause_controller.hold_paused(None),
        ark.agent_service.hold_agent_boundary(),
        service.lock,
    ):
        schedule = ark.schedule_service
        steps = ark.step_service
        if (
            getattr(schedule, "active_flow_advances", ())
            or getattr(steps, "active_steps", ())
            or service.store.list_steps(status=StepStatus.RUNNING)
            or ark.agent_service.has_running_agents()
        ):
            raise FlowStepValidationError("Provider switch requires zero active work")
        yield


def plan_provider_switch(
    service,
    *,
    scope_id: str,
    source_agent_ids: list[str],
    target_provider: str,
    historical_input_agent_flow_types: tuple[str, ...] = (),
    historical_failed_flow_ids: tuple[str, ...] = (),
):
    """Return a credential-free CAS plan. Caller holds switch_boundary."""
    agents = service.ark.agent_service
    sources = {key: agents.get_agent(key) for key in sorted(set(source_agent_ids))}
    if not sources or any(
        not _in_scope(a.scope_id, scope_id) for a in sources.values()
    ):
        raise FlowStepValidationError("Source Agents must belong to the selected scope")
    defaults = {}
    blockers = []
    for spec in agents.agent_types.list():
        provider, home_id = spec.provider_type, spec.default_home_id or spec.agent_type
        if provider != target_provider:
            blockers.append(f"default_provider_mismatch:{spec.agent_type}")
        home = agents.home_service.get_home(provider, home_id)
        if home.status != "active":
            blockers.append(f"inactive_target_home:{spec.agent_type}")
        defaults[spec.agent_type] = {
            "provider_type": provider,
            "home_id": home_id,
            "home_fingerprint": _digest(
                (home.materialization_manifest_hash, home.base_config_fingerprint)
            ),
        }
    groups = {
        key: {
            "source_agent_id": key,
            "source_status": a.status,
            "agent_type": a.agent_type,
            "scope_id": a.scope_id,
            "source_fingerprint": _digest(asdict(a)),
            "target": defaults[a.agent_type],
            "references": [],
        }
        for key, a in sources.items()
    }
    for key, a in sources.items():
        target = defaults[a.agent_type]
        if (a.provider_type, a.home_id) == (target["provider_type"], target["home_id"]):
            blockers.append(f"target_identity_unchanged:{key}")
        if a.status not in {"idle", "closed"}:
            blockers.append(f"source_not_settled:{key}")
    # Retire every off-target idle identity, including purely historical inheritance candidates.
    for a in agents.list_agents():
        if not _in_scope(a.scope_id, scope_id) or a.status == "closed":
            continue
        target = defaults.get(a.agent_type)
        if target is None:
            blockers.append(f"unknown_agent_type:{a.agent_id}")
        elif (a.provider_type, a.home_id) != (
            target["provider_type"],
            target["home_id"],
        ) and a.agent_id not in sources:
            blockers.append(f"unselected_source:{a.agent_id}")

    flows = service.store.list_flows()
    steps = service.store.list_steps()
    all_agents = {a.agent_id: a for a in agents.list_agents()}

    def validate_binding(key, location):
        if key in groups:
            return
        a = all_agents.get(key)
        if a is None:
            blockers.append(f"missing_agent:{location}")
            return
        target = defaults.get(a.agent_type, {})
        if a.status != "idle" or (a.provider_type, a.home_id) != (
            target.get("provider_type"),
            target.get("home_id"),
        ):
            blockers.append(f"unselected_live_binding:{location}")

    relevant = []
    unbound = []
    for flow in flows:
        scoped = _in_scope(flow.scope_id, scope_id)
        if flow.status in {FlowStatus.COMPLETED, FlowStatus.FAILED}:
            # A failed child beneath a completed parent is historical. Other failed
            # Flows need explicit business recovery; time order is not lineage proof.
            if (
                scoped
                and flow.status == FlowStatus.FAILED
                and flow.flow_id not in historical_failed_flow_ids
                and not any(
                    f.flow_id == flow.parent_flow_id
                    and f.status == FlowStatus.COMPLETED
                    for f in flows
                )
            ):
                blockers.append(f"failed_current_flow:{flow.flow_id}")
            continue
        input_data = (
            flow.input.model_dump(mode="json") if flow.input is not None else {}
        )
        if flow.flow_type in historical_input_agent_flow_types:
            input_data.pop("agent_id", None)
        if _contains(input_data, set(sources)):
            blockers.append(f"unsupported_flow_input_reference:{flow.flow_id}")
        refs = flow.agent_bindings.by_role
        if not scoped:
            if (
                _contains(flow.state.model_dump(mode="json"), set(sources))
                or any(a in sources for a in refs.values())
                or any(
                    _contains(s.model_dump(mode="json"), set(sources))
                    for s in steps
                    if s.flow_id == flow.flow_id
                    and (
                        s.status == StepStatus.CREATED
                        or s.step_id == flow.current_step_id
                    )
                )
            ):
                blockers.append(f"cross_scope_reference:{flow.flow_id}")
            continue
        relevant.append(flow.model_dump(mode="json"))
        for role, key in refs.items():
            if key in groups:
                groups[key]["references"].append(
                    {"kind": "flow", "id": flow.flow_id, "role": role}
                )
            else:
                validate_binding(key, f"{flow.flow_id}:{role}")
        if _contains(flow.state.model_dump(mode="json"), set(sources)):
            blockers.append(f"unsupported_flow_state_reference:{flow.flow_id}")
        for step in (s for s in steps if s.flow_id == flow.flow_id):
            if step.status not in {
                StepStatus.CREATED,
                StepStatus.SUSPENDED,
                StepStatus.RUNNING,
            }:
                continue
            # Old suspended Steps remain immutable after a replacement.
            if (
                step.status == StepStatus.SUSPENDED
                and step.step_id != flow.current_step_id
            ):
                continue
            relevant.append(step.model_dump(mode="json"))
            if isinstance(step, AgentStep):
                if step.step_id != flow.current_step_id or step.submission is not None:
                    blockers.append(f"unsupported_agent_step_boundary:{step.step_id}")
                    continue
                key = step.agent_bindings.get(step.state.agent_role) or refs.get(
                    step.state.agent_role
                )
                if key is not None:
                    validate_binding(key, step.step_id)
                    if (
                        key in all_agents
                        and step.state.agent_type is not None
                        and all_agents[key].agent_type != step.state.agent_type
                    ):
                        blockers.append(f"agent_type_mismatch:{step.step_id}")
                if step.status == StepStatus.CREATED and step.started_at is not None:
                    blockers.append(f"created_step_already_started:{step.step_id}")
                if step.status == StepStatus.SUSPENDED:
                    assessment = service._assess_agent_step_recovery(step.step_id)
                    if "resume_suspended" not in assessment.view.available_actions:
                        blockers.append(f"recovery_unavailable:{step.step_id}")
                    if key not in sources:
                        blockers.append(f"unselected_suspended_agent:{step.step_id}")
                if key in groups:
                    groups[key]["references"].append(
                        {
                            "kind": "step",
                            "id": step.step_id,
                            "role": step.state.agent_role,
                        }
                    )
                elif step.status == StepStatus.CREATED:
                    agent_type = (
                        all_agents[key].agent_type
                        if key in all_agents
                        else step.state.agent_type
                    )
                    if agent_type not in defaults or step.started_at is not None:
                        blockers.append(f"unsupported_unbound_step:{step.step_id}")
                    else:
                        target = defaults[agent_type]
                        if (step.state.provider_type, step.state.home_id) != (
                            target["provider_type"],
                            target["home_id"],
                        ):
                            unbound.append(step.step_id)
                if _contains(step.state.model_dump(mode="json"), set(sources)):
                    blockers.append(f"unsupported_agent_state_reference:{step.step_id}")
                for role, binding in step.agent_bindings.by_role.items():
                    if role != step.state.agent_role:
                        validate_binding(binding, f"{step.step_id}:{role}")
                        if binding in sources:
                            blockers.append(
                                f"additional_step_role:{step.step_id}:{role}"
                            )
            elif isinstance(step.state, DispatchStepState):
                if (
                    step.status != StepStatus.CREATED
                    or step.started_at
                    or step.state.created_children
                ):
                    blockers.append(f"partially_consumed_dispatch:{step.step_id}")
                    continue
                state = step.state.model_dump(mode="json")
                for index, request in enumerate(step.state.requests):
                    key = request.params.get("agent_id")
                    if key is not None:
                        validate_binding(key, f"{step.step_id}:{index}")
                    if key in groups:
                        groups[key]["references"].append(
                            {"kind": "request", "id": step.step_id, "index": index}
                        )
                        state["requests"][index]["params"]["agent_id"] = None
                if _contains(state, set(sources)):
                    blockers.append(f"unsupported_dispatch_reference:{step.step_id}")
            elif _contains(step.model_dump(mode="json"), set(sources)):
                blockers.append(f"unsupported_step_reference:{step.step_id}")
    payload = {
        "scope_id": scope_id,
        "target_provider": target_provider,
        "historical_failed_flow_ids": sorted(historical_failed_flow_ids),
        "groups": list(groups.values()),
        "defaults": defaults,
        "unbound_created_steps": unbound,
        "blockers": sorted(set(blockers)),
    }
    payload["plan_hash"] = _digest((payload, relevant))
    payload["complete"] = (
        not blockers
        and not unbound
        and all(
            a.status == "closed" and not groups[key]["references"]
            for key, a in sources.items()
        )
    )
    return payload


def apply_provider_switch(
    service,
    *,
    scope_id: str,
    source_agent_ids: list[str],
    target_provider: str,
    expected_plan_hash: str,
    historical_input_agent_flow_types: tuple[str, ...] = (),
    historical_failed_flow_ids: tuple[str, ...] = (),
):
    """Migrate each shared identity in one Flow/Step transaction; never unpause/enqueue."""
    with switch_boundary(service):
        args = dict(
            scope_id=scope_id,
            source_agent_ids=source_agent_ids,
            target_provider=target_provider,
            historical_input_agent_flow_types=historical_input_agent_flow_types,
            historical_failed_flow_ids=historical_failed_flow_ids,
        )
        plan = plan_provider_switch(service, **args)
        if plan["plan_hash"] != expected_plan_hash or plan["blockers"]:
            raise FlowStepValidationError(
                "Provider switch plan changed or has blockers"
            )
        agents = service.ark.agent_service
        # Preflight all future roles, including roles with no live Agent yet.
        for target in plan["defaults"].values():
            agents.home_service.build_execution_context(
                target["provider_type"], target["home_id"]
            )
        if plan_provider_switch(service, **args)["plan_hash"] != expected_plan_hash:
            raise FlowStepValidationError(
                "Provider switch target changed during preflight"
            )
        receipts = []
        for group in plan["groups"]:
            old_id = group["source_agent_id"]
            candidate = None
            committed = False
            receipt = {
                "source_agent_id": old_id,
                "replacement_agent_id": None,
                "replacement_step_ids": [],
                "bindings_committed": False,
                "retired": False,
            }
            try:
                if group["references"]:
                    target = group["target"]
                    candidate = agents.create_agent(
                        group["scope_id"],
                        group["agent_type"],
                        provider_type=target["provider_type"],
                        home_id=target["home_id"],
                    )
                    receipt["replacement_agent_id"] = candidate.agent_id
                    with service.store.edit_session() as tx:
                        for ref in group["references"]:
                            if ref["kind"] == "flow":
                                flow = tx.load_flow_for_update(ref["id"])
                                if flow.agent_bindings.get(ref["role"]) != old_id:
                                    raise FlowStepValidationError(
                                        "Flow binding changed"
                                    )
                                flow.agent_bindings.by_role[ref["role"]] = (
                                    candidate.agent_id
                                )
                            elif ref["kind"] == "request":
                                step = tx.load_step_for_update(ref["id"])
                                step.state.requests[ref["index"]].params["agent_id"] = (
                                    candidate.agent_id
                                )
                            else:
                                source = service.store.get_step(ref["id"])
                                if source.status == StepStatus.SUSPENDED:
                                    assessment = service._assess_agent_step_recovery(
                                        source.step_id
                                    )
                                    if (
                                        "resume_suspended"
                                        not in assessment.view.available_actions
                                    ):
                                        raise FlowStepValidationError(
                                            "Suspended recovery boundary changed"
                                        )
                                    step = service._build_replacement_step(
                                        source, replacement_agent_id=candidate.agent_id
                                    )
                                    step.state.provider_type, step.state.home_id = (
                                        target["provider_type"],
                                        target["home_id"],
                                    )
                                    tx.add_step(step)
                                    flow = tx.load_flow_for_update(source.flow_id)
                                    flow.step_ids.append(step.step_id)
                                    flow.current_step_id = step.step_id
                                    flow.status = FlowStatus.RUNNING
                                    receipt["replacement_step_ids"].append(step.step_id)
                                else:
                                    step = tx.load_step_for_update(source.step_id)
                                    if step.started_at is not None:
                                        raise FlowStepValidationError(
                                            "CREATED Step has started"
                                        )
                                    step.agent_bindings.by_role[ref["role"]] = (
                                        candidate.agent_id
                                    )
                                    step.state.provider_type, step.state.home_id = (
                                        target["provider_type"],
                                        target["home_id"],
                                    )
                    committed = True
                    receipt["bindings_committed"] = True
                agents.close_agent(old_id)
                receipt["retired"] = True
            except Exception as exc:
                receipt["error_type"] = type(exc).__name__
                if candidate is not None and not committed:
                    try:
                        agents.close_agent(candidate.agent_id)
                    except Exception as cleanup:
                        receipt["candidate_retirement_error_type"] = type(
                            cleanup
                        ).__name__
                receipts.append(receipt)
                return {
                    "complete": False,
                    "receipts": receipts,
                    "plan": plan_provider_switch(service, **args),
                }
            receipts.append(receipt)
        try:
            with service.store.edit_session() as tx:
                for step_id in plan["unbound_created_steps"]:
                    step = tx.load_step_for_update(step_id)
                    agent_id = step.agent_bindings.get(
                        step.state.agent_role
                    ) or tx.load_flow_for_update(step.flow_id).agent_bindings.get(
                        step.state.agent_role
                    )
                    agent_type = (
                        agents.get_agent(agent_id).agent_type
                        if agent_id
                        else step.state.agent_type
                    )
                    target = plan["defaults"][agent_type]
                    step.state.provider_type, step.state.home_id = (
                        target["provider_type"],
                        target["home_id"],
                    )
        except Exception as exc:
            return {
                "complete": False,
                "receipts": receipts,
                "error_type": type(exc).__name__,
                "plan": plan_provider_switch(service, **args),
            }
        final = plan_provider_switch(service, **args)
        return {"complete": final["complete"], "receipts": receipts, "plan": final}
