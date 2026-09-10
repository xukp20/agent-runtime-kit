from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Event, RLock, Thread, current_thread
from types import SimpleNamespace
from typing import ClassVar

import pytest
from pydantic import BaseModel

from agent_runtime_kit.flow import (
    ActiveStepRun,
    AgentStep,
    AgentStepIncompleteResult,
    AgentStepState,
    BaseFlow,
    BaseFlowState,
    BaseStep,
    BaseStepError,
    BaseStepResult,
    BaseStepState,
    BaseSubmission,
    FlowBuildContext,
    FlowRequest,
    FlowService,
    FlowStatus,
    FlowStepValidationError,
    FlowStepMutationSession,
    FlowTypeRegistry,
    RuntimeScheduleService,
    StepRunContext,
    StepService,
    StepStatus,
    StepTerminalReceipt,
    StepTypeRegistry,
)
from agent_runtime_kit.runtime import ARKServices, AppServices, RuntimePauseController


class OperatorFlowParams(BaseModel):
    pass


class OperatorFlowState(BaseFlowState):
    state_type: str = "operator_flow_state"


class OperatorFlow(BaseFlow):
    flow_type: ClassVar[str] = "operator_flow"
    Params: ClassVar[type[BaseModel]] = OperatorFlowParams
    State: ClassVar[type[BaseFlowState]] = OperatorFlowState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "OperatorFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=OperatorFlowState())


class OperatorAgentStep(AgentStep):
    step_type: ClassVar[str] = "operator_agent_step"

    def run(self, ctx: StepRunContext) -> StepTerminalReceipt:
        return ctx.complete_step(AgentStepIncompleteResult(summary="operator step done"))


class PlainStep(BaseStep):
    step_type: ClassVar[str] = "plain_step"
    State: ClassVar[type[BaseStepState]] = BaseStepState

    def run(self, ctx: StepRunContext) -> StepTerminalReceipt:
        return ctx.complete_step(BaseStepResult(result_type="plain_step_done"))


class FakeAgentService:
    def __init__(self) -> None:
        self.lock = RLock()
        self.running_agent_ids: set[str] = set()
        self.agent_types = SimpleNamespace(
            get=lambda _agent_type: SimpleNamespace(
                render_start_prompt=lambda variables: f"Default task: {variables['task']}"
            )
        )

    @contextmanager
    def hold_agent_boundary(self):  # noqa: ANN201
        with self.lock:
            yield

    def list_running_agents(self, scope_id: str | None = None):  # noqa: ANN201
        del scope_id
        return [SimpleNamespace(agent_id=agent_id) for agent_id in sorted(self.running_agent_ids)]

    def get_agent(self, agent_id: str):  # noqa: ANN201
        return SimpleNamespace(agent_id=agent_id, agent_type="operator_agent")


def _service(
    runtime_root: Path,
) -> tuple[
    FlowService,
    StepService,
    RuntimeScheduleService,
    RuntimePauseController,
    FakeAgentService,
    str,
    str,
]:
    flow_registry = FlowTypeRegistry()
    flow_registry.register(OperatorFlow)
    step_registry = StepTypeRegistry()
    step_registry.register(OperatorAgentStep)
    step_registry.register(PlainStep)
    pause = RuntimePauseController(global_paused=True)
    agents = FakeAgentService()
    ark = ARKServices(agent_service=agents, pause_controller=pause)
    flow_service = FlowService(
        runtime_root,
        flow_registry=flow_registry,
        step_registry=step_registry,
        ark_services=ark,
        app_services=AppServices(),
    )
    step_service = StepService(
        runtime_root,
        step_registry=step_registry,
        ark_services=ark,
        app_services=AppServices(),
    )
    scheduler = RuntimeScheduleService(ark_services=ark, app_services=AppServices())
    flow_id = flow_service.start_flow(
        FlowRequest(flow_type="operator_flow", scope_id="scope", params={}),
        enqueue=False,
    )
    step = OperatorAgentStep(
        step_id="operator-step",
        flow_id=flow_id,
        scope_id="scope",
        state=AgentStepState(
            agent_role="operator",
            agent_type="operator_agent",
            variables={"task": "inspect current truth"},
            prompt_override="Original business prompt.",
        ),
    )
    flow_service.store.create_step(step)

    def attach(flow: BaseFlow) -> None:
        flow.status = FlowStatus.RUNNING
        flow.step_ids.append(step.step_id)
        flow.current_step_id = step.step_id

    flow_service.store.update_flow_record(flow_id, attach)
    return flow_service, step_service, scheduler, pause, agents, flow_id, step.step_id


def _set_instruction(
    service: FlowService,
    flow_id: str,
    step_id: str,
    instruction: str | None,
):  # noqa: ANN201
    flow = service.get_flow(flow_id)
    step = service.get_step(step_id)
    return service.set_agent_step_operator_instruction(
        step_id=step_id,
        expected_step_updated_at=step.updated_at,
        expected_flow_updated_at=flow.updated_at,
        instruction=instruction,
    )


def test_operator_instruction_is_appended_once_after_initial_prompt(tmp_path: Path) -> None:
    service, _, _, _, _, flow_id, step_id = _service(tmp_path / ".runtime")
    _set_instruction(service, flow_id, step_id, "Inspect exact current sibling truth.")
    step = service.get_step(step_id)

    prompt = step.build_start_prompt(
        StepRunContext(
            ark=service.ark,
            app=AppServices(),
            step_id=step_id,
            flow_id=flow_id,
            scope_id="scope",
        ),
        "operator-agent",
    )

    assert prompt is not None
    assert prompt.startswith("Original business prompt.")
    assert prompt.endswith("Inspect exact current sibling truth.")
    assert prompt.count("Inspect exact current sibling truth.") == 1


def test_operator_instruction_renders_default_start_prompt_when_initial_prompt_is_none(
    tmp_path: Path,
) -> None:
    service, _, _, _, _, flow_id, step_id = _service(tmp_path / ".runtime")
    service.store.update_step_record(
        step_id,
        lambda step: setattr(step.state, "prompt_override", None),
    )
    _set_instruction(service, flow_id, step_id, "Use the fresh source evidence.")
    step = service.get_step(step_id)

    prompt = step.build_start_prompt(
        StepRunContext(
            ark=service.ark,
            app=AppServices(),
            step_id=step_id,
            flow_id=flow_id,
            scope_id="scope",
        ),
        "operator-agent",
    )

    assert prompt is not None
    assert prompt.startswith("Default task: inspect current truth")
    assert prompt.endswith("Use the fresh source evidence.")


def test_none_operator_instruction_preserves_start_and_continue_prompts(tmp_path: Path) -> None:
    service, _, _, _, _, flow_id, step_id = _service(tmp_path / ".runtime")
    step = service.get_step(step_id)
    ctx = StepRunContext(
        ark=service.ark,
        app=AppServices(),
        step_id=step_id,
        flow_id=flow_id,
        scope_id="scope",
    )

    assert step.build_start_prompt(ctx, "operator-agent") == "Original business prompt."
    assert (
        step.build_continue_prompt(
            ctx,
            "operator-agent",
            object(),
            SimpleNamespace(continue_prompt=None, reason="submission missing"),
        )
        == "Continue the current task. The previous turn did not complete the Step: "
        "submission missing. Submit a valid result when the task is ready."
    )


def test_set_agent_step_operator_instruction_sets_and_clears_with_exact_cas(
    tmp_path: Path,
) -> None:
    from agent_runtime_kit.flow import SetAgentStepOperatorInstructionReceipt

    service, _, _, _, _, flow_id, step_id = _service(tmp_path / ".runtime")
    flow_before = service.get_flow(flow_id).model_dump(mode="json")
    step_before = service.get_step(step_id).model_dump(mode="json")
    flow_bytes_before = service.store.resolve_flow_path(flow_id).read_bytes()

    receipt = _set_instruction(service, flow_id, step_id, "Inspect exact current truth.")

    assert isinstance(receipt, SetAgentStepOperatorInstructionReceipt)
    assert receipt.model_dump(mode="json") == {
        "step_id": step_id,
        "flow_id": flow_id,
        "scope_id": "scope",
        "instruction_before": None,
        "instruction_after": "Inspect exact current truth.",
        "instruction_present": True,
        "step_updated_at_before": step_before["updated_at"],
        "step_updated_at_after": receipt.step_updated_at_after,
        "flow_updated_at_before": flow_before["updated_at"],
        "flow_updated_at_after": flow_before["updated_at"],
        "summary": receipt.summary,
    }
    assert receipt.step_updated_at_after != step_before["updated_at"]
    assert service.get_flow(flow_id).model_dump(mode="json") == flow_before
    assert service.store.resolve_flow_path(flow_id).read_bytes() == flow_bytes_before
    set_step = service.get_step(step_id)
    assert set_step.state.operator_instruction == "Inspect exact current truth."
    expected_step = dict(step_before)
    expected_step["state"] = dict(expected_step["state"])
    expected_step["state"]["operator_instruction"] = "Inspect exact current truth."
    expected_step["updated_at"] = receipt.step_updated_at_after
    assert set_step.model_dump(mode="json") == expected_step

    clear_receipt = _set_instruction(service, flow_id, step_id, None)

    assert clear_receipt.instruction_before == "Inspect exact current truth."
    assert clear_receipt.instruction_after is None
    assert clear_receipt.instruction_present is False
    assert clear_receipt.step_updated_at_before == receipt.step_updated_at_after
    assert clear_receipt.step_updated_at_after == service.get_step(step_id).updated_at
    assert clear_receipt.flow_updated_at_before == flow_before["updated_at"]
    assert clear_receipt.flow_updated_at_after == flow_before["updated_at"]
    assert service.get_flow(flow_id).model_dump(mode="json") == flow_before
    assert service.store.resolve_flow_path(flow_id).read_bytes() == flow_bytes_before
    assert service.get_step(step_id).state.operator_instruction is None


@pytest.mark.parametrize(
    "failure",
    [
        "blank",
        "stale_step",
        "stale_flow",
        "non_current",
        "not_created",
        "submission",
        "result",
        "error",
        "started_at",
        "finished_at",
        "unpaused",
        "active_target_step",
        "active_other_step",
        "running_agent",
        "active_target_flow",
        "active_other_flow",
        "terminal_flow",
        "non_agent_step",
    ],
)
def test_set_agent_step_operator_instruction_rejects_unsafe_boundary_without_mutation(
    tmp_path: Path,
    failure: str,
) -> None:
    service, step_service, scheduler, pause, agents, flow_id, step_id = _service(
        tmp_path / ".runtime"
    )
    expected_step = service.get_step(step_id).updated_at
    expected_flow = service.get_flow(flow_id).updated_at
    instruction = "Inspect exact current truth."
    if failure == "blank":
        instruction = " \t\n"
    elif failure == "stale_step":
        expected_step = "stale-step-cas"
    elif failure == "stale_flow":
        expected_flow = "stale-flow-cas"
    elif failure == "non_current":
        service.store.update_flow_record(flow_id, lambda flow: setattr(flow, "current_step_id", None))
    elif failure == "not_created":
        service.store.update_step_record(
            step_id,
            lambda step: setattr(step, "status", StepStatus.SUSPENDED),
        )
    elif failure == "submission":
        service.store.update_step_record(
            step_id,
            lambda step: setattr(
                step,
                "submission",
                BaseSubmission(
                    submission_id="submitted",
                    submission_type="result",
                    tool_name="submit",
                ),
            ),
        )
    elif failure == "result":
        service.store.update_step_record(
            step_id,
            lambda step: setattr(
                step,
                "result",
                AgentStepIncompleteResult(summary="already finished"),
            ),
        )
    elif failure == "error":
        service.store.update_step_record(
            step_id,
            lambda step: setattr(step, "error", BaseStepError(error_type="failed", message="failed")),
        )
    elif failure in {"started_at", "finished_at"}:
        service.store.update_step_record(
            step_id,
            lambda step: setattr(step, failure, "2026-09-10T00:00:00Z"),
        )
    elif failure == "unpaused":
        pause.resume()
    elif failure in {"active_target_step", "active_other_step"}:
        active_step_id = step_id if failure == "active_target_step" else "other-step"
        step_service.active_steps[active_step_id] = ActiveStepRun(
            step_id=active_step_id,
            flow_id=flow_id,
            scope_id="scope",
            started_at="2026-09-10T00:00:00Z",
        )
    elif failure == "running_agent":
        agents.running_agent_ids.add("other-agent")
    elif failure in {"active_target_flow", "active_other_flow"}:
        active_flow_id = flow_id
        if failure == "active_other_flow":
            active_flow_id = service.start_flow(
                FlowRequest(flow_type="operator_flow", scope_id="scope", params={}),
                enqueue=False,
            )
        scheduler.active_flow_advances.add(active_flow_id)
    elif failure == "terminal_flow":
        service.store.update_flow_record(
            flow_id,
            lambda flow: setattr(flow, "status", FlowStatus.COMPLETED),
        )
    elif failure == "non_agent_step":
        plain = PlainStep(
            step_id="plain-step",
            flow_id=flow_id,
            scope_id="scope",
        )
        service.store.create_step(plain)

        def make_plain_current(flow: BaseFlow) -> None:
            flow.step_ids.append(plain.step_id)
            flow.current_step_id = plain.step_id

        service.store.update_flow_record(flow_id, make_plain_current)
        step_id = plain.step_id

    if failure != "stale_step":
        expected_step = service.get_step(step_id).updated_at
    if failure != "stale_flow":
        expected_flow = service.get_flow(flow_id).updated_at

    before_flow = service.get_flow(flow_id).model_dump(mode="json")
    before_step = service.get_step(step_id).model_dump(mode="json")
    flow_bytes_before = service.store.resolve_flow_path(flow_id).read_bytes()
    step_bytes_before = service.store.resolve_step_path(step_id).read_bytes()

    with pytest.raises((FlowStepValidationError, RuntimeError), match=".+"):
        service.set_agent_step_operator_instruction(
            step_id=step_id,
            expected_step_updated_at=expected_step,
            expected_flow_updated_at=expected_flow,
            instruction=instruction,
        )

    assert service.get_flow(flow_id).model_dump(mode="json") == before_flow
    assert service.get_step(step_id).model_dump(mode="json") == before_step
    assert service.store.resolve_flow_path(flow_id).read_bytes() == flow_bytes_before
    assert service.store.resolve_step_path(step_id).read_bytes() == step_bytes_before


def test_agent_step_operator_instruction_round_trips_current_schema(tmp_path: Path) -> None:
    service, _, _, _, _, flow_id, step_id = _service(tmp_path / ".runtime")
    initial = service.get_step(step_id)
    assert initial.state.operator_instruction is None
    legacy_state = initial.state.model_dump(mode="json")
    legacy_state.pop("operator_instruction", None)
    assert AgentStepState.model_validate(legacy_state).operator_instruction is None

    _set_instruction(service, flow_id, step_id, "Persist this instruction.")
    assert service.get_step(step_id).state.operator_instruction == "Persist this instruction."
    _set_instruction(service, flow_id, step_id, None)
    assert service.get_step(step_id).state.operator_instruction is None


def test_operator_instruction_isolated_from_developer_and_continue_surfaces(tmp_path: Path) -> None:
    service, _, _, _, _, flow_id, step_id = _service(tmp_path / ".runtime")
    _set_instruction(service, flow_id, step_id, "Only the start prompt may carry this.")
    step = service.get_step(step_id)
    ctx = StepRunContext(
        ark=service.ark,
        app=AppServices(),
        step_id=step_id,
        flow_id=flow_id,
        scope_id="scope",
    )

    assert step.build_developer_instructions_override(ctx, "operator-agent") is None
    assert "operator_instruction" not in step.state.variables
    assert "Only the start prompt may carry this." not in step.build_continue_prompt(
        ctx,
        "operator-agent",
        object(),
        SimpleNamespace(continue_prompt=None, reason="retry"),
    )


def test_set_before_bypass_start_linearizes_and_completes_without_deadlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, step_service, _, _, _, flow_id, step_id = _service(tmp_path / ".runtime")
    rendezvous = Barrier(2)
    set_at_boundary = Event()
    release_set = Event()
    original = service._assert_scope_quiescent

    def hold_set_boundary(scope_id: str, **kwargs) -> None:  # noqa: ANN003
        original(scope_id, **kwargs)
        set_at_boundary.set()
        assert release_set.wait(2)

    monkeypatch.setattr(service, "_assert_scope_quiescent", hold_set_boundary)
    results: dict[str, object] = {}

    def setter() -> None:
        rendezvous.wait()
        results["set"] = _set_instruction(service, flow_id, step_id, "Linearized first.")

    def starter() -> None:
        rendezvous.wait()
        assert set_at_boundary.wait(2)
        results["start"] = step_service.start_step(step_id, bypass_pause=True)

    set_thread = Thread(target=setter)
    start_thread = Thread(target=starter)
    set_thread.start()
    start_thread.start()
    assert set_at_boundary.wait(2)
    start_thread.join(0.05)
    assert start_thread.is_alive()
    release_set.set()
    set_thread.join(2)
    start_thread.join(2)

    assert not set_thread.is_alive() and not start_thread.is_alive()
    assert "set" in results and "start" in results
    assert step_service.wait_step(step_id, timeout_s=2).status is StepStatus.COMPLETED
    assert service.get_step(step_id).state.operator_instruction == "Linearized first."


def test_bypass_start_before_set_rejects_mutation_without_deadlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, step_service, _, _, _, flow_id, step_id = _service(tmp_path / ".runtime")
    rendezvous = Barrier(2)
    start_at_boundary = Event()
    release_start = Event()
    original = step_service.can_run_step

    def hold_start_boundary(candidate: str) -> bool:
        start_at_boundary.set()
        assert release_start.wait(2)
        return original(candidate)

    monkeypatch.setattr(step_service, "can_run_step", hold_start_boundary)
    results: dict[str, object] = {}

    def starter() -> None:
        rendezvous.wait()
        results["start"] = step_service.start_step(step_id, bypass_pause=True)

    def setter() -> None:
        rendezvous.wait()
        assert start_at_boundary.wait(2)
        try:
            results["set"] = _set_instruction(service, flow_id, step_id, "Too late.")
        except Exception as exc:  # noqa: BLE001 - asserted concurrent outcome.
            results["set_error"] = exc

    start_thread = Thread(target=starter)
    set_thread = Thread(target=setter)
    start_thread.start()
    set_thread.start()
    assert start_at_boundary.wait(2)
    set_thread.join(0.05)
    assert set_thread.is_alive()
    release_start.set()
    start_thread.join(2)
    set_thread.join(2)

    assert not start_thread.is_alive() and not set_thread.is_alive()
    assert "start" in results
    assert isinstance(results.get("set_error"), FlowStepValidationError)
    assert step_service.wait_step(step_id, timeout_s=2).status is StepStatus.COMPLETED
    assert service.get_step(step_id).state.operator_instruction is None


def test_concurrent_sets_keep_each_receipt_owned_by_its_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_runtime_kit.flow.store as flow_store_module

    service, _, _, _, _, flow_id, step_id = _service(tmp_path / ".runtime")
    first_step_updated_at = service.get_step(step_id).updated_at
    flow_updated_at = service.get_flow(flow_id).updated_at
    first_flushed = Event()
    second_done = Event()
    original_flush = FlowStepMutationSession.flush

    def observed_flush(session: FlowStepMutationSession) -> None:
        original_flush(session)
        if current_thread().name == "first-setter":
            first_flushed.set()

    monkeypatch.setattr(FlowStepMutationSession, "flush", observed_flush)
    timestamps = iter(["2026-09-10T14:00:01Z", "2026-09-10T14:00:02Z"])
    monkeypatch.setattr(flow_store_module, "utc_now_iso", lambda: next(timestamps))

    original_get_step = service.store.get_step
    first_get_count = 0

    def controlled_get_step(candidate: str):  # noqa: ANN201
        nonlocal first_get_count
        if current_thread().name == "first-setter":
            first_get_count += 1
            if first_get_count == 3:
                assert second_done.wait(2)
        return original_get_step(candidate)

    monkeypatch.setattr(service.store, "get_step", controlled_get_step)
    receipts: dict[str, object] = {}

    def first_setter() -> None:
        receipts["first"] = service.set_agent_step_operator_instruction(
            step_id=step_id,
            expected_step_updated_at=first_step_updated_at,
            expected_flow_updated_at=flow_updated_at,
            instruction="First instruction.",
        )

    def second_setter() -> None:
        assert first_flushed.wait(2)
        receipts["second"] = _set_instruction(
            service,
            flow_id,
            step_id,
            "Second instruction.",
        )
        second_done.set()

    first_thread = Thread(target=first_setter, name="first-setter")
    second_thread = Thread(target=second_setter, name="second-setter")
    first_thread.start()
    second_thread.start()
    first_thread.join(2)
    second_thread.join(2)

    assert not first_thread.is_alive() and not second_thread.is_alive()
    first = receipts["first"]
    second = receipts["second"]
    assert first.instruction_after == "First instruction."
    assert first.step_updated_at_after == "2026-09-10T14:00:01Z"
    assert second.instruction_before == "First instruction."
    assert second.instruction_after == "Second instruction."
    assert second.step_updated_at_after == "2026-09-10T14:00:02Z"
    assert service.get_step(step_id).state.operator_instruction == "Second instruction."
