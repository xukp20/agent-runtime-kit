from typing import ClassVar

from pydantic import BaseModel

from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowResult,
    BaseFlowState,
    BaseStep,
    BaseStepResult,
    BaseStepState,
    BaseSubmission,
    ChildFlowDispatchSubmission,
    FlowRequest,
    FlowStepTypeError,
    FlowStepValidationError,
    FlowTypeRegistry,
    StepTypeRegistry,
)


class DemoFlowParams(BaseModel):
    target: str
    attempts: int = 1


class DemoFlowState(BaseFlowState):
    state_type: str = "demo_flow_state"


class DemoFlow(BaseFlow):
    flow_type: ClassVar[str] = "demo_flow"
    Params: ClassVar[type[BaseModel]] = DemoFlowParams
    State: ClassVar[type[BaseFlowState]] = DemoFlowState

    @classmethod
    def build_from_request(cls, ctx: object) -> "DemoFlow":
        raise NotImplementedError


class DemoStepState(BaseStepState):
    state_type: str = "demo_step_state"


class DemoStep(BaseStep):
    step_type: ClassVar[str] = "demo_step"
    State: ClassVar[type[BaseStepState]] = DemoStepState


class DemoStepSuccessResult(BaseStepResult):
    result_type: str = "demo_success"
    value: int


class DemoStepBlockedResult(BaseStepResult):
    result_type: str = "demo_blocked"
    reason: str


class DemoSubmission(BaseSubmission):
    submission_type: str = "demo_submit"
    value: int


class PolymorphicStep(BaseStep):
    step_type: ClassVar[str] = "polymorphic_step"
    State: ClassVar[type[BaseStepState]] = DemoStepState
    Result: ClassVar[type[BaseStepResult]] = DemoStepSuccessResult
    Results: ClassVar[dict[str, type[BaseStepResult]]] = {
        "demo_success": DemoStepSuccessResult,
        "demo_blocked": DemoStepBlockedResult,
    }
    Submission: ClassVar[type[BaseSubmission] | None] = None
    Submissions: ClassVar[dict[str, type[BaseSubmission]]] = {
        "demo_submit": DemoSubmission,
        "child_flow_dispatch": ChildFlowDispatchSubmission,
    }


class DemoFlowDoneResult(BaseFlowResult):
    result_type: str = "flow_done"
    value: str


class DemoFlowBlockedResult(BaseFlowResult):
    result_type: str = "flow_blocked"
    reason: str


class PolymorphicFlow(DemoFlow):
    flow_type: ClassVar[str] = "polymorphic_flow"
    Result: ClassVar[type[BaseFlowResult]] = DemoFlowDoneResult
    Results: ClassVar[dict[str, type[BaseFlowResult]]] = {
        "flow_done": DemoFlowDoneResult,
        "flow_blocked": DemoFlowBlockedResult,
    }


def test_flow_registry_registers_gets_and_lists_flow_types() -> None:
    registry = FlowTypeRegistry()
    registry.register(DemoFlow)

    assert registry.get("demo_flow") is DemoFlow
    assert registry.list() == ["demo_flow"]
    assert registry.can_parse_state("demo_flow", "demo_flow_state") is True
    assert registry.can_parse_state("demo_flow", "other") is False


def test_flow_registry_validates_request_params_with_declared_params_model() -> None:
    registry = FlowTypeRegistry()
    registry.register(DemoFlow)

    params = registry.validate_request_params(FlowRequest(flow_type="demo_flow", scope_id="scope", params={"target": "T"}))

    assert isinstance(params, DemoFlowParams)
    assert params.target == "T"
    assert params.attempts == 1


def test_flow_registry_rejects_duplicate_and_unknown_types() -> None:
    registry = FlowTypeRegistry()
    registry.register(DemoFlow)

    try:
        registry.register(DemoFlow)
    except FlowStepTypeError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate flow type was accepted")

    try:
        registry.get("missing")
    except FlowStepTypeError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing flow type was returned")


def test_flow_registry_rejects_invalid_params() -> None:
    registry = FlowTypeRegistry()
    registry.register(DemoFlow)

    try:
        registry.validate_request_params(FlowRequest(flow_type="demo_flow", scope_id="scope", params={}))
    except FlowStepValidationError as exc:
        assert "demo_flow" in str(exc)
    else:
        raise AssertionError("invalid flow params were accepted")


def test_flow_registry_rejects_missing_protocol_fields() -> None:
    class NoParamsFlow(BaseFlow):
        flow_type: ClassVar[str] = "no_params"
        State: ClassVar[type[BaseFlowState]] = DemoFlowState

        @classmethod
        def build_from_request(cls, ctx: object) -> "NoParamsFlow":
            raise NotImplementedError

    registry = FlowTypeRegistry()

    try:
        registry.register(NoParamsFlow)
    except FlowStepTypeError as exc:
        assert "Params" in str(exc)
    else:
        raise AssertionError("flow without Params was accepted")


def test_step_registry_registers_gets_and_lists_step_types() -> None:
    registry = StepTypeRegistry()
    registry.register(DemoStep)

    assert registry.get("demo_step") is DemoStep
    assert registry.list() == ["demo_step"]
    assert registry.can_parse_state("demo_step", "demo_step_state") is True
    assert registry.can_parse_state("demo_step", "other") is False


def test_step_registry_rejects_duplicate_and_missing_state() -> None:
    class NoStateStep(BaseStep):
        step_type: ClassVar[str] = "no_state"

    registry = StepTypeRegistry()
    registry.register(DemoStep)

    try:
        registry.register(DemoStep)
    except FlowStepTypeError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate step type was accepted")

    try:
        registry.register(NoStateStep)
    except FlowStepTypeError as exc:
        assert "State" in str(exc)
    else:
        raise AssertionError("step without State was accepted")


def test_step_registry_parses_result_subclasses_by_result_type() -> None:
    registry = StepTypeRegistry()
    registry.register(PolymorphicStep)

    success = registry.parse_result("polymorphic_step", {"result_type": "demo_success", "value": 3})
    blocked = registry.parse_result("polymorphic_step", {"result_type": "demo_blocked", "reason": "need resource"})

    assert isinstance(success, DemoStepSuccessResult)
    assert success.value == 3
    assert isinstance(blocked, DemoStepBlockedResult)
    assert blocked.reason == "need resource"


def test_flow_registry_parses_result_subclasses_by_result_type() -> None:
    registry = FlowTypeRegistry()
    registry.register(PolymorphicFlow)

    done = registry.parse_result("polymorphic_flow", {"result_type": "flow_done", "value": "ok"})
    blocked = registry.parse_result("polymorphic_flow", {"result_type": "flow_blocked", "reason": "waiting"})

    assert isinstance(done, DemoFlowDoneResult)
    assert done.value == "ok"
    assert isinstance(blocked, DemoFlowBlockedResult)
    assert blocked.reason == "waiting"


def test_step_registry_parses_submission_subclasses_by_submission_type() -> None:
    registry = StepTypeRegistry()
    registry.register(PolymorphicStep)

    submission = registry.parse_submission(
        "polymorphic_step",
        {"submission_id": "sub-1", "submission_type": "demo_submit", "tool_name": "submit_demo", "value": 5},
    )
    dispatch = registry.parse_submission(
        "polymorphic_step",
        {
            "submission_id": "sub-2",
            "submission_type": "child_flow_dispatch",
            "tool_name": "submit_children",
            "requests": [{"flow_type": "demo_flow", "scope_id": "scope", "params": {"target": "T"}}],
        },
    )

    assert isinstance(submission, DemoSubmission)
    assert submission.value == 5
    assert isinstance(dispatch, ChildFlowDispatchSubmission)
    assert dispatch.requests[0].flow_type == "demo_flow"


def test_step_registry_rejects_unknown_submission_type_when_no_fallback_declared() -> None:
    registry = StepTypeRegistry()
    registry.register(PolymorphicStep)

    try:
        registry.parse_submission(
            "polymorphic_step",
            {"submission_id": "sub-1", "submission_type": "unknown", "tool_name": "submit_unknown"},
        )
    except FlowStepTypeError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown submission type was accepted")
