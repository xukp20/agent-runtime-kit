from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from agent_runtime_kit.agent.store_utils import read_json
from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowState,
    BaseStep,
    BaseStepState,
    BaseSubmission,
    FlowStatus,
    FlowStepStore,
    FlowTypeRegistry,
    StepStatus,
    StepTypeRegistry,
)


class TxFlowParams(BaseModel):
    target: str = "target"


class TxFlowState(BaseFlowState):
    state_type: str = "tx_flow_state"
    marker: str = "initial"


class TxFlow(BaseFlow):
    flow_type: ClassVar[str] = "tx_flow"
    Params: ClassVar[type[BaseModel]] = TxFlowParams
    State: ClassVar[type[BaseFlowState]] = TxFlowState

    @classmethod
    def build_from_request(cls, ctx: object) -> "TxFlow":
        raise NotImplementedError


class TxStepState(BaseStepState):
    state_type: str = "tx_step_state"
    marker: str = "initial"


class TxStep(BaseStep):
    step_type: ClassVar[str] = "tx_step"
    State: ClassVar[type[BaseStepState]] = TxStepState


def make_store(runtime_root: Path) -> FlowStepStore:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(TxFlow)
    step_registry.register(TxStep)
    return FlowStepStore(runtime_root, flow_registry=flow_registry, step_registry=step_registry)


def seed_store(store: FlowStepStore) -> None:
    store.create_flow(TxFlow(flow_id="flow-1", scope_id="scope", state=TxFlowState()))
    store.create_step(
        TxStep(
            step_id="step-1",
            flow_id="flow-1",
            scope_id="scope",
            state=TxStepState(),
            status=StepStatus.RUNNING,
        )
    )


def test_short_transaction_update_step_writes_submission_without_losing_state(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    seed_store(store)

    def write_submission(step: BaseStep) -> None:
        assert isinstance(step.state, TxStepState)
        step.state.marker = "after-submit"
        step.submission = BaseSubmission(
            submission_id="sub-1",
            submission_type="result",
            tool_name="submit_ready",
        )

    updated = store.update_step_record("step-1", write_submission)

    assert isinstance(updated.state, TxStepState)
    assert updated.state.marker == "after-submit"
    assert updated.submission is not None
    payload = read_json(store.resolve_step_path("step-1"))
    assert payload["submission"]["submission_id"] == "sub-1"
    assert payload["state"]["marker"] == "after-submit"


def test_mutation_session_persists_flow_state_on_normal_exit(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    seed_store(store)

    with store.edit_session("scope") as tx:
        flow = tx.load_flow_for_update("flow-1")
        assert isinstance(flow.state, TxFlowState)
        flow.state.marker = "changed"

    restored = store.get_flow("flow-1")
    assert isinstance(restored.state, TxFlowState)
    assert restored.state.marker == "changed"


def test_mutation_session_creates_step_and_updates_flow_history(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    seed_store(store)

    with store.edit_session("scope") as tx:
        flow = tx.load_flow_for_update("flow-1")
        step = TxStep(step_id="step-2", flow_id="flow-1", scope_id="scope", state=TxStepState(marker="new"))
        tx.add_step(step)
        flow.step_ids.append(step.step_id)
        flow.current_step_id = step.step_id

    restored_flow = store.get_flow("flow-1")
    restored_step = store.get_step("step-2")
    assert restored_flow.step_ids == ["step-2"]
    assert restored_flow.current_step_id == "step-2"
    assert isinstance(restored_step.state, TxStepState)
    assert restored_step.state.marker == "new"


def test_mutation_session_discards_changes_on_exception(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    seed_store(store)

    with pytest.raises(RuntimeError):
        with store.edit_session("scope") as tx:
            flow = tx.load_flow_for_update("flow-1")
            assert isinstance(flow.state, TxFlowState)
            flow.state.marker = "should-not-persist"
            raise RuntimeError("abort")

    restored = store.get_flow("flow-1")
    assert isinstance(restored.state, TxFlowState)
    assert restored.state.marker == "initial"


def test_consecutive_short_transactions_preserve_previous_fields(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    seed_store(store)

    store.update_step_record(
        "step-1",
        lambda step: setattr(
            step,
            "submission",
            BaseSubmission(submission_id="sub-1", submission_type="result", tool_name="submit_ready"),
        ),
    )
    store.update_step_record("step-1", lambda step: setattr(step.state, "marker", "second-update"))

    restored = store.get_step("step-1")
    assert restored.submission is not None
    assert restored.submission.submission_id == "sub-1"
    assert isinstance(restored.state, TxStepState)
    assert restored.state.marker == "second-update"


def test_terminal_query_helpers_filter_statuses(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    store.create_flow(TxFlow(flow_id="created", scope_id="scope", status=FlowStatus.CREATED))
    store.create_flow(TxFlow(flow_id="waiting", scope_id="scope", status=FlowStatus.WAITING))
    store.create_flow(TxFlow(flow_id="completed", scope_id="scope", status=FlowStatus.COMPLETED))
    store.create_flow(TxFlow(flow_id="failed", scope_id="scope", status=FlowStatus.FAILED))
    store.create_step(TxStep(step_id="created-step", flow_id="created", scope_id="scope", status=StepStatus.CREATED))
    store.create_step(TxStep(step_id="running-step", flow_id="created", scope_id="scope", status=StepStatus.RUNNING))

    assert [flow.flow_id for flow in store.list_non_terminal_flows(scope_id="scope")] == ["created", "waiting"]
    assert [step.step_id for step in store.list_created_steps(scope_id="scope")] == ["created-step"]
