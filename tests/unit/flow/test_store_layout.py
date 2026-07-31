from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from agent_runtime_kit.agent.store_utils import encode_scope_id, read_json, write_json_atomic
from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowState,
    BaseStep,
    BaseStepState,
    FlowStatus,
    FlowStepStore,
    FlowTypeRegistry,
    StepStatus,
    StepTypeRegistry,
)


class StoreFlowParams(BaseModel):
    target: str


class StoreFlowState(BaseFlowState):
    state_type: str = "store_flow_state"
    marker: str = "flow-marker"


class StoreFlow(BaseFlow):
    flow_type: ClassVar[str] = "store_flow"
    Params: ClassVar[type[BaseModel]] = StoreFlowParams
    State: ClassVar[type[BaseFlowState]] = StoreFlowState

    @classmethod
    def build_from_request(cls, ctx: object) -> "StoreFlow":
        raise NotImplementedError


class StoreStepState(BaseStepState):
    state_type: str = "store_step_state"
    marker: str = "step-marker"


class StoreStep(BaseStep):
    step_type: ClassVar[str] = "store_step"
    State: ClassVar[type[BaseStepState]] = StoreStepState


def make_store(runtime_root: Path) -> FlowStepStore:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(StoreFlow)
    step_registry.register(StoreStep)
    return FlowStepStore(runtime_root, flow_registry=flow_registry, step_registry=step_registry)


def test_store_writes_flow_json_in_scope_flow_directory(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    flow = StoreFlow(flow_id="flow-1", scope_id="repo:Main", state=StoreFlowState(marker="custom-flow"))

    store.create_flow(flow)

    flow_json = tmp_path / ".agent_runtime" / "scopes" / encode_scope_id("repo:Main") / "flows" / "flow-1" / "flow.json"
    payload = read_json(flow_json)
    assert payload["object_type"] == "flow"
    assert payload["flow_type"] == "store_flow"
    assert payload["state"]["marker"] == "custom-flow"
    assert payload["status"] == "created"


def test_store_writes_step_json_under_owning_flow_directory(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    flow = StoreFlow(flow_id="flow-1", scope_id="repo:Main")
    step = StoreStep(
        step_id="step-1",
        flow_id="flow-1",
        scope_id="repo:Main",
        state=StoreStepState(marker="custom-step"),
    )

    store.create_flow(flow)
    store.create_step(step)

    step_json = (
        tmp_path
        / ".agent_runtime"
        / "scopes"
        / encode_scope_id("repo:Main")
        / "flows"
        / "flow-1"
        / "steps"
        / "step-1"
        / "step.json"
    )
    payload = read_json(step_json)
    assert payload["object_type"] == "step"
    assert payload["step_type"] == "store_step"
    assert payload["state"]["marker"] == "custom-step"


def test_store_reads_truth_through_registered_types(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    flow = StoreFlow(flow_id="flow-1", scope_id="scope", state=StoreFlowState(marker="restored-flow"))
    step = StoreStep(
        step_id="step-1",
        flow_id="flow-1",
        scope_id="scope",
        state=StoreStepState(marker="restored-step"),
        status=StepStatus.CREATED,
    )

    store.create_flow(flow)
    store.create_step(step)

    restored_flow = store.get_flow("flow-1")
    restored_step = store.get_step("step-1")

    assert isinstance(restored_flow, StoreFlow)
    assert isinstance(restored_flow.state, StoreFlowState)
    assert restored_flow.state.marker == "restored-flow"
    assert restored_flow.status is FlowStatus.CREATED
    assert isinstance(restored_step, StoreStep)
    assert isinstance(restored_step.state, StoreStepState)
    assert restored_step.state.marker == "restored-step"


@pytest.mark.parametrize("observed_version", [None, 7])
def test_store_warns_but_reads_compatible_envelope_version(
    tmp_path: Path,
    caplog,
    observed_version: int | None,
) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    flow = StoreFlow(flow_id="flow-1", scope_id="scope")
    store.create_flow(flow)
    flow_path = (
        tmp_path
        / ".agent_runtime"
        / "scopes"
        / encode_scope_id("scope")
        / "flows"
        / "flow-1"
        / "flow.json"
    )
    payload = read_json(flow_path)
    if observed_version is None:
        payload.pop("schema_version")
    else:
        payload["schema_version"] = observed_version
    write_json_atomic(flow_path, payload)

    restored = store.get_flow("flow-1")

    assert restored.flow_id == "flow-1"
    assert "schema version differs" in caplog.text


def test_store_warns_but_reads_compatible_step_envelope_version(
    tmp_path: Path,
    caplog,
) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    flow = StoreFlow(flow_id="flow-1", scope_id="scope")
    step = StoreStep(step_id="step-1", flow_id="flow-1", scope_id="scope")
    store.create_flow(flow)
    store.create_step(step)
    step_path = (
        tmp_path
        / ".agent_runtime"
        / "scopes"
        / encode_scope_id("scope")
        / "flows"
        / "flow-1"
        / "steps"
        / "step-1"
        / "step.json"
    )
    payload = read_json(step_path)
    payload["schema_version"] = 7
    write_json_atomic(step_path, payload)

    restored = store.get_step("step-1")

    assert restored.step_id == "step-1"
    assert "schema version differs" in caplog.text
