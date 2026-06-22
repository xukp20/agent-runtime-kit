from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowState,
    BaseStep,
    BaseStepState,
    FlowStatus,
    FlowStepStore,
    FlowStepTypeError,
    FlowTypeRegistry,
    StepStatus,
    StepTypeRegistry,
)


class IndexedFlowParams(BaseModel):
    target: str = "target"


class IndexedFlowState(BaseFlowState):
    state_type: str = "indexed_flow_state"


class IndexedFlow(BaseFlow):
    flow_type: ClassVar[str] = "indexed_flow"
    Params: ClassVar[type[BaseModel]] = IndexedFlowParams
    State: ClassVar[type[BaseFlowState]] = IndexedFlowState

    @classmethod
    def build_from_request(cls, ctx: object) -> "IndexedFlow":
        raise NotImplementedError


class IndexedStepState(BaseStepState):
    state_type: str = "indexed_step_state"


class IndexedStep(BaseStep):
    step_type: ClassVar[str] = "indexed_step"
    State: ClassVar[type[BaseStepState]] = IndexedStepState


def make_store(runtime_root: Path) -> FlowStepStore:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(IndexedFlow)
    step_registry.register(IndexedStep)
    return FlowStepStore(runtime_root, flow_registry=flow_registry, step_registry=step_registry)


def seed_store(store: FlowStepStore) -> None:
    parent = IndexedFlow(flow_id="flow-parent", scope_id="scope", status=FlowStatus.RUNNING)
    child = IndexedFlow(
        flow_id="flow-child",
        scope_id="scope",
        parent_flow_id="flow-parent",
        parent_dispatch_step_id="dispatch-1",
    )
    step = IndexedStep(
        step_id="step-1",
        flow_id="flow-parent",
        scope_id="scope",
        status=StepStatus.CREATED,
    )
    store.create_flow(parent)
    store.create_flow(child)
    store.create_step(step)


def test_store_lists_by_scope_status_type_and_parent(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    seed_store(store)

    assert [flow.flow_id for flow in store.list_flows(scope_id="scope")] == ["flow-parent", "flow-child"]
    assert [flow.flow_id for flow in store.list_flows(status=FlowStatus.RUNNING)] == ["flow-parent"]
    assert [flow.flow_id for flow in store.list_flows(flow_type="indexed_flow")] == ["flow-parent", "flow-child"]
    assert [step.step_id for step in store.list_steps(flow_id="flow-parent")] == ["step-1"]
    assert [step.step_id for step in store.list_steps(status=StepStatus.CREATED)] == ["step-1"]
    assert [flow.flow_id for flow in store.list_child_flows(parent_flow_id="flow-parent")] == ["flow-child"]
    assert [
        flow.flow_id
        for flow in store.list_child_flows(parent_flow_id="flow-parent", parent_dispatch_step_id="dispatch-1")
    ] == ["flow-child"]


def test_store_rebuilds_indexes_from_json_truth(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    store = make_store(runtime_root)
    seed_store(store)

    (runtime_root / "index" / "global.sqlite").unlink()
    for scope_index in (runtime_root / "scopes").glob("*/index.sqlite"):
        scope_index.unlink()

    rebuilt = make_store(runtime_root)
    rebuilt.rebuild_all_indexes()

    assert [flow.flow_id for flow in rebuilt.list_flows(scope_id="scope")] == ["flow-parent", "flow-child"]
    assert [step.step_id for step in rebuilt.list_steps(scope_id="scope")] == ["step-1"]


def test_store_unknown_flow_type_raises_registry_error(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    store = make_store(runtime_root)
    seed_store(store)
    flow_path = store.resolve_flow_path("flow-parent")
    payload = flow_path.read_text(encoding="utf-8").replace('"flow_type": "indexed_flow"', '"flow_type": "missing_flow"')
    flow_path.write_text(payload, encoding="utf-8")

    with pytest.raises(FlowStepTypeError):
        store.get_flow("flow-parent")
