from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime_kit.flow import ChildFlowDispatchSubmission, FlowRequest, FlowStatus
from agent_runtime_kit.flow.standard_steps import AgentStep, DispatchStep, DispatchStepResult

from .support import (
    RealLogicChildResult,
    RealTerminalHandoffParentResult,
    RealTerminalHandoffParentState,
    make_real_flow_runtime,
    run_flow_until_terminal,
    run_scheduler_until_idle,
)


pytestmark = pytest.mark.real_codex


def test_real_dispatch_terminal_handoff_completes_parent_without_callback(tmp_path: Path) -> None:
    runtime = make_real_flow_runtime(tmp_path)
    try:
        flow_id = runtime.flow_service.start_flow(
            FlowRequest(
                flow_type="real_terminal_handoff_parent",
                scope_id="scope:rt-handoff",
                params={"names_csv": "handoff-alpha", "child_flow_type": "real_logic_child"},
            ),
            enqueue=True,
        )

        parent = run_flow_until_terminal(runtime, flow_id)
        parent_steps = runtime.flow_service.list_steps(flow_id=flow_id)
        child_flows = runtime.flow_service.store.list_child_flows(parent_flow_id=flow_id)
        run_scheduler_until_idle(runtime)
        completed_children = [runtime.flow_service.get_flow(child.flow_id) for child in child_flows]
    finally:
        runtime.close()

    assert parent.status is FlowStatus.COMPLETED
    assert isinstance(parent.result, RealTerminalHandoffParentResult)
    assert isinstance(parent.state, RealTerminalHandoffParentState)
    assert len(child_flows) == 1
    assert parent.result.child_flow_ids == [child_flows[0].flow_id]
    assert parent.state.handoff_child_flow_ids == [child_flows[0].flow_id]

    agent_steps = [step for step in parent_steps if isinstance(step, AgentStep)]
    dispatch_steps = [step for step in parent_steps if isinstance(step, DispatchStep)]
    assert len(agent_steps) == 1
    assert len(dispatch_steps) == 1

    initial_step = agent_steps[0]
    assert isinstance(initial_step.submission, ChildFlowDispatchSubmission)
    assert initial_step.submission.continuation == "terminal_handoff"

    dispatch_step = dispatch_steps[0]
    assert isinstance(dispatch_step.result, DispatchStepResult)
    assert dispatch_step.result.outcome == "dispatched"
    assert dispatch_step.result.continuation == "terminal_handoff"
    assert parent.state.handoff_dispatch_step_id == dispatch_step.step_id

    assert child_flows[0].parent_flow_id == flow_id
    assert child_flows[0].parent_dispatch_step_id == dispatch_step.step_id
    assert completed_children[0].status is FlowStatus.COMPLETED
    assert isinstance(completed_children[0].result, RealLogicChildResult)
    assert completed_children[0].result.name == "handoff-alpha"

    handoff_calls = [
        call
        for call in runtime.submit_bridge.call_log
        if call["tool_name"] == "ark_submit_child_flows"
    ]
    assert len(handoff_calls) == 1
    assert handoff_calls[0]["args"]["continuation"] == "terminal_handoff"
    assert not any(getattr(step.state, "prompt_mode", None) == "callback" for step in agent_steps)
