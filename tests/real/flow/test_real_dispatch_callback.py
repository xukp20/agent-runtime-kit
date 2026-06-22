from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime_kit.flow import ChildFlowDispatchSubmission, FlowRequest, FlowStatus, StepStatus
from agent_runtime_kit.flow.standard_steps import AgentStep, AgentStepSubmissionResult, DispatchStep, DispatchStepResult

from .support import make_real_flow_runtime, run_flow_until_terminal


pytestmark = pytest.mark.real_codex


def test_real_dispatch_callback_with_logic_child(tmp_path: Path) -> None:
    runtime = make_real_flow_runtime(tmp_path)
    try:
        flow_id = runtime.flow_service.start_flow(
            FlowRequest(
                flow_type="real_dispatch_parent",
                scope_id="scope:rt03",
                params={"names_csv": "alpha,beta", "child_flow_type": "real_logic_child"},
            ),
            enqueue=True,
        )

        flow = run_flow_until_terminal(runtime, flow_id)
        parent_steps = runtime.flow_service.list_steps(flow_id=flow_id)
        child_flows = runtime.flow_service.store.list_child_flows(parent_flow_id=flow_id)
    finally:
        runtime.close()

    assert flow.status is FlowStatus.COMPLETED
    assert flow.result is not None
    callback_summary = getattr(flow.result, "callback_summary", None)
    assert isinstance(callback_summary, str)
    assert callback_summary
    assert "alpha" in callback_summary or "beta" in callback_summary or "child" in callback_summary

    agent_steps = [step for step in parent_steps if isinstance(step, AgentStep)]
    dispatch_steps = [step for step in parent_steps if isinstance(step, DispatchStep)]
    assert len(agent_steps) == 2
    assert len(dispatch_steps) == 1

    initial_step, callback_step = agent_steps
    assert isinstance(initial_step.submission, ChildFlowDispatchSubmission)
    assert initial_step.submission.requests
    assert initial_step.agent_bindings.get("planner") == callback_step.agent_bindings.get("planner")
    assert getattr(callback_step.state, "prompt_mode", None) == "callback"
    assert isinstance(callback_step.result, AgentStepSubmissionResult)

    dispatch_step = dispatch_steps[0]
    assert isinstance(dispatch_step.result, DispatchStepResult)
    assert dispatch_step.result.outcome == "dispatched"
    assert len(dispatch_step.result.child_flow_ids) == 2
    assert len(child_flows) == 2
    assert {child.parent_flow_id for child in child_flows} == {flow_id}
    assert {child.parent_dispatch_step_id for child in child_flows} == {dispatch_step.step_id}
    assert {child.status for child in child_flows} == {FlowStatus.COMPLETED}
    assert {getattr(child.result, "name", None) for child in child_flows} == {"alpha", "beta"}
    assert any(call["tool_name"] == "ark_submit_child_flows" for call in runtime.submit_bridge.call_log)
    assert any(call["tool_name"] == "ark_submit_result" for call in runtime.submit_bridge.call_log)


@pytest.mark.slow
def test_real_dispatch_callback_with_agent_child(tmp_path: Path) -> None:
    runtime = make_real_flow_runtime(tmp_path)
    try:
        flow_id = runtime.flow_service.start_flow(
            FlowRequest(
                flow_type="real_dispatch_parent",
                scope_id="scope:rt04",
                params={"names_csv": "gamma", "child_flow_type": "real_agent_child"},
            ),
            enqueue=True,
        )

        flow = run_flow_until_terminal(runtime, flow_id, max_ticks=150)
        parent_steps = runtime.flow_service.list_steps(flow_id=flow_id)
        child_flows = runtime.flow_service.store.list_child_flows(parent_flow_id=flow_id)
        child_steps = [
            step
            for child in child_flows
            for step in runtime.flow_service.list_steps(flow_id=child.flow_id)
        ]
    finally:
        runtime.close()

    assert flow.status is FlowStatus.COMPLETED
    assert len(child_flows) == 1
    child = child_flows[0]
    assert child.status is FlowStatus.COMPLETED
    assert getattr(child.result, "name", None) == "gamma"
    assert child_steps
    assert all(step.status is StepStatus.COMPLETED for step in child_steps)
    assert any(isinstance(step, DispatchStep) for step in parent_steps)
    assert len([call for call in runtime.submit_bridge.call_log if call["tool_name"] == "ark_submit_result"]) >= 2
