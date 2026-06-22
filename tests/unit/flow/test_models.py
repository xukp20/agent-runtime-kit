from pydantic import ValidationError

from agent_runtime_kit.flow import (
    AgentRoleBindings,
    BaseFlow,
    BaseStep,
    ChildFlowDispatchSubmission,
    FlowRequest,
    FlowStatus,
    ManualPauseState,
    StepStatus,
    StepTerminalReceipt,
)


def test_status_enum_values_match_design() -> None:
    assert [item.value for item in StepStatus] == ["created", "running", "completed", "failed"]
    assert [item.value for item in FlowStatus] == ["created", "running", "waiting", "completed", "failed"]


def test_base_flow_and_step_minimal_objects_can_be_constructed() -> None:
    flow = BaseFlow(flow_id="flow-1", flow_type="test_flow", scope_id="scope")
    step = BaseStep(step_id="step-1", flow_id=flow.flow_id, step_type="test_step", scope_id=flow.scope_id)

    assert flow.status is FlowStatus.CREATED
    assert flow.input is None
    assert flow.state.position.phase == "initial"
    assert flow.state.position.round_index == 0
    assert flow.step_ids == []
    assert flow.manual_pause == ManualPauseState()
    assert step.status is StepStatus.CREATED
    assert step.submission is None
    assert step.result is None


def test_manual_pause_and_agent_bindings_defaults_are_empty() -> None:
    pause = ManualPauseState()
    bindings = AgentRoleBindings()

    assert pause.active is False
    assert pause.reason is None
    assert pause.paused_at is None
    assert bindings.by_role == {}
    assert bindings.get("planner") is None


def test_child_flow_dispatch_submission_saves_flow_requests() -> None:
    request = FlowRequest(flow_type="child", scope_id="child-scope", params={"node": "Main.Basic"})
    submission = ChildFlowDispatchSubmission(
        submission_id="sub-1",
        tool_name="submit_content_node_tasks",
        submitted_by_agent_id="agent-1",
        requests=[request],
        summary="run child task",
    )

    assert submission.submission_type == "child_flow_dispatch"
    assert submission.requests == [request]
    assert submission.submitted_at


def test_step_terminal_receipt_allows_completed_and_failed_statuses() -> None:
    completed = StepTerminalReceipt(
        step_id="step-1",
        flow_id="flow-1",
        scope_id="scope",
        status="completed",
        result_type="worker",
        finished_at="2026-06-22T00:00:00Z",
    )
    failed = StepTerminalReceipt(
        step_id="step-1",
        flow_id="flow-1",
        scope_id="scope",
        status="failed",
        error_type="runtime",
        finished_at="2026-06-22T00:00:01Z",
    )

    assert completed.result_type == "worker"
    assert failed.error_type == "runtime"


def test_step_terminal_receipt_rejects_waiting_status() -> None:
    try:
        StepTerminalReceipt(
            step_id="step-1",
            flow_id="flow-1",
            scope_id="scope",
            status="waiting",
            finished_at="2026-06-22T00:00:00Z",
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("StepTerminalReceipt accepted invalid waiting status")
