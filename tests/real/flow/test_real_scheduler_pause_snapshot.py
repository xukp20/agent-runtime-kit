from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime_kit.flow import FlowRequest, FlowStatus, StepStatus
from agent_runtime_kit.flow.standard_steps import DispatchStep

from .support import (
    DEFAULT_AGENT_TIMEOUT_S,
    make_real_flow_runtime,
    run_flow_until_terminal,
    run_step_in_thread,
)


pytestmark = pytest.mark.real_codex


def test_real_runtime_pause_blocks_flow_advance_and_step_run(tmp_path: Path) -> None:
    runtime = make_real_flow_runtime(tmp_path)
    scope_id = "scope:rt06"
    try:
        flow_id = runtime.flow_service.start_flow(
            FlowRequest(
                flow_type="real_single_agent",
                scope_id=scope_id,
                params={
                    "prompt": (
                        "Call MCP tool ark_submit_result exactly once with summary='rt06 resumed'. "
                        "Reply with exactly the returned tool result."
                    ),
                },
            ),
            enqueue=True,
        )

        assert runtime.ark.pause_controller is not None
        runtime.ark.pause_controller.pause(scope_id)

        paused_tick = runtime.schedule_service.schedule_ready()
        paused_flow = runtime.flow_service.get_flow(flow_id)

        assert paused_tick.advanced_flow_ids == []
        assert paused_tick.started_step_ids == []
        assert paused_flow.status is FlowStatus.CREATED
        assert paused_flow.step_ids == []

        runtime.ark.pause_controller.resume(scope_id)
        resumed_flow = run_flow_until_terminal(runtime, flow_id)
        steps = runtime.flow_service.list_steps(flow_id=flow_id)
    finally:
        runtime.close()

    assert resumed_flow.status is FlowStatus.COMPLETED
    assert steps
    assert all(step.status is StepStatus.COMPLETED for step in steps)


def test_real_scope_snapshot_restore_parent_waiting_stable_point(tmp_path: Path) -> None:
    runtime = make_real_flow_runtime(tmp_path)
    scope_id = "scope:rt07"
    try:
        flow_id = runtime.flow_service.start_flow(
            FlowRequest(
                flow_type="real_dispatch_parent",
                scope_id=scope_id,
                params={"names_csv": "alpha,beta", "child_flow_type": "real_logic_child"},
            ),
            enqueue=True,
        )

        first_tick = runtime.schedule_service.schedule_ready()
        assert first_tick.advanced_flow_ids == [flow_id]
        assert len(first_tick.started_step_ids) == 1
        runtime.step_service.wait_step(first_tick.started_step_ids[0], timeout_s=DEFAULT_AGENT_TIMEOUT_S)

        second_tick = runtime.schedule_service.schedule_ready()
        assert flow_id in second_tick.advanced_flow_ids
        assert len(second_tick.started_step_ids) == 1
        dispatch_step_id = second_tick.started_step_ids[0]
        dispatch_step = runtime.step_service.wait_step(dispatch_step_id, timeout_s=DEFAULT_AGENT_TIMEOUT_S)
        assert isinstance(dispatch_step, DispatchStep)

        waiting_parent = runtime.flow_service.get_flow(flow_id)
        child_flows = runtime.flow_service.store.list_child_flows(
            parent_flow_id=flow_id,
            parent_dispatch_step_id=dispatch_step_id,
        )
        assert waiting_parent.status is FlowStatus.WAITING
        assert waiting_parent.current_step_id is None
        assert len(child_flows) == 2
        assert {child.status for child in child_flows} == {FlowStatus.CREATED}

        snapshot = runtime.snapshot_service.create_scope_snapshot(scope_id)
        assert snapshot.status == "created"
        assert snapshot.snapshot_id is not None

        completed_before_restore = run_flow_until_terminal(runtime, flow_id)
        assert completed_before_restore.status is FlowStatus.COMPLETED

        restored = runtime.snapshot_service.restore_scope_snapshot(snapshot.snapshot_id)
        assert restored.status == "created"
        assert runtime.ark.pause_controller is not None
        assert runtime.ark.pause_controller.is_paused(scope_id) is True

        restored_parent = runtime.flow_service.get_flow(flow_id)
        restored_child_flows = runtime.flow_service.store.list_child_flows(
            parent_flow_id=flow_id,
            parent_dispatch_step_id=dispatch_step_id,
        )
        assert restored_parent.status is FlowStatus.WAITING
        assert restored_parent.current_step_id is None
        assert len(restored_child_flows) == 2
        assert {child.status for child in restored_child_flows} == {FlowStatus.CREATED}

        paused_tick = runtime.schedule_service.schedule_ready()
        assert paused_tick.advanced_flow_ids == []
        assert paused_tick.started_step_ids == []

        runtime.ark.pause_controller.resume(scope_id)
        completed_after_restore = run_flow_until_terminal(runtime, flow_id)
    finally:
        runtime.close()

    assert completed_after_restore.status is FlowStatus.COMPLETED
    assert completed_after_restore.result is not None


def test_real_runtime_snapshot_for_scopes_restore_dispatch_callback(tmp_path: Path) -> None:
    runtime = make_real_flow_runtime(tmp_path)
    scope_id = "scope:rt09"
    other_scope_id = "scope:rt09-other"
    try:
        other_flow_id = runtime.flow_service.start_flow(
            FlowRequest(
                flow_type="real_logic_child",
                scope_id=other_scope_id,
                params={"name": "other"},
            ),
            enqueue=True,
        )
        other_flow = run_flow_until_terminal(runtime, other_flow_id)
        assert other_flow.status is FlowStatus.COMPLETED
        other_scope_snapshot = runtime.snapshot_service.create_scope_snapshot(other_scope_id)
        assert other_scope_snapshot.status == "created"
        assert other_scope_snapshot.snapshot_id is not None

        flow_id = runtime.flow_service.start_flow(
            FlowRequest(
                flow_type="real_dispatch_parent",
                scope_id=scope_id,
                params={"names_csv": "alpha,beta", "child_flow_type": "real_logic_child"},
            ),
            enqueue=True,
        )

        first_tick = runtime.schedule_service.schedule_ready()
        assert first_tick.advanced_flow_ids == [flow_id]
        assert len(first_tick.started_step_ids) == 1
        runtime.step_service.wait_step(first_tick.started_step_ids[0], timeout_s=DEFAULT_AGENT_TIMEOUT_S)

        second_tick = runtime.schedule_service.schedule_ready()
        assert flow_id in second_tick.advanced_flow_ids
        assert len(second_tick.started_step_ids) == 1
        dispatch_step_id = second_tick.started_step_ids[0]
        dispatch_step = runtime.step_service.wait_step(dispatch_step_id, timeout_s=DEFAULT_AGENT_TIMEOUT_S)
        assert isinstance(dispatch_step, DispatchStep)

        waiting_parent = runtime.flow_service.get_flow(flow_id)
        child_flows = runtime.flow_service.store.list_child_flows(
            parent_flow_id=flow_id,
            parent_dispatch_step_id=dispatch_step_id,
        )
        assert waiting_parent.status is FlowStatus.WAITING
        assert len(child_flows) == 2

        for child_flow in child_flows:
            assert runtime.flow_service.advance_flow(child_flow.flow_id) is None
            completed_child = runtime.flow_service.get_flow(child_flow.flow_id)
            assert completed_child.status is FlowStatus.COMPLETED

        parent_before_snapshot = runtime.flow_service.get_flow(flow_id)
        child_flows_before_snapshot = runtime.flow_service.store.list_child_flows(
            parent_flow_id=flow_id,
            parent_dispatch_step_id=dispatch_step_id,
        )
        assert parent_before_snapshot.status is FlowStatus.WAITING
        assert parent_before_snapshot.current_step_id is None
        assert {child.status for child in child_flows_before_snapshot} == {FlowStatus.COMPLETED}

        runtime_snapshot = runtime.snapshot_service.create_runtime_snapshot_for_scopes(
            refresh_scope_ids=[scope_id],
            scope_ids=[scope_id, other_scope_id],
        )
        assert runtime_snapshot.status == "created"
        assert runtime_snapshot.snapshot_id is not None
        assert set(runtime_snapshot.scope_snapshot_ids) == {scope_id, other_scope_id}
        assert runtime_snapshot.scope_snapshot_ids[other_scope_id] == other_scope_snapshot.snapshot_id

        completed_before_restore = run_flow_until_terminal(runtime, flow_id)
        assert completed_before_restore.status is FlowStatus.COMPLETED

        restored = runtime.snapshot_service.restore_runtime_snapshot(runtime_snapshot.snapshot_id)
        assert restored.status == "created"
        assert runtime.ark.pause_controller is not None
        assert runtime.ark.pause_controller.is_paused(None) is True

        restored_parent = runtime.flow_service.get_flow(flow_id)
        restored_child_flows = runtime.flow_service.store.list_child_flows(
            parent_flow_id=flow_id,
            parent_dispatch_step_id=dispatch_step_id,
        )
        assert restored_parent.status is FlowStatus.WAITING
        assert restored_parent.current_step_id is None
        assert {child.status for child in restored_child_flows} == {FlowStatus.COMPLETED}

        runtime.ark.pause_controller.resume(None)
        completed_after_restore = run_flow_until_terminal(runtime, flow_id)
        parent_steps = runtime.flow_service.list_steps(flow_id=flow_id)
    finally:
        runtime.close()

    assert completed_after_restore.status is FlowStatus.COMPLETED
    assert completed_after_restore.result is not None
    assert len([step for step in parent_steps if step.status is StepStatus.COMPLETED]) >= 3


@pytest.mark.slow
def test_real_scope_snapshot_blocks_while_agent_step_running(tmp_path: Path) -> None:
    runtime = make_real_flow_runtime(tmp_path)
    scope_id = "scope:rt08"
    thread = None
    try:
        flow_id = runtime.flow_service.start_flow(
            FlowRequest(
                flow_type="real_single_agent",
                scope_id=scope_id,
                params={
                    "prompt": (
                        "Call MCP tool ark_sleep_then_submit exactly once with seconds=3 "
                        "and summary='rt08 slept'. Reply with exactly the returned tool result."
                    ),
                },
            ),
            enqueue=False,
        )
        step_id = runtime.flow_service.advance_flow(flow_id)
        assert step_id is not None

        thread = run_step_in_thread(runtime, step_id)
        assert runtime.submit_bridge.sleep_started.wait(timeout=DEFAULT_AGENT_TIMEOUT_S)

        blocked = runtime.snapshot_service.create_scope_snapshot(scope_id, wait=False)
        assert blocked.status == "blocked"
        assert step_id in blocked.running_step_ids

        thread.join(timeout=DEFAULT_AGENT_TIMEOUT_S)
        assert not thread.is_alive()

        step = runtime.flow_service.get_step(step_id)
        assert step.status is StepStatus.COMPLETED
        completed_flow = run_flow_until_terminal(runtime, flow_id)
        assert completed_flow.status is FlowStatus.COMPLETED

        created = runtime.snapshot_service.create_scope_snapshot(scope_id)
        assert created.status == "created"
        assert created.snapshot_id is not None
    finally:
        if thread is not None and thread.is_alive():
            thread.join(timeout=DEFAULT_AGENT_TIMEOUT_S)
        runtime.close()
