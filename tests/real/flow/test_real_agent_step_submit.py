from __future__ import annotations

import threading
from pathlib import Path

import pytest

from agent_runtime_kit.flow import FlowRequest, FlowStatus, StepStatus
from agent_runtime_kit.flow.standard_steps import AgentStepIncompleteResult, AgentStepSubmissionResult

from .support import (
    NO_TOOL_HOME_ID,
    make_real_flow_runtime,
    run_flow_until_terminal,
    run_step_in_thread,
)


pytestmark = pytest.mark.real_codex


def test_real_flow_home_config_renders_http_mcp_headers(tmp_path: Path) -> None:
    runtime = make_real_flow_runtime(tmp_path)
    try:
        home_root = runtime.agent_service.home_service.resolve_home_root("codex", "real_flow_submit_agent")
        rendered = (home_root / ".codex" / "config.toml").read_text(encoding="utf-8")
    finally:
        runtime.close()

    assert "[mcp_servers.ark_flow_submit]" in rendered
    assert 'url = "http://127.0.0.1:' in rendered
    assert "[mcp_servers.ark_flow_submit.env_http_headers]" in rendered
    assert 'X-Ark-Step-Id = "ARK_STEP_ID"' in rendered
    assert 'X-Ark-Flow-Id = "ARK_FLOW_ID"' in rendered
    assert 'X-Ark-Agent-Id = "ARK_AGENT_ID"' in rendered


def test_real_agent_step_submit_result_through_http_mcp(tmp_path: Path) -> None:
    runtime = make_real_flow_runtime(tmp_path)
    try:
        prompt = (
            "Call MCP tool ark_submit_result exactly once with summary='rt01 submitted'. "
            "Reply with exactly the returned tool result."
        )
        flow_id = runtime.flow_service.start_flow(
            FlowRequest(
                flow_type="real_single_agent",
                scope_id="scope:rt01",
                params={"prompt": prompt, "expected_summary": "rt01 submitted"},
            ),
            enqueue=True,
        )

        flow = run_flow_until_terminal(runtime, flow_id)
        steps = runtime.flow_service.list_steps(flow_id=flow_id)
    finally:
        runtime.close()

    assert flow.status is FlowStatus.COMPLETED
    assert flow.result is not None
    assert getattr(flow.result, "submitted_summary", None) == "rt01 submitted"
    assert len(steps) == 1
    step = steps[0]
    assert step.status is StepStatus.COMPLETED
    assert isinstance(step.result, AgentStepSubmissionResult)
    assert step.submission is not None
    assert step.submission.summary == "rt01 submitted"
    assert any(call["tool_name"] == "ark_submit_result" for call in runtime.submit_bridge.call_log)


def test_real_agent_step_without_submission_retries_then_completes_incomplete(tmp_path: Path) -> None:
    runtime = make_real_flow_runtime(tmp_path)
    try:
        prompt = (
            "Reply with exactly NO_SUBMISSION and do not call any MCP tool. "
            "This test intentionally expects no tool call."
        )
        flow_id = runtime.flow_service.start_flow(
            FlowRequest(
                flow_type="real_single_agent",
                scope_id="scope:rt02",
                params={
                    "prompt": prompt,
                    "home_id": NO_TOOL_HOME_ID,
                    "max_auto_continue_turns": 0,
                },
            ),
            enqueue=True,
        )

        flow = run_flow_until_terminal(runtime, flow_id)
        steps = runtime.flow_service.list_steps(flow_id=flow_id)
    finally:
        runtime.close()

    assert flow.status is FlowStatus.COMPLETED
    assert len(steps) == 1
    step = steps[0]
    assert step.status is StepStatus.COMPLETED
    assert step.submission is None
    assert isinstance(step.result, AgentStepIncompleteResult)
    assert step.result.outcome == "incomplete"


def test_real_same_home_concurrent_agent_steps_keep_identity_isolated(tmp_path: Path) -> None:
    runtime = make_real_flow_runtime(tmp_path, max_concurrent_steps=2)
    threads: list[threading.Thread] = []
    try:
        flow_ids: list[str] = []
        step_ids: list[str] = []
        for index in range(2):
            prompt = (
                "Call MCP tool ark_submit_result exactly once with "
                f"summary='concurrent summary {index}'. Reply with exactly the returned tool result."
            )
            flow_id = runtime.flow_service.start_flow(
                FlowRequest(
                    flow_type="real_single_agent",
                    scope_id="scope:rt05",
                    params={"prompt": prompt},
                ),
                enqueue=False,
            )
            step_id = runtime.flow_service.advance_flow(flow_id)
            assert step_id is not None
            flow_ids.append(flow_id)
            step_ids.append(step_id)

        threads = [run_step_in_thread(runtime, step_id) for step_id in step_ids]
        for thread in threads:
            thread.join(timeout=900)
            assert not thread.is_alive()

        for flow_id in flow_ids:
            run_flow_until_terminal(runtime, flow_id)

        completed_steps = [runtime.flow_service.get_step(step_id) for step_id in step_ids]
        call_log = list(runtime.submit_bridge.call_log)
    finally:
        runtime.close()

    assert {step.status for step in completed_steps} == {StepStatus.COMPLETED}
    assert all(step.submission is not None for step in completed_steps)
    submit_calls = [call for call in call_log if call["tool_name"] == "ark_submit_result"]
    assert len(submit_calls) >= 2
    call_by_step = {call["step_id"]: call for call in submit_calls}
    assert set(step_ids).issubset(call_by_step)
    assert call_by_step[step_ids[0]]["agent_id"] != call_by_step[step_ids[1]]["agent_id"]
    assert completed_steps[0].submission is not None
    assert completed_steps[1].submission is not None
    assert completed_steps[0].submission.summary != completed_steps[1].submission.summary
