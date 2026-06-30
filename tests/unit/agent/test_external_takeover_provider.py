from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_runtime_kit.agent.providers import (
    ExternalTakeoverCancelled,
    ExternalTakeoverProvider,
    ExternalTakeoverTurnResult,
)
from agent_runtime_kit.agent.store_utils import read_json, write_json_atomic


def test_external_takeover_provider_start_thread_writes_handoff_and_returns_completion(tmp_path: Path) -> None:
    provider = ExternalTakeoverProvider(runtime_root=tmp_path / ".agent_runtime", poll_interval_s=0.001, default_timeout_s=5)
    result_box: dict[str, object] = {}
    error_box: dict[str, BaseException] = {}

    thread = _start_provider_thread(
        result_box,
        error_box,
        provider.start_thread,
        home_id="worker-home",
        home_root=tmp_path / "home",
        env={"ARK_STEP_ID": "step-1", "LEAN_CONSTELLATION_AGENT_TYPE": "Worker"},
        workdir=str(tmp_path / "repo"),
        prompt="Do the controlled task.",
        developer_instructions="Developer override.",
        overwrite_developer_instructions=True,
        agent_id="agent-1",
    )
    handoff_dir = _wait_for_handoff(provider)
    handoff = read_json(handoff_dir / "handoff.json")

    assert handoff["status"] == "pending"
    assert handoff["thread_id"] is None
    assert handoff["agent_id"] == "agent-1"
    assert handoff["home_id"] == "worker-home"
    assert handoff["prompt"] == "Do the controlled task."
    assert handoff["developer_instructions"] == "Developer override."
    assert handoff["overwrite_developer_instructions"] is True
    assert handoff["env"]["ARK_STEP_ID"] == "step-1"
    assert handoff["env"]["LEAN_CONSTELLATION_AGENT_TYPE"] == "Worker"
    assert handoff["workdir"] == str(tmp_path / "repo")

    handoff_id = str(handoff["handoff_id"])
    write_json_atomic(
        handoff_dir / "completion.json",
        {
            "schema_version": 1,
            "handoff_id": handoff_id,
            "status": "completed",
            "thread_id": "external-thread-1",
            "turn_id": "external-turn-1",
            "rollout_relpath": "external/turn-1.json",
            "final_response": "done",
            "metadata": {"tool": "submit_result"},
        },
    )
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert error_box == {}
    result = result_box["result"]
    assert result.thread_id == "external-thread-1"
    assert result.rollout_relpath == "external/turn-1.json"
    assert isinstance(result.turn_result, ExternalTakeoverTurnResult)
    assert result.turn_result.id == "external-turn-1"
    assert result.turn_result.status == "completed"
    assert result.turn_result.final_response == "done"
    assert result.turn_result.handoff_id == handoff_id
    assert result.turn_result.metadata == {"tool": "submit_result"}

    agent = SimpleNamespace(thread_id="external-thread-1")
    latest = provider.read_latest_turn_result(agent, home_root=tmp_path / "home", env={})
    snapshot = provider.read_thread(agent, home_root=tmp_path / "home", env={})
    assert latest.id == "external-turn-1"
    assert snapshot.id == "external-thread-1"
    assert [turn.id for turn in snapshot.turns] == ["external-turn-1"]


def test_external_takeover_provider_resume_records_source_thread(tmp_path: Path) -> None:
    provider = ExternalTakeoverProvider(runtime_root=tmp_path / ".agent_runtime", poll_interval_s=0.001, default_timeout_s=5)
    result_box: dict[str, object] = {}
    error_box: dict[str, BaseException] = {}
    thread = _start_provider_thread(
        result_box,
        error_box,
        provider.resume_thread,
        home_id="worker-home",
        home_root=tmp_path / "home",
        env={"ARK_FLOW_ID": "flow-1"},
        thread_id="external-thread-source",
        workdir=None,
        prompt="Resume task.",
        developer_instructions=None,
        overwrite_developer_instructions=False,
        agent_id="agent-1",
    )
    handoff_dir = _wait_for_handoff(provider)
    handoff = read_json(handoff_dir / "handoff.json")

    assert handoff["thread_id"] == "external-thread-source"
    write_json_atomic(
        handoff_dir / "completion.json",
        {
            "schema_version": 1,
            "handoff_id": handoff["handoff_id"],
            "status": "completed",
            "final_response": "resumed",
        },
    )
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert error_box == {}
    assert result_box["result"].thread_id == "external-thread-source"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("failed", RuntimeError),
        ("cancelled", ExternalTakeoverCancelled),
    ],
)
def test_external_takeover_provider_completion_failure_states_raise(
    tmp_path: Path,
    status: str,
    expected: type[BaseException],
) -> None:
    provider = ExternalTakeoverProvider(runtime_root=tmp_path / ".agent_runtime", poll_interval_s=0.001, default_timeout_s=5)
    result_box: dict[str, object] = {}
    error_box: dict[str, BaseException] = {}
    thread = _start_provider_thread(
        result_box,
        error_box,
        provider.start_thread,
        home_id="worker-home",
        home_root=tmp_path / "home",
        env={},
        workdir=None,
        prompt="Run.",
        developer_instructions=None,
        agent_id="agent-1",
    )
    handoff_dir = _wait_for_handoff(provider)
    handoff = read_json(handoff_dir / "handoff.json")
    write_json_atomic(
        handoff_dir / "completion.json",
        {
            "schema_version": 1,
            "handoff_id": handoff["handoff_id"],
            "status": status,
            "error": "external controller stopped",
        },
    )
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert result_box == {}
    assert isinstance(error_box["error"], expected)


def test_external_takeover_provider_timeout_does_not_hang(tmp_path: Path) -> None:
    provider = ExternalTakeoverProvider(runtime_root=tmp_path / ".agent_runtime", poll_interval_s=0.001, default_timeout_s=0.01)

    with pytest.raises(TimeoutError):
        provider.start_thread(
            home_id="worker-home",
            home_root=tmp_path / "home",
            env={},
            workdir=None,
            prompt="Run.",
            developer_instructions=None,
            agent_id="agent-1",
        )

    assert len(list(provider.handoff_root.glob("*/handoff.json"))) == 1


def _start_provider_thread(
    result_box: dict[str, object],
    error_box: dict[str, BaseException],
    target,
    **kwargs,
) -> threading.Thread:
    def run() -> None:
        try:
            result_box["result"] = target(**kwargs)
        except BaseException as exc:  # noqa: BLE001 - tests need to inspect provider worker errors.
            error_box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _wait_for_handoff(provider: ExternalTakeoverProvider) -> Path:
    deadline = 2.0
    step = 0.001
    waited = 0.0
    while waited < deadline:
        handoffs = list(provider.handoff_root.glob("*/handoff.json"))
        if handoffs:
            return handoffs[0].parent
        threading.Event().wait(step)
        waited += step
    raise AssertionError("external takeover handoff was not written")
