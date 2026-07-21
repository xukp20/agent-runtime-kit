from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime_kit.agent.providers.codex_context import (
    capture_codex_compact_baseline,
    inspect_codex_compact_evidence,
    inspect_codex_rollout_context,
)


def test_inspect_codex_rollout_context_uses_last_usage_total_tokens(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.jsonl"
    _append(rollout, _token_count(total_tokens=80, context_window=100))

    usage = inspect_codex_rollout_context(rollout, session_id="thread-1")

    assert usage.available
    assert usage.used_tokens == 80
    assert usage.effective_context_window_tokens == 100


def test_inspect_codex_rollout_context_uses_latest_complete_token_event(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.jsonl"
    _append(rollout, _token_count(total_tokens=80, context_window=100))
    _append(rollout, _token_count(total_tokens=20, context_window=100))
    with rollout.open("ab") as handle:
        handle.write(b'{"type":"event_msg"')

    usage = inspect_codex_rollout_context(rollout, session_id="thread-1")

    assert usage.used_tokens == 20


def test_inspect_codex_rollout_context_reports_missing_fields(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.jsonl"
    _append(
        rollout,
        {
            "type": "event_msg",
            "payload": {"type": "token_count", "info": {"last_token_usage": {"total_tokens": 10}}},
        },
    )

    usage = inspect_codex_rollout_context(rollout, session_id="thread-1")

    assert not usage.available
    assert usage.reason == "context_window_missing"
    assert usage.used_tokens == 10


def test_compact_evidence_requires_new_markers_after_baseline(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.jsonl"
    _append(rollout, _token_count(total_tokens=80, context_window=100))
    baseline = capture_codex_compact_baseline(rollout, session_id="thread-1")

    _append(rollout, {"type": "compacted", "payload": {"message": "summary"}})
    _append(rollout, {"type": "event_msg", "payload": {"type": "context_compacted"}})
    _append(rollout, _token_count(total_tokens=20, context_window=100))

    evidence = inspect_codex_compact_evidence(
        rollout,
        session_id="thread-1",
        baseline=baseline,
    )

    assert evidence.complete
    assert evidence.usage.used_tokens == 20


def test_compact_evidence_does_not_reuse_old_markers(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.jsonl"
    _append(rollout, {"type": "compacted", "payload": {"message": "old"}})
    _append(rollout, {"type": "event_msg", "payload": {"type": "context_compacted"}})
    _append(rollout, _token_count(total_tokens=20, context_window=100))
    baseline = capture_codex_compact_baseline(rollout, session_id="thread-1")

    evidence = inspect_codex_compact_evidence(
        rollout,
        session_id="thread-1",
        baseline=baseline,
    )

    assert not evidence.complete
    assert not evidence.has_new_compacted
    assert not evidence.has_new_context_compacted
    assert not evidence.has_new_token_count


def test_compact_evidence_rejects_truncated_rollout(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.jsonl"
    _append(rollout, _token_count(total_tokens=80, context_window=100))
    baseline = capture_codex_compact_baseline(rollout, session_id="thread-1")
    rollout.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="truncated"):
        inspect_codex_compact_evidence(rollout, session_id="thread-1", baseline=baseline)


def test_complete_invalid_json_line_is_rejected(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="complete line 1"):
        inspect_codex_rollout_context(rollout, session_id="thread-1")


def _append(path: Path, event: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def _token_count(*, total_tokens: int, context_window: int) -> dict[str, object]:
    return {
        "timestamp": "2026-07-21T00:00:00Z",
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": {"total_tokens": 999},
                "last_token_usage": {"total_tokens": total_tokens},
                "model_context_window": context_window,
            },
            "rate_limits": None,
        },
    }
