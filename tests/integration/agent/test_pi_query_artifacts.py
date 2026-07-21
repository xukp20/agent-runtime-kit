from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent_runtime_kit.agent.provider_contracts import (
    ArtifactCaptureRequest,
    ArtifactRestoreRequest,
    ArtifactStabilityRequest,
    ProviderEventQuery,
    ProviderSessionListQuery,
    ProviderToolQuery,
    ProviderTurnQuery,
    ProviderUsageQuery,
)
from agent_runtime_kit.agent.providers.pi_bundle import build_pi_provider_bundle


def _write_session(runtime_root: Path, home_id: str = "demo") -> Path:
    root = runtime_root / "homes" / "pi" / home_id / ".pi" / "sessions"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "session.jsonl"
    records = [
        {"type": "session", "version": 3, "id": "s1", "timestamp": "2026-07-21T00:00:00Z", "cwd": "/w"},
        {"type": "message", "id": "u1", "parentId": None, "timestamp": "2026-07-21T00:00:01Z", "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}},
        {"type": "message", "id": "a1", "parentId": "u1", "timestamp": "2026-07-21T00:00:02Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}], "api": "openai-responses", "provider": "beeapi", "model": "gpt-5.4", "responseId": "r1", "usage": {"input": 2, "output": 1, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 3, "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}}, "stopReason": "stop"}},
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    return path


def test_pi_query_and_artifact_snapshot_restore(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    path = _write_session(runtime_root)
    home_manifest = runtime_root / "homes" / "pi" / "demo" / ".ark" / "home_materialization.json"
    home_manifest.parent.mkdir(parents=True, exist_ok=True)
    home_manifest.write_text("{}\n", encoding="utf-8")
    bundle = build_pi_provider_bundle(runtime_root=runtime_root)
    assert bundle.query is not None
    page = bundle.query.list_sessions(ProviderSessionListQuery(home_id="demo"))
    assert len(page.items) == 1
    session = page.items[0].locator
    turns = bundle.query.list_turns(ProviderTurnQuery(session=session))
    assert turns.items[0].result.final_text == "ok"
    events = bundle.query.list_events(ProviderEventQuery(session=session))
    assert [item.kind for item in events.items] == ["message.user", "message.assistant"]
    assert bundle.query.list_tool_calls(ProviderToolQuery(session=session)).items == ()
    usage = bundle.query.read_usage(
        ProviderUsageQuery(session=session, include_session_aggregate=True)
    )
    assert usage.token_usage.total_tokens == 3

    assert bundle.artifacts is not None
    assert bundle.artifacts.wait_quiescent(ArtifactStabilityRequest(session=session)).stable
    snapshot_root = tmp_path / "snapshot"
    captured = bundle.artifacts.capture(
        ArtifactCaptureRequest(session=session, snapshot_root=str(snapshot_root))
    )
    assert captured.manifest.stable
    assert {entry.capture_strategy for entry in captured.manifest.entries} == {
        "copy_file",
        "reference_hash",
    }
    session_entry = next(
        entry for entry in captured.manifest.entries if entry.kind == "session_transcript"
    )
    unsafe_manifest = replace(
        captured.manifest,
        entries=(replace(session_entry, snapshot_relpath="../escape.jsonl"),),
    )
    with pytest.raises(RuntimeError, match="escapes the snapshot root"):
        bundle.artifacts.restore(
            ArtifactRestoreRequest(manifest=unsafe_manifest, snapshot_root=str(snapshot_root))
        )
    path.unlink()
    restored = bundle.artifacts.restore(
        ArtifactRestoreRequest(manifest=captured.manifest, snapshot_root=str(snapshot_root))
    )
    assert restored.restored
    assert path.is_file()

    home_manifest.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Home materialization manifest"):
        bundle.artifacts.prepare_restore(
            ArtifactRestoreRequest(manifest=captured.manifest, snapshot_root=str(snapshot_root))
        )
    assert path.is_file()
    home_manifest.write_text("{}\n", encoding="utf-8")

    path.write_text("different\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="different content"):
        bundle.artifacts.restore(
            ArtifactRestoreRequest(manifest=captured.manifest, snapshot_root=str(snapshot_root))
        )
