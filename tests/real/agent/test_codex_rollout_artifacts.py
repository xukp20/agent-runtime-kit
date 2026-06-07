from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pytest

from agent_runtime_kit.agent.snapshots import AgentSnapshotService
from agent_runtime_kit.agent.store import AgentStoreService


pytestmark = pytest.mark.real_codex_artifact


def test_real_codex_rollout_artifacts_read_and_restore(tmp_path: Path) -> None:
    sample_dir = _artifact_dir()
    samples = _selected_samples(sample_dir, limit=2)
    runtime_root = tmp_path / "project" / ".agent_runtime"
    store = AgentStoreService(runtime_root)
    snapshot_service = AgentSnapshotService(runtime_root, store=store)
    agents = []
    original_checksums: dict[str, str] = {}
    original_line_counts: dict[str, int] = {}

    for index, sample in enumerate(samples, start=1):
        home_id = "artifact_reader"
        relpath = Path("sessions") / "artifacts" / sample.name
        target = runtime_root / "homes" / "codex" / home_id / ".codex" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sample, target)
        agent = store.create_agent_record(
            scope_id=f"repo-a:artifact-scope-{index}",
            agent_type="artifact_reader",
            cli_type="codex",
            home_id=home_id,
            thread_id=f"artifact-thread-{index}",
            rollout_relpath=str(relpath),
        )
        agents.append(agent)
        original_checksums[agent.agent_id] = _sha256(target)
        original_line_counts[agent.agent_id] = _line_count(target)

    for agent in agents:
        events = store.read_rollout_events(agent.agent_id)
        assert events
        assert len(events) == original_line_counts[agent.agent_id]

    first_scope_snapshot = snapshot_service.create_scope_snapshot(agents[0].scope_id)
    assert first_scope_snapshot.status == "created"
    assert first_scope_snapshot.snapshot_id is not None
    first_rollout = store.locate_rollout(agents[0].agent_id)
    assert first_rollout is not None
    first_rollout.unlink()

    restored_scope = snapshot_service.restore_scope_snapshot(first_scope_snapshot.snapshot_id)
    assert restored_scope.status == "created"
    assert first_rollout.exists()
    assert _sha256(first_rollout) == original_checksums[agents[0].agent_id]

    runtime_snapshot = snapshot_service.create_runtime_snapshot_synchronized()
    assert runtime_snapshot.status == "created"
    assert set(runtime_snapshot.scope_snapshot_ids) == {agent.scope_id for agent in agents}

    for agent in agents:
        rollout = store.locate_rollout(agent.agent_id)
        assert rollout is not None
        rollout.write_text("", encoding="utf-8")
    restored_runtime = snapshot_service.restore_runtime_snapshot(runtime_snapshot.snapshot_id)
    assert restored_runtime.status == "created"

    for agent in agents:
        rollout = store.locate_rollout(agent.agent_id)
        assert rollout is not None
        assert _sha256(rollout) == original_checksums[agent.agent_id]
        assert _line_count(rollout) == original_line_counts[agent.agent_id]


def _artifact_dir() -> Path:
    if os.environ.get("ARK_RUN_REAL_CODEX_ARTIFACTS") != "1":
        pytest.skip("set ARK_RUN_REAL_CODEX_ARTIFACTS=1 to run real Codex artifact tests")
    path = Path(os.environ.get("ARK_CODEX_SAMPLE_ROLLOUTS", "data/configs/codex/sample_rollouts"))
    if not path.exists():
        pytest.skip(f"Codex sample rollout directory does not exist: {path}")
    return path


def _selected_samples(sample_dir: Path, *, limit: int) -> list[Path]:
    samples = sorted(sample_dir.glob("*.jsonl"), key=lambda path: path.stat().st_size)
    if len(samples) < limit:
        pytest.skip(f"need at least {limit} Codex rollout samples in {sample_dir}")
    return samples[:limit]


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _line in handle)
