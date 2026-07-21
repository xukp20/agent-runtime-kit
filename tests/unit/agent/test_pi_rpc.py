from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from agent_runtime_kit.agent.providers.pi_rpc import PiRpcError, PiRpcProcess


FIXTURE = Path(__file__).parents[2] / "fixtures" / "pi_rpc_fixture.py"


def _rpc(tmp_path: Path) -> PiRpcProcess:
    return PiRpcProcess(
        [sys.executable, str(FIXTURE)],
        cwd=tmp_path,
        env=os.environ,
    )


def test_pi_rpc_correlates_response_and_preserves_interleaved_events(tmp_path: Path) -> None:
    rpc = _rpc(tmp_path)
    try:
        response = rpc.command("echo", {"value": "ok"})
        event, cursor = rpc.wait_for(lambda item: item.get("type") == "message_start")
        assert response["data"] == {"value": "ok"}
        assert event["value"] == "ok"
        assert cursor == 1
        assert [item["type"] for item in rpc.records] == ["message_start", "response"]
    finally:
        rpc.close()


def test_pi_rpc_surfaces_provider_error_timeout_malformed_output_and_stderr(
    tmp_path: Path,
) -> None:
    rpc = _rpc(tmp_path)
    try:
        with pytest.raises(PiRpcError, match="fixture failure"):
            rpc.command("fail")
        with pytest.raises(TimeoutError, match="timed out"):
            rpc.command("hang", timeout_s=0.01)
    finally:
        rpc.close()

    malformed = _rpc(tmp_path)
    try:
        with pytest.raises(PiRpcError, match="non-JSON"):
            malformed.command("malformed")
    finally:
        malformed.close()

    exited = _rpc(tmp_path)
    try:
        with pytest.raises(PiRpcError, match="exited with code 7"):
            exited.command("exit")
        assert "fixture stderr" in exited.stderr_tail
    finally:
        exited.close()
