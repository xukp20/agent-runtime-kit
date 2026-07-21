from __future__ import annotations

from dataclasses import asdict

import pytest

from agent_runtime_kit.agent.provider_contracts import (
    AgentContextUsage,
    ModelBackendIdentity,
    ProviderForkResult,
    ProviderRunRequest,
    ProviderRunState,
    ProviderSessionLocator,
    ProviderTurnResult,
    build_provider_payload,
)
from agent_runtime_kit.agent.models import to_jsonable


def _session() -> ProviderSessionLocator:
    return ProviderSessionLocator(
        provider_type="codex",
        session_id="thread-1",
        home_id="worker",
        created_at="2026-07-21T10:00:00Z",
    )


def test_run_request_rejects_cross_provider_session() -> None:
    session = ProviderSessionLocator(
        provider_type="claude_code",
        session_id="session-1",
        home_id="worker",
        created_at="2026-07-21T10:00:00Z",
    )
    with pytest.raises(ValueError, match="provider_type"):
        ProviderRunRequest(
            agent_id="a1",
            scope_id="s1",
            agent_type="Worker",
            provider_type="codex",
            home_id="worker",
            prompt="work",
            session_locator=session,
        )


def test_provider_turn_result_is_json_serializable_without_native_sdk_objects() -> None:
    result = ProviderTurnResult(
        provider_type="codex",
        run_id="run-1",
        session_locator=_session(),
        status=ProviderRunState.COMPLETED,
        started_at="2026-07-21T10:00:00Z",
        completed_at="2026-07-21T10:00:01Z",
        final_text="done",
    )

    payload = to_jsonable(result)
    assert payload["status"] == "completed"
    assert payload["session_locator"]["session_id"] == "thread-1"


def test_context_usage_exposes_legacy_aliases_without_inventing_values() -> None:
    usage = AgentContextUsage(
        agent_id="a1",
        provider_type="codex",
        session_id="thread-1",
        used_tokens=80,
        context_window_tokens=120,
        effective_context_window_tokens=100,
        observed_at="2026-07-21T10:00:00Z",
        source="provider_reported",
        measurement="provider_reported",
        available=True,
    )

    assert usage.total_tokens == 80
    assert usage.context_window == 100
    assert usage.usage_ratio == pytest.approx(0.8)


def test_provider_payload_redacts_secrets_but_preserves_usage_fields() -> None:
    payload = build_provider_payload(
        provider_type="codex",
        payload_type="usage",
        data={"access_token": "secret", "input_tokens": 42, "nested": {"api_key": "key"}},
    )

    assert payload.sanitized_data == {
        "access_token": "[REDACTED]",
        "input_tokens": 42,
        "nested": {"api_key": "[REDACTED]"},
    }


def test_fork_contract_never_claims_workspace_isolation() -> None:
    with pytest.raises(ValueError, match="workspace isolation"):
        ProviderForkResult(
            source_session=_session(),
            target_session=_session(),
            status="completed",
            workspace_isolated=True,
        )
