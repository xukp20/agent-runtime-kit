from types import SimpleNamespace
from unittest.mock import Mock

from agent_runtime_kit.agent.diagnostics import exception_diagnostics
from agent_runtime_kit.agent.provider_contracts import AgentEvent, ProviderSessionLocator
from agent_runtime_kit.agent.service import AgentService
from agent_runtime_kit.flow.services import _agent_step_exception_error


def test_unknown_failure_diagnostics_exclude_message_and_locals():
    try:
        raise RuntimeError("secret-token-sentinel")
    except RuntimeError as exc:
        result = exception_diagnostics(exc)
        error = _agent_step_exception_error(exc)
    assert "secret-token-sentinel" not in str(result)
    assert "secret-token-sentinel" not in str(error)
    assert result["frames"][-1]["function"] == "test_unknown_failure_diagnostics_exclude_message_and_locals"
    assert error.details["diagnostics"] == result
    assert error.details["operator_action_required"] is True


def test_created_session_upgrades_partial_identity():
    service = AgentService.__new__(AgentService)
    partial = ProviderSessionLocator(provider_type="grok", home_id="home", session_id="session", created_at="now")
    service.store = Mock()
    service.store.get_agent.return_value = SimpleNamespace(
        provider_type="grok", home_id="home", session_locator=partial,
    )
    full = ProviderSessionLocator(
        provider_type="grok", home_id="home", session_id="session", created_at="now",
        native_locator={"session_relpath": "sessions/session"},
    )
    event = AgentEvent(provider_type="grok", sequence=0, timestamp="now", kind="session.created",
                       session_id="session", data={"session_locator": full})
    service._on_provider_event("agent", event)
    service.store.update_session_locators.assert_called_once_with("agent", session_locator=full)


def test_home_commit_blocks_context_read_until_store_update(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from dataclasses import replace
    from agent_runtime_kit.agent.homes import HomeService
    from agent_runtime_kit.agent.provider_contracts import ProviderHomeSpec
    from agent_runtime_kit.agent.providers.codex_home import CodexHomeOptions

    homes = HomeService(tmp_path)
    homes.create_home(ProviderHomeSpec(provider_type="codex", home_id="worker", provider_options=CodexHomeOptions()))
    context = homes.build_execution_context("codex", "worker")
    staged, release, reading, attempted = Event(), Event(), Event(), Event()

    class Renderer:
        def commit_lifecycle_materialization(self, home, root, *, lifecycle):
            staged.set()
            assert release.wait(3)
            return replace(context.materialization_manifest, manifest_hash="updated")

        def build_execution_context(self, home, **kwargs):
            reading.set()
            assert home.materialization_manifest_hash == "updated"
            return "ok"

    homes.renderers["codex"] = Renderer()
    with ThreadPoolExecutor(max_workers=2) as pool:
        commit = pool.submit(homes.commit_provider_lifecycle_materialization, "codex", "worker", lifecycle="session_start")
        assert staged.wait(3)
        def read_context():
            attempted.set()
            return homes.build_execution_context("codex", "worker")

        read = pool.submit(read_context)
        assert attempted.wait(3)
        try:
            assert not reading.wait(0.1)
        finally:
            release.set()
        commit.result(timeout=3)
        assert read.result(timeout=3) == "ok"


def test_auth_reference_error_has_fixed_monitor_code():
    result = exception_diagnostics(RuntimeError("Grok Home auth reference changed"))
    assert result["code"] == "grok_home_auth_reference_changed"
    assert len(result["fingerprint"]) == 16
    assert "auth.json" not in str(result)


def test_loaded_session_does_not_replace_completed_turn_identity():
    service = AgentService.__new__(AgentService)
    existing = ProviderSessionLocator(provider_type="grok", home_id="home", session_id="session",
                                      created_at="original", native_locator={"session_relpath": "sessions/session"})
    service.store = Mock()
    service.store.get_agent.return_value = SimpleNamespace(provider_type="grok", home_id="home", session_locator=existing)
    loaded = ProviderSessionLocator(provider_type="grok", home_id="home", session_id="session",
                                    created_at="changed", native_locator=existing.native_locator)
    service._on_provider_event("agent", AgentEvent(provider_type="grok", sequence=0, timestamp="now",
                              kind="session.created", session_id="session", data={"session_locator": loaded}))
    service.store.update_session_locators.assert_not_called()
