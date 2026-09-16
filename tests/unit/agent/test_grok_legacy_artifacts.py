from urllib.parse import quote
import pytest
from agent_runtime_kit.agent.provider_contracts import ProviderSessionLocator
from agent_runtime_kit.agent.providers.grok_artifacts import GrokArtifactAdapter


def test_legacy_artifacts_resolve_unique_native_directory_without_rewriting_locator(tmp_path):
    adapter = GrokArtifactAdapter(runtime_root=tmp_path)
    locator = ProviderSessionLocator(provider_type="grok", home_id="home", session_id="session", created_at="now")
    root = tmp_path / "homes/grok/home/.grok/sessions"
    path = root / quote("/workspace", safe="") / "session"
    path.mkdir(parents=True)
    assert adapter._session_dir(locator) == path
    assert locator.native_locator is None
    (root / quote("/another", safe="") / "session").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="ambiguous"):
        adapter._session_dir(locator)


def test_legacy_artifacts_reject_missing_or_symlink(tmp_path):
    adapter = GrokArtifactAdapter(runtime_root=tmp_path)
    locator = ProviderSessionLocator(provider_type="grok", home_id="home", session_id="session", created_at="now")
    with pytest.raises(RuntimeError):
        adapter._session_dir(locator)
    root = tmp_path / "homes/grok/home/.grok/sessions" / quote("/workspace", safe="")
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "session").symlink_to(outside, target_is_directory=True)
    with pytest.raises((RuntimeError, ValueError)):
        adapter._session_dir(locator)
