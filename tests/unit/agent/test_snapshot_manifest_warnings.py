from __future__ import annotations

from agent_runtime_kit.agent.snapshots import _provider_artifact_manifest_from_dict


def test_provider_artifact_manifest_warns_for_omitted_compatibility_fields(caplog) -> None:
    manifest = _provider_artifact_manifest_from_dict(
        {
            "provider_type": "fake",
            "home_id": "home-1",
            "session_id": "session-1",
            "adapter_version": "1",
            "stable": True,
            "entries": [
                {
                    "artifact_id": "artifact-1",
                    "kind": "rollout",
                    "authority": "provider",
                    "capture_strategy": "copy",
                }
            ],
        }
    )

    assert manifest.entries[0].required_for_resume is False
    assert manifest.warnings == ()
    assert "omits required_for_resume" in caplog.text
    assert "omits warnings" in caplog.text
