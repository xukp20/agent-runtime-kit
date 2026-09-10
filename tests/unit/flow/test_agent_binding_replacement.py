from pathlib import Path

from agent_runtime_kit.agent.models import Agent
import pytest

from agent_runtime_kit.flow import FlowRequest, FlowStatus, FlowStepValidationError

from test_agent_step_restart import _service


def test_replace_bound_agent_fresh_preserves_source_and_updates_only_target_binding(tmp_path: Path) -> None:
    source = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, _, _, agents = _service(tmp_path, agent=source)
    flow_id = service.start_flow(
        FlowRequest(flow_type="restart_flow", scope_id="scope", params={}),
        enqueue=False,
    )
    service.store.update_flow_record(
        flow_id,
        lambda flow: flow.agent_bindings.by_role.__setitem__("reviewer", source.agent_id),
    )

    receipt = service.replace_bound_agent(
        flow_id=flow_id,
        role="reviewer",
        expected_agent_id=source.agent_id,
        replacement_mode="fresh",
    )

    assert receipt.replacement_agent_id == "fresh-1"
    assert service.get_flow(flow_id).agent_bindings.get("reviewer") == "fresh-1"
    assert agents.get_agent(source.agent_id) == source
    assert agents.get_agent("fresh-1").session_locator is None


@pytest.mark.parametrize("status", [FlowStatus.COMPLETED, FlowStatus.FAILED])
def test_replace_bound_agent_rejects_terminal_flow_before_creating_agent(
    tmp_path: Path,
    status: FlowStatus,
) -> None:
    source = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, _, _, agents = _service(tmp_path, agent=source)
    flow_id = service.start_flow(
        FlowRequest(flow_type="restart_flow", scope_id="scope", params={}),
        enqueue=False,
    )

    def terminal(flow):  # noqa: ANN001
        flow.status = status
        flow.agent_bindings.by_role["reviewer"] = source.agent_id

    service.store.update_flow_record(flow_id, terminal)
    preimage = service.get_flow(flow_id).model_dump(mode="json")

    with pytest.raises(FlowStepValidationError, match="terminal Flow"):
        service.replace_bound_agent(
            flow_id=flow_id,
            role="reviewer",
            expected_agent_id=source.agent_id,
            replacement_mode="fresh",
        )

    assert agents.created == []
    assert service.get_flow(flow_id).model_dump(mode="json") == preimage
