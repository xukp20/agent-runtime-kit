import pytest

from agent_runtime_kit.agent.templates import TemplateVariableError, render_template


def test_render_template_replaces_named_slots() -> None:
    assert render_template("hello {{ name }} in {{scope.id}}", {"name": "agent", "scope.id": "s1"}) == "hello agent in s1"


def test_render_template_allows_none_template() -> None:
    assert render_template(None, {}) is None


def test_render_template_raises_for_missing_slot() -> None:
    with pytest.raises(TemplateVariableError) as exc_info:
        render_template("hello {{ name }}", {})

    assert exc_info.value.name == "name"
