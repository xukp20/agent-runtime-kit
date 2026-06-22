from typing import Literal

from agent_runtime_kit.flow import BaseFlowInput, BaseFlowResult, BaseStepResult, RenderContext
from agent_runtime_kit.runtime import ARKServices, AppServices


def test_render_context_carries_runtime_and_viewer() -> None:
    ark = ARKServices()
    app = AppServices()
    ctx = RenderContext(ark=ark, app=app, scope_id="scope", viewer="admin")

    assert ctx.ark is ark
    assert ctx.app is app
    assert ctx.scope_id == "scope"
    assert ctx.viewer == "admin"


def test_base_render_methods_require_business_subclasses() -> None:
    ctx = RenderContext(ark=ARKServices(), app=AppServices(), scope_id="scope")

    for item in [
        BaseFlowInput(input_type="input", summary="input summary"),
        BaseFlowResult(result_type="flow_result", summary="flow summary"),
        BaseStepResult(result_type="step_result", summary="step summary"),
    ]:
        try:
            item.render_for_agent(ctx)
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"{type(item).__name__}.render_for_agent did not require override")


def test_business_render_subclasses_can_return_agent_text() -> None:
    class DemoFlowInput(BaseFlowInput):
        input_type: Literal["demo"] = "demo"
        target: str

        def render_for_agent(self, ctx: RenderContext) -> str:
            return f"Target: {self.target}"

    rendered = DemoFlowInput(target="Main theorem").render_for_agent(
        RenderContext(ark=ARKServices(), app=AppServices(), scope_id="scope")
    )

    assert rendered == "Target: Main theorem"
