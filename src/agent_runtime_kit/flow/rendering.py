from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent_runtime_kit.runtime import ARKServices, AppServices


@dataclass(frozen=True)
class RenderContext:
    ark: ARKServices
    app: AppServices
    scope_id: str
    viewer: Literal["agent", "admin"] = "agent"
