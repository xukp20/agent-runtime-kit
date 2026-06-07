from __future__ import annotations

from dataclasses import dataclass

from .services import ARKServices, AppServices


@dataclass(frozen=True)
class RuntimeContext:
    ark: ARKServices
    app: AppServices
