from __future__ import annotations

from dataclasses import dataclass


class AppServices:
    """Base object for application-provided services."""

    def validate(self) -> None:
        pass


@dataclass
class ARKServices:
    """Shared references to ARK services.

    The container is intentionally mutable so services can be constructed first
    and then registered into the same shared object.
    """

    agent_service: object | None = None
