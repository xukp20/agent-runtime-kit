from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock


class RuntimePausedError(RuntimeError):
    """Raised when runtime pause gate blocks a new run."""


@dataclass
class RuntimePauseController:
    global_paused: bool = False
    paused_scopes: set[str] = field(default_factory=set)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def pause(self, scope_id: str | None = None) -> None:
        with self._lock:
            if scope_id is None:
                self.global_paused = True
            else:
                self.paused_scopes.add(scope_id)

    def resume(self, scope_id: str | None = None) -> None:
        with self._lock:
            if scope_id is None:
                self.global_paused = False
            else:
                self.paused_scopes.discard(scope_id)

    def is_paused(self, scope_id: str | None = None) -> bool:
        with self._lock:
            if self.global_paused:
                return True
            if scope_id is None:
                return False
            return scope_id in self.paused_scopes

    def is_scope_directly_paused(self, scope_id: str) -> bool:
        with self._lock:
            return scope_id in self.paused_scopes

    def assert_can_start(self, scope_id: str) -> None:
        if self.is_paused(scope_id):
            raise RuntimePausedError(f"runtime is paused for scope: {scope_id}")


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
    flow_service: object | None = None
    step_service: object | None = None
    schedule_service: object | None = None
    snapshot_service: object | None = None
    pause_controller: object | None = None
