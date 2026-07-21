from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable

from ..provider_contracts import ProviderSessionLocator, ProviderSessionQuery, ProviderTurnQuery
from ..store_utils import utc_now_iso


class ClaudeCodeSdkUnavailable(RuntimeError):
    pass


class ClaudeCodeProvider:
    """Claude Code Provider facade with a lazy Claude Agent SDK dependency."""

    provider_type = "claude_code"

    def __init__(
        self,
        *,
        runtime_root: Path | None = None,
        sdk_loader: Callable[[], object] | None = None,
    ) -> None:
        self.runtime_root = Path(runtime_root) if runtime_root is not None else None
        self._sdk_loader = sdk_loader or _load_claude_sdk
        self._sdk_module: object | None = None

    def sdk(self) -> object:
        if self._sdk_module is None:
            self._sdk_module = self._sdk_loader()
        return self._sdk_module

    def build_provider_bundle(self, *, runtime_root: Path):
        from .claude_code_bundle import build_claude_code_provider_bundle

        return build_claude_code_provider_bundle(self, runtime_root=runtime_root)

    def read_thread(
        self,
        agent: object,
        *,
        home_root: Path,
        env: object | None = None,
        include_turns: bool = True,
    ) -> object:
        del env
        from .claude_code_query import ClaudeCodeQueryAdapter

        locator = _agent_locator(agent)
        return ClaudeCodeQueryAdapter(
            runtime_root=self._runtime_root(home_root),
            provider=self,
        ).read_session(ProviderSessionQuery(locator=locator, include_turns=include_turns))

    def read_latest_turn_result(
        self,
        agent: object,
        *,
        home_root: Path,
        env: object | None = None,
    ) -> object:
        del env
        from .claude_code_query import ClaudeCodeQueryAdapter

        locator = _agent_locator(agent)
        view = ClaudeCodeQueryAdapter(
            runtime_root=self._runtime_root(home_root),
            provider=self,
        ).read_turn(ProviderTurnQuery(session=locator, latest=True))
        if view is None or view.result is None:
            raise RuntimeError(f"Claude session has no completed turn: {locator.session_id}")
        return view.result

    def close(self) -> None:
        return None

    def _runtime_root(self, home_root: Path) -> Path:
        if self.runtime_root is not None:
            return self.runtime_root
        path = Path(home_root).resolve()
        if len(path.parents) < 3:
            raise RuntimeError(f"cannot infer runtime root from Claude Home: {path}")
        return path.parents[2]


def _load_claude_sdk() -> object:
    try:
        return importlib.import_module("claude_agent_sdk")
    except ImportError as exc:
        raise ClaudeCodeSdkUnavailable(
            "Claude Code Provider requires the optional claude-agent-sdk package"
        ) from exc


def _agent_locator(agent: object) -> ProviderSessionLocator:
    existing = getattr(agent, "session_locator", None)
    if isinstance(existing, ProviderSessionLocator):
        return existing
    session_id = getattr(agent, "thread_id", None)
    if not session_id:
        raise RuntimeError("Claude Agent has no session id")
    return ProviderSessionLocator(
        provider_type="claude_code",
        session_id=str(session_id),
        home_id=str(getattr(agent, "home_id")),
        created_at=str(getattr(agent, "created_at", "") or utc_now_iso()),
    )
