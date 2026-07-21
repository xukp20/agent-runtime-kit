from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable



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

    def close(self) -> None:
        return None


def _load_claude_sdk() -> object:
    try:
        return importlib.import_module("claude_agent_sdk")
    except ImportError as exc:
        raise ClaudeCodeSdkUnavailable(
            "Claude Code Provider requires the optional claude-agent-sdk package"
        ) from exc
