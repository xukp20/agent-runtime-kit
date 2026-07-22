from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..provider_contracts import ProviderContextUsage
from ..store_utils import utc_now_iso


@dataclass(frozen=True)
class CodexRolloutEventLocator:
    line_number: int
    byte_end: int
    timestamp: str | None


@dataclass(frozen=True)
class CodexCompactBaseline:
    path: str
    device: int
    inode: int
    complete_byte_count: int
    compacted_count: int
    context_compacted_count: int
    token_count_count: int
    latest_compacted: CodexRolloutEventLocator | None
    latest_context_compacted: CodexRolloutEventLocator | None
    latest_token_count: CodexRolloutEventLocator | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "device": self.device,
            "inode": self.inode,
            "complete_byte_count": self.complete_byte_count,
            "compacted_count": self.compacted_count,
            "context_compacted_count": self.context_compacted_count,
            "token_count_count": self.token_count_count,
            "latest_compacted": _locator_dict(self.latest_compacted),
            "latest_context_compacted": _locator_dict(self.latest_context_compacted),
            "latest_token_count": _locator_dict(self.latest_token_count),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CodexCompactBaseline":
        return cls(
            path=str(payload["path"]),
            device=int(payload["device"]),
            inode=int(payload["inode"]),
            complete_byte_count=int(payload["complete_byte_count"]),
            compacted_count=int(payload["compacted_count"]),
            context_compacted_count=int(payload["context_compacted_count"]),
            token_count_count=int(payload["token_count_count"]),
            latest_compacted=_locator_from_dict(payload.get("latest_compacted")),
            latest_context_compacted=_locator_from_dict(payload.get("latest_context_compacted")),
            latest_token_count=_locator_from_dict(payload.get("latest_token_count")),
        )


@dataclass(frozen=True)
class CodexCompactEvidence:
    baseline: CodexCompactBaseline
    current: CodexCompactBaseline
    usage: ProviderContextUsage
    has_new_compacted: bool
    has_new_context_compacted: bool
    has_new_token_count: bool

    @property
    def complete(self) -> bool:
        return self.has_new_compacted and self.has_new_context_compacted and self.has_new_token_count


@dataclass(frozen=True)
class _RolloutScan:
    baseline: CodexCompactBaseline
    latest_usage: ProviderContextUsage


def inspect_codex_rollout_context(
    rollout_path: Path,
    *,
    session_id: str | None,
) -> ProviderContextUsage:
    return _scan_rollout(Path(rollout_path), session_id=session_id).latest_usage


def capture_codex_compact_baseline(
    rollout_path: Path,
    *,
    session_id: str | None,
) -> CodexCompactBaseline:
    return _scan_rollout(Path(rollout_path), session_id=session_id).baseline


def inspect_codex_compact_evidence(
    rollout_path: Path,
    *,
    session_id: str | None,
    baseline: CodexCompactBaseline,
) -> CodexCompactEvidence:
    scan = _scan_rollout(Path(rollout_path), session_id=session_id)
    current = scan.baseline
    if current.device != baseline.device or current.inode != baseline.inode:
        raise ValueError("Codex rollout was replaced while compacting")
    if current.complete_byte_count < baseline.complete_byte_count:
        raise ValueError("Codex rollout was truncated while compacting")
    return CodexCompactEvidence(
        baseline=baseline,
        current=current,
        usage=scan.latest_usage,
        has_new_compacted=current.compacted_count > baseline.compacted_count,
        has_new_context_compacted=current.context_compacted_count > baseline.context_compacted_count,
        has_new_token_count=current.token_count_count > baseline.token_count_count,
    )


def _scan_rollout(path: Path, *, session_id: str | None) -> _RolloutScan:
    if not path.is_file():
        return _RolloutScan(
            baseline=_empty_baseline(path),
            latest_usage=_unavailable_usage(session_id, "rollout_missing"),
        )

    stat = path.stat()
    compacted_count = 0
    context_compacted_count = 0
    token_count_count = 0
    latest_compacted = None
    latest_context_compacted = None
    latest_token_count = None
    latest_usage = _unavailable_usage(session_id, "token_count_missing")
    complete_byte_count = 0

    with path.open("rb") as handle:
        line_number = 0
        while raw_line := handle.readline():
            line_number += 1
            if not raw_line.endswith(b"\n"):
                break
            complete_byte_count = handle.tell()
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid Codex rollout JSON at complete line {line_number}") from exc
            if not isinstance(event, dict):
                continue
            locator = CodexRolloutEventLocator(
                line_number=line_number,
                byte_end=complete_byte_count,
                timestamp=_optional_str(event.get("timestamp")),
            )
            event_type = event.get("type")
            payload = event.get("payload")
            payload = payload if isinstance(payload, dict) else {}
            payload_type = payload.get("type")
            if event_type == "compacted":
                compacted_count += 1
                latest_compacted = locator
            elif event_type == "event_msg" and payload_type == "context_compacted":
                context_compacted_count += 1
                latest_context_compacted = locator
            elif event_type == "event_msg" and payload_type == "token_count":
                token_count_count += 1
                latest_token_count = locator
                latest_usage = _usage_from_token_count(payload, session_id=session_id)

    return _RolloutScan(
        baseline=CodexCompactBaseline(
            path=str(path),
            device=stat.st_dev,
            inode=stat.st_ino,
            complete_byte_count=complete_byte_count,
            compacted_count=compacted_count,
            context_compacted_count=context_compacted_count,
            token_count_count=token_count_count,
            latest_compacted=latest_compacted,
            latest_context_compacted=latest_context_compacted,
            latest_token_count=latest_token_count,
        ),
        latest_usage=latest_usage,
    )


def _usage_from_token_count(payload: dict[str, Any], *, session_id: str | None) -> ProviderContextUsage:
    info = payload.get("info")
    if not isinstance(info, dict):
        return _unavailable_usage(session_id, "token_count_info_missing")
    last_usage = info.get("last_token_usage")
    if not isinstance(last_usage, dict):
        return _unavailable_usage(session_id, "last_token_usage_missing")
    total_tokens = _optional_non_negative_int(last_usage.get("total_tokens"))
    context_window = _optional_positive_int(info.get("model_context_window"))
    if total_tokens is None:
        return _unavailable_usage(session_id, "total_tokens_missing", context_window=context_window)
    if context_window is None:
        return _unavailable_usage(session_id, "context_window_missing", total_tokens=total_tokens)
    return ProviderContextUsage(
        session_id=session_id,
        observed_at=utc_now_iso(),
        source="artifact",
        available=True,
        used_tokens=total_tokens,
        context_window_tokens=context_window,
        effective_context_window_tokens=context_window,
        remaining_tokens=max(context_window - total_tokens, 0),
        measurement="provider_artifact",
    )


def _unavailable_usage(
    session_id: str | None,
    reason: str,
    *,
    total_tokens: int | None = None,
    context_window: int | None = None,
) -> ProviderContextUsage:
    return ProviderContextUsage(
        session_id=session_id,
        observed_at=utc_now_iso(),
        source="artifact",
        available=False,
        used_tokens=total_tokens,
        context_window_tokens=context_window,
        effective_context_window_tokens=context_window,
        measurement="provider_artifact" if total_tokens is not None else "unavailable",
        reason=reason,
    )


def _empty_baseline(path: Path) -> CodexCompactBaseline:
    return CodexCompactBaseline(
        path=str(path),
        device=0,
        inode=0,
        complete_byte_count=0,
        compacted_count=0,
        context_compacted_count=0,
        token_count_count=0,
        latest_compacted=None,
        latest_context_compacted=None,
        latest_token_count=None,
    )


def _optional_non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _locator_dict(locator: CodexRolloutEventLocator | None) -> dict[str, Any] | None:
    if locator is None:
        return None
    return {
        "line_number": locator.line_number,
        "byte_end": locator.byte_end,
        "timestamp": locator.timestamp,
    }


def _locator_from_dict(payload: object) -> CodexRolloutEventLocator | None:
    if not isinstance(payload, dict):
        return None
    return CodexRolloutEventLocator(
        line_number=int(payload["line_number"]),
        byte_end=int(payload["byte_end"]),
        timestamp=_optional_str(payload.get("timestamp")),
    )
