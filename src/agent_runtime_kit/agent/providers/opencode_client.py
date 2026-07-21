from __future__ import annotations

import base64
import json
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterator, Mapping


class OpenCodeClientError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class OpenCodeSseEvent:
    event: str
    data: object
    event_id: str | None = None


class OpenCodeClient:
    def __init__(
        self,
        base_url: str,
        *,
        password: str,
        directory: str,
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.password = password
        self.directory = directory
        self.timeout_s = timeout_s

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: object | None = None,
        query: Mapping[str, object] | None = None,
        timeout_s: float | None = None,
    ) -> object:
        url = self._url(path, query)
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, method=method)
        self._add_headers(request, json_body=payload is not None)
        try:
            with urllib.request.urlopen(request, timeout=timeout_s or self.timeout_s) as response:
                body = response.read()
                if not body:
                    return None
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OpenCodeClientError(
                f"OpenCode {method} {path} failed with HTTP {exc.code}: {_safe_body(body)}",
                status=exc.code,
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OpenCodeClientError(f"OpenCode {method} {path} failed: {exc}") from exc

    def health(self) -> Mapping[str, object]:
        value = self.request("GET", "/global/health", query={})
        if not isinstance(value, Mapping):
            raise OpenCodeClientError("OpenCode health response is not an object")
        return value

    def create_session(self) -> Mapping[str, object]:
        value = self.request("POST", "/session", payload={})
        if not isinstance(value, Mapping):
            raise OpenCodeClientError("OpenCode session response is not an object")
        return value

    def get_session(self, session_id: str) -> Mapping[str, object]:
        value = self.request("GET", f"/session/{_segment(session_id)}")
        if not isinstance(value, Mapping):
            raise OpenCodeClientError("OpenCode session response is not an object")
        return value

    def list_messages(self, session_id: str) -> list[object]:
        value = self.request("GET", f"/session/{_segment(session_id)}/message")
        if not isinstance(value, list):
            raise OpenCodeClientError("OpenCode messages response is not a list")
        return value

    def session_status(self) -> Mapping[str, object]:
        value = self.request("GET", "/session/status")
        if not isinstance(value, Mapping):
            raise OpenCodeClientError("OpenCode status response is not an object")
        return value

    def prompt_async(self, session_id: str, payload: Mapping[str, object]) -> None:
        self.request("POST", f"/session/{_segment(session_id)}/prompt_async", payload=payload)

    def abort(self, session_id: str) -> object:
        return self.request("POST", f"/session/{_segment(session_id)}/abort", payload={})

    def fork(self, session_id: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        value = self.request("POST", f"/session/{_segment(session_id)}/fork", payload=payload)
        if not isinstance(value, Mapping):
            raise OpenCodeClientError("OpenCode fork response is not an object")
        return value

    def summarize(self, session_id: str, payload: Mapping[str, object]) -> object:
        return self.request("POST", f"/session/{_segment(session_id)}/summarize", payload=payload)

    def reply_permission(self, permission_id: str, payload: Mapping[str, object]) -> object:
        return self.request("POST", f"/permission/{_segment(permission_id)}/reply", payload=payload)

    def reply_question(self, question_id: str, payload: Mapping[str, object]) -> object:
        return self.request("POST", f"/question/{_segment(question_id)}/reply", payload=payload)

    def reject_question(self, question_id: str) -> object:
        return self.request("POST", f"/question/{_segment(question_id)}/reject", payload={})

    def iter_events(self, stop: threading.Event) -> Iterator[OpenCodeSseEvent]:
        request = urllib.request.Request(self._url("/event", None), method="GET")
        self._add_headers(request, accept="text/event-stream")
        try:
            with urllib.request.urlopen(request, timeout=max(self.timeout_s, 60.0)) as response:
                event_name = "message"
                event_id: str | None = None
                data_lines: list[str] = []
                while not stop.is_set():
                    raw = response.readline()
                    if not raw:
                        if stop.is_set():
                            return
                        raise OpenCodeClientError("OpenCode SSE stream closed unexpectedly")
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line:
                        if data_lines:
                            text = "\n".join(data_lines)
                            try:
                                data: object = json.loads(text)
                            except json.JSONDecodeError:
                                data = text
                            yield OpenCodeSseEvent(event=event_name, data=data, event_id=event_id)
                        event_name, event_id, data_lines = "message", None, []
                        continue
                    if line.startswith(":"):
                        continue
                    field, _, value = line.partition(":")
                    value = value[1:] if value.startswith(" ") else value
                    if field == "event":
                        event_name = value
                    elif field == "id":
                        event_id = value
                    elif field == "data":
                        data_lines.append(value)
        except urllib.error.HTTPError as exc:
            raise OpenCodeClientError(
                f"OpenCode SSE failed with HTTP {exc.code}", status=exc.code
            ) from exc
        except urllib.error.URLError as exc:
            raise OpenCodeClientError(f"OpenCode SSE failed: {exc}") from exc

    def _url(self, path: str, query: Mapping[str, object] | None) -> str:
        values = {"directory": self.directory}
        if query is not None:
            values.update({key: str(value) for key, value in query.items()})
        return f"{self.base_url}{path}?{urllib.parse.urlencode(values)}"

    def _add_headers(
        self,
        request: urllib.request.Request,
        *,
        json_body: bool = False,
        accept: str = "application/json",
    ) -> None:
        token = base64.b64encode(f"opencode:{self.password}".encode()).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")
        request.add_header("Accept", accept)
        request.add_header("x-opencode-directory", self.directory)
        if json_body:
            request.add_header("Content-Type", "application/json")


def event_properties(value: object) -> tuple[str | None, Mapping[str, object]]:
    if not isinstance(value, Mapping):
        return None, {}
    event_type = value.get("type")
    properties = value.get("properties")
    return (
        str(event_type) if event_type is not None else None,
        properties if isinstance(properties, Mapping) else {},
    )


def _safe_body(body: str) -> str:
    # Provider error bodies can echo request/config data. Keep diagnostics bounded.
    value = body[:512].replace("\n", " ")
    value = re.sub(r"sk-[A-Za-z0-9_-]+", "<redacted>", value)
    value = re.sub(
        r'(?i)(authorization|api[_-]?key|token|secret)(["\s:=]+)([^,}\s"]+)',
        r"\1\2<redacted>",
        value,
    )
    return value


def _segment(value: str) -> str:
    return urllib.parse.quote(value, safe="")
