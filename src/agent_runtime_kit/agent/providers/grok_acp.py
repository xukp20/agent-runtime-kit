from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from time import monotonic, sleep


class GrokAcpError(RuntimeError):
    pass


class GrokAcpProcess:
    """Narrow synchronous ACP v1 JSONL controller for Grok 1.0.30."""

    def __init__(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        incoming_request: Callable[[dict[str, object]], dict[str, object]] | None = None,
        on_record: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.command_line = tuple(command)
        self.process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=dict(env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            start_new_session=True,
        )
        self.process_group_id = self.process.pid
        self._incoming_request = incoming_request
        self._on_record = on_record
        self._condition = threading.Condition()
        self._write_lock = threading.Lock()
        self._responses: dict[str, dict[str, object]] = {}
        self._pending: set[str] = set()
        self._records: list[dict[str, object]] = []
        self._stderr: list[str] = []
        self._error: BaseException | None = None
        self._closed = False
        self._cleanup_confirmed = False
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._reader.start()
        self._stderr_reader.start()

    @property
    def records(self) -> tuple[dict[str, object], ...]:
        with self._condition:
            return tuple(self._records)

    @property
    def stderr_tail(self) -> str:
        with self._condition:
            return "".join(self._stderr)[-4096:]

    @property
    def cleanup_confirmed(self) -> bool:
        return self._cleanup_confirmed

    def request(
        self,
        method: str,
        params: Mapping[str, object] | None = None,
        *,
        timeout_s: float = 30.0,
    ) -> object:
        request_id = f"ark-{uuid.uuid4().hex}"
        with self._condition:
            self._pending.add(request_id)
        try:
            self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params or {})})
            deadline = monotonic() + timeout_s
            with self._condition:
                while request_id not in self._responses:
                    self._raise_if_unusable()
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"Grok ACP request timed out: {method}")
                    self._condition.wait(min(remaining, 0.1))
                response = self._responses.pop(request_id)
        finally:
            with self._condition:
                self._pending.discard(request_id)
                self._responses.pop(request_id, None)
        if "error" in response:
            error = response.get("error")
            message = error.get("message") if isinstance(error, Mapping) else error
            raise GrokAcpError(str(message or f"Grok ACP {method} failed"))
        if "result" not in response:
            raise GrokAcpError(f"Grok ACP response has no result: {method}")
        return response["result"]

    def notify(self, method: str, params: Mapping[str, object] | None = None) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": dict(params or {})})

    def close_process_group(
        self,
        *,
        close_timeout_s: float = 1.0,
        term_timeout_s: float = 5.0,
        kill_timeout_s: float = 5.0,
    ) -> None:
        if self._cleanup_confirmed:
            return
        self._closed = True
        if self.process.stdin is not None and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        self._wait_leader(close_timeout_s)
        if self._group_exists():
            self._signal_group(signal.SIGTERM)
            self._wait_group(term_timeout_s)
        if self._group_exists():
            self._signal_group(signal.SIGKILL)
            self._wait_group(kill_timeout_s)
        self._wait_leader(0.2)
        if self._group_exists():
            raise GrokAcpError("Grok managed process group survived SIGKILL")
        self._cleanup_confirmed = True
        with self._condition:
            self._condition.notify_all()

    def terminate_process_group(self) -> None:
        self._closed = True
        self._signal_group(signal.SIGTERM)
        self._wait_group(2.0)
        if self._group_exists():
            self._signal_group(signal.SIGKILL)
            self._wait_group(3.0)
        self._wait_leader(0.2)
        if self._group_exists():
            raise GrokAcpError("Grok managed process group could not be terminated")
        self._cleanup_confirmed = True

    def _write(self, record: Mapping[str, object]) -> None:
        if self._closed or self.process.stdin is None or self.process.stdin.closed:
            raise GrokAcpError("Grok ACP process is closed")
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        try:
            with self._write_lock:
                self.process.stdin.write(encoded + "\n")
                self.process.stdin.flush()
        except OSError as exc:
            raise GrokAcpError("failed to write Grok ACP record") from exc

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise GrokAcpError("Grok ACP stdout contained non-JSON data") from exc
                if not isinstance(value, dict):
                    raise GrokAcpError("Grok ACP stdout record must be an object")
                response_id = value.get("id")
                method = value.get("method")
                is_incoming = isinstance(method, str) and response_id is not None
                with self._condition:
                    if response_id is not None and not is_incoming:
                        key = str(response_id)
                        if key in self._pending and key in self._responses:
                            raise GrokAcpError(f"duplicate Grok ACP response id: {key}")
                        if key in self._pending:
                            self._responses[key] = value
                    self._records.append(value)
                    self._condition.notify_all()
                if self._on_record is not None:
                    self._on_record(value)
                if is_incoming:
                    self._respond_to_incoming(value)
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            with self._condition:
                self._condition.notify_all()

    def _respond_to_incoming(self, record: dict[str, object]) -> None:
        response_id = record["id"]
        try:
            if self._incoming_request is None:
                raise GrokAcpError("client interaction is unsupported")
            result = self._incoming_request(record)
            self._write({"jsonrpc": "2.0", "id": response_id, "result": result})
        except BaseException as exc:
            try:
                self._write(
                    {
                        "jsonrpc": "2.0",
                        "id": response_id,
                        "error": {"code": -32601, "message": str(exc)},
                    }
                )
            except BaseException:
                pass

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            with self._condition:
                self._stderr.append(line)
                if sum(map(len, self._stderr)) > 8192:
                    self._stderr = ["".join(self._stderr)[-4096:]]

    def _raise_if_unusable(self) -> None:
        if self._error is not None:
            raise GrokAcpError(str(self._error)) from self._error
        code = self.process.poll()
        if code is not None:
            suffix = f": {self.stderr_tail}" if self.stderr_tail else ""
            raise GrokAcpError(f"Grok ACP exited with code {code}{suffix}")

    def _wait_leader(self, timeout_s: float) -> None:
        try:
            self.process.wait(timeout=max(timeout_s, 0.0))
        except subprocess.TimeoutExpired:
            pass

    def _group_exists(self) -> bool:
        try:
            os.killpg(self.process_group_id, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _signal_group(self, sig: signal.Signals) -> None:
        try:
            os.killpg(self.process_group_id, sig)
        except ProcessLookupError:
            pass

    def _wait_group(self, timeout_s: float) -> None:
        deadline = monotonic() + timeout_s
        while self._group_exists() and monotonic() < deadline:
            self._wait_leader(0.05)
            sleep(0.02)
