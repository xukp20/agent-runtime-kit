from __future__ import annotations

import json
import subprocess
import threading
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from time import monotonic


class PiRpcError(RuntimeError):
    pass


class PiRpcProcess:
    """Synchronous controller for Pi's LF-delimited JSONL RPC mode."""

    def __init__(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
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
        )
        self._condition = threading.Condition()
        self._write_lock = threading.Lock()
        self._responses: dict[str, dict[str, object]] = {}
        self._records: list[dict[str, object]] = []
        self._stderr: list[str] = []
        self._error: BaseException | None = None
        self._closed = False
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

    def command(
        self,
        command_type: str,
        payload: Mapping[str, object] | None = None,
        *,
        timeout_s: float = 30.0,
    ) -> dict[str, object]:
        request_id = f"ark-{uuid.uuid4().hex}"
        request = {"id": request_id, "type": command_type, **dict(payload or {})}
        self._write(request)
        deadline = monotonic() + timeout_s
        with self._condition:
            while request_id not in self._responses:
                self._raise_if_unusable()
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Pi RPC command timed out: {command_type}")
                self._condition.wait(min(remaining, 0.1))
            response = self._responses.pop(request_id)
        if response.get("type") != "response" or response.get("command") != command_type:
            raise PiRpcError(f"Pi RPC returned mismatched response for {command_type}")
        if response.get("success") is not True:
            raise PiRpcError(str(response.get("error") or f"Pi RPC {command_type} failed"))
        return response

    def wait_for(
        self,
        predicate: Callable[[dict[str, object]], bool],
        *,
        after_index: int = 0,
        timeout_s: float = 30.0,
    ) -> tuple[dict[str, object], int]:
        deadline = monotonic() + timeout_s
        cursor = max(after_index, 0)
        with self._condition:
            while True:
                while cursor < len(self._records):
                    record = self._records[cursor]
                    cursor += 1
                    if predicate(record):
                        return record, cursor
                self._raise_if_unusable()
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError("Pi RPC event wait timed out")
                self._condition.wait(min(remaining, 0.1))

    def send_extension_response(self, payload: Mapping[str, object]) -> None:
        if payload.get("type") != "extension_ui_response":
            raise ValueError("Pi extension response must use type=extension_ui_response")
        self._write(dict(payload))

    def close(self, *, timeout_s: float = 10.0) -> None:
        if self._closed:
            return
        self._closed = True
        if self.process.stdin is not None and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        with self._condition:
            self._condition.notify_all()

    def terminate(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
        self.close(timeout_s=5)

    def _write(self, record: Mapping[str, object]) -> None:
        if self._closed or self.process.stdin is None or self.process.stdin.closed:
            raise PiRpcError("Pi RPC process is closed")
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        try:
            with self._write_lock:
                self.process.stdin.write(encoded + "\n")
                self.process.stdin.flush()
        except OSError as exc:
            raise PiRpcError("failed to write Pi RPC command") from exc

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PiRpcError("Pi RPC stdout contained non-JSON data") from exc
                if not isinstance(value, dict):
                    raise PiRpcError("Pi RPC stdout record must be an object")
                with self._condition:
                    response_id = value.get("id")
                    if value.get("type") == "response" and isinstance(response_id, str):
                        if response_id in self._responses:
                            raise PiRpcError(f"duplicate Pi RPC response id: {response_id}")
                        self._responses[response_id] = value
                    self._records.append(value)
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            with self._condition:
                self._condition.notify_all()

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            with self._condition:
                self._stderr.append(line)
                if sum(map(len, self._stderr)) > 8192:
                    self._stderr = ["".join(self._stderr)[-4096:]]

    def _raise_if_unusable(self) -> None:
        if self._error is not None:
            raise PiRpcError(str(self._error)) from self._error
        returncode = self.process.poll()
        if returncode is not None:
            suffix = f": {self.stderr_tail}" if self.stderr_tail else ""
            raise PiRpcError(f"Pi RPC exited with code {returncode}{suffix}")
