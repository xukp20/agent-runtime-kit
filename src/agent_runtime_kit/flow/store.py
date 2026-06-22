from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Any, Callable

from agent_runtime_kit.agent.store_utils import encode_scope_id, read_json, utc_now_iso, write_json_atomic

from .models import BaseFlow, BaseStep, FlowStatus, StepStatus
from .registry import FlowTypeRegistry, StepTypeRegistry


class FlowStepStoreError(Exception):
    """Base store error for Flow / Step persistence."""


class FlowNotFoundError(FlowStepStoreError, KeyError):
    pass


class StepNotFoundError(FlowStepStoreError, KeyError):
    pass


class FlowStepStore:
    def __init__(
        self,
        runtime_root: Path,
        *,
        flow_registry: FlowTypeRegistry,
        step_registry: StepTypeRegistry,
    ) -> None:
        self.runtime_root = Path(runtime_root)
        self.scopes_root = self.runtime_root / "scopes"
        self.index_root = self.runtime_root / "index"
        self.global_index_path = self.index_root / "global.sqlite"
        self.flow_registry = flow_registry
        self.step_registry = step_registry
        self.lock = RLock()
        self.scopes_root.mkdir(parents=True, exist_ok=True)
        self.index_root.mkdir(parents=True, exist_ok=True)
        self._ensure_global_schema()

    def create_flow(self, flow: BaseFlow) -> BaseFlow:
        with self.lock:
            self._ensure_scope(flow.scope_id)
            path = self._flow_json_path(flow)
            if path.exists():
                raise FlowStepStoreError(f"flow already exists: {flow.flow_id}")
            self._write_flow(flow)
            self._upsert_flow_scope_index(flow)
            self._upsert_flow_global_index(flow)
            return flow

    def create_step(self, step: BaseStep) -> BaseStep:
        with self.lock:
            flow = self.get_flow(step.flow_id)
            if step.scope_id != flow.scope_id:
                raise FlowStepStoreError(
                    f"step {step.step_id} scope {step.scope_id} does not match flow {flow.flow_id} scope {flow.scope_id}"
                )
            path = self._step_json_path(step)
            if path.exists():
                raise FlowStepStoreError(f"step already exists: {step.step_id}")
            self._write_step(step)
            self._upsert_step_scope_index(step, flow)
            self._upsert_step_global_index(step, flow)
            return step

    def update_flow(self, flow: BaseFlow) -> BaseFlow:
        with self.lock:
            if not self._flow_json_path(flow).exists():
                raise FlowNotFoundError(f"unknown flow: {flow.flow_id}")
            flow.updated_at = utc_now_iso()
            self._write_flow(flow)
            self._upsert_flow_scope_index(flow)
            self._upsert_flow_global_index(flow)
            return flow

    def update_step(self, step: BaseStep) -> BaseStep:
        with self.lock:
            if not self._step_json_path(step).exists():
                raise StepNotFoundError(f"unknown step: {step.step_id}")
            flow = self.get_flow(step.flow_id)
            if step.scope_id != flow.scope_id:
                raise FlowStepStoreError(
                    f"step {step.step_id} scope {step.scope_id} does not match flow {flow.flow_id} scope {flow.scope_id}"
                )
            step.updated_at = utc_now_iso()
            self._write_step(step)
            self._upsert_step_scope_index(step, flow)
            self._upsert_step_global_index(step, flow)
            return step

    def update_flow_record(self, flow_id: str, mutator: Callable[[BaseFlow], None]) -> BaseFlow:
        flow = self.get_flow(flow_id)
        with self.edit_session(flow.scope_id) as tx:
            working = tx.load_flow_for_update(flow_id)
            mutator(working)
        return self.get_flow(flow_id)

    def update_step_record(self, step_id: str, mutator: Callable[[BaseStep], None]) -> BaseStep:
        step = self.get_step(step_id)
        with self.edit_session(step.scope_id) as tx:
            working = tx.load_step_for_update(step_id)
            mutator(working)
        return self.get_step(step_id)

    def edit_session(self, scope_id: str | None = None) -> "FlowStepMutationSession":
        return FlowStepMutationSession(self, scope_id=scope_id)

    def get_flow(self, flow_id: str) -> BaseFlow:
        return self._flow_from_payload(read_json(self.resolve_flow_path(flow_id)))

    def get_step(self, step_id: str) -> BaseStep:
        return self._step_from_payload(read_json(self.resolve_step_path(step_id)))

    def list_flows(
        self,
        *,
        scope_id: str | None = None,
        status: str | FlowStatus | None = None,
        flow_type: str | None = None,
    ) -> list[BaseFlow]:
        rows = self._query_rows(
            table="flows",
            relpath_column="flow_relpath",
            scope_id=scope_id,
            status=str(status) if status is not None else None,
            type_column="flow_type",
            type_value=flow_type,
            order_by="created_at, flow_id",
        )
        return [self._flow_from_payload(read_json(self.runtime_root / row[0])) for row in rows]

    def list_steps(
        self,
        *,
        scope_id: str | None = None,
        flow_id: str | None = None,
        status: str | StepStatus | None = None,
        step_type: str | None = None,
    ) -> list[BaseStep]:
        rows = self._query_rows(
            table="steps",
            relpath_column="step_relpath",
            scope_id=scope_id,
            status=str(status) if status is not None else None,
            type_column="step_type",
            type_value=step_type,
            extra_filters={"flow_id": flow_id} if flow_id is not None else None,
            order_by="created_at, step_id",
        )
        return [self._step_from_payload(read_json(self.runtime_root / row[0])) for row in rows]

    def list_non_terminal_flows(self, *, scope_id: str | None = None) -> list[BaseFlow]:
        return [
            flow
            for flow in self.list_flows(scope_id=scope_id)
            if flow.status not in {FlowStatus.COMPLETED, FlowStatus.FAILED}
        ]

    def list_created_steps(self, *, scope_id: str | None = None) -> list[BaseStep]:
        return self.list_steps(scope_id=scope_id, status=StepStatus.CREATED)

    def list_child_flows(
        self,
        *,
        parent_flow_id: str,
        parent_dispatch_step_id: str | None = None,
    ) -> list[BaseFlow]:
        query = "select flow_relpath from flows where parent_flow_id=?"
        params: list[str] = [parent_flow_id]
        if parent_dispatch_step_id is not None:
            query += " and parent_dispatch_step_id=?"
            params.append(parent_dispatch_step_id)
        query += " order by created_at, flow_id"
        with sqlite3.connect(self.global_index_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._flow_from_payload(read_json(self.runtime_root / row[0])) for row in rows]

    def resolve_flow_path(self, flow_id: str) -> Path:
        path = self._resolve_relpath_from_global("flows", "flow_id", "flow_relpath", flow_id)
        if path is not None and path.exists():
            return path
        self.rebuild_global_index()
        path = self._resolve_relpath_from_global("flows", "flow_id", "flow_relpath", flow_id)
        if path is None or not path.exists():
            raise FlowNotFoundError(f"unknown flow: {flow_id}")
        return path

    def resolve_step_path(self, step_id: str) -> Path:
        path = self._resolve_relpath_from_global("steps", "step_id", "step_relpath", step_id)
        if path is not None and path.exists():
            return path
        self.rebuild_global_index()
        path = self._resolve_relpath_from_global("steps", "step_id", "step_relpath", step_id)
        if path is None or not path.exists():
            raise StepNotFoundError(f"unknown step: {step_id}")
        return path

    def list_scope_ids(self) -> list[str]:
        scope_ids = []
        for scope_dir in sorted(self.scopes_root.iterdir()) if self.scopes_root.exists() else []:
            scope_json = scope_dir / "scope.json"
            if scope_json.exists():
                scope_ids.append(str(read_json(scope_json)["scope_id"]))
        return scope_ids

    def rebuild_scope_index(self, scope_id: str) -> None:
        with self.lock:
            scope_key = encode_scope_id(scope_id)
            self._ensure_scope_schema(scope_key)
            with sqlite3.connect(self._scope_index_path(scope_key)) as conn:
                conn.execute("delete from flows")
                conn.execute("delete from steps")
            flows_dir = self.scopes_root / scope_key / "flows"
            if not flows_dir.exists():
                return
            for flow_json in sorted(flows_dir.glob("*/flow.json")):
                flow = self._flow_from_payload(read_json(flow_json))
                self._upsert_flow_scope_index(flow)
                for step_json in sorted(flow_json.parent.glob("steps/*/step.json")):
                    step = self._step_from_payload(read_json(step_json))
                    self._upsert_step_scope_index(step, flow)

    def rebuild_global_index(self) -> None:
        with self.lock:
            self._ensure_global_schema()
            with sqlite3.connect(self.global_index_path) as conn:
                conn.execute("delete from flows")
                conn.execute("delete from steps")
            seen_flows: set[str] = set()
            seen_steps: set[str] = set()
            for scope_id in self.list_scope_ids():
                scope_key = encode_scope_id(scope_id)
                flows_dir = self.scopes_root / scope_key / "flows"
                if not flows_dir.exists():
                    continue
                for flow_json in sorted(flows_dir.glob("*/flow.json")):
                    flow = self._flow_from_payload(read_json(flow_json))
                    if flow.flow_id in seen_flows:
                        raise FlowStepStoreError(f"duplicate flow_id while rebuilding global index: {flow.flow_id}")
                    seen_flows.add(flow.flow_id)
                    self._upsert_flow_global_index(flow)
                    for step_json in sorted(flow_json.parent.glob("steps/*/step.json")):
                        step = self._step_from_payload(read_json(step_json))
                        if step.step_id in seen_steps:
                            raise FlowStepStoreError(f"duplicate step_id while rebuilding global index: {step.step_id}")
                        seen_steps.add(step.step_id)
                        self._upsert_step_global_index(step, flow)

    def rebuild_all_indexes(self) -> None:
        for scope_id in self.list_scope_ids():
            self.rebuild_scope_index(scope_id)
        self.rebuild_global_index()

    def assert_restorable_truth(self, *, scope_id: str | None = None) -> None:
        with self.lock:
            flows: dict[str, BaseFlow] = {}
            steps: dict[str, BaseStep] = {}
            target_scope_ids = [scope_id] if scope_id is not None else self.list_scope_ids()
            for target_scope_id in target_scope_ids:
                if target_scope_id is None:
                    continue
                scope_key = encode_scope_id(target_scope_id)
                flows_dir = self.scopes_root / scope_key / "flows"
                if not flows_dir.exists():
                    continue
                for flow_json in sorted(flows_dir.glob("*/flow.json")):
                    try:
                        flow = self._flow_from_payload(read_json(flow_json))
                    except BaseException as exc:
                        raise FlowStepStoreError(f"invalid flow truth at {flow_json}: {exc}") from exc
                    if flow.scope_id != target_scope_id:
                        raise FlowStepStoreError(
                            f"flow {flow.flow_id} scope {flow.scope_id} does not match directory scope {target_scope_id}"
                        )
                    if flow.flow_id in flows:
                        raise FlowStepStoreError(f"duplicate flow truth: {flow.flow_id}")
                    flows[flow.flow_id] = flow
                    for step_json in sorted(flow_json.parent.glob("steps/*/step.json")):
                        try:
                            step = self._step_from_payload(read_json(step_json))
                        except BaseException as exc:
                            raise FlowStepStoreError(f"invalid step truth at {step_json}: {exc}") from exc
                        if step.step_id in steps:
                            raise FlowStepStoreError(f"duplicate step truth: {step.step_id}")
                        if step.scope_id != flow.scope_id:
                            raise FlowStepStoreError(
                                f"step {step.step_id} scope {step.scope_id} does not match flow {flow.flow_id} scope {flow.scope_id}"
                            )
                        if step.flow_id != flow.flow_id:
                            raise FlowStepStoreError(
                                f"step {step.step_id} belongs to flow {step.flow_id}, expected {flow.flow_id}"
                            )
                        steps[step.step_id] = step

            for step in steps.values():
                if step.status is StepStatus.RUNNING:
                    raise FlowStepStoreError(f"step {step.step_id} is running")
                if step.flow_id not in flows:
                    raise FlowStepStoreError(f"step {step.step_id} references missing flow {step.flow_id}")

            for flow in flows.values():
                for step_id in flow.step_ids:
                    step = steps.get(step_id)
                    if step is None:
                        raise FlowStepStoreError(f"flow {flow.flow_id} step_ids references missing step {step_id}")
                    if step.flow_id != flow.flow_id:
                        raise FlowStepStoreError(
                            f"flow {flow.flow_id} step_ids references step {step_id} from flow {step.flow_id}"
                        )
                if flow.current_step_id is None:
                    continue
                current = steps.get(flow.current_step_id)
                if current is None:
                    raise FlowStepStoreError(
                        f"flow {flow.flow_id} current_step_id references missing step {flow.current_step_id}"
                    )
                if current.flow_id != flow.flow_id:
                    raise FlowStepStoreError(
                        f"flow {flow.flow_id} current_step_id references step {current.step_id} from flow {current.flow_id}"
                    )
                if flow.status in {FlowStatus.COMPLETED, FlowStatus.FAILED}:
                    raise FlowStepStoreError(f"terminal flow {flow.flow_id} still has current_step_id {current.step_id}")
                if current.status is not StepStatus.CREATED:
                    raise FlowStepStoreError(
                        f"flow {flow.flow_id} current step {current.step_id} is {current.status}, expected created"
                    )

    def _query_rows(
        self,
        *,
        table: str,
        relpath_column: str,
        scope_id: str | None,
        status: str | None,
        type_column: str,
        type_value: str | None,
        order_by: str,
        extra_filters: dict[str, str] | None = None,
    ) -> list[tuple[str]]:
        if scope_id is not None:
            scope_key = encode_scope_id(scope_id)
            index_path = self._scope_index_path(scope_key)
            index_exists = index_path.exists()
            self._ensure_scope_schema(scope_key)
            if not index_exists:
                self.rebuild_scope_index(scope_id)
        else:
            index_path = self.global_index_path
            if not index_path.exists():
                self.rebuild_global_index()

        clauses: list[str] = []
        params: list[str] = []
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        if type_value is not None:
            clauses.append(f"{type_column}=?")
            params.append(type_value)
        if extra_filters:
            for key, value in extra_filters.items():
                clauses.append(f"{key}=?")
                params.append(value)
        query = f"select {relpath_column} from {table}"
        if clauses:
            query += " where " + " and ".join(clauses)
        query += f" order by {order_by}"
        with sqlite3.connect(index_path) as conn:
            return conn.execute(query, params).fetchall()

    def _ensure_scope(self, scope_id: str) -> None:
        scope_key = encode_scope_id(scope_id)
        scope_dir = self.scopes_root / scope_key
        scope_dir.mkdir(parents=True, exist_ok=True)
        scope_json = scope_dir / "scope.json"
        now = utc_now_iso()
        if not scope_json.exists():
            write_json_atomic(
                scope_json,
                {
                    "schema_version": 1,
                    "object_type": "scope",
                    "scope_id": scope_id,
                    "scope_key": scope_key,
                    "created_at": now,
                    "updated_at": now,
                },
            )
        self._ensure_scope_schema(scope_key)

    def _flow_json_path(self, flow: BaseFlow) -> Path:
        return self.scopes_root / encode_scope_id(flow.scope_id) / "flows" / flow.flow_id / "flow.json"

    def _step_json_path(self, step: BaseStep) -> Path:
        return (
            self.scopes_root
            / encode_scope_id(step.scope_id)
            / "flows"
            / step.flow_id
            / "steps"
            / step.step_id
            / "step.json"
        )

    def _flow_relpath(self, flow: BaseFlow) -> str:
        return str(self._flow_json_path(flow).relative_to(self.runtime_root))

    def _step_relpath(self, step: BaseStep) -> str:
        return str(self._step_json_path(step).relative_to(self.runtime_root))

    def _write_flow(self, flow: BaseFlow) -> None:
        write_json_atomic(
            self._flow_json_path(flow),
            {
                "schema_version": 1,
                "object_type": "flow",
                **self._flow_payload(flow),
            },
        )

    def _write_step(self, step: BaseStep) -> None:
        write_json_atomic(
            self._step_json_path(step),
            {
                "schema_version": 1,
                "object_type": "step",
                **self._step_payload(step),
            },
        )

    def _flow_payload(self, flow: BaseFlow) -> dict[str, Any]:
        payload = flow.model_dump(mode="json")
        payload["flow_type"] = str(getattr(flow, "flow_type", payload.get("flow_type", "")))
        return payload

    def _flow_exists(self, flow_id: str) -> bool:
        try:
            self.resolve_flow_path(flow_id)
        except FlowNotFoundError:
            return False
        return True

    def _step_exists(self, step_id: str) -> bool:
        try:
            self.resolve_step_path(step_id)
        except StepNotFoundError:
            return False
        return True

    def _step_payload(self, step: BaseStep) -> dict[str, Any]:
        payload = step.model_dump(mode="json")
        payload["step_type"] = str(getattr(step, "step_type", payload.get("step_type", "")))
        return payload

    def _flow_from_payload(self, payload: dict[str, Any]) -> BaseFlow:
        data = self._strip_envelope(payload, expected_object_type="flow")
        flow_type = str(data["flow_type"])
        flow_cls = self.flow_registry.get(flow_type)
        data["state"] = self.flow_registry.parse_state(flow_type, data["state"])
        data["input"] = self.flow_registry.parse_input(flow_type, data.get("input"))
        data["result"] = self.flow_registry.parse_result(flow_type, data.get("result"))
        data["error"] = self.flow_registry.parse_error(flow_type, data.get("error"))
        if "flow_type" not in flow_cls.model_fields:
            data.pop("flow_type", None)
        return flow_cls.model_validate(data)

    def _step_from_payload(self, payload: dict[str, Any]) -> BaseStep:
        data = self._strip_envelope(payload, expected_object_type="step")
        step_type = str(data["step_type"])
        step_cls = self.step_registry.get(step_type)
        data["state"] = self.step_registry.parse_state(step_type, data["state"])
        data["submission"] = self.step_registry.parse_submission(step_type, data.get("submission"))
        data["result"] = self.step_registry.parse_result(step_type, data.get("result"))
        data["error"] = self.step_registry.parse_error(step_type, data.get("error"))
        if "step_type" not in step_cls.model_fields:
            data.pop("step_type", None)
        return step_cls.model_validate(data)

    def _strip_envelope(self, payload: dict[str, Any], *, expected_object_type: str) -> dict[str, Any]:
        object_type = payload.get("object_type")
        if object_type != expected_object_type:
            raise FlowStepStoreError(f"expected {expected_object_type} payload, got {object_type!r}")
        data = dict(payload)
        data.pop("schema_version", None)
        data.pop("object_type", None)
        return data

    def _scope_index_path(self, scope_key: str) -> Path:
        return self.scopes_root / scope_key / "index.sqlite"

    def _ensure_scope_schema(self, scope_key: str) -> None:
        path = self._scope_index_path(scope_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            _create_flow_step_schema(conn)

    def _ensure_global_schema(self) -> None:
        self.index_root.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.global_index_path) as conn:
            _create_flow_step_schema(conn)

    def _upsert_flow_scope_index(self, flow: BaseFlow) -> None:
        scope_key = encode_scope_id(flow.scope_id)
        self._ensure_scope_schema(scope_key)
        with sqlite3.connect(self._scope_index_path(scope_key)) as conn:
            _upsert_flow_row(conn, flow, scope_key, self._flow_relpath(flow))

    def _upsert_flow_global_index(self, flow: BaseFlow) -> None:
        self._ensure_global_schema()
        with sqlite3.connect(self.global_index_path) as conn:
            _upsert_flow_row(conn, flow, encode_scope_id(flow.scope_id), self._flow_relpath(flow))

    def _upsert_step_scope_index(self, step: BaseStep, flow: BaseFlow) -> None:
        scope_key = encode_scope_id(step.scope_id)
        self._ensure_scope_schema(scope_key)
        with sqlite3.connect(self._scope_index_path(scope_key)) as conn:
            _upsert_step_row(conn, step, flow, scope_key, self._step_relpath(step))

    def _upsert_step_global_index(self, step: BaseStep, flow: BaseFlow) -> None:
        self._ensure_global_schema()
        with sqlite3.connect(self.global_index_path) as conn:
            _upsert_step_row(conn, step, flow, encode_scope_id(step.scope_id), self._step_relpath(step))

    def _resolve_relpath_from_global(
        self,
        table: str,
        id_column: str,
        relpath_column: str,
        object_id: str,
    ) -> Path | None:
        if not self.global_index_path.exists():
            return None
        with sqlite3.connect(self.global_index_path) as conn:
            row = conn.execute(
                f"select {relpath_column} from {table} where {id_column}=?",
                (object_id,),
            ).fetchone()
        if row is None:
            return None
        return self.runtime_root / str(row[0])


def _create_flow_step_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        create table if not exists flows(
          flow_id text primary key,
          scope_id text not null,
          scope_key text not null,
          flow_type text not null,
          status text not null,
          parent_flow_id text,
          parent_dispatch_step_id text,
          current_step_id text,
          flow_relpath text not null,
          created_at text not null,
          updated_at text not null
        )
        """
    )
    conn.execute("create index if not exists idx_flows_type_status on flows(flow_type, status)")
    conn.execute("create index if not exists idx_flows_status_updated on flows(status, updated_at)")
    conn.execute("create index if not exists idx_flows_parent on flows(parent_flow_id, parent_dispatch_step_id)")
    conn.execute("create index if not exists idx_flows_scope_status on flows(scope_id, status)")
    conn.execute(
        """
        create table if not exists steps(
          step_id text primary key,
          scope_id text not null,
          scope_key text not null,
          flow_id text not null,
          flow_type text not null,
          step_type text not null,
          status text not null,
          step_relpath text not null,
          created_at text not null,
          updated_at text not null
        )
        """
    )
    conn.execute("create index if not exists idx_steps_flow on steps(flow_id, created_at)")
    conn.execute("create index if not exists idx_steps_type_status on steps(step_type, status)")
    conn.execute("create index if not exists idx_steps_status_updated on steps(status, updated_at)")
    conn.execute("create index if not exists idx_steps_scope_status on steps(scope_id, status)")


def _upsert_flow_row(conn: sqlite3.Connection, flow: BaseFlow, scope_key: str, flow_relpath: str) -> None:
    conn.execute(
        """
        insert into flows(
          flow_id, scope_id, scope_key, flow_type, status,
          parent_flow_id, parent_dispatch_step_id, current_step_id,
          flow_relpath, created_at, updated_at
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(flow_id) do update set
          scope_id=excluded.scope_id,
          scope_key=excluded.scope_key,
          flow_type=excluded.flow_type,
          status=excluded.status,
          parent_flow_id=excluded.parent_flow_id,
          parent_dispatch_step_id=excluded.parent_dispatch_step_id,
          current_step_id=excluded.current_step_id,
          flow_relpath=excluded.flow_relpath,
          updated_at=excluded.updated_at
        """,
        (
            flow.flow_id,
            flow.scope_id,
            scope_key,
            str(getattr(flow, "flow_type")),
            str(flow.status),
            flow.parent_flow_id,
            flow.parent_dispatch_step_id,
            flow.current_step_id,
            flow_relpath,
            flow.created_at,
            flow.updated_at,
        ),
    )


def _upsert_step_row(
    conn: sqlite3.Connection,
    step: BaseStep,
    flow: BaseFlow,
    scope_key: str,
    step_relpath: str,
) -> None:
    conn.execute(
        """
        insert into steps(
          step_id, scope_id, scope_key, flow_id, flow_type,
          step_type, status, step_relpath, created_at, updated_at
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(step_id) do update set
          scope_id=excluded.scope_id,
          scope_key=excluded.scope_key,
          flow_id=excluded.flow_id,
          flow_type=excluded.flow_type,
          step_type=excluded.step_type,
          status=excluded.status,
          step_relpath=excluded.step_relpath,
          updated_at=excluded.updated_at
        """,
        (
            step.step_id,
            step.scope_id,
            scope_key,
            step.flow_id,
            str(getattr(flow, "flow_type")),
            str(getattr(step, "step_type")),
            str(step.status),
            step_relpath,
            step.created_at,
            step.updated_at,
        ),
    )


class FlowStepMutationSession:
    def __init__(self, store: FlowStepStore, *, scope_id: str | None = None) -> None:
        self.store = store
        self.scope_id = scope_id
        self.working_flows: dict[str, BaseFlow] = {}
        self.working_steps: dict[str, BaseStep] = {}
        self.new_flows: dict[str, BaseFlow] = {}
        self.new_steps: dict[str, BaseStep] = {}
        self._active = False

    def __enter__(self) -> "FlowStepMutationSession":
        self.store.lock.acquire()
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        try:
            if exc_type is None:
                self.flush()
        finally:
            self._active = False
            self.store.lock.release()
        return False

    def load_flow_for_update(self, flow_id: str) -> BaseFlow:
        self._assert_active()
        if flow_id in self.new_flows:
            return self.new_flows[flow_id]
        if flow_id not in self.working_flows:
            flow = self.store.get_flow(flow_id)
            self._check_scope(flow.scope_id, object_id=flow_id)
            self.working_flows[flow_id] = flow
        return self.working_flows[flow_id]

    def load_step_for_update(self, step_id: str) -> BaseStep:
        self._assert_active()
        if step_id in self.new_steps:
            return self.new_steps[step_id]
        if step_id not in self.working_steps:
            step = self.store.get_step(step_id)
            self._check_scope(step.scope_id, object_id=step_id)
            self.working_steps[step_id] = step
        return self.working_steps[step_id]

    def load_flow(self, flow_id: str) -> BaseFlow:
        return self.load_flow_for_update(flow_id)

    def load_step(self, step_id: str) -> BaseStep:
        return self.load_step_for_update(step_id)

    def stage_flow(self, flow: BaseFlow) -> None:
        self._assert_active()
        self._check_scope(flow.scope_id, object_id=flow.flow_id)
        if flow.flow_id in self.new_flows:
            self.new_flows[flow.flow_id] = flow
        else:
            self.working_flows[flow.flow_id] = flow

    def stage_step(self, step: BaseStep) -> None:
        self._assert_active()
        self._check_scope(step.scope_id, object_id=step.step_id)
        if step.step_id in self.new_steps:
            self.new_steps[step.step_id] = step
        else:
            self.working_steps[step.step_id] = step

    def add_flow(self, flow: BaseFlow) -> str:
        self._assert_active()
        self._check_scope(flow.scope_id, object_id=flow.flow_id)
        if flow.flow_id in self.working_flows or flow.flow_id in self.new_flows or self.store._flow_exists(flow.flow_id):
            raise FlowStepStoreError(f"flow already exists: {flow.flow_id}")
        self.new_flows[flow.flow_id] = flow
        return flow.flow_id

    def add_step(self, step: BaseStep) -> str:
        self._assert_active()
        self._check_scope(step.scope_id, object_id=step.step_id)
        if step.step_id in self.working_steps or step.step_id in self.new_steps or self.store._step_exists(step.step_id):
            raise FlowStepStoreError(f"step already exists: {step.step_id}")
        self._flow_for_step(step)
        self.new_steps[step.step_id] = step
        return step.step_id

    def mark_dirty(self, obj: BaseFlow | BaseStep) -> None:
        self._assert_active()
        if isinstance(obj, BaseFlow):
            self.stage_flow(obj)
        elif isinstance(obj, BaseStep):
            self.stage_step(obj)
        else:
            raise TypeError(f"unsupported dirty object: {type(obj).__name__}")

    def flush(self) -> None:
        self._assert_active()
        for flow in self.new_flows.values():
            self.store._ensure_scope(flow.scope_id)
            self.store._write_flow(flow)
            self.store._upsert_flow_scope_index(flow)
            self.store._upsert_flow_global_index(flow)
        for flow in self.working_flows.values():
            flow.updated_at = utc_now_iso()
            self.store._write_flow(flow)
            self.store._upsert_flow_scope_index(flow)
            self.store._upsert_flow_global_index(flow)
        for step in self.new_steps.values():
            flow = self._flow_for_step(step)
            self.store._write_step(step)
            self.store._upsert_step_scope_index(step, flow)
            self.store._upsert_step_global_index(step, flow)
        for step in self.working_steps.values():
            step.updated_at = utc_now_iso()
            flow = self._flow_for_step(step)
            self.store._write_step(step)
            self.store._upsert_step_scope_index(step, flow)
            self.store._upsert_step_global_index(step, flow)

    def _flow_for_step(self, step: BaseStep) -> BaseFlow:
        flow = self.new_flows.get(step.flow_id) or self.working_flows.get(step.flow_id)
        if flow is None:
            flow = self.store.get_flow(step.flow_id)
        if step.scope_id != flow.scope_id:
            raise FlowStepStoreError(
                f"step {step.step_id} scope {step.scope_id} does not match flow {flow.flow_id} scope {flow.scope_id}"
            )
        return flow

    def _check_scope(self, scope_id: str, *, object_id: str) -> None:
        if self.scope_id is not None and scope_id != self.scope_id:
            raise FlowStepStoreError(f"object {object_id} is in scope {scope_id}, expected {self.scope_id}")

    def _assert_active(self) -> None:
        if not self._active:
            raise FlowStepStoreError("mutation session is not active")
