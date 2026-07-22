from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .models import MissingProviderEnvError, to_jsonable
from .provider_contracts import (
    HomeMaterializationResult,
    HomeValidationResult,
    ProviderHomeSpec,
)
from .store_utils import utc_now_iso, write_json_atomic


@dataclass(frozen=True)
class HomeRef:
    provider_type: str
    home_id: str


@dataclass
class HomeRecord:
    provider_type: str
    home_id: str
    home_relpath: str
    status: str = "active"
    created_at: str = ""
    updated_at: str = ""
    fixed_env: dict[str, str] = field(default_factory=dict)
    required_env: set[str] = field(default_factory=set)
    schema_version: int = 3
    materialization_manifest_ref: str | None = None
    materialization_manifest_hash: str | None = None
    base_config_fingerprint: str | None = None
    resolved_defaults: dict[str, object] | None = None
    capability_snapshot: dict[str, object] | None = None
    warnings: list[str] = field(default_factory=list)
    provider_payload: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 3:
            raise ValueError(f"unsupported Home schema_version: {self.schema_version}")
        if not self.provider_type.strip() or not self.home_id.strip():
            raise ValueError("Home provider_type and home_id must not be empty")

    @property
    def home_root_ref(self) -> str:
        return self.home_relpath


@dataclass
class McpServerSpec:
    name: str
    enabled: bool = True
    transport: str = "http"
    url: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    cwd: str | None = None
    startup_timeout_sec: int | None = None
    tool_timeout_sec: int | None = None
    required: bool = False
    enabled_tools: list[str] | None = None
    disabled_tools: list[str] | None = None
    env: dict[str, str] = field(default_factory=dict)
    env_vars: list[str] = field(default_factory=list)
    bearer_token_env_var: str | None = None
    http_headers: dict[str, str] = field(default_factory=dict)
    env_http_headers: dict[str, str] = field(default_factory=dict)


class HomeStore:
    def __init__(self, runtime_root: Path) -> None:
        self.runtime_root = Path(runtime_root)
        self.homes_root = self.runtime_root / "homes"
        self.index_path = self.homes_root / "index.sqlite"
        self.homes_root.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.index_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists homes(
                  provider_type text not null,
                  home_id text not null,
                  home_relpath text not null,
                  status text not null,
                  created_at text not null,
                  updated_at text not null,
                  fixed_env_json text not null default '{}',
                  required_env_csv text not null default '',
                  schema_version integer not null,
                  materialization_manifest_ref text,
                  materialization_manifest_hash text,
                  base_config_fingerprint text,
                  resolved_defaults_json text,
                  capability_snapshot_json text,
                  warnings_json text not null default '[]',
                  provider_payload_json text,
                  primary key(provider_type, home_id)
                )
                """
            )

    def upsert_home(self, record: HomeRecord) -> HomeRecord:
        import json

        with self._connect() as conn:
            conn.execute(
                """
                insert into homes(
                  provider_type, home_id, home_relpath, status, created_at, updated_at,
                  fixed_env_json, required_env_csv, schema_version,
                  materialization_manifest_ref, materialization_manifest_hash,
                  base_config_fingerprint, resolved_defaults_json, capability_snapshot_json,
                  warnings_json, provider_payload_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(provider_type, home_id) do update set
                  home_relpath=excluded.home_relpath,
                  status=excluded.status,
                  updated_at=excluded.updated_at,
                  fixed_env_json=excluded.fixed_env_json,
                  required_env_csv=excluded.required_env_csv,
                  schema_version=excluded.schema_version,
                  materialization_manifest_ref=excluded.materialization_manifest_ref,
                  materialization_manifest_hash=excluded.materialization_manifest_hash,
                  base_config_fingerprint=excluded.base_config_fingerprint,
                  resolved_defaults_json=excluded.resolved_defaults_json,
                  capability_snapshot_json=excluded.capability_snapshot_json,
                  warnings_json=excluded.warnings_json,
                  provider_payload_json=excluded.provider_payload_json
                """,
                (
                    record.provider_type,
                    record.home_id,
                    record.home_relpath,
                    record.status,
                    record.created_at,
                    record.updated_at,
                    json.dumps(record.fixed_env, sort_keys=True),
                    ",".join(sorted(record.required_env)),
                    record.schema_version,
                    record.materialization_manifest_ref,
                    record.materialization_manifest_hash,
                    record.base_config_fingerprint,
                    json.dumps(record.resolved_defaults, sort_keys=True) if record.resolved_defaults else None,
                    json.dumps(record.capability_snapshot, sort_keys=True)
                    if record.capability_snapshot
                    else None,
                    json.dumps(record.warnings, sort_keys=True),
                    json.dumps(record.provider_payload, sort_keys=True) if record.provider_payload else None,
                ),
            )
        return record

    def get_home(self, provider_type: str, home_id: str) -> HomeRecord:
        with self._connect() as conn:
            row = conn.execute(
                """
                select provider_type, home_id, home_relpath, status, created_at, updated_at,
                       fixed_env_json, required_env_csv, schema_version,
                       materialization_manifest_ref, materialization_manifest_hash,
                       base_config_fingerprint, resolved_defaults_json, capability_snapshot_json,
                       warnings_json, provider_payload_json
                from homes where provider_type=? and home_id=?
                """,
                (provider_type, home_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown home: {provider_type}/{home_id}")
        return _home_from_row(row)

    def list_homes(self, provider_type: str | None = None, status: str | None = None) -> list[HomeRecord]:
        clauses: list[str] = []
        params: list[str] = []
        if provider_type is not None:
            clauses.append("provider_type=?")
            params.append(provider_type)
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        where = f" where {' and '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                """
                select provider_type, home_id, home_relpath, status, created_at, updated_at,
                       fixed_env_json, required_env_csv, schema_version,
                       materialization_manifest_ref, materialization_manifest_hash,
                       base_config_fingerprint, resolved_defaults_json, capability_snapshot_json,
                       warnings_json, provider_payload_json
                from homes
                """
                + where
                + " order by provider_type, home_id",
                params,
            ).fetchall()
        return [_home_from_row(row) for row in rows]

    def disable_home(self, provider_type: str, home_id: str) -> HomeRecord:
        record = self.get_home(provider_type, home_id)
        record.status = "disabled"
        record.updated_at = utc_now_iso()
        return self.upsert_home(record)

    def resolve_home_root(self, provider_type: str, home_id: str) -> Path:
        record = self.get_home(provider_type, home_id)
        return self.runtime_root / record.home_relpath


class HomeService:
    def __init__(
        self,
        runtime_root: Path,
        *,
        renderers: Mapping[str, object] | None = None,
    ) -> None:
        self.runtime_root = Path(runtime_root)
        self.store = HomeStore(self.runtime_root)
        if renderers is None:
            from .providers.codex_home import CodexHomeRenderer

            renderers = {"codex": CodexHomeRenderer(runtime_root=self.runtime_root)}
        self.renderers = dict(renderers)

    def create_home(self, spec: ProviderHomeSpec) -> HomeRecord:
        provider_type = spec.provider_type.strip()
        home_id = spec.home_id.strip()
        home_root = self.runtime_root / "homes" / provider_type / home_id
        home_root.mkdir(parents=True, exist_ok=True)
        renderer = self.renderers.get(provider_type)
        if renderer is None:
            raise ValueError(f"no Home renderer registered for provider: {provider_type}")
        else:
            validation = renderer.validate(spec)
            if not isinstance(validation, HomeValidationResult):
                raise TypeError(f"invalid HomeValidationResult from provider renderer: {provider_type}")
            if not validation.valid:
                message = "; ".join(validation.errors) or "provider home validation failed"
                raise ValueError(message)
            materialization = renderer.materialize(spec, home_root)
            if not isinstance(materialization, HomeMaterializationResult):
                raise TypeError(f"invalid HomeMaterializationResult from provider renderer: {provider_type}")
            if materialization.provider_type != provider_type or materialization.home_id != home_id:
                raise ValueError("provider renderer returned materialization for a different home")
        write_json_atomic(
            home_root / ".ark" / "home_materialization.json",
            to_jsonable(materialization),
        )
        now = utc_now_iso()
        existing_created_at = now
        try:
            existing_created_at = self.store.get_home(provider_type, home_id).created_at
        except KeyError:
            pass
        record = HomeRecord(
            provider_type=provider_type,
            home_id=home_id,
            home_relpath=str(Path("homes") / provider_type / home_id),
            status="active",
            created_at=existing_created_at,
            updated_at=now,
            fixed_env=dict(spec.fixed_env),
            required_env=set(spec.required_env),
            materialization_manifest_ref=str(
                Path("homes") / provider_type / home_id / ".ark" / "home_materialization.json"
            ),
            materialization_manifest_hash=materialization.manifest_hash,
            resolved_defaults=(
                to_jsonable(materialization.resolved_defaults)
                if materialization.resolved_defaults is not None
                else None
            ),
            capability_snapshot=(
                to_jsonable(materialization.effective_capabilities)
                if materialization.effective_capabilities is not None
                else None
            ),
            warnings=list(materialization.warnings),
            provider_payload=(
                to_jsonable(materialization.provider_payload)
                if materialization.provider_payload is not None
                else None
            ),
        )
        return self.store.upsert_home(record)

    def get_home(self, provider_type: str, home_id: str) -> HomeRecord:
        return self.store.get_home(provider_type, home_id)

    def resolve_home_root(self, provider_type: str, home_id: str) -> Path:
        return self.store.resolve_home_root(provider_type, home_id)

    def build_execution_context(
        self,
        provider_type: str,
        home_id: str,
        *,
        run_env: Mapping[str, str] | None = None,
        workdir: str | None = None,
    ) -> object:
        renderer = self.renderers.get(provider_type)
        if renderer is None:
            raise ValueError(f"no Home renderer registered for provider: {provider_type}")
        home = self.get_home(provider_type, home_id)
        return renderer.build_execution_context(home, run_env=run_env, workdir=workdir)

    def seal_home_materialization(self, provider_type: str, home_id: str) -> HomeRecord:
        """Explicitly accept application post-processing and refresh the manifest.

        Applications should call this once after they intentionally modify
        provider-managed Home files. Later unsealed mutations still fail hash
        verification when an execution context is built.
        """

        renderer = self.renderers.get(provider_type)
        refresh = getattr(renderer, "refresh_materialization", None)
        if not callable(refresh):
            raise ValueError(f"provider does not support Home materialization sealing: {provider_type}")
        home = self.get_home(provider_type, home_id)
        home_root = self.resolve_home_root(provider_type, home_id)
        materialization = refresh(home, home_root)
        return self._store_refreshed_materialization(
            provider_type,
            home_id,
            home,
            materialization,
        )

    def commit_provider_lifecycle_materialization(
        self,
        provider_type: str,
        home_id: str,
        *,
        lifecycle: str,
    ) -> HomeRecord:
        """Commit only renderer-declared changes at a trusted provider boundary."""

        renderer = self.renderers.get(provider_type)
        commit = getattr(renderer, "commit_lifecycle_materialization", None)
        if not callable(commit):
            raise ValueError(
                f"provider does not support lifecycle Home materialization commits: {provider_type}"
            )
        home = self.get_home(provider_type, home_id)
        home_root = self.resolve_home_root(provider_type, home_id)
        materialization = commit(home, home_root, lifecycle=lifecycle)
        if materialization is None:
            return home
        return self._store_refreshed_materialization(
            provider_type,
            home_id,
            home,
            materialization,
        )

    def _store_refreshed_materialization(
        self,
        provider_type: str,
        home_id: str,
        home: HomeRecord,
        materialization: object,
    ) -> HomeRecord:
        if not isinstance(materialization, HomeMaterializationResult):
            raise TypeError(f"invalid HomeMaterializationResult from provider renderer: {provider_type}")
        if materialization.provider_type != provider_type or materialization.home_id != home_id:
            raise ValueError("provider renderer refreshed materialization for a different home")
        home.materialization_manifest_hash = materialization.manifest_hash
        home.resolved_defaults = (
            to_jsonable(materialization.resolved_defaults)
            if materialization.resolved_defaults is not None
            else None
        )
        home.capability_snapshot = (
            to_jsonable(materialization.effective_capabilities)
            if materialization.effective_capabilities is not None
            else home.capability_snapshot
        )
        home.warnings = list(materialization.warnings)
        home.updated_at = utc_now_iso()
        return self.store.upsert_home(home)


def _home_from_row(row: tuple) -> HomeRecord:
    required = {item for item in str(row[7]).split(",") if item}
    return HomeRecord(
        provider_type=str(row[0]),
        home_id=str(row[1]),
        home_relpath=str(row[2]),
        status=str(row[3]),
        created_at=str(row[4]),
        updated_at=str(row[5]),
        fixed_env=dict(json.loads(row[6] or "{}")),
        required_env=required,
        schema_version=int(row[8] or 1),
        materialization_manifest_ref=row[9],
        materialization_manifest_hash=row[10],
        base_config_fingerprint=row[11],
        resolved_defaults=dict(json.loads(row[12])) if row[12] else None,
        capability_snapshot=dict(json.loads(row[13])) if row[13] else None,
        warnings=list(json.loads(row[14] or "[]")),
        provider_payload=dict(json.loads(row[15])) if row[15] else None,
    )
