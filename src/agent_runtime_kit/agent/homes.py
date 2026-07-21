from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .models import MissingProviderEnvError, to_jsonable
from .provider_contracts import (
    BaseConfigSource,
    HomeMaterializationResult,
    HomeValidationResult,
    ProviderHomeSpec,
)
from .skills import SkillSpec
from .store_utils import utc_now_iso, write_json_atomic


@dataclass(frozen=True)
class HomeRef:
    cli_type: str
    home_id: str


@dataclass
class HomeRecord:
    cli_type: str
    home_id: str
    home_relpath: str
    status: str = "active"
    created_at: str = ""
    updated_at: str = ""
    fixed_env: dict[str, str] = field(default_factory=dict)
    required_env: set[str] = field(default_factory=set)
    schema_version: int = 2
    provider_type: str | None = None
    materialization_manifest_ref: str | None = None
    materialization_manifest_hash: str | None = None
    base_config_fingerprint: str | None = None
    resolved_defaults: dict[str, object] | None = None
    capability_snapshot: dict[str, object] | None = None
    warnings: list[str] = field(default_factory=list)
    provider_payload: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.provider_type is None:
            self.provider_type = self.cli_type
        if self.provider_type != self.cli_type:
            raise ValueError("provider_type and legacy cli_type must match during compatibility migration")

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


@dataclass(frozen=True)
class ModelConfigOverrides:
    """Provider-neutral model settings projected by a concrete Home renderer."""

    model: str | None = None
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("model", "reasoning_effort"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} override must be a non-empty string")


@dataclass
class HomeCreateSpec:
    cli_type: str
    home_id: str
    base_config_path: Path | None = None
    auth_json_path: Path | None = None
    skill_paths: dict[str, Path] = field(default_factory=dict)
    skill_specs: dict[str, SkillSpec] = field(default_factory=dict)
    mcp_servers: list[McpServerSpec] = field(default_factory=list)
    fixed_env: dict[str, str] = field(default_factory=dict)
    required_env: set[str] = field(default_factory=set)
    model_config_overrides: ModelConfigOverrides | None = None
    provider_type: str | None = None
    config_overrides: dict[str, object] = field(default_factory=dict)
    provider_options: object | None = None

    def resolved_provider_type(self) -> str:
        provider_type = (self.provider_type or self.cli_type).strip()
        cli_type = self.cli_type.strip()
        if provider_type != cli_type:
            raise ValueError("provider_type and legacy cli_type must match during compatibility migration")
        return provider_type


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
                  cli_type text not null,
                  home_id text not null,
                  home_relpath text not null,
                  status text not null,
                  created_at text not null,
                  updated_at text not null,
                  fixed_env_json text not null default '{}',
                  required_env_csv text not null default '',
                  primary key(cli_type, home_id)
                )
                """
            )
            columns = {str(row[1]) for row in conn.execute("pragma table_info(homes)").fetchall()}
            additions = {
                "schema_version": "integer not null default 1",
                "provider_type": "text",
                "materialization_manifest_ref": "text",
                "materialization_manifest_hash": "text",
                "base_config_fingerprint": "text",
                "resolved_defaults_json": "text",
                "capability_snapshot_json": "text",
                "warnings_json": "text not null default '[]'",
                "provider_payload_json": "text",
            }
            for name, declaration in additions.items():
                if name not in columns:
                    conn.execute(f"alter table homes add column {name} {declaration}")

    def upsert_home(self, record: HomeRecord) -> HomeRecord:
        import json

        with self._connect() as conn:
            conn.execute(
                """
                insert into homes(
                  cli_type, home_id, home_relpath, status, created_at, updated_at,
                  fixed_env_json, required_env_csv, schema_version, provider_type,
                  materialization_manifest_ref, materialization_manifest_hash,
                  base_config_fingerprint, resolved_defaults_json, capability_snapshot_json,
                  warnings_json, provider_payload_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(cli_type, home_id) do update set
                  home_relpath=excluded.home_relpath,
                  status=excluded.status,
                  updated_at=excluded.updated_at,
                  fixed_env_json=excluded.fixed_env_json,
                  required_env_csv=excluded.required_env_csv,
                  schema_version=excluded.schema_version,
                  provider_type=excluded.provider_type,
                  materialization_manifest_ref=excluded.materialization_manifest_ref,
                  materialization_manifest_hash=excluded.materialization_manifest_hash,
                  base_config_fingerprint=excluded.base_config_fingerprint,
                  resolved_defaults_json=excluded.resolved_defaults_json,
                  capability_snapshot_json=excluded.capability_snapshot_json,
                  warnings_json=excluded.warnings_json,
                  provider_payload_json=excluded.provider_payload_json
                """,
                (
                    record.cli_type,
                    record.home_id,
                    record.home_relpath,
                    record.status,
                    record.created_at,
                    record.updated_at,
                    json.dumps(record.fixed_env, sort_keys=True),
                    ",".join(sorted(record.required_env)),
                    record.schema_version,
                    record.provider_type,
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

    def get_home(self, cli_type: str, home_id: str) -> HomeRecord:
        with self._connect() as conn:
            row = conn.execute(
                """
                select cli_type, home_id, home_relpath, status, created_at, updated_at,
                       fixed_env_json, required_env_csv, schema_version, provider_type,
                       materialization_manifest_ref, materialization_manifest_hash,
                       base_config_fingerprint, resolved_defaults_json, capability_snapshot_json,
                       warnings_json, provider_payload_json
                from homes where cli_type=? and home_id=?
                """,
                (cli_type, home_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown home: {cli_type}/{home_id}")
        return _home_from_row(row)

    def list_homes(self, cli_type: str | None = None, status: str | None = None) -> list[HomeRecord]:
        clauses: list[str] = []
        params: list[str] = []
        if cli_type is not None:
            clauses.append("cli_type=?")
            params.append(cli_type)
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        where = f" where {' and '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                """
                select cli_type, home_id, home_relpath, status, created_at, updated_at,
                       fixed_env_json, required_env_csv, schema_version, provider_type,
                       materialization_manifest_ref, materialization_manifest_hash,
                       base_config_fingerprint, resolved_defaults_json, capability_snapshot_json,
                       warnings_json, provider_payload_json
                from homes
                """
                + where
                + " order by cli_type, home_id",
                params,
            ).fetchall()
        return [_home_from_row(row) for row in rows]

    def disable_home(self, cli_type: str, home_id: str) -> HomeRecord:
        record = self.get_home(cli_type, home_id)
        record.status = "disabled"
        record.updated_at = utc_now_iso()
        return self.upsert_home(record)

    def resolve_home_root(self, cli_type: str, home_id: str) -> Path:
        record = self.get_home(cli_type, home_id)
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

    def create_home(self, spec: HomeCreateSpec | ProviderHomeSpec) -> HomeRecord:
        provider_spec, legacy_spec = self._normalize_spec(spec)
        provider_type = provider_spec.provider_type.strip()
        home_id = provider_spec.home_id.strip()
        home_root = self.runtime_root / "homes" / provider_type / home_id
        home_root.mkdir(parents=True, exist_ok=True)
        renderer = self.renderers.get(provider_type)
        if renderer is None:
            materialization = self._materialize_legacy_empty_home(provider_spec, legacy_spec, home_root)
        else:
            validation = renderer.validate(provider_spec)
            if not isinstance(validation, HomeValidationResult):
                raise TypeError(f"invalid HomeValidationResult from provider renderer: {provider_type}")
            if not validation.valid:
                message = "; ".join(validation.errors) or "provider home validation failed"
                raise ValueError(message)
            materialization = renderer.materialize(provider_spec, home_root)
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
            cli_type=provider_type,
            provider_type=provider_type,
            home_id=home_id,
            home_relpath=str(Path("homes") / provider_type / home_id),
            status="active",
            created_at=existing_created_at,
            updated_at=now,
            fixed_env=dict(provider_spec.fixed_env),
            required_env=set(provider_spec.required_env),
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

    def _normalize_spec(
        self,
        spec: HomeCreateSpec | ProviderHomeSpec,
    ) -> tuple[ProviderHomeSpec, HomeCreateSpec | None]:
        if isinstance(spec, ProviderHomeSpec):
            return spec, None
        provider_type = spec.resolved_provider_type()
        base_config = BaseConfigSource(path=str(spec.base_config_path)) if spec.base_config_path else None
        from .providers.codex_home import CodexHomeOptions

        provider_options = spec.provider_options
        if provider_type == "codex":
            if provider_options is not None:
                raise ValueError("legacy Codex HomeCreateSpec cannot combine provider_options with Codex fields")
            provider_options = CodexHomeOptions(
                auth_json_path=spec.auth_json_path,
                skill_paths=dict(spec.skill_paths),
                skill_specs=dict(spec.skill_specs),
                mcp_servers=tuple(spec.mcp_servers),
                model_config_overrides=spec.model_config_overrides,
            )
        return (
            ProviderHomeSpec(
                provider_type=provider_type,
                home_id=spec.home_id,
                base_config=base_config,
                config_overrides=dict(spec.config_overrides),
                mcp_servers=tuple(spec.mcp_servers),
                skills=tuple(spec.skill_specs.values()),
                fixed_env=dict(spec.fixed_env),
                required_env=tuple(sorted(spec.required_env)),
                provider_options=provider_options,
            ),
            spec,
        )

    def _materialize_legacy_empty_home(
        self,
        provider_spec: ProviderHomeSpec,
        legacy_spec: HomeCreateSpec | None,
        home_root: Path,
    ) -> HomeMaterializationResult:
        # COMPAT(legacy-unregistered-home): older callers could create an empty
        # home for injected providers. Remove when every injected provider is
        # registered through ProviderRegistry with an explicit Home renderer.
        # Covered by external takeover and runtime-matrix compatibility tests.
        if legacy_spec is not None and legacy_spec.model_config_overrides is not None:
            raise ValueError(
                f"model configuration overrides are not supported for provider: {provider_spec.provider_type}"
            )
        if legacy_spec is None or any(
            (
                provider_spec.base_config is not None,
                bool(provider_spec.config_overrides),
                bool(provider_spec.skills),
                bool(provider_spec.mcp_servers),
                provider_spec.provider_options is not None,
            )
        ):
            raise ValueError(f"no Home renderer registered for provider: {provider_spec.provider_type}")
        result = HomeMaterializationResult(
            provider_type=provider_spec.provider_type,
            home_id=provider_spec.home_id,
            renderer_version="ark-empty-compat-1",
            manifest_schema_version=1,
            manifest_hash="ark-empty-home-v1",
            required_env=provider_spec.required_env,
            warnings=("legacy empty home without registered Provider renderer",),
        )
        manifest_path = home_root / ".ark" / "home_materialization.json"
        write_json_atomic(manifest_path, to_jsonable(result))
        return result

    def get_home(self, cli_type: str, home_id: str) -> HomeRecord:
        return self.store.get_home(cli_type, home_id)

    def resolve_home_root(self, cli_type: str, home_id: str) -> Path:
        return self.store.resolve_home_root(cli_type, home_id)

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


def build_provider_env(
    *,
    home: HomeRecord,
    home_root: Path,
    run_env: Mapping[str, str] | None = None,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env.update(home.fixed_env)
    if run_env:
        env.update(dict(run_env))
    for name in home.required_env:
        if not env.get(name):
            raise MissingProviderEnvError(name)
    # COMPAT(legacy-build-provider-env): legacy callers construct provider
    # environment without HomeService.build_execution_context(). Remove once
    # LC and all external providers consume ProviderExecutionContext.
    if home.cli_type == "codex":
        env["HOME"] = str(home_root)
        env["CODEX_HOME"] = str(home_root / ".codex")
    return env


def _home_from_row(row: tuple) -> HomeRecord:
    required = {item for item in str(row[7]).split(",") if item}
    return HomeRecord(
        cli_type=str(row[0]),
        home_id=str(row[1]),
        home_relpath=str(row[2]),
        status=str(row[3]),
        created_at=str(row[4]),
        updated_at=str(row[5]),
        fixed_env=dict(json.loads(row[6] or "{}")),
        required_env=required,
        schema_version=int(row[8] or 1),
        provider_type=str(row[9] or row[0]),
        materialization_manifest_ref=row[10],
        materialization_manifest_hash=row[11],
        base_config_fingerprint=row[12],
        resolved_defaults=dict(json.loads(row[13])) if row[13] else None,
        capability_snapshot=dict(json.loads(row[14])) if row[14] else None,
        warnings=list(json.loads(row[15] or "[]")),
        provider_payload=dict(json.loads(row[16])) if row[16] else None,
    )
