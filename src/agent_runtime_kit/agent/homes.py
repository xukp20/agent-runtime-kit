from __future__ import annotations

import os
import json
import re
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .models import MissingProviderEnvError
from .skills import SkillSpec, validate_skill_name, write_skill_spec
from .store_utils import utc_now_iso


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

    def upsert_home(self, record: HomeRecord) -> HomeRecord:
        import json

        with self._connect() as conn:
            conn.execute(
                """
                insert into homes(
                  cli_type, home_id, home_relpath, status, created_at, updated_at,
                  fixed_env_json, required_env_csv
                )
                values (?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(cli_type, home_id) do update set
                  home_relpath=excluded.home_relpath,
                  status=excluded.status,
                  updated_at=excluded.updated_at,
                  fixed_env_json=excluded.fixed_env_json,
                  required_env_csv=excluded.required_env_csv
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
                ),
            )
        return record

    def get_home(self, cli_type: str, home_id: str) -> HomeRecord:
        with self._connect() as conn:
            row = conn.execute(
                """
                select cli_type, home_id, home_relpath, status, created_at, updated_at,
                       fixed_env_json, required_env_csv
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
                       fixed_env_json, required_env_csv
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
    def __init__(self, runtime_root: Path) -> None:
        self.runtime_root = Path(runtime_root)
        self.store = HomeStore(self.runtime_root)

    def create_home(self, spec: HomeCreateSpec) -> HomeRecord:
        cli_type = spec.cli_type.strip()
        home_id = spec.home_id.strip()
        if not cli_type or not home_id:
            raise ValueError("cli_type and home_id must not be empty")
        home_root = self.runtime_root / "homes" / cli_type / home_id
        home_root.mkdir(parents=True, exist_ok=True)
        if cli_type == "codex":
            self._create_codex_home(home_root, spec)
        elif spec.model_config_overrides is not None:
            raise ValueError(f"model configuration overrides are not supported for provider: {cli_type}")
        now = utc_now_iso()
        existing_created_at = now
        try:
            existing_created_at = self.store.get_home(cli_type, home_id).created_at
        except KeyError:
            pass
        record = HomeRecord(
            cli_type=cli_type,
            home_id=home_id,
            home_relpath=str(Path("homes") / cli_type / home_id),
            status="active",
            created_at=existing_created_at,
            updated_at=now,
            fixed_env=dict(spec.fixed_env),
            required_env=set(spec.required_env),
        )
        return self.store.upsert_home(record)

    def _create_codex_home(self, home_root: Path, spec: HomeCreateSpec) -> None:
        codex_root = home_root / ".codex"
        agents_root = home_root / ".agents"
        skills_root = agents_root / "skills"
        codex_root.mkdir(parents=True, exist_ok=True)
        skills_root.mkdir(parents=True, exist_ok=True)
        self._write_codex_config(codex_root / "config.toml", spec)
        if spec.auth_json_path is not None:
            shutil.copyfile(spec.auth_json_path, codex_root / "auth.json")
        self._validate_skill_inputs(spec)
        for skill_name, skill_path in spec.skill_paths.items():
            validated_name = validate_skill_name(skill_name)
            if not skill_path.exists() or not skill_path.is_dir():
                raise ValueError(f"skill path must be an existing directory: {skill_path}")
            dest = skills_root / validated_name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(skill_path, dest)
        for skill_name, skill_spec in spec.skill_specs.items():
            validated_name = validate_skill_name(skill_name)
            if validated_name != skill_spec.name:
                raise ValueError(f"skill spec key must match SkillSpec.name: {skill_name} != {skill_spec.name}")
            write_skill_spec(skill_spec, skills_root / validated_name)

    def _validate_skill_inputs(self, spec: HomeCreateSpec) -> None:
        path_names = {validate_skill_name(name) for name in spec.skill_paths}
        spec_names = {validate_skill_name(name) for name in spec.skill_specs}
        duplicate_names = path_names & spec_names
        if duplicate_names:
            duplicates = ", ".join(sorted(duplicate_names))
            raise ValueError(f"duplicate skill names between skill_paths and skill_specs: {duplicates}")
        for skill_name in spec.skill_paths:
            if validate_skill_name(skill_name) != skill_name:
                raise ValueError(f"invalid skill path name: {skill_name}")
        for skill_name, skill_spec in spec.skill_specs.items():
            if validate_skill_name(skill_name) != skill_spec.name:
                raise ValueError(f"skill spec key must match SkillSpec.name: {skill_name} != {skill_spec.name}")

    def _write_codex_config(self, config_path: Path, spec: HomeCreateSpec) -> None:
        if spec.base_config_path is None and not spec.mcp_servers and spec.model_config_overrides is None:
            return
        if spec.base_config_path is not None:
            text = spec.base_config_path.read_text(encoding="utf-8")
        else:
            text = ""
        if spec.model_config_overrides is not None:
            text = _apply_codex_model_config_overrides(text, spec.model_config_overrides)
        if spec.mcp_servers:
            text = text.rstrip() + "\n\n" + _render_mcp_servers_toml(spec.mcp_servers)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(text, encoding="utf-8")

    def get_home(self, cli_type: str, home_id: str) -> HomeRecord:
        return self.store.get_home(cli_type, home_id)

    def resolve_home_root(self, cli_type: str, home_id: str) -> Path:
        return self.store.resolve_home_root(cli_type, home_id)


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
    if home.cli_type == "codex":
        env["HOME"] = str(home_root)
        env["CODEX_HOME"] = str(home_root / ".codex")
    return env


def _render_mcp_servers_toml(servers: list[McpServerSpec]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for server in servers:
        name = server.name.strip()
        if not name:
            raise ValueError("MCP server name must not be empty")
        if name in seen:
            raise ValueError(f"duplicate MCP server name: {name}")
        seen.add(name)
        lines.append(f"[mcp_servers.{_toml_key(name)}]")
        if server.transport and server.url and server.transport != "http":
            lines.append(f"transport = {_toml_value(server.transport)}")
        for field_name in [
            "enabled",
            "url",
            "command",
            "args",
            "cwd",
            "startup_timeout_sec",
            "tool_timeout_sec",
            "required",
            "enabled_tools",
            "disabled_tools",
            "env_vars",
            "bearer_token_env_var",
        ]:
            value = getattr(server, field_name)
            if value is None:
                continue
            if isinstance(value, list) and not value:
                continue
            lines.append(f"{field_name} = {_toml_value(value)}")
        for table_name in ["env", "http_headers", "env_http_headers"]:
            values = getattr(server, table_name)
            if not values:
                continue
            lines.append("")
            lines.append(f"[mcp_servers.{_toml_key(name)}.{table_name}]")
            for key, value in sorted(values.items()):
                lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _apply_codex_model_config_overrides(text: str, overrides: ModelConfigOverrides) -> str:
    replacements = {
        "model": overrides.model,
        "model_reasoning_effort": overrides.reasoning_effort,
    }
    active = {key: value for key, value in replacements.items() if value is not None}
    if not active:
        return text

    lines = text.splitlines()
    first_table = next(
        (index for index, line in enumerate(lines) if line.strip().startswith("[")),
        len(lines),
    )
    found: set[str] = set()
    rendered: list[str] = []
    for index, line in enumerate(lines):
        if index < first_table and "=" in line and not line.lstrip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in active:
                if key not in found:
                    rendered.append(f"{key} = {_toml_value(active[key])}")
                    found.add(key)
                continue
        rendered.append(line)

    missing = [key for key in active if key not in found]
    if missing:
        insert_at = next(
            (index for index, line in enumerate(rendered) if line.strip().startswith("[")),
            len(rendered),
        )
        additions = [f"{key} = {_toml_value(active[key])}" for key in missing]
        if insert_at and rendered[insert_at - 1].strip():
            additions.append("")
        rendered[insert_at:insert_at] = additions
    return "\n".join(rendered).rstrip() + "\n"


def _toml_key(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return value
    return json.dumps(value)


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {value!r}")


def _home_from_row(row: tuple) -> HomeRecord:
    import json

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
    )
