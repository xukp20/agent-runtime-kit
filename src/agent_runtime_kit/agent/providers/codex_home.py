from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, TYPE_CHECKING

from ..models import MissingProviderEnvError, to_jsonable
from ..provider_contracts import (
    CapabilityKey,
    CapabilityStatus,
    CapabilitySupport,
    HomeInitializationResult,
    HomeMaterializationResult,
    HomeMaterializedFile,
    HomeValidationResult,
    ModelBackendIdentity,
    ProviderCapabilities,
    ProviderExecutionContext,
    ProviderHomeSpec,
)
from ..skills import SkillSpec, validate_skill_name, write_skill_spec
from ..store_utils import read_json, write_json_atomic

if TYPE_CHECKING:
    from ..homes import HomeRecord, McpServerSpec, ModelConfigOverrides


@dataclass(frozen=True)
class CodexHomeOptions:
    auth_json_path: Path | None = None
    skill_paths: Mapping[str, Path] = field(default_factory=dict)
    skill_specs: Mapping[str, SkillSpec] = field(default_factory=dict)
    mcp_servers: tuple["McpServerSpec", ...] = ()
    model_config_overrides: "ModelConfigOverrides | None" = None


class CodexHomeRenderer:
    provider_type = "codex"
    renderer_version = "codex-home-v1"

    def __init__(self, *, runtime_root: Path, provider: object | None = None) -> None:
        self.runtime_root = Path(runtime_root)
        self.provider = provider

    def validate(self, spec: ProviderHomeSpec) -> HomeValidationResult:
        errors: list[str] = []
        if spec.provider_type != self.provider_type:
            errors.append(f"Codex renderer cannot materialize provider {spec.provider_type}")
        options = spec.provider_options
        if options is not None and not isinstance(options, CodexHomeOptions):
            errors.append("Codex provider_options must be CodexHomeOptions")
        if isinstance(options, CodexHomeOptions):
            try:
                _validate_skill_inputs(options)
            except ValueError as exc:
                errors.append(str(exc))
            if options.auth_json_path is not None and not options.auth_json_path.is_file():
                errors.append(f"auth_json_path must be an existing file: {options.auth_json_path}")
        return HomeValidationResult(valid=not errors, errors=tuple(errors))

    def materialize(self, spec: ProviderHomeSpec, home_root: Path) -> HomeMaterializationResult:
        validation = self.validate(spec)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        options = spec.provider_options if isinstance(spec.provider_options, CodexHomeOptions) else CodexHomeOptions()
        resolved_home_root = Path(home_root)
        codex_root = resolved_home_root / ".codex"
        skills_root = resolved_home_root / ".agents" / "skills"
        codex_root.mkdir(parents=True, exist_ok=True)
        skills_root.mkdir(parents=True, exist_ok=True)

        config_text = _read_base_config(spec)
        if spec.config_overrides:
            config_text = _apply_top_level_overrides(config_text, spec.config_overrides)
        if options.model_config_overrides is not None:
            config_text = _apply_codex_model_config_overrides(config_text, options.model_config_overrides)
        mcp_servers = options.mcp_servers or tuple(spec.mcp_servers)
        if mcp_servers:
            config_text = config_text.rstrip() + "\n\n" + _render_mcp_servers_toml(mcp_servers)
        config_path = codex_root / "config.toml"
        if config_text or spec.base_config is not None or mcp_servers:
            config_path.write_text(config_text, encoding="utf-8")

        if options.auth_json_path is not None:
            shutil.copyfile(options.auth_json_path, codex_root / "auth.json")
        _materialize_skills(options, skills_root)

        generated_files = _describe_generated_files(resolved_home_root)
        resolved_defaults = _resolve_model_defaults(config_text)
        result = HomeMaterializationResult(
            provider_type=self.provider_type,
            home_id=spec.home_id,
            renderer_version=self.renderer_version,
            manifest_schema_version=1,
            manifest_hash="",
            generated_files=generated_files,
            required_env=spec.required_env,
            auth_refs=spec.auth_refs,
            resolved_defaults=resolved_defaults,
            effective_capabilities=_codex_home_capabilities(spec.home_id),
        )
        result = replace(result, manifest_hash=_manifest_hash(to_jsonable(result)))
        write_json_atomic(
            resolved_home_root / ".ark" / "home_materialization.json",
            to_jsonable(result),
        )
        return result

    def initialize(
        self,
        home: "HomeRecord",
        ctx: ProviderExecutionContext,
    ) -> HomeInitializationResult:
        if self.provider is not None:
            initialize = getattr(self.provider, "ensure_home_initialized", None)
            if not callable(initialize):
                raise TypeError("Codex Home renderer provider lacks ensure_home_initialized")
            record = initialize(
                home_id=home.home_id,
                home_root=ctx.home_root,
                env=ctx.process_environment,
                workdir=ctx.workdir,
            )
            return HomeInitializationResult(
                initialized=True,
                marker_ref=str(getattr(record, "marker_path", "")) or None,
            )
        marker = ctx.home_root / ".ark" / "codex_home_initialized.json"
        return HomeInitializationResult(
            initialized=marker.exists(),
            marker_ref=str(marker) if marker.exists() else None,
        )

    def refresh_materialization(
        self,
        home: "HomeRecord",
        home_root: Path,
    ) -> HomeMaterializationResult:
        """Explicitly seal application post-processing into the Home manifest."""

        manifest_path = Path(home_root) / ".ark" / "home_materialization.json"
        payload = read_json(manifest_path)
        declared_hash = str(payload.get("manifest_hash", ""))
        if (
            declared_hash != (home.materialization_manifest_hash or "")
            or _manifest_hash(payload) != declared_hash
        ):
            raise RuntimeError(f"home materialization manifest hash mismatch: {home.home_id}")
        config_path = Path(home_root) / ".codex" / "config.toml"
        config_text = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
        result = HomeMaterializationResult(
            provider_type=self.provider_type,
            home_id=home.home_id,
            renderer_version=str(payload["renderer_version"]),
            manifest_schema_version=int(payload["manifest_schema_version"]),
            manifest_hash="",
            generated_files=_describe_generated_files(Path(home_root)),
            source_resource_hashes=dict(payload.get("source_resource_hashes") or {}),
            required_env=tuple(payload.get("required_env") or ()),
            auth_refs=tuple(payload.get("auth_refs") or ()),
            resolved_defaults=_resolve_model_defaults(config_text),
            warnings=tuple(payload.get("warnings") or ())
            + ("materialization manifest explicitly refreshed after application post-processing",),
            effective_capabilities=_codex_home_capabilities(home.home_id),
        )
        result = replace(result, manifest_hash=_manifest_hash(to_jsonable(result)))
        write_json_atomic(manifest_path, to_jsonable(result))
        return result

    def build_execution_context(
        self,
        home: "HomeRecord",
        *,
        run_env: Mapping[str, str] | None,
        workdir: str | None,
    ) -> ProviderExecutionContext:
        home_root = self.runtime_root / home.home_relpath
        env = dict(os.environ)
        env.update(home.fixed_env)
        if run_env:
            env.update(run_env)
        for name in home.required_env:
            if not env.get(name):
                raise MissingProviderEnvError(name)
        env["HOME"] = str(home_root)
        env["CODEX_HOME"] = str(home_root / ".codex")
        manifest_path = home_root / ".ark" / "home_materialization.json"
        manifest = None
        if manifest_path.exists():
            payload = read_json(manifest_path)
            declared_hash = str(payload.get("manifest_hash", ""))
            if (
                declared_hash != (home.materialization_manifest_hash or "")
                or _manifest_hash(payload) != declared_hash
            ):
                raise RuntimeError(f"home materialization manifest hash mismatch: {home.home_id}")
            for item in payload.get("generated_files", []):
                path = home_root / str(item["relpath"])
                if not path.is_file() or _sha256(path) != str(item["sha256"]):
                    raise RuntimeError(f"home materialized file hash mismatch: {item['relpath']}")
            defaults_payload = payload.get("resolved_defaults")
            resolved_defaults = None
            if isinstance(defaults_payload, Mapping):
                resolved_defaults = ModelBackendIdentity(
                    api_provider=str(defaults_payload["api_provider"]),
                    api_mode=str(defaults_payload["api_mode"]),
                    endpoint_id=defaults_payload.get("endpoint_id"),
                    requested_model=defaults_payload.get("requested_model"),
                    resolved_model=defaults_payload.get("resolved_model"),
                    model_version=defaults_payload.get("model_version"),
                    service_tier=defaults_payload.get("service_tier"),
                    reasoning_effort=defaults_payload.get("reasoning_effort"),
                    tokenizer_id=defaults_payload.get("tokenizer_id"),
                    model_config_hash=defaults_payload.get("model_config_hash"),
                )
            manifest = HomeMaterializationResult(
                provider_type=str(payload["provider_type"]),
                home_id=str(payload["home_id"]),
                renderer_version=str(payload["renderer_version"]),
                manifest_schema_version=int(payload["manifest_schema_version"]),
                manifest_hash=declared_hash,
                generated_files=tuple(
                    HomeMaterializedFile(
                        relpath=str(item["relpath"]),
                        sha256=str(item["sha256"]),
                        source_fingerprint=item.get("source_fingerprint"),
                        secret=bool(item.get("secret", False)),
                    )
                    for item in payload.get("generated_files", [])
                ),
                source_resource_hashes=dict(payload.get("source_resource_hashes") or {}),
                required_env=tuple(payload.get("required_env") or ()),
                auth_refs=tuple(payload.get("auth_refs") or ()),
                resolved_defaults=resolved_defaults,
                warnings=tuple(payload.get("warnings") or ()),
            )
        return ProviderExecutionContext(
            provider_type=self.provider_type,
            home_id=home.home_id,
            home_root=home_root,
            process_environment=env,
            materialization_manifest=manifest,
            workdir=workdir,
        )


def _read_base_config(spec: ProviderHomeSpec) -> str:
    source = spec.base_config
    if source is None:
        return ""
    if source.path is not None:
        return Path(source.path).read_text(encoding="utf-8")
    if source.text is not None:
        return source.text
    if source.mapping is not None:
        return _render_toml_mapping(source.mapping)
    return ""


def _materialize_skills(options: CodexHomeOptions, skills_root: Path) -> None:
    for skill_name, skill_path in options.skill_paths.items():
        validated_name = validate_skill_name(skill_name)
        if not Path(skill_path).exists() or not Path(skill_path).is_dir():
            raise ValueError(f"skill path must be an existing directory: {skill_path}")
        dest = skills_root / validated_name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(skill_path, dest)
    for skill_name, skill_spec in options.skill_specs.items():
        validated_name = validate_skill_name(skill_name)
        if validated_name != skill_spec.name:
            raise ValueError(f"skill spec key must match SkillSpec.name: {skill_name} != {skill_spec.name}")
        write_skill_spec(skill_spec, skills_root / validated_name)


def _validate_skill_inputs(options: CodexHomeOptions) -> None:
    path_names = {validate_skill_name(name) for name in options.skill_paths}
    spec_names = {validate_skill_name(name) for name in options.skill_specs}
    duplicate_names = path_names & spec_names
    if duplicate_names:
        duplicates = ", ".join(sorted(duplicate_names))
        raise ValueError(f"duplicate skill names between skill_paths and skill_specs: {duplicates}")
    for skill_name in options.skill_paths:
        if validate_skill_name(skill_name) != skill_name:
            raise ValueError(f"invalid skill path name: {skill_name}")
    for skill_name, skill_spec in options.skill_specs.items():
        if validate_skill_name(skill_name) != skill_spec.name:
            raise ValueError(f"skill spec key must match SkillSpec.name: {skill_name} != {skill_spec.name}")


def _describe_generated_files(home_root: Path) -> tuple[HomeMaterializedFile, ...]:
    files: list[HomeMaterializedFile] = []
    managed = [home_root / ".codex" / "config.toml", home_root / ".codex" / "auth.json"]
    skills_root = home_root / ".agents" / "skills"
    if skills_root.exists():
        managed.extend(sorted(item for item in skills_root.rglob("*") if item.is_file()))
    for path in managed:
        if not path.is_file():
            continue
        relpath = str(path.relative_to(home_root))
        files.append(
            HomeMaterializedFile(
                relpath=relpath,
                sha256=_sha256(path),
                secret=relpath == ".codex/auth.json",
            )
        )
    return tuple(files)


def _manifest_hash(payload: object) -> str:
    canonical = dict(payload) if isinstance(payload, Mapping) else {"payload": payload}
    canonical["manifest_hash"] = ""
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_model_defaults(text: str) -> ModelBackendIdentity | None:
    try:
        values = tomllib.loads(text) if text.strip() else {}
    except tomllib.TOMLDecodeError:
        return None
    api_provider = str(values.get("model_provider") or "openai")
    provider_configs = values.get("model_providers")
    provider_config = (
        provider_configs.get(api_provider)
        if isinstance(provider_configs, Mapping)
        else None
    )
    wire_api = provider_config.get("wire_api") if isinstance(provider_config, Mapping) else None
    api_mode = {
        "chat": "chat_completions",
        "chat_completions": "chat_completions",
        "responses": "responses",
    }.get(str(wire_api or "responses"), str(wire_api or "responses"))
    requested_model = values.get("model")
    reasoning_effort = values.get("model_reasoning_effort")
    if requested_model is None and reasoning_effort is None and "model_provider" not in values:
        return None
    return ModelBackendIdentity(
        api_provider=api_provider,
        api_mode=api_mode,
        requested_model=str(requested_model) if requested_model is not None else None,
        reasoning_effort=str(reasoning_effort) if reasoning_effort is not None else None,
    )


def _codex_home_capabilities(home_id: str) -> ProviderCapabilities:
    available = (
        CapabilityKey.HOME_BASE_CONFIG,
        CapabilityKey.HOME_TYPED_OVERRIDES,
        CapabilityKey.HOME_RAW_OVERRIDES,
        CapabilityKey.HOME_ENV,
        CapabilityKey.HOME_AUTH_REFS,
        CapabilityKey.HOME_SKILLS,
        CapabilityKey.HOME_MCP,
    )
    return ProviderCapabilities(
        provider_type="codex",
        resolved_for_home_id=home_id,
        supports={
            key: CapabilitySupport(
                capability=key,
                status=CapabilityStatus.NATIVE,
                available=True,
                resolved_for_home_id=home_id,
            )
            for key in available
        },
    )


def _apply_top_level_overrides(text: str, overrides: Mapping[str, object]) -> str:
    lines = text.splitlines()
    first_table = next((index for index, line in enumerate(lines) if line.strip().startswith("[")), len(lines))
    rendered: list[str] = []
    found: set[str] = set()
    for index, line in enumerate(lines):
        if index < first_table and "=" in line and not line.lstrip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in overrides:
                if key not in found:
                    rendered.append(f"{_toml_key(key)} = {_toml_value(overrides[key])}")
                    found.add(key)
                continue
        rendered.append(line)
    missing = [key for key in overrides if key not in found]
    if missing:
        insert_at = next((index for index, line in enumerate(rendered) if line.strip().startswith("[")), len(rendered))
        additions = [f"{_toml_key(key)} = {_toml_value(overrides[key])}" for key in missing]
        if insert_at and rendered[insert_at - 1].strip():
            additions.append("")
        rendered[insert_at:insert_at] = additions
    return "\n".join(rendered).rstrip() + "\n"


def _apply_codex_model_config_overrides(text: str, overrides: "ModelConfigOverrides") -> str:
    return _apply_top_level_overrides(
        text,
        {
            key: value
            for key, value in {
                "model": overrides.model,
                "model_reasoning_effort": overrides.reasoning_effort,
            }.items()
            if value is not None
        },
    )


def _render_mcp_servers_toml(servers: tuple["McpServerSpec", ...]) -> str:
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
        for field_name in (
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
        ):
            value = getattr(server, field_name)
            if value is None or isinstance(value, list) and not value:
                continue
            lines.append(f"{field_name} = {_toml_value(value)}")
        for table_name in ("env", "http_headers", "env_http_headers"):
            values = getattr(server, table_name)
            if not values:
                continue
            lines.append("")
            lines.append(f"[mcp_servers.{_toml_key(name)}.{table_name}]")
            for key, value in sorted(values.items()):
                lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_toml_mapping(mapping: Mapping[str, object]) -> str:
    scalars = {key: value for key, value in mapping.items() if not isinstance(value, Mapping)}
    tables = {key: value for key, value in mapping.items() if isinstance(value, Mapping)}
    lines = [f"{_toml_key(key)} = {_toml_value(value)}" for key, value in scalars.items()]
    for key, values in tables.items():
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(f"[{_toml_key(key)}]")
        lines.extend(f"{_toml_key(child)} = {_toml_value(value)}" for child, value in values.items())
    return "\n".join(lines).rstrip() + "\n"


def _toml_key(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return value
    return json.dumps(value)


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {value!r}")
