from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

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
    build_provider_payload,
)
from ..skills import SkillSpec, validate_skill_name, write_skill_spec
from ..store_utils import read_json, utc_now_iso, write_json_atomic

if TYPE_CHECKING:
    from ..homes import HomeRecord


_MANAGED_EXTRA_ARGS = {
    "enable-file-checkpointing",
    "fork-session",
    "mcp-config",
    "resume",
    "session-id",
    "setting-sources",
    "settings",
    "strict-mcp-config",
}
_SETTING_SOURCES = {"user", "project", "local"}
_PERMISSION_MODES = {"default", "acceptEdits", "bypassPermissions", "plan", "dontAsk", "auto"}


@dataclass(frozen=True)
class ClaudeCodeHomeOptions:
    cli_path: str | Path | None = None
    setting_sources: tuple[str, ...] | None = ("user",)
    tools: tuple[str, ...] | None = None
    allowed_tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    permission_mode: str | None = None
    strict_mcp_config: bool = True
    skills: tuple[str, ...] | str | None = None
    skill_paths: Mapping[str, Path] = field(default_factory=dict)
    system_prompt: str | None = None
    model: str | None = None
    fallback_model: str | None = None
    thinking: Mapping[str, object] | None = None
    effort: str | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    add_dirs: tuple[Path, ...] = ()
    extra_args: Mapping[str, str | None] = field(default_factory=dict)
    enable_file_checkpointing: bool = False
    minimum_context_cli_version: str = "2.1.216"


class ClaudeCodeHomeRenderer:
    provider_type = "claude_code"
    renderer_version = "claude-code-home-v1"

    def __init__(self, *, runtime_root: Path, provider: object | None = None) -> None:
        self.runtime_root = Path(runtime_root)
        self.provider = provider

    def validate(self, spec: ProviderHomeSpec) -> HomeValidationResult:
        errors: list[str] = []
        if spec.provider_type != self.provider_type:
            errors.append(f"Claude Code renderer cannot materialize provider {spec.provider_type}")
        options = _options(spec)
        if spec.provider_options is not None and not isinstance(
            spec.provider_options, ClaudeCodeHomeOptions
        ):
            errors.append("Claude Code provider_options must be ClaudeCodeHomeOptions")
            return HomeValidationResult(valid=False, errors=tuple(errors))
        if options.enable_file_checkpointing:
            errors.append("Claude Code file checkpointing is unsupported by the v1 snapshot adapter")
        if options.setting_sources is not None:
            invalid = set(options.setting_sources) - _SETTING_SOURCES
            if invalid:
                errors.append(f"unsupported Claude setting sources: {sorted(invalid)}")
        if options.permission_mode is not None and options.permission_mode not in _PERMISSION_MODES:
            errors.append(f"unsupported Claude permission mode: {options.permission_mode}")
        if options.max_turns is not None and options.max_turns <= 0:
            errors.append("Claude max_turns must be positive")
        if options.max_budget_usd is not None and options.max_budget_usd <= 0:
            errors.append("Claude max_budget_usd must be positive")
        conflicts = sorted(set(options.extra_args) & _MANAGED_EXTRA_ARGS)
        if conflicts:
            errors.append(f"Claude extra_args conflict with ARK-managed flags: {conflicts}")
        try:
            _validate_skills(spec, options)
            _validate_mcp_servers(tuple(spec.mcp_servers))
            _instruction_text(spec.instructions)
            _tool_names(spec.tools)
            _read_settings(spec)
        except (TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
        return HomeValidationResult(valid=not errors, errors=tuple(errors))

    def materialize(self, spec: ProviderHomeSpec, home_root: Path) -> HomeMaterializationResult:
        validation = self.validate(spec)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        options = _options(spec)
        home_root = Path(home_root)
        claude_root = home_root / ".claude"
        ark_root = home_root / ".ark"
        skills_root = claude_root / "skills"
        claude_root.mkdir(parents=True, exist_ok=True)
        ark_root.mkdir(parents=True, exist_ok=True)

        settings = _deep_merge(_read_settings(spec), dict(spec.config_overrides))
        settings_path = claude_root / "settings.json"
        if settings:
            write_json_atomic(settings_path, settings)
        elif settings_path.exists():
            settings_path.unlink()

        skill_names = _materialize_skills(spec, options, skills_root)
        runtime_config = {
            "schema_version": 1,
            "provider_type": self.provider_type,
            "cli_path": str(options.cli_path) if options.cli_path is not None else None,
            "setting_sources": (
                list(options.setting_sources) if options.setting_sources is not None else None
            ),
            "tools": list(options.tools) if options.tools is not None else _tool_names(spec.tools),
            "allowed_tools": list(options.allowed_tools),
            "disallowed_tools": list(options.disallowed_tools),
            "permission_mode": options.permission_mode,
            "strict_mcp_config": options.strict_mcp_config,
            "skills": _effective_skills(options.skills, skill_names),
            "system_prompt": _join_instructions(options.system_prompt, spec.instructions),
            "model": options.model,
            "fallback_model": options.fallback_model,
            "thinking": dict(options.thinking) if options.thinking is not None else None,
            "effort": options.effort,
            "max_turns": options.max_turns,
            "max_budget_usd": options.max_budget_usd,
            "add_dirs": [str(path) for path in options.add_dirs],
            "extra_args": dict(options.extra_args),
            "enable_file_checkpointing": False,
            "minimum_context_cli_version": options.minimum_context_cli_version,
            "mcp_servers": [_mcp_to_mapping(server) for server in spec.mcp_servers],
        }
        runtime_path = ark_root / "claude_code_home.json"
        write_json_atomic(runtime_path, runtime_config)

        resolved_defaults = spec.model_config or ModelBackendIdentity(
            api_provider="anthropic",
            api_mode="anthropic_messages",
            requested_model=options.model,
            reasoning_effort=options.effort,
        )
        generated_files = _describe_generated_files(home_root)
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
            effective_capabilities=_home_capabilities(spec.home_id),
            provider_payload=build_provider_payload(
                provider_type=self.provider_type,
                payload_type="home_materialization",
                adapter_version="1",
                data={
                    "setting_sources": runtime_config["setting_sources"],
                    "skill_names": skill_names,
                    "mcp_server_names": [item["name"] for item in runtime_config["mcp_servers"]],
                    "file_checkpointing": False,
                },
            ),
        )
        result = replace(result, manifest_hash=_manifest_hash(to_jsonable(result)))
        write_json_atomic(ark_root / "home_materialization.json", to_jsonable(result))
        return result

    def refresh_materialization(
        self,
        home: "HomeRecord",
        home_root: Path,
    ) -> HomeMaterializationResult:
        home_root = Path(home_root)
        payload = _validated_manifest(home, home_root)
        result = HomeMaterializationResult(
            provider_type=self.provider_type,
            home_id=home.home_id,
            renderer_version=str(payload["renderer_version"]),
            manifest_schema_version=int(payload["manifest_schema_version"]),
            manifest_hash="",
            generated_files=_describe_generated_files(home_root),
            source_resource_hashes=dict(payload.get("source_resource_hashes") or {}),
            required_env=tuple(payload.get("required_env") or ()),
            auth_refs=tuple(payload.get("auth_refs") or ()),
            resolved_defaults=_backend_from_mapping(payload.get("resolved_defaults")),
            warnings=tuple(payload.get("warnings") or ())
            + ("materialization manifest explicitly refreshed after application post-processing",),
            effective_capabilities=_home_capabilities(home.home_id),
        )
        result = replace(result, manifest_hash=_manifest_hash(to_jsonable(result)))
        write_json_atomic(home_root / ".ark" / "home_materialization.json", to_jsonable(result))
        return result

    def initialize(
        self,
        home: "HomeRecord",
        ctx: ProviderExecutionContext,
    ) -> HomeInitializationResult:
        if self.provider is not None:
            load_sdk = getattr(self.provider, "sdk", None)
            if callable(load_sdk):
                load_sdk()
        config = _runtime_config(ctx.home_root)
        cli_path = config.get("cli_path") or "claude"
        version = _probe_cli_version(str(cli_path), ctx.process_environment)
        marker = ctx.home_root / ".ark" / "claude_home_initialized.json"
        write_json_atomic(
            marker,
            {
                "provider_type": self.provider_type,
                "home_id": home.home_id,
                "initialized_at": utc_now_iso(),
                "cli_path": str(cli_path),
                "cli_version": version,
            },
        )
        return HomeInitializationResult(
            initialized=True,
            marker_ref=str(marker),
            warnings=() if version is not None else ("Claude CLI version could not be determined",),
        )

    def build_execution_context(
        self,
        home: "HomeRecord",
        *,
        run_env: Mapping[str, str] | None,
        workdir: str | None,
    ) -> ProviderExecutionContext:
        home_root = self.runtime_root / home.home_relpath
        payload = _validated_manifest(home, home_root)
        for item in payload.get("generated_files", []):
            path = _safe_join(home_root, str(item["relpath"]))
            if not path.is_file() or _sha256(path) != str(item["sha256"]):
                raise RuntimeError(f"home materialized file hash mismatch: {item['relpath']}")
        env = dict(os.environ)
        env.update(home.fixed_env)
        if run_env:
            env.update(run_env)
        for name in home.required_env:
            if not env.get(name):
                raise MissingProviderEnvError(name)
        env["HOME"] = str(home_root)
        env["CLAUDE_CONFIG_DIR"] = str(home_root / ".claude")
        runtime_config = _runtime_config(home_root)
        runtime_config["mcp_servers_resolved"] = _resolve_mcp_servers(
            runtime_config.get("mcp_servers") or [], env
        )
        return ProviderExecutionContext(
            provider_type=self.provider_type,
            home_id=home.home_id,
            home_root=home_root,
            process_environment=env,
            materialization_manifest=_materialization_from_payload(payload),
            workdir=workdir,
            resolved_defaults=_backend_from_mapping(payload.get("resolved_defaults")),
            runtime_payload=runtime_config,
        )


def _options(spec: ProviderHomeSpec) -> ClaudeCodeHomeOptions:
    return (
        spec.provider_options
        if isinstance(spec.provider_options, ClaudeCodeHomeOptions)
        else ClaudeCodeHomeOptions()
    )


def _read_settings(spec: ProviderHomeSpec) -> dict[str, object]:
    source = spec.base_config
    if source is None:
        return {}
    if source.path is not None:
        value = json.loads(Path(source.path).read_text(encoding="utf-8"))
    elif source.text is not None:
        value = json.loads(source.text)
    else:
        value = dict(source.mapping or {})
    if not isinstance(value, dict):
        raise ValueError("Claude base settings must be a JSON object")
    return value


def _deep_merge(base: Mapping[str, object], override: Mapping[str, object]) -> dict[str, object]:
    result = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result


def _instruction_text(values: tuple[object, ...]) -> str | None:
    parts: list[str] = []
    for item in values:
        text = item if isinstance(item, str) else getattr(item, "text", None)
        if not isinstance(text, str):
            raise TypeError("Claude instructions must be strings or TextFragment-like objects")
        if text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts) or None


def _join_instructions(base: str | None, values: tuple[object, ...]) -> str | None:
    parts = [item for item in (base.strip() if base else None, _instruction_text(values)) if item]
    return "\n\n".join(parts) or None


def _tool_names(values: tuple[object, ...]) -> list[str]:
    names: list[str] = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise TypeError("Claude tools must be non-empty string names")
        names.append(item.strip())
    return names


def _validate_skills(spec: ProviderHomeSpec, options: ClaudeCodeHomeOptions) -> None:
    names: set[str] = set()
    for name, path in options.skill_paths.items():
        resolved = validate_skill_name(name)
        if resolved in names:
            raise ValueError(f"duplicate Claude skill: {resolved}")
        if not Path(path).is_dir():
            raise ValueError(f"skill path must be an existing directory: {path}")
        names.add(resolved)
    for item in spec.skills:
        if not isinstance(item, SkillSpec):
            raise TypeError("Claude ProviderHomeSpec.skills must contain SkillSpec values")
        if item.name in names:
            raise ValueError(f"duplicate Claude skill: {item.name}")
        names.add(item.name)
    if isinstance(options.skills, str) and options.skills != "all":
        raise ValueError("Claude skills string must be 'all'")
    if isinstance(options.skills, tuple):
        for name in options.skills:
            validate_skill_name(name)


def _materialize_skills(
    spec: ProviderHomeSpec,
    options: ClaudeCodeHomeOptions,
    skills_root: Path,
) -> list[str]:
    if skills_root.exists():
        shutil.rmtree(skills_root)
    skills_root.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for name, source in options.skill_paths.items():
        resolved = validate_skill_name(name)
        shutil.copytree(source, skills_root / resolved)
        names.append(resolved)
    for item in spec.skills:
        assert isinstance(item, SkillSpec)
        write_skill_spec(item, skills_root / item.name)
        names.append(item.name)
    return sorted(names)


def _effective_skills(value: tuple[str, ...] | str | None, materialized: list[str]) -> object:
    if value is not None:
        return list(value) if isinstance(value, tuple) else value
    return materialized if materialized else []


def _validate_mcp_servers(servers: tuple[object, ...]) -> None:
    seen: set[str] = set()
    for raw in servers:
        name = str(getattr(raw, "name", "")).strip()
        if not name or name in seen:
            raise ValueError(f"invalid or duplicate MCP server name: {name!r}")
        seen.add(name)
        transport = str(getattr(raw, "transport", "http"))
        if transport not in {"stdio", "http", "sse"}:
            raise ValueError(f"Claude MCP transport is unsupported: {transport}")
        if transport == "stdio" and not getattr(raw, "command", None):
            raise ValueError(f"Claude stdio MCP requires command: {name}")
        if transport in {"http", "sse"} and not getattr(raw, "url", None):
            raise ValueError(f"Claude {transport} MCP requires url: {name}")
        unsupported = {
            "cwd": getattr(raw, "cwd", None),
            "startup_timeout_sec": getattr(raw, "startup_timeout_sec", None),
            "tool_timeout_sec": getattr(raw, "tool_timeout_sec", None),
            "enabled_tools": getattr(raw, "enabled_tools", None),
            "disabled_tools": getattr(raw, "disabled_tools", None),
        }
        present = [key for key, value in unsupported.items() if value not in (None, [], ())]
        if present:
            raise ValueError(f"Claude MCP fields cannot be mapped without loss for {name}: {present}")


def _mcp_to_mapping(server: object) -> dict[str, object]:
    return {
        "name": str(getattr(server, "name")),
        "enabled": bool(getattr(server, "enabled", True)),
        "required": bool(getattr(server, "required", False)),
        "transport": str(getattr(server, "transport", "http")),
        "url": getattr(server, "url", None),
        "command": getattr(server, "command", None),
        "args": list(getattr(server, "args", []) or []),
        "env": dict(getattr(server, "env", {}) or {}),
        "env_vars": list(getattr(server, "env_vars", []) or []),
        "bearer_token_env_var": getattr(server, "bearer_token_env_var", None),
        "http_headers": dict(getattr(server, "http_headers", {}) or {}),
        "env_http_headers": dict(getattr(server, "env_http_headers", {}) or {}),
    }


def _resolve_mcp_servers(values: list[object], env: Mapping[str, str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for raw in values:
        if not isinstance(raw, Mapping) or not raw.get("enabled", True):
            continue
        name = str(raw["name"])
        required = bool(raw.get("required", False))
        transport = str(raw.get("transport", "http"))
        if transport == "stdio":
            item: dict[str, object] = {
                "type": "stdio",
                "command": str(raw["command"]),
                "args": [str(value) for value in raw.get("args") or []],
            }
            mcp_env = {str(key): str(value) for key, value in dict(raw.get("env") or {}).items()}
            for name_ref in raw.get("env_vars") or []:
                value = env.get(str(name_ref))
                if value is None:
                    if required:
                        raise MissingProviderEnvError(str(name_ref))
                    continue
                mcp_env[str(name_ref)] = value
            if mcp_env:
                item["env"] = mcp_env
        else:
            item = {"type": transport, "url": str(raw["url"])}
            headers = {
                str(key): str(value) for key, value in dict(raw.get("http_headers") or {}).items()
            }
            for header, env_name in dict(raw.get("env_http_headers") or {}).items():
                value = env.get(str(env_name))
                if value is None:
                    if required:
                        raise MissingProviderEnvError(str(env_name))
                    continue
                headers[str(header)] = value
            bearer = raw.get("bearer_token_env_var")
            if bearer:
                value = env.get(str(bearer))
                if value is None and required:
                    raise MissingProviderEnvError(str(bearer))
                if value is not None:
                    headers["Authorization"] = f"Bearer {value}"
            if headers:
                item["headers"] = headers
        result[name] = item
    return result


def _runtime_config(home_root: Path) -> dict[str, object]:
    value = read_json(home_root / ".ark" / "claude_code_home.json")
    if not isinstance(value, dict) or value.get("provider_type") != "claude_code":
        raise RuntimeError("invalid Claude Code Home runtime config")
    return dict(value)


def _describe_generated_files(home_root: Path) -> tuple[HomeMaterializedFile, ...]:
    managed = [home_root / ".ark" / "claude_code_home.json", home_root / ".claude" / "settings.json"]
    skills_root = home_root / ".claude" / "skills"
    if skills_root.exists():
        managed.extend(sorted(path for path in skills_root.rglob("*") if path.is_file()))
    result: list[HomeMaterializedFile] = []
    for path in managed:
        if path.is_file():
            result.append(
                HomeMaterializedFile(
                    relpath=str(path.relative_to(home_root)),
                    sha256=_sha256(path),
                    secret=path.name == "settings.json" and _settings_may_contain_secrets(path),
                )
            )
    return tuple(result)


def _settings_may_contain_secrets(path: Path) -> bool:
    try:
        payload = read_json(path)
    except Exception:
        return True
    return isinstance(payload, Mapping) and bool(payload.get("env"))


def _validated_manifest(home: "HomeRecord", home_root: Path) -> dict[str, object]:
    path = home_root / ".ark" / "home_materialization.json"
    payload = read_json(path)
    declared = str(payload.get("manifest_hash", ""))
    if declared != (home.materialization_manifest_hash or "") or _manifest_hash(payload) != declared:
        raise RuntimeError(f"home materialization manifest hash mismatch: {home.home_id}")
    return payload


def _materialization_from_payload(payload: Mapping[str, object]) -> HomeMaterializationResult:
    return HomeMaterializationResult(
        provider_type=str(payload["provider_type"]),
        home_id=str(payload["home_id"]),
        renderer_version=str(payload["renderer_version"]),
        manifest_schema_version=int(payload["manifest_schema_version"]),
        manifest_hash=str(payload["manifest_hash"]),
        generated_files=tuple(
            HomeMaterializedFile(
                relpath=str(item["relpath"]),
                sha256=str(item["sha256"]),
                source_fingerprint=item.get("source_fingerprint"),
                secret=bool(item.get("secret", False)),
            )
            for item in payload.get("generated_files", [])
            if isinstance(item, Mapping)
        ),
        source_resource_hashes=dict(payload.get("source_resource_hashes") or {}),
        required_env=tuple(payload.get("required_env") or ()),
        auth_refs=tuple(payload.get("auth_refs") or ()),
        resolved_defaults=_backend_from_mapping(payload.get("resolved_defaults")),
        warnings=tuple(payload.get("warnings") or ()),
    )


def _backend_from_mapping(value: object) -> ModelBackendIdentity | None:
    if not isinstance(value, Mapping):
        return None
    return ModelBackendIdentity(
        api_provider=str(value["api_provider"]),
        api_mode=str(value["api_mode"]),
        endpoint_id=value.get("endpoint_id"),
        requested_model=value.get("requested_model"),
        resolved_model=value.get("resolved_model"),
        model_version=value.get("model_version"),
        service_tier=value.get("service_tier"),
        reasoning_effort=value.get("reasoning_effort"),
        tokenizer_id=value.get("tokenizer_id"),
        model_config_hash=value.get("model_config_hash"),
    )


def _home_capabilities(home_id: str) -> ProviderCapabilities:
    keys = (
        CapabilityKey.HOME_BASE_CONFIG,
        CapabilityKey.HOME_TYPED_OVERRIDES,
        CapabilityKey.HOME_RAW_OVERRIDES,
        CapabilityKey.HOME_ENV,
        CapabilityKey.HOME_AUTH_REFS,
        CapabilityKey.HOME_INSTRUCTIONS,
        CapabilityKey.HOME_SKILLS,
        CapabilityKey.HOME_MCP,
    )
    return ProviderCapabilities(
        provider_type="claude_code",
        resolved_for_home_id=home_id,
        supports={
            key: CapabilitySupport(
                capability=key,
                status=CapabilityStatus.NATIVE,
                available=True,
                resolved_for_home_id=home_id,
            )
            for key in keys
        },
    )


def _probe_cli_version(cli_path: str, env: Mapping[str, str]) -> str | None:
    try:
        completed = subprocess.run(
            [cli_path, "--version"],
            env=dict(env),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or completed.stderr.strip() or None


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


def _safe_join(root: Path, relpath: str) -> Path:
    root = root.resolve()
    target = (root / relpath).resolve()
    if target != root and root not in target.parents:
        raise RuntimeError(f"path escapes Claude Home: {relpath}")
    return target
