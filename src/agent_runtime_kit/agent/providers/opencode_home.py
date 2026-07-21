from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from ..instructions import TextFragment
from ..models import MissingProviderEnvError, to_jsonable
from ..provider_contracts import (
    HomeInitializationResult,
    HomeMaterializationResult,
    HomeMaterializedFile,
    HomeValidationResult,
    ModelBackendIdentity,
    ProviderPayload,
    ProviderExecutionContext,
    ProviderHomeSpec,
)
from ..skills import SkillSpec, write_skill_spec
from ..store_utils import read_json, write_json_atomic
from .opencode_models import OpenCodeHomeOptions, PROVIDER_TYPE


class OpenCodeHomeRenderer:
    provider_type = PROVIDER_TYPE
    renderer_version = "opencode-home-v1"

    def __init__(self, *, runtime_root: Path) -> None:
        self.runtime_root = Path(runtime_root)

    def validate(self, spec: ProviderHomeSpec) -> HomeValidationResult:
        errors: list[str] = []
        if spec.provider_type != self.provider_type:
            errors.append(f"OpenCode renderer cannot materialize provider {spec.provider_type}")
        if spec.provider_options is not None and not isinstance(
            spec.provider_options, OpenCodeHomeOptions
        ):
            errors.append("OpenCode provider_options must be OpenCodeHomeOptions")
        if spec.extensions:
            errors.append("OpenCode extensions/plugins are disabled in the first adapter version")
        try:
            config = _read_config(spec)
            _deep_merge(config, dict(spec.config_overrides))
            _tools_config(spec.tools)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
        for item in spec.skills:
            if not isinstance(item, SkillSpec):
                errors.append("OpenCode skills must be SkillSpec values")
        return HomeValidationResult(valid=not errors, errors=tuple(errors))

    def materialize(self, spec: ProviderHomeSpec, home_root: Path) -> HomeMaterializationResult:
        validation = self.validate(spec)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        options = _options(spec)
        root = Path(home_root)
        root.mkdir(parents=True, exist_ok=True)
        config = _deep_merge(_read_config(spec), dict(spec.config_overrides))
        config = _deep_merge(config, _mcp_config(spec.mcp_servers))
        config = _deep_merge(config, _tools_config(spec.tools))
        config.update(
            {
                "snapshot": False,
                "share": "disabled",
                "autoupdate": False,
                "plugin": [],
            }
        )
        _validate_secret_refs(config)
        required_env = tuple(
            sorted(
                set(spec.required_env)
                | set(spec.fixed_env_refs.values())
            )
        )
        config_path = root / options.config_filename
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        instructions = _instructions_text(spec.instructions)
        if instructions:
            (root / "AGENTS.md").write_text(instructions.rstrip() + "\n", encoding="utf-8")
        skills_root = root / "skills"
        for skill in spec.skills:
            assert isinstance(skill, SkillSpec)
            write_skill_spec(skill, skills_root / skill.name)
        _write_markdown_resources(root / "agents", options.agent_files)
        _write_markdown_resources(root / "commands", options.command_files)
        generated = _generated_files(root)
        defaults = spec.model_config or _model_defaults(config)
        result = HomeMaterializationResult(
            provider_type=self.provider_type,
            home_id=spec.home_id,
            renderer_version=self.renderer_version,
            manifest_schema_version=1,
            manifest_hash="",
            generated_files=generated,
            required_env=required_env,
            auth_refs=tuple(spec.auth_refs),
            resolved_defaults=defaults,
            warnings=(
                ("project-local OpenCode config and instructions are enabled",)
                if options.allow_project_config
                else ()
            ),
            provider_payload=ProviderPayload(
                provider_type=self.provider_type,
                payload_type="home_options",
                adapter_version="1",
                sanitized_data={
                    "binary_path": str(options.binary_path),
                    "server_start_timeout_s": options.server_start_timeout_s,
                    "allow_project_config": options.allow_project_config,
                    "config_filename": options.config_filename,
                    "fixed_env_refs": dict(spec.fixed_env_refs),
                },
            ),
        )
        result = replace(result, manifest_hash=_manifest_hash(to_jsonable(result)))
        write_json_atomic(root / ".ark" / "home_materialization.json", to_jsonable(result))
        return result

    def refresh_materialization(self, home: object, home_root: Path) -> HomeMaterializationResult:
        root = Path(home_root)
        payload = read_json(root / ".ark" / "home_materialization.json")
        result = HomeMaterializationResult(
            provider_type=self.provider_type,
            home_id=str(getattr(home, "home_id")),
            renderer_version=self.renderer_version,
            manifest_schema_version=1,
            manifest_hash="",
            generated_files=_generated_files(root),
            required_env=tuple(payload.get("required_env") or ()),
            auth_refs=tuple(payload.get("auth_refs") or ()),
            resolved_defaults=_identity_from_payload(payload.get("resolved_defaults")),
            warnings=tuple(payload.get("warnings") or ())
            + ("materialization explicitly refreshed",),
            provider_payload=_provider_payload_from_manifest(payload.get("provider_payload")),
        )
        result = replace(result, manifest_hash=_manifest_hash(to_jsonable(result)))
        write_json_atomic(root / ".ark" / "home_materialization.json", to_jsonable(result))
        return result

    def initialize(
        self, home: object, ctx: ProviderExecutionContext
    ) -> HomeInitializationResult:
        marker = ctx.home_root / ".ark" / "home_materialization.json"
        return HomeInitializationResult(initialized=marker.is_file(), marker_ref=str(marker))

    def build_execution_context(
        self,
        home: object,
        *,
        run_env: Mapping[str, str] | None,
        workdir: str | None,
    ) -> ProviderExecutionContext:
        home_root = self.runtime_root / str(getattr(home, "home_relpath"))
        manifest_path = home_root / ".ark" / "home_materialization.json"
        payload = read_json(manifest_path)
        declared = str(payload.get("manifest_hash", ""))
        if (
            declared != str(getattr(home, "materialization_manifest_hash", ""))
            or _manifest_hash(payload) != declared
        ):
            raise RuntimeError(f"home materialization manifest hash mismatch: {getattr(home, 'home_id')}")
        for item in payload.get("generated_files", []):
            path = home_root / str(item["relpath"])
            if not path.is_file() or _sha256(path) != str(item["sha256"]):
                raise RuntimeError(f"home materialized file hash mismatch: {item['relpath']}")
        env = dict(os.environ)
        env.update(dict(getattr(home, "fixed_env", {})))
        if run_env:
            env.update(run_env)
        env.pop("OPENCODE_CONFIG", None)
        env.pop("OPENCODE_CONFIG_CONTENT", None)
        options_payload = payload.get("provider_payload")
        options_data = (
            options_payload.get("sanitized_data")
            if isinstance(options_payload, Mapping)
            and isinstance(options_payload.get("sanitized_data"), Mapping)
            else {}
        )
        fixed_env_refs = options_data.get("fixed_env_refs")
        if isinstance(fixed_env_refs, Mapping):
            for target_name, source_name in fixed_env_refs.items():
                source = str(source_name)
                if not env.get(source):
                    raise MissingProviderEnvError(source)
                env[str(target_name)] = env[source]
        required_env = set(getattr(home, "required_env", set())) | set(
            str(name) for name in payload.get("required_env", [])
        )
        for name in required_env:
            if not env.get(name):
                raise MissingProviderEnvError(name)
        env["OPENCODE_CONFIG_DIR"] = str(home_root)
        allow_project_config = bool(options_data.get("allow_project_config", False))
        env["OPENCODE_PURE"] = "1"
        if allow_project_config:
            env.pop("OPENCODE_DISABLE_PROJECT_CONFIG", None)
        else:
            env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"
        if options_data.get("binary_path"):
            env["ARK_OPENCODE_BINARY"] = str(options_data["binary_path"])
        if options_data.get("server_start_timeout_s") is not None:
            env["ARK_OPENCODE_SERVER_START_TIMEOUT"] = str(
                options_data["server_start_timeout_s"]
            )
        return ProviderExecutionContext(
            provider_type=self.provider_type,
            home_id=str(getattr(home, "home_id")),
            home_root=home_root,
            process_environment=env,
            workdir=workdir,
            resolved_defaults=_identity_from_payload(payload.get("resolved_defaults")),
            runtime_payload={
                "runtime_root": str(self.runtime_root),
                "allow_project_config": allow_project_config,
            },
        )


def _options(spec: ProviderHomeSpec) -> OpenCodeHomeOptions:
    return spec.provider_options if isinstance(spec.provider_options, OpenCodeHomeOptions) else OpenCodeHomeOptions()


def _read_config(spec: ProviderHomeSpec) -> dict[str, object]:
    source = spec.base_config
    if source is None:
        return {}
    if source.mapping is not None:
        return dict(source.mapping)
    text = source.text
    if source.path is not None:
        text = Path(source.path).read_text(encoding="utf-8")
    if not text:
        return {}
    value = json.loads(_strip_json_trailing_commas(_strip_json_comments(text)))
    if not isinstance(value, dict):
        raise ValueError("OpenCode base config must be a JSON object")
    return value


def _strip_json_comments(text: str) -> str:
    out: list[str] = []
    index = 0
    quoted = False
    escaped = False
    while index < len(text):
        char = text[index]
        if quoted:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            index += 1
            continue
        if char == '"':
            quoted = True
            out.append(char)
            index += 1
            continue
        if text[index : index + 2] == "//":
            index = text.find("\n", index)
            if index < 0:
                break
            out.append("\n")
            index += 1
            continue
        if text[index : index + 2] == "/*":
            end = text.find("*/", index + 2)
            if end < 0:
                raise ValueError("unterminated JSONC block comment")
            index = end + 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _strip_json_trailing_commas(text: str) -> str:
    out: list[str] = []
    index = 0
    quoted = False
    escaped = False
    while index < len(text):
        char = text[index]
        if quoted:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            index += 1
            continue
        if char == '"':
            quoted = True
            out.append(char)
            index += 1
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
                continue
        out.append(char)
        index += 1
    return "".join(out)


def _deep_merge(base: dict[str, object], overrides: dict[str, object]) -> dict[str, object]:
    result = dict(base)
    for key, value in overrides.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(current, dict(value))
        else:
            result[key] = value
    return result


def _mcp_config(servers: tuple[object, ...]) -> dict[str, object]:
    values: dict[str, object] = {}
    for server in servers:
        name = str(getattr(server, "name", "")).strip()
        if not name:
            raise ValueError("OpenCode MCP server requires a name")
        enabled = bool(getattr(server, "enabled", True))
        command = getattr(server, "command", None)
        url = getattr(server, "url", None)
        if command:
            item: dict[str, object] = {
                "type": "local",
                "command": [str(command), *[str(arg) for arg in getattr(server, "args", [])]],
                "enabled": enabled,
            }
            environment = dict(getattr(server, "env", {}))
            for env_name in getattr(server, "env_vars", []):
                environment[str(env_name)] = "{" + f"env:{env_name}" + "}"
            if environment:
                item["environment"] = environment
        elif url:
            headers = dict(getattr(server, "http_headers", {}))
            for header, env_name in dict(getattr(server, "env_http_headers", {})).items():
                headers[str(header)] = "{" + f"env:{env_name}" + "}"
            bearer = getattr(server, "bearer_token_env_var", None)
            if bearer:
                headers["Authorization"] = "Bearer {" + f"env:{bearer}" + "}"
            item = {"type": "remote", "url": str(url), "enabled": enabled}
            if headers:
                item["headers"] = headers
        else:
            raise ValueError(f"OpenCode MCP server {name} requires command or url")
        values[name] = item
    return {"mcp": values} if values else {}


def _tools_config(tools: tuple[object, ...]) -> dict[str, object]:
    values: dict[str, bool] = {}
    for tool in tools:
        if isinstance(tool, Mapping):
            for name, enabled in tool.items():
                if not str(name).strip() or not isinstance(enabled, bool):
                    raise ValueError(
                        "OpenCode tool mappings require non-empty names and boolean values"
                    )
                values[str(name)] = enabled
            continue
        name = str(getattr(tool, "name", "")).strip()
        enabled = getattr(tool, "enabled", None)
        if not name or not isinstance(enabled, bool):
            raise ValueError("OpenCode tool specs require name and boolean enabled attributes")
        values[name] = enabled
    return {"tools": values} if values else {}


def _validate_secret_refs(value: object, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = (*path, key_text)
            sensitive = any(
                marker in key_text.lower()
                for marker in ("apikey", "api_key", "token", "secret", "authorization")
            )
            if sensitive and isinstance(child, str) and "{env:" not in child:
                raise ValueError(
                    f"OpenCode secret config must use an env reference: {'.'.join(child_path)}"
                )
            _validate_secret_refs(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_secret_refs(child, (*path, str(index)))


def _instructions_text(items: tuple[object, ...]) -> str:
    parts: list[str] = []
    for item in items:
        text = item.text if isinstance(item, TextFragment) else str(item)
        if text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)


def _write_markdown_resources(root: Path, values: Mapping[str, str]) -> None:
    for name, content in values.items():
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{name}.md").write_text(str(content).rstrip() + "\n", encoding="utf-8")


def _model_defaults(config: Mapping[str, object]) -> ModelBackendIdentity | None:
    model = config.get("model")
    if not isinstance(model, str) or not model:
        return None
    provider_id, _, model_id = model.partition("/")
    api_mode = "responses" if _provider_npm(config, provider_id) == "@ai-sdk/openai" else "chat_completions"
    return ModelBackendIdentity(
        api_provider=provider_id,
        api_mode=api_mode,
        requested_model=model_id or model,
        resolved_model=model_id or model,
    )


def _provider_npm(config: Mapping[str, object], provider_id: str) -> str | None:
    providers = config.get("provider")
    if not isinstance(providers, Mapping):
        return None
    provider = providers.get(provider_id)
    return str(provider.get("npm")) if isinstance(provider, Mapping) and provider.get("npm") else None


def _identity_from_payload(value: object) -> ModelBackendIdentity | None:
    if not isinstance(value, Mapping):
        return None
    return ModelBackendIdentity(
        api_provider=str(value["api_provider"]),
        api_mode=str(value["api_mode"]),
        endpoint_id=value.get("endpoint_id"),
        requested_model=value.get("requested_model"),
        resolved_model=value.get("resolved_model"),
        model_version=value.get("model_version"),
    )


def _provider_payload_from_manifest(value: object) -> ProviderPayload | None:
    if not isinstance(value, Mapping):
        return None
    return ProviderPayload(
        provider_type=str(value.get("provider_type") or PROVIDER_TYPE),
        payload_type=str(value.get("payload_type") or "home_options"),
        payload_schema_version=int(value.get("payload_schema_version") or 1),
        adapter_version=value.get("adapter_version"),
        sdk_or_cli_version=value.get("sdk_or_cli_version"),
        sanitized_data=value.get("sanitized_data"),
        truncated=bool(value.get("truncated", False)),
    )


def _generated_files(root: Path) -> tuple[HomeMaterializedFile, ...]:
    values: list[HomeMaterializedFile] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and ".ark" not in item.parts):
        values.append(HomeMaterializedFile(relpath=str(path.relative_to(root)), sha256=_sha256(path)))
    return tuple(values)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_hash(payload: object) -> str:
    value = dict(payload) if isinstance(payload, Mapping) else payload
    if isinstance(value, dict):
        value["manifest_hash"] = ""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
