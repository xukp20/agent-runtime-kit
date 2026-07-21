from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from ..models import MissingProviderEnvError, to_jsonable
from ..provider_contracts import (
    HomeInitializationResult,
    HomeMaterializationResult,
    HomeMaterializedFile,
    HomeValidationResult,
    ModelBackendIdentity,
    ProviderExecutionContext,
    ProviderHomeSpec,
    build_provider_payload,
)
from ..skills import SkillSpec, write_skill_spec
from ..store_utils import read_json, write_json_atomic
from .openai_agents import OpenAIAgentsHomeOptions, OpenAIAgentsResourceRegistry


class OpenAIAgentsHomeRenderer:
    provider_type = "openai_agents"
    renderer_version = "openai-agents-home-v1"

    def __init__(self, *, runtime_root: Path, registry: OpenAIAgentsResourceRegistry) -> None:
        self.runtime_root = Path(runtime_root)
        self.registry = registry

    def validate(self, spec: ProviderHomeSpec) -> HomeValidationResult:
        errors: list[str] = []
        if spec.provider_type != self.provider_type:
            errors.append(f"OpenAI Agents renderer cannot materialize {spec.provider_type}")
        if not isinstance(spec.provider_options, OpenAIAgentsHomeOptions):
            errors.append("provider_options must be OpenAIAgentsHomeOptions")
        else:
            try:
                self.registry.validate_ref(spec.provider_options.agent_factory_ref)
            except ValueError as exc:
                errors.append(str(exc))
            if spec.provider_options.api_key_env in spec.fixed_env:
                errors.append(
                    "API keys must be supplied through run environment, not persisted fixed_env"
                )
        if spec.model_config is None:
            errors.append("OpenAI Agents Home requires model_config")
        elif spec.model_config.api_mode not in {"responses", "chat_completions"}:
            errors.append(f"unsupported API mode: {spec.model_config.api_mode}")
        for item in spec.skills:
            if not isinstance(item, SkillSpec):
                errors.append("OpenAI Agents Home skills must be SkillSpec values")
        try:
            _assert_no_inline_secrets(_base_config(spec))
            _assert_no_inline_secrets(spec.config_overrides)
            _assert_mcp_secrets_use_env_refs(spec.mcp_servers)
        except ValueError as exc:
            errors.append(str(exc))
        return HomeValidationResult(valid=not errors, errors=tuple(errors))

    def materialize(self, spec: ProviderHomeSpec, home_root: Path) -> HomeMaterializationResult:
        validation = self.validate(spec)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        assert isinstance(spec.provider_options, OpenAIAgentsHomeOptions)
        assert spec.model_config is not None
        root = Path(home_root)
        root.mkdir(parents=True, exist_ok=True)
        skills_root = root / "skills"
        skills_root.mkdir(parents=True, exist_ok=True)
        for skill in spec.skills:
            assert isinstance(skill, SkillSpec)
            write_skill_spec(skill, skills_root / skill.name)

        config = _base_config(spec)
        config = _deep_merge(config, dict(spec.config_overrides))
        config.update(
            {
                "schema_version": 1,
                "agent_factory_ref": spec.provider_options.agent_factory_ref,
                "agent_factory_version": spec.provider_options.agent_factory_version,
                "resource_fingerprint": spec.provider_options.resource_fingerprint,
                "api_key_env": spec.provider_options.api_key_env,
                "base_url": spec.provider_options.base_url,
                "base_url_env": spec.provider_options.base_url_env,
                "session_backend": spec.provider_options.session_backend,
                "store": spec.provider_options.store,
                "compaction_mode": spec.provider_options.compaction_mode,
                "context_window_tokens": spec.provider_options.context_window_tokens,
                "max_output_tokens": spec.provider_options.max_output_tokens,
                "model_settings": dict(spec.provider_options.model_settings),
                "tracing_disabled": spec.provider_options.tracing_disabled,
                "model_identity": to_jsonable(spec.model_config),
                "instructions": [_instruction_text(item) for item in spec.instructions],
                "mcp_servers": [to_jsonable(item) for item in spec.mcp_servers],
                "required_env": sorted(
                    set(spec.required_env) | {spec.provider_options.api_key_env}
                ),
                "auth_refs": list(spec.auth_refs),
            }
        )
        provider_path = root / "provider.json"
        write_json_atomic(provider_path, config)

        generated = _generated_files(root)
        result = HomeMaterializationResult(
            provider_type=self.provider_type,
            home_id=spec.home_id,
            renderer_version=self.renderer_version,
            manifest_schema_version=1,
            manifest_hash="",
            generated_files=generated,
            source_resource_hashes={
                spec.provider_options.agent_factory_ref: (
                    spec.provider_options.resource_fingerprint
                    or spec.provider_options.agent_factory_version
                )
            },
            required_env=tuple(config["required_env"]),
            auth_refs=spec.auth_refs,
            resolved_defaults=spec.model_config,
            provider_payload=build_provider_payload(
                provider_type=self.provider_type,
                payload_type="home_options",
                data={
                    "compaction_mode": spec.provider_options.compaction_mode,
                    "session_backend": spec.provider_options.session_backend,
                    "store": spec.provider_options.store,
                },
                adapter_version="1",
                sdk_or_cli_version="0.18.3",
            ),
        )
        result = replace(result, manifest_hash=_manifest_hash(to_jsonable(result)))
        write_json_atomic(root / ".ark" / "home_materialization.json", to_jsonable(result))
        return result

    def refresh_materialization(self, home: object, home_root: Path) -> HomeMaterializationResult:
        payload = read_json(Path(home_root) / ".ark" / "home_materialization.json")
        result = _manifest_from_json(payload)
        if _manifest_hash(payload) != result.manifest_hash:
            raise RuntimeError(f"home materialization manifest hash mismatch: {result.home_id}")
        refreshed = replace(
            result,
            manifest_hash="",
            generated_files=_generated_files(Path(home_root)),
            warnings=result.warnings
            + ("materialization manifest explicitly refreshed after application post-processing",),
        )
        refreshed = replace(refreshed, manifest_hash=_manifest_hash(to_jsonable(refreshed)))
        write_json_atomic(Path(home_root) / ".ark" / "home_materialization.json", to_jsonable(refreshed))
        return refreshed

    def initialize(self, home: object, ctx: ProviderExecutionContext) -> HomeInitializationResult:
        config = _runtime_config(ctx.home_root)
        self.registry.resolve_agent_factory(
            str(config["agent_factory_ref"]),
            version=str(config["agent_factory_version"]),
            fingerprint=_optional_str(config.get("resource_fingerprint")),
        )
        marker = ctx.home_root / ".ark" / "openai_agents_initialized.json"
        write_json_atomic(marker, {"home_id": ctx.home_id, "manifest_hash": ctx.materialization_manifest.manifest_hash if ctx.materialization_manifest else None})
        return HomeInitializationResult(initialized=True, marker_ref=str(marker))

    def build_execution_context(
        self,
        home: object,
        *,
        run_env: Mapping[str, str] | None,
        workdir: str | None,
    ) -> ProviderExecutionContext:
        home_id = str(getattr(home, "home_id"))
        home_root = self.runtime_root / str(getattr(home, "home_relpath"))
        manifest_payload = read_json(home_root / ".ark" / "home_materialization.json")
        manifest = _manifest_from_json(manifest_payload)
        recorded_hash = getattr(home, "materialization_manifest_hash", None)
        if _manifest_hash(manifest_payload) != manifest.manifest_hash or (
            recorded_hash and recorded_hash != manifest.manifest_hash
        ):
            raise RuntimeError(f"home materialization manifest hash mismatch: {home_id}")
        for item in manifest.generated_files:
            path = home_root / item.relpath
            if not path.is_file() or _sha256(path) != item.sha256:
                raise RuntimeError(f"home materialized file hash mismatch: {item.relpath}")
        config = _runtime_config(home_root)
        env = dict(os.environ)
        env.update(dict(getattr(home, "fixed_env", {})))
        if run_env:
            env.update(run_env)
        for name in manifest.required_env:
            if not env.get(name):
                raise MissingProviderEnvError(name)
        base_url_env = _optional_str(config.get("base_url_env"))
        if base_url_env and not env.get(base_url_env):
            raise MissingProviderEnvError(base_url_env)
        return ProviderExecutionContext(
            provider_type=self.provider_type,
            home_id=home_id,
            home_root=home_root,
            process_environment=env,
            materialization_manifest=manifest,
            workdir=workdir,
            resolved_defaults=manifest.resolved_defaults,
            resource_handles=(self.registry,),
            runtime_payload=config,
        )


def _base_config(spec: ProviderHomeSpec) -> dict[str, object]:
    source = spec.base_config
    if source is None:
        return {}
    if source.mapping is not None:
        return dict(source.mapping)
    text = Path(source.path).read_text(encoding="utf-8") if source.path else source.text
    if not text or not text.strip():
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("OpenAI Agents base config must be a JSON object")
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


def _assert_no_inline_secrets(value: object, *, path: str = "config") -> None:
    if not isinstance(value, Mapping):
        return
    for raw_key, item in value.items():
        key = str(raw_key)
        lowered = key.lower()
        secret_key = any(
            fragment in lowered
            for fragment in ("api_key", "apikey", "authorization", "password", "secret", "token")
        )
        reference_key = lowered.endswith(("_env", "_env_var", "_ref"))
        if secret_key and not reference_key and item is not None and item != "":
            raise ValueError(f"inline secret-like value is forbidden at {path}.{key}; use an env/ref field")
        _assert_no_inline_secrets(item, path=f"{path}.{key}")


def _assert_mcp_secrets_use_env_refs(servers: tuple[object, ...]) -> None:
    for server in servers:
        name = str(getattr(server, "name", "unknown"))
        headers = getattr(server, "http_headers", {})
        if isinstance(headers, Mapping) and any(
            str(key).lower() in {"authorization", "proxy-authorization"} for key in headers
        ):
            raise ValueError(f"MCP server {name} must source authorization headers from env refs")
        env = getattr(server, "env", {})
        if isinstance(env, Mapping):
            for key, value in env.items():
                lowered = str(key).lower()
                if any(fragment in lowered for fragment in ("token", "api_key", "apikey", "secret", "password")) and value:
                    raise ValueError(f"MCP server {name} must source secret env {key} through env_vars")


def _instruction_text(value: object) -> str:
    text = getattr(value, "text", value)
    return str(text)


def _runtime_config(home_root: Path) -> dict[str, object]:
    payload = read_json(home_root / "provider.json")
    if not isinstance(payload, dict):
        raise RuntimeError("OpenAI Agents provider.json must contain an object")
    return payload


def _generated_files(root: Path) -> tuple[HomeMaterializedFile, ...]:
    paths = [root / "provider.json"]
    skills = root / "skills"
    if skills.exists():
        paths.extend(sorted(path for path in skills.rglob("*") if path.is_file()))
    return tuple(
        HomeMaterializedFile(relpath=str(path.relative_to(root)), sha256=_sha256(path))
        for path in paths
        if path.is_file()
    )


def _manifest_hash(value: object) -> str:
    payload = dict(value) if isinstance(value, Mapping) else {"payload": value}
    payload["manifest_hash"] = ""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_identity(value: object) -> ModelBackendIdentity | None:
    if not isinstance(value, Mapping):
        return None
    return ModelBackendIdentity(
        api_provider=str(value["api_provider"]),
        api_mode=str(value["api_mode"]),
        endpoint_id=_optional_str(value.get("endpoint_id")),
        requested_model=_optional_str(value.get("requested_model")),
        resolved_model=_optional_str(value.get("resolved_model")),
        model_version=_optional_str(value.get("model_version")),
        service_tier=_optional_str(value.get("service_tier")),
        reasoning_effort=_optional_str(value.get("reasoning_effort")),
        tokenizer_id=_optional_str(value.get("tokenizer_id")),
        model_config_hash=_optional_str(value.get("model_config_hash")),
    )


def _manifest_from_json(payload: Mapping[str, object]) -> HomeMaterializationResult:
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
                source_fingerprint=_optional_str(item.get("source_fingerprint")),
                secret=bool(item.get("secret", False)),
            )
            for item in payload.get("generated_files", [])
            if isinstance(item, Mapping)
        ),
        source_resource_hashes=dict(payload.get("source_resource_hashes") or {}),
        required_env=tuple(str(item) for item in payload.get("required_env", [])),
        auth_refs=tuple(str(item) for item in payload.get("auth_refs", [])),
        resolved_defaults=_model_identity(payload.get("resolved_defaults")),
        warnings=tuple(str(item) for item in payload.get("warnings", [])),
    )


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None
