from __future__ import annotations

import hashlib
import json
import os
import shutil
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
from ..store_utils import read_json, write_json_atomic
from .pi_session import PI_ADAPTER_VERSION, PI_CLI_VERSION

if TYPE_CHECKING:
    from ..homes import HomeRecord, McpServerSpec


@dataclass(frozen=True)
class PiHomeOptions:
    auth_json_path: Path | None = None
    models_json_path: Path | None = None
    models: Mapping[str, object] | None = None
    settings: Mapping[str, object] = field(default_factory=dict)
    skill_paths: Mapping[str, Path] = field(default_factory=dict)
    skill_specs: Mapping[str, SkillSpec] = field(default_factory=dict)
    extension_paths: tuple[Path, ...] = ()
    node_executable: str | None = None
    pi_cli_path: Path | None = None
    mcp_runtime_root: Path | None = None
    project_trust: str = "never"
    load_project_context: bool = False
    offline: bool = False
    tools: tuple[str, ...] = ()
    extra_cli_args: tuple[str, ...] = ()


class PiHomeRenderer:
    provider_type = "pi"
    renderer_version = "pi-home-v1"

    def __init__(self, *, runtime_root: Path) -> None:
        self.runtime_root = Path(runtime_root)

    def validate(self, spec: ProviderHomeSpec) -> HomeValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        if spec.provider_type != self.provider_type:
            errors.append(f"Pi renderer cannot materialize provider {spec.provider_type}")
        options = spec.provider_options
        if options is not None and not isinstance(options, PiHomeOptions):
            errors.append("Pi provider_options must be PiHomeOptions")
            return HomeValidationResult(valid=False, errors=tuple(errors))
        resolved = options if isinstance(options, PiHomeOptions) else PiHomeOptions()
        for path, label in (
            (resolved.auth_json_path, "auth_json_path"),
            (resolved.models_json_path, "models_json_path"),
            (resolved.pi_cli_path, "pi_cli_path"),
        ):
            if path is not None and not Path(path).is_file():
                errors.append(f"{label} must be an existing file: {path}")
        if resolved.models_json_path is not None and resolved.models is not None:
            errors.append("Pi models must use either models_json_path or models mapping")
        if resolved.project_trust not in {"never", "always", "ask"}:
            errors.append("Pi project_trust must be never, always, or ask")
        if resolved.project_trust == "ask":
            errors.append("Pi non-interactive Homes cannot use project_trust=ask")
        try:
            _validate_resources(resolved)
            _validate_extra_args(resolved.extra_cli_args)
            _read_json_config(spec)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
        if spec.mcp_servers and resolved.mcp_runtime_root is None:
            warnings.append("Pi MCP requires a prepared runtime root at Home initialization")
        return HomeValidationResult(valid=not errors, errors=tuple(errors), warnings=tuple(warnings))

    def materialize(self, spec: ProviderHomeSpec, home_root: Path) -> HomeMaterializationResult:
        validation = self.validate(spec)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        options = spec.provider_options if isinstance(spec.provider_options, PiHomeOptions) else PiHomeOptions()
        root = Path(home_root)
        pi_root = root / ".pi"
        ark_root = root / ".ark"
        sessions_root = pi_root / "sessions"
        skills_root = pi_root / "skills"
        extensions_root = pi_root / "extensions"
        for directory in (pi_root, ark_root, sessions_root, skills_root, extensions_root):
            directory.mkdir(parents=True, exist_ok=True)

        settings = _read_json_config(spec)
        settings.update(dict(spec.config_overrides))
        settings.update(dict(options.settings))
        settings["defaultProjectTrust"] = options.project_trust
        write_json_atomic(pi_root / "settings.json", settings)

        models = _read_models(options)
        if models is not None:
            write_json_atomic(pi_root / "models.json", models)
        if options.auth_json_path is not None:
            shutil.copyfile(options.auth_json_path, pi_root / "auth.json")
        _materialize_skills(spec, options, skills_root)
        extension_paths = list(options.extension_paths)
        for item in spec.extensions:
            if not isinstance(item, (str, Path)):
                raise ValueError("Pi extensions must be filesystem paths")
            extension_paths.append(Path(item))
        extension_relpaths = _materialize_extensions(tuple(extension_paths), extensions_root)

        instruction_text = _instruction_text(spec.instructions)
        instructions_relpath = None
        if instruction_text:
            instructions_relpath = ".ark/pi_instructions.md"
            (root / instructions_relpath).write_text(instruction_text, encoding="utf-8")

        mcp_servers = tuple(spec.mcp_servers)
        mcp_manifest_relpath = None
        if mcp_servers:
            mcp_manifest_relpath = ".ark/pi_mcp_manifest.json"
            write_json_atomic(root / mcp_manifest_relpath, _mcp_manifest(mcp_servers))
            shutil.copyfile(
                Path(__file__).with_name("pi_mcp_bridge.mjs"),
                extensions_root / "ark_pi_mcp_bridge.mjs",
            )

        runtime_payload = {
            "node_executable": options.node_executable,
            "pi_cli_path": str(options.pi_cli_path) if options.pi_cli_path is not None else None,
            "mcp_runtime_root": (
                str(options.mcp_runtime_root) if options.mcp_runtime_root is not None else None
            ),
            "session_dir": ".pi/sessions",
            "extensions": list(extension_relpaths),
            "mcp_manifest": mcp_manifest_relpath,
            "project_trust": options.project_trust,
            "load_project_context": options.load_project_context,
            "approve_project_resources": (
                options.project_trust == "always" or options.load_project_context
            ),
            "offline": options.offline,
            "tools": list(options.tools or tuple(str(item) for item in spec.tools)),
            "instructions": instructions_relpath,
            "extra_cli_args": list(options.extra_cli_args),
        }
        write_json_atomic(ark_root / "pi_runtime.json", runtime_payload)

        resolved_defaults = _resolve_defaults(settings, models)
        result = HomeMaterializationResult(
            provider_type="pi",
            home_id=spec.home_id,
            renderer_version=self.renderer_version,
            manifest_schema_version=1,
            manifest_hash="",
            generated_files=_describe_generated_files(root),
            required_env=spec.required_env,
            auth_refs=spec.auth_refs,
            resolved_defaults=resolved_defaults,
            warnings=validation.warnings,
            effective_capabilities=_home_capabilities(spec.home_id, resolved_defaults, bool(mcp_servers)),
            provider_payload=build_provider_payload(
                provider_type="pi",
                payload_type="home_materialization",
                data={
                    "runtime": runtime_payload,
                    "settings_keys": sorted(settings),
                    "mcp_server_names": [str(getattr(item, "name", "")) for item in mcp_servers],
                },
                adapter_version=PI_ADAPTER_VERSION,
                sdk_or_cli_version=PI_CLI_VERSION,
            ),
        )
        result = replace(result, manifest_hash=_manifest_hash(to_jsonable(result)))
        write_json_atomic(ark_root / "home_materialization.json", to_jsonable(result))
        return result

    def refresh_materialization(self, home: "HomeRecord", home_root: Path) -> HomeMaterializationResult:
        manifest = _load_manifest(Path(home_root), expected_hash=home.materialization_manifest_hash)
        result = replace(
            manifest,
            manifest_hash="",
            generated_files=_describe_generated_files(Path(home_root)),
            warnings=manifest.warnings
            + ("materialization manifest explicitly refreshed after application post-processing",),
        )
        result = replace(result, manifest_hash=_manifest_hash(to_jsonable(result)))
        write_json_atomic(Path(home_root) / ".ark" / "home_materialization.json", to_jsonable(result))
        return result

    def initialize(self, home: "HomeRecord", ctx: ProviderExecutionContext) -> HomeInitializationResult:
        runtime = _runtime_config(ctx.home_root)
        node = runtime.get("node_executable") or shutil.which("node")
        cli = runtime.get("pi_cli_path") or shutil.which("pi")
        if not node:
            raise RuntimeError("Pi Home initialization requires a Node executable")
        if not cli or not Path(str(cli)).is_file():
            raise RuntimeError("Pi Home initialization requires an existing Pi CLI path")
        mcp_manifest = runtime.get("mcp_manifest")
        if mcp_manifest:
            mcp_root = runtime.get("mcp_runtime_root")
            if not mcp_root or not (Path(str(mcp_root)) / "node_modules" / "@modelcontextprotocol" / "sdk").exists():
                raise RuntimeError("Pi MCP bridge requires a prepared @modelcontextprotocol/sdk runtime")
        marker = ctx.home_root / ".ark" / "pi_home_initialized.json"
        write_json_atomic(
            marker,
            {
                "provider_type": "pi",
                "home_id": home.home_id,
                "pi_cli_version": PI_CLI_VERSION,
                "node_executable": str(node),
                "pi_cli_path": str(cli),
            },
        )
        return HomeInitializationResult(initialized=True, marker_ref=str(marker))

    def build_execution_context(
        self,
        home: "HomeRecord",
        *,
        run_env: Mapping[str, str] | None,
        workdir: str | None,
    ) -> ProviderExecutionContext:
        root = self.runtime_root / home.home_relpath
        manifest = _load_manifest(root, expected_hash=home.materialization_manifest_hash)
        for item in manifest.generated_files:
            path = root / item.relpath
            if not path.is_file() or _sha256(path) != item.sha256:
                raise RuntimeError(f"Pi Home materialized file hash mismatch: {item.relpath}")
        env = dict(os.environ)
        env.update(home.fixed_env)
        if run_env:
            env.update(run_env)
        for name in home.required_env:
            if not env.get(name):
                raise MissingProviderEnvError(name)
        env.update(
            {
                "PI_CODING_AGENT_DIR": str(root / ".pi"),
                "PI_CODING_AGENT_SESSION_DIR": str(root / ".pi" / "sessions"),
                "PI_TELEMETRY": "0",
            }
        )
        runtime = _runtime_config(root)
        runtime = dict(runtime)
        runtime["node_executable"] = runtime.get("node_executable") or shutil.which("node")
        runtime["pi_cli_path"] = runtime.get("pi_cli_path") or shutil.which("pi")
        if runtime.get("mcp_manifest"):
            env["ARK_PI_MCP_MANIFEST"] = str(root / str(runtime["mcp_manifest"]))
            if runtime.get("mcp_runtime_root"):
                env["ARK_PI_MCP_RUNTIME_ROOT"] = str(runtime["mcp_runtime_root"])
        resolved_defaults = manifest.resolved_defaults
        return ProviderExecutionContext(
            provider_type="pi",
            home_id=home.home_id,
            home_root=root,
            process_environment=env,
            materialization_manifest=manifest,
            workdir=workdir,
            resolved_defaults=resolved_defaults,
            runtime_payload=runtime,
        )


def _read_json_config(spec: ProviderHomeSpec) -> dict[str, object]:
    source = spec.base_config
    if source is None:
        return {}
    if source.mapping is not None:
        return dict(source.mapping)
    if source.path is not None:
        value = json.loads(Path(source.path).read_text(encoding="utf-8"))
    elif source.text is not None:
        value = json.loads(source.text)
    else:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Pi base config must be a JSON object")
    return value


def _read_models(options: PiHomeOptions) -> dict[str, object] | None:
    if options.models is not None:
        return dict(options.models)
    if options.models_json_path is None:
        return None
    value = json.loads(options.models_json_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Pi models config must be a JSON object")
    return value


def _validate_resources(options: PiHomeOptions) -> None:
    duplicates = set(options.skill_paths) & set(options.skill_specs)
    if duplicates:
        raise ValueError(f"duplicate Pi skills: {', '.join(sorted(duplicates))}")
    for name, path in options.skill_paths.items():
        validate_skill_name(name)
        if not Path(path).is_dir():
            raise ValueError(f"Pi skill path must be an existing directory: {path}")
    for name, skill in options.skill_specs.items():
        if validate_skill_name(name) != skill.name:
            raise ValueError(f"Pi skill spec key must match SkillSpec.name: {name}")
    for path in options.extension_paths:
        if not Path(path).is_file():
            raise ValueError(f"Pi extension path must be an existing file: {path}")


def _validate_extra_args(args: tuple[str, ...]) -> None:
    managed = {
        "--mode", "--session", "--session-dir", "--session-id", "--provider", "--model",
        "--extension", "--approve", "--no-approve", "--offline",
    }
    for value in args:
        if value.split("=", 1)[0] in managed:
            raise ValueError(f"Pi extra_cli_args cannot override ARK-managed option: {value}")


def _materialize_skills(
    spec: ProviderHomeSpec,
    options: PiHomeOptions,
    target: Path,
) -> None:
    for name, source in options.skill_paths.items():
        destination = target / validate_skill_name(name)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    skills: dict[str, SkillSpec] = dict(options.skill_specs)
    for item in spec.skills:
        if isinstance(item, SkillSpec):
            skills.setdefault(item.name, item)
    for name, skill in skills.items():
        if name in options.skill_paths:
            continue
        write_skill_spec(skill, target / validate_skill_name(name))


def _materialize_extensions(paths: tuple[Path, ...], target: Path) -> tuple[str, ...]:
    relpaths: list[str] = []
    names: set[str] = set()
    for source in paths:
        name = Path(source).name
        if name in names:
            raise ValueError(f"duplicate Pi extension filename: {name}")
        names.add(name)
        destination = target / name
        shutil.copyfile(source, destination)
        relpaths.append(str(Path(".pi") / "extensions" / name))
    return tuple(relpaths)


def _instruction_text(items: tuple[object, ...]) -> str:
    parts: list[str] = []
    for item in items:
        value = getattr(item, "text", item)
        text = str(value).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts) + ("\n" if parts else "")


def _mcp_manifest(servers: tuple[object, ...]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in servers:
        name = str(getattr(item, "name", "")).strip()
        if not name or name in seen:
            raise ValueError(f"invalid or duplicate Pi MCP server name: {name}")
        seen.add(name)
        records.append(
            {
                "name": name,
                "enabled": bool(getattr(item, "enabled", True)),
                "required": bool(getattr(item, "required", False)),
                "transport": str(getattr(item, "transport", "http")),
                "url": getattr(item, "url", None),
                "command": getattr(item, "command", None),
                "args": list(getattr(item, "args", ()) or ()),
                "cwd": getattr(item, "cwd", None),
                "startup_timeout_sec": getattr(item, "startup_timeout_sec", None),
                "tool_timeout_sec": getattr(item, "tool_timeout_sec", None),
                "enabled_tools": getattr(item, "enabled_tools", None),
                "disabled_tools": getattr(item, "disabled_tools", None),
                "env": dict(getattr(item, "env", {}) or {}),
                "env_vars": list(getattr(item, "env_vars", ()) or ()),
                "bearer_token_env_var": getattr(item, "bearer_token_env_var", None),
                "http_headers": dict(getattr(item, "http_headers", {}) or {}),
                "env_http_headers": dict(getattr(item, "env_http_headers", {}) or {}),
            }
        )
    return {"schema_version": 1, "servers": records}


def _resolve_defaults(
    settings: Mapping[str, object],
    models: Mapping[str, object] | None,
) -> ModelBackendIdentity | None:
    provider = settings.get("defaultProvider")
    model = settings.get("defaultModel")
    if not isinstance(provider, str) or not provider or not isinstance(model, str) or not model:
        return None
    api = "other"
    endpoint = None
    if isinstance(models, Mapping):
        providers = models.get("providers")
        item = providers.get(provider) if isinstance(providers, Mapping) else None
        if isinstance(item, Mapping):
            api = str(item.get("api") or api)
            endpoint = str(item.get("baseUrl")) if item.get("baseUrl") is not None else None
    mode = {
        "openai-completions": "chat_completions",
        "openai-responses": "responses",
        "openai-codex-responses": "responses",
        "anthropic-messages": "messages",
    }.get(api, api)
    return ModelBackendIdentity(
        api_provider=provider,
        api_mode=mode,
        endpoint_id=endpoint,
        requested_model=model,
        model_config_hash=hashlib.sha256(
            json.dumps({"settings": settings, "models": models}, sort_keys=True).encode()
        ).hexdigest(),
    )


def _home_capabilities(
    home_id: str,
    backend: ModelBackendIdentity | None,
    has_mcp: bool,
) -> ProviderCapabilities:
    supports: dict[CapabilityKey, CapabilitySupport] = {}
    for key in (
        CapabilityKey.HOME_BASE_CONFIG,
        CapabilityKey.HOME_TYPED_OVERRIDES,
        CapabilityKey.HOME_RAW_OVERRIDES,
        CapabilityKey.HOME_ENV,
        CapabilityKey.HOME_AUTH_REFS,
        CapabilityKey.HOME_INSTRUCTIONS,
        CapabilityKey.HOME_SKILLS,
        CapabilityKey.HOME_EXTENSIONS,
    ):
        supports[key] = CapabilitySupport(
            capability=key,
            status=CapabilityStatus.NATIVE,
            available=True,
            resolved_for_home_id=home_id,
            evidence_version="pi-home-v1",
        )
    supports[CapabilityKey.HOME_MCP] = CapabilitySupport(
        capability=CapabilityKey.HOME_MCP,
        status=CapabilityStatus.ARK_OWNED,
        available=True,
        requirements=("prepared @modelcontextprotocol/sdk runtime",) if has_mcp else (),
        limitations=(
            "Pi 0.80.10 cannot unregister a projected MCP tool mid-session; removed tools fail closed until the next session",
        ),
        resolved_for_home_id=home_id,
        evidence_version="pi-mcp-bridge-v1",
    )
    for key, mode in (
        (CapabilityKey.MODEL_RESPONSES, "responses"),
        (CapabilityKey.MODEL_CHAT_COMPLETIONS, "chat_completions"),
    ):
        available = backend is not None and backend.api_mode == mode
        supports[key] = CapabilitySupport(
            capability=key,
            status=CapabilityStatus.NATIVE if available else CapabilityStatus.UNSUPPORTED,
            available=available,
            reason=None if available else "effective Pi backend uses a different API mode",
            resolved_for_home_id=home_id,
            resolved_for_backend=backend.backend_key if backend is not None else None,
            evidence_version="pi-home-v1",
        )
    messages = backend is not None and backend.api_mode == "messages"
    known = backend is not None and backend.api_mode in {
        "responses",
        "chat_completions",
        "messages",
    }
    supports[CapabilityKey.MODEL_OTHER_API] = CapabilitySupport(
        capability=CapabilityKey.MODEL_OTHER_API,
        status=(
            CapabilityStatus.NATIVE
            if messages
            else CapabilityStatus.UNSUPPORTED
            if known
            else CapabilityStatus.UNVERIFIED
        ),
        available=messages,
        reason=(
            None
            if messages
            else "effective Pi backend is not resolved to another verified API mode"
        ),
        resolved_for_home_id=home_id,
        resolved_for_backend=backend.backend_key if backend is not None else None,
        evidence_version="pi-home-v1",
    )
    return ProviderCapabilities(
        provider_type="pi",
        supports=supports,
        resolved_for_home_id=home_id,
        resolved_for_backend=backend.backend_key if backend is not None else None,
    )


def _describe_generated_files(root: Path) -> tuple[HomeMaterializedFile, ...]:
    managed: list[Path] = []
    for directory in (root / ".pi", root / ".ark"):
        if directory.exists():
            managed.extend(path for path in directory.rglob("*") if path.is_file())
    files: list[HomeMaterializedFile] = []
    for path in sorted(managed):
        if path.name == "home_materialization.json" or "sessions" in path.parts:
            continue
        relpath = str(path.relative_to(root))
        files.append(
            HomeMaterializedFile(
                relpath=relpath,
                sha256=_sha256(path),
                secret=relpath == ".pi/auth.json",
            )
        )
    return tuple(files)


def _load_manifest(root: Path, *, expected_hash: str | None) -> HomeMaterializationResult:
    payload = read_json(root / ".ark" / "home_materialization.json")
    declared = str(payload.get("manifest_hash") or "")
    if not declared or declared != (expected_hash or declared) or _manifest_hash(payload) != declared:
        raise RuntimeError("Pi Home materialization manifest hash mismatch")
    defaults = payload.get("resolved_defaults")
    identity = None
    if isinstance(defaults, Mapping):
        identity = ModelBackendIdentity(
            api_provider=str(defaults["api_provider"]),
            api_mode=str(defaults["api_mode"]),
            endpoint_id=defaults.get("endpoint_id"),
            requested_model=defaults.get("requested_model"),
            resolved_model=defaults.get("resolved_model"),
            model_config_hash=defaults.get("model_config_hash"),
        )
    return HomeMaterializationResult(
        provider_type="pi",
        home_id=str(payload["home_id"]),
        renderer_version=str(payload["renderer_version"]),
        manifest_schema_version=int(payload["manifest_schema_version"]),
        manifest_hash=declared,
        generated_files=tuple(
            HomeMaterializedFile(
                relpath=str(item["relpath"]),
                sha256=str(item["sha256"]),
                source_fingerprint=item.get("source_fingerprint"),
                secret=bool(item.get("secret", False)),
            )
            for item in payload.get("generated_files", ())
        ),
        required_env=tuple(payload.get("required_env") or ()),
        auth_refs=tuple(payload.get("auth_refs") or ()),
        resolved_defaults=identity,
        warnings=tuple(payload.get("warnings") or ()),
    )


def _runtime_config(root: Path) -> dict[str, object]:
    value = read_json(root / ".ark" / "pi_runtime.json")
    if not isinstance(value, dict):
        raise RuntimeError("invalid Pi runtime configuration")
    return value


def _manifest_hash(payload: object) -> str:
    canonical = dict(payload) if isinstance(payload, Mapping) else {"payload": payload}
    canonical["manifest_hash"] = ""
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
