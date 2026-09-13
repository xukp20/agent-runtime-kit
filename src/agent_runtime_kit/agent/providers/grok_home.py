from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, replace
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
from ..store_utils import read_json, write_json_atomic
from ..skills import SkillSpec, write_skill_spec

if TYPE_CHECKING:
    from ..homes import HomeRecord


GROK_ADAPTER_VERSION = "1"
GROK_CLI_VERSION = "1.0.30"
GROK_BINARY_SHA256 = "504dd6546ab991b75d36698242875ce461489cd1f8cd84285873cb55bd5c7d54"
DEFAULT_GROK_TOOLS = ("read_file", "list_dir", "grep")
MCP_GROK_TOOLS = ("search_tool", "use_tool")
SUPPORTED_GROK_TOOLS = frozenset(
    (*DEFAULT_GROK_TOOLS, "run_terminal_cmd", "search_replace", "web_search", "web_fetch", *MCP_GROK_TOOLS)
)
_PROJECT_CONFIG_MARKERS = (
    ".grok/config.toml",
    ".grok/hooks",
    ".grok/plugins",
    ".grok/lsp.json",
    ".mcp.json",
    ".claude/settings.json",
    ".claude/settings.local.json",
    ".claude/plugins",
    ".cursor/mcp.json",
    ".cursor/hooks.json",
    ".envrc",
)


@dataclass(frozen=True)
class GrokHomeOptions:
    auth_json_path: Path | None = Path("/root/.grok/auth.json")
    binary_path: Path | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    tools: tuple[str, ...] | None = None


class GrokHomeRenderer:
    provider_type = "grok"
    renderer_version = "grok-home-v1"

    def __init__(self, *, runtime_root: Path, binary_path: str | Path | None = None) -> None:
        self.runtime_root = Path(runtime_root)
        self.binary_path = Path(binary_path) if binary_path is not None else Path("/root/.grok/bin/grok")

    def validate(self, spec: ProviderHomeSpec) -> HomeValidationResult:
        errors: list[str] = []
        if spec.provider_type != self.provider_type:
            errors.append(f"Grok renderer cannot materialize provider {spec.provider_type}")
        if spec.base_config is not None:
            errors.append("Grok base_config is unsupported; use typed Home options and mcp_servers")
        if spec.config_overrides:
            errors.append("Grok raw config_overrides are unsupported")
        names = []
        for skill in spec.skills:
            if not isinstance(skill, SkillSpec):
                errors.append("Grok skills must be explicit SkillSpec objects")
            elif not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", skill.name):
                errors.append("Grok skill names must be canonical lowercase names with hyphens")
            else:
                names.append(skill.name)
        if len(names) != len(set(names)):
            errors.append("duplicate Grok skill names")
        if spec.extensions:
            errors.append("Grok extensions are unsupported by the isolated curated profile")
        options = spec.provider_options
        if options is not None and not isinstance(options, GrokHomeOptions):
            errors.append("Grok provider_options must be GrokHomeOptions")
            return HomeValidationResult(valid=False, errors=tuple(errors))
        resolved = options if isinstance(options, GrokHomeOptions) else GrokHomeOptions()
        if resolved.tools is not None and not resolved.tools:
            errors.append("Grok 1.0.30 rejects an explicitly empty curated toolset")
        if resolved.tools and spec.tools:
            errors.append("Grok tools must be declared in either spec.tools or GrokHomeOptions.tools, not both")
        tools = _resolve_tools(spec, resolved, errors)
        if any(tool not in SUPPORTED_GROK_TOOLS for tool in tools):
            errors.append("Grok Home contains an unsupported curated tool")
        auth = resolved.auth_json_path
        if auth is not None and not Path(auth).is_file():
            errors.append(f"Grok auth_json_path must be an existing file: {auth}")
        binary = resolved.binary_path or self.binary_path
        if binary is not None and not Path(binary).is_file():
            errors.append(f"Grok binary_path must be an existing file: {binary}")
        try:
            _validate_mcp_servers(tuple(spec.mcp_servers))
        except ValueError as exc:
            errors.append(str(exc))
        return HomeValidationResult(valid=not errors, errors=tuple(errors))

    def materialize(self, spec: ProviderHomeSpec, home_root: Path) -> HomeMaterializationResult:
        validation = self.validate(spec)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        options = spec.provider_options if isinstance(spec.provider_options, GrokHomeOptions) else GrokHomeOptions()
        root = Path(home_root)
        grok_root = root / ".grok"
        ark_root = root / ".ark"
        os_home = ark_root / "os_home"
        for directory in (grok_root, ark_root, os_home):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)

        auth = Path(options.auth_json_path).resolve() if options.auth_json_path is not None else None
        auth_link = grok_root / "auth.json"
        if auth_link.is_symlink() or auth_link.exists():
            auth_link.unlink()
        if auth is not None:
            auth_link.symlink_to(auth)

        mcp_servers = tuple(spec.mcp_servers)
        config_text = _render_mcp_config(mcp_servers)
        if "web_fetch" in _resolve_tools(spec, options, []):
            config_text += "\n[features]\nweb_fetch = true\n"
        if spec.skills:
            skills_root = grok_root / "skills"
            for skill in spec.skills:
                write_skill_spec(skill, skills_root / skill.name)
            config_text += "\n[skills]\npaths = [" + json.dumps(str(skills_root)) + "]\n"
        (grok_root / "config.toml").write_text(config_text, encoding="utf-8")
        (ark_root / "grok-config-authority.toml").write_text(config_text, encoding="utf-8")
        tools = list(_resolve_tools(spec, options, []))
        if mcp_servers:
            for tool in MCP_GROK_TOOLS:
                if tool not in tools:
                    tools.append(tool)
        instructions = _instruction_text(spec.instructions)
        profile = {
            "name": "agent-runtime-kit",
            "description": "Bounded ARK provider session",
            "discoverSkills": bool(spec.skills),
            "inheritSkills": False,
            "skills": [skill.name for skill in spec.skills],
            "agentsMd": True,
            "injectDefaultTools": False,
            "toolConfig": {"tools": [{"id": f"GrokBuild:{tool}"} for tool in tools]},
            "mcpInheritance": "all",
            "disallowedTools": ["Agent", "Task", "task", "workflow", "monitor", "scheduler_create"],
            "background": False,
        }
        profile_text = "---\n" + json.dumps(profile, sort_keys=True) + "\n---\n"
        profile_text += instructions or "Work only on the requested ARK turn.\n"
        profile_text += "Do not start background processes or subagents.\n"
        (ark_root / "grok-profile.md").write_text(profile_text, encoding="utf-8")

        binary = options.binary_path or self.binary_path or Path(shutil.which("grok") or "grok")
        runtime_payload = {
            "binary_path": str(binary),
            "auth_json_path": str(auth) if auth is not None else None,
            "profile_relpath": ".ark/grok-profile.md",
            "grok_home_relpath": ".grok",
            "os_home_relpath": ".ark/os_home",
            "model": options.model,
            "reasoning_effort": options.reasoning_effort,
            "tools": tools,
            "skills": [skill.name for skill in spec.skills],
            "mcp_server_names": [str(getattr(server, "name", "")) for server in mcp_servers],
            "required_mcp_server_names": [
                str(getattr(server, "name", ""))
                for server in mcp_servers
                if bool(getattr(server, "required", False))
            ],
        }
        write_json_atomic(ark_root / "grok_runtime.json", runtime_payload)
        defaults = (
            ModelBackendIdentity(
                api_provider="xai",
                api_mode="other",
                requested_model=options.model,
                reasoning_effort=options.reasoning_effort,
            )
            if options.model
            else None
        )
        result = HomeMaterializationResult(
            provider_type="grok",
            home_id=spec.home_id,
            renderer_version=self.renderer_version,
            manifest_schema_version=1,
            manifest_hash="",
            generated_files=_describe_generated_files(root),
            required_env=spec.required_env,
            auth_refs=spec.auth_refs,
            resolved_defaults=defaults,
            effective_capabilities=_home_capabilities(spec.home_id, bool(mcp_servers)),
            provider_payload=build_provider_payload(
                provider_type="grok",
                payload_type="home_materialization",
                data={
                    "tools": tools,
                    "mcp_server_names": runtime_payload["mcp_server_names"],
                    "auth_reference": str(auth) if auth is not None else None,
                },
                adapter_version=GROK_ADAPTER_VERSION,
                sdk_or_cli_version=GROK_CLI_VERSION,
            ),
        )
        result = replace(result, manifest_hash=_manifest_hash(to_jsonable(result)))
        write_json_atomic(ark_root / "home_materialization.json", to_jsonable(result))
        return result

    def refresh_materialization(self, home: "HomeRecord", home_root: Path) -> HomeMaterializationResult:
        manifest = _load_manifest(Path(home_root), expected_hash=home.materialization_manifest_hash)
        result = replace(manifest, manifest_hash="", generated_files=_describe_generated_files(Path(home_root)))
        result = replace(result, manifest_hash=_manifest_hash(to_jsonable(result)))
        write_json_atomic(Path(home_root) / ".ark" / "home_materialization.json", to_jsonable(result))
        return result

    def commit_lifecycle_materialization(
        self,
        home: "HomeRecord",
        home_root: Path,
        *,
        lifecycle: str,
    ) -> HomeMaterializationResult | None:
        if lifecycle != "session_start":
            raise ValueError(f"unsupported Grok Home materialization lifecycle: {lifecycle}")
        root = Path(home_root)
        manifest = _load_manifest(root, expected_hash=home.materialization_manifest_hash)
        expected = {item.relpath: item.sha256 for item in manifest.generated_files}
        actual = {item.relpath: item.sha256 for item in _describe_generated_files(root)}
        changed = sorted(
            relpath for relpath in expected.keys() | actual.keys() if expected.get(relpath) != actual.get(relpath)
        )
        if not changed:
            return None
        if changed != [".grok/config.toml"]:
            raise RuntimeError(
                "unexpected Grok Home materialization changes at session start: " + ", ".join(changed)
            )
        authority = (root / ".ark" / "grok-config-authority.toml").read_bytes()
        current = (root / ".grok" / "config.toml").read_bytes()
        _validate_marketplace_initialization(authority, current)
        refreshed = self.refresh_materialization(home, root)
        refreshed = replace(
            refreshed,
            manifest_hash="",
            warnings=refreshed.warnings
            + ("Grok marketplace initialization metadata sealed at trusted session-start boundary",),
        )
        refreshed = replace(refreshed, manifest_hash=_manifest_hash(to_jsonable(refreshed)))
        write_json_atomic(root / ".ark" / "home_materialization.json", to_jsonable(refreshed))
        return refreshed

    def initialize(self, home: "HomeRecord", ctx: ProviderExecutionContext) -> HomeInitializationResult:
        runtime = _runtime_config(ctx.home_root)
        binary = Path(str(runtime["binary_path"]))
        if not binary.is_file():
            raise RuntimeError(f"Grok Home initialization requires the configured binary: {binary}")
        digest = _sha256(binary)
        if digest != GROK_BINARY_SHA256:
            raise RuntimeError(f"Grok adapter requires the validated {GROK_CLI_VERSION} binary hash")
        completed = subprocess.run(
            [str(binary), "--version"],
            env=dict(ctx.process_environment),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        output = (completed.stdout + completed.stderr).strip()
        if completed.returncode != 0 or GROK_CLI_VERSION not in output:
            raise RuntimeError(f"Grok adapter requires CLI {GROK_CLI_VERSION}")
        return HomeInitializationResult(initialized=True, marker_ref=str(binary))

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
                raise RuntimeError(f"Grok Home materialized file changed: {item.relpath}")
        if {item.relpath for item in manifest.generated_files} != {item.relpath for item in _describe_generated_files(root)}:
            raise RuntimeError("Grok Home managed file set changed")
        runtime = _runtime_config(root)
        _validate_auth_link(root / ".grok" / "auth.json", runtime.get("auth_json_path"))
        env = dict(os.environ)
        env.update(home.fixed_env)
        env.update(dict(run_env or {}))
        for name in home.required_env:
            if not env.get(name):
                raise MissingProviderEnvError(name)
        env.update(
            {
                "HOME": str(root / str(runtime["os_home_relpath"])),
                "GROK_HOME": str(root / str(runtime["grok_home_relpath"])),
                "GROK_DISABLE_AUTOUPDATER": "1",
            }
        )
        resolved = manifest.resolved_defaults
        return ProviderExecutionContext(
            provider_type="grok",
            home_id=home.home_id,
            home_root=root,
            process_environment=env,
            materialization_manifest=manifest,
            workdir=workdir,
            resolved_defaults=resolved,
            runtime_payload=runtime,
        )


def validate_grok_workdir(workdir: Path, *, managed_skills: bool = False) -> None:
    canonical = Path(workdir).resolve(strict=True)
    root = _git_root(canonical)
    current = canonical
    while True:
        skill_markers = (".grok/skills", ".grok/commands", ".agents/skills", ".agents/commands") if managed_skills else ()
        for marker in (*_PROJECT_CONFIG_MARKERS, *skill_markers):
            candidate = current / marker
            try:
                candidate.lstat()
            except FileNotFoundError:
                continue
            raise RuntimeError(
                f"Grok requires a workspace without executable native configuration: {candidate}"
            )
        if current == root or current.parent == current:
            return
        current = current.parent


def _git_root(workdir: Path) -> Path:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    completed = subprocess.run(
        ["git", "-C", str(workdir), "rev-parse", "--show-toplevel"],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if completed.returncode == 0:
        try:
            root = Path(completed.stdout.strip()).resolve(strict=True)
            workdir.relative_to(root)
            return root
        except (OSError, ValueError):
            pass
    return Path(workdir.anchor)


def _resolve_tools(spec: ProviderHomeSpec, options: GrokHomeOptions, errors: list[str]) -> tuple[str, ...]:
    if options.tools is not None:
        raw = options.tools
    elif spec.tools:
        raw = tuple(str(item) for item in spec.tools)
    else:
        raw = DEFAULT_GROK_TOOLS
    values = tuple(str(item).strip() for item in raw)
    if not values or any(not item for item in values):
        if "Grok 1.0.30 rejects an explicitly empty curated toolset" not in errors:
            errors.append("Grok 1.0.30 rejects an explicitly empty curated toolset")
    if len(values) != len(set(values)):
        errors.append("Grok curated tools must not contain duplicates")
    return values


def _validate_mcp_servers(servers: tuple[object, ...]) -> None:
    seen: set[str] = set()
    for server in servers:
        name = str(getattr(server, "name", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in seen:
            raise ValueError(f"invalid or duplicate Grok MCP server name: {name}")
        seen.add(name)
        transport = str(getattr(server, "transport", "http") or "http")
        command = getattr(server, "command", None)
        url = getattr(server, "url", None)
        if transport == "stdio":
            if not command or url:
                raise ValueError(f"Grok stdio MCP server {name} requires command and no url")
        elif transport == "http":
            if not url or command:
                raise ValueError(f"Grok remote MCP server {name} requires url and no command")
        elif transport == "sse":
            raise ValueError("Grok 1.0.30 SSE MCP transport is unsupported; use streamable HTTP")
        else:
            raise ValueError(f"unsupported Grok MCP transport: {transport}")
        if getattr(server, "cwd", None) is not None:
            raise ValueError("Grok MCP cwd is not supported by native config")
        if getattr(server, "enabled_tools", None) is not None or getattr(server, "disabled_tools", None) is not None:
            raise ValueError("Grok MCP per-server tool filters are not verified")
        _mcp_environment_maps(server)
        if getattr(server, "bearer_token_env_var", None):
            raise ValueError("Grok MCP bearer_token_env_var is unsupported; use an Authorization ${ENV_NAME} header")


def _render_mcp_config(servers: tuple[object, ...]) -> str:
    lines = [
        "[compat.claude]",
        "mcps = false",
        "",
        "[compat.cursor]",
        "mcps = false",
        "",
        "[managed_mcps]",
        "enabled = false",
        "gateway_tools_enabled = false",
        "",
    ]
    for server in servers:
        name = str(getattr(server, "name"))
        lines.append(f"[mcp_servers.{name}]")
        command = getattr(server, "command", None)
        if command is not None:
            lines.append(f"command = {_toml_value(str(command))}")
            args = tuple(getattr(server, "args", ()) or ())
            if args:
                lines.append(f"args = {_toml_value(args)}")
        else:
            lines.append(f"url = {_toml_value(str(getattr(server, 'url')))}")
        lines.append(f"enabled = {_toml_value(bool(getattr(server, 'enabled', True)))}")
        for field in ("startup_timeout_sec", "tool_timeout_sec"):
            value = getattr(server, field, None)
            if value is not None:
                lines.append(f"{field} = {_toml_value(value)}")
        env, headers = _mcp_environment_maps(server)
        if env:
            lines.append(f"env = {_toml_value(env)}")
        if headers:
            lines.append(f"headers = {_toml_value(headers)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _mcp_environment_maps(server: object) -> tuple[dict[str, str], dict[str, str]]:
    env = dict(getattr(server, "env", {}) or {})
    headers = dict(getattr(server, "http_headers", {}) or {})
    for target, names, insensitive in (
        (env, {name: name for name in getattr(server, "env_vars", ()) or ()}, False),
        (headers, dict(getattr(server, "env_http_headers", {}) or {}), True),
    ):
        existing = {key.lower() if insensitive else key for key in target}
        for key, name in names.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"invalid Grok MCP environment variable name: {name}")
            normalized = key.lower() if insensitive else key
            if normalized in existing:
                raise ValueError(f"conflicting Grok MCP fixed/dynamic mapping: {key}")
            existing.add(normalized)
            target[key] = "${" + name + ":-}"
    return env, headers


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, Mapping):
        return "{ " + ", ".join(f"{json.dumps(str(k))} = {_toml_value(v)}" for k, v in sorted(value.items())) + " }"
    raise TypeError(f"unsupported TOML value: {value!r}")


def _instruction_text(items: tuple[object, ...]) -> str:
    parts = [str(getattr(item, "text", item)).strip() for item in items]
    return "\n\n".join(item for item in parts if item) + ("\n" if any(parts) else "")


def _describe_generated_files(root: Path) -> tuple[HomeMaterializedFile, ...]:
    files = []
    for relpath in (
        ".grok/config.toml",
        ".ark/grok-config-authority.toml",
        ".ark/grok-profile.md",
        ".ark/grok_runtime.json",
    ):
        path = root / relpath
        if path.is_file():
            files.append(HomeMaterializedFile(relpath=relpath, sha256=_sha256(path)))
    # Existing Homes retain their original sealed location; new Homes use the native path.
    for skills_root in (root / ".grok" / "skills", root / ".ark" / "grok-skills"):
        if skills_root.is_symlink():
            raise ValueError("Grok managed skill root must not be a symlink")
        for path in sorted(skills_root.rglob("*")):
            if path.is_symlink():
                raise ValueError("Grok managed skills must not contain symlinks")
            if path.is_file():
                files.append(HomeMaterializedFile(relpath=str(path.relative_to(root)), sha256=_sha256(path)))
    return tuple(files)


def _load_manifest(root: Path, *, expected_hash: str | None) -> HomeMaterializationResult:
    payload = read_json(root / ".ark" / "home_materialization.json")
    declared = str(payload.get("manifest_hash") or "")
    if not declared or declared != (expected_hash or declared) or _manifest_hash(payload) != declared:
        raise RuntimeError("Grok Home materialization manifest hash mismatch")
    defaults = payload.get("resolved_defaults")
    identity = None
    if isinstance(defaults, Mapping):
        identity = ModelBackendIdentity(
            api_provider=str(defaults["api_provider"]),
            api_mode=str(defaults["api_mode"]),
            endpoint_id=defaults.get("endpoint_id"),
            requested_model=defaults.get("requested_model"),
            resolved_model=defaults.get("resolved_model"),
            reasoning_effort=defaults.get("reasoning_effort"),
        )
    return HomeMaterializationResult(
        provider_type="grok",
        home_id=str(payload["home_id"]),
        renderer_version=str(payload["renderer_version"]),
        manifest_schema_version=int(payload["manifest_schema_version"]),
        manifest_hash=declared,
        generated_files=tuple(
            HomeMaterializedFile(
                relpath=str(item["relpath"]), sha256=str(item["sha256"]), secret=bool(item.get("secret", False))
            )
            for item in payload.get("generated_files", ())
        ),
        required_env=tuple(payload.get("required_env") or ()),
        auth_refs=tuple(payload.get("auth_refs") or ()),
        resolved_defaults=identity,
        warnings=tuple(payload.get("warnings") or ()),
    )


def _runtime_config(root: Path) -> dict[str, object]:
    value = read_json(root / ".ark" / "grok_runtime.json")
    if not isinstance(value, dict):
        raise RuntimeError("invalid Grok runtime configuration")
    return value


def _validate_auth_link(link: Path, expected: object) -> None:
    if expected is None:
        if link.exists() or link.is_symlink():
            raise RuntimeError("Grok Home has an undeclared auth reference")
        return
    if not link.is_symlink() or link.resolve(strict=True) != Path(str(expected)).resolve(strict=True):
        raise RuntimeError("Grok Home auth reference changed")


def _validate_marketplace_initialization(authority: bytes, current: bytes) -> None:
    expected = tomllib.loads(authority.decode("utf-8"))
    observed = tomllib.loads(current.decode("utf-8"))
    marketplace = observed.pop("marketplace", None)
    if observed != expected or marketplace != {
        "default_skills_installs_purged": True,
        "official_marketplace_auto_installed": True,
        "sources": [
            {
                "name": "xAI Official",
                "git": "https://github.com/xai-org/plugin-marketplace.git",
            }
        ],
    }:
        raise RuntimeError("Grok changed config.toml outside the verified marketplace initialization metadata")


def _home_capabilities(home_id: str, has_mcp: bool) -> ProviderCapabilities:
    supports = {}
    for key in (
        CapabilityKey.HOME_TYPED_OVERRIDES,
        CapabilityKey.HOME_ENV,
        CapabilityKey.HOME_AUTH_REFS,
        CapabilityKey.HOME_INSTRUCTIONS,
        CapabilityKey.HOME_SKILLS,
    ):
        supports[key] = CapabilitySupport(
            capability=key,
            status=CapabilityStatus.NATIVE,
            available=True,
            resolved_for_home_id=home_id,
            evidence_version="grok-1.0.30",
        )
    supports[CapabilityKey.HOME_MCP] = CapabilitySupport(
        capability=CapabilityKey.HOME_MCP,
        status=CapabilityStatus.NATIVE,
        available=True,
        requirements=("Grok native MCP server configuration",) if has_mcp else (),
        resolved_for_home_id=home_id,
        evidence_version="grok-1.0.30",
    )
    return ProviderCapabilities(provider_type="grok", supports=supports, resolved_for_home_id=home_id)


def _manifest_hash(payload: object) -> str:
    canonical = dict(payload) if isinstance(payload, Mapping) else {"payload": payload}
    canonical["manifest_hash"] = ""
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
