from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


PROVIDER_TYPE = "opencode"
ADAPTER_VERSION = "1"


@dataclass(frozen=True)
class OpenCodeHomeOptions:
    binary_path: str | Path = "opencode"
    server_start_timeout_s: float = 15.0
    allow_project_config: bool = False
    config_filename: str = "opencode.json"
    agent_files: Mapping[str, str] = field(default_factory=dict)
    command_files: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.server_start_timeout_s <= 0:
            raise ValueError("server_start_timeout_s must be positive")
        if Path(self.config_filename).name != self.config_filename:
            raise ValueError("config_filename must be a file name")
        for collection_name, values in (
            ("agent_files", self.agent_files),
            ("command_files", self.command_files),
        ):
            for name, content in values.items():
                if not name or Path(name).name != name or name in {".", ".."}:
                    raise ValueError(f"invalid OpenCode {collection_name} name: {name}")
                if not str(content).strip():
                    raise ValueError(f"OpenCode {collection_name} content must not be empty: {name}")


@dataclass(frozen=True)
class OpenCodeRunOptions:
    provider_id: str | None = None
    model_id: str | None = None
    agent: str | None = None
    variant: str | None = None
    tools: Mapping[str, bool] = field(default_factory=dict)
    output_format: object | None = None


@dataclass(frozen=True)
class OpenCodeNativeLocator:
    agent_id: str
    directory: str
    database_path: str
    runtime_relpath: str

    def as_dict(self) -> dict[str, str]:
        return {
            "agent_id": self.agent_id,
            "directory": self.directory,
            "database_path": self.database_path,
            "runtime_relpath": self.runtime_relpath,
        }


def parse_native_locator(value: object) -> OpenCodeNativeLocator:
    if not isinstance(value, Mapping):
        raise ValueError("OpenCode session locator requires a mapping native_locator")
    required = ("agent_id", "directory", "database_path", "runtime_relpath")
    missing = [key for key in required if not isinstance(value.get(key), str) or not value[key]]
    if missing:
        raise ValueError(f"OpenCode native locator is missing: {', '.join(missing)}")
    return OpenCodeNativeLocator(**{key: str(value[key]) for key in required})
