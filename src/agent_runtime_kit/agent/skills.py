from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    body: str
    group: str | None = None
    title: str | None = None
    metadata_short_description: str | None = None
    openai_metadata: Mapping[str, object] | None = None
    files: Mapping[str, str | bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = validate_skill_name(self.name)
        description = str(self.description).strip()
        body = str(self.body)
        group = str(self.group).strip() if self.group is not None else None
        title = str(self.title).strip() if self.title is not None else None
        short_description = (
            str(self.metadata_short_description).strip()
            if self.metadata_short_description is not None
            else None
        )
        files = dict(self.files)

        if not description:
            raise ValueError("skill description must not be empty")
        if not body.strip():
            raise ValueError("skill body must not be empty")
        for relpath in files:
            validate_skill_relative_path(relpath)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "group", group or None)
        object.__setattr__(self, "title", title or None)
        object.__setattr__(self, "metadata_short_description", short_description or None)
        object.__setattr__(self, "files", files)


class SkillService:
    """Named Codex skill registry used while defining homes in code."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}

    def register(self, skill: SkillSpec) -> None:
        if skill.name in self._skills:
            raise ValueError(f"duplicate skill: {skill.name}")
        self._skills[skill.name] = skill

    def replace(self, skill: SkillSpec) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillSpec:
        resolved = validate_skill_name(name)
        try:
            return self._skills[resolved]
        except KeyError as exc:
            raise KeyError(f"unknown skill: {resolved}") from exc

    def list(self, group: str | None = None) -> list[SkillSpec]:
        if group is None:
            return sorted(self._skills.values(), key=lambda item: item.name)
        resolved_group = str(group).strip()
        return sorted(
            (item for item in self._skills.values() if item.group == resolved_group),
            key=lambda item: item.name,
        )

    def bundle(self, *names: str) -> dict[str, SkillSpec]:
        return {validate_skill_name(name): self.get(name) for name in names}

    def describe(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for skill in self._skills.values():
            group = skill.group or "ungrouped"
            grouped.setdefault(group, []).append(skill.name)
        return {group: sorted(names) for group, names in sorted(grouped.items())}


def write_skill_spec(skill: SkillSpec, target_dir: Path) -> Path:
    """Write a code-defined skill directory in Codex-compatible layout."""

    target = Path(target_dir)
    if target.exists():
        if not target.is_dir():
            raise ValueError(f"skill target exists and is not a directory: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    (target / "SKILL.md").write_text(_render_skill_markdown(skill), encoding="utf-8")

    if skill.openai_metadata is not None:
        metadata_dir = target / "agents"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "openai.yaml").write_text(
            json.dumps(skill.openai_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    for relpath, content in skill.files.items():
        safe_relpath = validate_skill_relative_path(relpath)
        dest = target / safe_relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            dest.write_bytes(content)
        else:
            dest.write_text(str(content), encoding="utf-8")

    return target


def validate_skill_name(name: str) -> str:
    resolved = str(name).strip()
    if not resolved:
        raise ValueError("skill name must not be empty")
    if resolved in {".", ".."}:
        raise ValueError(f"invalid skill name: {resolved}")
    if "/" in resolved or "\\" in resolved or ".." in resolved:
        raise ValueError(f"invalid skill name: {resolved}")
    return resolved


def validate_skill_relative_path(relpath: str) -> Path:
    raw = str(relpath).strip()
    if not raw:
        raise ValueError("skill file path must not be empty")
    path = Path(raw)
    if path.is_absolute():
        raise ValueError(f"skill file path must be relative: {raw}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"invalid skill file path: {raw}")
    if path.parts and path.parts[0] == "SKILL.md":
        raise ValueError("extra files must not overwrite SKILL.md")
    return path


def _render_skill_markdown(skill: SkillSpec) -> str:
    lines = [
        "---",
        f"name: {_yaml_string(skill.name)}",
        f"description: {_yaml_string(skill.description)}",
    ]
    if skill.metadata_short_description is not None:
        lines.extend(
            [
                "metadata:",
                f"  short-description: {_yaml_string(skill.metadata_short_description)}",
            ]
        )
    lines.extend(["---", "", skill.body.rstrip(), ""])
    return "\n".join(lines)


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)

