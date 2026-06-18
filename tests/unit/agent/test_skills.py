from pathlib import Path

import pytest

from agent_runtime_kit.agent.skills import SkillService, SkillSpec, write_skill_spec


def test_skill_spec_normalizes_simple_fields() -> None:
    spec = SkillSpec(
        name=" mathlib-recon ",
        description="  Use Mathlib search. ",
        body="\nRead modules carefully.\n",
        group=" recon ",
        title=" Mathlib Recon ",
        metadata_short_description=" Search Mathlib ",
    )

    assert spec.name == "mathlib-recon"
    assert spec.description == "Use Mathlib search."
    assert spec.body == "\nRead modules carefully.\n"
    assert spec.group == "recon"
    assert spec.title == "Mathlib Recon"
    assert spec.metadata_short_description == "Search Mathlib"


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("name", {"name": ""}),
        ("description", {"description": ""}),
        ("body", {"body": "   "}),
        ("name", {"name": "../bad"}),
        ("name", {"name": "bad/name"}),
        ("name", {"name": "bad\\name"}),
    ],
)
def test_skill_spec_rejects_invalid_core_fields(field: str, kwargs: dict[str, str]) -> None:
    values = {"name": "demo", "description": "Demo skill.", "body": "Do the demo."}
    values.update(kwargs)

    with pytest.raises(ValueError, match=f"skill {field}|invalid skill name"):
        SkillSpec(**values)


@pytest.mark.parametrize("relpath", ["/abs/path", "../escape.md", "refs/../escape.md", "", "SKILL.md"])
def test_skill_spec_rejects_unsafe_extra_file_paths(relpath: str) -> None:
    with pytest.raises(ValueError):
        SkillSpec(
            name="demo",
            description="Demo skill.",
            body="Do the demo.",
            files={relpath: "content"},
        )


def test_write_skill_spec_writes_codex_skill_layout(tmp_path: Path) -> None:
    spec = SkillSpec(
        name="mathlib-recon",
        description="Use when searching Mathlib.",
        metadata_short_description="Search Mathlib",
        body="Read `references/search.md` before using Mathlib tools.",
        openai_metadata={
            "policy": {"allow_implicit_invocation": True},
            "dependencies": {"tools": ["mathlib_search"]},
        },
        files={
            "references/search.md": "# Search\n",
            "scripts/helper.py": b"print('ok')\n",
        },
    )

    target = write_skill_spec(spec, tmp_path / "skills" / "mathlib-recon")

    skill_md = (target / "SKILL.md").read_text(encoding="utf-8")
    assert 'name: "mathlib-recon"' in skill_md
    assert 'description: "Use when searching Mathlib."' in skill_md
    assert "metadata:" in skill_md
    assert 'short-description: "Search Mathlib"' in skill_md
    assert "Read `references/search.md`" in skill_md
    assert (target / "agents" / "openai.yaml").read_text(encoding="utf-8").startswith("{\n")
    assert (target / "references" / "search.md").read_text(encoding="utf-8") == "# Search\n"
    assert (target / "scripts" / "helper.py").read_bytes() == b"print('ok')\n"


def test_write_skill_spec_replaces_existing_directory(tmp_path: Path) -> None:
    target = tmp_path / "demo"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")

    write_skill_spec(
        SkillSpec(name="demo", description="Demo skill.", body="New body."),
        target,
    )

    assert not (target / "old.txt").exists()
    assert "New body." in (target / "SKILL.md").read_text(encoding="utf-8")


def test_skill_service_register_get_list_bundle_and_describe() -> None:
    service = SkillService()
    first = SkillSpec(name="a", description="Skill A.", body="A.", group="common")
    second = SkillSpec(name="b", description="Skill B.", body="B.", group="common")
    third = SkillSpec(name="c", description="Skill C.", body="C.")

    service.register(second)
    service.register(first)
    service.register(third)

    assert service.get("a") is first
    assert [item.name for item in service.list()] == ["a", "b", "c"]
    assert [item.name for item in service.list(group="common")] == ["a", "b"]
    assert service.bundle("b", "a") == {"b": second, "a": first}
    assert service.describe() == {"common": ["a", "b"], "ungrouped": ["c"]}


def test_skill_service_rejects_duplicate_register_and_allows_replace() -> None:
    service = SkillService()
    service.register(SkillSpec(name="demo", description="Demo skill.", body="Old."))

    with pytest.raises(ValueError, match="duplicate skill"):
        service.register(SkillSpec(name="demo", description="Demo skill.", body="Duplicate."))

    replacement = SkillSpec(name="demo", description="Demo skill.", body="New.")
    service.replace(replacement)

    assert service.get("demo") is replacement


def test_skill_service_unknown_bundle_raises_key_error() -> None:
    service = SkillService()

    with pytest.raises(KeyError, match="unknown skill"):
        service.bundle("missing")

