from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextFragment:
    key: str
    text: str
    group: str | None = None
    title: str | None = None

    def __post_init__(self) -> None:
        key = str(self.key).strip()
        text = str(self.text)
        group = str(self.group).strip() if self.group is not None else None
        title = str(self.title).strip() if self.title is not None else None
        if not key:
            raise ValueError("fragment key must not be empty")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "group", group or None)
        object.__setattr__(self, "title", title or None)


class InstructionService:
    """Named text fragment registry plus compose helper."""

    def __init__(self) -> None:
        self._fragments: dict[str, TextFragment] = {}

    def register(self, fragment: TextFragment) -> None:
        if fragment.key in self._fragments:
            raise ValueError(f"duplicate text fragment: {fragment.key}")
        if not fragment.text.strip():
            raise ValueError(f"text fragment is empty: {fragment.key}")
        self._fragments[fragment.key] = fragment

    def replace(self, fragment: TextFragment) -> None:
        if not fragment.text.strip():
            raise ValueError(f"text fragment is empty: {fragment.key}")
        self._fragments[fragment.key] = fragment

    def get(self, key: str) -> TextFragment:
        resolved = str(key).strip()
        try:
            return self._fragments[resolved]
        except KeyError as exc:
            raise KeyError(f"unknown text fragment: {resolved}") from exc

    def text(self, key: str) -> str:
        return self.get(key).text

    def list(self, group: str | None = None) -> list[TextFragment]:
        if group is None:
            return sorted(self._fragments.values(), key=lambda item: item.key)
        resolved_group = str(group).strip()
        return sorted(
            (item for item in self._fragments.values() if item.group == resolved_group),
            key=lambda item: item.key,
        )

    def compose(self, *items: str | TextFragment, sep: str = "\n\n") -> str:
        parts: list[str] = []
        for item in items:
            if isinstance(item, TextFragment):
                text = item.text
            else:
                value = str(item)
                text = self._fragments[value].text if value in self._fragments else value
            if text.strip():
                parts.append(text.strip())
        return sep.join(parts)

    def describe(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for fragment in self._fragments.values():
            group = fragment.group or "ungrouped"
            grouped.setdefault(group, []).append(fragment.key)
        return {group: sorted(keys) for group, keys in sorted(grouped.items())}
