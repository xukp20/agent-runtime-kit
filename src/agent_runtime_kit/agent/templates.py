from __future__ import annotations

import re
from collections.abc import Mapping

_SLOT_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\}\}")


class TemplateVariableError(KeyError):
    """Raised when a template slot has no provided value."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name

    def __str__(self) -> str:
        return f"missing template variable: {self.name}"


def render_template(template: str | None, variables: Mapping[str, object]) -> str | None:
    """Render a ``{{slot}}`` template with strict missing-variable checks."""

    if template is None:
        return None

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            raise TemplateVariableError(name)
        return str(variables[name])

    return _SLOT_RE.sub(replace, template)
