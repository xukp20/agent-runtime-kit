"""Provider-visible JSON Schema materialization for tool inputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence


_MAX_EXPANDED_NODES = 10_000
_MAX_EXPANDED_BYTES = 1_000_000
_MAX_EXPANSION_DEPTH = 128
_SUPPORTED_LOCAL_REF_ROOTS = {"$defs", "definitions"}
_DYNAMIC_REF_KEYS = {"$dynamicRef", "$recursiveRef"}
_ANNOTATION_KEYS = {
    "$comment",
    "default",
    "deprecated",
    "description",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
}


class ToolSchemaMaterializationError(ValueError):
    """Raised when a tool schema cannot be materialized without losing meaning."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        path: Sequence[str | int] = (),
        ref: str | None = None,
    ) -> None:
        self.code = code
        self.path = tuple(path)
        self.ref = ref
        location = _format_path(self.path)
        ref_suffix = f" (ref={ref!r})" if ref is not None else ""
        super().__init__(f"{code} at {location}: {message}{ref_suffix}")


class _Materializer:
    def __init__(self, schema: Mapping[str, object]) -> None:
        self._document = schema
        self._expanded_nodes = 0

    def materialize(self) -> dict[str, object]:
        result = self._visit_mapping(
            self._document,
            path=(),
            ref_stack=(),
            depth=0,
            omit_root_definitions=True,
        )
        encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > _MAX_EXPANDED_BYTES:
            raise ToolSchemaMaterializationError(
                code="schema_expansion_bytes_exceeded",
                message=(
                    "materialized schema exceeds the provider-visible byte budget "
                    f"of {_MAX_EXPANDED_BYTES}"
                ),
            )
        return result

    def _visit(
        self,
        value: object,
        *,
        path: tuple[str | int, ...],
        ref_stack: tuple[str, ...],
        depth: int,
    ) -> object:
        if isinstance(value, Mapping):
            return self._visit_mapping(value, path=path, ref_stack=ref_stack, depth=depth)
        self._consume_node(path=path, depth=depth)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [
                self._visit(child, path=(*path, index), ref_stack=ref_stack, depth=depth + 1)
                for index, child in enumerate(value)
            ]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise ToolSchemaMaterializationError(
            code="schema_value_not_json",
            message=f"schema contains unsupported value type {type(value).__name__}",
            path=path,
        )

    def _visit_mapping(
        self,
        value: Mapping[object, object],
        *,
        path: tuple[str | int, ...],
        ref_stack: tuple[str, ...],
        depth: int,
        omit_root_definitions: bool = False,
    ) -> dict[str, object]:
        self._consume_node(path=path, depth=depth)
        for key in value:
            if not isinstance(key, str):
                raise ToolSchemaMaterializationError(
                    code="schema_key_not_string",
                    message="JSON Schema object keys must be strings",
                    path=path,
                )
        for dynamic_key in _DYNAMIC_REF_KEYS:
            if dynamic_key in value:
                raise ToolSchemaMaterializationError(
                    code="schema_dynamic_ref_unsupported",
                    message=f"{dynamic_key} cannot be materialized safely for provider tools",
                    path=(*path, dynamic_key),
                    ref=str(value[dynamic_key]),
                )

        if "$ref" in value:
            raw_ref = value["$ref"]
            if not isinstance(raw_ref, str):
                raise ToolSchemaMaterializationError(
                    code="schema_ref_not_string",
                    message="$ref must be a string",
                    path=(*path, "$ref"),
                )
            target = self._resolve_local_ref(raw_ref, path=(*path, "$ref"))
            if raw_ref in ref_stack:
                raise ToolSchemaMaterializationError(
                    code="schema_ref_cycle",
                    message="cyclic local references are not provider-safe",
                    path=(*path, "$ref"),
                    ref=raw_ref,
                )
            expanded_target = self._visit(
                target,
                path=(*path, "$ref"),
                ref_stack=(*ref_stack, raw_ref),
                depth=depth + 1,
            )
            if not isinstance(expanded_target, dict):
                raise ToolSchemaMaterializationError(
                    code="schema_ref_target_not_object",
                    message="tool schema references must resolve to schema objects",
                    path=(*path, "$ref"),
                    ref=raw_ref,
                )
            siblings = {
                key: self._visit(
                    child,
                    path=(*path, key),
                    ref_stack=ref_stack,
                    depth=depth + 1,
                )
                for key, child in value.items()
                if key != "$ref" and not (omit_root_definitions and key in _SUPPORTED_LOCAL_REF_ROOTS)
            }
            return _conjoin_ref_siblings(expanded_target, siblings)

        result: dict[str, object] = {}
        for key, child in value.items():
            if omit_root_definitions and key in _SUPPORTED_LOCAL_REF_ROOTS:
                continue
            result[key] = self._visit(
                child,
                path=(*path, key),
                ref_stack=ref_stack,
                depth=depth + 1,
            )
        return result

    def _resolve_local_ref(self, ref: str, *, path: tuple[str | int, ...]) -> object:
        if not ref.startswith("#/"):
            code = "schema_external_ref_unsupported" if not ref.startswith("#") else "schema_ref_unsupported"
            raise ToolSchemaMaterializationError(
                code=code,
                message="only local $defs/definitions references are supported",
                path=path,
                ref=ref,
            )
        tokens = [_decode_json_pointer_token(token) for token in ref[2:].split("/")]
        if not tokens or tokens[0] not in _SUPPORTED_LOCAL_REF_ROOTS:
            raise ToolSchemaMaterializationError(
                code="schema_ref_unsupported",
                message="local references must target root $defs or definitions",
                path=path,
                ref=ref,
            )
        current: object = self._document
        for token in tokens:
            if not isinstance(current, Mapping) or token not in current:
                raise ToolSchemaMaterializationError(
                    code="schema_ref_unresolved",
                    message="local reference target does not exist",
                    path=path,
                    ref=ref,
                )
            current = current[token]
        return current

    def _consume_node(self, *, path: tuple[str | int, ...], depth: int) -> None:
        if depth > _MAX_EXPANSION_DEPTH:
            raise ToolSchemaMaterializationError(
                code="schema_expansion_depth_exceeded",
                message=f"schema expansion exceeds maximum depth {_MAX_EXPANSION_DEPTH}",
                path=path,
            )
        self._expanded_nodes += 1
        if self._expanded_nodes > _MAX_EXPANDED_NODES:
            raise ToolSchemaMaterializationError(
                code="schema_expansion_nodes_exceeded",
                message=f"schema expansion exceeds maximum node count {_MAX_EXPANDED_NODES}",
                path=path,
            )


def materialize_tool_input_schema(schema: Mapping[str, object]) -> dict[str, object]:
    """Return a self-contained provider-visible copy of a tool input schema.

    The canonical input is never mutated. Local references rooted at ``$defs`` or
    ``definitions`` are expanded; reference forms that cannot be represented safely
    are rejected with :class:`ToolSchemaMaterializationError`.
    """

    if not isinstance(schema, Mapping):
        raise ToolSchemaMaterializationError(
            code="schema_root_not_object",
            message="tool input schema must be a JSON object",
        )
    return _Materializer(schema).materialize()


def _conjoin_ref_siblings(
    target: dict[str, object],
    siblings: dict[str, object],
) -> dict[str, object]:
    if not siblings:
        return target
    merged = dict(target)
    conflicting_constraints: dict[str, object] = {}
    for key, value in siblings.items():
        if key not in merged or merged[key] == value:
            merged[key] = value
        elif key in _ANNOTATION_KEYS:
            merged[key] = value
        else:
            conflicting_constraints[key] = value
    if not conflicting_constraints:
        return merged

    base = {key: value for key, value in merged.items() if key not in conflicting_constraints}
    conflict_base = {key: target[key] for key in conflicting_constraints}
    base["allOf"] = [conflict_base, conflicting_constraints]
    return base


def _decode_json_pointer_token(token: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(token):
        char = token[index]
        if char != "~":
            result.append(char)
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
            raise ToolSchemaMaterializationError(
                code="schema_ref_invalid_pointer",
                message="JSON Pointer contains an invalid escape sequence",
                ref=token,
            )
        result.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(result)


def _format_path(path: Sequence[str | int]) -> str:
    if not path:
        return "$"
    rendered = "$"
    for item in path:
        if isinstance(item, int):
            rendered += f"[{item}]"
        else:
            rendered += f".{item}"
    return rendered
