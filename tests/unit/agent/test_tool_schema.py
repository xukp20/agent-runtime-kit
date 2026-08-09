from __future__ import annotations

import copy

import pytest

from agent_runtime_kit.agent.provider_contracts.tool_schema import (
    ToolSchemaMaterializationError,
    materialize_tool_input_schema,
)


def test_materializes_nested_array_item_refs_without_mutating_input() -> None:
    schema = {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "maxItems": 5,
                "items": {"$ref": "#/$defs/Candidate"},
            }
        },
        "required": ["candidates"],
        "$defs": {
            "Candidate": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "minLength": 1},
                    "handling": {"type": "string", "enum": ["local", "provider"]},
                },
                "required": ["target", "handling"],
                "additionalProperties": False,
            }
        },
    }
    original = copy.deepcopy(schema)

    result = materialize_tool_input_schema(schema)

    assert schema == original
    assert "$defs" not in result
    items = result["properties"]["candidates"]["items"]
    assert items["type"] == "object"
    assert items["required"] == ["target", "handling"]
    assert items["properties"]["handling"]["enum"] == ["local", "provider"]


def test_materializes_definition_refs_and_json_pointer_escaping() -> None:
    schema = {
        "$ref": "#/definitions/a~1b~0c",
        "definitions": {
            "a/b~c": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
            }
        },
    }

    assert materialize_tool_input_schema(schema) == {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
    }


def test_preserves_ref_sibling_annotations_and_constraints() -> None:
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "$ref": "#/$defs/Names",
                "description": "Exact declaration names.",
                "maxItems": 5,
            }
        },
        "$defs": {
            "Names": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
        },
    }

    result = materialize_tool_input_schema(schema)
    items = result["properties"]["items"]

    assert items == {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "description": "Exact declaration names.",
        "maxItems": 5,
    }


def test_preserves_conflicting_ref_sibling_constraints_as_conjunction() -> None:
    schema = {
        "$ref": "#/$defs/Value",
        "minLength": 3,
        "$defs": {"Value": {"type": "string", "minLength": 1}},
    }

    result = materialize_tool_input_schema(schema)

    assert result == {
        "type": "string",
        "allOf": [{"minLength": 1}, {"minLength": 3}],
    }


@pytest.mark.parametrize(
    ("schema", "code"),
    [
        ({"$ref": "#/$defs/Missing", "$defs": {}}, "schema_ref_unresolved"),
        ({"$ref": "https://example.test/schema.json"}, "schema_external_ref_unsupported"),
        ({"$ref": "#/properties/name"}, "schema_ref_unsupported"),
        ({"$dynamicRef": "#node"}, "schema_dynamic_ref_unsupported"),
    ],
)
def test_rejects_refs_that_cannot_be_materialized_safely(
    schema: dict[str, object],
    code: str,
) -> None:
    with pytest.raises(ToolSchemaMaterializationError) as exc_info:
        materialize_tool_input_schema(schema)

    assert exc_info.value.code == code


def test_rejects_cyclic_refs() -> None:
    schema = {
        "$ref": "#/$defs/Node",
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Node"}},
            }
        },
    }

    with pytest.raises(ToolSchemaMaterializationError) as exc_info:
        materialize_tool_input_schema(schema)

    assert exc_info.value.code == "schema_ref_cycle"
    assert exc_info.value.ref == "#/$defs/Node"


def test_rejects_expansion_beyond_node_budget() -> None:
    schema = {"type": "array", "examples": [None] * 10_001}

    with pytest.raises(ToolSchemaMaterializationError) as exc_info:
        materialize_tool_input_schema(schema)

    assert exc_info.value.code == "schema_expansion_nodes_exceeded"
