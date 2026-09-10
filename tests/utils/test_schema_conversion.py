"""Unit tests for src/utils/schema_conversion.py."""

import json
import re
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from src.utils.schema_conversion import json_response_schema_to_pydantic


def _object(properties: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"type": "object", "properties": properties, **extra}


class TestPrimitives:
    def test_flat_object_with_primitives(self):
        model = json_response_schema_to_pydantic(
            _object(
                {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "score": {"type": "number"},
                    "active": {"type": "boolean"},
                },
                required=["name", "age"],
            )
        )
        instance = model.model_validate(
            {"name": "ada", "age": 36, "score": 9.5, "active": True}
        )
        assert instance.name == "ada"  # pyright: ignore
        assert instance.age == 36  # pyright: ignore

    def test_required_field_missing_fails(self):
        model = json_response_schema_to_pydantic(
            _object({"name": {"type": "string"}}, required=["name"])
        )
        with pytest.raises(ValidationError):
            model.model_validate({})

    def test_optional_field_defaults_to_none(self):
        model = json_response_schema_to_pydantic(
            _object({"nickname": {"type": "string"}})
        )
        instance = model.model_validate({})
        assert instance.nickname is None  # pyright: ignore

    def test_default_value(self):
        model = json_response_schema_to_pydantic(
            _object({"count": {"type": "integer", "default": 3}})
        )
        assert model.model_validate({}).count == 3  # pyright: ignore

    def test_null_type(self):
        model = json_response_schema_to_pydantic(
            _object({"nothing": {"type": "null"}}, required=["nothing"])
        )
        assert model.model_validate({"nothing": None}).nothing is None  # pyright: ignore

    def test_default_wins_over_required(self):
        model = json_response_schema_to_pydantic(
            _object({"count": {"type": "integer", "default": 3}}, required=["count"])
        )
        assert model.model_validate({}).count == 3  # pyright: ignore


class TestNesting:
    def test_nested_object(self):
        model = json_response_schema_to_pydantic(
            _object(
                {
                    "address": _object(
                        {
                            "city": {"type": "string"},
                            "geo": _object(
                                {"lat": {"type": "number"}}, required=["lat"]
                            ),
                        },
                        required=["city", "geo"],
                    )
                },
                required=["address"],
            )
        )
        instance = model.model_validate(
            {"address": {"city": "oakland", "geo": {"lat": 37.8}}}
        )
        assert instance.address.geo.lat == 37.8  # pyright: ignore

    def test_array_of_objects(self):
        model = json_response_schema_to_pydantic(
            _object(
                {
                    "items": {
                        "type": "array",
                        "items": _object(
                            {"food": {"type": "string"}}, required=["food"]
                        ),
                    }
                },
                required=["items"],
            )
        )
        instance = model.model_validate({"items": [{"food": "sushi"}]})
        assert instance.items[0].food == "sushi"  # pyright: ignore

    def test_array_without_items_accepts_anything(self):
        model = json_response_schema_to_pydantic(
            _object({"stuff": {"type": "array"}}, required=["stuff"])
        )
        instance = model.model_validate({"stuff": [1, "two", {"three": 3}]})
        assert len(instance.stuff) == 3  # pyright: ignore

    def test_nested_model_name_collision(self):
        # Two sibling objects whose name hints collide must not clash.
        model = json_response_schema_to_pydantic(
            _object(
                {
                    "a": _object({"x b": _object({"v": {"type": "string"}})}),
                    "a_x": _object({"b": _object({"v": {"type": "integer"}})}),
                }
            )
        )
        instance = model.model_validate(
            {"a": {"x b": {"v": "s"}}, "a_x": {"b": {"v": 1}}}
        )
        assert instance.a_x.b.v == 1  # pyright: ignore


class TestEnumsAndUnions:
    def test_string_enum(self):
        model = json_response_schema_to_pydantic(
            _object(
                {"sentiment": {"enum": ["loves", "hates"]}},
                required=["sentiment"],
            )
        )
        assert model.model_validate({"sentiment": "loves"}).sentiment == "loves"  # pyright: ignore
        with pytest.raises(ValidationError):
            model.model_validate({"sentiment": "meh"})

    def test_int_enum_and_null_member(self):
        model = json_response_schema_to_pydantic(
            _object({"level": {"enum": [1, 2, None]}}, required=["level"])
        )
        assert model.model_validate({"level": None}).level is None  # pyright: ignore
        assert model.model_validate({"level": 2}).level == 2  # pyright: ignore

    def test_invalid_enum_value_type(self):
        with pytest.raises(ValueError, match="enum values"):
            json_response_schema_to_pydantic(_object({"bad": {"enum": [[1]]}}))

    def test_anyof_with_null_is_optional(self):
        model = json_response_schema_to_pydantic(
            _object(
                {"maybe": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
                required=["maybe"],
            )
        )
        assert model.model_validate({"maybe": None}).maybe is None  # pyright: ignore
        assert model.model_validate({"maybe": "x"}).maybe == "x"  # pyright: ignore

    def test_oneof_union(self):
        model = json_response_schema_to_pydantic(
            _object(
                {"value": {"oneOf": [{"type": "integer"}, {"type": "string"}]}},
                required=["value"],
            )
        )
        assert model.model_validate({"value": 5}).value == 5  # pyright: ignore

    def test_type_list_form(self):
        model = json_response_schema_to_pydantic(
            _object({"name": {"type": ["string", "null"]}}, required=["name"])
        )
        assert model.model_validate({"name": None}).name is None  # pyright: ignore

    def test_all_null_enum_degenerates_to_none(self):
        model = json_response_schema_to_pydantic(
            _object({"nothing": {"enum": [None]}}, required=["nothing"])
        )
        assert model.model_validate({"nothing": None}).nothing is None  # pyright: ignore
        with pytest.raises(ValidationError):
            model.model_validate({"nothing": "x"})

    def test_union_of_objects(self):
        model = json_response_schema_to_pydantic(
            _object(
                {
                    "pet": {
                        "anyOf": [
                            _object({"meows": {"type": "boolean"}}, required=["meows"]),
                            _object({"barks": {"type": "boolean"}}, required=["barks"]),
                        ]
                    }
                },
                required=["pet"],
            )
        )
        instance = model.model_validate({"pet": {"barks": True}})
        assert instance.pet.barks is True  # pyright: ignore


class TestRejections:
    @pytest.mark.parametrize(
        "construct,schema",
        [
            ("$defs", _object({"a": {"type": "object", "$defs": {}}})),
            ("definitions", _object({"a": {"type": "object", "definitions": {}}})),
            ("allOf", _object({"a": {"allOf": [{"type": "string"}]}})),
            ("not", _object({"a": {"not": {"type": "string"}}})),
            ("if", _object({"a": {"if": {"type": "string"}}})),
            (
                "patternProperties",
                _object({"a": {"type": "object", "patternProperties": {}}}),
            ),
        ],
    )
    def test_unsupported_constructs(self, construct: str, schema: dict[str, Any]):
        with pytest.raises(ValueError, match=re.escape(construct)):
            json_response_schema_to_pydantic(schema)

    def test_error_message_includes_path(self):
        with pytest.raises(
            ValueError, match=r"unsupported \$ref '#/x'.*at properties\.address"
        ):
            json_response_schema_to_pydantic(_object({"address": {"$ref": "#/x"}}))

    def test_schema_valued_additional_properties(self):
        with pytest.raises(ValueError, match="additionalProperties"):
            json_response_schema_to_pydantic(
                _object(
                    {
                        "map": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        }
                    }
                )
            )

    def test_boolean_schema(self):
        with pytest.raises(ValueError, match="boolean schemas"):
            json_response_schema_to_pydantic(_object({"anything": True}))

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="unsupported type 'date'"):
            json_response_schema_to_pydantic(_object({"when": {"type": "date"}}))


class TestRefs:
    def test_ref_into_defs(self):
        model = json_response_schema_to_pydantic(
            _object(
                {"address": {"$ref": "#/$defs/Address"}},
                required=["address"],
                **{
                    "$defs": {
                        "Address": _object(
                            {"city": {"type": "string"}}, required=["city"]
                        )
                    }
                },
            )
        )
        instance = model.model_validate({"address": {"city": "Berlin"}})
        assert instance.address.city == "Berlin"  # pyright: ignore

    def test_ref_into_definitions_alias(self):
        model = json_response_schema_to_pydantic(
            _object(
                {"item": {"$ref": "#/definitions/Item"}},
                required=["item"],
                definitions={"Item": {"type": "string"}},
            )
        )
        assert model.model_validate({"item": "x"}).item == "x"  # pyright: ignore

    def test_pydantic_nested_model_schema(self):
        """The real-world motivation: model_json_schema() of a nested Pydantic
        model emits $defs/$ref and must convert cleanly."""

        class Preference(BaseModel):
            food: str
            confidence: float

        class Preferences(BaseModel):
            preferences: list[Preference]
            summary: str

        model = json_response_schema_to_pydantic(Preferences.model_json_schema())
        instance = model.model_validate(
            {
                "preferences": [{"food": "sushi", "confidence": 0.9}],
                "summary": "likes sushi",
            }
        )
        assert instance.preferences[0].food == "sushi"  # pyright: ignore

    def test_root_ref(self):
        model = json_response_schema_to_pydantic(
            {
                "$ref": "#/$defs/Root",
                "$defs": {
                    "Root": _object({"ok": {"type": "boolean"}}, required=["ok"])
                },
            }
        )
        assert model.model_validate({"ok": True}).ok is True  # pyright: ignore

    def test_ref_sibling_keys_overlay_target(self):
        model = json_response_schema_to_pydantic(
            _object(
                {"count": {"$ref": "#/$defs/Count", "default": 3}},
                **{"$defs": {"Count": {"type": "integer"}}},
            )
        )
        assert model.model_validate({}).count == 3  # pyright: ignore

    def test_same_def_referenced_twice(self):
        model = json_response_schema_to_pydantic(
            _object(
                {
                    "home": {"$ref": "#/$defs/Address"},
                    "work": {"$ref": "#/$defs/Address"},
                },
                required=["home", "work"],
                **{"$defs": {"Address": _object({"city": {"type": "string"}})}},
            )
        )
        instance = model.model_validate(
            {"home": {"city": "Berlin"}, "work": {"city": "Kyiv"}}
        )
        assert instance.work.city == "Kyiv"  # pyright: ignore

    def test_chained_refs(self):
        model = json_response_schema_to_pydantic(
            _object(
                {"a": {"$ref": "#/$defs/A"}},
                required=["a"],
                **{"$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"type": "string"}}},
            )
        )
        assert model.model_validate({"a": "x"}).a == "x"  # pyright: ignore

    def test_unreferenced_invalid_def_is_ignored(self):
        model = json_response_schema_to_pydantic(
            _object(
                {"name": {"type": "string"}},
                **{"$defs": {"Broken": {"allOf": [{"type": "string"}]}}},
            )
        )
        assert model.model_validate({"name": "x"}).name == "x"  # pyright: ignore

    @pytest.mark.parametrize(
        "ref",
        ["#", "#/x", "#/$defs/a/b", "#/properties/a", "https://x.dev/s.json#/$defs/X"],
    )
    def test_unsupported_ref_forms(self, ref: str):
        with pytest.raises(ValueError, match=r"unsupported \$ref"):
            json_response_schema_to_pydantic(
                _object(
                    {"a": {"$ref": ref}},
                    **{"$defs": {"a": {"type": "string"}}},
                )
            )

    def test_unknown_definition(self):
        with pytest.raises(ValueError, match="unknown definition"):
            json_response_schema_to_pydantic(
                _object({"a": {"$ref": "#/$defs/Missing"}}, **{"$defs": {}})
            )

    def test_direct_recursion_rejected(self):
        with pytest.raises(ValueError, match=r"recursive \$ref.*cycle: Node -> Node"):
            json_response_schema_to_pydantic(
                _object(
                    {"tree": {"$ref": "#/$defs/Node"}},
                    **{
                        "$defs": {
                            "Node": _object(
                                {
                                    "children": {
                                        "type": "array",
                                        "items": {"$ref": "#/$defs/Node"},
                                    }
                                }
                            )
                        }
                    },
                )
            )

    def test_mutual_recursion_rejected(self):
        with pytest.raises(ValueError, match=r"cycle: A -> B -> A"):
            json_response_schema_to_pydantic(
                _object(
                    {"a": {"$ref": "#/$defs/A"}},
                    **{
                        "$defs": {
                            "A": _object({"b": {"$ref": "#/$defs/B"}}),
                            "B": _object({"a": {"$ref": "#/$defs/A"}}),
                        }
                    },
                )
            )

    def test_recursive_pydantic_model_rejected(self):
        class Node(BaseModel):
            value: str
            children: list["Node"] = []

        with pytest.raises(ValueError, match=r"recursive \$ref"):
            json_response_schema_to_pydantic(Node.model_json_schema())

    def test_ref_expansion_counts_against_node_budget(self):
        """A doubling ref chain (billion laughs) is stopped by max_nodes."""
        defs = {
            f"L{i}": _object(
                {
                    "a": {"$ref": f"#/$defs/L{i + 1}"},
                    "b": {"$ref": f"#/$defs/L{i + 1}"},
                }
            )
            for i in range(10)
        }
        defs["L10"] = {"type": "string"}
        with pytest.raises(ValueError, match="maximum of .* nodes"):
            json_response_schema_to_pydantic(
                _object({"root": {"$ref": "#/$defs/L0"}}, **{"$defs": defs})
            )

    def test_duplicate_name_across_defs_and_definitions(self):
        with pytest.raises(ValueError, match="appears in both"):
            json_response_schema_to_pydantic(
                _object(
                    {"a": {"$ref": "#/$defs/X"}},
                    **{
                        "$defs": {"X": {"type": "string"}},
                        "definitions": {"X": {"type": "integer"}},
                    },
                )
            )

    def test_root_must_be_object(self):
        with pytest.raises(ValueError, match="root schema"):
            json_response_schema_to_pydantic({"type": "string"})

    def test_root_must_be_dict(self):
        with pytest.raises(ValueError, match="JSON Schema object"):
            json_response_schema_to_pydantic(["not", "a", "schema"])  # pyright: ignore

    def test_no_recognizable_type(self):
        with pytest.raises(ValueError, match="no recognizable type"):
            json_response_schema_to_pydantic(_object({"mystery": {}}))

    def test_depth_limit(self):
        schema: dict[str, Any] = {"type": "string"}
        for _ in range(25):
            schema = _object({"inner": schema})
        with pytest.raises(ValueError, match="maximum depth"):
            json_response_schema_to_pydantic(schema)

    def test_node_limit(self):
        schema = _object({f"field_{i}": {"type": "string"} for i in range(600)})
        with pytest.raises(ValueError, match="maximum of 500 nodes"):
            json_response_schema_to_pydantic(schema)

    def test_property_schema_not_an_object(self):
        with pytest.raises(ValueError, match="schema must be an object"):
            json_response_schema_to_pydantic(_object({"a": "string"}))

    @pytest.mark.parametrize("members", [[], "not-a-list"])
    def test_malformed_anyof(self, members: Any):
        with pytest.raises(ValueError, match="'anyOf' must be a non-empty array"):
            json_response_schema_to_pydantic(_object({"a": {"anyOf": members}}))

    def test_empty_type_list(self):
        with pytest.raises(ValueError, match="'type' array must not be empty"):
            json_response_schema_to_pydantic(_object({"a": {"type": []}}))

    @pytest.mark.parametrize("values", [[], "loves"])
    def test_malformed_enum(self, values: Any):
        with pytest.raises(ValueError, match="'enum' must be a non-empty array"):
            json_response_schema_to_pydantic(_object({"a": {"enum": values}}))

    def test_properties_not_an_object(self):
        with pytest.raises(ValueError, match="'properties' must be an object"):
            json_response_schema_to_pydantic({"type": "object", "properties": []})

    @pytest.mark.parametrize("required", ["a", [1]])
    def test_malformed_required(self, required: Any):
        with pytest.raises(ValueError, match="'required' must be an array of strings"):
            json_response_schema_to_pydantic(
                _object({"a": {"type": "string"}}, required=required)
            )

    def test_empty_property_name(self):
        with pytest.raises(ValueError, match="property names"):
            json_response_schema_to_pydantic(_object({"": {"type": "string"}}))


class TestLenientAcceptance:
    def test_additional_properties_false_ignored(self):
        model = json_response_schema_to_pydantic(
            _object(
                {"known": {"type": "string"}},
                required=["known"],
                additionalProperties=False,
            )
        )
        instance = model.model_validate({"known": "x", "extra": "dropped"})
        assert instance.model_dump() == {"known": "x"}

    def test_root_dollar_schema_ignored(self):
        model = json_response_schema_to_pydantic(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {"a": {"type": "string"}},
            }
        )
        assert issubclass(model, BaseModel)

    def test_empty_properties(self):
        model = json_response_schema_to_pydantic({"type": "object", "properties": {}})
        assert model.model_validate({}).model_dump() == {}

    def test_missing_properties_with_object_type(self):
        model = json_response_schema_to_pydantic({"type": "object"})
        assert model.model_validate({"anything": 1}).model_dump() == {}

    def test_missing_type_with_properties_treated_as_object(self):
        model = json_response_schema_to_pydantic(
            {"properties": {"a": {"type": "string"}}, "required": ["a"]}
        )
        assert model.model_validate({"a": "x"}).a == "x"  # pyright: ignore

    def test_required_naming_unknown_property_ignored(self):
        model = json_response_schema_to_pydantic(
            _object({"a": {"type": "string"}}, required=["a", "ghost"])
        )
        assert model.model_validate({"a": "x"}).a == "x"  # pyright: ignore


class TestFieldMetadata:
    def test_description_propagates(self):
        model = json_response_schema_to_pydantic(
            _object({"food": {"type": "string", "description": "A food item"}})
        )
        generated = model.model_json_schema()
        assert generated["properties"]["food"]["description"] == "A food item"

    def test_constraint_hints_pass_through_unenforced(self):
        model = json_response_schema_to_pydantic(
            _object(
                {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 3,
                    }
                },
                required=["tags"],
            )
        )
        generated = model.model_json_schema()
        assert generated["properties"]["tags"]["maxItems"] == 3
        # Not enforced: more than maxItems still validates.
        instance = model.model_validate({"tags": ["a", "b", "c", "d"]})
        assert len(instance.tags) == 4  # pyright: ignore

    def test_non_identifier_key_alias_round_trip(self):
        model = json_response_schema_to_pydantic(
            _object(
                {"my-key": {"type": "string"}, "_private": {"type": "integer"}},
                required=["my-key"],
            )
        )
        instance = model.model_validate({"my-key": "v", "_private": 7})
        dumped = instance.model_dump_json(by_alias=True)
        assert '"my-key":"v"' in dumped
        assert '"_private":7' in dumped

    def test_digit_leading_key_gets_field_prefix(self):
        model = json_response_schema_to_pydantic(
            _object({"123": {"type": "integer"}}, required=["123"])
        )
        instance = model.model_validate({"123": 7})
        assert instance.model_dump(by_alias=True) == {"123": 7}

    def test_sanitized_key_collision_round_trip(self):
        # "my-key" sanitizes to "my_key", which then collides with the real
        # "my_key" property; both must survive with their original JSON keys.
        model = json_response_schema_to_pydantic(
            _object(
                {"my-key": {"type": "string"}, "my_key": {"type": "integer"}},
                required=["my-key", "my_key"],
            )
        )
        instance = model.model_validate({"my-key": "v", "my_key": 7})
        assert instance.model_dump(by_alias=True) == {"my-key": "v", "my_key": 7}

    def test_digit_leading_model_name(self):
        model = json_response_schema_to_pydantic(
            _object({"a": {"type": "string"}}), model_name="123"
        )
        assert model.__name__ == "Model123"


class TestZodCompatibility:
    def test_zod4_tojsonschema_output_converts(self):
        # Captured shape of zod 4's z.toJSONSchema() for
        # z.object({ preferences: z.array(z.object({ food: z.string(),
        #   sentiment: z.enum(["loves","hates"]) })), summary: z.string(),
        #   note: z.string().optional() })
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "preferences": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "food": {"type": "string"},
                            "sentiment": {
                                "type": "string",
                                "enum": ["loves", "hates"],
                            },
                        },
                        "required": ["food", "sentiment"],
                        "additionalProperties": False,
                    },
                },
                "summary": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["preferences", "summary"],
            "additionalProperties": False,
        }
        model = json_response_schema_to_pydantic(schema)
        instance = model.model_validate(
            {
                "preferences": [{"food": "sushi", "sentiment": "loves"}],
                "summary": "likes sushi",
            }
        )
        assert instance.preferences[0].sentiment == "loves"  # pyright: ignore
        assert instance.note is None  # pyright: ignore


class TestCustomGuardLimits:
    def test_custom_max_depth(self):
        schema: dict[str, Any] = {"type": "string"}
        for _ in range(5):
            schema = _object({"inner": schema})
        with pytest.raises(ValueError, match="maximum depth of 3"):
            json_response_schema_to_pydantic(schema, max_depth=3)

    def test_custom_max_nodes(self):
        schema = _object({f"f{i}": {"type": "string"} for i in range(20)})
        with pytest.raises(ValueError, match="maximum of 10 nodes"):
            json_response_schema_to_pydantic(schema, max_nodes=10)

    def test_depth_exactly_at_limit_allowed(self):
        # Leaf sits at depth == max_depth; only depth > max_depth must fail.
        schema: dict[str, Any] = {"type": "string"}
        for _ in range(3):
            schema = _object({"inner": schema})
        model = json_response_schema_to_pydantic(schema, max_depth=3)
        assert issubclass(model, BaseModel)


# The wiki spec's own request example (dialectic-enhancements §3.A.1).
SPEC_EXAMPLE_SCHEMA = _object(
    {
        "preferences": {
            "type": "array",
            "items": _object(
                {
                    "food": {"type": "string"},
                    "sentiment": {
                        "type": "string",
                        "enum": ["loves", "likes", "neutral", "dislikes", "hates"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                required=["food", "sentiment"],
            ),
            "maxItems": 3,
        },
        "summary": {"type": "string"},
    },
    required=["preferences", "summary"],
)


class TestEndToEnd:
    """Table tests running the full pipeline the server runs: convert the
    caller's schema, validate a payload against the generated model, and
    serialize it back with model_dump_json(by_alias=True)."""

    @pytest.mark.parametrize(
        "schema,payload,expected",
        [
            pytest.param(
                SPEC_EXAMPLE_SCHEMA,
                {
                    "preferences": [
                        {
                            "food": "dark roast coffee",
                            "sentiment": "loves",
                            "confidence": 0.95,
                        },
                        {"food": "sushi", "sentiment": "likes"},
                    ],
                    "summary": "Coffee enthusiast.",
                },
                {
                    "preferences": [
                        {
                            "food": "dark roast coffee",
                            "sentiment": "loves",
                            "confidence": 0.95,
                        },
                        {"food": "sushi", "sentiment": "likes", "confidence": None},
                    ],
                    "summary": "Coffee enthusiast.",
                },
                id="spec-example",
            ),
            pytest.param(
                _object(
                    {
                        "user": _object(
                            {
                                "name": {"type": "string"},
                                "location": _object(
                                    {
                                        "lat": {"type": "number"},
                                        "lon": {"type": "number"},
                                    },
                                    required=["lat", "lon"],
                                ),
                            },
                            required=["name", "location"],
                        )
                    },
                    required=["user"],
                ),
                {"user": {"name": "ada", "location": {"lat": 37.8, "lon": -122.3}}},
                {"user": {"name": "ada", "location": {"lat": 37.8, "lon": -122.3}}},
                id="nested-three-levels",
            ),
            pytest.param(
                _object(
                    {"my-key": {"type": "string"}, "first name": {"type": "string"}},
                    required=["my-key"],
                ),
                {"my-key": "v", "first name": "Ada"},
                {"my-key": "v", "first name": "Ada"},
                id="alias-keys-round-trip",
            ),
            pytest.param(
                _object(
                    {
                        "count": {"type": "integer", "default": 3},
                        "tag": {"type": "string", "default": "none"},
                    }
                ),
                {},
                {"count": 3, "tag": "none"},
                id="defaults-fill-omitted-fields",
            ),
            pytest.param(
                _object(
                    {
                        "a": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "b": {"type": ["integer", "null"]},
                    },
                    required=["a", "b"],
                ),
                {"a": None, "b": 2},
                {"a": None, "b": 2},
                id="nullable-via-anyof-and-type-list",
            ),
            pytest.param(
                _object(
                    {"value": {"oneOf": [{"type": "integer"}, {"type": "string"}]}},
                    required=["value"],
                ),
                {"value": "five"},
                {"value": "five"},
                id="oneof-union-string-member",
            ),
            pytest.param(
                _object({"level": {"enum": [1, 2, None]}}, required=["level"]),
                {"level": None},
                {"level": None},
                id="enum-with-null-member",
            ),
            pytest.param(
                _object({"stuff": {"type": "array"}}, required=["stuff"]),
                {"stuff": [1, "two", {"three": 3}, None]},
                {"stuff": [1, "two", {"three": 3}, None]},
                id="array-without-items-accepts-anything",
            ),
            pytest.param(
                _object(
                    {
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 2,
                        }
                    },
                    required=["tags"],
                ),
                {"tags": ["a", "b", "c", "d"]},
                {"tags": ["a", "b", "c", "d"]},
                id="constraint-hints-not-enforced",
            ),
            pytest.param(
                _object({"known": {"type": "string"}}, required=["known"]),
                {"known": "x", "hallucinated": "dropped"},
                {"known": "x"},
                id="extra-keys-dropped",
            ),
            pytest.param(
                {"type": "object", "properties": {}},
                {},
                {},
                id="empty-object",
            ),
        ],
    )
    def test_construct_validate_serialize(
        self,
        schema: dict[str, Any],
        payload: dict[str, Any],
        expected: dict[str, Any],
    ):
        model = json_response_schema_to_pydantic(schema)
        instance = model.model_validate(payload)
        # by_alias=True mirrors DialecticAgent.answer's serialization.
        assert json.loads(instance.model_dump_json(by_alias=True)) == expected

    @pytest.mark.parametrize(
        "schema,payload",
        [
            pytest.param(
                SPEC_EXAMPLE_SCHEMA,
                {"preferences": [{"food": "sushi"}], "summary": "s"},
                id="missing-required-in-array-item",
            ),
            pytest.param(
                SPEC_EXAMPLE_SCHEMA,
                {
                    "preferences": [{"food": "sushi", "sentiment": "adores"}],
                    "summary": "s",
                },
                id="invalid-enum-value",
            ),
            pytest.param(
                SPEC_EXAMPLE_SCHEMA,
                {"preferences": [{"food": "sushi", "sentiment": "likes"}]},
                id="missing-required-top-level",
            ),
            pytest.param(
                _object(
                    {"user": _object({"name": {"type": "string"}}, required=["name"])},
                    required=["user"],
                ),
                {"user": {}},
                id="missing-required-nested",
            ),
            pytest.param(
                _object({"a": {"type": "string"}}, required=["a"]),
                {"a": None},
                id="null-for-non-nullable",
            ),
            pytest.param(
                _object({"n": {"type": "integer"}}, required=["n"]),
                {"n": {"nested": "dict"}},
                id="wrong-type-for-integer",
            ),
        ],
    )
    def test_rejects_nonconforming_payloads(
        self, schema: dict[str, Any], payload: dict[str, Any]
    ):
        model = json_response_schema_to_pydantic(schema)
        with pytest.raises(ValidationError):
            model.model_validate(payload)
