"""Unit tests for src/utils/schema_conversion.py."""

import re

import pytest
from pydantic import BaseModel, ValidationError

from src.utils.schema_conversion import json_response_schema_to_pydantic


def _object(properties: dict, **extra) -> dict:
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
        model = json_response_schema_to_pydantic(_object({"nickname": {"type": "string"}}))
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
                {
                    "maybe": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    }
                },
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
            _object(
                {"name": {"type": ["string", "null"]}}, required=["name"]
            )
        )
        assert model.model_validate({"name": None}).name is None  # pyright: ignore


class TestRejections:
    @pytest.mark.parametrize(
        "construct,schema",
        [
            ("$ref", _object({"a": {"$ref": "#/x"}})),
            ("$defs", {"type": "object", "properties": {}, "$defs": {}}),
            ("allOf", _object({"a": {"allOf": [{"type": "string"}]}})),
            ("not", _object({"a": {"not": {"type": "string"}}})),
            ("if", _object({"a": {"if": {"type": "string"}}})),
            (
                "patternProperties",
                _object({"a": {"type": "object", "patternProperties": {}}}),
            ),
        ],
    )
    def test_unsupported_constructs(self, construct: str, schema: dict):
        with pytest.raises(ValueError, match=re.escape(construct)):
            json_response_schema_to_pydantic(schema)

    def test_error_message_includes_path(self):
        with pytest.raises(
            ValueError, match=r"'\$ref' at properties\.address"
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
        schema: dict = {"type": "string"}
        for _ in range(25):
            schema = _object({"inner": schema})
        with pytest.raises(ValueError, match="maximum depth"):
            json_response_schema_to_pydantic(schema)

    def test_node_limit(self):
        schema = _object(
            {f"field_{i}": {"type": "string"} for i in range(600)}
        )
        with pytest.raises(ValueError, match="maximum of 500 nodes"):
            json_response_schema_to_pydantic(schema)


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
            _object(
                {"food": {"type": "string", "description": "A food item"}}
            )
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
        schema: dict = {"type": "string"}
        for _ in range(5):
            schema = _object({"inner": schema})
        with pytest.raises(ValueError, match="maximum depth of 3"):
            json_response_schema_to_pydantic(schema, max_depth=3)

    def test_custom_max_nodes(self):
        schema = _object({f"f{i}": {"type": "string"} for i in range(20)})
        with pytest.raises(ValueError, match="maximum of 10 nodes"):
            json_response_schema_to_pydantic(schema, max_nodes=10)
