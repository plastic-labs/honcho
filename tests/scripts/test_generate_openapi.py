"""Tests for the generate_openapi script."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.generate_openapi import (
    SpecGenerationError,
    _hoist_defs,  # pyright: ignore[reportPrivateUsage]
    describe_drift,
    render,
)


def _spec_with_defs(defs: dict[str, Any], ref: str) -> dict[str, Any]:
    """A minimal spec whose only response schema carries a local ``$defs``."""
    return {
        "openapi": "3.1.0",
        "paths": {
            "/chat": {
                "post": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$defs": defs,
                                        "properties": {
                                            "evidence": {"$ref": ref},
                                        },
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {"schemas": {}},
    }


class TestHoistDefs:
    def test_hoists_and_rewrites_the_ref(self):
        """Mintlify cannot resolve `#/$defs/`, so nothing may be left pointing there."""
        spec = _spec_with_defs(
            {"Evidence": {"type": "object", "title": "Evidence"}},
            "#/$defs/Evidence",
        )

        assert _hoist_defs(spec) == ["Evidence"]

        assert spec["components"]["schemas"]["Evidence"] == {
            "type": "object",
            "title": "Evidence",
        }
        schema = spec["paths"]["/chat"]["post"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        assert "$defs" not in schema
        assert schema["properties"]["evidence"]["$ref"] == (
            "#/components/schemas/Evidence"
        )

    def test_hoists_defs_nested_inside_a_def(self):
        """`model_json_schema()` nests refs between `$defs` entries, not just at the top."""
        spec = _spec_with_defs(
            {
                "Evidence": {
                    "type": "object",
                    "properties": {"tool_calls": {"$ref": "#/$defs/EvidenceToolCall"}},
                    "$defs": {"EvidenceToolCall": {"type": "object"}},
                }
            },
            "#/$defs/Evidence",
        )

        assert _hoist_defs(spec) == ["Evidence", "EvidenceToolCall"]

        schemas = spec["components"]["schemas"]
        assert set(schemas) == {"Evidence", "EvidenceToolCall"}
        assert schemas["Evidence"]["properties"]["tool_calls"]["$ref"] == (
            "#/components/schemas/EvidenceToolCall"
        )
        assert "$defs" not in schemas["Evidence"]

    def test_identical_duplicate_is_not_a_conflict(self):
        """Both chat routes emit the same evidence schemas; that must stay silent."""
        shared = {"type": "object", "title": "Evidence"}
        spec = _spec_with_defs({"Evidence": shared}, "#/$defs/Evidence")
        spec["paths"]["/other"] = {
            "post": {
                "responses": {
                    "200": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$defs": {"Evidence": dict(shared)},
                                    "properties": {"e": {"$ref": "#/$defs/Evidence"}},
                                }
                            }
                        }
                    }
                }
            }
        }

        assert _hoist_defs(spec) == ["Evidence"]
        assert spec["components"]["schemas"]["Evidence"] == shared

    def test_conflicting_duplicate_raises(self):
        """Silently keeping one of two same-named schemas would publish a wrong shape."""
        spec = _spec_with_defs({"Evidence": {"type": "object"}}, "#/$defs/Evidence")
        spec["paths"]["/other"] = {
            "post": {
                "responses": {
                    "200": {
                        "content": {
                            "application/json": {
                                "schema": {"$defs": {"Evidence": {"type": "string"}}}
                            }
                        }
                    }
                }
            }
        }

        with pytest.raises(SpecGenerationError, match="both named 'Evidence'"):
            _hoist_defs(spec)

    def test_collision_with_an_existing_component_raises(self):
        spec = _spec_with_defs({"Evidence": {"type": "object"}}, "#/$defs/Evidence")
        spec["components"]["schemas"]["Evidence"] = {"type": "string"}

        with pytest.raises(SpecGenerationError, match="would overwrite"):
            _hoist_defs(spec)

    def test_leaves_a_spec_without_defs_alone(self):
        spec = {
            "openapi": "3.1.0",
            "components": {"schemas": {"Peer": {"type": "object"}}},
            "paths": {"/p": {"get": {"responses": {"200": {"description": "ok"}}}}},
        }
        original = render(spec)

        assert _hoist_defs(spec) == []
        assert render(spec) == original


class TestRender:
    def test_ends_with_exactly_one_newline(self):
        """The drift check compares bytes, so the trailing newline is part of the contract."""
        rendered = render({"openapi": "3.1.0"})
        assert rendered.endswith("}\n")
        assert not rendered.endswith("\n\n")

    def test_is_stable_across_calls(self):
        spec = {"b": 1, "a": {"d": [1, 2], "c": None}}
        assert render(spec) == render(spec)


class TestDescribeDrift:
    def test_no_drift_reports_nothing(self):
        spec: dict[str, Any] = {"paths": {"/a": {"get": {}}}}
        assert describe_drift(spec, dict(spec)) == []

    def test_names_a_path_the_app_gained(self):
        drift = describe_drift({"paths": {}}, {"paths": {"/new": {}}})
        assert drift == ["/paths//new: missing from committed spec"]

    def test_names_a_path_the_app_dropped(self):
        drift = describe_drift({"paths": {"/gone": {}}}, {"paths": {}})
        assert drift == ["/paths//gone: no longer produced by the app"]

    def test_reports_a_changed_value_and_a_length_change(self):
        committed = {"info": {"version": "3.1.1"}, "tags": ["a"]}
        generated = {"info": {"version": "3.2.0"}, "tags": ["a", "b"]}

        assert set(describe_drift(committed, generated)) == {
            "/info/version: differs",
            "/tags: 1 entries in committed, 2 generated",
        }
