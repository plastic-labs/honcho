"""Tests for the generate_openapi script."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts import generate_openapi
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


class TestCheckMode:
    """`--check` has to stay legible when the committed file is unusable.

    Both cases already exited nonzero, but via a traceback -- which tells the
    author nothing about how to fix it.
    """

    @staticmethod
    def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Point the script at a scratch spec and skip importing the app."""
        spec_path = tmp_path / "openapi.json"
        monkeypatch.setattr(generate_openapi, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(generate_openapi, "SPEC_PATH", spec_path)
        monkeypatch.setattr(
            generate_openapi, "build_spec", lambda: {"openapi": "3.1.0"}
        )
        return spec_path

    def test_missing_spec_names_the_file_and_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        _ = self._isolate(tmp_path, monkeypatch)

        assert generate_openapi.main(["--check"]) == 1

        out = capsys.readouterr().out
        assert "openapi.json is missing" in out
        assert "uv run python -m scripts.generate_openapi" in out

    def test_malformed_spec_says_so_instead_of_reporting_drift(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        spec_path = self._isolate(tmp_path, monkeypatch)
        _ = spec_path.write_text("{ not json")

        assert generate_openapi.main(["--check"]) == 1

        out = capsys.readouterr().out
        assert "is not valid JSON" in out
        # The drift report would otherwise bury the real problem under every
        # key in the spec.
        assert "out of date" not in out
        assert "missing from committed spec" not in out

    def test_matching_spec_passes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        spec_path = self._isolate(tmp_path, monkeypatch)
        _ = spec_path.write_text(render({"openapi": "3.1.0"}))

        assert generate_openapi.main(["--check"]) == 0
        assert "is up to date" in capsys.readouterr().out

    def test_drifted_spec_reports_the_difference(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        spec_path = self._isolate(tmp_path, monkeypatch)
        _ = spec_path.write_text(render({"openapi": "3.0.0"}))

        assert generate_openapi.main(["--check"]) == 1

        out = capsys.readouterr().out
        assert "is out of date" in out
        assert "/openapi: differs" in out
