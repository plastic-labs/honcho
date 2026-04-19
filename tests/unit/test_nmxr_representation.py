"""Unit tests for the NMXR grounded-representation Pydantic schema.

Covers:
1. Schema construction with valid facts.
2. Pydantic validation of required fields.
3. Negative source_message_idx values are permitted at this layer
   (grounding verifiers enforce bounds).
4. ``to_prompt_representation()`` strips ``source_*`` provenance fields.
5. JSON roundtrip fidelity via ``model_dump()`` / ``model_validate()``.
6. ``model_json_schema()`` returns a valid JSON-schema dict -- this is
   what minccino's ``nmxr_teacher_trace`` stage hands to OpenRouter as
   ``response_format``.
"""

from typing import Any, cast

import pytest
from pydantic import ValidationError

from src.utils.representation import (
    ExplicitObservationBase,
    NmxrExplicitFact,
    NmxrGroundedRepresentation,
    PromptRepresentation,
)


def _sample_facts() -> list[NmxrExplicitFact]:
    """Build a canonical three-fact sample for tests."""
    return [
        NmxrExplicitFact(
            fact="Alice lives in Berlin.",
            source_message_idx=0,
            source_span="I live in Berlin",
        ),
        NmxrExplicitFact(
            fact="Alice is a machine-learning researcher.",
            source_message_idx=2,
            source_span="I do ML research",
        ),
        NmxrExplicitFact(
            fact="Alice owns a cat named Nibbler.",
            source_message_idx=4,
            source_span="my cat Nibbler",
        ),
    ]


class TestNmxrGroundedRepresentationConstruction:
    """Construction and Pydantic validation tests."""

    def test_build_three_facts(self) -> None:
        """A well-formed representation with three facts validates."""
        rep = NmxrGroundedRepresentation(
            explicit=_sample_facts(),
            subject_peer="alice",
        )
        assert len(rep.explicit) == 3
        assert rep.subject_peer == "alice"
        assert rep.explicit[0].fact == "Alice lives in Berlin."
        assert rep.explicit[0].source_message_idx == 0
        assert rep.explicit[0].source_span == "I live in Berlin"

    def test_empty_explicit_allowed(self) -> None:
        """An empty ``explicit`` list is valid (default_factory=list)."""
        rep = NmxrGroundedRepresentation(subject_peer="bob")
        assert rep.explicit == []
        assert rep.subject_peer == "bob"

    def test_missing_source_span_raises(self) -> None:
        """Omitting ``source_span`` on a fact triggers a ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            NmxrExplicitFact.model_validate(
                {
                    "fact": "Alice lives in Berlin.",
                    "source_message_idx": 0,
                    # source_span deliberately missing
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("source_span",) for err in errors)

    def test_missing_fact_raises(self) -> None:
        """Omitting ``fact`` triggers a ValidationError."""
        with pytest.raises(ValidationError):
            NmxrExplicitFact.model_validate(
                {
                    "source_message_idx": 0,
                    "source_span": "hello",
                }
            )

    def test_missing_source_message_idx_raises(self) -> None:
        """Omitting ``source_message_idx`` triggers a ValidationError."""
        with pytest.raises(ValidationError):
            NmxrExplicitFact.model_validate(
                {
                    "fact": "Alice lives in Berlin.",
                    "source_span": "hello",
                }
            )

    def test_missing_subject_peer_raises(self) -> None:
        """``subject_peer`` is required on the outer model."""
        with pytest.raises(ValidationError):
            NmxrGroundedRepresentation.model_validate({"explicit": []})

    def test_negative_source_message_idx_allowed(self) -> None:
        """Negative indices are *allowed* at the Pydantic layer.

        The grounding verifier (downstream, in minccino) is responsible
        for bound-checking; Honcho's mirror must not reject them.
        """
        fact = NmxrExplicitFact(
            fact="Alice was described in an earlier context.",
            source_message_idx=-1,
            source_span="earlier",
        )
        assert fact.source_message_idx == -1

        rep = NmxrGroundedRepresentation(
            explicit=[fact],
            subject_peer="alice",
        )
        assert rep.explicit[0].source_message_idx == -1


class TestToPromptRepresentation:
    """``to_prompt_representation()`` forward-compatibility adapter."""

    def test_strips_source_fields(self) -> None:
        """The projection drops all ``source_*`` provenance fields."""
        rep = NmxrGroundedRepresentation(
            explicit=_sample_facts(),
            subject_peer="alice",
        )
        prompt_rep = rep.to_prompt_representation()

        assert isinstance(prompt_rep, PromptRepresentation)
        assert len(prompt_rep.explicit) == 3
        for observation in prompt_rep.explicit:
            assert isinstance(observation, ExplicitObservationBase)
            # Dumped model must not leak source_* keys.
            dumped = observation.model_dump()
            assert "source_message_idx" not in dumped
            assert "source_span" not in dumped
            assert "content" in dumped

    def test_fact_text_preserved_in_content(self) -> None:
        """Each fact's text is copied verbatim into ``content``."""
        facts = _sample_facts()
        rep = NmxrGroundedRepresentation(explicit=facts, subject_peer="alice")
        prompt_rep = rep.to_prompt_representation()

        assert [obs.content for obs in prompt_rep.explicit] == [
            f.fact for f in facts
        ]

    def test_empty_representation_projects_to_empty(self) -> None:
        """An empty grounded rep projects to an empty prompt rep."""
        rep = NmxrGroundedRepresentation(subject_peer="alice")
        prompt_rep = rep.to_prompt_representation()
        assert isinstance(prompt_rep, PromptRepresentation)
        assert prompt_rep.explicit == []


class TestJsonRoundtrip:
    """JSON serialization / deserialization fidelity."""

    def test_model_dump_validate_roundtrip(self) -> None:
        """``model_dump`` -> ``model_validate`` yields an equal object."""
        original = NmxrGroundedRepresentation(
            explicit=_sample_facts(),
            subject_peer="alice",
        )
        dumped = original.model_dump()
        restored = NmxrGroundedRepresentation.model_validate(dumped)
        assert restored == original

    def test_json_string_roundtrip(self) -> None:
        """``model_dump_json`` -> ``model_validate_json`` yields equality."""
        original = NmxrGroundedRepresentation(
            explicit=_sample_facts(),
            subject_peer="alice",
        )
        raw = original.model_dump_json()
        restored = NmxrGroundedRepresentation.model_validate_json(raw)
        assert restored == original


class TestJsonSchema:
    """``model_json_schema()`` is consumed by OpenRouter via minccino."""

    def test_schema_is_dict_with_expected_keys(self) -> None:
        """The generated JSON schema is a well-formed dict."""
        schema: dict[str, Any] = NmxrGroundedRepresentation.model_json_schema()
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert "properties" in schema
        properties = schema["properties"]
        assert "explicit" in properties
        assert "subject_peer" in properties
        # subject_peer is required; explicit has a default_factory so is not.
        assert "required" in schema
        assert "subject_peer" in schema["required"]

    def test_schema_references_explicit_fact(self) -> None:
        """The schema exposes ``NmxrExplicitFact`` in its ``$defs``."""
        schema: dict[str, Any] = NmxrGroundedRepresentation.model_json_schema()
        defs = cast(
            dict[str, Any], schema.get("$defs") or schema.get("definitions") or {}
        )
        assert "NmxrExplicitFact" in defs
        fact_schema = cast(dict[str, Any], defs["NmxrExplicitFact"])
        assert fact_schema["type"] == "object"
        required_raw = cast(list[str], fact_schema.get("required", []))
        required: set[str] = set(required_raw)
        assert {"fact", "source_message_idx", "source_span"} <= required
