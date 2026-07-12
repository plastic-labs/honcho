"""
Tests for PromptRepresentation input normalization.

Providers without ``json_schema`` structured-output support (e.g. DeepSeek via
LiteLLM, some vLLM configs) run in ``json_object`` mode and infer the JSON shape
from the prompt/schema text. They frequently return the ``explicit`` field as a
list of bare strings instead of ``{"content": ...}`` objects. These tests pin
that the bare-string shape is coerced rather than silently dropped (#893).
"""

import pytest
from pydantic import ValidationError

from src.utils.representation import ExplicitObservationBase, PromptRepresentation


def test_explicit_bare_strings_are_coerced_to_objects() -> None:
    rep = PromptRepresentation.model_validate(
        {"explicit": ["alice is 25 years old", "alice has a dog"]}
    )

    assert rep.explicit == [
        ExplicitObservationBase(content="alice is 25 years old"),
        ExplicitObservationBase(content="alice has a dog"),
    ]


def test_explicit_objects_still_accepted() -> None:
    rep = PromptRepresentation.model_validate(
        {"explicit": [{"content": "alice is 25 years old"}]}
    )

    assert rep.explicit == [ExplicitObservationBase(content="alice is 25 years old")]


def test_explicit_mixed_shapes_are_normalized() -> None:
    rep = PromptRepresentation.model_validate(
        {"explicit": ["bare fact", {"content": "object fact"}]}
    )

    assert [o.content for o in rep.explicit] == ["bare fact", "object fact"]


def test_explicit_none_becomes_empty_list() -> None:
    rep = PromptRepresentation.model_validate({"explicit": None})

    assert rep.explicit == []


def test_explicit_non_string_scalar_still_rejected() -> None:
    # Coercion is limited to strings; a stray int is a genuine shape error and
    # should not be silently turned into content.
    with pytest.raises(ValidationError):
        PromptRepresentation.model_validate({"explicit": [123]})
