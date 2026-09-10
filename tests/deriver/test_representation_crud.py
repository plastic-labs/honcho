import datetime

import pytest

from src.utils.representation import (
    DeductiveObservation,
    ExplicitObservation,
    ExplicitObservationBase,
    PromptRepresentation,
    Representation,
)


def test_representation_is_empty_and_diff():
    """is_empty and diff_representation behave per the new definitions."""
    now = datetime.datetime.now(datetime.timezone.utc)
    shared_time = now - datetime.timedelta(seconds=10)
    exp_shared_1 = ExplicitObservation(
        content="A",
        created_at=shared_time,
        message_ids=[1],
        session_name="s",
    )
    exp_shared_2 = ExplicitObservation(
        content="B",
        created_at=shared_time,
        message_ids=[1],
        session_name="s",
    )
    rep1 = Representation(explicit=[exp_shared_1], deductive=[])
    rep2 = Representation(
        explicit=[
            ExplicitObservation(
                content="A",
                created_at=shared_time,
                message_ids=[1],
                session_name="s",
            ),
            exp_shared_2,
        ]
    )

    assert not rep1.is_empty()
    assert Representation().is_empty()

    diff = rep1.diff_representation(rep2)
    assert [e.content for e in diff.explicit] == ["B"]
    assert diff.deductive == []


def test_representation_formatting_methods():
    """__str__ and format_as_markdown produce expected section headers and content."""
    now = datetime.datetime.now(datetime.timezone.utc)
    e = ExplicitObservation(
        content="has a dog",
        created_at=now,
        message_ids=[1],
        session_name="s",
    )
    d = DeductiveObservation(
        created_at=now,
        message_ids=[1],
        session_name="s",
        conclusion="owns a pet",
        premises=[e.content],
    )
    rep = Representation(explicit=[e], deductive=[d])

    s = str(rep)
    assert "EXPLICIT:" in s
    assert "DEDUCTIVE:" in s
    assert "owns a pet" in s

    md = rep.format_as_markdown()
    assert "## Explicit Observations" in md
    assert "## Deductive Observations" in md
    assert "owns a pet" in md
    assert "Premises:" in md


def test_prompt_representation_conversion():
    """PromptRepresentation.to_representation maps strings to observation objects.

    Note: In the current architecture, the Deriver only creates explicit observations.
    Deductive and inductive observations are created by the Dreamer agent.
    Therefore, from_prompt_representation only converts explicit observations.
    """
    pr = PromptRepresentation(
        explicit=[ExplicitObservationBase(content="A")],
        # Deductive observations in PromptRepresentation are ignored by from_prompt_representation
        # because the Deriver only produces explicit observations
        # deductive=[
        #     DeductiveObservationBase(
        #         conclusion="C", premises=["P1"], source_ids=["id1"]
        #     )
        # ],
    )
    timestamp = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    rep = Representation.from_prompt_representation(
        pr,
        message_ids=[1],
        session_name="s",
        created_at=timestamp,
    )
    assert isinstance(rep, Representation)
    assert [e.content for e in rep.explicit] == ["A"]
    # Deductive observations from PromptRepresentation are not converted
    # (they would be created directly by the Dreamer via the create_observations tool)
    assert len(rep.deductive) == 0
    assert rep.explicit[0].created_at == timestamp


def test_prompt_representation_normalizes_top_level_array_and_alias_keys():
    arr = PromptRepresentation.model_validate(["I live in Berlin", {"text": "I use Cubase"}])
    assert [item.content for item in arr.explicit] == ["I live in Berlin", "I use Cubase"]

    aliased = PromptRepresentation.model_validate({"observations": ["I am a musician"]})
    assert [item.content for item in aliased.explicit] == ["I am a musician"]

    nested = PromptRepresentation.model_validate({"result": {"observations": ["I work remotely"]}})
    assert [item.content for item in nested.explicit] == ["I work remotely"]


def test_prompt_representation_truncates_overlong_content():
    prompt_rep = PromptRepresentation.model_validate({"explicit": [{"content": "x" * 5000}]})
    assert len(prompt_rep.explicit) == 1
    assert len(prompt_rep.explicit[0].content) == 2000


def test_prompt_representation_rejects_unknown_top_level_dict_shape():
    with pytest.raises(Exception):
        PromptRepresentation.model_validate({"metadata": {"foo": "bar"}})
