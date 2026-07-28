import datetime
import logging

import pytest

from src import models
from src.deriver.deriver import (
    _format_messages_for_prompt,  # pyright: ignore[reportPrivateUsage]
)
from src.utils.representation import (
    DeductiveObservation,
    ExplicitObservation,
    ExplicitObservationBase,
    PromptRepresentation,
    Representation,
)


def test_prompt_representation_schema_orders_citations_before_content() -> None:
    schema = PromptRepresentation.model_json_schema()
    explicit_properties = schema["$defs"]["ExplicitObservationBase"]["properties"]

    assert list(explicit_properties) == ["source_indices", "content"]


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
        batch_message_ids=[1],
        session_name="s",
        created_at=timestamp,
    )
    assert isinstance(rep, Representation)
    assert [e.content for e in rep.explicit] == ["A"]
    # Deductive observations from PromptRepresentation are not converted
    # (they would be created directly by the Dreamer via the create_observations tool)
    assert len(rep.deductive) == 0
    assert rep.explicit[0].created_at == timestamp


def test_mixed_peer_source_indices_resolve_against_prompt_order(
    caplog: pytest.LogCaptureFixture,
) -> None:
    created_at = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    messages = [
        models.Message(id=10, peer_name="bob", content="Which option?"),
        models.Message(id=20, peer_name="alice", content="The first one"),
        models.Message(id=30, peer_name="bob", content="Got it"),
    ]
    for message in messages:
        message.created_at = created_at

    formatted_messages, batch_message_ids = _format_messages_for_prompt(messages)
    prompt_representation = PromptRepresentation(
        explicit=[
            ExplicitObservationBase(
                content="Alice chose the first option",
                source_indices=[0, 1, 3],
            )
        ]
    )

    with caplog.at_level(logging.WARNING, logger="src.utils.representation"):
        representation = Representation.from_prompt_representation(
            prompt_representation,
            message_ids=[20],
            batch_message_ids=batch_message_ids,
            session_name="s",
            created_at=created_at,
        )

    observation = representation.explicit[0]
    assert observation.source_indices == [0, 1]
    assert observation.batch_message_ids == [10, 20, 30]
    assert [
        (line[:3], message_id, line.split(": ", 1)[1])
        for message_id, line in zip(
            batch_message_ids, formatted_messages.splitlines(), strict=True
        )
    ] == [
        ("[0]", 10, "Which option?"),
        ("[1]", 20, "The first one"),
        ("[2]", 30, "Got it"),
    ]
    assert [
        observation.batch_message_ids[index] for index in observation.source_indices
    ] == [10, 20]
    assert "Dropping out-of-range source_indices [3]" in caplog.text
