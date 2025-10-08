import datetime

from src.utils.representation import (
    DeductiveObservation,
    DeductiveObservationBase,
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
        message_ids=[(1, 1)],
        session_name="s",
    )
    exp_shared_2 = ExplicitObservation(
        content="B",
        created_at=shared_time,
        message_ids=[(1, 1)],
        session_name="s",
    )
    rep1 = Representation(explicit=[exp_shared_1], deductive=[])
    rep2 = Representation(
        explicit=[
            ExplicitObservation(
                content="A",
                created_at=shared_time,
                message_ids=[(1, 1)],
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
        message_ids=[(1, 1)],
        session_name="s",
    )
    d = DeductiveObservation(
        created_at=now,
        message_ids=[(1, 1)],
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
    assert "**Conclusion**: owns a pet" in md


def test_prompt_representation_conversion():
    """PromptRepresentation.to_representation maps strings to observation objects."""
    pr = PromptRepresentation(
        explicit=[ExplicitObservationBase(content="A")],
        deductive=[DeductiveObservationBase(conclusion="C", premises=["P1"])],
    )
    timestamp = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    rep = Representation.from_prompt_representation(
        pr,
        message_ids=(1, 1),
        session_name="s",
        created_at=timestamp,
    )
    assert isinstance(rep, Representation)
    assert [e.content for e in rep.explicit] == ["A"]
    assert rep.deductive[0].conclusion == "C"
    assert rep.deductive[0].premises == ["P1"]
    assert rep.explicit[0].created_at == timestamp
    assert rep.deductive[0].created_at == timestamp
