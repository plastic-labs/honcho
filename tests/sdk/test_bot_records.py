"""filter_bot_records: bot records are kept only when a person interacted."""

import datetime
from typing import Any
from zoneinfo import ZoneInfo

from sdks.python.src.honcho.api_types import IngestRecord
from sdks.python.src.honcho.bot_records import filter_bot_records

ALEX = {"id": "1", "name": "alex"}
SAM = {"id": "2", "name": "sam"}
RABBIT = {"id": "9", "name": "coderabbitai", "bot": True}
CI = {"name": "ci[bot]", "bot": True}


def _record(
    source_id: str,
    minute: int,
    author: dict[str, Any] | None,
    *,
    container: str = "repo",
    reply_to: str | None = None,
    thread: str | None = None,
    participants: list[dict[str, Any]] | None = None,
    instance: str = "acme",
    content: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if reply_to:
        metadata["reply_to"] = reply_to
    if thread:
        metadata["thread"] = thread
    return {
        "source": "github",
        "source_instance": instance,
        "source_id": source_id,
        "created_at": f"2026-11-04T10:{minute:02d}:00Z",
        "container": {"id": container},
        "author": author,
        "participants": participants or [],
        "content": content or source_id,
        "metadata": metadata,
    }


def _ids(records: list[Any]) -> list[str]:
    return [r["source_id"] if isinstance(r, dict) else r.source_id for r in records]


def test_unanswered_bot_records_are_held_back() -> None:
    records = [
        _record("pr", 0, ALEX),
        _record("walkthrough", 1, RABBIT),
        _record("ci", 2, CI),
        _record("lgtm", 3, SAM),
    ]

    kept, held_back = filter_bot_records(records)

    assert _ids(kept) == ["pr", "lgtm"]
    assert held_back == 2


def test_a_reply_chain_with_a_person_keeps_its_bot_records() -> None:
    records = [
        # A review thread: replies point at the thread's first comment.
        _record("nit", 0, RABBIT),
        _record("why", 1, ALEX, reply_to="nit"),
        _record("because", 2, RABBIT, reply_to="nit"),
        # A chain: each reply points at the one before it.
        _record("question", 3, SAM),
        _record("answer", 4, RABBIT, reply_to="question"),
        _record("thanks", 5, SAM, reply_to="answer"),
        # A bot replying only to a bot.
        _record("alert", 6, CI),
        _record("ack", 7, RABBIT, reply_to="alert"),
    ]

    kept, held_back = filter_bot_records(records)

    assert _ids(kept) == ["nit", "why", "because", "question", "answer", "thanks"]
    assert held_back == 2


def test_a_mention_keeps_the_bots_post_before_it_and_its_answer() -> None:
    records = [
        _record("old", 0, RABBIT, thread="pr:1"),
        _record("summary", 1, RABBIT, thread="pr:1"),
        _record(
            "ask",
            2,
            ALEX,
            thread="pr:1",
            participants=[RABBIT],
            content="@coderabbitai what about the cache?",
        ),
        _record("reply", 3, RABBIT, thread="pr:1"),
        _record("later", 4, RABBIT, thread="pr:1"),
        # Same repository, another pull request: a different conversation.
        _record("elsewhere", 1, RABBIT, thread="pr:2"),
    ]

    kept, held_back = filter_bot_records(records)

    assert _ids(kept) == ["summary", "ask", "reply"]
    assert held_back == 3


def test_a_conversation_is_the_container_without_a_thread() -> None:
    records = [
        _record("stats", 0, CI, container="stats"),
        _record(
            "ping",
            1,
            ALEX,
            container="general",
            participants=[CI],
            content="@ci run it again",
        ),
        _record("pong", 2, CI, container="general"),
    ]

    kept, _ = filter_bot_records(records)

    assert _ids(kept) == ["ping", "pong"]


def test_a_conversation_is_the_session_ingest_maps_a_record_to() -> None:
    mention = "@ci run it again"
    records = [
        # An explicit session_id is its own conversation, whatever the container.
        {**_record("stats", 0, CI), "session_id": "ops"},
        {
            **_record("ping", 1, ALEX, participants=[CI], content=mention),
            "session_id": "ops",
        },
        _record("pong", 2, CI),
        # Without a container, each record is a session of its own.
        {**_record("alone", 3, CI), "container": None},
        {
            **_record("asks", 4, ALEX, participants=[CI], content=mention),
            "container": None,
        },
        {**_record("after", 5, CI), "container": None},
    ]

    kept, held_back = filter_bot_records(records)

    assert _ids(kept) == ["stats", "ping", "asks"]
    assert held_back == 3


def test_another_source_instance_never_interacts() -> None:
    records = [
        _record("alert", 0, RABBIT, instance="acme"),
        _record("reply", 1, ALEX, instance="other", reply_to="alert"),
    ]

    kept, _ = filter_bot_records(records)

    assert _ids(kept) == ["reply"]


def test_all_none_and_records_without_an_author() -> None:
    records = [
        _record("pr", 0, ALEX),
        _record("walkthrough", 1, RABBIT),
        _record("anonymous", 2, None),
    ]

    assert _ids(filter_bot_records(records, keep="all")[0]) == [
        "pr",
        "walkthrough",
        "anonymous",
    ]
    assert filter_bot_records(records, keep="none") == (
        [records[0], records[2]],
        1,
    )
    # A record without an author isn't a person: it keeps no bot record.
    records.append(_record("reply", 3, None, reply_to="walkthrough"))
    assert "walkthrough" not in _ids(filter_bot_records(records)[0])


def test_models_and_dicts_alike_in_order() -> None:
    records = [
        IngestRecord.model_validate(_record("nit", 0, RABBIT)),
        IngestRecord.model_validate(_record("why", 1, ALEX, reply_to="nit")),
        IngestRecord.model_validate(_record("noise", 2, CI)),
    ]

    kept, held_back = filter_bot_records(records)

    assert _ids(kept) == ["nit", "why"]
    assert all(isinstance(r, IngestRecord) for r in kept)
    assert held_back == 1


def test_the_same_thread_in_two_repositories_is_two_conversations() -> None:
    records = [
        _record("a-post", 0, RABBIT, container="repo-a", thread="pr:1"),
        _record("b-post", 1, RABBIT, container="repo-b", thread="pr:1"),
        _record(
            "ask",
            2,
            ALEX,
            container="repo-a",
            thread="pr:1",
            participants=[RABBIT],
            content="@coderabbitai again?",
        ),
    ]

    kept, _ = filter_bot_records(records)

    assert _ids(kept) == ["a-post", "ask"]


def test_long_chains_and_cycles_stay_one_conversation() -> None:
    chain = [_record("start", 0, ALEX)]
    for n in range(1, 1500):
        chain.append(
            _record(f"bot-{n}", n % 60, RABBIT, reply_to=chain[-1]["source_id"])
        )
    kept, held_back = filter_bot_records(chain)
    assert held_back == 0
    assert len(kept) == 1500

    cycle = [
        _record("x", 0, RABBIT, reply_to="y"),
        _record("y", 1, ALEX, reply_to="x"),
    ]
    assert _ids(filter_bot_records(cycle)[0]) == ["x", "y"]


def test_a_bot_named_without_an_id_is_the_same_bot_named_with_one() -> None:
    records = [
        _record("bump", 0, {"name": "dependabot[bot]", "bot": True}, thread="pr:7"),
        _record(
            "ask",
            1,
            ALEX,
            thread="pr:7",
            participants=[{"id": "42", "name": "dependabot", "bot": True}],
            content="@dependabot rebase",
        ),
        _record(
            "reply", 2, {"id": "42", "name": "dependabot", "bot": True}, thread="pr:7"
        ),
    ]

    kept, held_back = filter_bot_records(records)

    assert _ids(kept) == ["bump", "ask", "reply"]
    assert held_back == 0


def test_metadata_hints_that_arent_ids_are_ignored() -> None:
    records = [
        _record("pr", 0, ALEX),
        {
            **_record("odd", 1, RABBIT),
            "metadata": {"reply_to": ["pr"], "thread": {"id": 1}},
        },
    ]

    kept, held_back = filter_bot_records(records)

    assert _ids(kept) == ["pr"]
    assert held_back == 1


def test_a_dict_is_read_the_way_ingest_validates_it() -> None:
    # "false" validates as a person; a lowercase "t"/"z" timestamp is valid.
    person = {**_record("hi", 0, {"id": "1", "name": "alex", "bot": "false"})}
    person["created_at"] = "2026-11-04t10:00:00z"

    kept, held_back = filter_bot_records([person], keep="none")

    assert _ids(kept) == ["hi"]
    assert held_back == 0


def test_aware_times_compare_in_utc_across_a_dst_change() -> None:
    zone = ZoneInfo("America/New_York")
    post = IngestRecord.model_validate(_record("post", 0, RABBIT))
    ask = IngestRecord.model_validate(
        _record("ask", 0, ALEX, participants=[RABBIT], content="@coderabbitai?")
    )
    answer = IngestRecord.model_validate(_record("answer", 0, RABBIT))
    # One zone, the hour that repeats: 01:50 EDT (the first 01:50) is before
    # 01:10 EST (fold=1, the second pass), which is before 01:40 EST. Compared
    # by wall clock, as Python compares datetimes sharing a zone, it isn't.
    post.created_at = datetime.datetime(2026, 11, 1, 1, 50, tzinfo=zone)
    ask.created_at = datetime.datetime(2026, 11, 1, 1, 10, fold=1, tzinfo=zone)
    answer.created_at = datetime.datetime(2026, 11, 1, 1, 40, fold=1, tzinfo=zone)

    kept, _ = filter_bot_records([post, ask, answer])

    assert _ids(kept) == ["post", "ask", "answer"]


def test_replies_to_a_missing_root_are_one_thread() -> None:
    # The thread's first comment was dropped as empty; both replies point at it.
    records = [
        _record("bot-reply", 0, RABBIT, reply_to="root"),
        _record("person-reply", 1, ALEX, reply_to="root"),
    ]

    assert _ids(filter_bot_records(records)[0]) == ["bot-reply", "person-reply"]


def test_a_reply_alone_is_not_a_mention() -> None:
    # The replied-to author is a participant, but the text doesn't mention it.
    records = [
        _record("a", 0, RABBIT, thread="pr:1"),
        _record("b", 1, RABBIT, thread="pr:1"),
        _record("fixed", 2, ALEX, thread="pr:1", reply_to="a", participants=[RABBIT]),
        _record("c", 3, RABBIT, thread="pr:1"),
    ]

    kept, held_back = filter_bot_records(records)

    assert _ids(kept) == ["a", "fixed"]
    assert held_back == 2


def test_name_matching_doesnt_depend_on_order() -> None:
    same_name_a = {"id": "1", "name": "helper", "bot": True}
    same_name_b = {"id": "2", "name": "helper", "bot": True}
    b_post = _record("b-post", 0, same_name_b, container="repo-b")
    ask = _record(
        "ask",
        1,
        ALEX,
        container="repo-b",
        participants=[{"name": "helper", "bot": True}],
        content="@helper hi",
    )
    a_post = _record("a-post", 0, same_name_a, container="repo-a")

    for records in ([a_post, b_post, ask], [b_post, a_post, ask]):
        assert "b-post" in _ids(filter_bot_records(records)[0])


def test_a_mention_names_the_whole_bot_name() -> None:
    def ask(text: str) -> list[dict[str, Any]]:
        return [
            _record("root", 0, CI),
            _record("other", 1, CI),
            _record("reply", 2, ALEX, reply_to="root", participants=[CI], content=text),
            _record("later", 3, CI),
        ]

    # `@city` isn't `@ci`: only the reply chain counts.
    assert _ids(filter_bot_records(ask("@city please look"))[0]) == ["root", "reply"]
    for text in ("@ci please look", "@CI[bot] again", "thanks @ci."):
        assert _ids(filter_bot_records(ask(text))[0]) == [
            "root",
            "other",
            "reply",
            "later",
        ]
