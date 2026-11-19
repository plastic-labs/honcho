"""Discord messages to ingest records."""

from typing import Any

from honcho_discord.mapping import Container, account, display_name, to_record

CHANNEL = Container(id="200", name="dev", parent_id="100", type=0)
THREAD = Container(id="300", name="sso-rollout", parent_id="200", type=11)


def user(uid: str, username: str, **extra: Any) -> dict[str, Any]:
    return {"id": uid, "username": username, "global_name": None, **extra}


def message(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "9001",
        "type": 0,
        "channel_id": "200",
        "author": user("1", "jess", global_name="Jess"),
        "content": "Northstar needs SSO before <@2> ships it.",
        "timestamp": "2026-11-04T20:47:00.000000+00:00",
        "edited_timestamp": None,
        "mentions": [user("2", "aakash")],
        "attachments": [],
        "embeds": [],
    }
    base.update(overrides)
    return base


def test_a_message_keeps_what_the_source_sent() -> None:
    record = to_record(message(), CHANNEL, "77", members={})
    assert record == {
        "source": "discord",
        "source_instance": "77",
        "source_id": "9001",
        "created_at": "2026-11-04T20:47:00.000000+00:00",
        "revised_at": None,
        "container": {"id": "200", "name": "dev", "parent_id": "100"},
        "author": {"id": "1", "name": "Jess", "handle": "jess"},
        "participants": [{"id": "2", "name": "aakash", "handle": "aakash"}],
        # Mentions stay as written; resolving them is Honcho's job.
        "content": "Northstar needs SSO before <@2> ships it.",
        "metadata": {"discord_message_type": "default"},
    }


def test_a_thread_names_its_channel_as_parent() -> None:
    record = to_record(message(channel_id="300"), THREAD, "77", members={})
    assert record is not None
    assert record["container"] == {
        "id": "300",
        "name": "sso-rollout",
        "parent_id": "200",
    }


def test_names_prefer_the_server_nickname() -> None:
    members = {"1": {"nick": "JessD", "user": {"id": "1"}}}
    record = to_record(message(), CHANNEL, "77", members)
    assert record is not None
    assert record["author"] == {
        "id": "1",
        "name": "JessD",
        "handle": "jess",
        "aliases": ["Jess"],
    }
    assert display_name(user("5", "plain")) == "plain"
    assert display_name(user("5", "plain", global_name="  ")) == "plain"


def test_a_reply_names_who_it_answers_and_links_the_message() -> None:
    replied = message(id="8000", author=user("3", "maya"), mentions=[])
    record = to_record(
        message(
            type=19,
            mentions=[],
            message_reference={"message_id": "8000"},
            referenced_message=replied,
        ),
        CHANNEL,
        "77",
        members={},
    )
    assert record is not None
    assert record["participants"] == [{"id": "3", "name": "maya", "handle": "maya"}]
    assert record["metadata"] == {"discord_message_type": "reply", "reply_to": "8000"}


def test_the_author_is_never_their_own_participant() -> None:
    record = to_record(
        message(mentions=[user("1", "jess"), user("2", "aakash"), user("2", "aakash")]),
        CHANNEL,
        "77",
        members={},
    )
    assert record is not None
    assert [p["id"] for p in record["participants"]] == ["2"]


def test_an_edit_is_carried_as_the_revision_time() -> None:
    record = to_record(
        message(edited_timestamp="2026-11-05T09:00:00+00:00"), CHANNEL, "77", {}
    )
    assert record is not None
    assert record["revised_at"] == "2026-11-05T09:00:00+00:00"


def test_system_notices_and_empty_messages_are_not_records() -> None:
    assert to_record(message(type=7, content=""), CHANNEL, "77", {}) is None  # join
    assert to_record(message(type=6), CHANNEL, "77", {}) is None  # pin
    assert to_record(message(content="   "), CHANNEL, "77", {}) is None


def test_a_message_without_text_keeps_a_note_of_what_it_carried() -> None:
    record = to_record(
        message(
            content="",
            mentions=[],
            attachments=[{"filename": "plan.pdf"}],
            embeds=[{"title": "Deploy failed", "description": "api: exit 1"}],
            sticker_items=[{"name": "wave"}],
        ),
        CHANNEL,
        "77",
        {},
    )
    assert record is not None
    assert record["content"] == (
        "Deploy failed\napi: exit 1\n[attachment: plan.pdf]\n[sticker: wave]"
    )
    assert record["metadata"]["attachments"] == ["plan.pdf"]


def test_bots_and_webhooks_are_marked() -> None:
    bot = to_record(
        message(author=user("9", "sentry", bot=True), mentions=[]), CHANNEL, "77", {}
    )
    hook = to_record(
        message(author=user("8", "deploys"), webhook_id="8", mentions=[]),
        CHANNEL,
        "77",
        {},
    )
    assert bot is not None and bot["author"]["bot"] is True
    assert hook is not None and hook["author"]["bot"] is True
    assert "bot" not in bot["metadata"]
    assert account(user("9", "sentry", bot=True))["bot"] is True
    assert account(user("1", "jess"))["bot"] is False


def test_a_reply_to_a_webhook_message_marks_its_author_a_bot() -> None:
    replied = message(
        id="8000", author=user("8", "deploys"), webhook_id="8", mentions=[]
    )
    record = to_record(
        message(
            type=19,
            mentions=[],
            message_reference={"message_id": "8000"},
            referenced_message=replied,
        ),
        CHANNEL,
        "77",
        members={},
    )
    assert record is not None
    assert {"id": "8", "name": "deploys", "handle": "deploys", "bot": True} in record[
        "participants"
    ]


def test_a_webhook_mentioned_and_replied_to_is_still_a_bot() -> None:
    replied = message(
        id="8000", author=user("8", "deploys"), webhook_id="8", mentions=[]
    )
    record = to_record(
        message(
            type=19,
            mentions=[user("8", "deploys")],
            message_reference={"message_id": "8000"},
            referenced_message=replied,
        ),
        CHANNEL,
        "77",
        members={},
    )
    assert record is not None
    assert record["participants"] == [
        {"id": "8", "name": "deploys", "handle": "deploys", "bot": True}
    ]


def test_a_bot_named_in_a_message_is_marked() -> None:
    record = to_record(
        message(mentions=[user("9", "sentry", bot=True), user("2", "aakash")]),
        CHANNEL,
        "77",
        {},
    )
    assert record is not None
    assert record["participants"] == [
        {"id": "9", "name": "sentry", "handle": "sentry", "bot": True},
        {"id": "2", "name": "aakash", "handle": "aakash"},
    ]


def test_participants_stay_within_ingests_limit() -> None:
    many = [user(str(100 + i), f"u{i}") for i in range(105)]
    record = to_record(message(mentions=many), CHANNEL, "77", {})
    assert record is not None
    assert len(record["participants"]) == 100
    assert record["metadata"]["participants_dropped"] == 5


def test_a_name_that_is_only_nul_falls_back() -> None:
    assert display_name(user("5", "plain", global_name="\x00\x00")) == "plain"
    assert display_name({"id": "6", "username": "\x00"}) == "6"


def test_a_forward_keeps_the_forwarded_text_and_its_mentions() -> None:
    record = to_record(
        message(
            content="",
            mentions=[],
            message_reference={"type": 1, "message_id": "555"},
            message_snapshots=[
                {
                    "message": {
                        "content": "ship it <@4>",
                        "mentions": [user("4", "omar")],
                    }
                }
            ],
        ),
        CHANNEL,
        "77",
        {},
    )
    assert record is not None
    assert record["content"] == "[forwarded message]\nship it <@4>"
    assert record["author"]["id"] == "1"
    assert record["participants"] == [{"id": "4", "name": "omar", "handle": "omar"}]
    assert record["metadata"]["forwarded"] is True
    assert record["metadata"]["forwarded_from"] == "555"


def test_text_components_and_embed_fields_are_kept() -> None:
    record = to_record(
        message(
            content="",
            mentions=[],
            embeds=[{"fields": [{"name": "Status", "value": "failed"}]}],
            components=[
                {
                    "type": 17,
                    "components": [{"type": 10, "content": "Deploy summary"}],
                }
            ],
        ),
        CHANNEL,
        "77",
        {},
    )
    assert record is not None
    assert record["content"] == "Status: failed\nDeploy summary"


def test_very_long_content_is_cut_to_ingests_limit() -> None:
    record = to_record(message(content="x" * 30_000), CHANNEL, "77", {})
    assert record is not None
    assert len(record["content"]) == 25_000
    assert record["metadata"]["truncated"] is True
