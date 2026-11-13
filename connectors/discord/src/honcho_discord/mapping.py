"""Discord REST objects to Honcho ingest records.

Pure functions over the JSON Discord's REST API returns, so they can be tested
without a server. A record keeps what the source sent: the message text as
written (mentions stay `<@id>`), its timestamps, the channel or thread it was
posted in, its author and the people it names. Interpreting any of that
(resolving mentions, quoting replies) is left to Honcho.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

SOURCE = "discord"

# Channel types (https://discord.com/developers/docs/resources/channel).
TEXT = 0
VOICE = 2
CATEGORY = 4
ANNOUNCEMENT = 5
ANNOUNCEMENT_THREAD = 10
PUBLIC_THREAD = 11
PRIVATE_THREAD = 12
FORUM = 15
STAGE = 13
MEDIA = 16
THREAD_TYPES = frozenset({ANNOUNCEMENT_THREAD, PUBLIC_THREAD, PRIVATE_THREAD})
# Channels whose own history holds messages (voice and stage channels have a
# text chat). Forum and media channels hold messages only in their threads.
MESSAGE_CHANNEL_TYPES = frozenset({TEXT, ANNOUNCEMENT, VOICE, STAGE})
THREAD_PARENT_TYPES = frozenset({TEXT, ANNOUNCEMENT, FORUM, MEDIA})
EXPORTABLE_TYPES = MESSAGE_CHANNEL_TYPES | THREAD_PARENT_TYPES

# The ingest API's limits, so one message can't get a whole batch refused.
MAX_NAME = 512
MAX_CONTENT = 25_000
MAX_PARTICIPANTS = 100

# Message types that carry what someone wrote. The rest (joins, pins, thread
# created, boosts) are system notices and aren't records.
# https://discord.com/developers/docs/resources/message#message-object-message-types
MESSAGE_TYPES: Mapping[int, str] = {
    0: "default",
    19: "reply",
    20: "chat_input_command",
    23: "context_menu_command",
}

Json = Mapping[str, Any]


def items(value: Any) -> list[Json]:
    """A JSON list of objects, or [] when the field is missing or null."""
    return cast(list[Json], value) if isinstance(value, list) else []


def obj(value: Any) -> Json:
    """A JSON object, or {} when the field is missing or null."""
    return cast(Json, value) if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class Container:
    """A channel or thread, as a record's container.

    `parent_id` is a thread's channel, or a channel's category: the next level
    up in the server's own structure.
    """

    id: str
    name: str
    parent_id: str | None
    type: int

    @classmethod
    def from_channel(cls, channel: Json) -> "Container":
        return cls(
            id=str(channel["id"]),
            name=clean_name(channel.get("name")) or str(channel["id"]),
            parent_id=str(channel["parent_id"]) if channel.get("parent_id") else None,
            type=int(channel.get("type", TEXT)),
        )

    def as_record(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "parent_id": self.parent_id}


def clean_name(value: Any) -> str | None:
    """A usable display name: NUL-free, trimmed, at most 512 characters."""
    if not isinstance(value, str):
        return None
    name = value.replace("\x00", "").strip()
    return name[:MAX_NAME] or None


def display_name(user: Json, member: Json | None = None) -> str:
    """The name the server shows: nickname, then global display name, then username."""
    for candidate in (
        (member or {}).get("nick"),
        user.get("global_name"),
        user.get("username"),
    ):
        name = clean_name(candidate)
        if name:
            return name
    return str(user["id"])


def is_bot(message_or_user: Json) -> bool:
    """A bot account, or a message posted through a webhook."""
    if message_or_user.get("webhook_id"):
        return True
    author = obj(message_or_user.get("author", message_or_user))
    return bool(author.get("bot") or author.get("system"))


def account(
    user: Json, member: Json | None = None, *, bot: bool = False
) -> dict[str, Any]:
    """What an export keeps about one Discord account, for identity checks."""
    return {
        "id": str(user["id"]),
        "username": user.get("username"),
        "global_name": user.get("global_name"),
        "nick": (member or {}).get("nick"),
        "display_name": display_name(user, member),
        "bot": bot or bool(user.get("bot") or user.get("system")),
    }


def _person(user: Json, members: Mapping[str, Json]) -> dict[str, Any]:
    user_id = str(user["id"])
    return {"id": user_id, "name": display_name(user, members.get(user_id))}


def _component_text(components: list[Json]) -> list[str]:
    """Text Display components' text, read through containers and sections."""
    lines: list[str] = []
    for component in components:
        if component.get("type") == 10 and isinstance(component.get("content"), str):
            lines.append(component["content"])
        lines.extend(_component_text(items(component.get("components"))))
    return lines


def _body(message: Json) -> str:
    """Everything a message (or a forwarded snapshot of one) says, as text.

    The text as written, then embed titles, descriptions and fields, text
    components, and a note for each attachment, sticker and poll.
    """
    lines: list[str] = []
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        lines.append(content)
    for embed in items(message.get("embeds")):
        parts = [embed.get("title"), embed.get("description")]
        for embed_field in items(embed.get("fields")):
            parts.append(
                f"{embed_field.get('name', '')}: {embed_field.get('value', '')}"
            )
        text = "\n".join(p for p in parts if isinstance(p, str) and p.strip())
        if text:
            lines.append(text)
    lines.extend(_component_text(items(message.get("components"))))
    for attachment in items(message.get("attachments")):
        lines.append(f"[attachment: {attachment.get('filename', 'file')}]")
    for sticker in items(message.get("sticker_items")):
        lines.append(f"[sticker: {sticker.get('name', 'sticker')}]")
    question = obj(obj(message.get("poll")).get("question")).get("text")
    if isinstance(question, str) and question:
        lines.append(f"[poll: {question}]")
    return "\n".join(lines)


def _snapshots(message: Json) -> list[Json]:
    """The forwarded messages a message carries, if it is a forward."""
    return [obj(s.get("message")) for s in items(message.get("message_snapshots"))]


def to_record(
    message: Json,
    container: Container,
    guild_id: str,
    members: Mapping[str, Json],
) -> dict[str, Any] | None:
    """One Discord message as an ingest record, or None if it isn't one.

    None for system notices and for messages with nothing to keep. A forward's
    author is whoever forwarded it; the forwarded text follows a
    `[forwarded message]` line (Discord doesn't say who wrote the original).
    """
    message_type = int(message.get("type", 0))
    if message_type not in MESSAGE_TYPES:
        return None
    sections = [_body(message)]
    snapshots = _snapshots(message)
    for snapshot in snapshots:
        forwarded = _body(snapshot)
        if forwarded:
            sections.append(f"[forwarded message]\n{forwarded}")
    text = "\n".join(part for part in sections if part)
    if not text:
        return None

    metadata: dict[str, Any] = {"discord_message_type": MESSAGE_TYPES[message_type]}
    if len(text) > MAX_CONTENT:
        text = text[:MAX_CONTENT]
        metadata["truncated"] = True

    author_person = _person(obj(message["author"]), members)
    participants: list[dict[str, Any]] = []
    seen = {author_person["id"]}
    named = list(items(message.get("mentions")))
    for snapshot in snapshots:
        named.extend(items(snapshot.get("mentions")))
    replied_author = obj(obj(message.get("referenced_message")).get("author"))
    if replied_author:
        named.append(replied_author)
    for user in named:
        person = _person(user, members)
        if person["id"] not in seen:
            seen.add(person["id"])
            participants.append(person)
    if len(participants) > MAX_PARTICIPANTS:
        metadata["participants_dropped"] = len(participants) - MAX_PARTICIPANTS
        participants = participants[:MAX_PARTICIPANTS]

    reference = obj(message.get("message_reference"))
    if message_type == 19 and reference.get("message_id"):
        metadata["reply_to"] = str(reference["message_id"])
    if snapshots:
        metadata["forwarded"] = True
        if reference.get("message_id"):
            metadata["forwarded_from"] = str(reference["message_id"])
    attachments = [
        str(a["filename"])
        for a in items(message.get("attachments"))
        if a.get("filename")
    ]
    if attachments:
        metadata["attachments"] = attachments
    if is_bot(message):
        metadata["bot"] = True

    return {
        "source": SOURCE,
        "source_instance": guild_id,
        "source_id": str(message["id"]),
        "created_at": message["timestamp"],
        "revised_at": message.get("edited_timestamp"),
        "container": container.as_record(),
        "author": author_person,
        "participants": participants,
        "content": text,
        "metadata": metadata,
    }
