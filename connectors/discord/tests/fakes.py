"""A fake Discord REST API, served the way Discord pages it, for the tests."""

from collections.abc import Callable
from typing import Any, final

import httpx
from honcho_discord.client import DiscordClient

GUILD = "77"


def user(uid: str, name: str, **extra: Any) -> dict[str, Any]:
    return {"id": uid, "username": name, "global_name": None, **extra}


def message(
    mid: int, channel: str, author: dict[str, Any], content: str, **extra: Any
) -> dict[str, Any]:
    return {
        "id": str(mid),
        "type": 0,
        "channel_id": channel,
        "author": author,
        "content": content,
        "timestamp": f"2026-11-04T10:{mid % 60:02d}:00+00:00",
        "edited_timestamp": None,
        "mentions": [],
        "attachments": [],
        "embeds": [],
        **extra,
    }


JESS, MAYA, BOT = user("1", "jess"), user("3", "maya"), user("9", "ci", bot=True)


@final
class FakeDiscord:
    """Channels, threads, members and messages, served the way Discord pages them."""

    def __init__(self, page: int = 100) -> None:
        self.page = page
        self.requests: list[str] = []
        self.rate_limited_once = False
        self.channels: list[dict[str, Any]] = [
            {"id": "100", "type": 4, "name": "Engineering", "position": 0},
            {"id": "200", "type": 0, "name": "dev", "parent_id": "100", "position": 1},
            {
                "id": "210",
                "type": 0,
                "name": "random",
                "parent_id": None,
                "position": 2,
            },
            {
                "id": "220",
                "type": 15,
                "name": "ideas",
                "parent_id": "100",
                "position": 3,
            },
            {"id": "230", "type": 2, "name": "voice", "position": 4},
        ]
        self.active: list[dict[str, Any]] = [
            {"id": "300", "type": 11, "name": "sso", "parent_id": "200"}
        ]
        self.archived: dict[str, list[dict[str, Any]]] = {
            "220": [
                {
                    "id": "310",
                    "type": 11,
                    "name": "post-a",
                    "parent_id": "220",
                    "thread_metadata": {
                        "archive_timestamp": "2026-11-01T00:00:00+00:00"
                    },
                }
            ]
        }
        self.messages: dict[str, list[dict[str, Any]]] = {
            "200": [
                message(i, "200", JESS if i % 2 else MAYA, f"dev {i}")
                for i in range(1, 8)
            ]
            + [message(8, "200", JESS, "", type=7)],
            "300": [message(20, "300", MAYA, "thread reply", mentions=[JESS])],
            "210": [message(30, "210", BOT, "build green")],
            "310": [message(40, "310", JESS, "forum post")],
            "230": [message(50, "230", MAYA, "voice chat")],
        }
        self.members: list[dict[str, Any]] = [
            {"user": JESS, "nick": "JessD"},
            {"user": MAYA, "nick": None},
        ]
        # Called with the channel ID whenever a page of its history is read.
        self.on_history: Callable[[str], None] | None = None
        # Threads found by ID but in no listing (a private archived thread).
        self.unlisted: list[dict[str, Any]] = []
        # Server errors to answer, by request (as `requests` records it), first.
        self.failures: dict[str, int] = {}

    def all_channels(self) -> list[dict[str, Any]]:
        archived = [t for threads in self.archived.values() for t in threads]
        return [*self.channels, *self.active, *archived, *self.unlisted]

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/api/v10")
        parts = path.split("/")
        key = path
        if path.endswith("/messages") and request.url.params.get("limit") == "1":
            key = f"{path}?newest"
        if self.failures.get(key):
            self.failures[key] -= 1
            self.requests.append(key)
            return httpx.Response(500, json={"message": "unavailable"})
        if key.endswith("?newest"):
            self.requests.append(key)
            newest = sorted(self.messages.get(parts[2], []), key=lambda m: int(m["id"]))
            return httpx.Response(200, json=newest[-1:])
        self.requests.append(path)
        if len(parts) == 3 and parts[1] == "channels":
            for channel in self.all_channels():
                if channel["id"] == parts[2]:
                    return httpx.Response(200, json={**channel, "guild_id": GUILD})
            return httpx.Response(404, json={"message": "Unknown Channel"})
        if len(parts) == 5 and parts[3] == "messages":
            for found in self.messages.get(parts[2], []):
                if found["id"] == parts[4]:
                    return httpx.Response(200, json=found)
            return httpx.Response(404, json={"message": "Unknown Message"})
        if path.endswith("/messages") and self.on_history is not None:
            self.on_history(parts[2])
        if path.endswith("/messages") and not self.rate_limited_once:
            self.rate_limited_once = True
            return httpx.Response(429, json={"retry_after": 0.01})
        if path == f"/guilds/{GUILD}/channels":
            return httpx.Response(200, json=self.channels)
        if path == f"/guilds/{GUILD}/threads/active":
            return httpx.Response(200, json={"threads": self.active})
        if path.endswith("/threads/archived/public"):
            channel = path.split("/")[2]
            return httpx.Response(
                200, json={"threads": self.archived.get(channel, []), "has_more": False}
            )
        if path == f"/guilds/{GUILD}/members":
            return httpx.Response(200, json=self.members)
        if path.endswith("/messages"):
            channel = path.split("/")[2]
            after = int(request.url.params.get("after", "0"))
            newer = [m for m in self.messages.get(channel, []) if int(m["id"]) > after]
            # Discord answers with the page closest to `after`, newest first.
            page = sorted(newer, key=lambda m: int(m["id"]))[: self.page]
            return httpx.Response(200, json=list(reversed(page)))
        return httpx.Response(404, json={"message": "Unknown"})


def client_for(fake: FakeDiscord) -> DiscordClient:
    return DiscordClient(
        "test-token",
        base_url="https://discord.test/api/v10",
        transport=httpx.MockTransport(fake.handler),
        sleep=lambda _s: None,
    )
