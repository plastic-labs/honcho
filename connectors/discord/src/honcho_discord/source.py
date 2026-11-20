"""`honcho ingest from discord`: this connector as a Honcho CLI source.

Registered under the ``honcho.ingest_sources`` entry point. The CLI owns the
workspace, credentials, bot filtering and sending; this reads the selected
channels and threads and hands their records over, one at a time. With
``--follow`` (``honcho ingest from discord --follow``) it then follows the
server live: see `honcho_discord.follow`.
"""

import argparse
import logging
import sys
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from honcho_discord.client import DiscordClient
from honcho_discord.export import SelectionError, export
from honcho_discord.follow import FOLLOW_METHODS, Follower, FollowSink, GatewayEvents
from honcho_discord.mapping import Container
from honcho_discord.options import (
    add_limit_option,
    add_read_options,
    channel_ids,
    read_token,
)


class RecordSink(Protocol):
    """What the CLI hands the source: see `honcho_cli.ingest_sources`."""

    def container(self, name: str) -> None: ...

    def record(self, record: Mapping[str, Any]) -> None: ...


class _ToCli:
    """The export's sink, handing each channel's or thread's records to the CLI."""

    def __init__(self, sink: RecordSink) -> None:
        self._sink: RecordSink = sink

    def container(self, container: Container) -> None:
        self._sink.container(f"#{container.name}")

    def record(self, record: dict[str, Any]) -> None:
        self._sink.record(record)

    def close(
        self, accounts: Iterable[dict[str, Any]], summary: dict[str, Any]
    ) -> None:
        _ = accounts, summary


class DiscordSource:
    help: str = "A Discord server's channels and threads"
    follows: bool = True

    def main(self, argv: list[str], sink: RecordSink) -> int:
        parser = argparse.ArgumentParser(
            prog="honcho ingest from discord",
            description="Read a Discord server's history and import it. The bot"
            + " needs View Channels and Read Message History, and the Message"
            + " Content intent. The token is read from a file, so it never appears"
            + " in a command line.",
        )
        add_read_options(parser)
        add_limit_option(parser)
        parser.add_argument(
            "--follow",
            action="store_true",
            help="Catch up from the saved position, then follow the server live"
            + " until stopped (passed by `honcho ingest from discord --follow`)",
        )
        args = parser.parse_args(argv)
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("discord").setLevel(logging.WARNING)

        selected = channel_ids(args)
        if args.follow:
            if args.limit_per_container is not None:
                raise SystemExit("--follow and --limit-per-container don't go together")
            if not all(hasattr(sink, name) for name in FOLLOW_METHODS):
                raise SystemExit("--follow needs `honcho ingest from discord --follow`")
            return _follow(args.guild, selected, read_token(args.token_file), sink)  # pyright: ignore[reportArgumentType]
        token = read_token(args.token_file)
        try:
            with DiscordClient(token) as client:
                stats, _ = export(
                    client,
                    args.guild,
                    lambda: _ToCli(sink),
                    channel_ids=selected,
                    limit_per_container=args.limit_per_container,
                )
        except SelectionError as error:
            raise SystemExit(str(error)) from None
        print(
            f"read {stats.records} records from {stats.containers} channels and"
            + " threads",
            file=sys.stderr,
        )
        if stats.skipped:
            print(f"skipped: {dict(stats.skipped)}", file=sys.stderr)
        if not stats.complete:
            print(
                f"INCOMPLETE: {len(stats.unreadable)} unreadable channels or threads,"
                + f" {len(stats.incomplete)} channels whose threads weren't all listed",
                file=sys.stderr,
            )
            return 1
        return 0


def _follow(guild: str, selected: set[str] | None, token: str, sink: FollowSink) -> int:
    with DiscordClient(token) as client:
        stats = Follower(
            client, guild, sink, GatewayEvents(token), channel_ids=selected
        ).run()
    print(
        f"followed: {stats.records} new records, {stats.edits} edits,"
        + f" {stats.deletes} deletes; {stats.catch_ups} catch-ups",
        file=sys.stderr,
    )
    return 0


SOURCE = DiscordSource()
