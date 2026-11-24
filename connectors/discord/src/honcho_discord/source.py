"""`honcho ingest from discord`: this connector as a Honcho CLI source.

Registered under the ``honcho.ingest_sources`` entry point. The CLI owns the
workspace, credentials, bot filtering and sending; this reads the selected
channels and threads and hands their records over, one at a time.
"""

import argparse
import logging
import sys
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from honcho_discord.client import DiscordClient, DiscordError
from honcho_discord.export import SelectionError, export
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
        args = parser.parse_args(argv)
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logging.getLogger("httpx").setLevel(logging.WARNING)

        selected = channel_ids(args)
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
        except (SelectionError, DiscordError) as error:
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


SOURCE = DiscordSource()
