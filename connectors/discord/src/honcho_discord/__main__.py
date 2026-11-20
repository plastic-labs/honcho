"""Command line: list a server's channels, or export its history.

Experimental: ingest, and this connector, may change or stop working without
notice.

    honcho-discord channels --token-file F --guild ID
    honcho-discord export --token-file F --guild ID [--channel ID ...] --out DIR
    honcho-discord export --token-file F --guild ID [--channel ID ...] \\
        --ingest-url URL --workspace NAME [--api-key-env VAR]

The token is a bot token, read from a file so it never appears in a command
line or shell history. The bot needs View Channels and Read Message History in
the channels it exports, the Message Content intent for message text, and
(optionally) the Server Members intent for nicknames.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from honcho_discord.client import DiscordClient
from honcho_discord.export import (
    IngestSink,
    JsonlSink,
    SelectionError,
    Sink,
    export,
    inside_git_worktree,
    select,
)
from honcho_discord.options import (
    add_limit_option,
    add_read_options,
    channel_ids,
    read_token,
)


def _channels(args: argparse.Namespace) -> int:
    with DiscordClient(read_token(args.token_file)) as client:
        selection = select(client, args.guild, channel_ids(args))
    for channel in selection.channels:
        category = selection.categories.get(channel.parent_id or "", "")
        threads = len(selection.threads.get(channel.id, []))
        print(
            f"{channel.id}\t{category}\t{channel.name}\ttype={channel.type}\tthreads={threads}"
        )
    if selection.incomplete:
        print(
            f"INCOMPLETE: {len(selection.incomplete)} channels whose threads"
            + " weren't all listed; their thread counts are low"
        )
        return 1
    return 0


def _export(args: argparse.Namespace) -> int:
    # Checked before anything is read; the sink checks again when it opens.
    out = Path(args.out).expanduser() if args.out else None
    if out is not None and inside_git_worktree(out) and not args.allow_git_dir:
        raise SystemExit(
            f"{args.out} is inside a git checkout, where exported messages could be"
            + " committed. Choose another directory, or pass --allow-git-dir."
        )
    selected = channel_ids(args)
    token = read_token(args.token_file)

    def make_sink() -> Sink:
        if out is not None:
            return JsonlSink(out, allow_git_dir=args.allow_git_dir)
        api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
        return IngestSink(args.ingest_url, args.workspace, api_key=api_key)

    with DiscordClient(token) as client:
        stats, sink = export(
            client,
            args.guild,
            make_sink,
            channel_ids=selected,
            limit_per_container=args.limit_per_container,
        )
    print(f"{stats.records} records from {stats.containers} channels and threads")
    if stats.skipped:
        print(f"skipped: {dict(stats.skipped)}")
    if isinstance(sink, IngestSink):
        print(f"ingest outcomes: {dict(sink.outcomes)}")
        if sink.rejections:
            print(f"rejections: {dict(sink.rejections)}")
    failed = False
    if not stats.complete:
        print(
            f"INCOMPLETE: {len(stats.unreadable)} unreadable channels or threads,"
            + f" {len(stats.incomplete)} channels whose threads weren't all listed"
        )
        failed = True
    if isinstance(sink, IngestSink) and sink.rejections:
        print(f"REJECTED: {sum(sink.rejections.values())} records")
        failed = True
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="honcho-discord", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    channels = sub.add_parser("channels", help="List the channels an export would read")
    add_read_options(channels)

    exporting = sub.add_parser("export", help="Export history as ingest records")
    add_read_options(exporting)
    target = exporting.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", help="Directory for the JSONL files")
    target.add_argument("--ingest-url", help="Honcho base URL to send records to")
    exporting.add_argument("--workspace", help="Workspace for --ingest-url")
    exporting.add_argument(
        "--api-key-env", help="Environment variable holding the Honcho API key"
    )
    add_limit_option(exporting)
    exporting.add_argument(
        "--allow-git-dir", action="store_true", help="Allow --out inside a git checkout"
    )

    args = parser.parse_args(argv)
    if args.command == "export" and args.ingest_url and not args.workspace:
        parser.error("--ingest-url needs --workspace")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        return _channels(args) if args.command == "channels" else _export(args)
    except SelectionError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    sys.exit(main())
