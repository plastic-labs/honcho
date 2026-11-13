"""Command line: list a server's channels, or export its history.

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


def _token(path: str) -> str:
    """The bot token: one line of printable ASCII, never echoed."""
    token = Path(path).expanduser().read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit(f"{path} is empty")
    if (
        not token.isascii()
        or not token.isprintable()
        or any(c.isspace() for c in token)
    ):
        raise SystemExit(f"{path} should hold just the token, on one line")
    return token


def _channel_ids(args: argparse.Namespace) -> set[str] | None:
    """The channels asked for, or None to export every channel.

    A channels file that lists nothing is an error, not "everything".
    """
    if not args.channel and not args.channels_file:
        return None
    ids = set(args.channel or [])
    if args.channels_file:
        for line in Path(args.channels_file).expanduser().read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                ids.add(line)
    if not ids:
        raise SystemExit(f"{args.channels_file} lists no channel IDs")
    return ids


def _channels(args: argparse.Namespace) -> int:
    with DiscordClient(_token(args.token_file)) as client:
        selection = select(client, args.guild, _channel_ids(args))
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
    channel_ids = _channel_ids(args)
    token = _token(args.token_file)

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
            channel_ids=channel_ids,
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

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--token-file", required=True, help="File holding the bot token")
        p.add_argument("--guild", required=True, help="Server (guild) ID")
        p.add_argument("--channel", action="append", help="Channel ID; repeatable")
        p.add_argument(
            "--channels-file", help="File of channel IDs, one per line; # comments"
        )

    channels = sub.add_parser("channels", help="List the channels an export would read")
    common(channels)

    exporting = sub.add_parser("export", help="Export history as ingest records")
    common(exporting)
    target = exporting.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", help="Directory for the JSONL files")
    target.add_argument("--ingest-url", help="Honcho base URL to send records to")
    exporting.add_argument("--workspace", help="Workspace for --ingest-url")
    exporting.add_argument(
        "--api-key-env", help="Environment variable holding the Honcho API key"
    )
    exporting.add_argument(
        "--limit-per-container", type=int, help="At most this many records per channel"
    )
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
