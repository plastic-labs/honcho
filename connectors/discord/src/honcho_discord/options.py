"""Options shared by `honcho-discord` and `honcho ingest from discord`."""

import argparse
from pathlib import Path


def add_read_options(parser: argparse.ArgumentParser) -> None:
    """What to read and with what token: --token-file, --guild, --channel(s)."""
    parser.add_argument(
        "--token-file", required=True, help="File holding the bot token"
    )
    parser.add_argument("--guild", required=True, help="Server (guild) ID")
    parser.add_argument("--channel", action="append", help="Channel ID; repeatable")
    parser.add_argument(
        "--channels-file", help="File of channel IDs, one per line; # comments"
    )


def add_limit_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--limit-per-container", type=int, help="At most this many records per channel"
    )


def read_token(path: str) -> str:
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


def channel_ids(args: argparse.Namespace) -> set[str] | None:
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
