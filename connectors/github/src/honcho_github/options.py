"""Options shared by `honcho-github` and `honcho ingest from github`."""

import argparse
import datetime
from pathlib import Path


def add_read_options(parser: argparse.ArgumentParser) -> None:
    """What to read and with what token: --token-file, --repo, --org, --host."""
    parser.add_argument("--token-file", required=True, help="File holding the token")
    parser.add_argument(
        "--repo", action="append", help="A repository, OWNER/NAME; repeatable"
    )
    parser.add_argument(
        "--org",
        action="append",
        help="An organization or account whose public repositories to read;"
        + " repeatable",
    )
    parser.add_argument(
        "--host", default="github.com", help="GitHub host (default github.com)"
    )


def add_since_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--since",
        help="Only issues and pull requests updated since this date, and commits"
        + " made since it",
    )


def read_token(path: str) -> str:
    """The token: one line of printable ASCII, never echoed."""
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


def parse_since(value: str | None) -> str | None:
    """`--since` as GitHub's timestamp format, UTC: 2026-11-01T00:00:00Z."""
    if value is None:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise SystemExit(f"--since {value!r} isn't a date (YYYY-MM-DD)") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.UTC)
    return parsed.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
