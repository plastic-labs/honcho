"""Command line: list the repositories an export would read, or export them.

    honcho-github repos --token-file F (--repo OWNER/NAME ... | --org LOGIN ...)
    honcho-github export --token-file F (--repo ... | --org ...) [--since DATE] --out DIR
    honcho-github export --token-file F (--repo ... | --org ...) [--since DATE] \\
        --ingest-url URL --workspace NAME [--api-key-env VAR]

The token is read from a file so it never appears in a command line or shell
history. A read-only fine-grained token with Metadata, Contents, Issues and Pull
requests read access is enough. `--org` reads an account's or organization's public
repositories only; name a private repository with `--repo` to read it.
"""

import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

from honcho_github.client import GitHubClient, GitHubError
from honcho_github.export import (
    IngestError,
    IngestSink,
    JsonlSink,
    SelectionError,
    Sink,
    export,
    inside_git_worktree,
    select,
)


def _token(path: str) -> str:
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


def _since(value: str | None) -> str | None:
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


def _repos(args: argparse.Namespace) -> int:
    with GitHubClient(_token(args.token_file), host=args.host) as client:
        selected = select(client, args.repo or [], args.org or [])
    for listed in selected:
        visibility = "private" if listed.repo.private else "public"
        print(
            f"{listed.repo.name_with_owner}\t{visibility}"
            + f"\tprs={listed.pull_requests}\tissues={listed.issues}"
        )
    return 0


def _export(args: argparse.Namespace) -> int:
    # Checked before anything is read; the sink checks again when it opens.
    out = Path(args.out).expanduser() if args.out else None
    if out is not None and inside_git_worktree(out) and not args.allow_git_dir:
        raise SystemExit(
            f"{args.out} is inside a git checkout, where exported text could be"
            + " committed. Choose another directory, or pass --allow-git-dir."
        )
    token = _token(args.token_file)
    since = _since(args.since)

    def make_sink() -> Sink:
        if out is not None:
            return JsonlSink(out, allow_git_dir=args.allow_git_dir)
        api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
        return IngestSink(
            args.ingest_url, args.workspace, api_key=api_key, bots=args.bots
        )

    with GitHubClient(token, host=args.host) as client:
        stats, sink = export(
            client, args.repo or [], args.org or [], make_sink, since=since
        )
    print(f"{stats.records} records from {stats.repos} repositories")
    if stats.skipped:
        print(f"skipped: {dict(stats.skipped)}")
    if isinstance(sink, IngestSink):
        print(f"ingest outcomes: {dict(sink.outcomes)}")
        if sink.held_back:
            print(
                f"{sink.held_back} bot records held back (--bots {args.bots});"
                + " --bots all sends them"
            )
        if sink.rejections:
            print(f"rejections: {dict(sink.rejections)}")
    failed = False
    if not stats.complete:
        print(f"INCOMPLETE: couldn't read {', '.join(sorted(stats.unreadable))}")
        failed = True
    if isinstance(sink, IngestSink) and sink.rejections:
        print(f"REJECTED: {sum(sink.rejections.values())} records")
        failed = True
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="honcho-github", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--token-file", required=True, help="File holding the token")
        p.add_argument(
            "--repo", action="append", help="A repository, OWNER/NAME; repeatable"
        )
        p.add_argument(
            "--org",
            action="append",
            help="An organization or account whose public repositories to read;"
            + " repeatable",
        )
        p.add_argument(
            "--host", default="github.com", help="GitHub host (default github.com)"
        )

    listing = sub.add_parser("repos", help="List the repositories an export would read")
    common(listing)

    exporting = sub.add_parser("export", help="Export history as ingest records")
    common(exporting)
    exporting.add_argument(
        "--since",
        help="Only issues and pull requests updated since this date, and commits"
        + " made since it",
    )
    target = exporting.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", help="Directory for the JSONL files")
    target.add_argument("--ingest-url", help="Honcho base URL to send records to")
    exporting.add_argument("--workspace", help="Workspace for --ingest-url")
    exporting.add_argument(
        "--api-key-env", help="Environment variable holding the Honcho API key"
    )
    exporting.add_argument(
        "--allow-git-dir", action="store_true", help="Allow --out inside a git checkout"
    )
    exporting.add_argument(
        "--bots",
        choices=("interacted", "all", "none"),
        default="interacted",
        help="Bot records sent with --ingest-url: interacted (default) sends only"
        + " those a person replied to or mentioned the bot about. --out always"
        + " writes every record; `honcho ingest import --bots` filters a file.",
    )

    args = parser.parse_args(argv)
    if args.command == "export" and args.ingest_url and not args.workspace:
        parser.error("--ingest-url needs --workspace")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        return _repos(args) if args.command == "repos" else _export(args)
    except SelectionError as error:
        raise SystemExit(str(error)) from None
    except GitHubError as error:
        raise SystemExit(str(error)) from None
    except IngestError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    sys.exit(main())
