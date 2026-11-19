"""`honcho ingest from github`: this connector as a Honcho CLI source.

Registered under the ``honcho.ingest_sources`` entry point. The CLI owns the
workspace, credentials, bot filtering and sending; this reads the selected
repositories and hands their records over, one repository at a time.
"""

import argparse
import logging
import sys
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from honcho_github.client import GitHubClient, GitHubError
from honcho_github.export import SelectionError, export
from honcho_github.mapping import Repo
from honcho_github.options import (
    add_read_options,
    add_since_option,
    parse_since,
    read_token,
)


class RecordSink(Protocol):
    """What the CLI hands the source: see `honcho_cli.ingest_sources`."""

    def container(self, name: str) -> None: ...

    def record(self, record: Mapping[str, Any]) -> None: ...


class _ToCli:
    """The export's sink, handing each repository's records to the CLI."""

    def __init__(self, sink: RecordSink) -> None:
        self._sink: RecordSink = sink

    def container(self, repo: Repo) -> None:
        self._sink.container(repo.name_with_owner)

    def record(self, record: dict[str, Any]) -> None:
        self._sink.record(record)

    def close(
        self, accounts: Iterable[dict[str, Any]], summary: dict[str, Any]
    ) -> None:
        _ = accounts, summary


class GitHubSource:
    help: str = "GitHub repositories: issues, pull requests, reviews and commits"

    def main(self, argv: list[str], sink: RecordSink) -> int:
        parser = argparse.ArgumentParser(
            prog="honcho ingest from github",
            description="Read GitHub repositories' history and import it."
            + " `--org` reads public repositories only; name a private one with"
            + " `--repo`. The token is read from a file, so it never appears in a"
            + " command line.",
        )
        add_read_options(parser)
        add_since_option(parser)
        args = parser.parse_args(argv)
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logging.getLogger("httpx").setLevel(logging.WARNING)

        token = read_token(args.token_file)
        since = parse_since(args.since)
        try:
            with GitHubClient(token, host=args.host) as client:
                stats, _ = export(
                    client,
                    args.repo or [],
                    args.org or [],
                    lambda: _ToCli(sink),
                    since=since,
                )
        except (SelectionError, GitHubError) as error:
            raise SystemExit(str(error)) from None
        print(
            f"read {stats.records} records from {stats.repos} repositories",
            file=sys.stderr,
        )
        if stats.skipped:
            print(f"skipped: {dict(stats.skipped)}", file=sys.stderr)
        if not stats.complete:
            print(
                f"INCOMPLETE: couldn't read {', '.join(sorted(stats.unreadable))}",
                file=sys.stderr,
            )
            return 1
        return 0


SOURCE = GitHubSource()
