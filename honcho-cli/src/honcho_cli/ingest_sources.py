"""Connectors that `honcho ingest from <source>` runs.

A connector package makes itself available by registering an object under the
``honcho.ingest_sources`` entry point group, named for its source::

    [project.entry-points."honcho.ingest_sources"]
    github = "honcho_github.source:SOURCE"

The object provides two things:

* ``help``: one line, shown when ``honcho ingest from`` lists the sources.
* ``main(argv, sink) -> int``: parse the connector's own options from ``argv``
  (``--help`` included), read the source, and hand the records to ``sink``:
  ``sink.container(name)`` before a container's records, then
  ``sink.record(record)`` for each one, a mapping in the ingest record format.
  Return the exit code, non-zero when the read was incomplete.

The CLI does the rest: the workspace and credentials from its own
configuration, validation, holding back bot records no person interacted with
(per container), batching, sending with derive later, and the summary. A
connector needs nothing from this package; the protocols below document what
it provides and what it gets.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Protocol

GROUP = "honcho.ingest_sources"


class RecordSink(Protocol):
    """What a source hands its records to."""

    def container(self, name: str) -> None:
        """A container's records follow; `name` is for progress and errors."""
        ...

    def record(self, record: Mapping[str, Any]) -> None:
        """One ingest record."""
        ...


class IngestSource(Protocol):
    """What a connector registers under the entry point group."""

    help: str

    def main(self, argv: list[str], sink: RecordSink) -> int:
        """Read the source into `sink`; return the exit code."""
        ...


class SourceError(Exception):
    """A registered source can't be loaded or isn't shaped like one."""


def installed() -> dict[str, EntryPoint]:
    """The registered sources by name, in name order."""
    found = {point.name: point for point in entry_points(group=GROUP)}
    return dict(sorted(found.items()))


def load(point: EntryPoint) -> IngestSource:
    """The source an entry point names, checked for `help` and `main`."""
    try:
        source = point.load()
    except Exception as e:  # any import error in the connector package
        raise SourceError(f"couldn't load {point.value}: {e}") from e
    if not isinstance(getattr(source, "help", None), str) or not callable(
        getattr(source, "main", None)
    ):
        raise SourceError(f"{point.value} doesn't provide `help` and `main(argv, sink)`")
    return source  # pyright: ignore[reportReturnType]
