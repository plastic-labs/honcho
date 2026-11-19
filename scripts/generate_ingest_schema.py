"""Regenerate ``docs/schemas/ingest-record.v0.json``, the published ingest record schema.

    uv run python -m scripts.generate_ingest_schema            # rewrite it
    uv run python -m scripts.generate_ingest_schema --check    # exit 1 on drift

The schema is the server's own request model for one record
(``src.schemas.api.IngestRecordCreate``), the same model the OpenAPI spec
documents, so an exporter validates its output against exactly what ingest
accepts. Unknown fields are rejected, as the SDK and `honcho ingest import`
reject them, so a misspelled field fails validation instead of being dropped
(the API itself ignores unknown fields). It is v0: the format may change until
edits and deletes settle. Do not edit the file by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
# The record's nested types; their unknown fields are rejected like the record's.
_STRICT_DEFS = (
    "IngestContainer",
    "IngestParticipant",
    "MessageConfiguration",
    "ReasoningConfiguration",
)
SCHEMA_PATH = REPO_ROOT / "docs" / "schemas" / "ingest-record.v0.json"
SCHEMA_ID = (
    "https://raw.githubusercontent.com/plastic-labs/honcho/main/"
    + "docs/schemas/ingest-record.v0.json"
)


def build_schema() -> dict[str, Any]:
    """The JSON Schema for one ingest record, ready to publish."""
    from src.schemas.api import IngestRecordCreate

    schema = IngestRecordCreate.model_json_schema(mode="validation")
    published: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "Honcho ingest record (v0)",
        "description": (
            "One source record for Honcho's ingest API: a line of a JSONL file for"
            + " `honcho ingest import`, or an item of the `records` array of"
            + " `POST /v3/workspaces/{workspace_id}/ingest`. Experimental: v0 may"
            + " change until edits and deletes settle. Unknown fields are rejected,"
            + " as `honcho ingest import` and the SDKs reject them; the API itself"
            + " ignores them."
        ),
    }
    published.update(
        {
            key: value
            for key, value in schema.items()
            if key not in ("title", "description")
        }
    )
    published["additionalProperties"] = False
    for name in _STRICT_DEFS:
        published["$defs"][name]["additionalProperties"] = False
    return published


def render(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if the committed schema differs from the model",
    )
    args = parser.parse_args(argv)

    rendered = render(build_schema())
    if args.check:
        current = SCHEMA_PATH.read_text() if SCHEMA_PATH.exists() else None
        if current != rendered:
            print(
                f"{SCHEMA_PATH.relative_to(REPO_ROOT)} is out of date with"
                + " src.schemas.api.IngestRecordCreate.\nRegenerate it with:\n"
                + "  uv run python -m scripts.generate_ingest_schema",
                file=sys.stderr,
            )
            return 1
        print(f"{SCHEMA_PATH.relative_to(REPO_ROOT)} is up to date.")
        return 0
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(rendered)
    print(f"Wrote {SCHEMA_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
