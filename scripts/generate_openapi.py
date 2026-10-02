"""Regenerate ``docs/v3/openapi.json`` from the FastAPI app.

    uv run python -m scripts.generate_openapi            # rewrite the spec
    uv run python -m scripts.generate_openapi --check    # exit 1 on drift

The committed spec is what ``honcho.dev/docs/v3/api-reference`` renders: every
page under it is a three-line Mintlify stub whose content comes entirely from
this file. Historically it was hand-edited, so it drifted behind ``main`` --
endpoints that shipped months ago were missing from the reference. This script
makes the file a build artifact instead. Do not edit it by hand.

Formatting is ``json.dumps(indent=2)`` rather than prettier, even though the
file was previously prettier-formatted. The drift check has to produce the same
bytes on every machine, and a stdlib serializer has no version to drift with and
needs no Node toolchain in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

from src.utils.json_coerce import as_dict, as_list

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO_ROOT / "docs" / "v3" / "openapi.json"

_DEFS_KEY = "$defs"
_REF_KEY = "$ref"
_DEFS_PREFIX = "#/$defs/"
_COMPONENTS_PREFIX = "#/components/schemas/"

_REGENERATE_HINT = (
    "\nRegenerate it with:\n"
    + "  uv run python -m scripts.generate_openapi\n"
    + "\nAdding an endpoint also needs a stub page and a nav entry:\n"
    + "  cd docs && bun run openapi"
    + "   # writes missing stubs, prints the nav block\n"
    + "then paste the new group into docs/docs.json."
)

_MAX_REPORTED_DIFFS = 40


class SpecGenerationError(RuntimeError):
    """The app's schema cannot be turned into a publishable spec."""


def build_spec() -> dict[str, Any]:
    """Return the app's OpenAPI schema, normalized for publication."""
    # Imported lazily so `--help` doesn't pay for importing the whole app.
    from fastapi_pagination import add_pagination

    from src.main import app

    # `src.main` calls add_pagination() before include_router(), so at import
    # time it patches an empty route table. The app gets away with it because
    # add_pagination() also wraps the lifespan and re-patches on startup -- but
    # this script never starts the app, so without this call the generated spec
    # would silently omit `page`/`size` from every paginated endpoint. Calling
    # it again is safe: fastapi-pagination tags the dependency it injects and
    # skips any route that already has it.
    add_pagination(app)
    app.openapi_schema = None

    spec = app.openapi()
    _ = _hoist_defs(spec)
    return spec


def _hoist_defs(spec: dict[str, Any]) -> list[str]:
    """Move local ``$defs`` schemas into ``components/schemas``.

    Both chat routes hand-roll their 200 response as
    ``DialecticResponse.model_json_schema()`` because the same operation also
    returns ``text/event-stream``, which rules out ``response_model``. FastAPI
    only hoists nested models into ``components`` for ``response_model``, so
    those land in a local ``$defs`` block instead.

    That is legal OpenAPI 3.1, but Mintlify cannot resolve ``#/$defs/`` refs and
    aborts the entire docs build on them, so the refs are rewritten here.

    Returns the sorted names that were hoisted.
    """
    collected: dict[str, object] = {}

    def collect(node: object) -> None:
        mapping = as_dict(node)
        if mapping is not None:
            defs = as_dict(mapping.pop(_DEFS_KEY, None))
            if defs is not None:
                for name, subschema in defs.items():
                    collect(subschema)
                    previous = collected.get(name)
                    if previous is not None and previous != subschema:
                        raise SpecGenerationError(
                            f"Two different schemas are both named {name!r} in a"
                            + f" local {_DEFS_KEY} block, so neither can be hoisted"
                            + " into components/schemas. Rename one of the models."
                        )
                    collected[name] = subschema
            for value in list(mapping.values()):
                collect(value)
            return

        sequence = as_list(node)
        if sequence is not None:
            for item in sequence:
                collect(item)

    collect(spec)

    components = cast(dict[str, object], spec.setdefault("components", {}))
    schemas = cast(dict[str, object], components.setdefault("schemas", {}))
    for name, subschema in collected.items():
        existing = schemas.get(name)
        if existing is not None and existing != subschema:
            raise SpecGenerationError(
                f"Hoisting {name!r} out of a local {_DEFS_KEY} block would"
                + " overwrite a different schema of the same name already in"
                + " components/schemas. Rename one of the models."
            )
        schemas[name] = subschema

    _rewrite_refs(spec)
    return sorted(collected)


def _rewrite_refs(node: object) -> None:
    """Repoint every ``#/$defs/`` ref at ``#/components/schemas/``."""
    mapping = as_dict(node)
    if mapping is not None:
        ref = mapping.get(_REF_KEY)
        if isinstance(ref, str) and ref.startswith(_DEFS_PREFIX):
            mapping[_REF_KEY] = _COMPONENTS_PREFIX + ref[len(_DEFS_PREFIX) :]
        for value in mapping.values():
            _rewrite_refs(value)
        return

    sequence = as_list(node)
    if sequence is not None:
        for item in sequence:
            _rewrite_refs(item)


def render(spec: dict[str, Any]) -> str:
    return json.dumps(spec, indent=2) + "\n"


def describe_drift(committed: dict[str, Any], generated: dict[str, Any]) -> list[str]:
    """Summarize where two specs disagree, as JSON-pointer-ish paths."""
    found: list[str] = []

    def walk(a: object, b: object, path: str) -> None:
        if len(found) > _MAX_REPORTED_DIFFS:
            return

        if type(a) is not type(b):
            found.append(
                f"{path}: {type(a).__name__} in committed,"
                + f" {type(b).__name__} in generated"
            )
            return

        a_mapping, b_mapping = as_dict(a), as_dict(b)
        if a_mapping is not None and b_mapping is not None:
            for key in sorted(set(a_mapping) | set(b_mapping)):
                if key not in a_mapping:
                    found.append(f"{path}/{key}: missing from committed spec")
                elif key not in b_mapping:
                    found.append(f"{path}/{key}: no longer produced by the app")
                else:
                    walk(a_mapping[key], b_mapping[key], f"{path}/{key}")
            return

        a_sequence, b_sequence = as_list(a), as_list(b)
        if a_sequence is not None and b_sequence is not None:
            if len(a_sequence) != len(b_sequence):
                found.append(
                    f"{path}: {len(a_sequence)} entries in committed,"
                    + f" {len(b_sequence)} generated"
                )
            else:
                pairs = zip(a_sequence, b_sequence, strict=True)
                for index, (x, y) in enumerate(pairs):
                    walk(x, y, f"{path}[{index}]")
            return

        if a != b:
            found.append(f"{path}: differs")

    walk(committed, generated, "")
    return found


def _report_drift(committed_text: str, spec: dict[str, Any], relative: Path) -> None:
    try:
        committed = cast(dict[str, Any], json.loads(committed_text))
    except json.JSONDecodeError as exc:
        # Checked before the "out of date" header: a file this broken has no
        # drift to describe, and 40 lines of "missing from committed spec" would
        # bury the actual problem.
        print(f"{relative} is not valid JSON: {exc}")
        print(_REGENERATE_HINT)
        return

    print(f"{relative} is out of date.\n")

    drift = describe_drift(committed, spec)
    if not drift:
        print("  The content matches; only the formatting differs.")
    else:
        shown = drift[:_MAX_REPORTED_DIFFS]
        for line in shown:
            print(f"  {line or '/'}")
        if len(drift) > len(shown):
            print(f"  ... and {len(drift) - len(shown)} more")

    print(_REGENERATE_HINT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift and exit 1 instead of rewriting the spec.",
    )
    args = parser.parse_args(argv)

    try:
        spec = build_spec()
    except SpecGenerationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rendered = render(spec)
    relative = SPEC_PATH.relative_to(REPO_ROOT)

    if not args.check:
        _ = SPEC_PATH.write_text(rendered)
        print(f"Wrote {relative}")
        return 0

    try:
        committed_text = SPEC_PATH.read_text()
    except FileNotFoundError:
        print(f"{relative} is missing.")
        print(_REGENERATE_HINT)
        return 1

    if committed_text == rendered:
        print(f"{relative} is up to date.")
        return 0

    _report_drift(committed_text, spec, relative)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
