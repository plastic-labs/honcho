"""Chat flags that map onto the 3.4.0 dialectic request body."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from honcho_cli.output import print_error, status


def parse_response_format(raw: str | None) -> dict[str, Any] | None:
    """Parse ``--response-format`` into the JSON Schema object the API expects.

    The value is inline JSON, or ``@path`` to a file containing that JSON.
    The server requires a schema whose root ``type`` is ``object`` and stores
    the schema as an object, so this returns a dict and never a JSON string.
    """
    if raw is None:
        return None
    text = raw
    if raw.startswith("@"):
        path = Path(raw[1:])
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            print_error(
                "INVALID_RESPONSE_FORMAT",
                f"Could not read response format {path}: {e}",
            )
            raise typer.Exit(1) from e
    try:
        schema = json.loads(text)
    except json.JSONDecodeError as e:
        print_error(
            "INVALID_RESPONSE_FORMAT",
            f"--response-format must be a JSON object: {e}",
        )
        raise typer.Exit(1) from e
    if not isinstance(schema, dict):
        kind = type(schema).__name__
        print_error(
            "INVALID_RESPONSE_FORMAT",
            f"--response-format must be a JSON object (a JSON Schema), not a {kind}",
        )
        raise typer.Exit(1)
    if schema.get("type") != "object":
        print_error(
            "INVALID_RESPONSE_FORMAT",
            "response_format root schema must have type 'object'",
        )
        raise typer.Exit(1)
    return schema


def present_chat_content(content: Any, *, structured: bool) -> Any:
    """Return chat content for CLI output.

    With a response format the API's ``content`` is a JSON string. Parse it so
    ``--json`` nests the object instead of escaping the string. An ordinary
    answer stays a string even when it happens to look like JSON.
    """
    if not structured or not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def named_scope_recall(
    client: Any,
    *,
    session_id: str | None,
    session_from_flag: bool,
    scope_names: list[str] | None,
    sessions: list[str] | None,
) -> tuple[str | None, list[str] | None]:
    """Recall bounds to forward for this client.

    A named-scope key's projection is the key's scope. ``session_id`` and a
    session allowlist are mutually exclusive with that projection, and another
    scope name is rejected. An ambient ``HONCHO_SESSION_ID`` is dropped so it
    does not turn an allowed chat into a 422; an explicit ``-s`` is an error.
    Other clients are unchanged.
    """
    from honcho_cli.credentials import is_named_scope_client, named_scope_name

    if not is_named_scope_client(client):
        return session_id, scope_names
    if session_from_flag or sessions:
        print_error(
            "INCOMPATIBLE_FLAGS",
            "A named-scope key already confines recall to its own scope. "
            + "Do not pass -s/--session or --sessions.",
        )
        raise typer.Exit(1)
    bound = named_scope_name(client)
    if scope_names and bound and scope_names != [bound]:
        print_error(
            "PERMISSION_ERROR",
            f"This key is limited to scope '{bound}'.",
            {"scope": bound},
        )
        raise typer.Exit(1)
    if session_id:
        status(
            "Ignoring HONCHO_SESSION_ID: a named-scope key confines recall to its scope."
        )
    return None, scope_names
