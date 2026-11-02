"""Helpers for characters Postgres cannot store in text columns.

Postgres rejects NUL (0x00) in ``text``/``varchar`` values and in ``jsonb``
strings, so any string bound into a query or persisted to those columns has to
have NUL removed first. This applies to model-generated text as much as to
user-supplied input: an LLM can emit a ``\\u0000`` escape in its tool-call
arguments, which the JSON parser decodes into a real NUL byte.
"""

import re
from typing import Any, cast, overload

from pydantic import BeforeValidator

__all__ = ["NulStripped", "escape_unstorable", "strip_nul"]

# NUL and UTF-16 surrogate code points: the only characters Postgres can't store
# in text. A Python str can hold surrogates (a JSON "\\ud83d" escape decodes to
# one), and they can't be UTF-8 encoded on the way to the driver.
_UNSTORABLE = re.compile("[\x00\ud800-\udfff]")


@overload
def strip_nul(value: str) -> str: ...


@overload
def strip_nul(value: Any) -> Any: ...


def strip_nul(value: Any) -> Any:
    """Recursively remove NUL bytes from strings, including nested ones.

    Dict keys are stripped alongside values. Anything that is not a string,
    dict, or list -- ``None`` included -- is returned unchanged, so this can be
    applied to an optional field without a guard.
    """
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        d = cast(dict[str, Any], value)
        return {strip_nul(k): strip_nul(v) for k, v in d.items()}
    if isinstance(value, list):
        lst = cast(list[Any], value)
        return [strip_nul(item) for item in lst]
    return value


def escape_unstorable(text: str) -> str:
    """Replace NUL and lone surrogates with their escape text, e.g. ``\\x00``.

    For diagnostic text such as a stored error message, where the bad character
    is evidence worth keeping. Use ``strip_nul`` for content instead. A literal
    backslash escape already in ``text`` is left alone, so it reads the same as
    an escaped character.
    """
    return _UNSTORABLE.sub(
        lambda m: "\\x00" if m.group() == "\x00" else f"\\u{ord(m.group()):04x}",
        text,
    )


# Reusable annotation for string fields; composes with a per-field Field(...).
# Runs *before* the field's own constraints, so `min_length` is checked against
# the stripped value and all-NUL input is rejected instead of becoming "".
NulStripped = BeforeValidator(strip_nul)
