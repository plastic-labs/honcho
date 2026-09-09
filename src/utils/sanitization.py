"""Helpers for stripping bytes Postgres cannot store in text columns.

Postgres rejects NUL (0x00) in ``text``/``varchar`` values and in ``jsonb``
strings, so any string bound into a query or persisted to those columns has to
have NUL removed first. This applies to model-generated text as much as to
user-supplied input: an LLM can emit a ``\\u0000`` escape in its tool-call
arguments, which the JSON parser decodes into a real NUL byte.
"""

from typing import Any, cast, overload

from pydantic import BeforeValidator

__all__ = ["NulStripped", "strip_nul"]


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


# Reusable annotation for string fields; composes with a per-field Field(...).
# Runs *before* the field's own constraints, so `min_length` is checked against
# the stripped value and all-NUL input is rejected instead of becoming "".
NulStripped = BeforeValidator(strip_nul)
