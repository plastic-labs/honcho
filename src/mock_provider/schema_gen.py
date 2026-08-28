"""Generate a conforming instance from a JSON Schema.

The deriver is a structured-output caller: it sends a schema and parses the
reply back into a Pydantic model. A mock that answers with prose does not fail
loudly — ``repair_response_model_json`` swallows the error and hands back an
empty ``PromptRepresentation``, which reads as "the deriver found nothing"
rather than "the mock is wrong". So generation is driven by the schema that was
actually sent, ``$ref`` indirection and all.

Values are derived from a hash of the property path, so the same schema always
produces the same instance and two different fields never collide.
"""

from __future__ import annotations

import hashlib
from typing import Any

from src.mock_provider.coerce import as_dict, as_int, as_list, as_str

# Depth cap for self-referential schemas. Reasoning-tree models nest premises
# inside conclusions, so a $ref cycle is normal input, not a malformed schema.
MAX_DEPTH = 6

_WORDS = (
    "synthetic",
    "placeholder",
    "mock",
    "sample",
    "fixture",
    "stub",
    "generated",
    "example",
    "inert",
    "dummy",
)


def _seed(path: str) -> int:
    return int.from_bytes(hashlib.sha256(path.encode()).digest()[:8], "big")


def _phrase(path: str, words: int = 6) -> str:
    """An obviously-synthetic sentence, stable for a given path."""
    seed = _seed(path)
    picked = [_WORDS[(seed >> (i * 5)) % len(_WORDS)] for i in range(words)]
    return f"[mock] {' '.join(picked)}"


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Follow a local ``$ref`` chain to the schema it points at.

    Only local refs are supported: the mock never fetches over the network, and
    Pydantic's ``model_json_schema()`` only ever emits ``#/$defs/...``.
    """
    seen: set[str] = set()
    current = schema
    while "$ref" in current:
        ref = as_str(current["$ref"])
        if ref is None or not ref.startswith("#/") or ref in seen:
            return {}
        seen.add(ref)

        target: dict[str, Any] | None = root
        for part in ref[2:].split("/"):
            if target is None or part not in target:
                return {}
            target = as_dict(target[part])
        if target is None:
            return {}
        current = target
    return current


def _merge_all_of(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Flatten ``allOf`` into the parent so one pass can read properties off it."""
    branches = as_list(schema.get("allOf"))
    if branches is None:
        return schema

    merged: dict[str, Any] = {k: v for k, v in schema.items() if k != "allOf"}
    for branch in branches:
        resolved_branch = as_dict(branch)
        if resolved_branch is None:
            continue
        resolved = _resolve(resolved_branch, root)
        for key, value in resolved.items():
            if key == "properties":
                properties = as_dict(value)
                if properties is not None:
                    existing = as_dict(merged.get("properties")) or {}
                    merged["properties"] = {**properties, **existing}
                    continue
            if key == "required":
                required = as_list(value)
                if required is not None:
                    previous = as_list(merged.get("required")) or []
                    merged["required"] = list({*previous, *required})
                    continue
            merged.setdefault(key, value)
    return merged


def _infer_type(schema: dict[str, Any]) -> str:
    """Best-effort type when the schema omits an explicit ``type``."""
    declared = schema.get("type")
    if (name := as_str(declared)) is not None:
        return name
    if (names := as_list(declared)) is not None:
        # Nullable unions arrive as ["string", "null"]; prefer the real type.
        for candidate in names:
            if (candidate_name := as_str(candidate)) and candidate_name != "null":
                return candidate_name
        return "null"
    if "properties" in schema:
        return "object"
    if "items" in schema:
        return "array"
    return "string"


def generate(schema: dict[str, Any], root: dict[str, Any] | None = None) -> Any:
    """Build a value satisfying ``schema``.

    ``root`` carries the document that ``$ref`` resolves against; it defaults to
    ``schema`` itself, which is the shape Pydantic emits.
    """
    return _generate(schema, root if root is not None else schema, "$", 0)


def _generate(
    schema: dict[str, Any], root: dict[str, Any], path: str, depth: int
) -> Any:
    resolved = _merge_all_of(_resolve(schema, root), root)

    if "const" in resolved:
        return resolved["const"]

    enum = as_list(resolved.get("enum"))
    if enum:
        return enum[_seed(path) % len(enum)]

    if depth >= MAX_DEPTH and "default" in resolved:
        return resolved["default"]

    for key in ("anyOf", "oneOf"):
        branches = as_list(resolved.get(key))
        if branches:
            return _generate(_pick_branch(branches, root, depth), root, path, depth)

    kind = _infer_type(resolved)
    if kind == "object":
        return _generate_object(resolved, root, path, depth)
    if kind == "array":
        return _generate_array(resolved, root, path, depth)
    if kind == "integer":
        return _bounded_int(resolved, path)
    if kind == "number":
        return float(_bounded_int(resolved, path))
    if kind == "boolean":
        return _seed(path) % 2 == 0
    if kind == "null":
        return None
    return _generate_string(resolved, path)


def _pick_branch(
    branches: list[Any], root: dict[str, Any], depth: int
) -> dict[str, Any]:
    """Choose a union member, preferring a non-null one.

    Past the depth cap the order flips: a nullable recursive field terminates on
    ``null`` instead of nesting another level.
    """
    resolved: list[dict[str, Any]] = []
    for branch in branches:
        branch_dict = as_dict(branch)
        if branch_dict is not None:
            resolved.append(_resolve(branch_dict, root))
    if not resolved:
        return {}

    if depth >= MAX_DEPTH:
        nulls = [b for b in resolved if _infer_type(b) == "null"]
        if nulls:
            return nulls[0]
    non_null = [b for b in resolved if _infer_type(b) != "null"]
    return non_null[0] if non_null else resolved[0]


def _generate_object(
    schema: dict[str, Any], root: dict[str, Any], path: str, depth: int
) -> dict[str, Any]:
    properties = as_dict(schema.get("properties"))
    if properties is None:
        return {}

    # OpenAI structured outputs run in strict mode, where every property is
    # required. Emitting the full property set satisfies both strict and loose
    # schemas, so `required` is only consulted to decide what to drop once the
    # depth cap has been hit.
    declared_required = as_list(schema.get("required"))
    required: set[str] = (
        {name for name in (as_str(item) for item in declared_required) if name}
        if declared_required is not None
        else set(properties)
    )

    result: dict[str, Any] = {}
    for name, subschema in properties.items():
        if depth >= MAX_DEPTH and name not in required:
            continue
        child = as_dict(subschema)
        if child is None:
            continue
        result[name] = _generate(child, root, f"{path}.{name}", depth + 1)
    return result


def _generate_array(
    schema: dict[str, Any], root: dict[str, Any], path: str, depth: int
) -> list[Any]:
    items = as_dict(schema.get("items"))
    min_items = as_int(schema.get("minItems"))
    max_items = as_int(schema.get("maxItems"))

    count = 2
    if min_items is not None:
        count = max(count, min_items)
    if max_items is not None:
        count = min(count, max_items)
    if depth >= MAX_DEPTH:
        count = min_items or 0
    if items is None or count <= 0:
        return []

    return [_generate(items, root, f"{path}[{i}]", depth + 1) for i in range(count)]


def _generate_string(schema: dict[str, Any], path: str) -> str:
    fmt = as_str(schema.get("format"))
    if fmt == "date-time":
        return "2020-01-01T00:00:00Z"
    if fmt == "date":
        return "2020-01-01"
    if fmt == "uuid":
        stem = hashlib.sha256(path.encode()).hexdigest()[:8]
        return f"{stem}-0000-4000-8000-000000000000"
    if fmt in ("uri", "url"):
        return "https://mock.invalid/placeholder"
    if fmt == "email":
        return "placeholder@mock.invalid"

    value = _phrase(path)
    min_length = as_int(schema.get("minLength"))
    max_length = as_int(schema.get("maxLength"))
    if min_length is not None and len(value) < min_length:
        value = value.ljust(min_length, "x")
    if max_length is not None and len(value) > max_length:
        value = value[:max_length]
    return value


def _bounded_int(schema: dict[str, Any], path: str) -> int:
    low = as_int(schema.get("minimum"))
    if (
        low is None
        and (exclusive := as_int(schema.get("exclusiveMinimum"))) is not None
    ):
        low = exclusive + 1
    high = as_int(schema.get("maximum"))
    if (
        high is None
        and (exclusive := as_int(schema.get("exclusiveMaximum"))) is not None
    ):
        high = exclusive - 1

    if low is not None and high is not None:
        span = high - low
        return low + (_seed(path) % (span + 1) if span > 0 else 0)
    if low is not None:
        return low + (_seed(path) % 8)
    if high is not None:
        return high - (_seed(path) % 8)
    return _seed(path) % 100
