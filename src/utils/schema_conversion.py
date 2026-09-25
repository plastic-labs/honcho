"""Convert caller-supplied JSON Schema objects into dynamic Pydantic models.

Used by the dialectic chat endpoint's ``response_format`` option: the caller
sends a JSON Schema dict, and the resulting model is passed as
``response_model`` to ``honcho_llm_call()`` so providers return conforming
JSON.

Only a conservative subset of JSON Schema is supported (see
``json_response_schema_to_pydantic``). Conversion doubles as validation: any
unsupported construct raises ``ValueError`` with the offending path, which the
router surfaces as a 422.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Literal, NoReturn, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

# "$defs"/"definitions" are extracted at the root before the walk and are
# rejected anywhere else. "$ref" nodes are resolved by _resolve_ref before
# node validation, so they never reach this check.
_UNSUPPORTED_KEYS = (
    "$defs",
    "definitions",
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "patternProperties",
)

_REF_PREFIXES = ("#/$defs/", "#/definitions/")

_PRIMITIVE_TYPES: dict[str, Any] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "null": type(None),
}

# Constraint keywords are forwarded to the model's json_schema_extra so the
# LLM sees them, but Pydantic does not enforce them (they are hints only).
_HINT_KEYS = (
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "pattern",
    "format",
    "minProperties",
    "maxProperties",
    "uniqueItems",
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass
class _Ctx:
    """Mutable state shared across one conversion walk."""

    max_depth: int
    max_nodes: int
    defs: dict[str, Any] = field(default_factory=dict)
    used_names: set[str] = field(default_factory=set)
    ref_stack: list[str] = field(default_factory=list)
    node_count: int = 0


def json_response_schema_to_pydantic(
    schema: dict[str, Any],
    *,
    model_name: str = "ResponseFormat",
    max_depth: int = 20,
    max_nodes: int = 500,
) -> type[BaseModel]:
    """Convert a JSON Schema dict (root type ``object``) into a Pydantic model.

    Supported constructs: primitive types (``string``/``number``/``integer``/
    ``boolean``/``null``), nested ``object`` with ``properties``, ``array``
    with ``items`` (missing ``items`` yields ``list[Any]``), ``enum`` of
    strings/integers/booleans/null, ``anyOf``/``oneOf`` unions (a
    ``{"type": "null"}`` member yields an optional), ``type`` given as a list,
    ``required``, ``default``, and ``description``. A root-level ``$schema``
    key is ignored. Boolean ``additionalProperties`` is accepted and ignored;
    extra keys in LLM output are silently dropped (``extra="ignore"``).

    ``$ref`` is supported for references of the form ``#/$defs/<name>`` or
    ``#/definitions/<name>`` into root-level ``$defs``/``definitions`` (this
    is what Pydantic's ``model_json_schema()`` and Zod's ``toJSONSchema``
    emit). References are resolved by inlining; sibling keys next to ``$ref``
    overlay the referenced definition (siblings win). Recursive references
    are rejected — the error names the cycle. Unreferenced definitions are
    ignored without validation.

    Constraint keywords (``minItems``, ``maxLength``, ``minimum``,
    ``pattern``, ...) are passed through to the generated schema as hints but
    are not enforced by Pydantic.

    Args:
        schema: The JSON Schema object. Root must resolve to type ``object``.
        model_name: Name for the generated root model class.
        max_depth: Maximum nesting depth. Guard against excessive or
            malicious schemas (e.g. pathologically deep nesting); exceeding
            it raises ``ValueError``.
        max_nodes: Maximum total nodes visited across the whole schema.
            Guard against excessive or malicious schemas (e.g. enormous
            property fan-out); exceeding it raises ``ValueError``.

    Returns:
        A dynamically created Pydantic model class.

    Raises:
        ValueError: If the schema is malformed or uses an unsupported
            construct (``allOf``, ``not``, ``if``/``then``/``else``,
            ``patternProperties``, schema-valued ``additionalProperties``,
            boolean schemas, unknown types, a non-object root, a ``$ref``
            that is recursive, malformed, or targets an unknown definition,
            or ``$defs``/``definitions`` anywhere but the root). The message
            names the construct and its path.
    """
    schema_obj: Any = schema
    if not isinstance(schema_obj, dict):
        raise ValueError("response_format must be a JSON Schema object")

    root = {k: v for k, v in schema.items() if k != "$schema"}
    defs = _extract_defs(root)
    root_type = root.get("type")
    # A "$ref" root is allowed through here; the post-conversion check below
    # still enforces that it resolves to an object.
    is_object_root = root_type == "object" or (
        root_type is None and ("properties" in root or "$ref" in root)
    )
    if not is_object_root:
        raise ValueError("root schema must have type 'object'")

    ctx = _Ctx(max_depth=max_depth, max_nodes=max_nodes, defs=defs)
    annotation = _convert_schema(root, "", model_name, ctx, depth=0)
    # An object root always converts to a model class; this is a safety net.
    if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
        raise ValueError("root schema must have type 'object'")
    return annotation


def _fail(msg: str, path: str) -> NoReturn:
    raise ValueError(f"{msg} at {path or 'root'}")


def _union(members: tuple[Any, ...]) -> Any:
    """Build ``A | B | ...`` from a dynamic tuple of annotations."""
    result: Any = members[0]
    for member in members[1:]:
        result = result | member
    return result


def _convert_schema(
    raw_node: Any, path: str, name_hint: str, ctx: _Ctx, depth: int
) -> Any:
    """Convert one schema node into a type annotation.

    Dispatch order matters: $ref resolution comes first (a ref node is
    replaced by its target before anything else looks at it), then enum (a
    value constraint) wins over unions, which win over "type"-based
    conversion.
    """
    if isinstance(raw_node, dict) and "$ref" in raw_node:
        return _resolve_ref(cast(dict[str, Any], raw_node), path, ctx, depth)

    node = _validate_node(raw_node, path, ctx, depth)

    if "enum" in node:
        return _convert_enum(node["enum"], path)

    if "anyOf" in node or "oneOf" in node:
        return _convert_union(node, path, name_hint, ctx, depth)

    node_type = node.get("type")
    if isinstance(node_type, list):
        return _convert_type_list(
            node, cast(list[Any], node_type), path, name_hint, ctx, depth
        )

    # Tolerate an omitted "type" when "properties" makes the intent clear.
    if node_type is None and "properties" in node:
        node_type = "object"

    if node_type == "object":
        return _build_object_model(node, path, name_hint, ctx, depth)

    if node_type == "array":
        return _convert_array(node, path, name_hint, ctx, depth)

    if node_type in _PRIMITIVE_TYPES:
        return _PRIMITIVE_TYPES[node_type]

    if node_type is None:
        _fail("schema has no recognizable type", path)
    _fail(f"unsupported type '{node_type}'", path)


def _extract_defs(root: dict[str, Any]) -> dict[str, Any]:
    """Pop root-level ``$defs``/``definitions`` and merge them into one
    registry. Entries are validated lazily, when (and only when) referenced."""
    defs: dict[str, Any] = {}
    for key in ("$defs", "definitions"):
        raw = root.pop(key, None)
        if raw is None:
            continue
        if not isinstance(raw, dict):
            _fail(f"'{key}' must be an object", "")
        for name, definition in cast(dict[Any, Any], raw).items():
            if not isinstance(name, str) or not name:
                _fail(f"'{key}' definition names must be non-empty strings", "")
            if name in defs:
                _fail(
                    f"definition '{name}' appears in both '$defs' and 'definitions'",
                    "",
                )
            defs[name] = definition
    return defs


def _resolve_ref(node: dict[str, Any], path: str, ctx: _Ctx, depth: int) -> Any:
    """Inline a ``$ref`` node: resolve the target definition, overlay any
    sibling keys (siblings win), and convert the result in place.

    Only root-relative refs into ``$defs``/``definitions`` are supported.
    Cycles are rejected — recursion cannot be inlined. The resolved node is
    converted at the same depth (replacement semantics); the node budget in
    ``_validate_node`` still counts every expansion, so a definition that is
    referenced many times cannot blow up the walk.
    """
    ref: Any = node["$ref"]
    if not isinstance(ref, str):
        _fail("'$ref' must be a string", path)
    name: str | None = None
    for prefix in _REF_PREFIXES:
        if ref.startswith(prefix):
            name = ref[len(prefix) :]
            break
    # "/", "~", and "%" would make the remainder a deeper or escaped JSON
    # pointer (e.g. "#/$defs/a/b", "~1" escapes, %-encoding) rather than a
    # plain definition name, so their presence means an unsupported form.
    if not name or "/" in name or "~" in name or "%" in name:
        _fail(
            f"unsupported $ref '{ref}': only '#/$defs/<name>' or "
            + "'#/definitions/<name>' references are supported",
            path,
        )
    if name not in ctx.defs:
        _fail(f"$ref '{ref}' points to an unknown definition", path)
    # ref_stack holds the definitions currently being expanded on this branch
    # of the walk, so membership means the definition (transitively)
    # references itself. Slicing from the first occurrence yields the cycle
    # for the error message, e.g. stack [A, B] + name A -> "A -> B -> A".
    if name in ctx.ref_stack:
        cycle = " -> ".join([*ctx.ref_stack[ctx.ref_stack.index(name) :], name])
        _fail(
            f"recursive $ref is not supported (cycle: {cycle}); "
            + "restructure the schema so definitions do not reference themselves",
            path,
        )
    target: Any = ctx.defs[name]
    siblings = {k: v for k, v in node.items() if k != "$ref"}
    resolved: Any = target
    if siblings and isinstance(target, dict):
        resolved = {**cast(dict[str, Any], target), **siblings}
    # Pop after converting (not just on success) so the stack tracks only the
    # current branch: a diamond — the same definition referenced from two
    # sibling nodes — is legitimate reuse, not a cycle.
    ctx.ref_stack.append(name)
    try:
        return _convert_schema(resolved, path, name, ctx, depth)
    finally:
        ctx.ref_stack.pop()


def _validate_node(raw_node: Any, path: str, ctx: _Ctx, depth: int) -> dict[str, Any]:
    """Enforce size budgets and node shape; reject unsupported constructs."""
    # node_count is cumulative across the whole walk; depth tracks only the
    # current branch.
    ctx.node_count += 1
    if ctx.node_count > ctx.max_nodes:
        raise ValueError(f"schema exceeds the maximum of {ctx.max_nodes} nodes")
    if depth > ctx.max_depth:
        raise ValueError(f"schema nesting exceeds the maximum depth of {ctx.max_depth}")
    if isinstance(raw_node, bool):
        # A special case of the object requirement, with its own message:
        # boolean schemas are legal JSON Schema, just deliberately unsupported.
        _fail("boolean schemas are not supported", path)
    if not isinstance(raw_node, dict):
        _fail("schema must be an object", path)
    node = cast(dict[str, Any], raw_node)
    for key in _UNSUPPORTED_KEYS:
        if key in node:
            _fail(f"unsupported construct '{key}'", path)
    if isinstance(node.get("additionalProperties"), dict):
        _fail("additionalProperties with a schema is not supported", path)
    return node


def _convert_union(
    node: dict[str, Any], path: str, name_hint: str, ctx: _Ctx, depth: int
) -> Any:
    """Convert anyOf/oneOf, treated identically: a plain union of the member
    schemas (a {"type": "null"} member makes the union optional)."""
    union_key = "anyOf" if "anyOf" in node else "oneOf"
    members = node[union_key]
    if not isinstance(members, list) or not members:
        _fail(f"'{union_key}' must be a non-empty array", path)
    converted = tuple(
        _convert_schema(
            member,
            _child_path(path, f"{union_key}[{i}]"),
            f"{name_hint}Option{i}",
            ctx,
            depth + 1,
        )
        for i, member in enumerate(cast(list[Any], members))
    )
    return _union(converted)


def _convert_type_list(
    node: dict[str, Any],
    types: list[Any],
    path: str,
    name_hint: str,
    ctx: _Ctx,
    depth: int,
) -> Any:
    """Convert the type: ["string", "null"] sugar — re-convert the node once
    per entry (keeping sibling keys, at the same depth since it's the same
    source node) and union the results."""
    if not types:
        _fail("'type' array must not be empty", path)
    variants = tuple(
        _convert_schema(
            {**{k: v for k, v in node.items() if k != "type"}, "type": t},
            path,
            name_hint,
            ctx,
            depth,
        )
        for t in types
    )
    return _union(variants)


def _convert_array(
    node: dict[str, Any], path: str, name_hint: str, ctx: _Ctx, depth: int
) -> Any:
    """Convert an array schema into a list annotation."""
    items = node.get("items")
    # A missing "items" constraint means any element type is allowed.
    if items is None:
        return list[Any]
    item_annotation = _convert_schema(
        items, _child_path(path, "items"), f"{name_hint}Item", ctx, depth + 1
    )
    return list[item_annotation]


def _convert_enum(raw_values: Any, path: str) -> Any:
    if not isinstance(raw_values, list) or not raw_values:
        _fail("'enum' must be a non-empty array", path)
    literal_values: list[Any] = []
    # None is not Literal-legal, so collect it separately and union NoneType
    # back in at the end (an all-null enum degenerates to NoneType).
    has_null = False
    for value in cast(list[Any], raw_values):
        if value is None:
            has_null = True
        elif isinstance(value, str | int | bool):
            literal_values.append(value)
        else:
            _fail("enum values must be strings, integers, booleans, or null", path)
    if not literal_values:
        return type(None)
    annotation: Any = Literal[tuple(literal_values)]
    return _union((annotation, type(None))) if has_null else annotation


def _build_object_model(
    node: dict[str, Any], path: str, name_hint: str, ctx: _Ctx, depth: int
) -> type[BaseModel]:
    raw_properties: Any = node.get("properties", {})
    if not isinstance(raw_properties, dict):
        _fail("'properties' must be an object", path)
    properties = cast(dict[Any, Any], raw_properties)
    raw_required: Any = node.get("required", [])
    if not isinstance(raw_required, list) or not all(
        isinstance(entry, str) for entry in cast(list[Any], raw_required)
    ):
        _fail("'required' must be an array of strings", path)
    # Entries naming properties that don't exist are tolerated; they just
    # have no effect.
    required = set(cast(list[str], raw_required))

    fields: dict[str, tuple[Any, Any]] = {}
    for prop_key, prop_schema in properties.items():
        if not isinstance(prop_key, str) or not prop_key:
            _fail("property names must be non-empty strings", path)
        annotation = _convert_schema(
            prop_schema,
            _child_path(path, f"properties.{prop_key}"),
            f"{name_hint} {prop_key}",
            ctx,
            depth + 1,
        )
        # Non-identifier keys ("my-key") get a sanitized field name, with the
        # original key preserved as the alias for validation/serialization.
        field_name = _field_name(prop_key, fields)
        alias = prop_key if field_name != prop_key else None
        fields[field_name] = _make_field(
            cast(dict[str, Any], prop_schema) if isinstance(prop_schema, dict) else {},
            annotation,
            is_required=prop_key in required,
            alias=alias,
        )

    # create_model's overloads can't type dynamic **fields; the values are
    # (annotation, FieldInfo) tuples, which is the documented calling form.
    model: Any = create_model(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
        _unique_model_name(name_hint, ctx),
        __config__=ConfigDict(extra="ignore", populate_by_name=True),
        **fields,  # pyright: ignore[reportArgumentType]
    )
    return cast(type[BaseModel], model)


def _make_field(
    prop_schema: dict[str, Any],
    annotation: Any,
    *,
    is_required: bool,
    alias: str | None,
) -> tuple[Any, Any]:
    kwargs: dict[str, Any] = {}
    if alias is not None:
        kwargs["alias"] = alias
    description = prop_schema.get("description")
    if isinstance(description, str):
        kwargs["description"] = description
    hints = {key: prop_schema[key] for key in _HINT_KEYS if key in prop_schema}
    if hints:
        kwargs["json_schema_extra"] = hints

    # Precedence: an explicit default wins (even for required fields), then
    # required, then optional — which widens to `T | None` defaulting to None.
    if "default" in prop_schema:
        return annotation, Field(default=prop_schema["default"], **kwargs)
    if is_required:
        return annotation, Field(**kwargs)
    return annotation | None, Field(default=None, **kwargs)


def _child_path(path: str, segment: str) -> str:
    return f"{path}.{segment}" if path else segment


def _field_name(prop_key: str, existing: dict[str, Any]) -> str:
    """Return a valid, unique Python field name for a JSON property key.

    Keys that aren't valid identifiers (or would be Pydantic-private via a
    leading underscore) are sanitized; the original key is preserved as the
    field alias by the caller.
    """
    if _IDENTIFIER_RE.match(prop_key) and prop_key not in existing:
        return prop_key
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", prop_key).lstrip("_")
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"field_{sanitized}"
    candidate = sanitized
    suffix = 2
    while candidate in existing:
        candidate = f"{sanitized}_{suffix}"
        suffix += 1
    return candidate


def _unique_model_name(name_hint: str, ctx: _Ctx) -> str:
    # PascalCase the hint (e.g. "ResponseFormat address geo" -> "ResponseFormatAddressGeo").
    parts = re.split(r"[^A-Za-z0-9]+", name_hint)
    name = "".join(part[:1].upper() + part[1:] for part in parts if part)
    # Class names can't be empty or start with a digit.
    if not name or name[0].isdigit():
        name = f"Model{name}"
    # Distinct hints can sanitize to the same name; suffix _2, _3, ... to disambiguate.
    candidate = name
    suffix = 2
    while candidate in ctx.used_names:
        candidate = f"{name}_{suffix}"
        suffix += 1
    ctx.used_names.add(candidate)
    return candidate
