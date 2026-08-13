import datetime
import re
from collections.abc import Callable, Sequence
from decimal import Decimal
from logging import getLogger
from typing import Any, TypeVar, get_args
from typing import cast as typing_cast

from sqlalchemy import ColumnElement, Select, and_, case, cast, literal, or_
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import Numeric

from ..exceptions import FilterError
from ..schemas.api import RESOURCE_NAME_PATTERN
from .formatting import ILIKE_ESCAPE_CHAR, escape_ilike_pattern, parse_datetime_iso
from .types import DocumentLevel, VectorSyncState

logger = getLogger(__name__)

# Type variable for SQLAlchemy model classes
T = TypeVar("T")

# Module-level constants for comparison operators
COMPARISON_OPERATORS = {
    "gte",
    "lte",
    "gt",
    "lt",
    "ne",
    "in",
    "contains",
    "icontains",
}

NUMERIC_OPERATORS = {"gte", "lte", "gt", "lt", "ne"}

# JSONB columns keep containment semantics: bare lists are not membership
# sugar, and dict values map to nested-metadata conditions rather than IN/Eq.
JSONB_COLUMNS = ("h_metadata", "configuration", "internal_metadata")

ALLOWED_EXTERNAL_TO_INTERNAL_COLUMN_MAPPING = {
    "id": "name",
    "created_at": "created_at",
    "is_active": "is_active",
    "workspace_id": "workspace_name",
    "session_id": "session_name",
    "peer_id": "peer_name",
    "metadata": "h_metadata",
}

ALLOWED_EXTERNAL_TO_INTERNAL_COLUMN_MAPPING_MESSAGES = {
    "workspace_id": "workspace_name",
    "session_id": "session_name",
    "peer_id": "peer_name",
    "token_count": "token_count",
    "created_at": "created_at",
    "metadata": "h_metadata",
}

ALLOWED_EXTERNAL_TO_INTERNAL_COLUMN_MAPPING_DOCUMENTS = {
    "session_id": "session_name",
    "workspace_id": "workspace_name",
    "observer_id": "observer",
    "observed_id": "observed",
    "metadata": "internal_metadata",
}


MAX_SESSION_ALLOWLIST_ENTRIES = 1000

# Columns whose values come from a closed set. Derived from the Literal types
# themselves, so adding a level (e.g. "abduction") or a sync state updates
# filter validation with no change here — an unlisted value is a 422 rather
# than a filter that silently matches nothing.
ENUM_COLUMN_VALUES: dict[str, frozenset[str]] = {
    "level": frozenset(get_args(DocumentLevel)),
    "sync_state": frozenset(get_args(VectorSyncState)),
}


def _coerce_numeric(op_value: Any) -> float | Decimal:
    """Validate a numeric operand without losing precision or overflowing.

    Integers become Decimal rather than staying int. SQLAlchemy types the bind
    from the operand, so a plain int renders an ``::INTEGER`` cast and anything
    past int4 fails at execute time with "integer out of range" — even when the
    comparison itself is meaningful. Decimal renders no cast, matching what
    float() used to do, but keeps the value exact: float() rounds any int past
    2**53 and would silently compare against a different number.

    bool is narrowed first because it subclasses int; binding it as a boolean
    against a numeric column produces SQL Postgres has no operator for.

    Args:
        op_value: The operand to validate.

    Returns:
        The operand as an exact numeric value that binds without a cast.

    Raises:
        ValueError, TypeError: If the operand is not numeric. Callers convert
            these to FilterError.
    """
    if isinstance(op_value, bool):
        return Decimal(int(op_value))
    if isinstance(op_value, float):
        return op_value
    if isinstance(op_value, int | Decimal):
        return Decimal(op_value)
    try:
        return Decimal(str(op_value))
    except ArithmeticError:
        raise ValueError(f"not a number: {op_value!r}") from None


def _column_python_type(column: Any) -> type | None:
    """Return a column's Python type, or None when it doesn't declare one.

    pgvector's Vector raises NotImplementedError rather than returning a type,
    so this must not be called bare.
    """
    try:
        return typing_cast("type | None", column.type.python_type)
    except (AttributeError, NotImplementedError):
        return None


def _coerce_operand(
    column: Any, column_name: str, value: Any, operator: str = ""
) -> Any:
    """Return ``value`` ready to bind against ``column``, or raise FilterError.

    SQLAlchemy types a bind from the *operand*, not the column, and the psycopg
    dialect renders that type as an explicit cast. So an operand whose type
    doesn't match its column compiles into valid-looking SQL and then fails at
    execute time — ``operator does not exist: text = integer``. Postgres will
    not implicitly bridge these, so the mismatch has to be caught here.

    Every operand passes through this one function, whatever the operator, so
    ``eq``/``ne``/``gt``/``in`` cannot drift apart by construction. Callers
    handle None (a null check) and ``*`` (a wildcard) before calling.

    Args:
        column: SQLAlchemy column object.
        column_name: Internal column name, for error messages.
        value: The operand to coerce.
        operator: The comparison operator, or "" for bare equality.

    Returns:
        The operand, coerced where a lossless coercion exists.

    Raises:
        FilterError: If the operand cannot be bound to this column.
    """
    # JSONB keeps containment semantics: the operand is a JSON document, not a
    # scalar to compare. `jsonb >= 5` and `jsonb @> 'text'` have no operator.
    if isinstance(column.type, JSONB):
        if operator in ("", "contains") and isinstance(value, dict | list):
            return typing_cast("Any", value)
        raise FilterError(
            f"Invalid filter for column '{column_name}': a JSONB column takes an object, optionally under 'contains'"
        )

    python_type = _column_python_type(column)
    if python_type is None:
        raise FilterError(f"Column '{column_name}' cannot be filtered on")

    # contains/icontains build an ILIKE pattern, so the operand is stringified
    # and its own type doesn't matter — but the column must be text, or
    # Postgres has no `~~` operator for it.
    if operator in ("contains", "icontains"):
        if python_type is not str:
            raise FilterError(
                f"Operator '{operator}' requires a text column, but '{column_name}' is {python_type.__name__}"
            )
        return value

    # bool is checked before the numeric branch: it subclasses int, so a boolean
    # column would otherwise have `true` coerced to 1, which Postgres rejects
    # against a boolean column ("operator does not exist: boolean <> integer").
    if python_type is bool:
        if isinstance(value, bool):
            return value
        raise FilterError(
            f"Invalid value for column '{column_name}': expected true or false, got {type(value).__name__}"
        )

    if issubclass(python_type, datetime.datetime | datetime.date):
        if isinstance(value, datetime.datetime | datetime.date):
            return value
        if isinstance(value, str):
            validated = _validate_datetime_string(value)
            if validated is None:
                raise FilterError(f"Invalid datetime value: {value}")
            return validated
        raise FilterError(
            f"Invalid value for column '{column_name}': expected a datetime, got {type(value).__name__}"
        )

    if issubclass(python_type, int | float | Decimal):
        try:
            return _coerce_numeric(value)
        except (TypeError, ValueError):
            raise FilterError(
                f"Invalid numeric value: {value}. Expected a number, got {type(value).__name__}"
            ) from None

    if python_type is str:
        if not isinstance(value, str):
            raise FilterError(
                f"Invalid value for column '{column_name}': expected a string, got {type(value).__name__}"
            )
        allowed = ENUM_COLUMN_VALUES.get(column_name)
        if allowed is not None and value not in allowed:
            raise FilterError(
                f"Invalid value for column '{column_name}': {value!r}. Expected one of {sorted(allowed)}"
            )
        return value

    raise FilterError(f"Column '{column_name}' cannot be filtered on")


def extract_session_allowlist(
    filters: dict[str, Any] | None,
    must_include: str | None = None,
) -> list[str] | None:
    """Parse a recall-path ``filters`` body into a session allowlist.

    The dialectic and representation endpoints accept a constrained subset of
    the filter DSL: only the ``session_id`` key, valued as a single id, a bare
    list of ids, or ``{"in": [...]}``. Unsupported keys or shapes raise
    FilterError (422) rather than being silently ignored — a dropped filter
    on these endpoints would widen recall scope.

    Entries must be well-formed session ids. Wildcards are not part of this
    subset: the DSL treats ``*`` as "match everything" while the non-DSL
    consumers of the allowlist treat it as a literal name, so it is rejected
    rather than meaning two things at once.

    Args:
        filters: The raw ``filters`` body, or None.
        must_include: A session id that must appear in the parsed allowlist —
            used by routes that also accept a top-level ``session_id``, so the
            two can't contradict each other. Ignored when filters is None.

    Returns:
        None when filters is None. An explicit empty list is preserved so
        downstream consumers fail closed.

    Raises:
        FilterError: On an unsupported key or shape, an over-cap list, or a
            ``must_include`` session missing from the allowlist.
    """
    if filters is None:
        return None

    unsupported = set(filters) - {"session_id"}
    if unsupported:
        raise FilterError(
            f"Unsupported filter key(s) for this endpoint: {sorted(unsupported)}. Only 'session_id' is supported."
        )
    if "session_id" not in filters:
        raise FilterError("filters must contain 'session_id'")

    value = filters["session_id"]
    entries: list[Any]
    if isinstance(value, str):
        entries = [value]
    elif isinstance(value, list):
        entries = list(typing_cast(Sequence[Any], value))
    elif (
        isinstance(value, dict)
        and set(typing_cast("dict[str, Any]", value)) == {"in"}
        and isinstance(value["in"], list)
    ):
        entries = list(typing_cast(Sequence[Any], value["in"]))
    else:
        raise FilterError(
            'filters.session_id must be a session id, a list of session ids, or {"in": [...]}'
        )

    if len(entries) > MAX_SESSION_ALLOWLIST_ENTRIES:
        raise FilterError(
            f"filters.session_id supports at most {MAX_SESSION_ALLOWLIST_ENTRIES} sessions per request"
        )

    allowlist: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, str) or not entry:
            raise FilterError("filters.session_id entries must be non-empty strings")
        # Only names a session could actually have. The allowlist reaches
        # queries three ways — direct `IN`, the filter DSL, and a Python
        # membership test — and they don't agree on a value like "*", which the
        # DSL reads as "drop the condition" while the others treat as a literal.
        # Rejecting it here keeps the divergent value away from all three, and
        # matches this endpoint's documented contract (an id, a list of ids, or
        # {"in": [...]}) which never included wildcards.
        if not re.fullmatch(RESOURCE_NAME_PATTERN, entry):
            raise FilterError(
                f"Invalid session id in filters.session_id: {entry!r}. Session ids match {RESOURCE_NAME_PATTERN}"
            )
        if entry not in seen:
            seen.add(entry)
            allowlist.append(entry)

    if must_include is not None and must_include not in seen:
        raise FilterError("session_id must be included in filters.session_id")

    return allowlist


def apply_filter(
    stmt: Select[tuple[T]], model_class: type[T], filters: dict[str, Any] | None = None
) -> Select[tuple[T]]:
    """
    Apply advanced filter to a SQL statement based on filter dictionary.

    Supports logical operators (AND, OR, NOT), comparison operators
    (gte, lte, gt, lt, ne, contains, icontains, in), and wildcard character (*).

    Note that the filter refers to column names from the user perspective:
    that means all `*_name` fields are actually `*_id` fields and `h_metadata`
    is actually `metadata`.

    Examples:
        # Simple filters (backward compatible)
        {"peer_id": "alice", "metadata": {"type": "user"}}

        # Logical operators
        {"AND": [{"peer_id": "alice"}, {"created_at": {"gte": "2024-01-01"}}]}
        {"OR": [{"peer_id": "alice"}, {"peer_id": "bob"}]}
        {"NOT": [{"peer_id": "alice"}]}

        # Comparison operators
        {"created_at": {"gte": "2024-01-01", "lte": "2024-12-31"}}
        {"peer_id": {"in": ["alice", "bob"]}}

        # Wildcards (matches everything for that field)
        {"peer_id": "*"}

    Args:
        stmt: SQLAlchemy Select statement to modify
        model_class: SQLAlchemy model class for column access
        filters: Optional filter dictionary

    Returns:
        Modified Select statement with filter applied if provided

    Raises:
        FilterError: When the filter contains invalid configuration or values
    """
    if filters is None:
        return stmt

    # Fail closed. The filter body is arbitrary client JSON, so any shape the
    # DSL doesn't recognize must become a 422, not an unhandled 500 from
    # somewhere deep in SQLAlchemy. The exception is still logged in full so a
    # genuine bug in the builder stays visible rather than being swallowed.
    try:
        conditions = _build_filter_conditions(filters, model_class)
        if conditions is not None:
            stmt = stmt.where(conditions)
    except FilterError:
        raise
    except Exception:
        # Keys only, not the body: filter operands are client-supplied and carry
        # peer/session ids and free-text `contains` values. The traceback plus
        # the entry shape is what actually locates a builder bug.
        logger.exception(
            "Unexpected error building filter for %s; filter keys: %s",
            model_class.__name__,
            sorted(filters),
        )
        raise FilterError("Invalid filter configuration") from None

    return stmt


def _build_filter_conditions(
    filter_dict: dict[str, Any], model_class: type[Any], *, _depth: int = 0
) -> ColumnElement[bool] | None:
    """
    Recursively build filter conditions from a filter dictionary.

    Args:
        filter_dict: Filter dictionary that may contain logical operators
        model_class: SQLAlchemy model class for column access

    Returns:
        SQLAlchemy condition object or None
    """
    if _depth > 5:
        raise FilterError("Filter nesting exceeds maximum depth of 5")

    conditions: list[ColumnElement[bool]] = []

    # Handle logical operators
    if "AND" in filter_dict:
        if not isinstance(filter_dict["AND"], list):
            raise FilterError(
                f"AND operator must contain a list, got {type(filter_dict['AND']).__name__}"
            )
        and_conditions: list[ColumnElement[bool]] = []
        for sub_filter in filter_dict["AND"]:  # pyright: ignore
            sub_condition = _build_filter_conditions(
                sub_filter,  # pyright: ignore[reportUnknownArgumentType]
                model_class,
                _depth=_depth + 1,
            )
            if sub_condition is not None:
                and_conditions.append(sub_condition)
        if and_conditions:
            conditions.append(and_(*and_conditions))

    if "OR" in filter_dict:
        if not isinstance(filter_dict["OR"], list):
            raise FilterError(
                f"OR operator must contain a list, got {type(filter_dict['OR']).__name__}"
            )
        or_conditions: list[ColumnElement[bool]] = []
        for sub_filter in filter_dict["OR"]:  # pyright: ignore
            sub_condition = _build_filter_conditions(
                sub_filter,  # pyright: ignore[reportUnknownArgumentType]
                model_class,
                _depth=_depth + 1,
            )
            if sub_condition is not None:
                or_conditions.append(sub_condition)
        if or_conditions:
            conditions.append(or_(*or_conditions))

    if "NOT" in filter_dict:
        if filter_dict["NOT"] is None:
            raise FilterError("NOT operator cannot be None")
        if not isinstance(filter_dict["NOT"], list):
            raise FilterError(
                f"NOT operator must contain a list, got {type(filter_dict['NOT']).__name__}"
            )
        not_conditions: list[ColumnElement[bool]] = []
        for sub_filter in filter_dict["NOT"]:  # pyright: ignore
            sub_condition = _build_filter_conditions(
                sub_filter,  # pyright: ignore[reportUnknownArgumentType]
                model_class,
                _depth=_depth + 1,
            )
            if sub_condition is not None:
                # `IS NOT TRUE` rather than `NOT`: under SQL's three-valued
                # logic a comparison against a NULL column is NULL, and `NOT
                # NULL` is NULL, so plain negation drops rows whose column is
                # unset — even though an unset column does not match what is
                # being excluded. NOT [{"session_id": "abc"}] must include
                # documents that have no session, since those are not "abc".
                # Composes over compound sub-conditions: (a AND b) IS NOT TRUE.
                not_conditions.append(
                    sub_condition.is_not(True)
                )  # Apply NOT to each condition individually
        if not_conditions:
            conditions.append(and_(*not_conditions))  # Then AND them together

    # Handle field-level conditions (skip logical operator keys)
    logical_keys = {"AND", "OR", "NOT"}
    for key, value in filter_dict.items():
        if key in logical_keys:
            continue

        condition = _build_field_condition(key, value, model_class)
        if condition is not None:
            conditions.append(condition)

    # Combine all conditions with AND
    if len(conditions) == 0:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return and_(*conditions)


def _build_field_condition(
    key: str, value: Any, model_class: type[Any]
) -> ColumnElement[bool] | None:
    """
    Build a condition for a single field.

    Args:
        key: Field name
        value: Field value or comparison dict
        model_class: SQLAlchemy model class

    Returns:
        SQLAlchemy condition object or None
    """
    if model_class.__name__ == "Message":
        column_name = ALLOWED_EXTERNAL_TO_INTERNAL_COLUMN_MAPPING_MESSAGES.get(key)
    elif model_class.__name__ == "Document":
        # NOTE: unlike Message/Workspace, Document falls back to the raw key so
        # internal callers can filter on internal column names. The session
        # allowlist depends on this: recall passes {"session_name": {"in": ...}}
        # (see search_memory in utils/agent_tools.py and RepresentationManager),
        # and "session_name" is deliberately absent from the mapping below.
        # Tightening this to a strict allowlist would break session scoping —
        # fail-closed, since an unmapped key raises, but silently.
        column_name = ALLOWED_EXTERNAL_TO_INTERNAL_COLUMN_MAPPING_DOCUMENTS.get(
            key,
            key,  # fallback to the key itself if not found in the mapping for internal use here
        )
    else:
        column_name = ALLOWED_EXTERNAL_TO_INTERNAL_COLUMN_MAPPING.get(key)

    if column_name is None:
        raise FilterError(
            f"Column '{key}' is not allowed to be filtered on or does not exist on {model_class.__name__}"
        )

    # Check if the column exists on the model
    if not hasattr(model_class, column_name):
        raise FilterError(f"Column '{key}' does not exist on {model_class.__name__}")

    column = getattr(model_class, column_name)

    # Handle wildcard - matches everything, so no condition needed
    if value == "*":
        return None

    # A null operand is a null check, not a value to compare. Every branch below
    # binds the operand against the column's type, and no type accepts None, so
    # this has to short-circuit or `{"col": null}` raises instead of matching the
    # rows it names. Keeps bare null agreeing with `{"ne": null}` (IS NOT NULL)
    # and with `NOT [{"col": null}]`.
    if value is None:
        return column.is_(None)

    # Bare-list sugar on regular columns: {"session_id": ["a", "b"]} is
    # shorthand for {"session_id": {"in": ["a", "b"]}}. JSONB columns are
    # excluded — a bare list there keeps JSONB containment semantics.
    if isinstance(value, list | tuple | set) and column_name not in JSONB_COLUMNS:
        value = {"in": list(typing_cast(Sequence[Any], value))}

    # Handle comparison operators vs regular values
    if isinstance(value, dict):
        # Check if this is a comparison operators dict by looking for known operators
        is_comparison_dict = any(op_key in COMPARISON_OPERATORS for op_key in value)  # pyright: ignore

        if is_comparison_dict:
            return _build_comparison_conditions(column, column_name, value)  # pyright: ignore
        else:
            # This is a regular value that happens to be a dict
            # For JSONB fields (metadata, configuration), check if it contains nested comparison operators
            if column_name in JSONB_COLUMNS:
                return _build_nested_metadata_conditions(column, value)  # pyright: ignore
            elif not isinstance(column.type, JSONB):
                # A dict against a scalar column compiles fine but fails in
                # psycopg at execute time ("cannot adapt type 'dict'") as a 500.
                # Reject unknown operator dicts here as a 422 instead.
                keys = sorted(typing_cast("dict[str, Any]", value))
                raise FilterError(
                    f"Invalid filter for column '{key}': unsupported operator(s) {keys}. Expected one of {sorted(COMPARISON_OPERATORS)} or a scalar value."
                )
            else:
                return column == value
    else:
        if column_name in JSONB_COLUMNS:
            return column.contains(_coerce_operand(column, column_name, value))
        else:
            return column == _coerce_operand(column, column_name, value)


def _safe_numeric_cast(
    column_accessor: ColumnElement[Any], op_value: Any
) -> tuple[ColumnElement[Any], Any]:
    """
    Safely cast JSONB column accessor to appropriate type for comparison.

    Args:
        column_accessor: SQLAlchemy JSONB column accessor (.astext)
        op_value: The value to compare against

    Returns:
        Tuple of (cast_column_accessor, cast_op_value) for typed comparison
        or (column_accessor, str_op_value) for string comparison
    """
    try:
        if isinstance(op_value, bool):
            # For boolean values, compare with the string representation
            # PostgreSQL JSONB stores booleans as "true"/"false" strings when extracted with ->>
            return column_accessor, str(op_value).lower()

        # For numeric values, use a safer cast that handles empty strings and invalid values
        # We use CASE WHEN to handle empty strings and non-numeric values gracefully
        safe_cast = case(
            (column_accessor == "", literal(None)),  # Empty string -> NULL
            (column_accessor.is_(None), literal(None)),  # NULL -> NULL
            else_=cast(column_accessor, Numeric()),
        )

        if isinstance(op_value, int | float):
            return safe_cast, op_value
        else:
            # Try to parse as numeric (handles both strings and other types)
            try:
                # Try int first, then float
                parsed_value = int(op_value)
                return safe_cast, parsed_value
            except (ValueError, TypeError):
                try:
                    parsed_value = float(op_value)
                    return safe_cast, parsed_value
                except (ValueError, TypeError):
                    if isinstance(op_value, str):
                        # If it's not numeric, treat as string comparison (e.g., dates, text)
                        # This allows date strings like "2024-02-01" to be compared lexicographically
                        return column_accessor, str(op_value)
                    else:
                        raise FilterError(
                            f"Invalid value for numeric operator: {op_value}. Expected a number, got {type(op_value).__name__}"
                        ) from None
    except Exception as e:
        raise FilterError(
            f"Failed to process numeric cast for value '{op_value}': {str(e)}"
        ) from e


def _build_comparison_condition(
    column: Any, field_name: str, operator: str, op_value: Any
) -> ColumnElement[bool] | None:
    """
    Build a single comparison condition for a JSONB field.

    Args:
        column: SQLAlchemy JSONB column object
        field_name: Name of the field in the JSONB column
        operator: Comparison operator
        op_value: Value to compare against

    Returns:
        SQLAlchemy condition object or None
    """
    # Validate that the operator is supported
    if operator not in COMPARISON_OPERATORS:
        raise FilterError(f"Unsupported comparison operator: {operator}")

    # Handle wildcard - matches everything, so no condition needed
    if op_value == "*":
        return None

    field_accessor = column[field_name].astext

    # Mapping of operators to their SQLAlchemy methods
    if operator in NUMERIC_OPERATORS:
        try:
            safe_accessor, safe_value = _safe_numeric_cast(field_accessor, op_value)
            operator_map: dict[str, Callable[[Any, Any], ColumnElement[bool]]] = {
                "gte": lambda a, v: a >= v,
                "lte": lambda a, v: a <= v,
                "gt": lambda a, v: a > v,
                "lt": lambda a, v: a < v,
                "ne": lambda a, v: a != v,
            }
            return operator_map[operator](safe_accessor, safe_value)
        except Exception as e:
            raise FilterError(
                f"Failed to build numeric comparison condition for operator '{operator}' with value '{op_value}': {str(e)}"
            ) from e
    elif operator == "in":
        if hasattr(op_value, "__iter__") and not isinstance(op_value, str | bytes):
            # Handle wildcard in iterable - if present, matches everything, so no condition needed
            if "*" in op_value:
                return None
            return field_accessor.in_([str(v) for v in op_value])
        else:
            raise FilterError(
                f"Invalid value for 'in' operator: {op_value}. Expected an iterable (list, tuple, set), got {type(op_value).__name__}"
            )
    elif operator in ("contains", "icontains"):
        escaped_value = escape_ilike_pattern(str(op_value))
        return field_accessor.ilike(f"%{escaped_value}%", escape=ILIKE_ESCAPE_CHAR)

    return None


def _build_nested_metadata_conditions(
    column: Any, metadata_dict: dict[str, Any]
) -> ColumnElement[bool] | None:
    """
    Build conditions for nested metadata fields with comparison operators.

    Args:
        column: SQLAlchemy JSONB column object
        metadata_dict: Dictionary containing nested field conditions

    Returns:
        Combined SQLAlchemy condition object or None
    """
    conditions: list[ColumnElement[bool]] = []

    for field_name, field_value in metadata_dict.items():
        if isinstance(field_value, dict) and any(
            op in COMPARISON_OPERATORS
            for op in field_value  # pyright: ignore
        ):
            # This field has comparison operators
            field_conditions: list[ColumnElement[bool]] = []
            for operator, op_value in field_value.items():  # pyright: ignore
                condition = _build_comparison_condition(
                    column,
                    field_name,
                    operator,  # pyright: ignore
                    op_value,
                )
                if condition is not None:
                    field_conditions.append(condition)

            if field_conditions:
                conditions.append(
                    field_conditions[0]
                    if len(field_conditions) == 1
                    else and_(*field_conditions)
                )
        else:
            # Handle wildcard - matches everything, so no condition needed
            if field_value == "*":
                continue
            # Regular field equality - use JSONB contains for nested object matching
            conditions.append(column.contains({field_name: field_value}))

    # Combine all field conditions with AND
    return _combine_conditions_with_and(conditions)


def _combine_conditions_with_and(
    conditions: list[ColumnElement[bool]],
) -> ColumnElement[bool] | None:
    """
    Combine a list of conditions with AND logic.

    Args:
        conditions: List of SQLAlchemy condition objects

    Returns:
        Combined condition object or None if no conditions
    """
    if not conditions:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return and_(*conditions)


def _build_comparison_conditions(
    column: Any, column_name: str, comparisons: dict[str, Any]
) -> ColumnElement[bool] | None:
    """
    Build comparison conditions for a single column.

    Args:
        column: SQLAlchemy column object
        column_name: Name of the column
        comparisons: Dictionary of comparison operators and values

    Returns:
        Combined SQLAlchemy condition object or None
    """
    conditions: list[ColumnElement[bool]] = []

    for operator, op_value in comparisons.items():
        # Validate that the operator is supported
        if operator not in COMPARISON_OPERATORS:
            raise FilterError(f"Unsupported comparison operator: {operator}")

        # Handle wildcard - matches everything, so no condition needed
        if op_value == "*":
            continue

        # A null operand is a null check, not a value comparison, on every
        # column type. Only `ne` is meaningful: {"col": None} covers IS NULL
        # via the null guard in _build_field_condition.
        if op_value is None:
            if operator != "ne":
                raise FilterError(
                    f"Operator '{operator}' does not accept null. Use {{\"ne\": null}} for a not-null check, or null on its own for a null check."
                )
            conditions.append(column.is_not(None))
            continue

        condition = None

        # `in` coerces element-wise below; every other operator has one operand.
        casted_value = (
            op_value
            if operator == "in"
            else _coerce_operand(column, column_name, op_value, operator)
        )

        if operator == "gte":
            condition = column >= casted_value
        elif operator == "lte":
            condition = column <= casted_value
        elif operator == "gt":
            condition = column > casted_value
        elif operator == "lt":
            condition = column < casted_value
        elif operator == "ne":
            # IS DISTINCT FROM, not <>: `NULL <> 'abc'` is NULL, so plain
            # inequality drops rows whose column is unset. Identical to <>
            # whenever no NULL is involved, and keeps `ne` agreeing with the
            # NOT operator instead of quietly returning a different row set.
            condition = column.is_distinct_from(casted_value)
        elif operator == "in":
            if hasattr(op_value, "__iter__") and not isinstance(op_value, str | bytes):
                # Handle wildcard in iterable - if present, matches everything, so no condition needed
                if "*" in op_value:
                    continue
                else:
                    # Element-wise: one bad element poisons the whole IN, since
                    # its type decides the cast rendered for that parameter.
                    # An empty list is applied, not skipped: `in: []` must match
                    # nothing. Dropping the condition would widen the query to
                    # every row, and session scoping relies on an empty
                    # allowlist failing closed (see extract_session_allowlist).
                    condition = column.in_(
                        [
                            _coerce_operand(column, column_name, val, operator)
                            for val in op_value
                        ]
                    )
            else:
                raise FilterError(
                    f"Invalid value for 'in' operator: {op_value}. Expected an iterable (list, tuple, set), got {type(op_value).__name__}"
                )
        elif operator == "contains":
            if isinstance(column.type, JSONB):
                # Keyed on the column type, not the name: internal_metadata is
                # equally JSONB and was falling through to ILIKE, which
                # Postgres rejects as `jsonb ~~* text`.
                condition = column.contains(casted_value)
            else:
                # For text columns, use ILIKE with escaped pattern
                escaped_value = escape_ilike_pattern(str(op_value))
                condition = column.ilike(f"%{escaped_value}%", escape=ILIKE_ESCAPE_CHAR)
        elif operator == "icontains":
            # Case-insensitive contains for text columns with escaped pattern
            escaped_value = escape_ilike_pattern(str(op_value))
            condition = column.ilike(f"%{escaped_value}%", escape=ILIKE_ESCAPE_CHAR)

        if condition is not None:
            conditions.append(condition)

    # Combine all conditions for this field with AND
    if len(conditions) == 0:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return and_(*conditions)


def _validate_datetime_string(value: str) -> datetime.datetime | None:
    """
    Safely validate and parse a datetime string to prevent SQL injection.

    This function prioritizes timezone-aware datetime formats and uses the
    consistent parse_datetime_iso utility for proper timezone handling.

    Args:
        value: String value to validate as datetime

    Returns:
        Parsed datetime object if valid, None if invalid
    """
    # Strip whitespace
    value = value.strip()

    # First try the standard ISO format with timezone info using our utility
    try:
        return parse_datetime_iso(value)
    except ValueError:
        pass

    # Fallback to naive formats (assume UTC timezone for compatibility)
    naive_formats = [
        "%Y-%m-%dT%H:%M:%S",  # 2024-01-01T12:00:00 (ISO format, assume UTC)
        "%Y-%m-%dT%H:%M:%S.%f",  # 2024-01-01T12:00:00.123456 (assume UTC)
        "%Y-%m-%d %H:%M:%S",  # 2024-01-01 12:00:00 (assume UTC)
        "%Y-%m-%d %H:%M:%S.%f",  # 2024-01-01 12:00:00.123456 (assume UTC)
        "%Y-%m-%d",  # 2024-01-01 (assume UTC, start of day)
    ]

    for fmt in naive_formats:
        try:
            parsed = datetime.datetime.strptime(value, fmt)
            # Assume UTC timezone for naive datetimes
            return parsed.replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue

    # Return None for invalid datetime - let the caller handle the error
    return None
