import datetime
from logging import getLogger
from typing import Any

from sqlalchemy import Select, and_, cast, not_, or_
from sqlalchemy.types import Numeric

from ..exceptions import FilterError

logger = getLogger(__name__)

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

EXTERNAL_TO_INTERNAL_COLUMN_MAPPING = {
    "metadata": "h_metadata",
    "id": "name",
    "workspace_id": "workspace_name",
    "session_id": "session_name",
    "peer_id": "peer_name",
}

EXTERNAL_TO_INTERNAL_COLUMN_MAPPING_MESSAGES = {
    "metadata": "h_metadata",
    "id": "public_id",
    "workspace_id": "workspace_name",
    "session_id": "session_name",
    "peer_id": "peer_name",
}

DISALLOWED_INTERNAL_COLUMNS = [
    "internal_metadata",
    ## TODO any others?
]


def apply_filter(
    stmt: Select, model_class, filter: dict[str, Any] | None = None
) -> Select:
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
        {"content": {"contains": "hello"}}

        # Wildcards (matches everything for that field)
        {"peer_id": "*"}

    Args:
        stmt: SQLAlchemy Select statement to modify
        model_class: SQLAlchemy model class for column access
        filter: Optional filter dictionary

    Returns:
        Modified Select statement with filter applied if provided

    Raises:
        FilterError: When the filter contains invalid configuration or values
    """
    if filter is None:
        return stmt

    conditions = _build_filter_conditions(filter, model_class)
    if conditions is not None:
        stmt = stmt.where(conditions)

    return stmt


def _build_filter_conditions(filter_dict: dict[str, Any], model_class) -> Any:
    """
    Recursively build filter conditions from a filter dictionary.

    Args:
        filter_dict: Filter dictionary that may contain logical operators
        model_class: SQLAlchemy model class for column access

    Returns:
        SQLAlchemy condition object or None
    """
    conditions = []

    # Handle logical operators
    if "AND" in filter_dict:
        if not isinstance(filter_dict["AND"], list):
            raise FilterError(
                f"AND operator must contain a list, got {type(filter_dict['AND']).__name__}"
            )
        and_conditions = []
        for sub_filter in filter_dict["AND"]:
            sub_condition = _build_filter_conditions(sub_filter, model_class)
            if sub_condition is not None:
                and_conditions.append(sub_condition)
        if and_conditions:
            conditions.append(and_(*and_conditions))

    if "OR" in filter_dict:
        if not isinstance(filter_dict["OR"], list):
            raise FilterError(
                f"OR operator must contain a list, got {type(filter_dict['OR']).__name__}"
            )
        or_conditions = []
        for sub_filter in filter_dict["OR"]:
            sub_condition = _build_filter_conditions(sub_filter, model_class)
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
        not_conditions = []
        for sub_filter in filter_dict["NOT"]:
            sub_condition = _build_filter_conditions(sub_filter, model_class)
            if sub_condition is not None:
                not_conditions.append(
                    not_(sub_condition)
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


def _build_field_condition(key: str, value: Any, model_class) -> Any:
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
        column_name = EXTERNAL_TO_INTERNAL_COLUMN_MAPPING_MESSAGES.get(key, key)
    else:
        column_name = EXTERNAL_TO_INTERNAL_COLUMN_MAPPING.get(key, key)

    if column_name in DISALLOWED_INTERNAL_COLUMNS:
        raise FilterError(f"Column '{key}' is not allowed to be filtered on")

    # Check if the column exists on the model
    if not hasattr(model_class, column_name):
        raise FilterError(f"Column '{key}' does not exist on {model_class.__name__}")

    column = getattr(model_class, column_name)

    # Handle wildcard - matches everything, so no condition needed
    if value == "*":
        return None

    # Handle comparison operators vs regular values
    if isinstance(value, dict):
        # Check if this is a comparison operators dict by looking for known operators
        is_comparison_dict = any(key in COMPARISON_OPERATORS for key in value)

        if is_comparison_dict:
            return _build_comparison_conditions(column, column_name, value)
        else:
            # This is a regular value that happens to be a dict
            # For JSONB fields (metadata, configuration), check if it contains nested comparison operators
            if column_name in ("h_metadata", "configuration"):
                return _build_nested_metadata_conditions(column, value)
            else:
                return column == value
    else:
        if column_name in ("h_metadata", "configuration"):
            return column.contains(value)
        else:
            return column == value


def _safe_numeric_cast(column_accessor, op_value):
    """
    Safely cast JSONB column accessor to appropriate type for comparison.

    Args:
        column_accessor: SQLAlchemy JSONB column accessor (.astext)
        op_value: The value to compare against

    Returns:
        Tuple of (cast_column_accessor, cast_op_value) for typed comparison
        or (column_accessor, str_op_value) for string comparison
    """
    if isinstance(op_value, bool):
        # For boolean values, compare with the string representation
        # PostgreSQL JSONB stores booleans as "true"/"false" strings when extracted with ->>
        return column_accessor, str(op_value).lower()
    elif isinstance(op_value, int | float):
        # Cast to numeric for proper numeric comparison
        numeric_accessor = cast(column_accessor, Numeric)
        return numeric_accessor, op_value
    else:
        # Try to parse as numeric (handles both strings and other types)
        try:
            # Try int first, then float, then string for lexicographic comparison
            parsed_value = int(op_value)
            numeric_accessor = cast(column_accessor, Numeric)
            return numeric_accessor, parsed_value
        except (ValueError, TypeError):
            try:
                parsed_value = float(op_value)
                numeric_accessor = cast(column_accessor, Numeric)
                return numeric_accessor, parsed_value
            except (ValueError, TypeError):
                if isinstance(op_value, str):
                    # If it's not numeric, treat as string comparison (e.g., dates, text)
                    # This allows date strings like "2024-02-01" to be compared lexicographically
                    return column_accessor, str(op_value)
                else:
                    raise FilterError(
                        f"Invalid value for numeric operator: {op_value}. Expected a number, got {type(op_value).__name__}"
                    ) from None


def _build_comparison_condition(
    column, field_name: str, operator: str, op_value: Any
) -> Any:
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
        safe_accessor, safe_value = _safe_numeric_cast(field_accessor, op_value)
        operator_map = {
            "gte": lambda a, v: a >= v,
            "lte": lambda a, v: a <= v,
            "gt": lambda a, v: a > v,
            "lt": lambda a, v: a < v,
            "ne": lambda a, v: a != v,
        }
        return operator_map[operator](safe_accessor, safe_value)
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
        return field_accessor.ilike(f"%{op_value}%")

    return None


def _build_nested_metadata_conditions(column, metadata_dict: dict[str, Any]) -> Any:
    """
    Build conditions for nested metadata fields with comparison operators.

    Args:
        column: SQLAlchemy JSONB column object
        metadata_dict: Dictionary containing nested field conditions

    Returns:
        Combined SQLAlchemy condition object or None
    """
    conditions = []

    for field_name, field_value in metadata_dict.items():
        if isinstance(field_value, dict) and any(
            op in COMPARISON_OPERATORS for op in field_value
        ):
            # This field has comparison operators
            field_conditions = []
            for operator, op_value in field_value.items():
                condition = _build_comparison_condition(
                    column, field_name, operator, op_value
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


def _combine_conditions_with_and(conditions: list) -> Any:
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
    column, column_name: str, comparisons: dict[str, Any]
) -> Any:
    """
    Build comparison conditions for a single column.

    Args:
        column: SQLAlchemy column object
        column_name: Name of the column
        comparisons: Dictionary of comparison operators and values

    Returns:
        Combined SQLAlchemy condition object or None
    """
    conditions = []

    # Check if this is a datetime column
    is_datetime_column = hasattr(column.type, "python_type") and issubclass(
        column.type.python_type, datetime.datetime
    )

    for operator, op_value in comparisons.items():
        # Validate that the operator is supported
        if operator not in COMPARISON_OPERATORS:
            raise FilterError(f"Unsupported comparison operator: {operator}")

        # Handle wildcard - matches everything, so no condition needed
        if op_value == "*":
            continue

        condition = None

        # For datetime columns, cast string values to timestamp
        if is_datetime_column and isinstance(op_value, str):
            # Validate datetime string to prevent SQL injection
            validated_datetime = _validate_datetime_string(op_value)
            if validated_datetime is None:
                # Raise error if datetime validation fails
                raise FilterError(f"Invalid datetime value: {op_value}")

            # Use the validated datetime object directly instead of string interpolation
            casted_value = validated_datetime
        else:
            # if the operator is a numeric operator, the value must cast to a number
            if operator in NUMERIC_OPERATORS:
                try:
                    casted_value = float(op_value)
                except ValueError:
                    raise FilterError(
                        f"Invalid numeric value: {op_value}. Expected a number, got {type(op_value).__name__}"
                    ) from None
            else:
                casted_value = op_value

        if operator == "gte":
            condition = column >= casted_value
        elif operator == "lte":
            condition = column <= casted_value
        elif operator == "gt":
            condition = column > casted_value
        elif operator == "lt":
            condition = column < casted_value
        elif operator == "ne":
            condition = column != casted_value
        elif operator == "in":
            if hasattr(op_value, "__iter__") and not isinstance(op_value, str | bytes):
                # Handle wildcard in iterable - if present, matches everything, so no condition needed
                if "*" in op_value:
                    continue
                else:
                    if is_datetime_column:
                        # Validate and cast each datetime string value
                        casted_values = []
                        for val in op_value:
                            if isinstance(val, str):
                                validated_datetime = _validate_datetime_string(val)
                                if validated_datetime is None:
                                    raise FilterError(
                                        f"Invalid datetime value in list: {val}"
                                    )
                                casted_values.append(validated_datetime)
                            else:
                                casted_values.append(val)
                        if casted_values:
                            condition = column.in_(casted_values)
                    else:
                        condition = column.in_(list(op_value))
            else:
                raise FilterError(
                    f"Invalid value for 'in' operator: {op_value}. Expected an iterable (list, tuple, set), got {type(op_value).__name__}"
                )
        elif operator == "contains":
            if column_name == "h_metadata":
                # For JSONB columns, use JSONB contains
                condition = column.contains(op_value)
            else:
                # For text columns, use ILIKE
                condition = column.ilike(f"%{op_value}%")
        elif operator == "icontains":
            # Case-insensitive contains for text columns
            condition = column.ilike(f"%{op_value}%")

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

    This function attempts to parse the datetime string using multiple common formats
    to ensure it's a valid datetime before allowing it to be used in SQL queries.

    Args:
        value: String value to validate as datetime

    Returns:
        Parsed datetime object if valid, None if invalid
    """
    # Strip whitespace
    value = value.strip()

    # Try to parse with various common datetime formats
    datetime_formats = [
        "%Y-%m-%d %H:%M:%S",  # 2024-01-01 12:00:00
        "%Y-%m-%d %H:%M:%S.%f",  # 2024-01-01 12:00:00.123456
        "%Y-%m-%dT%H:%M:%S",  # 2024-01-01T12:00:00 (ISO format)
        "%Y-%m-%dT%H:%M:%S.%f",  # 2024-01-01T12:00:00.123456
        "%Y-%m-%dT%H:%M:%SZ",  # 2024-01-01T12:00:00Z (UTC)
        "%Y-%m-%dT%H:%M:%S.%fZ",  # 2024-01-01T12:00:00.123456Z
        "%Y-%m-%d",  # 2024-01-01
    ]

    for fmt in datetime_formats:
        try:
            return datetime.datetime.strptime(value, fmt)
        except ValueError:
            continue

    # Try fromisoformat as a fallback (Python 3.7+)
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        pass

    # Return None for invalid datetime - let the caller handle the error
    return None
