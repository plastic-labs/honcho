import datetime
from logging import getLogger
from typing import Any

from sqlalchemy import Select, and_, cast, not_, or_
from sqlalchemy.types import Numeric

logger = getLogger(__name__)


def apply_filter(stmt: Select, filter: dict[str, Any] | None = None) -> Select:
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
        filter: Optional filter dictionary

    Returns:
        Modified Select statement with filter applied if provided
    """
    if filter is None:
        return stmt

    # Get the model class from the statement's columns
    model_class = stmt.column_descriptions[0]["entity"]

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
        and_conditions = []
        for sub_filter in filter_dict["AND"]:
            sub_condition = _build_filter_conditions(sub_filter, model_class)
            if sub_condition is not None:
                and_conditions.append(sub_condition)
        if and_conditions:
            conditions.append(and_(*and_conditions))

    if "OR" in filter_dict:
        or_conditions = []
        for sub_filter in filter_dict["OR"]:
            sub_condition = _build_filter_conditions(sub_filter, model_class)
            if sub_condition is not None:
                or_conditions.append(sub_condition)
        if or_conditions:
            conditions.append(or_(*or_conditions))

    if "NOT" in filter_dict:
        not_conditions = []
        for sub_filter in filter_dict["NOT"]:
            sub_condition = _build_filter_conditions(sub_filter, model_class)
            if sub_condition is not None:
                not_conditions.append(sub_condition)
        if not_conditions:
            conditions.append(not_(and_(*not_conditions)))

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
    # Map 'metadata' to 'h_metadata' for all models
    column_name = "h_metadata" if key == "metadata" else key
    if column_name == "id":
        # For Message model, id maps to public_id, for others it maps to name
        column_name = "public_id" if model_class.__name__ == "Message" else "name"
    # Only apply _id to _name mapping for fields that are actually foreign keys
    # and not already mapped to public_id
    elif column_name.endswith("_id"):
        column_name = column_name[:-3] + "_name"

    # Check if the column exists on the model
    if not hasattr(model_class, column_name):
        logger.debug(
            f"Column {column_name} does not exist on model {model_class.__name__}"
        )
        return None

    column = getattr(model_class, column_name)

    # Handle wildcard - matches everything, so no condition needed
    if value == "*":
        return None

    # Handle comparison operators vs regular values
    if isinstance(value, dict):
        # Check if this is a comparison operators dict by looking for known operators
        comparison_operators = {
            "gte",
            "lte",
            "gt",
            "lt",
            "ne",
            "in",
            "contains",
            "icontains",
        }
        is_comparison_dict = any(key in comparison_operators for key in value)

        if is_comparison_dict:
            return _build_comparison_conditions(column, column_name, value)
        else:
            # This is a regular value that happens to be a dict
            # For JSONB fields (metadata, configuration), check if it contains nested comparison operators
            # Check if this column is a JSONB column by checking the column type
            is_jsonb_column = (
                hasattr(column.type, "python_type")
                and hasattr(column.type, "impl")
                and "JSONB" in str(column.type)
            )
            if column_name in ("h_metadata", "configuration") or is_jsonb_column:
                return _build_nested_metadata_conditions(column, value)
            else:
                return column == value
    else:
        # Simple equality or contains for JSONB
        is_jsonb_column = (
            hasattr(column.type, "python_type")
            and hasattr(column.type, "impl")
            and "JSONB" in str(column.type)
        )
        if column_name in ("h_metadata", "configuration") or is_jsonb_column:
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
    elif isinstance(op_value, (int, float)):
        try:
            # Cast to numeric for proper numeric comparison
            numeric_accessor = cast(column_accessor, Numeric)
            return numeric_accessor, op_value
        except Exception:
            # Fall back to string comparison if casting fails
            logger.debug(
                "Failed to cast JSONB value to numeric, using string comparison"
            )
            return column_accessor, str(op_value)
    else:
        # Non-numeric, non-boolean value, use string comparison
        return column_accessor, str(op_value)


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
    if op_value == "*":
        return None

    field_accessor = column[field_name].astext

    # Mapping of operators to their SQLAlchemy methods
    numeric_operators = {"gte", "lte", "gt", "lt", "ne"}

    if operator in numeric_operators:
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
        if isinstance(op_value, list):
            return field_accessor.in_([str(v) for v in op_value])
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
    comparison_operators = {
        "gte",
        "lte",
        "gt",
        "lt",
        "ne",
        "in",
        "contains",
        "icontains",
    }

    for field_name, field_value in metadata_dict.items():
        if isinstance(field_value, dict) and any(
            op in comparison_operators for op in field_value
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
    from sqlalchemy import text

    conditions = []

    # Check if this is a datetime column
    is_datetime_column = hasattr(column.type, "python_type") and issubclass(
        column.type.python_type, datetime.datetime
    )

    for operator, op_value in comparisons.items():
        if op_value == "*":
            # Wildcard for this operator - skip condition
            continue

        condition = None

        # For datetime columns, cast string values to timestamp
        if is_datetime_column and isinstance(op_value, str):
            # Use PostgreSQL's timestamp casting
            casted_value = text(f"'{op_value}'::timestamp")
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
            if isinstance(op_value, list):
                # Check if the list contains wildcard - if so, skip condition (match all)
                if "*" in op_value:
                    continue
                else:
                    if is_datetime_column:
                        # Cast each datetime string value
                        casted_values = [
                            text(f"'{val}'::timestamp") if isinstance(val, str) else val
                            for val in op_value
                        ]
                        condition = column.in_(casted_values)
                    else:
                        condition = column.in_(op_value)
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
