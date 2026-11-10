from typing import Any, TypeVar

from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


def schema_from_path(
    schema: type[ModelT], path_param: str, name: str, **fields: Any
) -> ModelT:
    """Build a create schema for the resource named by a request path parameter.

    Raises:
        RequestValidationError: If `name` fails the schema, located at
            `path.<path_param>`
    """
    try:
        return schema(name=name, **fields)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_input=False)
        raise RequestValidationError(
            [
                {**error, "loc": ("path", path_param, *error["loc"][1:])}
                if error["loc"][:1] == ("name",)
                else error
                for error in errors
            ]
        ) from None
