import datetime
import logging
import os

from fastapi import APIRouter, Depends

from src.exceptions import DisabledException, ValidationException
from src.security import (
    JWTParams,
    create_jwt,
    require_auth,
)

logger = logging.getLogger(__name__)

USE_AUTH = os.getenv("USE_AUTH", "False").lower() == "true"

router = APIRouter(
    prefix="/keys",
    tags=["keys"],
    dependencies=[Depends(require_auth(admin=True))],
)


@router.post("")
async def create_key(
    expires_at: datetime.datetime | None = None,
    app_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    collection_id: str | None = None,
):
    """Create a new Key"""
    if not USE_AUTH:
        raise DisabledException()

    # Validate that at least one parameter is provided for proper scoping
    if not any([app_id, user_id, session_id, collection_id]):
        raise ValidationException(
            "At least one of app_id, user_id, session_id, or collection_id must be provided"
        )

    key_str = create_jwt(
        JWTParams(
            exp=expires_at.isoformat() if expires_at else None,
            ap=app_id,
            us=user_id,
            se=session_id,
            co=collection_id,
        )
    )
    return {
        "key": key_str,
    }
