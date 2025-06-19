import datetime
import logging

from fastapi import APIRouter, Depends, Query

from src.config import settings
from src.exceptions import DisabledException, ValidationException
from src.security import (
    JWTParams,
    create_jwt,
    require_auth,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/keys",
    tags=["keys"],
    dependencies=[Depends(require_auth(admin=True))],
)


@router.post("")
async def create_key(
    workspace_id: str | None = Query(
        None, description="ID of the workspace to scope the key to"
    ),
    peer_id: str | None = Query(None, description="ID of the peer to scope the key to"),
    session_id: str | None = Query(
        None, description="ID of the session to scope the key to"
    ),
    expires_at: datetime.datetime | None = None,
):
    """Create a new Key"""
    if not settings.AUTH.USE_AUTH:
        raise DisabledException()

    # Validate that at least one parameter is provided for proper scoping
    if not any([workspace_id, peer_id, session_id]):
        raise ValidationException(
            "At least one of workspace_id, peer_id, or session_id must be provided"
        )

    key_str = create_jwt(
        JWTParams(
            exp=expires_at.isoformat() if expires_at else None,
            w=workspace_id,
            p=peer_id,
            s=session_id,
        )
    )
    return {
        "key": key_str,
    }
