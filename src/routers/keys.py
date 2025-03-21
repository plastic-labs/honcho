import logging

from fastapi import APIRouter, Depends

from src import crud
from src.dependencies import db
from src.security import JWTParams, create_jwt, require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/keys",
    tags=["keys"],
    dependencies=[Depends(require_auth(admin=True))],
)


@router.post("")
async def create_key(
    app_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    collection_id: str | None = None,
    db=db,
):
    """Create a new Key"""
    key_str = create_jwt(
        JWTParams(
            ap=app_id,
            us=user_id,
            se=session_id,
            co=collection_id,
        )
    )
    key = await crud.create_key(db, key_str)
    return {
        "key": key_str,
        "created_at": key.created_at,
    }


@router.post("/revoke")
async def revoke_key(
    key: str,
    db=db,
):
    """Revoke a Key"""
    await crud.revoke_key(db, key)
    return {"revoked": key}
