import logging
import os

from fastapi import APIRouter, Depends

from src import crud
from src.dependencies import db
from src.exceptions import DisabledException
from src.security import (
    JWTParams,
    clear_api_key_cache,
    create_jwt,
    require_auth,
    rotate_jwt_secret,
)

logger = logging.getLogger(__name__)

USE_AUTH = os.getenv("USE_AUTH", "False").lower() == "true"

router = APIRouter(
    prefix="/keys",
    tags=["keys"],
    dependencies=[Depends(require_auth(admin=True))],
)


@router.get("/")
async def get_keys(
    db=db,
):
    """Get all Keys"""
    return await crud.get_keys(db)


@router.post("")
async def create_key(
    app_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    collection_id: str | None = None,
    db=db,
):
    """Create a new Key"""
    if not USE_AUTH:
        raise DisabledException()

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
    if not USE_AUTH:
        raise DisabledException()

    await crud.revoke_key(db, key)
    clear_api_key_cache()
    return {"revoked": key}


@router.post("/rotate")
async def rotate(
    new_secret: str | None = None,
    db=db,
):
    """Rotate the JWT secret and return admin JWT"""
    if not USE_AUTH:
        raise DisabledException()

    new_jwt = rotate_jwt_secret(new_secret)

    key = await crud.create_key(db, new_jwt)

    return {
        "key": new_jwt,
        "created_at": key.created_at,
    }
