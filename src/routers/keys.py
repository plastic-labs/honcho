import logging

from fastapi import APIRouter, Depends
from src.security import require_auth, JWTParams, create_jwt

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/keys",
    tags=["keys"],
    dependencies=[Depends(require_auth(admin=True))],
)


@router.post("")
async def create_key(
    params: JWTParams,
):
    """Create a new Key"""
    key = create_jwt(params)
    return {"key": key}
