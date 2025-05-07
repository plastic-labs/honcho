import datetime
import logging

from fastapi import APIRouter, Depends, Path, Query, Request

from src import crud
from src.dependencies import db
from src.security import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/transactions",
    tags=["transactions"],
)


async def get_transaction_id(request: Request) -> int | None:
    transaction_id = request.headers.get("X-Transaction-ID")
    if transaction_id:
        try:
            return int(transaction_id)
        except ValueError:
            return None
    return None


@router.post(
    "/begin",
    dependencies=[Depends(require_auth(admin=True))],
)
async def create_transaction(
    db=db,
    expires_at: datetime.datetime | None = Query(
        None, description="Expiration time of the transaction"
    ),
) -> int:
    """Create a new Transaction"""
    transaction = await crud.create_transaction(db, expires_at=expires_at)
    return transaction.transaction_id


@router.post(
    "/{transaction_id}/commit",
    dependencies=[Depends(require_auth(admin=True))],
)
async def commit_transaction(
    transaction_id: int = Path(..., description="ID of the transaction to commit"),
    db=db,
):
    """Commit a Transaction"""
    await crud.commit_transaction(db, transaction_id)


@router.post(
    "/{transaction_id}/rollback",
    dependencies=[Depends(require_auth(admin=True))],
)
async def rollback_transaction(
    transaction_id: int = Path(..., description="ID of the transaction to rollback"),
    db=db,
):
    """Rollback a Transaction"""
    await crud.rollback_transaction(db, transaction_id)
