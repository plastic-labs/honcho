import logging
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import crud, schemas
from src.dependencies import db
from src.exceptions import ResourceNotFoundException, ValidationException, ConflictException
from src.security import auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/apps/{app_id}/users/{user_id}/collections",
    tags=["collections"],
    dependencies=[Depends(auth)],
)


@router.post("/list", response_model=Page[schemas.Collection])
async def get_collections(
    app_id: str,
    user_id: str,
    options: schemas.CollectionGet,
    reverse: Optional[bool] = False,
    db=db,
):
    """Get All Collections for a User"""
    return await paginate(
        db,
        await crud.get_collections(
            db, app_id=app_id, user_id=user_id, filter=options.filter, reverse=reverse
        ),
    )


@router.get("/name/{name}", response_model=schemas.Collection)
async def get_collection_by_name(
    app_id: str,
    user_id: str,
    name: str,
    db=db,
) -> schemas.Collection:
    """Get a Collection by Name"""
    honcho_collection = await crud.get_collection_by_name(
        db, app_id=app_id, user_id=user_id, name=name
    )
    return honcho_collection


@router.get("/{collection_id}", response_model=schemas.Collection)
async def get_collection_by_id(
    app_id: str,
    user_id: str,
    collection_id: str,
    db=db,
) -> schemas.Collection:
    """Get a Collection by ID"""
    honcho_collection = await crud.get_collection_by_id(
        db, app_id=app_id, user_id=user_id, collection_id=collection_id
    )
    return honcho_collection


@router.post("", response_model=schemas.Collection)
async def create_collection(
    app_id: str,
    user_id: str,
    collection: schemas.CollectionCreate,
    db=db,
):
    """Create a new Collection"""
    return await crud.create_collection(
        db, collection=collection, app_id=app_id, user_id=user_id
    )


@router.put("/{collection_id}", response_model=schemas.Collection)
async def update_collection(
    app_id: str,
    user_id: str,
    collection_id: str,
    collection: schemas.CollectionUpdate,
    db=db,
):
    "Update a Collection's name or metadata"
    honcho_collection = await crud.update_collection(
        db,
        collection=collection,
        app_id=app_id,
        user_id=user_id,
        collection_id=collection_id,
    )
    return honcho_collection


@router.delete("/{collection_id}")
async def delete_collection(
    app_id: str,
    user_id: str,
    collection_id: str,
    db=db,
):
    """Delete a Collection and its documents"""
    await crud.delete_collection(
        db, app_id=app_id, user_id=user_id, collection_id=collection_id
    )
    return {"message": "Collection deleted successfully"}
