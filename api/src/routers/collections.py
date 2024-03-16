import json
from typing import Optional
import uuid
from fastapi import APIRouter, HTTPException, Request
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import crud, schemas
from src.dependencies import db


router = APIRouter(
    prefix="/apps/{app_id}/users/{user_id}/collections",
    tags=["collections"],
)


@router.get("", response_model=Page[schemas.Collection])
async def get_collections(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    reverse: Optional[bool] = False,
    filter: Optional[str] = None,
    db=db,
):
    """Get All Collections for a User

    Args:
        app_id (uuid.UUID): The ID of the app representing the client
        application using honcho
        user_id (uuid.UUID): The User ID representing the user, managed by the user

    Returns:
        list[schemas.Collection]: List of Collection objects

    """
    data = None
    if filter is not None:
        data = json.loads(filter)
    return await paginate(
        db,
        await crud.get_collections(
            db, app_id=app_id, user_id=user_id, filter=data, reverse=reverse
        ),
    )


# @router.get("/id/{collection_id}", response_model=schemas.Collection)
# def get_collection_by_id(
#     request: Request,
#     app_id: uuid.UUID,
#     user_id: uuid.UUID,
#     collection_id: uuid.UUID,
#     db=db,
# ) -> schemas.Collection:
#     honcho_collection = crud.get_collection_by_id(
#         db, app_id=app_id, user_id=user_id, collection_id=collection_id
#     )
#     if honcho_collection is None:
#         raise HTTPException(
#             status_code=404, detail="collection not found or does not belong to user"
#         )
#     return honcho_collection


@router.get("/{name}", response_model=schemas.Collection)
async def get_collection_by_name(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    name: str,
    db=db,
) -> schemas.Collection:
    honcho_collection = await crud.get_collection_by_name(
        db, app_id=app_id, user_id=user_id, name=name
    )
    if honcho_collection is None:
        raise HTTPException(
            status_code=404, detail="collection not found or does not belong to user"
        )
    return honcho_collection


@router.post("", response_model=schemas.Collection)
async def create_collection(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    collection: schemas.CollectionCreate,
    db=db,
):
    if collection.name == "honcho":
        raise HTTPException(
            status_code=406,
            detail="error invalid collection configuration - honcho is a reserved name",
        )
    try:
        return await crud.create_collection(
            db, collection=collection, app_id=app_id, user_id=user_id
        )
    except ValueError:
        raise HTTPException(
            status_code=406,
            detail="Error invalid collection configuration - name may already exist",
        ) from None


@router.put("/{collection_id}", response_model=schemas.Collection)
async def update_collection(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
    collection: schemas.CollectionUpdate,
    db=db,
):
    if collection.name is None:
        raise HTTPException(
            status_code=400, detail="invalid request - name cannot be None"
        )
    if collection.name == "honcho":
        raise HTTPException(
            status_code=406,
            detail="error invalid collection configuration - honcho is a reserved name",
        )

    try:
        honcho_collection = await crud.update_collection(
            db,
            collection=collection,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
        )
    except ValueError:
        raise HTTPException(
            status_code=406,
            detail="Error invalid collection configuration - name may already exist",
        ) from None
    return honcho_collection


@router.delete("/{collection_id}")
async def delete_collection(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
    db=db,
):
    response = await crud.delete_collection(
        db, app_id=app_id, user_id=user_id, collection_id=collection_id
    )
    if response:
        return {"message": "Collection deleted successfully"}
    else:
        raise HTTPException(
            status_code=404, detail="collection not found or does not belong to user"
        )
