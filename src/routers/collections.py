from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import crud, schemas
from src.dependencies import db
from src.security import auth

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
    """Get All Collections for a User

    Args:
        app_id (str): The ID of the app representing the client
        application using honcho
        user_id (str): The User ID representing the user, managed by the user

    Returns:
        list[schemas.Collection]: List of Collection objects

    """
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
    honcho_collection = await crud.get_collection_by_name(
        db, app_id=app_id, user_id=user_id, name=name
    )
    if honcho_collection is None:
        raise HTTPException(
            status_code=404, detail="collection not found or does not belong to user"
        )
    return honcho_collection


@router.get("/{collection_id}", response_model=schemas.Collection)
async def get_collection_by_id(
    app_id: str,
    user_id: str,
    collection_id: str,
    db=db,
) -> schemas.Collection:
    honcho_collection = await crud.get_collection_by_id(
        db, app_id=app_id, user_id=user_id, collection_id=collection_id
    )
    if honcho_collection is None:
        raise HTTPException(
            status_code=404, detail="collection not found or does not belong to user"
        )
    return honcho_collection


@router.post("", response_model=schemas.Collection)
async def create_collection(
    app_id: str,
    user_id: str,
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
    app_id: str,
    user_id: str,
    collection_id: str,
    collection: schemas.CollectionUpdate,
    db=db,
):
    if collection.name is None and collection.metadata is None:
        raise HTTPException(
            status_code=406,
            detail="error invalid collection configuration - atleast 1 field must be provided",
        )
    if collection.name is not None and collection.name == "honcho":
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
    app_id: str,
    user_id: str,
    collection_id: str,
    db=db,
):
    try:
        await crud.delete_collection(
            db, app_id=app_id, user_id=user_id, collection_id=collection_id
        )
        return {"message": "Collection deleted successfully"}
    except ValueError:
        raise HTTPException(
            status_code=404, detail="collection not found or does not belong to user"
        ) from None
