from typing import Optional

from fastapi import APIRouter, Depends
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import crud, schemas
from src.dependencies import db
from src.exceptions import AuthenticationException
from src.security import JWTParams, auth, require_auth

router = APIRouter(
    prefix="/apps/{app_id}/users/{user_id}/collections",
    tags=["collections"],
)


@router.get(
    "",
    response_model=schemas.Collection,
)
async def get_collection_from_token(
    app_id: str,
    user_id: str,
    jwt_params: JWTParams = Depends(auth),
    db=db,
):
    """Get a specific collection for a user by collection_id provided in the JWT"""
    if jwt_params.co is None:
        raise AuthenticationException("Collection not found in JWT")
    return await crud.get_collection_by_id(
        db, app_id=app_id, collection_id=jwt_params.co, user_id=user_id
    )


@router.post(
    "/list",
    response_model=Page[schemas.Collection],
    dependencies=[Depends(require_auth(app_id="app_id", user_id="user_id"))],
)
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


@router.get(
    "/name/{name}",
    response_model=schemas.Collection,
    dependencies=[Depends(require_auth(app_id="app_id", user_id="user_id"))],
)
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


@router.get(
    "/{collection_id}",
    response_model=schemas.Collection,
    dependencies=[
        Depends(
            require_auth(
                app_id="app_id", user_id="user_id", collection_id="collection_id"
            )
        )
    ],
)
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


@router.post(
    "",
    response_model=schemas.Collection,
    dependencies=[Depends(require_auth(app_id="app_id", user_id="user_id"))],
)
async def create_collection(
    app_id: str,
    user_id: str,
    collection: schemas.CollectionCreate,
    db=db,
):
    """Create a new Collection"""
    # ValidationException will be caught by global handler if collection is invalid
    # ConflictException will be caught by global handler if collection name already exists
    return await crud.create_collection(
        db, collection=collection, app_id=app_id, user_id=user_id
    )


@router.put(
    "/{collection_id}",
    response_model=schemas.Collection,
    dependencies=[
        Depends(
            require_auth(
                app_id="app_id", user_id="user_id", collection_id="collection_id"
            )
        )
    ],
)
async def update_collection(
    app_id: str,
    user_id: str,
    collection_id: str,
    collection: schemas.CollectionUpdate,
    db=db,
):
    "Update a Collection's name or metadata"
    # ResourceNotFoundException will be caught by global handler if collection not found
    # ValidationException will be caught by global handler if update data is invalid
    honcho_collection = await crud.update_collection(
        db,
        collection=collection,
        app_id=app_id,
        user_id=user_id,
        collection_id=collection_id,
    )
    return honcho_collection


@router.delete(
    "/{collection_id}",
    dependencies=[
        Depends(
            require_auth(
                app_id="app_id", user_id="user_id", collection_id="collection_id"
            )
        )
    ],
)
async def delete_collection(
    app_id: str,
    user_id: str,
    collection_id: str,
    db=db,
):
    """Delete a Collection and its documents"""
    # ResourceNotFoundException will be caught by global handler if collection not found
    await crud.delete_collection(
        db, app_id=app_id, user_id=user_id, collection_id=collection_id
    )
    return {"message": "Collection deleted successfully"}
