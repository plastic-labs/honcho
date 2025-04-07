from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import crud, schemas
from src.dependencies import db
from src.exceptions import AuthenticationException, ResourceNotFoundException
from src.security import JWTParams, require_auth

router = APIRouter(
    prefix="/apps/{app_id}/users/{user_id}/collections",
    tags=["collections"],
)


@router.get(
    "",
    response_model=schemas.Collection,
)
async def get_collection(
    app_id: str,
    user_id: str,
    collection_id: Optional[str] = Query(
        None, description="Collection ID to retrieve. If not provided, uses JWT token"
    ),
    jwt_params: JWTParams = Depends(require_auth()),
    db=db,
):
    """
    Get a specific collection for a user.

    If collection_id is provided as a query parameter, it uses that (must match JWT collection_id).
    Otherwise, it uses the collection_id from the JWT token.
    """
    # Verify JWT has access to the requested resource
    if not jwt_params.ad:
        if jwt_params.ap is not None and jwt_params.ap != app_id:
            raise AuthenticationException("Unauthorized access to resource")
        if jwt_params.us is not None and jwt_params.us != user_id:
            raise AuthenticationException("Unauthorized access to resource")
    # If collection_id provided in query, check if it matches jwt or user is admin
    if collection_id:
        if (
            not jwt_params.ad
            and jwt_params.co is not None
            and jwt_params.co != collection_id
        ):
            raise AuthenticationException("Unauthorized access to resource")
        target_collection_id = collection_id
    else:
        # Use collection_id from JWT
        if not jwt_params.co:
            raise AuthenticationException(
                "Collection ID not found in query parameter or JWT"
            )
        target_collection_id = jwt_params.co

    # Let crud function handle the ResourceNotFoundException
    return await crud.get_collection_by_id(
        db, app_id=app_id, collection_id=target_collection_id, user_id=user_id
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
