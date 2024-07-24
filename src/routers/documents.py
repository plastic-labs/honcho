import json
import uuid
from typing import Optional, Sequence

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import crud, schemas
from src.dependencies import db
from src.security import auth

router = APIRouter(
    prefix="/apps/{app_id}/users/{user_id}/collections/{collection_id}",
    tags=["documents"],
)


@router.get("/documents", response_model=Page[schemas.Document])
async def get_documents(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
    reverse: Optional[bool] = False,
    filter: Optional[str] = None,
    db=db,
    auth=Depends(auth),
):
    try:
        data = None
        if filter is not None:
            data = json.loads(filter)
        return await paginate(
            db,
            await crud.get_documents(
                db,
                app_id=app_id,
                user_id=user_id,
                collection_id=collection_id,
                filter=data,
                reverse=reverse,
            ),
        )
    except (
        ValueError
    ):  # TODO can probably remove this exception ok to return empty here
        raise HTTPException(
            status_code=404, detail="collection not found or does not belong to user"
        ) from None


@router.get(
    "/documents/{document_id}",
    response_model=schemas.Document,
)
async def get_document(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    db=db,
    auth=Depends(auth),
):
    honcho_document = await crud.get_document(
        db,
        app_id=app_id,
        user_id=user_id,
        collection_id=collection_id,
        document_id=document_id,
    )
    if honcho_document is None:
        raise HTTPException(
            status_code=404, detail="document not found or does not belong to user"
        )
    return honcho_document


@router.get("/query", response_model=Sequence[schemas.Document])
async def query_documents(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
    query: str,
    top_k: int = 5,
    filter: Optional[str] = None,
    db=db,
    auth=Depends(auth),
):
    if top_k is not None and top_k > 50:
        top_k = 50  # TODO see if we need to paginate this
    data = None
    if filter is not None:
        data = json.loads(filter)
    return await crud.query_documents(
        db=db,
        app_id=app_id,
        user_id=user_id,
        collection_id=collection_id,
        query=query,
        filter=data,
        top_k=top_k,
    )


@router.post("/documents", response_model=schemas.Document)
async def create_document(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
    document: schemas.DocumentCreate,
    db=db,
    auth=Depends(auth),
):
    try:
        return await crud.create_document(
            db,
            document=document,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
        )
    except ValueError:
        raise HTTPException(
            status_code=404, detail="collection not found or does not belong to user"
        ) from None


@router.put(
    "/documents/{document_id}",
    response_model=schemas.Document,
)
async def update_document(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    document: schemas.DocumentUpdate,
    db=db,
    auth=Depends(auth),
):
    if document.content is None and document.metadata is None:
        raise HTTPException(
            status_code=400, detail="content and metadata cannot both be None"
        )
    try:
        return await crud.update_document(
            db,
            document=document,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
            document_id=document_id,
        )
    except ValueError:
        raise HTTPException(
            status_code=404, detail="collection not found or does not belong to user"
        ) from None


@router.delete("/documents/{document_id}")
async def delete_document(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    db=db,
    auth=Depends(auth),
):
    response = await crud.delete_document(
        db,
        app_id=app_id,
        user_id=user_id,
        collection_id=collection_id,
        document_id=document_id,
    )
    if response:
        return {"message": "Document deleted successfully"}
    else:
        raise HTTPException(
            status_code=404, detail="document not found or does not belong to user"
        )
