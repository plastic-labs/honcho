import json
from collections.abc import Sequence
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import crud, schemas
from src.dependencies import db
from src.security import auth

router = APIRouter(
    prefix="/apps/{app_id}/users/{user_id}/collections/{collection_id}/documents",
    tags=["documents"],
    dependencies=[Depends(auth)],
)


@router.post("/list", response_model=Page[schemas.Document])
async def get_documents(
    app_id: str,
    user_id: str,
    collection_id: str,
    options: schemas.DocumentGet,
    reverse: Optional[bool] = False,
    db=db,
):
    """Get all of the Documents in a Collection"""
    try:
        return await paginate(
            db,
            await crud.get_documents(
                db,
                app_id=app_id,
                user_id=user_id,
                collection_id=collection_id,
                filter=options.filter,
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
    "/{document_id}",
    response_model=schemas.Document,
)
async def get_document(
    app_id: str,
    user_id: str,
    collection_id: str,
    document_id: str,
    db=db,
):
    """Get a document by ID"""
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


@router.post("/query", response_model=Sequence[schemas.Document])
async def query_documents(
    app_id: str,
    user_id: str,
    collection_id: str,
    options: schemas.DocumentQuery,
    db=db,
):
    """Cosine Similarity Search for Documents"""

    try:
        top_k = options.top_k
        filter = options.filter
        if options.filter == {}:
            filter = None
        return await crud.query_documents(
            db=db,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
            query=options.query,
            filter=filter,
            top_k=top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Error Query Documents") from e


@router.post("", response_model=schemas.Document)
async def create_document(
    app_id: str,
    user_id: str,
    collection_id: str,
    document: schemas.DocumentCreate,
    db=db,
):
    """Embed text as a vector and create a Document"""
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
    "/{document_id}",
    response_model=schemas.Document,
)
async def update_document(
    app_id: str,
    user_id: str,
    collection_id: str,
    document_id: str,
    document: schemas.DocumentUpdate,
    db=db,
):
    """Update the content and/or the metadata of a Document"""
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


@router.delete("/{document_id}")
async def delete_document(
    app_id: str,
    user_id: str,
    collection_id: str,
    document_id: str,
    db=db,
):
    """Delete a Document by ID"""
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
