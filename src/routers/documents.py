import logging
from collections.abc import Sequence
from typing import Optional

from fastapi import APIRouter, Depends, Query, Path, Body
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import crud, schemas
from src.dependencies import db
from src.exceptions import ResourceNotFoundException, ValidationException
from src.security import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/apps/{app_id}/users/{user_id}/collections/{collection_id}/documents",
    tags=["documents"],
    dependencies=[
        Depends(
            require_auth(
                app_id="app_id", user_id="user_id", collection_id="collection_id"
            )
        )
    ],
)


@router.post("/list", response_model=Page[schemas.Document])
async def get_documents(
    app_id: str = Path(..., description="ID of the app"),
    user_id: str = Path(..., description="ID of the user"),
    collection_id: str = Path(..., description="ID of the collection"),
    options: schemas.DocumentGet = Body(..., description="Filtering options for the documents list"),
    reverse: Optional[bool] = Query(False, description="Whether to reverse the order of results"),
    db=db,
):
    """Get all of the Documents in a Collection"""
    try:
        documents_query = await crud.get_documents(
            db,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
            filter=options.filter,
            reverse=reverse,
        )

        return await paginate(db, documents_query)
    except ValueError as e:
        logger.warning(
            f"Failed to get documents for collection {collection_id}: {str(e)}"
        )
        raise ResourceNotFoundException(
            "Collection not found or does not belong to user"
        ) from e


@router.get("/{document_id}", response_model=schemas.Document)
async def get_document(
    app_id: str = Path(..., description="ID of the app"),
    user_id: str = Path(..., description="ID of the user"),
    collection_id: str = Path(..., description="ID of the collection"),
    document_id: str = Path(..., description="ID of the document to retrieve"),
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
    return honcho_document


@router.post("/query", response_model=Sequence[schemas.Document])
async def query_documents(
    app_id: str = Path(..., description="ID of the app"),
    user_id: str = Path(..., description="ID of the user"),
    collection_id: str = Path(..., description="ID of the collection"),
    options: schemas.DocumentQuery = Body(..., description="Query parameters for document search"),
    db=db,
):
    """Cosine Similarity Search for Documents"""

    try:
        top_k = options.top_k
        filter = options.filter
        if options.filter == {}:
            filter = None

        documents = await crud.query_documents(
            db=db,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
            query=options.query,
            filter=filter,
            top_k=top_k,
        )

        logger.info(f"Query documents successful for collection {collection_id}")
        return documents
    except ValueError as e:
        logger.error(
            f"Error querying documents in collection {collection_id}: {str(e)}"
        )
        raise ValidationException("Error querying documents") from e


@router.post("", response_model=schemas.Document)
async def create_document(
    app_id: str = Path(..., description="ID of the app"),
    user_id: str = Path(..., description="ID of the user"),
    collection_id: str = Path(..., description="ID of the collection"),
    document: schemas.DocumentCreate = Body(..., description="Document creation parameters"),
    db=db,
):
    """Embed text as a vector and create a Document"""
    try:
        document_obj = await crud.create_document(
            db,
            document=document,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
        )
        logger.info(f"Document created successfully in collection {collection_id}")
        return document_obj
    except ValueError as e:
        logger.warning(
            f"Failed to create document in collection {collection_id}: {str(e)}"
        )
        raise ResourceNotFoundException(
            "Collection not found or does not belong to user"
        ) from e


@router.put(
    "/{document_id}",
    response_model=schemas.Document,
)
async def update_document(
    app_id: str = Path(..., description="ID of the app"),
    user_id: str = Path(..., description="ID of the user"),
    collection_id: str = Path(..., description="ID of the collection"),
    document_id: str = Path(..., description="ID of the document to update"),
    document: schemas.DocumentUpdate = Body(..., description="Updated document parameters"),
    db=db,
):
    """Update the content and/or the metadata of a Document"""
    if document.content is None and document.metadata is None:
        logger.warning(
            f"Document update attempted with empty content and metadata for document {document_id}"
        )
        raise ValidationException("Content and metadata cannot both be None")

    try:
        updated_document = await crud.update_document(
            db,
            document=document,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
            document_id=document_id,
        )
        logger.info(f"Document {document_id} updated successfully")
        return updated_document
    except ValueError as e:
        logger.warning(f"Failed to update document {document_id}: {str(e)}")
        raise ResourceNotFoundException("Collection or document not found") from e


@router.delete("/{document_id}")
async def delete_document(
    app_id: str = Path(..., description="ID of the app"),
    user_id: str = Path(..., description="ID of the user"),
    collection_id: str = Path(..., description="ID of the collection"),
    document_id: str = Path(..., description="ID of the document to delete"),
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
        logger.info(f"Document {document_id} deleted successfully")
        return {"message": "Document deleted successfully"}
    else:
        logger.warning(f"Document {document_id} not found or could not be deleted")
        raise ResourceNotFoundException("Document not found or does not belong to user")
