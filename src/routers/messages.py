import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    Path,
    Query,
    UploadFile,
)
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import apaginate
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from src import crud, schemas
from src.config import settings
from src.dependencies import db
from src.deriver import enqueue
from src.exceptions import FileTooLargeError, ResourceNotFoundException
from src.security import require_auth
from src.telemetry import otel_metrics
from src.utils.files import process_file_uploads_for_messages

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/sessions/{session_id}/messages",
    tags=["messages"],
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)


async def parse_upload_form(
    peer_id: str = Form(...),
    metadata: str | None = Form(None),
    configuration: str | None = Form(None),
    created_at: str | None = Form(None),
) -> schemas.MessageUploadCreate:
    """Parse form data for file upload requests"""
    import json
    from datetime import datetime

    parsed_metadata = None
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse metadata JSON: {metadata}")
            parsed_metadata = None

    parsed_configuration = None
    if configuration:
        try:
            parsed_configuration = json.loads(configuration)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse configuration JSON: {configuration}")
            parsed_configuration = None

    parsed_created_at = None
    if created_at:
        try:
            parsed_created_at = datetime.fromisoformat(
                created_at.replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            logger.warning(f"Failed to parse created_at: {created_at}")
            parsed_created_at = None

    return schemas.MessageUploadCreate(
        peer_id=peer_id,
        metadata=parsed_metadata,
        configuration=parsed_configuration,
        created_at=parsed_created_at,
    )


@router.post("", response_model=list[schemas.Message], status_code=201)
@router.post(
    "/", response_model=list[schemas.Message], status_code=201, include_in_schema=False
)  # backwards compatibility with pre-2.6.0 faulty route endpoint
async def create_messages_for_session(
    background_tasks: BackgroundTasks,
    messages: schemas.MessageBatchCreate,
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    db: AsyncSession = db,
):
    """Add new message(s) to a session."""
    try:
        created_messages = await crud.create_messages(
            db,
            messages=messages.messages,
            workspace_name=workspace_id,
            session_name=session_id,
        )

        # OTel metrics (push-based)
        if settings.OTEL.ENABLED:
            otel_metrics.record_messages_created(
                count=len(created_messages),
                workspace_name=workspace_id,
            )

        # Enqueue for processing (existing logic)
        payloads = [
            {
                "workspace_name": workspace_id,
                "session_name": session_id,
                "message_id": message.id,
                "content": message.content,
                "peer_name": message.peer_name,
                "created_at": message.created_at,
                "message_public_id": message.public_id,
                "seq_in_session": message.seq_in_session,
                "configuration": original.configuration,
            }
            for message, original in zip(
                created_messages, messages.messages, strict=True
            )
        ]

        # Enqueue all messages in one call
        background_tasks.add_task(enqueue, payloads)

        return created_messages
    except ValueError as e:
        logger.warning(f"Failed to create messages for session {session_id}: {str(e)}")
        raise


@router.post("/upload", response_model=list[schemas.Message], status_code=201)
async def create_messages_with_file(
    background_tasks: BackgroundTasks,
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    form_data: schemas.MessageUploadCreate = Depends(parse_upload_form),
    file: UploadFile = File(...),
    db: AsyncSession = db,
):
    """Create messages from uploaded files. Files are converted to text and split into multiple messages."""

    # Validate file size
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise FileTooLargeError(
            f"File size ({file.size} bytes) exceeds maximum allowed size ({settings.MAX_FILE_SIZE} bytes)",
        )

    # Process files using shared utility function
    all_message_data = await process_file_uploads_for_messages(
        file=file,
        peer_id=form_data.peer_id,
        metadata=form_data.metadata,
        configuration=form_data.configuration,
        created_at=form_data.created_at,
    )

    # Create messages
    message_creates = [item["message_create"] for item in all_message_data]
    created_messages = await crud.create_messages(
        db,
        messages=message_creates,
        workspace_name=workspace_id,
        session_name=session_id,
    )

    # Update internal_metadata for file-related messages
    for i, message in enumerate(created_messages):
        file_metadata = all_message_data[i]["file_metadata"]
        message.internal_metadata.update(file_metadata)
        flag_modified(message, "internal_metadata")

    await db.commit()

    # Enqueue for processing (same as regular messages)
    payloads = [
        {
            "workspace_name": workspace_id,
            "session_name": session_id,
            "message_id": message.id,
            "content": message.content,
            "peer_name": message.peer_name,
            "created_at": message.created_at,
            "message_public_id": message.public_id,
            "seq_in_session": message.seq_in_session,
            "configuration": form_data.configuration,
        }
        for message in created_messages
    ]

    background_tasks.add_task(enqueue, payloads)
    logger.debug(
        "Batch of %s messages created from file uploads and queued for processing",
        len(created_messages),
    )

    # OTel metrics (push-based)
    if settings.OTEL.ENABLED:
        otel_metrics.record_messages_created(
            count=len(created_messages),
            workspace_name=workspace_id,
        )

    return created_messages


@router.post("/list", response_model=Page[schemas.Message])
async def get_messages(
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    options: schemas.MessageGet | None = Body(
        None, description="Filtering options for the message list"
    ),
    reverse: bool | None = Query(
        False, description="Whether to reverse the order of results"
    ),
    db: AsyncSession = db,
):
    """Get all messages for a Session with optional filters. Results are paginated."""
    try:
        filters = None
        if options and hasattr(options, "filters"):
            filters = options.filters
            if filters == {}:
                filters = None

        messages_query = await crud.get_messages(
            workspace_name=workspace_id,
            session_name=session_id,
            filters=filters,
            reverse=reverse,
        )

        return await apaginate(db, messages_query)
    except ValueError as e:
        logger.warning(f"Failed to get messages for session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.get("/{message_id}", response_model=schemas.Message)
async def get_message(
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    message_id: str = Path(...),
    db: AsyncSession = db,
):
    """Get a single message by ID from a Session."""
    honcho_message = await crud.get_message(
        db, workspace_name=workspace_id, session_name=session_id, message_id=message_id
    )
    if honcho_message is None:
        logger.warning(f"Message {message_id} not found in session {session_id}")
        raise ResourceNotFoundException(f"Message with ID {message_id} not found")
    return honcho_message


@router.put("/{message_id}", response_model=schemas.Message)
async def update_message(
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    message_id: str = Path(...),
    message: schemas.MessageUpdate = Body(
        ..., description="Updated message parameters"
    ),
    db: AsyncSession = db,
):
    """
    Update the metadata of a message.

    This will overwrite any existing metadata for the message.
    """
    try:
        updated_message = await crud.update_message(
            db,
            message=message,
            workspace_name=workspace_id,
            session_name=session_id,
            message_id=message_id,
        )
        logger.debug("Message %s updated successfully", message_id)
        return updated_message
    except ValueError as e:
        logger.warning(f"Failed to update message {message_id}: {str(e)}")
        raise ResourceNotFoundException("Message not found") from e
