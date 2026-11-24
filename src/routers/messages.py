import logging
from collections.abc import Sequence
from typing import Any

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
from fastapi.exceptions import RequestValidationError
from fastapi_pagination.bases import AbstractParams
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.crud.message import CreateMessagesResult
from src.db import tenant_context
from src.dependencies import db, read_db
from src.deriver.enqueue import (
    cancel_dreams_for_activity,
    commit_with_retry,
    stage_enqueue,
)
from src.exceptions import FileTooLargeError, ResourceNotFoundException
from src.reconciler.embed_now import embed_task_gate
from src.security import require_auth
from src.telemetry import prometheus_metrics
from src.telemetry.events import FileUploadedEvent, MessageCreatedEvent, emit
from src.utils.files import process_file_uploads_for_messages
from src.utils.pagination import (
    CursorPage,
    Page,
    paginate_offset_or_cursor,
    pagination_params,
)
from src.utils.retryable_errors import is_retryable_conflict

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/sessions/{session_id}/messages",
    tags=["messages"],
)

# Read routes additionally allow a peer-scoped key whose peer is a member of the
# session; write routes stay session-scoped only. Applied per-route rather than
# on the router so the two policies can differ.
require_session_read = require_auth(
    workspace_name="workspace_id",
    session_name="session_id",
    allow_member_read=True,
    allow_scope=True,
)
require_session_write = require_auth(
    workspace_name="workspace_id",
    session_name="session_id",
    allow_scope=True,
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

    try:
        return schemas.MessageUploadCreate(
            peer_id=peer_id,
            metadata=parsed_metadata,
            configuration=parsed_configuration,
            created_at=parsed_created_at,
        )
    except ValidationError as exc:
        raise RequestValidationError(
            [
                {**error, "loc": ("body", *error["loc"])}
                for error in exc.errors(include_url=False, include_input=False)
            ]
        ) from None


async def _create_and_enqueue(
    db: AsyncSession,
    background_tasks: BackgroundTasks,
    *,
    workspace_name: str,
    session_name: str,
    messages: list[schemas.MessageCreate],
    configurations: Sequence[schemas.MessageConfiguration | None],
    internal_metadata: Sequence[dict[str, Any]] | None = None,
) -> list[models.Message]:
    """Insert messages and their queue rows in one transaction, then commit.

    A message is never saved without the work that derives it: if the queue
    insert fails, the whole write fails. A deadlock or serialization failure
    reruns the transaction from the start.
    """
    payloads: list[dict[str, Any]] = []

    async def stage() -> CreateMessagesResult:
        result = await crud.create_messages(
            db,
            messages=messages,
            workspace_name=workspace_name,
            session_name=session_name,
            internal_metadata=internal_metadata,
        )
        payloads[:] = [
            {
                "workspace_name": workspace_name,
                "session_name": session_name,
                "message_id": message.id,
                "content": message.content,
                "peer_name": message.peer_name,
                "created_at": message.created_at,
                "message_public_id": message.public_id,
                "seq_in_session": message.seq_in_session,
                "configuration": configuration,
            }
            for message, configuration in zip(
                result.messages, configurations, strict=True
            )
        ]
        await stage_enqueue(db, payloads, session=result.session)
        return result

    result = await commit_with_retry(db, stage, retryable=is_retryable_conflict)
    await result.post_commit()

    # ai: tenant read now, in the request, for the reason EmbedTaskGate.try_schedule gives
    background_tasks.add_task(
        cancel_dreams_for_activity, payloads, tenant_id=tenant_context.get()
    )
    return result.messages


@router.post(
    "",
    response_model=list[schemas.Message],
    status_code=201,
    dependencies=[Depends(require_session_write)],
)
@router.post(
    "/",
    response_model=list[schemas.Message],
    status_code=201,
    include_in_schema=False,
    dependencies=[Depends(require_session_write)],
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
        created_messages = await _create_and_enqueue(
            db,
            background_tasks,
            workspace_name=workspace_id,
            session_name=session_id,
            messages=messages.messages,
            configurations=[message.configuration for message in messages.messages],
        )

        # Prometheus metrics
        if settings.METRICS.ENABLED:
            prometheus_metrics.record_messages_created(
                count=len(created_messages),
                workspace_name=workspace_id,
            )

        emit(
            MessageCreatedEvent(
                workspace_name=workspace_id,
                session_name=session_id,
                message_count=len(created_messages),
                total_tokens=sum(message.token_count for message in created_messages),
                source="api",
                last_message_id=created_messages[-1].public_id,
            )
        )

        # Embed immediately so messages are searchable within seconds; the
        # reconciler is the fallback for anything left pending. Scheduling is
        # capped per process — when saturated, the reconciler picks them up.
        if settings.EMBED_MESSAGES and created_messages:
            scheduled = embed_task_gate.try_schedule(
                background_tasks, [m.public_id for m in created_messages]
            )
            if not scheduled:
                logger.debug(
                    "Immediate-embed tasks saturated; deferring %s message(s) to reconciler",
                    len(created_messages),
                )

        return created_messages
    except ValueError as e:
        logger.warning(f"Failed to create messages for session {session_id}: {str(e)}")
        raise


@router.post(
    "/upload",
    response_model=list[schemas.Message],
    status_code=201,
    dependencies=[Depends(require_session_write)],
)
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

    # Create messages, each carrying its file metadata in the same insert, and
    # queue them for processing in the same transaction
    created_messages = await _create_and_enqueue(
        db,
        background_tasks,
        workspace_name=workspace_id,
        session_name=session_id,
        messages=[item["message_create"] for item in all_message_data],
        configurations=[form_data.configuration] * len(all_message_data),
        internal_metadata=[item["file_metadata"] for item in all_message_data],
    )

    # Embed immediately so messages are searchable within seconds; the
    # reconciler is the fallback for anything left pending. Scheduling is
    # capped per process — when saturated, the reconciler picks them up.
    if settings.EMBED_MESSAGES and created_messages:
        scheduled = embed_task_gate.try_schedule(
            background_tasks, [m.public_id for m in created_messages]
        )
        if not scheduled:
            logger.debug(
                "Immediate-embed tasks saturated; deferring %s message(s) to reconciler",
                len(created_messages),
            )

    logger.debug(
        "Batch of %s messages created from file uploads and queued for processing",
        len(created_messages),
    )

    # Prometheus metrics
    if settings.METRICS.ENABLED:
        prometheus_metrics.record_messages_created(
            count=len(created_messages),
            workspace_name=workspace_id,
        )

    # An empty extracted file (no chunks) leaves both lists empty. Skip the
    # telemetry in that case rather than indexing into [].
    if all_message_data and created_messages:
        file_metadata = all_message_data[0]["file_metadata"]
        total_tokens = sum(message.token_count for message in created_messages)
        emit(
            FileUploadedEvent(
                workspace_name=workspace_id,
                session_name=session_id,
                peer_name=form_data.peer_id,
                file_id=str(file_metadata["file_id"]),
                filename=file.filename,
                content_type=file.content_type,
                file_size_bytes=file.size,
                message_count=len(created_messages),
                total_tokens=total_tokens,
            )
        )
        emit(
            MessageCreatedEvent(
                workspace_name=workspace_id,
                session_name=session_id,
                message_count=len(created_messages),
                total_tokens=total_tokens,
                source="file_upload",
                last_message_id=created_messages[-1].public_id,
            )
        )

    return created_messages


@router.post(
    "/list",
    response_model=Page[schemas.Message] | CursorPage[schemas.Message],
    dependencies=[Depends(require_session_read)],
)
async def get_messages(
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    options: schemas.MessageGet | None = Body(
        None, description="Filtering options for the message list"
    ),
    reverse: bool | None = Query(
        False, description="Whether to reverse the order of results"
    ),
    params: AbstractParams = Depends(pagination_params),
    db: AsyncSession = read_db,
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

        return await paginate_offset_or_cursor(db, messages_query, params)
    except ValueError as e:
        logger.warning(f"Failed to get messages for session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.get(
    "/{message_id}",
    response_model=schemas.Message,
    dependencies=[Depends(require_session_read)],
)
async def get_message(
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    message_id: str = Path(...),
    db: AsyncSession = read_db,
):
    """Get a single message by ID from a Session."""
    honcho_message = await crud.get_message(
        db, workspace_name=workspace_id, session_name=session_id, message_id=message_id
    )
    if honcho_message is None:
        logger.warning(f"Message {message_id} not found in session {session_id}")
        raise ResourceNotFoundException(f"Message with ID {message_id} not found")
    return honcho_message


@router.put(
    "/{message_id}",
    response_model=schemas.Message,
    dependencies=[Depends(require_session_write)],
)
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
