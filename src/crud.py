import datetime
import functools
import inspect
import os
from collections.abc import Coroutine, Sequence
from logging import getLogger
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from sqlalchemy import Select, cast, delete, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func
from sqlalchemy.types import BigInteger

from . import models, schemas
from .exceptions import (
    ConflictException,
    ResourceNotFoundException,
    ValidationException,
)

load_dotenv(override=True)

openai_client = AsyncOpenAI()

logger = getLogger(__name__)

DEF_PROTECTED_COLLECTION_NAME = "honcho"
MAX_STAGED_OPERATIONS = int(os.getenv("MAX_STAGED_OPERATIONS", 10))


async def _get_reconstructed_schema_arg(
    target_handler_func_obj: Callable,
    schema_arg_name: str | None,
    is_list_schema_arg: bool,
    raw_payload: dict | list | None,
    context_for_logging: str = "operation",
) -> Any | None:
    """Helper to reconstruct Pydantic model(s) from stored data."""
    if not schema_arg_name or raw_payload is None:
        return None

    reconstructed_value = None
    original_target_func = inspect.unwrap(target_handler_func_obj)
    try:
        annotations = inspect.get_annotations(
            original_target_func, globals=globals(), eval_str=True
        )
    except NameError as e:
        msg = f"Could not evaluate annotations for {original_target_func.__name__} to reconstruct schema for {context_for_logging}: {e}"
        logger.error(msg)
        raise ValueError(msg) from e

    param_type_hint = annotations.get(schema_arg_name)
    if not param_type_hint:
        msg = f"Could not find annotation for schema param '{schema_arg_name}' in {original_target_func.__name__} for {context_for_logging}"
        logger.error(msg)
        raise ValueError(msg)

    actual_item_type = None
    if is_list_schema_arg:
        if (
            hasattr(param_type_hint, "__origin__")
            and param_type_hint.__origin__ in (list, Sequence)
            and param_type_hint.__args__
        ):
            actual_item_type = param_type_hint.__args__[0]
        else:
            msg = f"Schema '{schema_arg_name}' in {original_target_func.__name__} marked as list, but type hint is not List[ModelType]: {param_type_hint} for {context_for_logging}"
            logger.error(msg)
            raise ValueError(msg)
    else:
        actual_item_type = param_type_hint

    if actual_item_type and hasattr(actual_item_type, "model_validate"):
        try:
            if is_list_schema_arg:
                if not isinstance(raw_payload, list):
                    msg = f"Payload for list schema '{schema_arg_name}' of {original_target_func.__name__} is not a list: {raw_payload} for {context_for_logging}"
                    logger.error(msg)
                    raise ValueError(msg)
                else:
                    reconstructed_value = [
                        actual_item_type.model_validate(item_data)
                        for item_data in raw_payload
                    ]
            else:
                if not isinstance(raw_payload, dict):
                    msg = f"Payload for schema '{schema_arg_name}' of {original_target_func.__name__} is not a dict: {raw_payload} for {context_for_logging}"
                    logger.error(msg)
                    raise ValueError(msg)
                else:
                    reconstructed_value = actual_item_type.model_validate(raw_payload)
        except Exception as e_reconstruct:
            logger.error(
                f"Error reconstructing schema for '{schema_arg_name}' in {original_target_func.__name__} during {context_for_logging}: {e_reconstruct}"
            )
            raise  # Propagate original Pydantic validation error or other exceptions
    elif actual_item_type:
        logger.warning(
            f"Type {actual_item_type} for '{schema_arg_name}' in {original_target_func.__name__} is not a Pydantic model for {context_for_logging}. Skipping reconstruction."
        )

    return reconstructed_value


def _extract_staging_params_and_payload(
    bound_arguments: inspect.BoundArguments,
    schema_payload_arg_name_from_decorator: str | None,
    is_list_payload_from_decorator: bool,
    db_session_param_name: str,
) -> tuple[dict, dict | list]:
    """Extracts parameters and payload for staging from bound arguments of a function call."""
    operation_params = {}
    operation_payload_data: dict | list = {}

    for name, value in bound_arguments.arguments.items():
        if name == db_session_param_name or name == "transaction_id":
            continue

        if (
            schema_payload_arg_name_from_decorator
            and name == schema_payload_arg_name_from_decorator
        ):
            if value is not None:
                if is_list_payload_from_decorator:
                    if isinstance(value, list) and all(
                        hasattr(item, "model_dump") for item in value
                    ):
                        operation_payload_data = [item.model_dump() for item in value]
                    else:
                        operation_payload_data = []
                else:
                    if hasattr(value, "model_dump"):
                        operation_payload_data = value.model_dump()

        else:
            if value is not None:
                operation_params[name] = value

    return operation_params, operation_payload_data


def _apply_forced_public_id(
    target_kwargs: dict,
    staged_op: models.StagedOperation,
    handler_func_obj: Callable,
    context_for_logging: str = "operation",
) -> None:
    """Modifies target_kwargs in place to include public_id if applicable."""
    if staged_op.resource_public_id:
        unwrapped_func = inspect.unwrap(handler_func_obj)
        func_sig = inspect.signature(unwrapped_func)
        if "public_id" in func_sig.parameters:
            target_kwargs["public_id"] = staged_op.resource_public_id
            logger.debug(
                f"Forcing public_id='{staged_op.resource_public_id}' for {context_for_logging} of {staged_op.handler_function}"
            )


def stageable(
    schema_payload_arg_name: Optional[str] = None, is_list_payload: bool = False
):
    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]):
        sig = inspect.signature(func)
        db_param_name = next(iter(sig.parameters))

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            db_session: AsyncSession = args[0]
            transaction_id_val: Optional[int] = kwargs.get("transaction_id")

            if transaction_id_val is not None:
                logger.debug(
                    f"Staging operation for function {func.__name__} within transaction {transaction_id_val}"
                )

                current_sequence_number = await get_next_sequence_number(
                    db_session, transaction_id_val
                )

                async with db_session.begin_nested() as func_execution_tx:
                    logger.debug(
                        f"Replaying prior operations for transaction {transaction_id_val} before executing {func.__name__}"
                    )

                    prior_staged_ops_stmt = (
                        select(models.StagedOperation)
                        .where(
                            models.StagedOperation.transaction_id == transaction_id_val
                        )
                        .where(
                            models.StagedOperation.sequence_number
                            < current_sequence_number
                        )
                        .order_by(models.StagedOperation.sequence_number)
                    )
                    prior_staged_ops_result = await db_session.execute(
                        prior_staged_ops_stmt
                    )
                    prior_staged_ops = prior_staged_ops_result.scalars().all()

                    for op in prior_staged_ops:
                        logger.debug(
                            f"Replaying staged op: {op.handler_function} (seq: {op.sequence_number}, res_id: {op.resource_public_id}) in tx {transaction_id_val}"
                        )
                        handler_func_to_replay_obj = globals().get(op.handler_function)
                        if not handler_func_to_replay_obj:
                            # This is a critical error for transaction integrity if a handler is missing.
                            msg = f"Handler function {op.handler_function} not found during replay."
                            logger.error(msg)
                            # Depending on desired behavior, could mark transaction as failed here too.
                            raise RuntimeError(msg)

                        replay_kwargs = {**op.parameters}
                        replay_kwargs.pop("transaction_id", None)

                        try:
                            reconstructed_schema_model = await _get_reconstructed_schema_arg(
                                handler_func_to_replay_obj,
                                op.schema_arg_name,
                                op.is_list_schema,
                                op.payload,
                                context_for_logging=f"replay of {op.handler_function}",
                            )
                            if (
                                reconstructed_schema_model is not None
                                and op.schema_arg_name
                            ):
                                replay_kwargs[op.schema_arg_name] = (
                                    reconstructed_schema_model
                                )

                            _apply_forced_public_id(
                                replay_kwargs,
                                op,
                                handler_func_to_replay_obj,
                                context_for_logging="replay",
                            )

                            await handler_func_to_replay_obj(
                                db_session, **replay_kwargs
                            )
                        except ValueError as e_replay_arg_prep:
                            # Error during Pydantic reconstruction or other value error from helpers
                            logger.error(
                                f"Error preparing arguments for replayed operation {op.handler_function}: {e_replay_arg_prep}"
                            )
                            # This error should lead to the failure of the overall staging operation.
                            # The func_execution_tx will be rolled back by the context manager's exit if an exception propagates.
                            raise  # Re-raise to ensure the staging of the current `func` call fails.

                    func_call_result_for_staging = await func(*args, **kwargs)

                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    operation_params_for_staging, operation_payload_for_staging = (
                        _extract_staging_params_and_payload(
                            bound_args,
                            schema_payload_arg_name,
                            is_list_payload,
                            db_param_name,
                        )
                    )
                    
                    if isinstance(func_call_result_for_staging, Select):
                        logger.debug(
                            f"func_call_result_for_staging is a Select: {func_call_result_for_staging}, therefore not rolling back"
                        )
                        return func_call_result_for_staging
                    
                    resource_public_id_for_staging = None
                    if func_call_result_for_staging and hasattr(
                        func_call_result_for_staging, "public_id"
                    ):
                        resource_public_id_for_staging = (
                            func_call_result_for_staging.public_id
                        )

                    # Rollback happens automatically if an exception (like ValueError from helpers) occurred above.
                    # If no exception, this explicit rollback is for the successful path of func execution.
                    await func_execution_tx.rollback()
                    logger.debug(
                        f"Rolled back nested transaction for {func.__name__} in transaction {transaction_id_val} after replay and execution."
                    )

                logger.debug(
                    f"Creating staged operation for {func.__name__}, sequence {current_sequence_number}"
                )
                await create_staged_operation(
                    db_session,
                    transaction_id_val,
                    current_sequence_number,
                    operation_params_for_staging,
                    operation_payload_for_staging,
                    func.__name__,
                    resource_public_id_for_staging,
                    schema_arg_name_for_db=schema_payload_arg_name,
                    is_list_schema_for_db=is_list_payload,
                )

                return func_call_result_for_staging
            else:
                original_func_result = await func(*args, **kwargs)

                if db_session.in_nested_transaction():
                    logger.debug(
                        f"Executed {func.__name__} within a nested transaction (replay context). Flushing."
                    )
                    await db_session.flush()
                elif not db_session.in_transaction():
                    logger.debug(
                        f"Executed {func.__name__} as a standalone operation (no active transaction). Committing."
                    )
                    await db_session.commit()
                else:
                    logger.debug(
                        f"Executed {func.__name__} within an existing main transaction (not staging related). Committing."
                    )
                    await db_session.commit()

                return original_func_result

        return wrapper

    return decorator


########################################################
# app methods
########################################################


@stageable()
async def get_app(
    db: AsyncSession,
    app_id: str,
    *,
    transaction_id: int | None = None,
) -> models.App:
    """
    Get an app by its ID.

    Args:
        db: Database session
        app_id: Public ID of the app
        transaction_id: Optional transaction ID for staging.

    Returns:
        The app if found

    Raises:
        ResourceNotFoundException: If the app does not exist
    """
    stmt = select(models.App).where(models.App.public_id == app_id)
    result = await db.execute(stmt)
    app = result.scalar_one_or_none()
    if app is None:
        logger.warning(f"App with ID {app_id} not found")
        raise ResourceNotFoundException(f"App with ID {app_id} not found")
    return app


async def get_all_apps(
    db: AsyncSession,
    reverse: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    """
    Get all apps.

    Args:
        db: Database session
        reverse: Whether to reverse the order of the apps
        filter: Filter the apps by a dictionary of metadata
    """
    stmt = select(models.App)
    if reverse:
        stmt = stmt.order_by(models.App.id.desc())
    else:
        stmt = stmt.order_by(models.App.id)
    if filter is not None:
        stmt = stmt.where(models.App.h_metadata.contains(filter))
    return stmt


@stageable()
async def get_app_by_name(
    db: AsyncSession, name: str, *, transaction_id: int | None = None
) -> models.App:
    """
    Get an app by its name.

    Args:
        db: Database session
        name: Name of the app
        transaction_id: Optional transaction ID for staging.

    Returns:
        The app if found

    Raises:
        ResourceNotFoundException: If the app does not exist
    """
    stmt = select(models.App).where(models.App.name == name)
    result = await db.execute(stmt)
    app = result.scalar_one_or_none()
    if app is None:
        logger.warning(f"App with name '{name}' not found")
        raise ResourceNotFoundException(f"App with name '{name}' not found")
    return app


@stageable(schema_payload_arg_name="app")
async def create_app(
    db: AsyncSession,
    app: schemas.AppCreate,
    *,
    transaction_id: int | None = None,
    public_id: str | None = None,
) -> models.App:
    """
    Create a new app.

    Args:
        db: Database session
        app: App creation schema
        transaction_id: Optional transaction ID for staging.
        public_id: Optional public ID for the app.
    Returns:
        The created app

    Raises:
        ConflictException: If an app with the same name already exists
    """
    try:
        honcho_app = models.App(
            name=app.name, h_metadata=app.metadata, public_id=public_id
        )
        db.add(honcho_app)
        await db.flush()
        await db.refresh(honcho_app)
        logger.info(f"App created successfully: {app.name}")
        return honcho_app
    except IntegrityError as e:
        await db.rollback()
        logger.error(f"IntegrityError creating app with name '{app.name}': {str(e)}")
        raise ConflictException(f"App with name '{app.name}' already exists") from e


@stageable(schema_payload_arg_name="app")
async def update_app(
    db: AsyncSession,
    app_id: str,
    app: schemas.AppUpdate,
    *,
    transaction_id: int | None = None,
) -> models.App:
    """
    Update an app.

    Args:
        db: Database session
        app_id: Public ID of the app
        app: App update schema
        transaction_id: Optional transaction ID for staging.
    Returns:
        The updated app

    Raises:
        ResourceNotFoundException: If the app does not exist
    """
    try:
        honcho_app = await get_app(db, app_id)

        if app.name is not None:
            honcho_app.name = app.name
        if app.metadata is not None:
            honcho_app.h_metadata = app.metadata

        await db.flush()
        await db.refresh(honcho_app)
        logger.info(f"App with ID {app_id} updated successfully")
        return honcho_app
    except IntegrityError as e:
        await db.rollback()
        logger.error(f"IntegrityError updating app {app_id}: {str(e)}")
        raise ConflictException(
            "App update failed - unique constraint violation"
        ) from e


########################################################
# user methods
########################################################


@stageable(schema_payload_arg_name="user")
async def create_user(
    db: AsyncSession,
    app_id: str,
    user: schemas.UserCreate,
    *,
    transaction_id: int | None = None,
    public_id: str | None = None,
) -> models.User:
    """
    Create a new user.

    Args:
        db: Database session
        app_id: Public ID of the app
        user: User creation schema
        transaction_id: Optional transaction ID for staging.
        public_id: Optional public ID for the user.
    Returns:
        The created user

    Raises:
        ConflictException: If a user with the same name already exists in this app
    """
    try:
        honcho_user = models.User(
            app_id=app_id,
            name=user.name,
            h_metadata=user.metadata,
            public_id=public_id,
        )
        db.add(honcho_user)
        await db.flush()
        await db.refresh(honcho_user)
        logger.info(f"User created successfully: {user.name} for app {app_id}")
        return honcho_user
    except IntegrityError as e:
        await db.rollback()
        logger.warning(f"Failed to create user - integrity error: {str(e)}")
        raise ConflictException("User with this name already exists") from e


@stageable()
async def get_user(
    db: AsyncSession, app_id: str, user_id: str, *, transaction_id: int | None = None
) -> models.User:
    """
    Get a user by app ID and user ID.

    Args:
        db: Database session
        app_id: Public ID of the app
        user_id: Public ID of the user
        transaction_id: Optional transaction ID for staging.

    Returns:
        The user if found

    Raises:
        ResourceNotFoundException: If the user does not exist
    """
    stmt = (
        select(models.User)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
    )
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    if user is None:
        logger.warning(f"User with ID '{user_id}' not found in app {app_id}")
        raise ResourceNotFoundException(f"User with ID '{user_id}' not found")
    return user


@stageable()
async def get_user_by_name(
    db: AsyncSession, app_id: str, name: str, *, transaction_id: int | None = None
) -> models.User:
    """
    Get a user by app ID and name.

    Args:
        db: Database session
        app_id: Public ID of the app
        name: Name of the user
        transaction_id: Optional transaction ID for staging.

    Returns:
        The user if found

    Raises:
        ResourceNotFoundException: If the user does not exist
    """
    stmt = (
        select(models.User)
        .where(models.User.app_id == app_id)
        .where(models.User.name == name)
    )
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    if user is None:
        logger.warning(f"User with name '{name}' not found in app {app_id}")
        raise ResourceNotFoundException(f"User with name '{name}' not found")
    return user


async def get_users(
    db: AsyncSession,
    app_id: str,
    reverse: bool = False,
    filter: Optional[dict] = None,
) -> Select:
    stmt = select(models.User).where(models.User.app_id == app_id)

    if filter is not None:
        stmt = stmt.where(models.User.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.User.id.desc())
    else:
        stmt = stmt.order_by(models.User.id)

    return stmt


@stageable(schema_payload_arg_name="user")
async def update_user(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    user: schemas.UserUpdate,
    *,
    transaction_id: int | None = None,
) -> models.User:
    """
    Update a user.

    Args:
        db: Database session
        app_id: Public ID of the app
        user_id: Public ID of the user
        user: User update schema
        transaction_id: Optional transaction ID for staging.

    Returns:
        The updated user

    Raises:
        ResourceNotFoundException: If the user does not exist
        ValidationException: If the update data is invalid
        ConflictException: If the update violates a unique constraint
    """
    try:
        # get_user will raise ResourceNotFoundException if not found
        honcho_user = await get_user(db, app_id, user_id)

        if user.name is not None:
            honcho_user.name = user.name
        if user.metadata is not None:
            honcho_user.h_metadata = user.metadata

        await db.flush()
        await db.refresh(honcho_user)
        logger.info(f"User {user_id} updated successfully")
        return honcho_user
    except IntegrityError as e:
        await db.rollback()
        logger.warning(f"User update failed due to integrity error: {str(e)}")
        raise ConflictException(
            "User update failed - unique constraint violation"
        ) from e


########################################################
# session methods
########################################################


@stageable()
async def get_session(
    db: AsyncSession,
    app_id: str,
    session_id: str,
    user_id: Optional[str] = None,
    *,
    transaction_id: int | None = None,
) -> models.Session:
    """
    Get a session by ID for a specific user and app.

    Args:
        db: Database session
        app_id: Public ID of the app
        session_id: Public ID of the session
        user_id: Optional public ID of the user
        transaction_id: Optional transaction ID for staging.

    Returns:
        The session if found

    Raises:
        ResourceNotFoundException: If the session does not exist or doesn't belong to the user
    """
    stmt = (
        select(models.Session)
        .join(models.User, models.User.public_id == models.Session.user_id)
        .where(models.User.app_id == app_id)
        .where(models.Session.public_id == session_id)
    )
    if user_id is not None:
        stmt = stmt.where(models.Session.user_id == user_id)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()
    if session is None:
        logger.warning(f"Session with ID '{session_id}' not found for user {user_id}")
        raise ResourceNotFoundException("Session not found or does not belong to user")
    return session


async def get_sessions(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    reverse: Optional[bool] = False,
    is_active: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    stmt = (
        select(models.Session)
        .join(models.User, models.User.public_id == models.Session.user_id)
        .where(models.User.app_id == app_id)
        .where(models.Session.user_id == user_id)
    )

    if is_active:
        stmt = stmt.where(models.Session.is_active.is_(True))

    if filter is not None:
        stmt = stmt.where(models.Session.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.Session.id.desc())
    else:
        stmt = stmt.order_by(models.Session.id)

    return stmt


@stageable(schema_payload_arg_name="session")
async def create_session(
    db: AsyncSession,
    session: schemas.SessionCreate,
    app_id: str,
    user_id: str,
    *,
    transaction_id: int | None = None,
    public_id: str | None = None,
) -> models.Session:
    """
    Create a new session for a user.

    Args:
        db: Database session
        session: Session creation schema
        app_id: ID of the app
        user_id: ID of the user
        transaction_id: Optional transaction ID for staging.
        public_id: Optional public ID for the session.
    Returns:
        The created session

    Raises:
        ResourceNotFoundException: If the user does not exist
    """
    try:
        # This will raise ResourceNotFoundException if user not found
        _honcho_user = await get_user(db, app_id=app_id, user_id=user_id)

        honcho_session = models.Session(
            user_id=user_id,
            h_metadata=session.metadata,
            public_id=public_id,
        )
        db.add(honcho_session)
        await db.flush()
        await db.refresh(honcho_session)
        logger.info(f"Session created successfully for user {user_id}")
        return honcho_session
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating session for user {user_id}: {str(e)}")
        raise


@stageable(schema_payload_arg_name="session")
async def update_session(
    db: AsyncSession,
    session: schemas.SessionUpdate,
    app_id: str,
    user_id: str,
    session_id: str,
    *,
    transaction_id: int | None = None,
) -> models.Session:
    """
    Update a session.

    Args:
        db: Database session
        session: Session update schema
        app_id: ID of the app
        user_id: ID of the user
        session_id: ID of the session
        transaction_id: Optional transaction ID for staging.

    Returns:
        The updated session

    Raises:
        ResourceNotFoundException: If the session does not exist or doesn't belong to the user
    """
    honcho_session = await get_session(
        db, app_id=app_id, session_id=session_id, user_id=user_id
    )
    if honcho_session is None:
        logger.warning(f"Session {session_id} not found for user {user_id}")
        raise ResourceNotFoundException("Session not found or does not belong to user")

    if (
        session.metadata is not None
    ):  # Need to explicitly be there won't make it empty by default
        honcho_session.h_metadata = session.metadata

    await db.flush()
    await db.refresh(honcho_session)
    logger.info(f"Session {session_id} updated successfully")
    return honcho_session


@stageable()
async def delete_session(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    session_id: str,
    *,
    transaction_id: int | None = None,
) -> bool:
    """
    Mark a session as inactive (soft delete).

    Args:
        db: Database session
        app_id: ID of the app
        user_id: ID of the user
        session_id: ID of the session
        transaction_id: Optional transaction ID for staging.

    Returns:
        True if the session was deleted successfully

    Raises:
        ResourceNotFoundException: If the session does not exist or doesn't belong to the user
    """
    stmt = (
        select(models.Session)
        .join(models.User, models.User.public_id == models.Session.user_id)
        .where(models.Session.public_id == session_id)
        .where(models.User.app_id == app_id)
        .where(models.Session.user_id == user_id)
    )
    result = await db.execute(stmt)
    honcho_session = result.scalar_one_or_none()

    if honcho_session is None:
        logger.warning(f"Session {session_id} not found for user {user_id}")
        raise ResourceNotFoundException("Session not found or does not belong to user")

    honcho_session.is_active = False
    logger.info(f"Session {session_id} marked as inactive")
    return True


async def clone_session(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    original_session_id: str,
    cutoff_message_id: Optional[str] = None,
    deep_copy: bool = True,
) -> models.Session:
    """
    Clone a session and its messages. If cutoff_message_id is provided,
    only clone messages up to and including that message.

    Args:
        db: SQLAlchemy session
        app_id: ID of the app the target session is in
        user_id: ID of the user the target session belongs to
        original_session_id: ID of the session to clone
        cutoff_message_id: Optional ID of the last message to include in the clone
        deep_copy: bool for deep copy of metamessages

    Returns:
        The newly created session
    """
    # Get the original session
    stmt = (
        select(models.Session)
        .join(models.User, models.User.public_id == models.Session.user_id)
        .where(models.Session.public_id == original_session_id)
        .where(models.Session.user_id == user_id)
        .where(models.User.app_id == app_id)
    )
    original_session = await db.scalar(stmt)
    if not original_session:
        raise ValueError("Original session not found")

    # If cutoff_message_id is provided, verify it belongs to the session
    cutoff_message = None
    if cutoff_message_id is not None:
        stmt = select(models.Message).where(
            models.Message.public_id == cutoff_message_id,
            models.Message.session_id == original_session_id,
        )
        cutoff_message = await db.scalar(stmt)
        if not cutoff_message:
            raise ValueError(
                "Message not found or doesn't belong to the specified session"
            )

    # Create new session
    new_session = models.Session(
        user_id=original_session.user_id,
        h_metadata=original_session.h_metadata,
    )
    db.add(new_session)
    await db.flush()  # Flush to get the new session ID

    # Build query for messages to clone
    stmt = select(models.Message).where(
        models.Message.session_id == original_session_id
    )
    if cutoff_message_id is not None and cutoff_message is not None:
        stmt = stmt.where(models.Message.id <= cast(cutoff_message.id, BigInteger))
    stmt = stmt.order_by(models.Message.id)

    # Fetch messages to clone
    messages_to_clone_scalars = await db.scalars(stmt)
    messages_to_clone = messages_to_clone_scalars.all()

    if not messages_to_clone:
        return new_session

    # Prepare bulk insert data
    new_messages = [
        {
            "session_id": new_session.public_id,
            "content": message.content,
            "is_user": message.is_user,
            "h_metadata": message.h_metadata,
        }
        for message in messages_to_clone
    ]

    stmt = insert(models.Message).returning(models.Message.public_id)
    result = await db.execute(stmt, new_messages)
    new_message_ids = result.scalars().all()

    # Create mapping of old to new message IDs
    message_id_map = dict(
        zip([message.public_id for message in messages_to_clone], new_message_ids)
    )

    # Handle metamessages if deep copy is requested
    if deep_copy:
        # Fetch all metamessages tied to the session in a single query
        stmt = select(models.Metamessage).where(
            models.Metamessage.session_id == original_session_id
        )
        if cutoff_message_id is not None and cutoff_message is not None:
            # Only get metamessages related to messages we're cloning
            message_ids = [message.public_id for message in messages_to_clone]
            stmt = stmt.where(
                (models.Metamessage.message_id.is_(None))
                | (models.Metamessage.message_id.in_(message_ids))
            )

        metamessages_result = await db.scalars(stmt)
        metamessages = metamessages_result.all()

        if metamessages:
            # Prepare bulk insert data for metamessages
            new_metamessages = []

            for meta in metamessages:
                # Base metamessage data
                meta_data = {
                    "user_id": meta.user_id,  # Preserve original user
                    "session_id": new_session.public_id,
                    "metamessage_type": meta.metamessage_type,
                    "content": meta.content,
                    "h_metadata": meta.h_metadata,
                }

                # If the metamessage was tied to a message, tie it to the corresponding new message
                if meta.message_id is not None and meta.message_id in message_id_map:
                    meta_data["message_id"] = message_id_map[meta.message_id]

                new_metamessages.append(meta_data)

            # Bulk insert metamessages using modern insert syntax
            if new_metamessages:
                stmt = insert(models.Metamessage)
                await db.execute(stmt, new_metamessages)

    await db.commit()

    return new_session


########################################################
# Message Methods
########################################################


@stageable(schema_payload_arg_name="message")
async def create_message(
    db: AsyncSession,
    message: schemas.MessageCreate,
    app_id: str,
    user_id: str,
    session_id: str,
    *,
    transaction_id: int | None = None,
    public_id: str | None = None,
) -> models.Message:
    _honcho_session = await get_session(
        db, app_id=app_id, session_id=session_id, user_id=user_id
    )
    honcho_message = models.Message(
        session_id=session_id,
        is_user=message.is_user,
        content=message.content,
        h_metadata=message.metadata,
        public_id=public_id,
    )
    db.add(honcho_message)
    await db.flush()
    await db.refresh(honcho_message)
    return honcho_message


@stageable(schema_payload_arg_name="messages", is_list_payload=True)
async def create_messages(
    db: AsyncSession,
    messages: list[schemas.MessageCreate],
    app_id: str,
    user_id: str,
    session_id: str,
    *,
    transaction_id: int | None = None,
) -> list[models.Message]:
    """
    Bulk create messages for a session while maintaining order.

    Args:
        transaction_id: Optional transaction ID for staging.
    """
    # Verify session exists and belongs to user
    _honcho_session = await get_session(
        db, app_id=app_id, session_id=session_id, user_id=user_id
    )

    # Create list of message records
    message_records = [
        {
            "session_id": session_id,
            "is_user": message.is_user,
            "content": message.content,
            "h_metadata": message.metadata,
        }
        for message in messages
    ]

    # Bulk insert messages and return them in order
    stmt = insert(models.Message).returning(models.Message)
    result = await db.execute(stmt, message_records)

    return list(result.scalars().all())


async def get_messages(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    session_id: str,
    reverse: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    stmt = (
        select(models.Message)
        .join(models.Session, models.Session.public_id == models.Message.session_id)
        .join(models.User, models.User.public_id == models.Session.user_id)
        .join(models.App, models.App.public_id == models.User.app_id)
        .where(models.App.public_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Message.session_id == session_id)
    )

    if filter is not None:
        stmt = stmt.where(models.Message.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.Message.id.desc())
    else:
        stmt = stmt.order_by(models.Message.id)

    return stmt


@stageable()
async def get_message(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    session_id: str,
    message_id: str,
    *,
    transaction_id: int | None = None,
) -> Optional[models.Message]:
    stmt = (
        select(models.Message)
        .join(models.Session, models.Session.public_id == models.Message.session_id)
        .join(models.User, models.User.public_id == models.Session.user_id)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Session.public_id == session_id)
        .where(models.Message.public_id == message_id)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


@stageable(schema_payload_arg_name="message")
async def update_message(
    db: AsyncSession,
    message: schemas.MessageUpdate,
    app_id: str,
    user_id: str,
    session_id: str,
    message_id: str,
    *,
    transaction_id: int | None = None,
) -> models.Message:
    honcho_message = await get_message(
        db, app_id=app_id, session_id=session_id, user_id=user_id, message_id=message_id
    )
    if honcho_message is None:
        raise ValueError("Message not found or does not belong to user")
    if (
        message.metadata is not None
    ):  # Need to explicitly be there won't make it empty by default
        honcho_message.h_metadata = message.metadata

    await db.flush()
    await db.refresh(honcho_message)
    return honcho_message


########################################################
# metamessage methods
########################################################


@stageable(schema_payload_arg_name="metamessage")
async def create_metamessage(
    db: AsyncSession,
    user_id: str,
    metamessage: schemas.MetamessageCreate,
    app_id: str,
    *,
    transaction_id: int | None = None,
) -> models.Metamessage:
    # Validate user exists
    _user = await get_user(db, app_id=app_id, user_id=user_id)

    metamessage_data = {
        "user_id": user_id,
        "metamessage_type": metamessage.metamessage_type,
        "content": metamessage.content,
        "h_metadata": metamessage.metadata,
    }

    # Validate session_id if provided
    if metamessage.session_id is not None:
        _session = await get_session(
            db, app_id=app_id, user_id=user_id, session_id=metamessage.session_id
        )
        metamessage_data["session_id"] = metamessage.session_id

        # Validate message_id if provided
        if metamessage.message_id is not None:
            _message = await get_message(
                db,
                app_id=app_id,
                session_id=metamessage.session_id,
                user_id=user_id,
                message_id=metamessage.message_id,
            )
            metamessage_data["message_id"] = metamessage.message_id
    elif metamessage.message_id is not None:
        # If message_id provided but no session_id, that's an error
        raise ValidationException("Cannot specify message_id without session_id")

    honcho_metamessage = models.Metamessage(**metamessage_data)
    db.add(honcho_metamessage)
    await db.flush()
    await db.refresh(honcho_metamessage)
    return honcho_metamessage


async def get_metamessages(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    session_id: Optional[str] = None,
    message_id: Optional[str] = None,
    metamessage_type: Optional[str] = None,
    filter: Optional[dict] = None,
    reverse: Optional[bool] = False,
) -> Select:
    # Base query starts with metamessage and user relationship
    stmt = (
        select(models.Metamessage)
        .join(models.User, models.User.public_id == models.Metamessage.user_id)
        .join(models.App, models.App.public_id == models.User.app_id)
        .where(models.App.public_id == app_id)
        .where(models.User.public_id == user_id)
    )

    # If session_id is provided, filter by it
    if session_id is not None:
        stmt = stmt.where(models.Metamessage.session_id == session_id)

    # If message_id is provided, filter by it
    if message_id is not None:
        stmt = stmt.where(models.Metamessage.message_id == message_id)

    # Filter by metamessage_type if provided
    if metamessage_type is not None:
        stmt = stmt.where(models.Metamessage.metamessage_type == metamessage_type)

    # Apply metadata filter if provided
    if filter is not None:
        stmt = stmt.where(models.Metamessage.h_metadata.contains(filter))

    # Apply sort order
    if reverse:
        stmt = stmt.order_by(models.Metamessage.id.desc())
    else:
        stmt = stmt.order_by(models.Metamessage.id)

    return stmt


@stageable()
async def get_metamessage(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    metamessage_id: str,
    session_id: Optional[str] = None,
    message_id: Optional[str] = None,
    *,
    transaction_id: int | None = None,
) -> Optional[models.Metamessage]:
    # Base query for metamessage by ID
    stmt = (
        select(models.Metamessage)
        .join(models.User, models.User.public_id == models.Metamessage.user_id)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Metamessage.public_id == metamessage_id)
    )

    # Add optional filters
    if session_id is not None:
        stmt = stmt.where(models.Metamessage.session_id == session_id)
    if message_id is not None:
        stmt = stmt.where(models.Metamessage.message_id == message_id)

    result = await db.execute(stmt)
    return result.scalar_one_or_none()


@stageable(schema_payload_arg_name="metamessage")
async def update_metamessage(
    db: AsyncSession,
    metamessage: schemas.MetamessageUpdate,
    app_id: str,
    user_id: str,
    metamessage_id: str,
    *,
    transaction_id: int | None = None,
) -> models.Metamessage:
    metamessage_obj = await get_metamessage(
        db, app_id=app_id, user_id=user_id, metamessage_id=metamessage_id
    )
    if metamessage_obj is None:
        raise ResourceNotFoundException(
            f"Metamessage with ID {metamessage_id} not found"
        )

    if metamessage.message_id is not None and metamessage.session_id is None:
        metamessage.session_id = metamessage_obj.session_id
        if metamessage.session_id is None:
            raise ValidationException("Cannot specify message_id without session_id")

    if metamessage.session_id is not None and metamessage.message_id is not None:
        _message = await get_message(
            db,
            app_id=app_id,
            session_id=metamessage.session_id,
            user_id=user_id,
            message_id=metamessage.message_id,
        )
        # Assuming get_message raises if not found.

    if metamessage.session_id is not None:
        metamessage_obj.session_id = metamessage.session_id
    if metamessage.message_id is not None:
        metamessage_obj.message_id = metamessage.message_id
    if metamessage.metadata is not None:
        metamessage_obj.h_metadata = metamessage.metadata
    if metamessage.metamessage_type is not None:
        metamessage_obj.metamessage_type = metamessage.metamessage_type

    await db.flush()
    await db.refresh(metamessage_obj)
    return metamessage_obj


########################################################
# collection methods
########################################################


async def get_collections(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    reverse: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    """Get a distinct list of the names of collections associated with a user
    Args:
        transaction_id: Optional transaction ID for staging.
    """
    stmt = (
        select(models.Collection)
        .join(models.User, models.User.public_id == models.Collection.user_id)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
    )

    if filter is not None:
        stmt = stmt.where(models.Collection.h_metadata.contains(filter))
    if reverse:
        stmt = stmt.order_by(models.Collection.id.desc())
    else:
        stmt = stmt.order_by(models.Collection.id)

    return stmt


@stageable()
async def get_collection_by_id(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    collection_id: str,
    *,
    transaction_id: int | None = None,
) -> models.Collection:
    """
    Get a collection by ID for a specific user and app.

    Args:
        db: Database session
        app_id: Public ID of the app
        user_id: Public ID of the user
        collection_id: Public ID of the collection
        transaction_id: Optional transaction ID for staging.

    Returns:
        The collection if found

    Raises:
        ResourceNotFoundException: If the collection does not exist
    """
    stmt = (
        select(models.Collection)
        .join(models.User, models.User.public_id == models.Collection.user_id)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Collection.public_id == collection_id)
    )
    result = await db.execute(stmt)
    collection = result.scalar_one_or_none()
    if collection is None:
        logger.warning(
            f"Collection with ID '{collection_id}' not found for user {user_id}"
        )
        raise ResourceNotFoundException(
            "Collection not found or does not belong to user"
        )
    return collection


@stageable()
async def get_collection_by_name(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    name: str,
    *,
    transaction_id: int | None = None,
) -> models.Collection:
    """
    Get a collection by name for a specific user and app.

    Args:
        db: Database session
        app_id: Public ID of the app
        user_id: Public ID of the user
        name: Name of the collection
        transaction_id: Optional transaction ID for staging.

    Returns:
        The collection if found

    Raises:
        ResourceNotFoundException: If the collection does not exist
    """
    stmt = (
        select(models.Collection)
        .join(models.User, models.User.public_id == models.Collection.user_id)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Collection.name == name)
    )
    result = await db.execute(stmt)
    collection = result.scalar_one_or_none()
    if collection is None:
        logger.warning(f"Collection with name '{name}' not found for user {user_id}")
        raise ResourceNotFoundException(f"Collection with name '{name}' not found")
    return collection


@stageable(schema_payload_arg_name="collection")
async def create_collection(
    db: AsyncSession,
    collection: schemas.CollectionCreate,
    app_id: str,
    user_id: str,
    *,
    transaction_id: int | None = None,
    public_id: str | None = None,
) -> models.Collection:
    """
    Create a new collection for a user.

    Args:
        db: Database session
        collection: Collection creation schema
        app_id: ID of the app
        user_id: ID of the user
        transaction_id: Optional transaction ID for staging.
        public_id: Optional public ID for the collection.
    Returns:
        The created collection

    Raises:
        ConflictException: If a collection with the same name already exists for this user
        ValidationException: If the collection configuration is invalid
        ResourceNotFoundException: If the user does not exist
    """
    try:
        _user = await get_user(db, app_id=app_id, user_id=user_id)

        if collection.name == "honcho":
            logger.warning(
                f"Attempted to create collection with reserved name 'honcho' for user {user_id}"
            )
            raise ValidationException(
                "Invalid collection configuration - 'honcho' is a reserved name"
            )

        honcho_collection = models.Collection(
            user_id=user_id,  # from func param
            name=collection.name,
            h_metadata=collection.metadata,
            public_id=public_id,
        )
        db.add(honcho_collection)
        await db.flush()
        await db.refresh(honcho_collection)
        logger.info(
            f"Collection '{collection.name}' created successfully for user {user_id}"
        )
        return honcho_collection
    except IntegrityError as e:
        await db.rollback()
        logger.warning(f"Failed to create collection - integrity error: {str(e)}")
        raise ConflictException(
            f"Collection with name '{collection.name}' already exists"
        ) from e


async def create_user_protected_collection(
    db: AsyncSession,
    app_id: str,
    user_id: str,
) -> models.Collection:
    honcho_collection = models.Collection(
        user_id=user_id,
        name=DEF_PROTECTED_COLLECTION_NAME,
    )
    try:
        db.add(honcho_collection)
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise ValueError("Collection already exists") from None
    return honcho_collection


async def get_or_create_user_protected_collection(
    db: AsyncSession,
    app_id: str,
    user_id: str,
) -> models.Collection:
    try:
        honcho_collection = await get_collection_by_name(
            db, app_id, user_id, DEF_PROTECTED_COLLECTION_NAME
        )
        return honcho_collection
    except ResourceNotFoundException:
        honcho_collection = await create_user_protected_collection(db, app_id, user_id)
        return honcho_collection


@stageable(schema_payload_arg_name="collection")
async def update_collection(
    db: AsyncSession,
    collection: schemas.CollectionUpdate,
    app_id: str,
    user_id: str,
    collection_id: str,
    *,
    transaction_id: int | None = None,
) -> models.Collection:
    """
    Update a collection.

    Args:
        db: Database session
        collection: Collection update schema
        app_id: ID of the app
        user_id: ID of the user
        collection_id: ID of the collection
        transaction_id: Optional transaction ID for staging.

    Returns:
        The updated collection

    Raises:
        ResourceNotFoundException: If the collection does not exist
        ValidationException: If the update data is invalid
        ConflictException: If the update violates a unique constraint
    """
    try:
        # Validate input
        if collection.name is None and collection.metadata is None:
            logger.warning(
                f"Collection update attempted with no fields provided for collection {collection_id}"
            )
            raise ValidationException(
                "Invalid collection configuration - at least one field must be provided"
            )

        # This will raise ResourceNotFoundException if not found
        honcho_collection = await get_collection_by_id(
            db, app_id=app_id, user_id=user_id, collection_id=collection_id
        )

        # Check for reserved names if name is being updated
        if collection.name == "honcho":
            logger.warning(
                f"Attempted to rename collection to reserved name 'honcho' for user {user_id}"
            )
            raise ValidationException(
                "Invalid collection configuration - 'honcho' is a reserved name"
            )

        if collection.metadata is not None:
            honcho_collection.h_metadata = collection.metadata

        if collection.name is not None:
            honcho_collection.name = collection.name

        await db.flush()
        await db.refresh(honcho_collection)
        logger.info(f"Collection {collection_id} updated successfully")
        return honcho_collection
    except IntegrityError as e:
        await db.rollback()
        logger.warning(f"Collection update failed due to integrity error: {str(e)}")
        raise ConflictException("Collection update failed - name already in use") from e


@stageable()
async def delete_collection(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    collection_id: str,
    *,
    transaction_id: int | None = None,
) -> bool:
    """
    Delete a Collection and all documents associated with it. Takes advantage of
    the orm cascade feature.

    Args:
        db: Database session
        app_id: ID of the app
        user_id: ID of the user
        collection_id: ID of the collection
        transaction_id: Optional transaction ID for staging.

    Returns:
        True if the collection was deleted successfully

    Raises:
        ResourceNotFoundException: If the collection does not exist
    """
    try:
        # This will raise ResourceNotFoundException if not found
        honcho_collection = await get_collection_by_id(
            db, app_id=app_id, user_id=user_id, collection_id=collection_id
        )

        if honcho_collection.name == "honcho":
            logger.warning(
                f"Attempted to delete collection with reserved name 'honcho' for user {user_id}"
            )
            raise ValidationException(
                "Invalid collection configuration - 'honcho' is a reserved name"
            )

        await db.delete(honcho_collection)
        logger.info(f"Collection {collection_id} deleted successfully")
        return True
    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting collection {collection_id}: {str(e)}")
        raise


########################################################
# document methods
########################################################

# Should be similar to the messages methods outside of query


async def get_documents(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    collection_id: str,
    reverse: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    stmt = (
        select(models.Document)
        .join(
            models.Collection,
            models.Collection.public_id == models.Document.collection_id,
        )
        .join(models.User, models.User.public_id == models.Collection.user_id)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Document.collection_id == collection_id)
    )

    if filter is not None:
        stmt = stmt.where(models.Document.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.Document.id.desc())
    else:
        stmt = stmt.order_by(models.Document.id)

    return stmt


@stageable()
async def get_document(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    collection_id: str,
    document_id: str,
    *,
    transaction_id: int | None = None,
) -> models.Document:
    """
    Get a document by ID.

    Args:
        db: Database session
        app_id: Public ID of the app
        user_id: Public ID of the user
        collection_id: Public ID of the collection
        document_id: Public ID of the document
        transaction_id: Optional transaction ID for staging.

    Returns:
        The document if found

    Raises:
        ResourceNotFoundException: If the document does not exist
    """
    stmt = (
        select(models.Document)
        .join(
            models.Collection,
            models.Collection.public_id == models.Document.collection_id,
        )
        .join(models.User, models.User.public_id == models.Collection.user_id)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Document.collection_id == collection_id)
        .where(models.Document.public_id == document_id)
    )

    result = await db.execute(stmt)
    document = result.scalar_one_or_none()
    if document is None:
        logger.warning(
            f"Document with ID '{document_id}' not found in collection {collection_id}"
        )
        raise ResourceNotFoundException(f"Document with ID '{document_id}' not found")
    return document


async def query_documents(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    collection_id: str,
    query: str,
    filter: Optional[dict] = None,
    max_distance: Optional[float] = None,
    top_k: int = 5,
) -> Sequence[models.Document]:
    # Using async client with await
    response = await openai_client.embeddings.create(
        model="text-embedding-3-small", input=query
    )
    embedding_query = response.data[0].embedding
    stmt = (
        select(models.Document)
        .join(
            models.Collection,
            models.Collection.public_id == models.Document.collection_id,
        )
        .join(models.User, models.User.public_id == models.Collection.user_id)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Document.collection_id == collection_id)
        # .limit(top_k)
    )
    if max_distance is not None:
        stmt = stmt.where(
            models.Document.embedding.cosine_distance(embedding_query) < max_distance
        )
    if filter is not None:
        stmt = stmt.where(models.Document.h_metadata.contains(filter))
    stmt = stmt.limit(top_k).order_by(
        models.Document.embedding.cosine_distance(embedding_query)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


@stageable(schema_payload_arg_name="document")
async def create_document(
    db: AsyncSession,
    document: schemas.DocumentCreate,
    app_id: str,
    user_id: str,
    collection_id: str,
    duplicate_threshold: Optional[float] = None,
    *,
    transaction_id: int | None = None,
    public_id: str | None = None,
) -> models.Document:
    """
    Embed text as a vector and create a document.

    Args:
        db: Database session
        document: Document creation schema
        app_id: ID of the app
        user_id: ID of the user
        collection_id: ID of the collection
        duplicate_threshold: Optional threshold for preventing duplicates.
        transaction_id: Optional transaction ID for staging.
        public_id: Optional public ID for the document.
    Returns:
        The created document

    Raises:
        ResourceNotFoundException: If the collection does not exist
        ValidationException: If the document data is invalid
    """
    _collection = await get_collection_by_id(
        db, app_id=app_id, collection_id=collection_id, user_id=user_id
    )

    response = await openai_client.embeddings.create(
        input=document.content, model="text-embedding-3-small"
    )

    embedding = response.data[0].embedding

    if duplicate_threshold is not None:
        # Check if there are duplicates within the threshold
        stmt = (
            select(models.Document)
            .where(models.Document.collection_id == collection_id)
            .where(
                models.Document.embedding.cosine_distance(embedding)
                < duplicate_threshold
            )
            .order_by(models.Document.embedding.cosine_distance(embedding))
            .limit(1)
        )
        result = await db.execute(stmt)
        duplicate = result.scalar_one_or_none()  # Get the closest match if any exist
        if duplicate is not None:
            logger.info(f"Duplicate found: {duplicate.content}. Ignoring new document.")
            return duplicate

    honcho_document = models.Document(
        collection_id=collection_id,
        content=document.content,
        h_metadata=document.metadata,
        embedding=embedding,
        public_id=public_id,
    )
    db.add(honcho_document)
    await db.flush()
    await db.refresh(honcho_document)
    return honcho_document


@stageable(schema_payload_arg_name="document")
async def update_document(
    db: AsyncSession,
    document: schemas.DocumentUpdate,
    app_id: str,
    user_id: str,
    collection_id: str,
    document_id: str,
    *,
    transaction_id: int | None = None,
) -> models.Document:
    honcho_document = await get_document(
        db,
        app_id=app_id,
        collection_id=collection_id,
        user_id=user_id,
        document_id=document_id,
    )
    if honcho_document is None:
        raise ResourceNotFoundException("Document not found or does not belong to user")
    if document.content is not None:
        honcho_document.content = document.content
        # Using async client with await
        response = await openai_client.embeddings.create(
            input=document.content, model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        honcho_document.embedding = embedding
        honcho_document.created_at = func.now()

    if document.metadata is not None:
        honcho_document.h_metadata = document.metadata

    await db.flush()
    await db.refresh(honcho_document)
    return honcho_document


@stageable()
async def delete_document(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    collection_id: str,
    document_id: str,
    *,
    transaction_id: int | None = None,
) -> bool:
    _honcho_collection = await get_collection_by_id(
        db, app_id=app_id, collection_id=collection_id, user_id=user_id
    )
    if _honcho_collection is None:
        raise ResourceNotFoundException("Collection or Document not found")
    if _honcho_collection.name == "honcho":
        logger.warning(
            f"Attempted to delete collection with reserved name 'honcho' for user {user_id}"
        )
        raise ValidationException(
            "Cannot delete collection with reserved name 'honcho'"
        )
    stmt = (
        select(models.Document)
        .join(
            models.Collection,
            models.Collection.public_id == models.Document.collection_id,
        )
        .join(models.User, models.User.public_id == models.Collection.user_id)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Document.collection_id == collection_id)
        .where(models.Document.public_id == document_id)
    )
    result = await db.execute(stmt)
    document = result.scalar_one_or_none()
    if document is None:
        return False
    await db.delete(document)
    return True


async def get_duplicate_documents(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    collection_id: str,
    content: str,
    similarity_threshold: float = 0.85,
) -> list[models.Document]:
    """Check if a document with similar content already exists in the collection.

    Args:
        db: Database session
        app_id: Application ID
        user_id: User ID
        collection_id: Collection ID
        content: Document content to check for duplicates
        similarity_threshold: Similarity threshold (0-1) for considering documents as duplicates

    Returns:
        List of documents that are similar to the provided content
    """
    # Get embedding for the content
    # Using async client with await
    response = await openai_client.embeddings.create(
        input=content, model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding

    # Find documents with similar embeddings
    stmt = (
        select(models.Document)
        .where(models.Document.collection_id == collection_id)
        .where(
            models.Document.embedding.cosine_distance(embedding)
            < (1 - similarity_threshold)
        )  # Convert similarity to distance
        .order_by(models.Document.embedding.cosine_distance(embedding))
    )

    result = await db.execute(stmt)
    return list(result.scalars().all())  # Convert to list to match the return type


########################################################
# transaction methods
########################################################
# These methods are NOT decorated with @stageable


async def create_transaction(
    db: AsyncSession, expires_at: datetime.datetime | None = None
) -> models.Transaction:
    transaction = models.Transaction(
        status="pending",
        expires_at=expires_at,
    )
    db.add(transaction)
    await db.commit()
    return transaction


async def get_transaction(db: AsyncSession, transaction_id: int) -> models.Transaction:
    stmt = select(models.Transaction).where(
        models.Transaction.transaction_id == transaction_id
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def commit_transaction(db: AsyncSession, transaction_id: int) -> None:
    stmt = select(models.Transaction).where(
        models.Transaction.transaction_id == transaction_id
    )
    result = await db.execute(stmt)
    transaction = result.scalar_one_or_none()
    if transaction is None:
        raise ResourceNotFoundException("Transaction not found")
    if transaction.status != "pending":
        raise ValidationException("Transaction is not pending")
    if transaction.expires_at < datetime.datetime.now(datetime.timezone.utc):
        transaction.status = "expired"
        await db.commit()
        raise ValidationException("Transaction has expired")

    staged_operations = await get_staged_operations(db, transaction_id)

    if not db.in_transaction():
        await db.begin()
        logger.warning(
            "commit_transaction was called with a db session not in an active transaction. Began a new one."
        )

    for staged_op in staged_operations:
        handler_func_obj = globals().get(staged_op.handler_function)
        if not handler_func_obj:
            logger.error(
                f"Handler function {staged_op.handler_function} not found during commit."
            )
            await db.rollback()
            transaction.status = "failed"
            await db.commit()
            raise RuntimeError(
                f"Handler function {staged_op.handler_function} not found."
            )

        commit_kwargs = {**staged_op.parameters}
        commit_kwargs.pop("transaction_id", None)

        try:
            reconstructed_schema_model = await _get_reconstructed_schema_arg(
                handler_func_obj,
                staged_op.schema_arg_name,
                staged_op.is_list_schema,
                staged_op.payload,
                context_for_logging=f"commit of {staged_op.handler_function}",
            )
            if reconstructed_schema_model is not None and staged_op.schema_arg_name:
                commit_kwargs[staged_op.schema_arg_name] = reconstructed_schema_model

            _apply_forced_public_id(
                commit_kwargs, staged_op, handler_func_obj, context_for_logging="commit"
            )

            await handler_func_obj(db, **commit_kwargs)
        except (
            ValueError,
            IntegrityError,
            ConflictException,
            ResourceNotFoundException,
        ) as e_commit_exec:
            logger.error(
                f"Error executing staged operation {staged_op.handler_function} during commit: {e_commit_exec}"
            )
            await db.rollback()
            transaction.status = "failed"
            await db.commit()
            raise
        except (
            Exception
        ) as e_unexpected_commit_exec:  # Catch any other unexpected errors
            logger.error(
                f"Unexpected error executing staged operation {staged_op.handler_function} during commit: {e_unexpected_commit_exec}"
            )
            await db.rollback()
            transaction.status = "failed"
            await db.commit()
            raise RuntimeError(
                f"Unexpected error during commit of {staged_op.handler_function}"
            ) from e_unexpected_commit_exec

    stmt = delete(models.StagedOperation).where(
        models.StagedOperation.transaction_id == transaction_id
    )
    await db.execute(stmt)

    transaction.status = "committed"
    await db.commit()


async def rollback_transaction(db: AsyncSession, transaction_id: int) -> None:
    stmt = select(models.Transaction).where(
        models.Transaction.transaction_id == transaction_id
    )
    result = await db.execute(stmt)
    transaction = result.scalar_one_or_none()
    if transaction is None:
        raise ResourceNotFoundException("Transaction not found")
    if transaction.status != "pending":
        raise ValidationException("Transaction is not pending")
    if transaction.expires_at < datetime.datetime.now(datetime.timezone.utc):
        raise ValidationException("Transaction has expired")

    # Delete all staged operations for the transaction
    stmt = delete(models.StagedOperation).where(
        models.StagedOperation.transaction_id == transaction_id
    )
    await db.execute(stmt)

    # Update the transaction status to rolled back
    transaction.status = "rolled_back"
    await db.commit()


async def create_staged_operation(
    db: AsyncSession,
    transaction_id: int,
    sequence_number: int,
    operation_params: dict,
    operation_payload: dict | list,
    operation_handler: str,
    resource_public_id: str | None = None,
    schema_arg_name_for_db: str | None = None,
    is_list_schema_for_db: bool = False,
) -> models.StagedOperation:
    if sequence_number > MAX_STAGED_OPERATIONS:
        raise ValidationException(
            f"Sequence number {sequence_number} is greater than the maximum allowed value of {MAX_STAGED_OPERATIONS}"
        )
    staged_operation = models.StagedOperation(
        transaction_id=transaction_id,
        sequence_number=sequence_number,
        parameters=operation_params,
        payload=operation_payload,
        handler_function=operation_handler,
        resource_public_id=resource_public_id,
        schema_arg_name=schema_arg_name_for_db,
        is_list_schema=is_list_schema_for_db,
    )
    db.add(staged_operation)
    await db.commit()
    logger.info(
        f"Staged operation {operation_handler} created for transaction {transaction_id} with sequence {sequence_number}"
    )
    return staged_operation


async def get_staged_operations(
    db: AsyncSession, transaction_id: int
) -> Sequence[models.StagedOperation]:
    stmt = (
        select(models.StagedOperation)
        .where(models.StagedOperation.transaction_id == transaction_id)
        .order_by(models.StagedOperation.sequence_number)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


async def get_next_sequence_number(db: AsyncSession, transaction_id: int) -> int:
    stmt = select(func.max(models.StagedOperation.sequence_number)).where(
        models.StagedOperation.transaction_id == transaction_id
    )
    result = await db.execute(stmt)
    max_seq = result.scalar_one_or_none()
    return (max_seq + 1) if max_seq is not None else 1
