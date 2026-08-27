from logging import getLogger

from sqlalchemy import select

from src import models
from src.dependencies import tracked_db
from src.schemas import DreamType

logger = getLogger(__name__)


async def execute_dream(
    workspace_name: str,
    dream_type: DreamType,
    *,
    observer: str,
    observed: str,
    trigger_reason: str | None = None,
    delay_reason: str | None = None,
    documents_since_last_dream_at_schedule: int | None = None,
    document_threshold: int | None = None,
) -> None:
    """Execute the dream by enqueueing it."""
    from src import crud
    from src.deriver.enqueue import enqueue_dream
    from src.utils.config_helpers import get_configuration

    async with tracked_db("dream_session_lookup") as db:
        stmt = (
            select(models.Document.session_name)
            .where(
                models.Document.workspace_name == workspace_name,
                models.Document.observer == observer,
                models.Document.observed == observed,
                models.Document.level == "explicit",
            )
            .order_by(models.Document.created_at.desc())
            .limit(1)
        )
        session_name = await db.scalar(stmt)

        if not session_name:
            logger.warning(
                f"No documents found for {workspace_name}/{observer}/{observed}, skipping dream"
            )
            return

        session = await crud.get_session(
            db, workspace_name=workspace_name, session_name=session_name
        )
        workspace = await crud.get_workspace(db, workspace_name=workspace_name)

        configuration = get_configuration(None, session, workspace)

        if not configuration.dream.enabled:
            logger.debug(
                f"Dreams disabled for {workspace_name}/{session_name}, skipping dream"
            )
            return

    await enqueue_dream(
        workspace_name,
        observer=observer,
        observed=observed,
        dream_type=dream_type,
        session_name=session_name,
        trigger_reason=trigger_reason,
        delay_reason=delay_reason,
        documents_since_last_dream_at_schedule=documents_since_last_dream_at_schedule,
        document_threshold=document_threshold,
    )
