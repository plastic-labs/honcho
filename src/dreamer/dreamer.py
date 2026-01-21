import logging

import sentry_sdk

from src.config import settings
from src.dependencies import tracked_db
from src.dreamer.orchestrator import run_dream
from src.schemas import DreamType
from src.telemetry.events import DreamRunEvent, emit
from src.utils.queue_payload import DreamPayload

logger = logging.getLogger(__name__)


@sentry_sdk.trace
async def process_dream(
    payload: DreamPayload,
    workspace_name: str,
) -> None:
    """
    Process a dream task by performing collection maintenance operations.

    Args:
        payload: The dream task payload containing workspace, peer, and dream type information
    """
    logger.info(
        f"""
(っ- ‸ - ς)ᶻ z 𐰁 ᶻ z 𐰁 ᶻ z 𐰁\n
DREAM: {payload.dream_type} documents for {workspace_name}/{payload.observer}/{payload.observed}\n
𐰁 z ᶻ 𐰁 z ᶻ 𐰁 z ᶻ(っ- ‸ - ς)"""
    )

    try:
        match payload.dream_type:
            case DreamType.OMNI:
                async with tracked_db("dream_orchestrator") as db:
                    result = await run_dream(
                        db=db,
                        workspace_name=workspace_name,
                        observer=payload.observer,
                        observed=payload.observed,
                        session_name=payload.session_name,
                    )

                # Emit telemetry event if dream ran
                if result is not None:
                    emit(
                        DreamRunEvent(
                            run_id=result.run_id,
                            workspace_id=workspace_name,
                            workspace_name=workspace_name,
                            session_name=payload.session_name,
                            observer=payload.observer,
                            observed=payload.observed,
                            specialists_run=result.specialists_run,
                            deduction_success=result.deduction_success,
                            induction_success=result.induction_success,
                            surprisal_enabled=result.surprisal_enabled,
                            surprisal_conclusion_count=result.surprisal_conclusion_count,
                            total_iterations=result.total_iterations,
                            total_input_tokens=result.input_tokens,
                            total_output_tokens=result.output_tokens,
                            total_duration_ms=result.total_duration_ms,
                        )
                    )

    except Exception as e:
        logger.error(
            f"Error processing dream task {payload.dream_type} for {payload.observer}/{payload.observed}: {str(e)}",
            exc_info=True,
        )
        if settings.SENTRY.ENABLED:
            sentry_sdk.capture_exception(e)
        # Don't re-raise - we want to mark the dream task as processed even if it fails
