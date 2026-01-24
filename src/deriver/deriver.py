import logging
import time

from src import crud
from src.config import settings
from src.crud.representation import RepresentationManager
from src.dependencies import tracked_db
from src.models import Message
from src.schemas import ResolvedConfiguration
from src.telemetry import prometheus_metrics
from src.telemetry.events import RepresentationCompletedEvent, emit
from src.telemetry.logging import accumulate_metric, log_performance_metrics
from src.telemetry.prometheus.metrics import (
    DeriverComponents,
    DeriverTaskTypes,
    TokenTypes,
)
from src.telemetry.sentry import with_sentry_transaction
from src.utils.clients import honcho_llm_call
from src.utils.config_helpers import get_configuration
from src.utils.formatting import format_new_turn_with_timestamp
from src.utils.representation import PromptRepresentation, Representation
from src.utils.tokens import estimate_tokens, track_deriver_input_tokens

from .prompts import estimate_minimal_deriver_prompt_tokens, minimal_deriver_prompt

logger = logging.getLogger(__name__)


@with_sentry_transaction("minimal_deriver_batch", op="deriver")
async def process_representation_tasks_batch(
    messages: list[Message],
    message_level_configuration: ResolvedConfiguration | None,
    *,
    observers: list[str],
    observed: str,
    queue_items_count: int,
) -> None:
    """
    Process messages with minimal overhead - single LLM call, save to multiple collections.

    Args:
        messages: List of messages to process (includes interleaving context).
        message_level_configuration: Optional configuration override.
        observers: List of observer peer IDs (collections to save to).
        observed: The observed peer ID.
        queue_items_count: Number of QueueItem records being processed in this batch.
    """
    if not messages:
        return

    overall_start = time.perf_counter()

    messages.sort(key=lambda x: x.id)
    latest_message = messages[-1]
    earliest_message = messages[0]

    # Get configuration if not provided
    # TODO: this appears to be a very rare edge case coming out of `get_queue_item_batch` in queue_manager.py,
    # possible that we can remove this and require configuration to come through with the payload.
    if message_level_configuration is None:
        async with tracked_db("minimal_deriver.get_config") as db:
            message_level_configuration = get_configuration(
                None,
                await crud.get_session(
                    db, latest_message.session_name, latest_message.workspace_name
                ),
                await crud.get_workspace(
                    db, workspace_name=latest_message.workspace_name
                ),
            )

    # Skip if disabled
    if message_level_configuration.reasoning.enabled is False:
        return

    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "starting_message_id",
        earliest_message.id,
        "id",
    )
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "ending_message_id",
        latest_message.id,
        "id",
    )

    # Format messages with timestamps
    formatted_messages = "\n".join(
        format_new_turn_with_timestamp(msg.content, msg.created_at, msg.peer_name)
        for msg in messages
    )

    # Track token usage
    prompt_tokens = estimate_minimal_deriver_prompt_tokens()
    messages_tokens = estimate_tokens(formatted_messages)
    track_deriver_input_tokens(
        task_type=DeriverTaskTypes.INGESTION,
        components={
            DeriverComponents.PROMPT: prompt_tokens,
            DeriverComponents.MESSAGES: messages_tokens,
        },
    )

    # Build prompt
    prompt = minimal_deriver_prompt(peer_id=observed, messages=formatted_messages)

    context_prep_duration = (time.perf_counter() - overall_start) * 1000
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "context_preparation",
        context_prep_duration,
        "ms",
    )

    # validation on settings means max_tokens will always be > 0
    max_tokens = settings.DERIVER.MAX_OUTPUT_TOKENS or settings.LLM.DEFAULT_MAX_TOKENS

    # Single LLM call
    llm_start = time.perf_counter()
    response = await honcho_llm_call(
        llm_settings=settings.DERIVER,
        prompt=prompt,
        max_tokens=max_tokens,
        track_name="Minimal Deriver",
        response_model=PromptRepresentation,
        json_mode=True,
        temperature=settings.DERIVER.TEMPERATURE,
        stop_seqs=["   \n", "\n\n\n\n"],
        thinking_budget_tokens=settings.DERIVER.THINKING_BUDGET_TOKENS,
        max_input_tokens=settings.DERIVER.MAX_INPUT_TOKENS,
        reasoning_effort="minimal",
        enable_retry=True,
        retry_attempts=3,
        trace_name="minimal_deriver",
    )
    llm_duration = (time.perf_counter() - llm_start) * 1000

    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "llm_call_duration",
        llm_duration,
        "ms",
    )

    # Prometheus metrics
    if settings.METRICS.ENABLED:
        prometheus_metrics.record_deriver_tokens(
            count=response.output_tokens,
            task_type=DeriverTaskTypes.INGESTION.value,
            token_type=TokenTypes.OUTPUT.value,
            component=DeriverComponents.OUTPUT_TOTAL.value,
        )

    message_ids = [m.id for m in messages if m.peer_name == observed]

    # Convert to Representation and save
    observations = Representation.from_prompt_representation(
        response.content,
        message_ids,
        latest_message.session_name,
        latest_message.created_at,
    )

    if observations.is_empty() or not message_ids:
        logger.warning(
            "Deriver generated zero observations for messages %s:%s in %s/%s!",
            earliest_message.id,
            latest_message.id,
            latest_message.workspace_name,
            latest_message.session_name,
        )
    else:
        # Save to all observer collections
        for observer in observers:
            representation_manager = RepresentationManager(
                workspace_name=latest_message.workspace_name,
                observer=observer,
                observed=observed,
            )

            try:
                await representation_manager.save_representation(
                    observations,
                    message_ids,
                    latest_message.session_name,
                    latest_message.created_at,
                    message_level_configuration,
                )
            except Exception as e:
                logger.error(
                    "Failed to save representation for observer %s: %s", observer, e
                )

    # Log metrics
    overall_duration = (time.perf_counter() - overall_start) * 1000
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "total_processing_time",
        overall_duration,
        "ms",
    )

    total_observations = len(observations.explicit) + len(observations.deductive)
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "observation_count",
        total_observations,
        "count",
    )

    if settings.DERIVER.LOG_OBSERVATIONS:
        # Log messages fed into deriver
        accumulate_metric(
            f"minimal_deriver_{latest_message.id}_{observed}",
            "messages",
            formatted_messages,
            "blob",
        )
        # Log actual observations created as blob metrics
        accumulate_metric(
            f"minimal_deriver_{latest_message.id}_{observed}",
            "explicit_observations",
            "\n".join(f"  • {obs}" for obs in observations.explicit),
            "blob",
        )

    log_performance_metrics("minimal_deriver", f"{latest_message.id}_{observed}")

    # Emit telemetry event
    emit(
        RepresentationCompletedEvent(
            workspace_name=latest_message.workspace_name,
            session_name=latest_message.session_name,
            observed=observed,
            queue_items_processed=queue_items_count,
            earliest_message_id=earliest_message.public_id,
            latest_message_id=latest_message.public_id,
            message_count=len(messages),
            explicit_conclusion_count=len(observations.explicit),
            context_preparation_ms=context_prep_duration,
            llm_call_ms=llm_duration,
            total_duration_ms=overall_duration,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
    )
