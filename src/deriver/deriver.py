import logging
import time

from src import crud
from src.config import settings
from src.crud.representation import RepresentationManager
from src.dependencies import tracked_db
from src.models import Message
from src.schemas import ResolvedConfiguration
from src.telemetry import otel_metrics, prometheus
from src.telemetry.logging import accumulate_metric, log_performance_metrics
from src.telemetry.tracing import with_sentry_transaction
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
    observer: str,
    observed: str,
) -> None:
    """
    Process messages with minimal overhead - single LLM call, no peer card.

    Args:
        messages: List of messages to process (includes interleaving context).
        message_level_configuration: Optional configuration override.
        observer: The observer peer ID.
        observed: The observed peer ID.
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
        f"minimal_deriver_{latest_message.id}_{observer}",
        "starting_message_id",
        earliest_message.id,
        "id",
    )
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observer}",
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
        task_type=prometheus.DeriverTaskTypes.INGESTION,
        components={
            prometheus.DeriverComponents.PROMPT: prompt_tokens,
            prometheus.DeriverComponents.MESSAGES: messages_tokens,
        },
    )

    # Build prompt
    prompt = minimal_deriver_prompt(peer_id=observed, messages=formatted_messages)

    context_prep_duration = (time.perf_counter() - overall_start) * 1000
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observer}",
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
        f"minimal_deriver_{latest_message.id}_{observer}",
        "llm_call_duration",
        llm_duration,
        "ms",
    )

    # OTel metrics (push-based)
    if settings.OTEL.ENABLED:
        otel_metrics.record_deriver_tokens(
            count=response.output_tokens,
            task_type=prometheus.DeriverTaskTypes.INGESTION.value,
            token_type=prometheus.TokenTypes.OUTPUT.value,
            component=prometheus.DeriverComponents.OUTPUT_TOTAL.value,
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
        representation_manager = RepresentationManager(
            workspace_name=latest_message.workspace_name,
            observer=observer,
            observed=observed,
        )

        await representation_manager.save_representation(
            observations,
            message_ids,
            latest_message.session_name,
            latest_message.created_at,
            message_level_configuration,
        )

    # Log metrics
    overall_duration = (time.perf_counter() - overall_start) * 1000
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observer}",
        "total_processing_time",
        overall_duration,
        "ms",
    )

    total_observations = len(observations.explicit) + len(observations.deductive)
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observer}",
        "observation_count",
        total_observations,
        "count",
    )

    if settings.DERIVER.LOG_OBSERVATIONS:
        # Log messages fed into deriver
        accumulate_metric(
            f"minimal_deriver_{latest_message.id}_{observer}",
            "messages",
            formatted_messages,
            "blob",
        )
        # Log actual observations created as blob metrics
        accumulate_metric(
            f"minimal_deriver_{latest_message.id}_{observer}",
            "explicit_observations",
            "\n".join(f"  • {obs}" for obs in observations.explicit),
            "blob",
        )

    log_performance_metrics("minimal_deriver", f"{latest_message.id}_{observer}")
