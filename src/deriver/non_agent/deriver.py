import logging
import time

from src import prometheus
from src.config import settings
from src.crud import message as message_crud
from src.crud.representation import RepresentationManager
from src.dependencies import tracked_db
from src.models import Message
from src.schemas import ResolvedConfiguration
from src.utils.clients import honcho_llm_call
from src.utils.config_helpers import get_configuration
from src.utils.finetuning_traces import log_finetuning_trace
from src.utils.formatting import format_new_turn_with_timestamp
from src.utils.logging import accumulate_metric, log_performance_metrics
from src.utils.representation import PromptRepresentation, Representation
from src.utils.tokens import estimate_tokens, track_input_tokens
from src.utils.tracing import with_sentry_transaction

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
        messages: List of messages to process.
        message_level_configuration: Optional configuration override.
        observer: The observer peer ID.
        observed: The observed peer ID.
    """
    if not messages:
        return

    messages.sort(key=lambda x: x.id)
    latest_message = messages[-1]
    earliest_message = messages[0]

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

    overall_start = time.perf_counter()

    # Get configuration if not provided
    if message_level_configuration is None:
        async with tracked_db("minimal_deriver.get_config") as db:
            from src import crud

            message_level_configuration = get_configuration(
                None,
                await crud.get_session(
                    db, latest_message.session_name, latest_message.workspace_name
                ),
                await crud.get_workspace(
                    db, workspace_name=latest_message.workspace_name
                ),
            )

    # Skip if deriver disabled
    if message_level_configuration.deriver.enabled is False:
        return

    # Calculate available tokens for messages
    prompt_tokens = estimate_minimal_deriver_prompt_tokens()
    safety_buffer = 500
    available_context_tokens = max(
        0,
        settings.DERIVER.MAX_INPUT_TOKENS - prompt_tokens - safety_buffer,
    )

    # Get interleaving messages from starting_message_id to ending_message_id (inclusive)
    async with tracked_db("minimal_deriver.get_interleaving_messages") as db:
        starting_message_id = earliest_message.id
        ending_message_id = latest_message.id

        # Get all messages in the range (inclusive on both ends)
        # Note: get_messages_id_range uses exclusive end_id, so we pass ending_message_id + 1
        interleaving_messages = await message_crud.get_messages_id_range(
            db,
            latest_message.workspace_name,
            latest_message.session_name,
            start_id=starting_message_id,
            end_id=ending_message_id + 1,
            token_limit=available_context_tokens,
        )

        # Create a dict of messages being processed for quick lookup and deduplication
        processing_messages_dict: dict[int, Message] = {m.id: m for m in messages}

        # Combine interleaving messages with messages being processed
        # Messages from DB may not include the ones being processed if they're not yet committed
        # Use processing messages when available (they're more up-to-date), otherwise use DB messages
        all_message_ids = {msg.id for msg in interleaving_messages} | {
            m.id for m in messages
        }

        # Build combined list sorted by ID for contiguous presentation
        combined_messages: list[Message] = []
        for msg_id in sorted(all_message_ids):
            if msg_id in processing_messages_dict:
                combined_messages.append(processing_messages_dict[msg_id])
            else:
                # Find the message in interleaving_messages
                for msg in interleaving_messages:
                    if msg.id == msg_id:
                        combined_messages.append(msg)
                        break

        # Format all messages with timestamps for contiguous presentation
        formatted_messages = "\n".join(
            [
                format_new_turn_with_timestamp(
                    msg.content, msg.created_at, msg.peer_name
                )
                for msg in combined_messages
            ]
        )

    messages_tokens = estimate_tokens(formatted_messages)

    # Track input tokens
    track_input_tokens(
        task_type="minimal_representation",
        components={
            "prompt": prompt_tokens,
            "messages": messages_tokens,
        },
    )

    # Build prompt
    prompt = minimal_deriver_prompt(
        peer_id=observed,
        message_created_at=latest_message.created_at,
        messages=formatted_messages,
    )

    context_prep_duration = (time.perf_counter() - overall_start) * 1000
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observer}",
        "context_preparation",
        context_prep_duration,
        "ms",
    )

    # Single LLM call
    llm_start = time.perf_counter()
    response = await honcho_llm_call(
        llm_settings=settings.DERIVER,
        prompt=prompt,
        max_tokens=settings.DERIVER.MAX_OUTPUT_TOKENS
        or settings.LLM.DEFAULT_MAX_TOKENS,
        track_name="Minimal Deriver",
        response_model=PromptRepresentation,
        json_mode=True,
        stop_seqs=["   \n", "\n\n\n\n"],
        thinking_budget_tokens=settings.DERIVER.THINKING_BUDGET_TOKENS,
        reasoning_effort="minimal",
        enable_retry=True,
        retry_attempts=3,
    )
    llm_duration = (time.perf_counter() - llm_start) * 1000

    # Log fine-tuning trace
    log_finetuning_trace(
        task_type="minimal_deriver",
        llm_settings=settings.DERIVER,
        prompt=prompt,
        response=response,
        max_tokens=settings.DERIVER.MAX_OUTPUT_TOKENS
        or settings.LLM.DEFAULT_MAX_TOKENS,
        thinking_budget_tokens=settings.DERIVER.THINKING_BUDGET_TOKENS,
        reasoning_effort="minimal",
        json_mode=True,
        stop_seqs=["   \n", "\n\n\n\n"],
    )

    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observer}",
        "llm_call_duration",
        llm_duration,
        "ms",
    )

    prometheus.DERIVER_TOKENS_PROCESSED.labels(  # nosec B106 <- dumb false positive
        task_type="minimal_representation",
        token_type="output",
        component="total",
    ).inc(response.output_tokens)

    # Convert to Representation and save
    observations = Representation.from_prompt_representation(
        response.content,
        [earliest_message.id, latest_message.id],
        latest_message.session_name,
        latest_message.created_at,
    )

    if not observations.is_empty():
        representation_manager = RepresentationManager(
            workspace_name=latest_message.workspace_name,
            observer=observer,
            observed=observed,
        )
        message_ids = [m.id for m in messages]
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
        if observations.explicit:
            accumulate_metric(
                f"minimal_deriver_{latest_message.id}_{observer}",
                "explicit_observations",
                "\n".join(f"  • {obs}" for obs in observations.explicit),
                "blob",
            )
        if observations.deductive:
            accumulate_metric(
                f"minimal_deriver_{latest_message.id}_{observer}",
                "deductive_observations",
                "\n".join(f"  • {obs}" for obs in observations.deductive),
                "blob",
            )

    log_performance_metrics("minimal_deriver", f"{latest_message.id}_{observer}")
