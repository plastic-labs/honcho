import logging
import time

from src import crud
from src.config import ConfiguredModelSettings, settings
from src.crud.representation import RepresentationManager
from src.dependencies import tracked_db
from src.llm import honcho_llm_call
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
from src.utils.config_helpers import get_configuration
from src.utils.formatting import format_new_turn_with_timestamp
from src.utils.representation import PromptRepresentation, Representation
from src.utils.tokens import track_deriver_input_tokens

from .prompts import estimate_minimal_deriver_prompt_tokens, minimal_deriver_prompt

logger = logging.getLogger(__name__)


def _get_deriver_model_config() -> ConfiguredModelSettings:
    return settings.DERIVER.MODEL_CONFIG


# Message metadata.type values that mark a message as not the observed peer's
# authored prose. The deriver should still record these messages (for replay,
# audit, and message-stream completeness) but must NOT extract peer-attributed
# facts from them — they ride role: "user" per the Anthropic Messages API but
# carry assistant-authored content (pasted diffs, tool_result blocks) or
# explicit assistant tool actions performed on the user's behalf.
#
# The corresponding tags are set by the claude-honcho plugin
# (plastic-labs/claude-honcho#34) and may be set by any client that wants its
# tool-action / paste messages preserved without contributing to peer
# representation.
NON_PROSE_METADATA_TYPES: frozenset[str] = frozenset(
    {
        "user_paste_not_speech",
        "tool_action",
    }
)


def _is_extraction_eligible(msg: Message) -> bool:
    """Return False if the message metadata.type marks it as not authored prose.

    Returns True for messages without a recognized non-prose tag (the default).
    Type-guards against non-dict h_metadata: a JSONB column can technically
    hold a scalar from a rogue write or a botched migration; we don't want a
    stray string to crash the whole batch with an AttributeError.
    """
    metadata = msg.h_metadata
    if not isinstance(metadata, dict):
        return True
    return metadata.get("type") not in NON_PROSE_METADATA_TYPES


@with_sentry_transaction("minimal_deriver_batch", op="deriver")
async def process_representation_tasks_batch(
    messages: list[Message],
    message_level_configuration: ResolvedConfiguration | None,
    *,
    observers: list[str],
    observed: str,
    queue_item_message_ids: list[int],
) -> None:
    """
    Process messages with minimal overhead - single LLM call, save to multiple collections.

    Messages tagged with metadata.type in NON_PROSE_METADATA_TYPES (pasted
    code/diffs, tool actions) are excluded from the LLM-facing prompt so the
    extractor cannot misattribute their content to the observed peer. The
    messages themselves remain in the database; only their contribution to
    peer representation is suppressed. Earliest/latest message tracking is
    unaffected so progress accounting stays correct.

    Args:
        messages: List of messages to process (includes interleaving context).
        message_level_configuration: Optional configuration override.
        observers: List of observer peer IDs (collections to save to).
        observed: The observed peer ID.
        queue_item_message_ids: Message IDs from queue items being processed
    """
    if not messages:
        return

    overall_start = time.perf_counter()

    messages.sort(key=lambda x: x.id)
    latest_message = messages[-1]
    earliest_message = messages[0]

    # Filter messages by extraction eligibility. We compute two related sets:
    #
    #   eligible_messages       — every message in the batch that is NOT
    #                             tagged non-prose. Used for the LLM-facing
    #                             prompt (we want to keep eligible context
    #                             from any peer, not just the observed peer,
    #                             so the LLM sees enough surrounding text).
    #
    #   eligible_source_messages — eligible messages that are ALSO target
    #                              queue items for the observed peer. These
    #                              are what the deriver is being asked to
    #                              produce facts about. If this set is empty,
    #                              every queue item we were told to process
    #                              is non-prose and we should skip the LLM
    #                              call entirely — even if other-peer context
    #                              messages happen to be in the same batch.
    queue_item_message_ids_set = set(queue_item_message_ids)
    eligible_messages = [m for m in messages if _is_extraction_eligible(m)]
    eligible_source_messages = [
        m
        for m in eligible_messages
        if m.peer_name == observed and m.id in queue_item_message_ids_set
    ]
    if not eligible_source_messages:
        logger.debug(
            "All %d target queue items in batch tagged as non-prose; skipping "
            "representation extraction (observed=%s, latest_message_id=%s)",
            len(queue_item_message_ids_set),
            observed,
            latest_message.id,
        )
        # Emit a minimal completion event + metrics so the observability
        # layer captures all batch outcomes uniformly. Downstream
        # dashboards / replay systems otherwise lose all trace of
        # all-non-prose batches.
        skipped_duration = (time.perf_counter() - overall_start) * 1000
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
        accumulate_metric(
            f"minimal_deriver_{latest_message.id}_{observed}",
            "total_processing_time",
            skipped_duration,
            "ms",
        )
        accumulate_metric(
            f"minimal_deriver_{latest_message.id}_{observed}",
            "observation_count",
            0,
            "count",
        )
        log_performance_metrics(
            "minimal_deriver", f"{latest_message.id}_{observed}"
        )
        emit(
            RepresentationCompletedEvent(
                workspace_name=latest_message.workspace_name,
                session_name=latest_message.session_name,
                observed=observed,
                queue_items_processed=len(queue_item_message_ids),
                earliest_message_id=earliest_message.public_id,
                latest_message_id=latest_message.public_id,
                message_count=len(messages),
                explicit_conclusion_count=0,
                context_preparation_ms=0.0,
                llm_call_ms=0.0,
                total_duration_ms=skipped_duration,
                input_tokens=0,
                output_tokens=0,
            )
        )
        return

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

    # Format messages with timestamps. Use eligible_messages (non-prose tags
    # filtered) so the LLM-facing prompt does not contain pasted code/diffs
    # or assistant tool actions.
    formatted_messages = "\n".join(
        format_new_turn_with_timestamp(msg.content, msg.created_at, msg.peer_name)
        for msg in eligible_messages
    )

    # Track token usage - count only tokens from messages being processed.
    # Use eligible_messages so we don't bill against tokens we elected to
    # filter from the prompt. (queue_item_message_ids_set is already built
    # above for the target-message gating.)
    prompt_tokens = estimate_minimal_deriver_prompt_tokens()
    messages_tokens = sum(
        msg.token_count
        for msg in eligible_messages
        if msg.id in queue_item_message_ids_set
    )
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
    base_model_config = _get_deriver_model_config()
    max_tokens = base_model_config.max_output_tokens or settings.LLM.DEFAULT_MAX_TOKENS
    model_config = base_model_config

    # Single LLM call
    llm_start = time.perf_counter()
    response = await honcho_llm_call(
        model_config=model_config,
        prompt=prompt,
        max_tokens=max_tokens,
        track_name="Minimal Deriver",
        response_model=PromptRepresentation,
        json_mode=True,
        max_input_tokens=settings.DERIVER.MAX_INPUT_TOKENS,
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

    # Provenance: observations cite only eligible_source_messages — i.e.
    # target queue items for the observed peer that survived the non-prose
    # filter. This prevents tagged messages (pasted diffs, tool actions)
    # from appearing as evidence for derived facts about the user.
    message_ids = [m.id for m in eligible_source_messages]
    latest_eligible = eligible_source_messages[-1]

    # Convert to Representation and save
    observations = Representation.from_prompt_representation(
        response.content,
        message_ids,
        latest_eligible.session_name,
        latest_eligible.created_at,
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
        # Save to all observer collections. Use the latest_eligible source
        # message for session/timestamp so saved provenance points to a
        # message that actually contributed to the prompt.
        for observer in observers:
            representation_manager = RepresentationManager(
                workspace_name=latest_eligible.workspace_name,
                observer=observer,
                observed=observed,
            )

            try:
                await representation_manager.save_representation(
                    observations,
                    message_ids,
                    latest_eligible.session_name,
                    latest_eligible.created_at,
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
            queue_items_processed=len(queue_item_message_ids),
            earliest_message_id=earliest_message.public_id,
            latest_message_id=latest_message.public_id,
            message_count=len(messages),
            explicit_conclusion_count=len(observations.explicit),
            context_preparation_ms=context_prep_duration,
            llm_call_ms=llm_duration,
            total_duration_ms=overall_duration,
            input_tokens=messages_tokens,
            output_tokens=response.output_tokens,
        )
    )
