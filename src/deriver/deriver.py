import datetime
import logging
import time

import sentry_sdk

from src import crud, exceptions, prometheus
from src.config import settings
from src.crud.representation import RepresentationManager
from src.dependencies import tracked_db
from src.models import Message
from src.utils import summarizer
from src.utils.clients import honcho_llm_call
from src.utils.formatting import format_new_turn_with_timestamp
from src.utils.langfuse_client import get_langfuse_client
from src.utils.logging import (
    accumulate_metric,
    conditional_observe,
    log_performance_metrics,
    log_representation,
)
from src.utils.peer_card import PeerCardQuery
from src.utils.representation import PromptRepresentation, Representation
from src.utils.tokens import estimate_tokens
from src.utils.tracing import with_sentry_transaction

from src.deriver.reasoner.explicit import ExplicitReasoner
from src.deriver.reasoner.xr import XRReasoner

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

lf = get_langfuse_client() if settings.LANGFUSE_PUBLIC_KEY else None


@with_sentry_transaction("process_representation_tasks_batch", op="deriver")
async def process_representation_tasks_batch(
    messages: list[Message],
    *,
    observer: str,
    observed: str,
) -> None:
    """
    Process a batch of representation tasks by extracting insights and updating working representations.
    """
    if not messages or len(messages) == 0:
        return

    messages.sort(key=lambda x: x.id)

    latest_message = messages[-1]
    earliest_message = messages[0]

    accumulate_metric(
        f"deriver_{latest_message.id}_{observer}",
        "starting_message_id",
        earliest_message.id,
        "id",
    )

    accumulate_metric(
        f"deriver_{latest_message.id}_{observer}",
        "ending_message_id",
        latest_message.id,
        "id",
    )

    # Start overall timing
    overall_start = time.perf_counter()

    # Run Reasoner Operations
    res_ops = [XRReasoner]
    for res_op in res_ops:
        # Time Reasoner Operation
        res_op_start = time.perf_counter()

        # Time Context Preparation
        ctx_pr_start = time.perf_counter()

        working_representation = await crud.get_working_representation(
            latest_message.workspace_name,
            observer=observer,
            observed=observed,
            # include_semantic_query=latest_message.content,
            # include_most_derived=False,
        )

        if settings.PEER_CARD.ENABLED:
            async with tracked_db("deriver.get_peer_card") as db:
                speaker_peer_card: list[str] | None = await crud.get_peer_card(
                    db,
                    latest_message.workspace_name,
                    observer=observer,
                    observed=observed,
                )
        else:
            speaker_peer_card = None

        # Estimate Tokens
        peer_card_tokens = estimate_tokens(speaker_peer_card)
        work_repr_tokens = estimate_tokens(
            str(working_representation) if not working_representation.is_empty() else None
        )
        sys_prompt_tokens = estimate_tokens(res_op.base_prompt_render('system'))
        usr_prompt_tokens = estimate_tokens(res_op.base_prompt_render('user'))

        # Estimate Tokens for New Turns
        new_turns = [
            format_new_turn_with_timestamp(m.content, m.created_at, m.peer_name)
            for m in messages
        ]
        new_turns_tokens = estimate_tokens(new_turns)

        # Estimate Total Tokens for Reasoner Input
        estimated_input_tokens = (
            peer_card_tokens + work_repr_tokens + \
            sys_prompt_tokens + usr_prompt_tokens + \
            new_turns_tokens
        )

        # Calculate Available Tokens for Context
        safety_buffer = 500
        available_context_tokens = max(
            0,
            settings.DERIVER.MAX_INPUT_TOKENS - estimated_input_tokens - safety_buffer,
        )

        logger.debug(
            "Token estimation - Peer card: %d, Working rep: %d, Sys prompt: %d, Usr prompt: %d, "
            + "New turns: %d, Total estimated: %d, Available for context: %d",
            peer_card_tokens,
            work_repr_tokens,
            sys_prompt_tokens,
            usr_prompt_tokens,
            new_turns_tokens,
            estimated_input_tokens,
            available_context_tokens,
        )

        # Get Session Context
        async with tracked_db("deriver.get_session_context_formatted") as db:
            formatted_history = await summarizer.get_session_context_formatted(
                db,
                latest_message.workspace_name,
                latest_message.session_name,
                token_limit=available_context_tokens,
                cutoff=earliest_message.id,
                include_summary=True,
            )

        # Estimate Tokens for Session Context
        session_context_tokens = estimate_tokens(formatted_history)

        # Log Timing
        ctx_pr_duration = (time.perf_counter() - ctx_pr_start) * 1000
        accumulate_metric(
            f"deriver_{latest_message.id}_{observer}",
            "context_preparation",
            ctx_pr_duration,
            "ms",
        )

        logger.debug(
            "Working Representation Counts:\n"
            + f"Explicit: {len(working_representation.explicit)}\n"
            + f"Implicit: {len(working_representation.implicit)}\n"
            + f"Deductive: {len(working_representation.deductive)}\n"
        )

        # Utilize Representation Manager
        representation_manager = RepresentationManager(
            workspace_name=latest_message.workspace_name,
            observer=observer,
            observed=observed,
        )

        # instantiate representation manager from collection
        # if the sender is also the target, we're handling a global representation task.
        # otherwise, we're handling a directional representation task where the sender is
        # being observed by the target.

        reasoner = res_op(
            representation_manager=representation_manager,
            ctx=messages,
            observed=observed,
            observer=observer,
            estimated_input_tokens=estimated_input_tokens,
        )

        final_observations = await reasoner.reason(
            working_representation,
            formatted_history,
            speaker_peer_card,
        )

        # Log Final Observations
        log_representation(final_observations)

        # Calculate and Log Reasoning Timing
        rsr_duration = (time.perf_counter() - res_op_start) * 1000
        accumulate_metric(
            f"deriver_{latest_message.id}_{observer}",
            "reasoning",
            rsr_duration,
            "ms",
        )

    # Calculate and log overall timing
    overall_duration = (time.perf_counter() - overall_start) * 1000
    accumulate_metric(
        f"deriver_{latest_message.id}_{observer}",
        "total_processing_time",
        overall_duration,
        "ms",
    )

    # Compute and Log Total Observations
    total_observations = \
        len(final_observations.explicit) + \
        len(final_observations.implicit) + \
        len(final_observations.deductive)

    accumulate_metric(
        f"deriver_{latest_message.id}_{observer}",
        "observation_count",
        total_observations,
        "count",
    )

    log_performance_metrics("deriver", f"{latest_message.id}_{observer}")

    if lf:
        lf.update_current_trace(output=final_observations.format_as_markdown())
