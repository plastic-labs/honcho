import datetime
import logging
import time

import sentry_sdk

from src import crud, exceptions, prometheus
from src.config import settings
from src.crud.representation import RepresentationManager
from src.dependencies import tracked_db
from src.models import Message
from src.schemas import ResolvedSessionConfiguration
from src.utils import summarizer
from src.utils.clients import honcho_llm_call
from src.utils.config_helpers import get_configuration
from src.utils.formatting import format_new_turn_with_timestamp
from src.utils.logging import (
    accumulate_metric,
    conditional_observe,
    log_performance_metrics,
    # log_representation,
)
from src.utils.peer_card import PeerCardQuery
from src.utils.representation import PromptRepresentation, Representation
from src.utils.tokens import estimate_tokens
from src.utils.tracing import with_sentry_transaction

from .prompts import (
    critical_analysis_prompt,
    estimate_base_prompt_tokens,
    peer_card_prompt,
)

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True


@conditional_observe(name="Critical Analysis Call")
async def critical_analysis_call(
    peer_id: str,
    peer_card: list[str] | None,
    message_created_at: datetime.datetime,
    working_representation: Representation,
    history: str,
    new_turns: list[str],
    estimated_input_tokens: int,
) -> PromptRepresentation:
    prompt = critical_analysis_prompt(
        peer_id=peer_id,
        peer_card=peer_card,
        message_created_at=message_created_at,
        working_representation=working_representation,
        history=history,
        new_turns=new_turns,
    )

    response = await honcho_llm_call(
        provider=settings.DERIVER.PROVIDER,
        model=settings.DERIVER.MODEL,
        prompt=prompt,
        max_tokens=settings.DERIVER.MAX_OUTPUT_TOKENS
        or settings.LLM.DEFAULT_MAX_TOKENS,
        track_name="Critical Analysis Call",
        response_model=PromptRepresentation,
        json_mode=True,
        stop_seqs=["   \n", "\n\n\n\n"],
        thinking_budget_tokens=settings.DERIVER.THINKING_BUDGET_TOKENS,
        reasoning_effort="minimal",
        verbosity="medium",
        enable_retry=True,
        retry_attempts=3,
    )

    prometheus.DERIVER_TOKENS_PROCESSED.labels(
        task_type="representation",
    ).inc(response.output_tokens + estimated_input_tokens)

    return response.content


@conditional_observe(name="Peer Card Call")
async def peer_card_call(
    old_peer_card: list[str] | None,
    new_observations: Representation,
) -> PeerCardQuery:
    """
    Generate peer card prompt, call LLM with response model.
    """
    prompt = peer_card_prompt(
        old_peer_card=old_peer_card,
        new_observations=new_observations.str_no_timestamps(),
    )

    response = await honcho_llm_call(
        provider=settings.PEER_CARD.PROVIDER,
        model=settings.PEER_CARD.MODEL,
        prompt=prompt,
        max_tokens=settings.PEER_CARD.MAX_OUTPUT_TOKENS
        or settings.LLM.DEFAULT_MAX_TOKENS,
        track_name="Peer Card Call",
        response_model=PeerCardQuery,
        json_mode=True,
        reasoning_effort="minimal",
        enable_retry=True,
        retry_attempts=3,
    )

    return response.content


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

    # Time context preparation
    context_prep_start = time.perf_counter()

    # Use get_session_context_formatted with configurable token limit

    working_representation = await crud.get_working_representation(
        latest_message.workspace_name,
        observer=observer,
        observed=observed,
        # include_semantic_query=latest_message.content,
        # include_most_derived=False,
    )

    async with tracked_db("deriver.get_peer_card") as db:
        session_level_configuration = get_configuration(
            await crud.get_session(
                db, latest_message.session_name, latest_message.workspace_name
            ),
            await crud.get_workspace(db, workspace_name=latest_message.workspace_name),
        )
        if session_level_configuration.peer_cards_enabled is False:
            speaker_peer_card = None
        else:
            speaker_peer_card = await crud.get_peer_card(
                db,
                latest_message.workspace_name,
                observer=observer,
                observed=observed,
            )

    if session_level_configuration.deriver_enabled is False:
        return

    # Estimate tokens for deriver input
    peer_card_tokens = estimate_tokens(speaker_peer_card)
    working_rep_tokens = estimate_tokens(
        str(working_representation) if not working_representation.is_empty() else None
    )
    base_prompt_tokens = estimate_base_prompt_tokens()

    # Estimate tokens for new conversation turns
    new_turns = [
        format_new_turn_with_timestamp(m.content, m.created_at, m.peer_name)
        for m in messages
    ]
    new_turns_tokens = estimate_tokens(new_turns)

    estimated_input_tokens = (
        peer_card_tokens + working_rep_tokens + base_prompt_tokens + new_turns_tokens
    )

    # Calculate available tokens for context
    safety_buffer = 500
    available_context_tokens = max(
        0,
        settings.DERIVER.MAX_INPUT_TOKENS - estimated_input_tokens - safety_buffer,
    )

    logger.debug(
        "Token estimation - Peer card: %d, Working rep: %d, Base prompt: %d, "
        + "New turns: %d, Total estimated: %d, Available for context: %d",
        peer_card_tokens,
        working_rep_tokens,
        base_prompt_tokens,
        new_turns_tokens,
        estimated_input_tokens,
        available_context_tokens,
    )

    async with tracked_db("deriver.get_session_context_formatted") as db:
        formatted_history = await summarizer.get_session_context_formatted(
            db,
            latest_message.workspace_name,
            latest_message.session_name,
            token_limit=available_context_tokens,
            cutoff=earliest_message.id,
            include_summary=True,
        )

    session_context_tokens = estimate_tokens(formatted_history)

    # got working representation and peer card, log timing
    context_prep_duration = (time.perf_counter() - context_prep_start) * 1000
    accumulate_metric(
        f"deriver_{latest_message.id}_{observer}",
        "context_preparation",
        context_prep_duration,
        "ms",
    )

    logger.debug(
        "Using working representation with %s explicit, %s deductive observations",
        len(working_representation.explicit),
        len(working_representation.deductive),
    )

    # instantiate representation manager from collection
    # if the sender is also the target, we're handling a global representation task.
    # otherwise, we're handling a directional representation task where the sender is
    # being observed by the target.

    # Use the representation manager directly
    representation_manager = RepresentationManager(
        workspace_name=latest_message.workspace_name,
        observer=observer,
        observed=observed,
    )

    reasoner = CertaintyReasoner(
        representation_manager=representation_manager,
        ctx=messages,
        observed=observed,
        observer=observer,
        estimated_input_tokens=estimated_input_tokens + session_context_tokens,
        session_level_configuration=session_level_configuration,
    )

    # Run single-pass reasoning
    final_observations = await reasoner.reason(
        working_representation,
        formatted_history,
        speaker_peer_card,
    )

    # Display final observations in a beautiful tree
    # log_representation(final_observations)

    # Calculate and log overall timing
    overall_duration = (time.perf_counter() - overall_start) * 1000
    accumulate_metric(
        f"deriver_{latest_message.id}_{observer}",
        "total_processing_time",
        overall_duration,
        "ms",
    )

    total_observations = len(final_observations.explicit) + len(
        final_observations.deductive
    )

    accumulate_metric(
        f"deriver_{latest_message.id}_{observer}",
        "observation_count",
        total_observations,
        "count",
    )

    log_performance_metrics("deriver", f"{latest_message.id}_{observer}")


class CertaintyReasoner:
    """Certainty reasoner for analyzing and deriving insights."""

    representation_manager: RepresentationManager
    ctx: list[Message]
    observer: str
    observed: str
    session_level_configuration: ResolvedSessionConfiguration

    def __init__(
        self,
        representation_manager: RepresentationManager,
        ctx: list[Message],
        *,
        observed: str,
        observer: str,
        estimated_input_tokens: int,
        session_level_configuration: ResolvedSessionConfiguration,
    ) -> None:
        self.representation_manager = representation_manager
        self.ctx = ctx
        self.observed = observed
        self.observer = observer
        self.estimated_input_tokens: int = estimated_input_tokens
        self.session_level_configuration = session_level_configuration

    @conditional_observe(name="Deriver")
    @sentry_sdk.trace
    async def reason(
        self,
        working_representation: Representation,
        history: str,
        speaker_peer_card: list[str] | None,
    ) -> Representation:
        """
        Single-pass reasoning function that critically analyzes and derives insights.
        Performs one analysis pass and returns the final observations.

        Returns:
            Representation: Final observations
        """
        analysis_start = time.perf_counter()

        earliest_message = self.ctx[0]
        latest_message = self.ctx[-1]

        new_turns = [
            format_new_turn_with_timestamp(m.content, m.created_at, m.peer_name)
            for m in self.ctx
        ]

        logger.debug(
            "CRITICAL ANALYSIS: message_created_at='%s', new_turns_count=%s",
            latest_message.created_at,
            len(new_turns),
        )

        try:
            reasoning_response = await critical_analysis_call(
                peer_id=self.observed,
                peer_card=speaker_peer_card,
                message_created_at=latest_message.created_at,
                working_representation=working_representation,
                history=history,
                new_turns=new_turns,
                estimated_input_tokens=self.estimated_input_tokens,
            )
        except Exception as e:
            raise exceptions.LLMError(
                speaker_peer_card=speaker_peer_card,
                working_representation=working_representation,
                history=history,
                new_turns=new_turns,
            ) from e

        reasoning_response = Representation.from_prompt_representation(
            reasoning_response,
            (earliest_message.id, latest_message.id),
            latest_message.session_name,
            latest_message.created_at,
        )

        analysis_duration_ms = (time.perf_counter() - analysis_start) * 1000
        accumulate_metric(
            f"deriver_{latest_message.id}_{self.observer}",
            "critical_analysis_duration",
            analysis_duration_ms,
            "ms",
        )

        # Save only the new observations that weren't in the original context
        new_observations = working_representation.diff_representation(
            reasoning_response
        )
        if not new_observations.is_empty():
            await self.representation_manager.save_representation(
                new_observations,
                (earliest_message.id, latest_message.id),
                latest_message.session_name,
                latest_message.created_at,
                self.session_level_configuration,
            )

        if self.session_level_configuration.peer_cards_enabled:
            update_peer_card_start = time.perf_counter()
            if not new_observations.is_empty():
                await self._update_peer_card(speaker_peer_card, new_observations)
            update_peer_card_duration = (
                time.perf_counter() - update_peer_card_start
            ) * 1000
            accumulate_metric(
                f"deriver_{latest_message.id}_{self.observer}",
                "update_peer_card",
                update_peer_card_duration,
                "ms",
            )

        return reasoning_response

    @sentry_sdk.trace
    async def _update_peer_card(
        self,
        old_peer_card: list[str] | None,
        new_observations: Representation,
    ) -> None:
        """
        Update the peer card by calling LLM with the old peer card and new observations.
        The new peer card is returned by the LLM and saved to peer internal metadata.
        """
        try:
            response = await peer_card_call(old_peer_card, new_observations)
            new_peer_card = response.card
            if not new_peer_card:
                # no changes
                return
            # even with a dedicated notes field, we still need to prune notes out of the card
            new_peer_card = [
                observation
                for observation in new_peer_card
                if not observation.lower().startswith(("note", "notes"))
            ]
            accumulate_metric(
                f"deriver_{self.ctx[-1].id}_{self.observer}",
                "new_peer_card",
                "\n".join(new_peer_card),
                "blob",
            )
            async with tracked_db("deriver.update_peer_card") as db:
                await crud.set_peer_card(
                    db,
                    self.ctx[0].workspace_name,
                    new_peer_card,
                    observer=self.observer,
                    observed=self.observed,
                )
        except Exception as e:
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_exception(e)
            logger.error("Error updating peer card! Skipping... %s", e)
