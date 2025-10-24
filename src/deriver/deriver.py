import datetime
import json
import logging
import time
from typing import Any

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
    log_reasoning_step,
    log_representation,
)
from src.utils.metrics_collector import (
    save_abductive_trace,
    save_deductive_trace,
    save_explicit_trace,
    save_inductive_trace,
)
from src.utils.peer_card import PeerCardQuery
from src.utils.representation import (
    AbductiveReasoningResponse,
    DeductiveReasoningResponse,
    InductiveReasoningResponse,
    PromptRepresentation,
    Representation,
)
from src.utils.tokens import estimate_tokens
from src.utils.tracing import with_sentry_transaction

from .prompts import (
    abductive_reasoning_prompt,
    deductive_reasoning_prompt,
    estimate_base_prompt_tokens,
    explict_reasoning_prompt,
    inductive_reasoning_prompt,
    peer_card_prompt,
)

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

lf = get_langfuse_client() if settings.LANGFUSE_PUBLIC_KEY else None

MAKE_LOCAL_TRACE = settings.COLLECT_METRICS_LOCAL


async def critical_analysis_call(
    peer_id: str,
    peer_card: list[str] | None,
    message_created_at: datetime.datetime,
    working_representation: Representation,
    history: str,
    new_turns: list[str],
    estimated_input_tokens: int,
    message_metadata: dict[str, Any] | None = None,
) -> PromptRepresentation:
    # Step 1: Explicit Reasoning
    explicit_prompt = explict_reasoning_prompt(
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
        prompt=explicit_prompt,
        max_tokens=settings.DERIVER.MAX_OUTPUT_TOKENS
        or settings.LLM.DEFAULT_MAX_TOKENS,
        track_name="Explicit Reasoning Call",
        response_model=PromptRepresentation,
        json_mode=True,
        stop_seqs=["   \n", "\n\n\n\n"],
        thinking_budget_tokens=settings.DERIVER.THINKING_BUDGET_TOKENS,
        reasoning_effort="minimal",
        verbosity="medium",
        enable_retry=True,
        retry_attempts=3,
    )

    # Log explicit reasoning step
    logger.debug("=" * 80)
    logger.debug("STEP 1: EXPLICIT REASONING")
    log_reasoning_step(
        "Explicit Reasoning",
        explicit_prompt,
        response.content.model_dump(),
        step_number=1,
    )

    if MAKE_LOCAL_TRACE:
        save_explicit_trace(
            provider=settings.DERIVER.PROVIDER,
            model=settings.DERIVER.MODEL,
            max_tokens=settings.DERIVER.MAX_OUTPUT_TOKENS
            or settings.LLM.DEFAULT_MAX_TOKENS,
            peer_id=peer_id,
            peer_card=peer_card,
            message_created_at=message_created_at,
            working_representation=working_representation,
            history=history,
            new_turns=new_turns,
            prompt=explicit_prompt,
            response=json.dumps(response.content.model_dump()),
            thinking=response.think_trace,
            message_metadata=message_metadata,
        )

    prometheus.DERIVER_TOKENS_PROCESSED.labels(
        task_type="representation",
    ).inc(response.output_tokens + estimated_input_tokens)

    # Step 2: Deductive Reasoning
    # Combine explicit and implicit facts as atomic propositions for deductive reasoning
    atomic_propositions = response.content.explicit + response.content.implicit

    # Get existing deductions from working representation
    existing_deductions = [
        obs.conclusion for obs in working_representation.deductive
    ]

    # Only perform deductive reasoning if we have atomic propositions
    deductive_response = None
    if atomic_propositions:
        deductive_prompt = deductive_reasoning_prompt(
            peer_id=peer_id,
            peer_card=peer_card,
            message_created_at=message_created_at,
            existing_deductions=existing_deductions,
            atomic_propositions=atomic_propositions,
            history=history,
            new_turns=new_turns,
        )

        deductive_response = await honcho_llm_call(
            provider=settings.DERIVER.PROVIDER,
            model=settings.DERIVER.MODEL,
            prompt=deductive_prompt,
            max_tokens=settings.DERIVER.MAX_OUTPUT_TOKENS
            or settings.LLM.DEFAULT_MAX_TOKENS,
            track_name="Deductive Reasoning Call",
            response_model=DeductiveReasoningResponse,
            json_mode=True,
            thinking_budget_tokens=settings.DERIVER.THINKING_BUDGET_TOKENS,
            reasoning_effort="minimal",
            verbosity="medium",
            enable_retry=True,
            retry_attempts=3,
        )

        # Log deductive reasoning step
        logger.debug("=" * 80)
        logger.debug("STEP 2: DEDUCTIVE REASONING")
        log_reasoning_step(
            "Deductive Reasoning",
            deductive_prompt,
            deductive_response.content.model_dump(),
            step_number=2,
        )

        if MAKE_LOCAL_TRACE:
            save_deductive_trace(
                provider=settings.DERIVER.PROVIDER,
                model=settings.DERIVER.MODEL,
                max_tokens=settings.DERIVER.MAX_OUTPUT_TOKENS
                or settings.LLM.DEFAULT_MAX_TOKENS,
                peer_id=peer_id,
                peer_card=peer_card,
                message_created_at=message_created_at,
                existing_deductions=existing_deductions,
                atomic_propositions=atomic_propositions,
                history=history,
                new_turns=new_turns,
                prompt=deductive_prompt,
                response=json.dumps(deductive_response.content.model_dump()),
                thinking=deductive_response.think_trace,
                message_metadata=message_metadata,
            )

        prometheus.DERIVER_TOKENS_PROCESSED.labels(
            task_type="representation",
        ).inc(deductive_response.output_tokens + estimated_input_tokens)

    # Step 3: Inductive Reasoning
    # Collect all existing inductions from working representation
    existing_inductions = [
        obs.conclusion for obs in working_representation.inductive
    ]

    # Collect explicit conclusions (from step 1) and deductive conclusions (from step 2)
    explicit_conclusions = response.content.explicit
    deductive_conclusions = [
        ded.conclusion for ded in (deductive_response.content.deductions if deductive_response else [])
    ]

    # Only perform inductive reasoning if we have enough conclusions (need at least 3 for patterns)
    inductive_response = None
    if len(explicit_conclusions) + len(deductive_conclusions) >= 3:
        inductive_prompt = inductive_reasoning_prompt(
            peer_id=peer_id,
            peer_card=peer_card,
            message_created_at=message_created_at,
            existing_inductions=existing_inductions,
            explicit_conclusions=explicit_conclusions,
            deductive_conclusions=deductive_conclusions,
            history=history,
            new_turns=new_turns,
        )

        inductive_response = await honcho_llm_call(
            provider=settings.DERIVER.PROVIDER,
            model=settings.DERIVER.MODEL,
            prompt=inductive_prompt,
            max_tokens=settings.DERIVER.MAX_OUTPUT_TOKENS
            or settings.LLM.DEFAULT_MAX_TOKENS,
            track_name="Inductive Reasoning Call",
            response_model=InductiveReasoningResponse,
            json_mode=True,
            thinking_budget_tokens=settings.DERIVER.THINKING_BUDGET_TOKENS,
            reasoning_effort="minimal",
            verbosity="medium",
            enable_retry=True,
            retry_attempts=3,
        )

        # Log inductive reasoning step
        logger.debug("=" * 80)
        logger.debug("STEP 3: INDUCTIVE REASONING")
        log_reasoning_step(
            "Inductive Reasoning",
            inductive_prompt,
            inductive_response.content.model_dump(),
            step_number=3,
        )

        if MAKE_LOCAL_TRACE:
            save_inductive_trace(
                provider=settings.DERIVER.PROVIDER,
                model=settings.DERIVER.MODEL,
                max_tokens=settings.DERIVER.MAX_OUTPUT_TOKENS
                or settings.LLM.DEFAULT_MAX_TOKENS,
                peer_id=peer_id,
                peer_card=peer_card,
                message_created_at=message_created_at,
                existing_inductions=existing_inductions,
                explicit_conclusions=explicit_conclusions,
                deductive_conclusions=deductive_conclusions,
                history=history,
                new_turns=new_turns,
                prompt=inductive_prompt,
                response=json.dumps(inductive_response.content.model_dump()),
                thinking=inductive_response.think_trace,
                message_metadata=message_metadata,
            )

        prometheus.DERIVER_TOKENS_PROCESSED.labels(
            task_type="representation",
        ).inc(inductive_response.output_tokens + estimated_input_tokens)

    # Step 4: Abductive Reasoning
    # Collect all existing abductions from working representation
    existing_abductions = [
        obs.conclusion for obs in working_representation.abductive
    ]

    # Collect explicit, deductive, and inductive conclusions
    explicit_conclusions = response.content.explicit
    deductive_conclusions = [
        ded.conclusion for ded in (deductive_response.content.deductions if deductive_response else [])
    ]
    inductive_conclusions = [
        ind.conclusion for ind in (inductive_response.content.inductions if inductive_response else [])
    ]

    # Only perform abductive reasoning if we have enough conclusions to explain
    abductive_response = None
    if len(explicit_conclusions) + len(deductive_conclusions) + len(inductive_conclusions) >= 3:
        abductive_prompt = abductive_reasoning_prompt(
            peer_id=peer_id,
            peer_card=peer_card,
            message_created_at=message_created_at,
            existing_abductions=existing_abductions,
            explicit_conclusions=explicit_conclusions,
            deductive_conclusions=deductive_conclusions,
            inductive_conclusions=inductive_conclusions,
            history=history,
            new_turns=new_turns,
        )

        abductive_response = await honcho_llm_call(
            provider=settings.DERIVER.PROVIDER,
            model=settings.DERIVER.MODEL,
            prompt=abductive_prompt,
            max_tokens=settings.DERIVER.MAX_OUTPUT_TOKENS
            or settings.LLM.DEFAULT_MAX_TOKENS,
            track_name="Abductive Reasoning Call",
            response_model=AbductiveReasoningResponse,
            json_mode=True,
            thinking_budget_tokens=settings.DERIVER.THINKING_BUDGET_TOKENS,
            reasoning_effort="minimal",
            verbosity="medium",
            enable_retry=True,
            retry_attempts=3,
        )

        # Log abductive reasoning step
        logger.debug("=" * 80)
        logger.debug("STEP 4: ABDUCTIVE REASONING")
        log_reasoning_step(
            "Abductive Reasoning",
            abductive_prompt,
            abductive_response.content.model_dump(),
            step_number=4,
        )

        if MAKE_LOCAL_TRACE:
            save_abductive_trace(
                provider=settings.DERIVER.PROVIDER,
                model=settings.DERIVER.MODEL,
                max_tokens=settings.DERIVER.MAX_OUTPUT_TOKENS
                or settings.LLM.DEFAULT_MAX_TOKENS,
                peer_id=peer_id,
                peer_card=peer_card,
                message_created_at=message_created_at,
                existing_abductions=existing_abductions,
                explicit_conclusions=explicit_conclusions,
                deductive_conclusions=deductive_conclusions,
                inductive_conclusions=inductive_conclusions,
                history=history,
                new_turns=new_turns,
                prompt=abductive_prompt,
                response=json.dumps(abductive_response.content.model_dump()),
                thinking=abductive_response.think_trace,
                message_metadata=message_metadata,
            )

        prometheus.DERIVER_TOKENS_PROCESSED.labels(
            task_type="representation",
        ).inc(abductive_response.output_tokens + estimated_input_tokens)

    # Create combined result with explicit, implicit, deductive, inductive, and abductive
    combined_result = PromptRepresentation(
        explicit=response.content.explicit,
        implicit=response.content.implicit,
    )

    # Store deductive results
    if deductive_response:
        combined_result.deductions = deductive_response.content.deductions
    else:
        combined_result.deductions = []

    # Store inductive results
    if inductive_response:
        combined_result.inductions = inductive_response.content.inductions
    else:
        combined_result.inductions = []

    # Store abductive results
    if abductive_response:
        combined_result.abductions = abductive_response.content.abductions
    else:
        combined_result.abductions = []

    # Log final combined result
    logger.debug("=" * 80)
    logger.debug("FINAL COMBINED RESULT")
    from src.utils.representation import Representation
    final_rep = Representation.from_prompt_representation(
        combined_result,
        (0, 0),  # dummy message ids for display
        "session",  # dummy session name for display
        message_created_at,
    )
    log_representation(final_rep)

    return combined_result


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
        "Using working representation with %s explicit, %s implicit, %s deductive, %s inductive, %s abductive observations",
        len(working_representation.explicit),
        len(working_representation.implicit),
        len(working_representation.deductive),
        len(working_representation.inductive),
        len(working_representation.abductive),
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
    )

    # Run single-pass reasoning
    final_observations = await reasoner.reason(
        working_representation,
        formatted_history,
        speaker_peer_card,
    )

    # Display final observations in a beautiful tree
    log_representation(final_observations)

    # Calculate and log overall timing
    overall_duration = (time.perf_counter() - overall_start) * 1000
    accumulate_metric(
        f"deriver_{latest_message.id}_{observer}",
        "total_processing_time",
        overall_duration,
        "ms",
    )

    total_observations = (
        len(final_observations.explicit)
        + len(final_observations.implicit)
        + len(final_observations.deductive)
        + len(final_observations.inductive)
        + len(final_observations.abductive)
    )

    accumulate_metric(
        f"deriver_{latest_message.id}_{observer}",
        "observation_count",
        total_observations,
        "count",
    )

    log_performance_metrics("deriver", f"{latest_message.id}_{observer}")

    if lf:
        lf.update_current_trace(output=final_observations.format_as_markdown())


class CertaintyReasoner:
    """Certainty reasoner for analyzing and deriving insights."""

    representation_manager: RepresentationManager
    ctx: list[Message]
    observer: str
    observed: str

    def __init__(
        self,
        representation_manager: RepresentationManager,
        ctx: list[Message],
        *,
        observed: str,
        observer: str,
        estimated_input_tokens: int,
    ) -> None:
        self.representation_manager = representation_manager
        self.ctx = ctx
        self.observed = observed
        self.observer = observer
        self.estimated_input_tokens: int = estimated_input_tokens

    @conditional_observe
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
                message_metadata=latest_message.h_metadata if hasattr(latest_message, 'h_metadata') else None,
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

        if lf:
            lf.update_current_generation(
                output=reasoning_response.format_as_markdown(),
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
            )

        # not currently deduplicating at the save_representation step, so this isn't useful
        # accumulate_metric(
        #     f"deriver_{latest_payload.message_id}_{latest_payload.observer}",
        #     "new_observation_count",
        #     new_observations_saved,
        #     "count",
        # )

        if settings.PEER_CARD.ENABLED:
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

    @conditional_observe
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
            # Log peer card update count instead of full content to reduce debug noise
            accumulate_metric(
                f"deriver_{self.ctx[-1].id}_{self.observer}",
                "peer_card_update_count",
                len(new_peer_card),
                "count",
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