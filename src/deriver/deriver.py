import datetime
import json
import logging
import time
from typing import Any

import sentry_sdk
from langfuse import get_client

from src import crud, exceptions
from src.config import settings
from src.crud.representation import GLOBAL_REPRESENTATION_COLLECTION_NAME
from src.dependencies import tracked_db
from src.deriver.utils import estimate_tokens
from src.utils import summarizer
from src.utils.clients import honcho_llm_call
from src.utils.embedding_store import EmbeddingStore
from src.utils.formatting import (
    REASONING_LEVELS,
    extract_observation_content,
    find_new_observations,
    format_context_for_prompt,
    format_new_turn_with_timestamp,
    utc_now_iso,
)
from src.utils.logging import (
    accumulate_metric,
    conditional_observe,
    format_reasoning_inputs_as_markdown,
    format_reasoning_response_as_markdown,
    log_observations_tree,
    log_performance_metrics,
    log_thinking_panel,
)
from src.utils.shared_models import (
    DeductiveObservation,
    ObservationContext,
    PeerCardQuery,
    ReasoningResponse,
    ReasoningResponseWithThinking,
    UnifiedObservation,
)

from .prompts import (
    critical_analysis_prompt,
    estimate_base_prompt_tokens,
    peer_card_prompt,
)
from .queue_payload import (
    RepresentationPayload,
)

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

lf = get_client()


async def critical_analysis_call(
    peer_id: str,
    peer_card: list[str] | None,
    message_created_at: datetime.datetime,
    working_representation: str | None,
    history: str,
    new_turns: list[str],
) -> ReasoningResponse:
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
        response_model=ReasoningResponse,
        json_mode=True,
        thinking_budget_tokens=settings.DERIVER.THINKING_BUDGET_TOKENS,
        enable_retry=True,
        retry_attempts=3,
    )

    return response.content


async def peer_card_call(
    old_peer_card: list[str] | None,
    new_observations: list[str],
) -> PeerCardQuery:
    """
    Generate peer card prompt, call LLM with response model.
    """
    prompt = peer_card_prompt(
        old_peer_card=old_peer_card,
        new_observations=new_observations,
    )

    response = await honcho_llm_call(
        provider=settings.DERIVER.PEER_CARD_PROVIDER,
        model=settings.DERIVER.PEER_CARD_MODEL,
        prompt=prompt,
        max_tokens=settings.DERIVER.PEER_CARD_MAX_OUTPUT_TOKENS
        or settings.LLM.DEFAULT_MAX_TOKENS,
        track_name="Peer Card Call",
        response_model=PeerCardQuery,
        json_mode=True,
        reasoning_effort="minimal",
        enable_retry=True,
        retry_attempts=3,
    )

    return response.content


@sentry_sdk.trace
async def process_representation_tasks_batch(
    payloads: list[RepresentationPayload],
) -> None:
    """
    Process a batch of representation tasks by extracting insights and updating working representations.
    """
    if not payloads or len(payloads) == 0:
        return

    payloads.sort(key=lambda x: x.message_id)

    latest_payload = payloads[-1]
    earliest_payload = payloads[0]

    # Start overall timing
    overall_start = time.perf_counter()

    logger.debug(
        "Starting insight extraction for message batch starting with: %s",
        earliest_payload.message_id,
    )

    async with tracked_db("deriver.get_peer_card") as db:
        speaker_peer_card: list[str] | None = await crud.get_peer_card(
            db,
            latest_payload.workspace_name,
            latest_payload.sender_name,
            latest_payload.target_name,
        )
    if speaker_peer_card is None:
        logger.warning("No peer card found for %s", latest_payload.sender_name)
    else:
        logger.info("Using peer card: %s", speaker_peer_card)

    # Get working representation data early for token estimation
    async with tracked_db("deriver.get_working_representation_data") as db:
        working_rep_data: (
            dict[str, Any] | str | None
        ) = await crud.get_working_representation_data(
            db,
            latest_payload.workspace_name,
            latest_payload.target_name,
            latest_payload.sender_name,
            latest_payload.session_name,
        )

    # Estimate tokens for deriver input
    peer_card_tokens = estimate_tokens(speaker_peer_card)
    working_rep_tokens = _estimate_working_representation_tokens(working_rep_data)
    base_prompt_tokens = estimate_base_prompt_tokens(logger)

    # Estimate tokens for new conversation turns
    new_turns = [
        format_new_turn_with_timestamp(p.content, p.created_at, p.sender_name)
        for p in payloads
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

    # Use get_session_context_formatted with dynamic token limit
    async with tracked_db("deriver.get_session_context") as db:
        formatted_history = await summarizer.get_session_context_formatted(
            db,
            latest_payload.workspace_name,
            latest_payload.session_name,
            token_limit=available_context_tokens,
            cutoff=earliest_payload.message_id,
            include_summary=True,
        )

    # instantiate embedding store from collection
    # if the sender is also the target, we're handling a global representation task.
    # otherwise, we're handling a directional representation task where the sender is
    # being observed by the target.
    collection_name = (
        crud.construct_collection_name(
            observer=latest_payload.target_name, observed=latest_payload.sender_name
        )
        if latest_payload.sender_name != latest_payload.target_name
        else GLOBAL_REPRESENTATION_COLLECTION_NAME
    )

    # get_or_create_collection already handles IntegrityError with rollback and a retry
    async with tracked_db("deriver.get_or_create_collection") as db:
        collection = await crud.get_or_create_collection(
            db,
            latest_payload.workspace_name,
            collection_name,
            latest_payload.sender_name,
        )
        collection_name_loaded = collection.name

    # Use the embedding store directly
    embedding_store = EmbeddingStore(
        workspace_name=latest_payload.workspace_name,
        peer_name=latest_payload.sender_name,
        collection_name=collection_name_loaded,
    )

    # Create reasoner instance
    reasoner = CertaintyReasoner(embedding_store=embedding_store, ctx=payloads)

    # Time context preparation
    context_prep_start = time.perf_counter()
    if (
        working_rep_data
        and isinstance(working_rep_data, dict)
        and working_rep_data.get("final_observations")
    ):
        # Reconstruct ReasoningResponse from stored peer data
        final_obs: dict[str, Any] = working_rep_data["final_observations"]
        deductive_observations: list[DeductiveObservation] = []
        for deductive_data in final_obs.get("deductive", []):
            deductive_observations.append(
                DeductiveObservation(
                    conclusion=deductive_data["conclusion"],
                    premises=deductive_data.get("premises", []),
                )
            )

        working_representation = ReasoningResponseWithThinking(
            thinking=final_obs.get("thinking"),
            explicit=final_obs.get("explicit", []),
            deductive=deductive_observations,
        )
        logger.info(
            "Using existing working representation with %s explicit, %s deductive observations",
            len(working_representation.explicit),
            len(working_representation.deductive),
        )
    else:
        # No existing working representation, use global search
        query_text = [payload.content for payload in payloads]
        query_text = "\n".join(
            query_text
        )  # TODO: consider a smarter strategy than concatenation
        working_representation = await embedding_store.get_relevant_observations(
            query=query_text,
            conversation_context=formatted_history,
            for_reasoning=True,
        )

        working_representation = observation_context_to_reasoning_response(
            working_representation
        )
        logger.info("No working representation found, using global semantic search")
    context_prep_duration = (time.perf_counter() - context_prep_start) * 1000
    accumulate_metric(
        f"deriver_representation_{latest_payload.message_id}_{latest_payload.target_name}",
        "context_preparation",
        context_prep_duration,
        "ms",
    )

    # Run consolidated reasoning that handles explicit and deductive levels
    logger.debug(
        "REASONING: Running unified insight derivation across explicit and deductive reasoning levels"
    )

    # Run single-pass reasoning
    final_observations = await reasoner.reason(
        working_representation,
        formatted_history,
        speaker_peer_card,
    )

    logger.debug("REASONING COMPLETION: Unified reasoning completed across all levels.")

    # Display final observations in a beautiful tree
    final_obs_dict = {
        level: getattr(final_observations, level, []) for level in REASONING_LEVELS
    }
    log_observations_tree(final_obs_dict)

    # Always save working representation to peer for dialectic access
    await save_working_representation_to_peer(latest_payload, final_observations)
    # Calculate and log overall timing
    overall_duration = (time.perf_counter() - overall_start) * 1000
    accumulate_metric(
        f"deriver_representation_{latest_payload.message_id}_{latest_payload.target_name}",
        "total_processing_time",
        overall_duration,
        "ms",
    )

    total_observations = sum(len(obs_list) for obs_list in final_obs_dict.values())

    accumulate_metric(
        f"deriver_representation_{latest_payload.message_id}_{latest_payload.target_name}",
        "final_observation_count",
        total_observations,
        "count",
    )
    log_performance_metrics(
        f"deriver_representation_{latest_payload.message_id}_{latest_payload.target_name}"
    )

    if settings.LANGFUSE_PUBLIC_KEY:
        lf.update_current_trace(
            output=format_reasoning_response_as_markdown(final_observations)
        )


# The old function now just calls the batch processor with a single payload
async def process_representation_task(
    payload: RepresentationPayload,
) -> None:
    await process_representation_tasks_batch([payload])


class CertaintyReasoner:
    """Certainty reasoner for analyzing and deriving insights."""

    embedding_store: EmbeddingStore
    ctx: list[RepresentationPayload]

    def __init__(
        self, embedding_store: EmbeddingStore, ctx: list[RepresentationPayload]
    ) -> None:
        self.embedding_store = embedding_store
        self.ctx = ctx

    @conditional_observe
    @sentry_sdk.trace
    async def derive_new_insights(
        self,
        working_representation: ReasoningResponseWithThinking,
        history: str,
        speaker_peer_card: list[str] | None,
    ) -> ReasoningResponseWithThinking:
        """
        Critically analyzes and revises understanding, returning structured observations.
        """
        # For logging, we can just show the content of the last message
        latest_payload = self.ctx[-1]

        if settings.LANGFUSE_PUBLIC_KEY:
            lf.update_current_generation(
                input=format_reasoning_inputs_as_markdown(
                    working_representation,
                    history,
                    latest_payload.content,
                    latest_payload.created_at,
                )
            )

        new_turns = [
            format_new_turn_with_timestamp(p.content, p.created_at, p.sender_name)
            for p in self.ctx
        ]

        formatted_working_representation = format_context_for_prompt(
            working_representation
        )

        logger.debug(
            "CRITICAL ANALYSIS: message_created_at='%s', new_turns_count=%s",
            latest_payload.created_at,
            len(new_turns),
        )

        try:
            response_obj = await critical_analysis_call(
                peer_id=latest_payload.sender_name,
                peer_card=speaker_peer_card,
                message_created_at=latest_payload.created_at,
                working_representation=formatted_working_representation,
                history=history,
                new_turns=new_turns,
            )
        except Exception as e:
            raise exceptions.LLMError(
                speaker_peer_card=speaker_peer_card,
                working_representation=formatted_working_representation,
                history=history,
                new_turns=new_turns,
            ) from e

        # If response is a string, try to parse as JSON
        if isinstance(response_obj, str):
            try:
                response_data = json.loads(response_obj)
                new_insights = ReasoningResponse(
                    explicit=response_data.get("explicit", []),
                    deductive=[
                        DeductiveObservation(**item)
                        for item in response_data.get("deductive", [])
                    ],
                )
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                if settings.SENTRY.ENABLED:
                    sentry_sdk.capture_exception(e)
                logger.warning("Failed to parse string response as JSON: %s", e)
                new_insights = ReasoningResponse(explicit=[], deductive=[])
        else:
            # If response is already a ReasoningResponse object
            new_insights = response_obj

        # Extract thinking content from the response
        thinking: str | None = None
        try:
            # Try to get thinking from the response object using getattr for safety
            response_attr = getattr(response_obj, "_response", None)
            if response_attr:
                thinking = getattr(response_attr, "thinking", None)
            else:
                thinking = getattr(response_obj, "thinking", None)

            if thinking is None:
                logger.debug("No thinking content found in response")
        except (AttributeError, TypeError) as e:
            logger.warning("Error accessing thinking content: %s, setting to None", e)
            thinking = None

        response = ReasoningResponseWithThinking(
            thinking=thinking,
            explicit=new_insights.explicit,
            deductive=new_insights.deductive,
        )

        logger.debug(
            "🚀 DEBUG: new_insights=%s, thinking_length=%s",
            new_insights,
            len(thinking) if thinking else 0,
        )

        if settings.LANGFUSE_PUBLIC_KEY:
            lf.update_current_generation(
                output=format_reasoning_response_as_markdown(response),
            )

        return response

    @conditional_observe
    @sentry_sdk.trace
    async def reason(
        self,
        working_representation: ReasoningResponseWithThinking,
        history: str,
        speaker_peer_card: list[str] | None,
    ) -> ReasoningResponseWithThinking:
        """
        Single-pass reasoning function that critically analyzes and derives insights.
        Performs one analysis pass and returns the final observations.
        """
        latest_payload = self.ctx[-1]
        analysis_start = time.perf_counter()

        # Perform critical analysis to get observation lists
        reasoning_response = await self.derive_new_insights(
            working_representation,
            history,
            speaker_peer_card,
        )

        # Output the thinking content for this analysis
        log_thinking_panel(reasoning_response.thinking)

        analysis_duration_ms = (time.perf_counter() - analysis_start) * 1000
        accumulate_metric(
            f"deriver_representation_{latest_payload.message_id}_{latest_payload.target_name}",
            "critical_analysis_duration",
            analysis_duration_ms,
            "ms",
        )

        save_observations_start = time.perf_counter()
        # Save only the NEW observations that weren't in the original context
        new_observations_by_level: dict[
            str, list[str]
        ] = await self._save_new_observations(
            working_representation, reasoning_response, latest_payload
        )
        save_observations_duration = (
            time.perf_counter() - save_observations_start
        ) * 1000
        accumulate_metric(
            f"deriver_representation_{latest_payload.message_id}_{latest_payload.target_name}",
            "save_new_observations",
            save_observations_duration,
            "ms",
        )

        update_peer_card_start = time.perf_counter()
        # flatten new observations by level into a list
        new_observations = [
            extract_observation_content(observation)
            for level in new_observations_by_level.values()
            for observation in level
        ]
        if new_observations:
            await self._update_peer_card(speaker_peer_card, new_observations)
        update_peer_card_duration = (
            time.perf_counter() - update_peer_card_start
        ) * 1000
        accumulate_metric(
            f"deriver_representation_{latest_payload.message_id}_{latest_payload.target_name}",
            "update_peer_card",
            update_peer_card_duration,
            "ms",
        )

        return reasoning_response

    @conditional_observe
    @sentry_sdk.trace
    async def _save_new_observations(
        self,
        original_working_representation: ReasoningResponse
        | ReasoningResponseWithThinking,
        revised_observations: ReasoningResponse | ReasoningResponseWithThinking,
        latest_payload: RepresentationPayload,
    ) -> dict[str, list[str]]:
        """Save only the observations that are new compared to the original context."""
        # Use the utility function to find new observations
        new_observations_by_level: dict[str, list[str]] = find_new_observations(
            original_working_representation, revised_observations
        )

        all_unified_observations: list[UnifiedObservation] = []
        total_observations_count: int = 0

        for level, new_observations in new_observations_by_level.items():
            if not new_observations:
                logger.debug("No new observations to save for %s level", level)
                continue

            logger.debug("Found %s new %s observations", len(new_observations), level)

            # Convert each observation to UnifiedObservation with proper premises and level
            for observation in new_observations:
                if isinstance(observation, DeductiveObservation):
                    # Create UnifiedObservation with premises from DeductiveObservation
                    unified_obs = UnifiedObservation(
                        conclusion=observation.conclusion,
                        premises=observation.premises,
                        level=level,
                    )
                    all_unified_observations.append(unified_obs)
                    logger.debug(
                        "Added %s observation: %s... with %s premises",
                        level,
                        observation.conclusion[:50],
                        len(observation.premises),
                    )

                else:
                    # String observations (explicit) have no premises
                    unified_obs = UnifiedObservation.from_string(
                        observation, level=level
                    )
                    all_unified_observations.append(unified_obs)
                    logger.debug("Added %s observation: %s...", level, observation[:50])

                total_observations_count += 1

        if all_unified_observations:
            await self.embedding_store.save_unified_observations(
                all_unified_observations,
                latest_payload.message_id,
                latest_payload.session_name,
                latest_payload.created_at,
            )
        else:
            logger.debug("No new observations to save")

        return new_observations_by_level

    @conditional_observe
    @sentry_sdk.trace
    async def _update_peer_card(
        self,
        old_peer_card: list[str] | None,
        new_observations: list[str],
    ) -> None:
        """
        Update the peer card by calling LLM with the old peer card and new observations.
        The new peer card is returned by the LLM and saved to peer internal metadata.
        """
        try:
            response = await peer_card_call(old_peer_card, new_observations)
            new_peer_card = response.card
            if not new_peer_card:
                logger.info("No changes to peer card")
                return
            # even with a dedicated notes field, we still need to prune notes out of the card
            new_peer_card = [
                observation
                for observation in new_peer_card
                if not observation.lower().startswith("notes")
            ]
            logger.info("New peer card: %s", new_peer_card)
            async with tracked_db("deriver.update_peer_card") as db:
                await crud.set_peer_card(
                    db,
                    self.ctx[0].workspace_name,
                    self.ctx[0].sender_name,
                    self.ctx[0].target_name,
                    new_peer_card,
                )
        except Exception as e:
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_exception(e)
            logger.error("Error updating peer card! Skipping... %s", e)


def observation_context_to_reasoning_response(
    context: ObservationContext,
) -> ReasoningResponseWithThinking:
    """Convert ObservationContext to ReasoningResponse for compatibility."""
    thinking = context.thinking

    # Convert explicit observations to new structure
    explicit: list[str] = []
    for obs in context.explicit:
        explicit.append(obs.content)

    # Convert deductive observations
    deductive: list[DeductiveObservation] = []
    for obs in context.deductive:
        deductive_obs = DeductiveObservation(
            conclusion=obs.content,
            premises=obs.metadata.premises if obs.metadata else [],
        )
        deductive.append(deductive_obs)

    return ReasoningResponseWithThinking(
        thinking=thinking,
        explicit=explicit,
        deductive=deductive,
    )


@sentry_sdk.trace
async def save_working_representation_to_peer(
    payload: RepresentationPayload,
    final_observations: ReasoningResponseWithThinking,
) -> None:
    """Save working representation to peer internal_metadata for dialectic access."""

    # Convert ReasoningResponse to serializable dict
    final_obs_dict = {
        "thinking": final_observations.thinking,
        "explicit": final_observations.explicit,
        "deductive": [
            {
                "conclusion": obs.conclusion,
                "premises": obs.premises,
            }
            for obs in final_observations.deductive
        ],
    }

    working_rep_data = {
        "final_observations": final_obs_dict,
        "message_id": payload.message_id,
        "created_at": utc_now_iso(),
    }

    async with tracked_db("deriver.save_working_representation") as db:
        await crud.set_working_representation(
            db,
            working_rep_data,
            payload.workspace_name,
            payload.target_name,
            payload.sender_name,
            payload.session_name,
        )


def _estimate_working_representation_tokens(
    working_rep_data: dict[str, Any] | str | None,
) -> int:
    """Estimate tokens for working representation data."""
    if isinstance(working_rep_data, str):
        return estimate_tokens(working_rep_data)

    if (
        not isinstance(working_rep_data, dict)
        or "final_observations" not in working_rep_data
    ):
        return 0

    final_obs = working_rep_data["final_observations"]

    explicit_tokens = estimate_tokens(final_obs.get("explicit"))
    thinking_tokens = estimate_tokens(final_obs.get("thinking"))
    deductive_tokens = sum(
        estimate_tokens(d.get("conclusion")) + estimate_tokens(d.get("premises"))
        for d in final_obs.get("deductive", [])
    )

    return explicit_tokens + thinking_tokens + deductive_tokens
