import datetime
import json
import logging
import time
from typing import Any

import sentry_sdk
from langfuse.decorators import langfuse_context
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud
from src.config import settings
from src.utils import summarizer
from src.utils.clients import honcho_llm_call
from src.utils.embedding_store import EmbeddingStore
from src.utils.formatting import (
    REASONING_LEVELS,
    find_new_observations,
    format_context_for_prompt,
    format_new_turn_with_timestamp,
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
    ReasoningResponse,
    ReasoningResponseWithThinking,
    UnifiedObservation,
)

from .prompts import NO_CHANGES_RESPONSE, critical_analysis_prompt, peer_card_prompt
from .queue_payload import (
    RepresentationPayload,
)

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True


@honcho_llm_call(
    provider=settings.DERIVER.PROVIDER,
    model=settings.DERIVER.MODEL,
    track_name="Critical Analysis Call",
    response_model=ReasoningResponse,
    json_mode=True,
    max_tokens=settings.DERIVER.MAX_OUTPUT_TOKENS or settings.LLM.DEFAULT_MAX_TOKENS,
    thinking_budget_tokens=settings.DERIVER.THINKING_BUDGET_TOKENS
    if settings.DERIVER.PROVIDER == "anthropic"
    else None,
    enable_retry=True,
    retry_attempts=3,
)
async def critical_analysis_call(
    peer_card: str | None,
    message_created_at: datetime.datetime,
    working_representation: str | None,
    history: str,
    new_turn: str,
):
    return critical_analysis_prompt(
        peer_card=peer_card,
        message_created_at=message_created_at,
        working_representation=working_representation,
        history=history,
        new_turn=new_turn,
    )


@honcho_llm_call(
    provider=settings.DERIVER.PEER_CARD_PROVIDER,
    model=settings.DERIVER.PEER_CARD_MODEL,
    track_name="Peer Card Call",
    max_tokens=settings.DERIVER.PEER_CARD_MAX_OUTPUT_TOKENS
    or settings.LLM.DEFAULT_MAX_TOKENS,
    thinking_budget_tokens=None,
    reasoning_effort="minimal",
    verbosity="low",
    enable_retry=True,
    retry_attempts=1,  # unstructured output means we shouldn't need to retry, 1 just in case
)
async def peer_card_call(
    old_peer_card: str | None,
    new_observations: list[str],
):
    return peer_card_prompt(
        old_peer_card=old_peer_card,
        new_observations=new_observations,
    )


@conditional_observe
@sentry_sdk.trace
async def process_representation_task(
    db: AsyncSession,
    payload: RepresentationPayload,
) -> None:
    """
    Process a representation task by extracting insights and updating working representations.
    """
    # Start overall timing
    overall_start = time.perf_counter()

    logger.debug("Starting insight extraction for user message: %s", payload.message_id)

    # Use get_session_context_formatted with configurable token limit
    formatted_history = await summarizer.get_session_context_formatted(
        db,
        payload.workspace_name,
        payload.session_name,
        token_limit=settings.DERIVER.CONTEXT_TOKEN_LIMIT,
        cutoff=payload.message_id,
        include_summary=True,
    )

    # instantiate embedding store from collection
    collection_name = (
        crud.construct_collection_name(
            observer=payload.target_name, observed=payload.sender_name
        )
        if payload.sender_name != payload.target_name
        else "global_representation"
    )

    try:
        collection = await crud.get_or_create_collection(
            db,
            payload.workspace_name,
            collection_name,
            payload.sender_name,
        )
    except Exception as e:
        # Handle race condition from concurrent processing
        if "duplicate key" in str(e).lower():
            # Rollback the failed transaction
            await db.rollback()
            # Collection already exists, fetch it
            collection = await crud.get_collection(
                db,
                payload.workspace_name,
                collection_name,
                payload.sender_name,
            )
        else:
            raise

    # Use the embedding store directly
    embedding_store = EmbeddingStore(
        workspace_name=payload.workspace_name,
        peer_name=payload.sender_name,
        collection_name=collection.name,
    )

    # Create reasoner instance
    reasoner = CertaintyReasoner(embedding_store=embedding_store, ctx=payload)

    # Check for existing working representation first, fall back to global search
    working_rep_data: (
        dict[str, Any] | str | None
    ) = await crud.get_working_representation_data(
        db,
        payload.workspace_name,
        payload.target_name,
        payload.sender_name,
        payload.session_name,
    )

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
        working_representation = await embedding_store.get_relevant_observations(
            query=payload.content,
            conversation_context=formatted_history,
            for_reasoning=True,
        )

        working_representation = observation_context_to_reasoning_response(
            working_representation
        )
        logger.info("No working representation found, using global semantic search")
    context_prep_duration = (time.perf_counter() - context_prep_start) * 1000
    accumulate_metric(
        f"deriver_representation_{payload.message_id}_{payload.target_name}",
        "context_preparation",
        context_prep_duration,
        "ms",
    )

    # Run consolidated reasoning that handles explicit and deductive levels
    logger.debug(
        "REASONING: Running unified insight derivation across explicit and deductive reasoning levels"
    )

    # We currently only use Peer Cards in Honcho-level representation derivation.
    if payload.sender_name == payload.target_name:
        sender_peer_card: str | None = await crud.get_peer_card(
            db, payload.workspace_name, payload.sender_name
        )
        if sender_peer_card is None:
            logger.warning("No peer card found for %s", payload.sender_name)
        else:
            logger.info("Using peer card: %s", sender_peer_card)
    else:
        logger.info("No peer card used for directional representation derivation")
        sender_peer_card = None

    # Run single-pass reasoning
    final_observations = await reasoner.reason(
        db,
        working_representation,
        formatted_history,
        sender_peer_card,
    )

    logger.debug("REASONING COMPLETION: Unified reasoning completed across all levels.")

    # Display final observations in a beautiful tree
    final_obs_dict = {
        level: getattr(final_observations, level, []) for level in REASONING_LEVELS
    }
    log_observations_tree(final_obs_dict)

    # Always save working representation to peer for dialectic access
    await save_working_representation_to_peer(db, payload, final_observations)

    # Calculate and log overall timing
    overall_duration = (time.perf_counter() - overall_start) * 1000
    accumulate_metric(
        f"deriver_representation_{payload.message_id}_{payload.target_name}",
        "total_processing_time",
        overall_duration,
        "ms",
    )

    total_observations = sum(len(obs_list) for obs_list in final_obs_dict.values())

    accumulate_metric(
        f"deriver_representation_{payload.message_id}_{payload.target_name}",
        "final_observation_count",
        total_observations,
        "",
    )
    log_performance_metrics(
        f"deriver_representation_{payload.message_id}_{payload.target_name}"
    )

    if settings.LANGFUSE_PUBLIC_KEY:
        langfuse_context.update_current_trace(
            output=format_reasoning_response_as_markdown(final_observations)
        )


class CertaintyReasoner:
    """Certainty reasoner for analyzing and deriving insights."""

    embedding_store: EmbeddingStore
    ctx: RepresentationPayload

    def __init__(
        self, embedding_store: EmbeddingStore, ctx: RepresentationPayload
    ) -> None:
        self.embedding_store = embedding_store
        self.ctx = ctx

    @conditional_observe
    @sentry_sdk.trace
    async def derive_new_insights(
        self,
        working_representation: ReasoningResponseWithThinking,
        history: str,
        speaker_peer_card: str | None,
    ) -> ReasoningResponseWithThinking:
        """
        Critically analyzes and revises understanding, returning structured observations.
        """

        if settings.LANGFUSE_PUBLIC_KEY:
            langfuse_context.update_current_observation(
                input=format_reasoning_inputs_as_markdown(
                    working_representation,
                    history,
                    self.ctx.content,
                    self.ctx.created_at,
                )
            )

        formatted_new_turn = format_new_turn_with_timestamp(
            self.ctx.content,
            self.ctx.created_at,
            self.ctx.sender_name,
        )
        formatted_working_representation = format_context_for_prompt(
            working_representation
        )

        logger.debug(
            "CRITICAL ANALYSIS: message_created_at='%s', formatted_new_turn='%s'",
            self.ctx.created_at,
            formatted_new_turn,
        )

        # Call the standalone LLM function (now with Tenacity retries)
        response_obj = await critical_analysis_call(
            peer_card=speaker_peer_card,
            message_created_at=self.ctx.created_at,
            working_representation=formatted_working_representation,
            history=history,
            new_turn=formatted_new_turn,
        )

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
            langfuse_context.update_current_observation(
                output=format_reasoning_response_as_markdown(response),
            )

        return response

    @conditional_observe
    @sentry_sdk.trace
    async def reason(
        self,
        db: AsyncSession,
        working_representation: ReasoningResponseWithThinking,
        history: str,
        speaker_peer_card: str | None,
    ) -> ReasoningResponseWithThinking:
        """
        Single-pass reasoning function that critically analyzes and derives insights.
        Performs one analysis pass and returns the final observations.
        """
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
            f"deriver_representation_{self.ctx.message_id}_{self.ctx.target_name}",
            "critical_analysis_duration",
            analysis_duration_ms,
            "ms",
        )

        save_observations_start = time.perf_counter()
        # Save only the NEW observations that weren't in the original context
        new_observations_by_level: dict[
            str, list[str]
        ] = await self._save_new_observations(
            working_representation, reasoning_response
        )
        save_observations_duration = (
            time.perf_counter() - save_observations_start
        ) * 1000
        accumulate_metric(
            f"deriver_representation_{self.ctx.message_id}_{self.ctx.target_name}",
            "save_new_observations",
            save_observations_duration,
            "ms",
        )

        # Only update peer card if we are in Honcho-level representation derivation.
        if self.ctx.sender_name == self.ctx.target_name:
            update_peer_card_start = time.perf_counter()
            # flatten new observations by level into a list
            new_observations = [
                observation
                for level in new_observations_by_level.values()
                for observation in level
            ]
            await self._update_peer_card(db, speaker_peer_card, new_observations)
            update_peer_card_duration = (
                time.perf_counter() - update_peer_card_start
            ) * 1000
            accumulate_metric(
                f"deriver_representation_{self.ctx.message_id}_{self.ctx.target_name}",
                "update_peer_card",
                update_peer_card_duration,
                "ms",
            )

        return reasoning_response

    @conditional_observe
    @sentry_sdk.trace
    async def _save_new_observations(
        self,
        original_working_representation: ReasoningResponse,
        revised_observations: ReasoningResponse,
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
                self.ctx.message_id,
                self.ctx.session_name,
                self.ctx.created_at,
            )
        else:
            logger.debug("No new observations to save")

        return new_observations_by_level

    @conditional_observe
    @sentry_sdk.trace
    async def _update_peer_card(
        self,
        db: AsyncSession,
        old_peer_card: str | None,
        new_observations: list[str],
    ) -> None:
        """
        Update the peer card by calling LLM with the old peer card and new observations.
        The new peer card is returned by the LLM and saved to peer internal metadata.
        """
        try:
            response = await peer_card_call(old_peer_card, new_observations)
            new_peer_card = str(response)
            if NO_CHANGES_RESPONSE in new_peer_card or new_peer_card == "":
                logger.info("No changes to peer card")
                return
            logger.info("New peer card: %s", new_peer_card)
            await crud.set_peer_card(
                db, self.ctx.workspace_name, self.ctx.sender_name, new_peer_card
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
    db: AsyncSession,
    payload: RepresentationPayload,
    final_observations: ReasoningResponseWithThinking,
) -> None:
    """Save working representation to peer internal_metadata for dialectic access."""
    from sqlalchemy import update

    from src import models

    # Determine metadata key based on observer/observed relationship
    if payload.sender_name == payload.target_name:
        metadata_key = "global_representation"
    else:
        metadata_key = crud.construct_collection_name(
            observer=payload.target_name, observed=payload.sender_name
        )

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
        "created_at": datetime.datetime.now().isoformat(),
    }

    # if session_name is supplied, save working representation to session peer
    stmt = (
        update(models.SessionPeer)
        .where(
            models.SessionPeer.workspace_name == payload.workspace_name,
            models.SessionPeer.session_name == payload.session_name,
            models.SessionPeer.peer_name == payload.target_name,
        )
        .values(
            internal_metadata=models.SessionPeer.internal_metadata.op("||")(
                {metadata_key: working_rep_data}
            )
        )
    )
    await db.execute(stmt)
    await db.commit()
    logger.info(
        "Saved working representation to session peer %s - %s with key %s",
        payload.session_name,
        payload.target_name,
        metadata_key,
    )
