import datetime
import logging
import time

import sentry_sdk
from langfuse import get_client

from src import crud, exceptions
from src.config import settings
from src.crud.representation import GLOBAL_REPRESENTATION_COLLECTION_NAME
from src.dependencies import tracked_db
from src.utils import summarizer
from src.utils.clients import honcho_llm_call
from src.utils.embedding_store import EmbeddingStore
from src.utils.formatting import format_new_turn_with_timestamp
from src.utils.logging import (
    accumulate_metric,
    conditional_observe,
    format_reasoning_inputs_as_markdown,
    log_performance_metrics,
    log_representation,
)
from src.utils.representation import PromptRepresentation, Representation
from src.utils.shared_models import PeerCardQuery

from .prompts import critical_analysis_prompt, peer_card_prompt
from .queue_payload import RepresentationPayload

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

lf = get_client()


async def critical_analysis_call(
    peer_id: str,
    peer_card: list[str] | None,
    message_created_at: datetime.datetime,
    working_representation: str | None,
    history: str,
    new_turn: str,
) -> PromptRepresentation:
    prompt = critical_analysis_prompt(
        peer_id=peer_id,
        peer_card=peer_card,
        message_created_at=message_created_at,
        working_representation=working_representation,
        history=history,
        new_turn=new_turn,
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
        thinking_budget_tokens=settings.DERIVER.THINKING_BUDGET_TOKENS,
        enable_retry=True,
        retry_attempts=3,
    )

    return response.content


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


@conditional_observe
@sentry_sdk.trace
async def process_representation_task(
    payload: RepresentationPayload,
) -> None:
    """
    Process a representation task by extracting insights and updating working representations.
    """
    # Start overall timing
    overall_start = time.perf_counter()

    logger.debug("Starting insight extraction for user message: %s", payload.message_id)

    # Use get_session_context_formatted with configurable token limit
    async with tracked_db("deriver.get_session_context") as db:
        formatted_history = await summarizer.get_session_context_formatted(
            db,
            payload.workspace_name,
            payload.session_name,
            token_limit=settings.DERIVER.CONTEXT_TOKEN_LIMIT,
            cutoff=payload.message_id,
            include_summary=True,
        )

    # instantiate embedding store from collection
    # if the sender is also the target, we're handling a global representation task.
    # otherwise, we're handling a directional representation task where the sender is
    # being observed by the target.
    collection_name = (
        crud.construct_collection_name(
            observer=payload.target_name, observed=payload.sender_name
        )
        if payload.sender_name != payload.target_name
        else GLOBAL_REPRESENTATION_COLLECTION_NAME
    )

    # get_or_create_collection already handles IntegrityError with rollback and a retry
    async with tracked_db("deriver.get_or_create_collection") as db:
        collection = await crud.get_or_create_collection(
            db,
            payload.workspace_name,
            collection_name,
            payload.sender_name,
        )
        collection_name_loaded = collection.name

    # Use the embedding store directly
    embedding_store = EmbeddingStore(
        workspace_name=payload.workspace_name,
        peer_name=payload.sender_name,
        collection_name=collection_name_loaded,
    )

    # Create reasoner instance
    reasoner = CertaintyReasoner(embedding_store=embedding_store, ctx=payload)

    # Check for existing working representation first, fall back to global search
    async with tracked_db("deriver.get_working_representation_data") as db:
        working_representation = await crud.get_working_representation(
            db, payload.workspace_name, payload.target_name, payload.sender_name
        )

    # Time context preparation
    context_prep_start = time.perf_counter()
    if working_representation:
        logger.info(
            "Using existing working representation with %s explicit, %s deductive observations",
            len(working_representation.explicit),
            len(working_representation.deductive),
        )
    else:
        # No existing working representation, use global search
        logger.info("No working representation found, using global semantic search")
        working_representation = await embedding_store.get_relevant_observations(
            query=payload.content,
            conversation_context=formatted_history,
        )

    context_prep_duration = (time.perf_counter() - context_prep_start) * 1000
    accumulate_metric(
        f"deriver_representation_{payload.message_id}_{payload.target_name}",
        "context_preparation",
        context_prep_duration,
        "ms",
    )

    async with tracked_db("deriver.get_peer_card") as db:
        speaker_peer_card: list[str] | None = await crud.get_peer_card(
            db, payload.workspace_name, payload.sender_name, payload.target_name
        )
    if speaker_peer_card is None:
        logger.warning("No peer card found for %s", payload.sender_name)
    else:
        logger.info("Using peer card: %s", speaker_peer_card)

    # Run single-pass reasoning
    final_observations, new_observations_count = await reasoner.reason(
        working_representation,
        formatted_history,
        speaker_peer_card,
    )

    # Display final observations in a beautiful tree
    log_representation(final_observations)

    # Calculate and log overall timing
    overall_duration = (time.perf_counter() - overall_start) * 1000
    accumulate_metric(
        f"deriver_representation_{payload.message_id}_{payload.target_name}",
        "total_processing_time",
        overall_duration,
        "ms",
    )

    total_observations = len(final_observations.explicit) + len(
        final_observations.deductive
    )

    accumulate_metric(
        f"deriver_representation_{payload.message_id}_{payload.target_name}",
        "final_observation_count",
        total_observations,
        "",
    )

    accumulate_metric(
        f"deriver_representation_{payload.message_id}_{payload.target_name}",
        "new_observation_count",
        new_observations_count,
        "",
    )

    log_performance_metrics(
        f"deriver_representation_{payload.message_id}_{payload.target_name}"
    )

    if settings.LANGFUSE_PUBLIC_KEY:
        lf.update_current_trace(output=final_observations.format_as_markdown())


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
    async def reason(
        self,
        working_representation: Representation,
        history: str,
        speaker_peer_card: list[str] | None,
    ) -> tuple[Representation, int]:
        """
        Single-pass reasoning function that critically analyzes and derives insights.
        Performs one analysis pass and returns the final observations and count of new observations.

        Returns:
            tuple[Representation, int]: Final observations and count of new observations added
        """
        analysis_start = time.perf_counter()

        # Perform critical analysis to get observation lists
        if settings.LANGFUSE_PUBLIC_KEY:
            lf.update_current_generation(
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
        formatted_working_representation = str(working_representation)

        logger.debug(
            "CRITICAL ANALYSIS: message_created_at='%s', formatted_new_turn='%s'",
            self.ctx.created_at,
            formatted_new_turn,
        )

        try:
            reasoning_response = await critical_analysis_call(
                peer_id=self.ctx.sender_name,
                peer_card=speaker_peer_card,
                message_created_at=self.ctx.created_at,
                working_representation=formatted_working_representation,
                history=history,
                new_turn=formatted_new_turn,
            )
        except Exception as e:
            raise exceptions.LLMError(
                speaker_peer_card=speaker_peer_card,
                working_representation=formatted_working_representation,
                history=history,
                new_turn=formatted_new_turn,
            ) from e

        reasoning_response = reasoning_response.to_representation(
            self.ctx.message_id, self.ctx.session_name
        )

        if settings.LANGFUSE_PUBLIC_KEY:
            lf.update_current_generation(
                output=reasoning_response.format_as_markdown(),
            )

        analysis_duration_ms = (time.perf_counter() - analysis_start) * 1000
        accumulate_metric(
            f"deriver_representation_{self.ctx.message_id}_{self.ctx.target_name}",
            "critical_analysis_duration",
            analysis_duration_ms,
            "ms",
        )

        save_observations_start = time.perf_counter()
        # Save only the new observations that weren't in the original context
        new_observations = working_representation.diff_representation(
            reasoning_response
        )
        if not new_observations.is_empty():
            await self.embedding_store.save_representation(
                new_observations,
                self.ctx.message_id,
                self.ctx.session_name,
                self.ctx.created_at,
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

        # Store the count of new observations for metrics
        new_observations_count = len(new_observations.explicit) + len(
            new_observations.deductive
        )

        update_peer_card_start = time.perf_counter()
        if not new_observations.is_empty():
            await self._update_peer_card(speaker_peer_card, new_observations)
        update_peer_card_duration = (
            time.perf_counter() - update_peer_card_start
        ) * 1000
        accumulate_metric(
            f"deriver_representation_{self.ctx.message_id}_{self.ctx.target_name}",
            "update_peer_card",
            update_peer_card_duration,
            "ms",
        )

        return reasoning_response, new_observations_count

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
                    self.ctx.workspace_name,
                    self.ctx.sender_name,
                    self.ctx.target_name,
                    new_peer_card,
                )
        except Exception as e:
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_exception(e)
            logger.error("Error updating peer card! Skipping... %s", e)
