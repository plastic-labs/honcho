import datetime
import logging
import time

import sentry_sdk

from src import crud, exceptions
from src.config import settings
from src.crud.representation import GLOBAL_REPRESENTATION_COLLECTION_NAME
from src.dependencies import tracked_db
from src.utils import summarizer
from src.utils.clients import honcho_llm_call
from src.utils.embedding_store import EmbeddingStore
from src.utils.formatting import format_new_turn_with_timestamp
from src.utils.langfuse_client import get_langfuse_client
from src.utils.logging import (
    accumulate_metric,
    conditional_observe,
    format_reasoning_inputs_as_markdown,
    log_performance_metrics,
    log_representation,
)
from src.utils.queue_payload import RepresentationPayload
from src.utils.representation import PromptRepresentation, Representation
from src.utils.shared_models import PeerCardQuery

from .prompts import critical_analysis_prompt, peer_card_prompt

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

lf = get_langfuse_client()


async def critical_analysis_call(
    peer_id: str,
    peer_card: list[str] | None,
    message_created_at: datetime.datetime,
    working_representation: Representation,
    history: str,
    new_turns: list[str],
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

    accumulate_metric(
        f"deriver_{latest_payload.message_id}_{latest_payload.target_name}",
        "starting_message_id",
        earliest_payload.message_id,
        "id",
    )

    accumulate_metric(
        f"deriver_{latest_payload.message_id}_{latest_payload.target_name}",
        "ending_message_id",
        latest_payload.message_id,
        "id",
    )

    # Start overall timing
    overall_start = time.perf_counter()

    # Time context preparation
    context_prep_start = time.perf_counter()

    # Use get_session_context_formatted with configurable token limit
    async with tracked_db("deriver.get_context+representation+peer_card") as db:
        formatted_history = await summarizer.get_session_context_formatted(
            db,
            latest_payload.workspace_name,
            latest_payload.session_name,
            token_limit=settings.DERIVER.CONTEXT_TOKEN_LIMIT,
            cutoff=latest_payload.message_id,
            include_summary=True,
        )

        working_representation = await crud.get_working_representation(
            db,
            latest_payload.workspace_name,
            latest_payload.target_name,
            latest_payload.sender_name,
            # include_semantic_query=latest_payload.content,
            # include_most_derived=False,
        )

        if settings.DERIVER.USE_PEER_CARD:
            speaker_peer_card: list[str] | None = await crud.get_peer_card(
                db,
                latest_payload.workspace_name,
                latest_payload.sender_name,
                latest_payload.target_name,
            )
            if speaker_peer_card is None:
                logger.warning(
                    "No peer card found for %s. Normal if brand-new peer.",
                    latest_payload.sender_name,
                )
            else:
                logger.info("Using peer card: %s", speaker_peer_card)
        else:
            speaker_peer_card = None

    # got working representation and peer card, log timing
    context_prep_duration = (time.perf_counter() - context_prep_start) * 1000
    accumulate_metric(
        f"deriver_{latest_payload.message_id}_{latest_payload.target_name}",
        "context_preparation",
        context_prep_duration,
        "ms",
    )

    logger.info(
        "Using working representation with %s explicit, %s deductive observations",
        len(working_representation.explicit),
        len(working_representation.deductive),
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

    # Use the embedding store directly
    embedding_store = EmbeddingStore(
        workspace_name=latest_payload.workspace_name,
        peer_name=latest_payload.sender_name,
        collection_name=collection_name,
    )

    # Create reasoner instance
    reasoner = CertaintyReasoner(embedding_store=embedding_store, ctx=payloads)

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
        f"deriver_{latest_payload.message_id}_{latest_payload.target_name}",
        "total_processing_time",
        overall_duration,
        "ms",
    )

    total_observations = len(final_observations.explicit) + len(
        final_observations.deductive
    )

    accumulate_metric(
        f"deriver_{latest_payload.message_id}_{latest_payload.target_name}",
        "observation_count",
        total_observations,
        "count",
    )

    log_performance_metrics(
        "deriver", f"{latest_payload.message_id}_{latest_payload.target_name}"
    )

    if settings.LANGFUSE_PUBLIC_KEY:
        lf.update_current_trace(output=final_observations.format_as_markdown())


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
        # For logging, we can just show the content of the last message
        earliest_payload = self.ctx[0]
        latest_payload = self.ctx[-1]

        # Perform critical analysis to get observation lists
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

        logger.debug(
            "CRITICAL ANALYSIS: message_created_at='%s', new_turns_count=%s",
            latest_payload.created_at,
            len(new_turns),
        )

        try:
            reasoning_response = await critical_analysis_call(
                peer_id=latest_payload.sender_name,
                peer_card=speaker_peer_card,
                message_created_at=latest_payload.created_at,
                working_representation=working_representation,
                history=history,
                new_turns=new_turns,
            )
        except Exception as e:
            raise exceptions.LLMError(
                speaker_peer_card=speaker_peer_card,
                working_representation=working_representation,
                history=history,
                new_turns=new_turns,
            ) from e

        reasoning_response = reasoning_response.to_representation(
            (earliest_payload.message_id, latest_payload.message_id),
            latest_payload.session_name,
            latest_payload.created_at,
        )

        if settings.LANGFUSE_PUBLIC_KEY:
            lf.update_current_generation(
                output=reasoning_response.format_as_markdown(),
            )

        analysis_duration_ms = (time.perf_counter() - analysis_start) * 1000
        accumulate_metric(
            f"deriver_{latest_payload.message_id}_{latest_payload.target_name}",
            "critical_analysis_duration",
            analysis_duration_ms,
            "ms",
        )

        # Save only the new observations that weren't in the original context
        new_observations = working_representation.diff_representation(
            reasoning_response
        )
        if not new_observations.is_empty():
            await self.embedding_store.save_representation(
                new_observations,
                (earliest_payload.message_id, latest_payload.message_id),
                latest_payload.session_name,
                latest_payload.created_at,
            )

        # not currently deduplicating at the save_representation step, so this isn't useful
        # accumulate_metric(
        #     f"deriver_{latest_payload.message_id}_{latest_payload.target_name}",
        #     "new_observation_count",
        #     new_observations_saved,
        #     "count",
        # )

        if settings.DERIVER.USE_PEER_CARD:
            update_peer_card_start = time.perf_counter()
            if not new_observations.is_empty():
                await self._update_peer_card(speaker_peer_card, new_observations)
            update_peer_card_duration = (
                time.perf_counter() - update_peer_card_start
            ) * 1000
            accumulate_metric(
                f"deriver_{latest_payload.message_id}_{latest_payload.target_name}",
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
            logger.info("Jettisoned notes from peer card: %s", response.notes)
            new_peer_card = response.card
            if not new_peer_card:
                logger.info("No changes to peer card")
                return
            # even with a dedicated notes field, we still need to prune notes out of the card
            new_peer_card = [
                observation
                for observation in new_peer_card
                if not observation.lower().startswith(("note", "notes"))
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
