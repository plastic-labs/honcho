import datetime
import logging
import time
from typing import Any

from httpx import AsyncClient
from langfuse.decorators import langfuse_context
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud
from src.config import settings
from src.dependencies import tracked_db
from src.utils import summarizer
from src.utils.clients import honcho_llm_call
from src.utils.embedding_store import EmbeddingStore
from src.utils.formatting import (
    REASONING_LEVELS,
    extract_observation_content,
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
from src.webhooks.webhook_delivery import webhook_delivery_service

from .prompts import critical_analysis_prompt
from .queue_payload import (
    DeriverQueuePayload,
    RepresentationPayload,
    SummaryPayload,
    WebhookQueuePayload,
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
    peer_name: str,
    message_created_at: datetime.datetime,
    context: str,
    history: str,
    new_turn: str,
):
    return critical_analysis_prompt(
        peer_name=peer_name,
        message_created_at=message_created_at,
        context=context,
        history=history,
        new_turn=new_turn,
    )


@conditional_observe
class Deriver:
    """Deriver class for processing messages and extracting insights."""

    async def process_webhook(
        self,
        client: AsyncClient,
        payload: WebhookQueuePayload,
    ) -> None:
        async with tracked_db() as db:
            await webhook_delivery_service.deliver_webhook(db, client, payload)

    async def process_message(
        self,
        task_type: str,
        payload: DeriverQueuePayload,
    ) -> None:
        """
        Process a user message by extracting insights and saving them to the vector store.
        This runs as a background process after a user message is logged.
        """

        if settings.LANGFUSE_PUBLIC_KEY:
            langfuse_context.update_current_trace(
                metadata={
                    "critical_analysis_model": settings.DERIVER.MODEL,
                }
            )

        # Open a DB session only for the duration of the processing call
        async with tracked_db("deriver") as db:
            if task_type == "summary":
                await self.process_summary_task(db, payload)
            elif task_type == "representation":
                await self.process_representation_task(db, payload)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

    async def process_summary_task(
        self,
        db: AsyncSession,
        payload: SummaryPayload,
    ) -> None:
        """
        Process a summary task by generating summaries if needed.
        """
        await summarizer.summarize_if_needed(
            db,
            payload.workspace_name,
            payload.session_name,
            payload.message_id,
            payload.message_seq_in_session,
        )
        log_performance_metrics(f"deriver_message_{payload.message_id}")

    async def process_representation_task(
        self,
        db: AsyncSession,
        payload: RepresentationPayload,
    ) -> None:
        """
        Process a representation task by extracting insights and updating working representations.
        """
        # Start overall timing
        overall_start = time.perf_counter()

        # Extract variables from payload for cleaner access
        content = payload.content
        workspace_name = payload.workspace_name
        session_name = payload.session_name
        message_id = payload.message_id
        sender_name = payload.sender_name
        target_name = payload.target_name
        created_at = payload.created_at

        logger.debug("Starting insight extraction for user message: %s", message_id)

        # Use message timestamp instead of wall-clock time for reasoning/insight dating
        # created_at is now always a datetime object from Pydantic validation
        message_dt_obj = created_at

        formatted_history = await summarizer.get_summarized_history(
            db,
            workspace_name,
            session_name,
            cutoff=message_id,
            summary_type=summarizer.SummaryType.SHORT,
        )

        # instantiate embedding store from collection
        collection_name = (
            crud.construct_collection_name(observer=target_name, observed=sender_name)
            if sender_name != target_name
            else "global_representation"
        )
        try:
            collection = await crud.get_or_create_collection(
                db, workspace_name, collection_name, sender_name
            )
        except Exception as e:
            # Handle race condition from concurrent processing
            if "duplicate key" in str(e).lower():
                # Rollback the failed transaction
                await db.rollback()
                # Collection already exists, fetch it
                collection = await crud.get_collection(
                    db, workspace_name, collection_name, sender_name
                )
            else:
                raise

        # Use the embedding store directly
        embedding_store = EmbeddingStore(
            workspace_name=workspace_name,
            peer_name=sender_name,
            collection_name=collection.name,
        )

        # Create reasoner instance
        reasoner = CertaintyReasoner(embedding_store=embedding_store)

        # Check for existing working representation first, fall back to global search
        working_rep_data: (
            dict[str, Any] | str | None
        ) = await crud.get_working_representation_data(
            db, workspace_name, target_name, sender_name, session_name
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

            initial_reasoning_context = ReasoningResponseWithThinking(
                thinking=final_obs.get("thinking"),
                explicit=final_obs.get("explicit", []),
                deductive=deductive_observations,
            )
            logger.info(
                "Using existing working representation with %s explicit, %s deductive observations",
                len(initial_reasoning_context.explicit),
                len(initial_reasoning_context.deductive),
            )
        else:
            # No working representation, use global search
            initial_context = await embedding_store.get_relevant_observations(
                query=content,
                conversation_context=formatted_history,
                for_reasoning=True,
            )

            initial_reasoning_context = (
                reasoner.observation_context_to_reasoning_response(initial_context)
            )
            logger.info("No working representation found, using global semantic search")
        context_prep_duration = (time.perf_counter() - context_prep_start) * 1000
        accumulate_metric(
            f"deriver_message_{message_id}",
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
            initial_reasoning_context,
            formatted_history,
            content,
            str(message_id),  # Convert int to str
            session_name,
            message_dt_obj,
            sender_name,  # Pass the speaker name
        )

        logger.debug(
            "REASONING COMPLETION: Unified reasoning completed across all levels."
        )

        # Display final observations in a beautiful tree
        final_obs_dict = {
            level: getattr(final_observations, level, []) for level in REASONING_LEVELS
        }
        log_observations_tree(final_obs_dict)

        # Always save working representation to peer for dialectic access
        await save_working_representation_to_peer(
            db,
            workspace_name,
            target_name,  # observer (whose metadata we update)
            sender_name,  # observed (for key calculation)
            session_name,
            final_observations,
            message_id,
        )

        # Calculate and log overall timing
        overall_duration = (time.perf_counter() - overall_start) * 1000
        accumulate_metric(
            f"deriver_message_{message_id}",
            "total_processing_time",
            overall_duration,
            "ms",
        )

        total_observations = sum(len(obs_list) for obs_list in final_obs_dict.values())

        accumulate_metric(
            f"deriver_message_{message_id}",
            "final_observation_count",
            total_observations,
            "",
        )
        log_performance_metrics(f"deriver_message_{message_id}")

        if settings.LANGFUSE_PUBLIC_KEY:
            langfuse_context.update_current_trace(
                output=format_reasoning_response_as_markdown(final_observations)
            )


class CertaintyReasoner:
    """Certainty reasoner for analyzing and deriving insights."""

    embedding_store: EmbeddingStore

    def __init__(self, embedding_store: EmbeddingStore) -> None:
        self.embedding_store = embedding_store

    def observation_context_to_reasoning_response(
        self, context: "ObservationContext"
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

    @conditional_observe
    async def derive_new_insights(
        self,
        context: ReasoningResponseWithThinking,
        history: str,
        new_turn: str,
        message_created_at: datetime.datetime,
        speaker: str,
    ) -> ReasoningResponseWithThinking:
        """
        Critically analyzes and revises understanding, returning structured observations.
        """

        if settings.LANGFUSE_PUBLIC_KEY:
            langfuse_context.update_current_observation(
                input=format_reasoning_inputs_as_markdown(
                    context, history, new_turn, message_created_at
                )
            )

        formatted_new_turn = format_new_turn_with_timestamp(
            new_turn, message_created_at, speaker
        )
        formatted_context = format_context_for_prompt(context)

        logger.debug(
            "CRITICAL ANALYSIS: message_created_at='%s', formatted_new_turn='%s'",
            message_created_at,
            formatted_new_turn,
        )

        # Call the standalone LLM function (now with Tenacity retries)
        response_obj = await critical_analysis_call(
            peer_name=speaker,
            message_created_at=message_created_at,
            context=formatted_context,
            history=history,
            new_turn=formatted_new_turn,
        )

        # Handle different response types
        if isinstance(response_obj, str):
            # If response is a string, try to parse as JSON
            import json

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
                logger.warning(f"Failed to parse string response as JSON: {e}")
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
            logger.warning(f"Error accessing thinking content: {e}, setting to None")
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
    async def reason(
        self,
        context: ReasoningResponseWithThinking,
        history: str,
        new_turn: str,
        message_id: str,
        session_name: str | None = None,
        message_created_at: datetime.datetime | None = None,
        speaker: str = "user",
    ) -> ReasoningResponseWithThinking:
        """
        Single-pass reasoning function that critically analyzes and derives insights.
        Performs one analysis pass and returns the final observations.
        """
        if message_created_at is None:
            message_created_at = datetime.datetime.now(datetime.timezone.utc)

        analysis_start = time.perf_counter()

        # Perform critical analysis to get observation lists
        reasoning_response = await self.derive_new_insights(
            context, history, new_turn, message_created_at, speaker
        )

        # Output the thinking content for this analysis
        log_thinking_panel(reasoning_response.thinking)

        analysis_duration_ms = (time.perf_counter() - analysis_start) * 1000
        accumulate_metric(
            f"deriver_message_{message_id}",
            "critical_analysis_duration",
            analysis_duration_ms,
            "ms",
        )

        save_observations_start = time.perf_counter()
        # Save only the NEW observations that weren't in the original context
        await self._save_new_observations(
            context,
            reasoning_response,
            message_id,
            session_name,
            message_created_at,
        )
        save_observations_duration = (
            time.perf_counter() - save_observations_start
        ) * 1000
        accumulate_metric(
            f"deriver_message_{message_id}",
            "save_new_observations",
            save_observations_duration,
            "ms",
        )

        return reasoning_response

    @conditional_observe
    async def _save_new_observations(
        self,
        original_context: ReasoningResponse,
        revised_observations: ReasoningResponse,
        message_id: str,
        session_name: str | None = None,
        message_created_at: datetime.datetime | None = None,
    ) -> None:
        """Save only the observations that are new compared to the original context."""
        # Use the utility function to find new observations
        new_observations_by_level = find_new_observations(
            original_context, revised_observations
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

                elif isinstance(observation, str):
                    # String observations (explicit) have no premises
                    unified_obs = UnifiedObservation.from_string(
                        observation, level=level
                    )
                    all_unified_observations.append(unified_obs)
                    logger.debug("Added %s observation: %s...", level, observation[:50])

                else:
                    # Handle unexpected types
                    content = extract_observation_content(observation)
                    unified_obs = UnifiedObservation.from_string(content, level=level)
                    all_unified_observations.append(unified_obs)
                    logger.warning(
                        f"Added unexpected observation type: {type(observation)} as {level}"
                    )

                total_observations_count += 1

        if not all_unified_observations:
            logger.debug("No new observations to save")
            return

        await self.embedding_store.save_unified_observations(
            all_unified_observations,
            message_id=message_id,
            session_name=session_name,
            message_created_at=message_created_at,
        )


async def save_working_representation_to_peer(
    db: AsyncSession,
    workspace_name: str,
    observer_name: str,  # renamed from peer_name for clarity
    observed_name: str,  # new parameter
    session_name: str | None,
    final_observations: ReasoningResponseWithThinking,
    message_id: int,
) -> None:
    """Save working representation to peer internal_metadata for dialectic access."""
    from sqlalchemy import update

    from src import models

    # Determine metadata key based on observer/observed relationship
    if observer_name == observed_name:
        metadata_key = "global_representation"
    else:
        metadata_key = crud.construct_collection_name(
            observer=observer_name, observed=observed_name
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
        "message_id": message_id,
        "created_at": datetime.datetime.now().isoformat(),
    }

    # if session_name is supplied, save working representation to session peer
    if session_name:
        stmt = (
            update(models.SessionPeer)
            .where(
                models.SessionPeer.workspace_name == workspace_name,
                models.SessionPeer.session_name == session_name,
                models.SessionPeer.peer_name == observer_name,
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
            f"Saved working representation to session peer {session_name} - {observer_name} with key {metadata_key}"
        )
    else:
        # For peer-level messages (session_name=None), only save global representations
        if observer_name == observed_name:
            stmt = (
                update(models.Peer)
                .where(
                    models.Peer.workspace_name == workspace_name,
                    models.Peer.name == observer_name,
                )
                .values(
                    internal_metadata=models.Peer.internal_metadata.op("||")(
                        {metadata_key: working_rep_data}
                    )
                )
            )

            await db.execute(stmt)
            await db.commit()

            logger.debug(
                "Saved working representation to peer %s with key %s",
                observer_name,
                metadata_key,
            )
        else:
            logger.debug(
                "Skipping peer-level local representation save: observer=%s, observed=%s",
                observer_name,
                observed_name,
            )
