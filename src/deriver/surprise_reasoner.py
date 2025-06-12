import logging
import time
from inspect import cleandoc as c
import datetime
import asyncio
import os

from langfuse.decorators import observe
from mirascope import llm
from sentry_sdk.ai.monitoring import ai_track
from sqlalchemy.ext.asyncio import AsyncSession

from src.deriver.models import (
    ObservationContext,
    ReasoningResponse,
    StructuredObservation,
)
from src.deriver.tom.embeddings import CollectionEmbeddingStore
from src.utils.deriver import (
    REASONING_LEVELS,
    analyze_observation_changes,
    extract_observation_content,
    find_new_observations,
    format_context_for_prompt,
    format_context_for_trace,
    format_new_turn_with_timestamp,
)
from src.utils.logging import (
    log_changes_table,
    log_error_panel,
    log_observations_tree,
    log_performance_metrics,
    log_thinking_panel,
)

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True


# TODO: Re-enable when Mirascope-Langfuse compatibility issue is fixed
# @with_langfuse()
@llm.call(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    response_model=ReasoningResponse,
    call_params={"max_tokens": 1500, "temperature": 0.7},
    json_mode=True,
)
async def critical_analysis_call(
    current_time: str, context: str, history: str, new_turn: str
):
    """
    Standalone LLM call function for critical analysis using Mirascope.
    """
    return c(
        f"""
        You are an expert at critically analyzing user understanding through rigorous logical reasoning while maintaining appropriate stability. Your goal is to IMPROVE understanding through careful analysis, not to change things unnecessarily.

        **REASONING MODES - STRICT DEFINITIONS:**

        1. **EXPLICIT OBSERVATIONS**: 
        - Direct, literal facts stated by the user in their messages
        - No interpretation or inference - only what is explicitly written
        - Examples: "I am 25 years old", "I work as a teacher", "I live in NYC"

        2. **DEDUCTIVE OBSERVATIONS**: 
        - Facts that MUST be true given explicit observations OR previous deductive observations as premises
        - Follow strict logical necessity - if premises are true, conclusion MUST be true
        - Can scaffold: use both explicit facts and previously deduced facts as premises
        - Example: If user says "I'm teaching my 5th grade class" → They teach elementary school

        3. **INDUCTIVE OBSERVATIONS**:
        - Highly probable generalizations based on patterns in certain observations (explicit and deductive)
        - Strong likelihood based on repeated evidence, but not logically guaranteed
        - Example: User mentions coding problems across 5 conversations → They likely work in tech

        4. **ABDUCTIVE OBSERVATIONS**: 
        - Plausible explanatory hypotheses that best explain the totality of observations
        - Provisional conclusions that offer the most coherent explanation for observed patterns
        - Example: Technical discussions + late night messages + coffee mentions → Possibly a startup founder

        **REASONING HIERARCHY & MOVEMENT:**
        - Explicit → Deductive: When logical necessity allows certain conclusion
        - Explicit/Deductive → Further Deductive: Can use certain facts to deduce more certain facts
        - Explicit/Deductive → Inductive: When patterns strongly support generalization
        - All observations → Abductive: When seeking best explanation for overall pattern
        - Abductive → Inductive: When hypothesis gains strong repeated support
        - Inductive → Deductive: When generalization becomes explicitly confirmed

        **CONSERVATIVE ANALYSIS PRINCIPLE:**
        Only make changes when evidence COMPELS change. The current understanding may be accurate and well-founded. Stability is valuable.

        **EVIDENCE STANDARDS BY TYPE:**
        1. **Explicit**: Only what is literally stated - no interpretation
        2. **Deductive**: Must follow with logical necessity - no exceptions possible
        3. **Inductive**: Requires multiple supporting instances, strong pattern
        4. **Abductive**: Must be best available explanation, coherent with all observations

        **TEMPORAL CONTEXT AWARENESS:**
        Facts may include temporal metadata showing access patterns:
        - **High access count**: Indicates core, stable traits
        - **Recent access**: Shows immediately relevant patterns  
        - **Cross-session consistency**: Suggests reliable patterns
        - **Low/single access**: May indicate situational information

        **QUALITY OVER CHANGE:**
        Maintaining accurate existing insights is better than unnecessary modifications. Every change should demonstrably improve understanding.

        You will respond with a structured analysis containing four types of observations. Be thorough but conservative - only include observations that meet the strict evidence standards for each reasoning type.


        Current date and time: {current_time}

        Here's the current user understanding to be CRITICALLY EXAMINED:
        <current_context>
        {context}
        </current_context>

        Recent conversation history for context:
        <history>
        {history}
        </history>

        New conversation turn to analyze:
        <new_turn>
        {new_turn}
        </new_turn>

        **YOUR CRITICAL MISSION:**
        Apply rigorous logical reasoning to analyze what we know about the user. ONLY make changes when evidence compels them. Work through each reasoning type systematically in your thinking process:

        1. **EXPLICIT OBSERVATIONS**:
        - What facts are LITERALLY stated in the messages? When writing these premises, append "User said:" to the beginning.
        - No inference, no reading between lines - only explicit statements
        - Include only direct quotes or clear paraphrases, be sure to denote them as such.

        2. **DEDUCTIVE OBSERVATIONS**:
        - Given the explicit facts as premises, what MUST be true?
        - Apply strict logical deduction - if premises are true, what conclusions are CERTAIN?
        - Test each deduction: "If X is true, then Y MUST be true" (not just likely)

        3. **INDUCTIVE OBSERVATIONS**:
        - What patterns emerge from multiple observations?
        - What generalizations have STRONG support (not just single instances)?
        - How confident can we be in these patterns? (High probability required)

        4. **ABDUCTIVE OBSERVATIONS**:
        - What hypotheses BEST explain all our observations?
        - What plausible explanations make sense of the overall pattern?
        - Which explanations are most coherent with everything we know?

        For each category, ask:
        - Is there COMPELLING evidence to change existing observations?
        - Are current observations still well-supported?
        - Would changes genuinely improve understanding?

        Provide your analysis in a structured JSON format:
        {{
            "thinking": "Your critical analysis of the user's understanding, including your reasoning process and the changes you made to the observations.",
            "explicit": [
                "List of facts LITERALLY stated by the user",
                "Direct quotes or clear paraphrases only",
                "No interpretation or inference"
            ],
            "deductive": [
                {{
                    "conclusion": "Conclusions that MUST be true given explicit facts and premises as well as strict logical necessities from premises",
                    "premises": ["Explicit fact or prior deductive conclusion used", "Another premise if needed"]
                }},
                {{
                    "conclusion": "...",
                    "premises": ["..."]
                }}
            ],
            "inductive": [
                {{
                    "conclusion": "Highly probable patterns based on multiple observations, strong generalizations with substantial support, high confidence predictions about user behavior/preferences",
                    "premises": ["Multiple observations supporting this", "Evidence from explicit/deductive observations", "Repeated instances"]
                }},
                {{
                    "conclusion": "...",
                    "premises": ["..."]
                }}
            ],
            "abductive": [
                {{
                    "conclusion": "Best explanatory hypotheses for all observations, plausible theories about identity/motivations/context, coherent explanations that fit the overall pattern",
                    "premises": ["All types of observations this explains", "Patterns it accounts for", "Facts it makes coherent"]
                }},
                {{
                    "conclusion": "...",
                    "premises": ["..."]
                }}
            ],
            "queries": [
                "Query to learn more about the user, to either confirm or inform future reasoning",
                "Query to learn more about the user, to either confirm or inform future reasoning",
                ...
            ]
        }}
        
        Remember: STABILITY is valuable. Only change what strong evidence demands.
        """
    )


class SurpriseReasoner:
    def __init__(self, embedding_store: CollectionEmbeddingStore):
        self.max_recursion_depth = 3  # Reduced from 5 to prevent excessive recursion
        self.current_depth = 0
        self.embedding_store = embedding_store
        # Minimum percentage of observations that must change to continue recursion
        self.significance_threshold = 0.2  # 20% of observations must change
        # Trace capture for analysis
        self.trace = None

    def _observation_context_to_reasoning_response(
        self, context: "ObservationContext"
    ) -> ReasoningResponse:
        """Convert ObservationContext to ReasoningResponse for compatibility."""
        thinking = context.thinking

        # Convert explicit observations (strings) without modifying content
        explicit = [obs.content for obs in context.explicit]

        # Convert structured observations
        deductive = []
        for obs in context.deductive:
            structured_obs = StructuredObservation(
                conclusion=obs.content,
                premises=obs.metadata.premises if obs.metadata else [],
            )
            deductive.append(structured_obs)

        inductive = []
        for obs in context.inductive:
            structured_obs = StructuredObservation(
                conclusion=obs.content,
                premises=obs.metadata.premises if obs.metadata else [],
            )
            inductive.append(structured_obs)

        abductive = []
        for obs in context.abductive:
            structured_obs = StructuredObservation(
                conclusion=obs.content,
                premises=obs.metadata.premises if obs.metadata else [],
            )
            abductive.append(structured_obs)

        return ReasoningResponse(
            thinking=thinking,
            explicit=explicit,
            deductive=deductive,
            inductive=inductive,
            abductive=abductive,
        )

    def _init_trace(
        self,
        message_id: str,
        session_id: str | None,
        user_message: str,
        initial_context: ReasoningResponse,
    ):
        """Initialize trace capture for this reasoning session."""
        self.trace = {
            "message_id": message_id,
            "session_id": session_id or "unknown",
            "user_message": user_message,
            "timestamp": time.time(),
            "initial_context": format_context_for_trace(
                initial_context, include_similarity_scores=True
            ),
            "reasoning_iterations": [],
            "final_observations": {},
            "saved_documents": {},
            "summary": {},
        }

    def _capture_iteration(
        self,
        depth: int,
        input_context: ReasoningResponse,
        thinking: str,
        output_observations: dict,
        changes_detected: dict,
        significance_score: float,
        threshold_met: bool,
        continue_reasoning: bool,
        duration_ms: int,
    ):
        """Capture data for a single reasoning iteration."""
        if not self.trace:
            return

        iteration = {
            "depth": depth,
            "input_context": format_context_for_trace(input_context),
            "thinking": thinking,
            "output_observations": output_observations,
            "changes_detected": changes_detected,
            "significance_score": significance_score,
            "threshold_met": threshold_met,
            "continue_reasoning": continue_reasoning,
            "iteration_duration_ms": duration_ms,
        }

        self.trace["reasoning_iterations"].append(iteration)

    def _finalize_trace(
        self, final_observations: dict, convergence_reason: str, total_duration_ms: int
    ):
        """Finalize the trace with summary information."""
        if not self.trace:
            return

        self.trace["final_observations"] = final_observations
        self.trace["summary"] = {
            "total_iterations": len(self.trace["reasoning_iterations"]),
            "total_observations_added": sum(
                [
                    len(it["changes_detected"].get(level, {}).get("added", []))
                    for it in self.trace["reasoning_iterations"]
                    for level in REASONING_LEVELS
                ]
            ),
            "total_observations_modified": sum(
                [
                    len(it["changes_detected"].get(level, {}).get("modified", []))
                    for it in self.trace["reasoning_iterations"]
                    for level in REASONING_LEVELS
                ]
            ),
            "total_duration_ms": total_duration_ms,
            "convergence_reason": convergence_reason,
        }

    def _adjust_context_for_depth(self):
        """
        Adjust the embedding store's observation retrieval counts based on recursion depth
        to progressively focus on the most relevant observations.
        """
        if not self.embedding_store:
            return

        # Progressively reduce context size as depth increases
        depth_factor = max(0.4, 1.0 - (self.current_depth * 0.15))

        # Calculate adjusted counts (minimum of 1 for each level)
        adjusted_abductive = max(1, int(2 * depth_factor))
        adjusted_inductive = max(2, int(4 * depth_factor))
        adjusted_deductive = max(3, int(6 * depth_factor))

        logger.debug(
            f"Depth {self.current_depth}: Adjusting context to {adjusted_abductive}/{adjusted_inductive}/{adjusted_deductive} observations"
        )

        self.embedding_store.set_observation_counts(
            abductive=adjusted_abductive,
            inductive=adjusted_inductive,
            deductive=adjusted_deductive,
        )

    @observe()
    @ai_track("Derive New Insights")
    async def derive_new_insights(
        self, context, history, new_turn, current_time: str
    ) -> ReasoningResponse:
        """
        Critically analyzes and revises understanding, returning structured observations.
        """
        formatted_new_turn = format_new_turn_with_timestamp(new_turn, current_time)
        formatted_context = format_context_for_prompt(context)
        logger.debug(
            f"CRITICAL ANALYSIS: current_time='{current_time}', formatted_new_turn='{formatted_new_turn}'"
        )

        # Call the standalone LLM function with retries & rich error logging
        return await self._critical_analysis_with_retry(
            current_time=current_time,
            context=formatted_context,
            history=history,
            new_turn=formatted_new_turn,
        )

    async def recursive_reason_with_trace(
        self,
        db: AsyncSession,
        context: ReasoningResponse,
        history: str,
        new_turn: str,
        message_id: str,
        session_id: str | None = None,
        current_time: str | None = None,
        message_created_at: datetime.datetime | None = None,
    ) -> tuple[ReasoningResponse, dict]:
        """
        Main entry point for recursive reasoning with full trace capture.
        Returns final observations and complete trace.
        """
        start_time = time.time()

        # Initialize trace capture
        self._init_trace(message_id, session_id, new_turn, context)

        # Reset depth counter
        self.current_depth = 0

        try:
            # Run recursive reasoning
            final_observations = await self.recursive_reason(
                db, context, history, new_turn, message_id, session_id, current_time, message_created_at
            )

            # Build obs dict with timestamps for trace
            final_obs_with_dates = self._attach_created_at(final_observations, message_created_at)

            # Finalize trace
            total_duration_ms = int((time.time() - start_time) * 1000)
            convergence_reason = "completed"
            self._finalize_trace(
                final_obs_with_dates,
                convergence_reason,
                total_duration_ms,
            )

            # Ensure trace is not None before returning
            return final_observations, self.trace or {}

        except Exception as e:
            # Finalize trace even on error
            total_duration_ms = int((time.time() - start_time) * 1000)
            convergence_reason = f"error: {str(e)}"
            self._finalize_trace({}, convergence_reason, total_duration_ms)

            # Log error with rich formatting
            log_error_panel(
                logger,
                e,
                f"Recursive reasoning failed\nMessage ID: {message_id}\nSession ID: {session_id}\nDepth: {self.current_depth}",
            )

            raise e

    async def recursive_reason(
        self,
        db: AsyncSession,
        context: ReasoningResponse,
        history: str,
        new_turn: str,
        message_id: str,
        session_id: str | None = None,
        current_time: str | None = None,
        message_created_at: datetime.datetime | None = None,
        query_results: dict[str, list[str]] | None = None,
    ) -> ReasoningResponse:
        """
        Main recursive reasoning function that critically analyzes and revises understanding.
        Continues recursion only if the LLM makes changes (is "surprised").
        """
        # Check if we've hit the maximum recursion depth
        if self.current_depth >= self.max_recursion_depth:
            logger.warning(
                f"[bold red]🚨 Maximum recursion depth ({self.max_recursion_depth}) reached![/]\n"
                f"Session ID: {session_id}\nMessage ID: {message_id}",
                extra={"markup": True},
            )
            return context

        # Increment the recursion depth counter
        self.current_depth += 1

        try:
            iteration_start = time.time()

            # Perform critical analysis to get revised observation lists
            reasoning_response = await self.derive_new_insights(
                context, history, new_turn, current_time
            )

            # Process queries if any were generated
            if reasoning_response.queries:
                logger.info(
                    f"Executing {len(reasoning_response.queries)} queries: {reasoning_response.queries}"
                )
                query_execution = await self.embedding_store.execute_queries(
                    reasoning_response.queries
                )
                logger.debug(
                    f"Query execution returned {query_execution.total_observations} observations"
                )

            # Output the thinking content for this recursive iteration
            log_thinking_panel(reasoning_response.thinking, self.current_depth)

            # Compare input context with output to detect changes (surprise)
            # Apply depth-based conservatism - require more significant changes at deeper levels
            effective_threshold = self.significance_threshold * (
                1 + 0.5 * self.current_depth
            )
            if self.current_depth > 0:
                logger.info(
                    f"Depth {self.current_depth}: Using higher significance threshold: {effective_threshold:.1%}"
                )

            # Use the unified change analysis function
            result = analyze_observation_changes(
                context, reasoning_response, effective_threshold, include_details=True
            )

            # Unpack the result - it should be a tuple when include_details=True
            if isinstance(result, tuple) and len(result) == 3:
                has_changes, changes_detected, significance_score = result
            else:
                # Fallback if something went wrong
                has_changes = False
                changes_detected = {}
                significance_score = 0.0
                logger.error(
                    f"Unexpected result from analyze_observation_changes: {result}"
                )

            # Calculate iteration duration
            iteration_duration_ms = int((time.time() - iteration_start) * 1000)

            # Capture this iteration in trace
            self._capture_iteration(
                depth=self.current_depth,
                input_context=context,
                thinking=reasoning_response.thinking,
                output_observations=reasoning_response.model_dump(),
                changes_detected=changes_detected or {},  # Ensure it's always a dict
                significance_score=significance_score,
                threshold_met=has_changes,
                continue_reasoning=has_changes,
                duration_ms=iteration_duration_ms,
            )

            # Log changes in a beautiful table
            log_changes_table(
                changes_detected or {}, significance_score, effective_threshold
            )

            # If no changes were made, the LLM wasn't surprised - exit recursion
            if not has_changes:
                if self.current_depth > 0:
                    logger.info(
                        f"[yellow]🔄 Depth {self.current_depth}:[/] LLM stabilized. Exiting recursion.",
                        extra={"markup": True},
                    )
                else:
                    logger.info(
                        "[yellow]🔄 LLM was not surprised. Exiting recursion.[/]",
                        extra={"markup": True},
                    )
                return context

            logger.info(
                "[green]✨ Changes detected - LLM was surprised. Continuing analysis.[/]",
                extra={"markup": True},
            )

            # Save only the NEW observations that weren't in the original context
            await self._save_new_observations(
                db, context, reasoning_response, message_id, session_id, message_created_at
            )

            # Display observations in a tree structure and performance metrics
            observations = {
                level: getattr(reasoning_response, level, [])
                for level in REASONING_LEVELS
            }
            log_observations_tree(
                observations,
                f"📊 REVISED OBSERVATIONS (Depth {self.current_depth})",
            )

            # Log performance metrics for this iteration
            metrics = {
                "iteration_duration": iteration_duration_ms,
                "significance_score": significance_score,
                "recursion_depth": self.current_depth,
            }
            log_performance_metrics(
                metrics, f"⚡ ITERATION {self.current_depth} METRICS"
            )

            # Recursively analyze with the revised observations from the LLM
            return await self.recursive_reason(
                db,
                reasoning_response,
                history,
                new_turn,
                message_id,
                session_id,
                current_time,
                message_created_at,
            )

        finally:
            # Decrement the recursion depth counter when we're done
            self.current_depth -= 1

    async def _save_new_observations(
        self,
        db: AsyncSession,
        original_context: ReasoningResponse,
        revised_observations: ReasoningResponse,
        message_id: str,
        session_id: str | None = None,
        message_created_at: datetime.datetime | None = None,
    ):
        """Save only the observations that are new compared to the original context."""
        if not self.embedding_store:
            return

        # Use the utility function to find new observations
        new_observations_by_level = find_new_observations(
            original_context, revised_observations
        )

        for level, new_observations in new_observations_by_level.items():
            if new_observations:
                logger.debug(
                    f"Saving {len(new_observations)} new {level} observations: {new_observations}"
                )
                await self._save_structured_observations(
                    db,
                    new_observations,
                    message_id=message_id,
                    level=level,
                    session_id=session_id,
                    message_created_at=message_created_at,
                )
            else:
                logger.debug(f"No new observations to save for {level} level")

    async def _save_structured_observations(
        self,
        db: AsyncSession,
        observations: list,
        message_id: str,
        level: str,
        session_id: str | None = None,
        message_created_at: datetime.datetime | None = None,
    ):
        """Save observations with proper handling of structured data including premises."""
        if not self.embedding_store:
            return

        for observation in observations:
            if isinstance(observation, StructuredObservation):
                # Handle StructuredObservation Pydantic objects
                conclusion = observation.conclusion
                premises = observation.premises
                await self.embedding_store.save_observations(
                    db,
                    [conclusion],  # Only save the conclusion as content
                    message_id=message_id,
                    level=level,
                    session_id=session_id,
                    premises=premises,  # Pass premises in metadata
                    message_created_at=message_created_at,
                )

                logger.debug(
                    f"Saved structured observation: conclusion='{conclusion}' with {len(premises)} premises"
                )

            elif isinstance(observation, str):
                # Handle simple string observations (like explicit observations)
                await self.embedding_store.save_observations(
                    db,
                    [observation],
                    message_id=message_id,
                    level=level,
                    session_id=session_id,
                    message_created_at=message_created_at,
                )

                logger.debug(f"Saved simple observation: '{observation[:50]}...'")

            else:
                # Fallback: extract content for unknown types
                observation_content = extract_observation_content(observation)
                await self.embedding_store.save_observations(
                    db,
                    [observation_content],
                    message_id=message_id,
                    level=level,
                    session_id=session_id,
                    message_created_at=message_created_at,
                )

                logger.debug(
                    f"Saved fallback observation: '{observation_content[:50]}...'"
                )
                logger.warning(f"Unexpected observation type: {type(observation)}")

    # ---------------------------------------------------------------------
    # Helper: attach created_at timestamps to ReasoningResponse for trace
    # ---------------------------------------------------------------------

    def _attach_created_at(
        self,
        observations: ReasoningResponse,
        created_at: datetime.datetime | None,
    ) -> dict:
        """Return a dict version of observations with created_at on every item."""

        ts = created_at.isoformat() if created_at else None

        def explicit_item(content: str):
            return {"content": content, "created_at": ts}

        def structured_item(ob):
            return {
                "conclusion": ob.conclusion,
                "premises": ob.premises,
                "created_at": ts,
            }

        return {
            "explicit": [explicit_item(c) for c in observations.explicit],
            "deductive": [structured_item(o) for o in observations.deductive],
            "inductive": [structured_item(o) for o in observations.inductive],
            "abductive": [structured_item(o) for o in observations.abductive],
        }

    # ------------------------------------------------------------------
    # Lightweight retry logic for LLM call with rich error logging
    # ------------------------------------------------------------------

    async def _critical_analysis_with_retry(
        self,
        *,
        current_time: str,
        context: str,
        history: str,
        new_turn: str,
    ) -> ReasoningResponse:
        """Call ``critical_analysis_call`` with simple exponential-backoff retries.

        The maximum number of retries can be configured with the ``LLM_MAX_RETRIES``
        environment variable (defaults to 3).  Failures are logged and stored in
        the deriver trace (``llm_errors`` list) including input/output token counts
        when Mirascope attaches a ``_response`` object to the exception.
        """

        max_retries = int(os.getenv("LLM_MAX_RETRIES", 3))

        for attempt in range(1, max_retries + 1):
            try:
                return await critical_analysis_call(
                    current_time=current_time,
                    context=context,
                    history=history,
                    new_turn=new_turn,
                )
            except Exception as e:  # noqa: BLE001
                # Log and record the failure; re-raise if final attempt
                self._log_llm_error(e, attempt, max_retries)

                if attempt >= max_retries:
                    raise

                # Simple exponential backoff: 1s, 2s, 4s, ...
                await asyncio.sleep(2 ** (attempt - 1))

        # This line is never reached due to raise above, but keeps mypy happy
        raise RuntimeError("_critical_analysis_with_retry exhausted without return")

    def _log_llm_error(self, exc: Exception, attempt: int, max_retries: int) -> None:  # noqa: D401
        """Log rich info about an LLM exception and append to trace."""

        # Basic info
        logger.warning(
            f"[LLM retry {attempt}/{max_retries}] {type(exc).__name__}: {exc}",
            extra={"markup": False},
        )

        # Extract response if Mirascope attached it
        resp = getattr(exc, "_response", None)

        error_record: dict[str, object] = {
            "attempt": attempt,
            "max_retries": max_retries,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

        try:
            if resp is not None:
                error_record.update(
                    {
                        "provider": getattr(resp, "provider", None),
                        "model": getattr(resp, "model", None),
                        "input_tokens": getattr(resp, "input_tokens", None),
                        "output_tokens": getattr(resp, "output_tokens", None),
                        "cached_tokens": getattr(resp, "cached_tokens", None),
                        "cost": getattr(resp, "cost", None),
                        "max_tokens": getattr(resp.call_params, "max_tokens", None)
                        if hasattr(resp, "call_params")
                        else None,
                        "temperature": getattr(resp.call_params, "temperature", None)
                        if hasattr(resp, "call_params")
                        else None,
                        "message_count": len(getattr(resp, "messages", [])),
                    }
                )
        except Exception as inner_exc:  # noqa: BLE001
            # Ensure logging itself never blows up
            logger.debug(f"Failed to enrich LLM error metadata: {inner_exc}")

        # Store in trace
        if self.trace is not None:
            self.trace.setdefault("llm_errors", []).append(error_record)
