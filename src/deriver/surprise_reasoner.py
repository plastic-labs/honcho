import logging
import time
from inspect import cleandoc as c

from langfuse.decorators import observe
from mirascope import llm, prompt_template
from sentry_sdk.ai.monitoring import ai_track

from src.deriver.models import ObservationContext, ReasoningResponse
from src.utils.deriver import (
    REASONING_LEVELS,
    analyze_observation_changes,
    extract_observation_content,
    find_new_observations,
    format_context_for_prompt,
    format_context_for_trace,
    format_new_turn_with_timestamp,
)

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True


@prompt_template()
def reasoning_system_prompt() -> str:
    return """

"""


# TODO: Re-enable when Mirascope-Langfuse compatibility issue is fixed
# @with_langfuse()
@llm.call(
    provider="anthropic",
    model="claude-3-7-sonnet-20250219",
    response_model=ReasoningResponse,
    call_params={"max_tokens": 1500, "temperature": 0.7},
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
        Apply rigorous logical reasoning to analyze what we know about the user. ONLY make changes when evidence compels them.

        Work through each reasoning type systematically in your thinking process:

        1. **EXPLICIT OBSERVATIONS**:
        - What facts are LITERALLY stated in the messages? Each observation should be a separate list item.
        - No inference, no reading between lines - only explicit statements
        - Format each as: "User said: [exact quote or clear paraphrase]"
        - Return as a list of individual observations, not a single concatenated string
        - Examples: ["User said: I am 25 years old", "User said: I work as a teacher"]

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

        Remember: STABILITY is valuable. Only change what strong evidence demands.

        Provide your analysis in a structured JSON format:
        - "thinking": string (the thinking process of the LLM)
        - "explicit": array of strings (each starting with "User said:")
        - "deductive": array of objects with "conclusion" and "premises" fields
        - "inductive": array of objects with "conclusion" and "premises" fields  
        - "abductive": array of objects with "conclusion" and "premises" fields

        CRITICAL: The "explicit" field must be an array of strings, not a single string.
        """
    )


class SurpriseReasoner:
    def __init__(self, embedding_store=None):
        self.max_recursion_depth = 3  # Reduced from 5 to prevent excessive recursion
        self.current_depth = 0
        self.embedding_store = embedding_store
        # Minimum percentage of observations that must change to continue recursion
        self.significance_threshold = 0.2  # 20% of observations must change
        # Trace capture for analysis
        self.trace = None

    def _reasoning_response_to_dict(
        self, reasoning_response: ReasoningResponse
    ) -> dict:
        """Convert a ReasoningResponse to a dictionary format for compatibility with existing code."""
        return {
            "explicit": reasoning_response.explicit,
            "deductive": [
                {"conclusion": obs.conclusion, "premises": obs.premises}
                for obs in reasoning_response.deductive
            ],
            "inductive": [
                {"conclusion": obs.conclusion, "premises": obs.premises}
                for obs in reasoning_response.inductive
            ],
            "abductive": [
                {"conclusion": obs.conclusion, "premises": obs.premises}
                for obs in reasoning_response.abductive
            ],
        }

    def _observation_context_to_reasoning_response(
        self, context: "ObservationContext"
    ) -> ReasoningResponse:
        """Convert ObservationContext to ReasoningResponse for compatibility."""
        from src.deriver.models import StructuredObservation

        thinking = context.thinking

        # Convert explicit observations (strings)
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
            "max_depth_reached": max(
                [it["depth"] for it in self.trace["reasoning_iterations"]], default=0
            ),
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
        self, context, history, new_turn, message_id: str, current_time: str
    ) -> ReasoningResponse:
        """
        Critically analyzes and revises understanding, returning structured observations.
        """
        formatted_new_turn = format_new_turn_with_timestamp(new_turn, current_time)
        formatted_context = format_context_for_prompt(context)
        logger.debug(
            f"CRITICAL ANALYSIS: current_time='{current_time}', formatted_new_turn='{formatted_new_turn}'"
        )

        # Log prompts for debugging
        logger.info(f"CRITICAL ANALYSIS - Context length: {len(formatted_context)}")

        # Call the standalone LLM function
        return await critical_analysis_call(
            current_time=current_time,
            context=formatted_context,
            history=history,
            new_turn=formatted_new_turn,
        )

    async def recursive_reason_with_trace(
        self,
        context: ReasoningResponse,
        history: str,
        new_turn: str,
        message_id: str,
        session_id: str | None = None,
        current_time: str | None = None,
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
                context, history, new_turn, message_id, session_id, current_time
            )

            # Finalize trace
            total_duration_ms = int((time.time() - start_time) * 1000)
            convergence_reason = "completed"
            self._finalize_trace(
                self._reasoning_response_to_dict(final_observations),
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

            raise e

    async def recursive_reason(
        self,
        context: ReasoningResponse,
        history: str,
        new_turn: str,
        message_id: str,
        session_id: str | None = None,
        current_time: str | None = None,
    ) -> ReasoningResponse:
        """
        Main recursive reasoning function that critically analyzes and revises understanding.
        Continues recursion only if the LLM makes changes (is "surprised").
        """
        # Check if we've hit the maximum recursion depth
        if self.current_depth >= self.max_recursion_depth:
            logger.warning(
                f"Maximum recursion depth reached.\nSession ID: {session_id}\nMessage ID: {message_id}"
            )
            return context

        # Increment the recursion depth counter
        self.current_depth += 1

        try:
            iteration_start = time.time()

            # Perform critical analysis to get revised observation lists
            reasoning_response = await self.derive_new_insights(
                context, history, new_turn, message_id, current_time
            )

            # Convert ReasoningResponse to dict for compatibility with existing code
            revised_observations = self._reasoning_response_to_dict(reasoning_response)

            logger.debug(f"Critical analysis result: {revised_observations}")

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
                output_observations=revised_observations,
                changes_detected=changes_detected or {},  # Ensure it's always a dict
                significance_score=significance_score,
                threshold_met=has_changes,
                continue_reasoning=has_changes,
                duration_ms=iteration_duration_ms,
            )

            # If no changes were made, the LLM wasn't surprised - exit recursion
            if not has_changes:
                if self.current_depth > 0:
                    logger.info(
                        f"No significant changes detected at depth {self.current_depth} - LLM stabilized. Exiting recursion."
                    )
                else:
                    logger.info(
                        "No changes detected - LLM was not surprised. Exiting recursion."
                    )
                return context

            logger.info("Changes detected - LLM was surprised. Continuing analysis.")

            # Save only the NEW observations that weren't in the original context
            await self._save_new_observations(
                context, reasoning_response, message_id, session_id
            )

            # Pass the revised observations directly to the next iteration
            # The LLM has already curated the most relevant observations in reasoning_response
            logger.debug(
                "Passing revised observations directly to next recursive iteration"
            )
            for level in REASONING_LEVELS:
                level_observations = getattr(reasoning_response, level, [])
                logger.debug(
                    f"Passing {level}: {len(level_observations)} revised observations to next iteration"
                )

            # Recursively analyze with the revised observations from the LLM
            return await self.recursive_reason(
                reasoning_response,
                history,
                new_turn,
                message_id,
                session_id,
                current_time,
            )

        finally:
            # Decrement the recursion depth counter when we're done
            self.current_depth -= 1

    async def _save_new_observations(
        self,
        original_context: ReasoningResponse,
        revised_observations: ReasoningResponse,
        message_id: str,
        session_id: str | None = None,
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
                    new_observations,
                    message_id=message_id,
                    level=level,
                    session_id=session_id,
                )
            else:
                logger.debug(f"No new observations to save for {level} level")

    async def _save_structured_observations(
        self,
        observations: list,
        message_id: str,
        level: str,
        session_id: str | None = None,
    ):
        """Save observations with proper handling of structured data including premises."""
        if not self.embedding_store:
            return

        for observation in observations:
            if (
                isinstance(observation, dict)
                and "conclusion" in observation
                and "premises" in observation
            ):
                # This is a structured observation with premises
                conclusion = observation["conclusion"]
                premises = observation.get("premises", [])

                # Save the conclusion with premises in metadata
                await self.embedding_store.save_observations(
                    [conclusion],  # Only save the conclusion as content
                    message_id=message_id,
                    level=level,
                    session_id=session_id,
                    premises=premises,  # Pass premises directly
                )

                logger.debug(
                    f"Saved structured observation: conclusion='{conclusion}' with {len(premises)} premises in metadata"
                )

            else:
                # Simple observation (string or dict without structure)
                observation_content = extract_observation_content(observation)

                # Save simple observations normally (no premises)
                await self.embedding_store.save_observations(
                    [observation_content],
                    message_id=message_id,
                    level=level,
                    session_id=session_id,
                )
