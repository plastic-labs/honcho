import json
import logging
import time
from typing import Any

import sentry_sdk
from langfuse.decorators import langfuse_context, observe
from sentry_sdk.ai.monitoring import ai_track

from src.agent import parse_xml_content
from src.utils.model_client import ModelClient, ModelProvider
from src.utils.deriver import (
    REASONING_LEVELS,
    extract_observation_content,
    ensure_context_structure,
    format_new_turn_with_timestamp,
    format_context_for_prompt,
    format_context_for_trace,
    analyze_observation_changes,
    find_new_observations
)

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

PROVIDER = ModelProvider.ANTHROPIC
MODEL = "claude-3-7-sonnet-20250219"

REASONING_SYSTEM_PROMPT = """
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
"""

REASONING_USER_PROMPT = """
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

<thinking>
Work through each reasoning type systematically:

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

Remember: STABILITY is valuable. Only change what strong evidence demands.
</thinking>

Format your response as follows:
<response>
{{
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
    ]
}}
</response>

If evidence isn't compelling enough to warrant changes, return the SAME observations. Unnecessary changes degrade system quality.
"""

class SurpriseReasoner:
    def __init__(self, embedding_store=None):
        self.max_recursion_depth = 3  # Reduced from 5 to prevent excessive recursion
        self.current_depth = 0
        self.embedding_store = embedding_store
        # Minimum percentage of observations that must change to continue recursion
        self.significance_threshold = 0.2  # 20% of observations must change
        # Trace capture for analysis
        self.trace = None
    
    def _init_trace(self, message_id: str, session_id: str | None, user_message: str, initial_context: dict):
        """Initialize trace capture for this reasoning session."""
        self.trace = {
            "message_id": message_id,
            "session_id": session_id or "unknown",
            "user_message": user_message,
            "timestamp": time.time(),
            "initial_context": format_context_for_trace(initial_context, include_similarity_scores=True),
            "reasoning_iterations": [],
            "final_observations": {},
            "saved_documents": {},
            "summary": {}
        }
    
    def _capture_iteration(self, depth: int, input_context: dict, raw_response: str, 
                          output_observations: dict, changes_detected: dict, significance_score: float, 
                          threshold_met: bool, continue_reasoning: bool, duration_ms: int):
        """Capture data for a single reasoning iteration."""
        if not self.trace:
            return
            
        # Extract reasoning trace from <thinking> tags
        reasoning_trace = parse_xml_content(raw_response, "thinking") or "No thinking content found"
        
        iteration = {
            "depth": depth,
            "input_context": format_context_for_trace(input_context),
            "reasoning_trace": reasoning_trace,
            "output_observations": output_observations,
            "changes_detected": changes_detected,
            "significance_score": significance_score,
            "threshold_met": threshold_met,
            "continue_reasoning": continue_reasoning,
            "iteration_duration_ms": duration_ms
        }
        
        self.trace["reasoning_iterations"].append(iteration)
    
    def _finalize_trace(self, final_observations: dict, convergence_reason: str, total_duration_ms: int):
        """Finalize the trace with summary information."""
        if not self.trace:
            return
            
        self.trace["final_observations"] = final_observations
        self.trace["summary"] = {
            "total_iterations": len(self.trace["reasoning_iterations"]),
            "max_depth_reached": max([it["depth"] for it in self.trace["reasoning_iterations"]], default=0),
            "total_observations_added": sum([
                len(it["changes_detected"].get(level, {}).get("added", []))
                for it in self.trace["reasoning_iterations"]
                for level in REASONING_LEVELS
            ]),
            "total_observations_modified": sum([
                len(it["changes_detected"].get(level, {}).get("modified", []))
                for it in self.trace["reasoning_iterations"]
                for level in REASONING_LEVELS
            ]),
            "total_duration_ms": total_duration_ms,
            "convergence_reason": convergence_reason
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
        
        logger.debug(f"Depth {self.current_depth}: Adjusting context to {adjusted_abductive}/{adjusted_inductive}/{adjusted_deductive} observations")
        
        self.embedding_store.set_observation_counts(
            abductive=adjusted_abductive,
            inductive=adjusted_inductive,
            deductive=adjusted_deductive
        )

    @observe()
    @ai_track("Derive New Insights")
    async def derive_new_insights(self, context, history, new_turn, message_id: str, current_time: str) -> tuple[dict[str, list[str]], str]:
        """
        Critically analyzes and revises understanding, returning final observation lists at each level.
        """
        system_prompt = REASONING_SYSTEM_PROMPT
        formatted_new_turn = format_new_turn_with_timestamp(new_turn, current_time)
        formatted_context = format_context_for_prompt(context)
        logger.debug(f"CRITICAL ANALYSIS: current_time='{current_time}', formatted_new_turn='{formatted_new_turn}'")
        user_prompt = REASONING_USER_PROMPT.format(
            current_time=current_time,
            context=formatted_context, 
            history=history, 
            new_turn=formatted_new_turn
        )

        # Log prompts for debugging
        logger.info(f"CRITICAL ANALYSIS PROMPT:\n{user_prompt}")

        client = ModelClient(provider=PROVIDER, model=MODEL)

        # prepare the messages
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt},
        ]

        langfuse_context.update_current_observation(input=messages, model=MODEL)

        try:
            # generate the response
            response = await client.generate(
                messages=messages,
                system=system_prompt,
                max_tokens=1500,  # Increased for more complex responses
                temperature=0.7,  # Slightly lower for more consistent critical analysis
                use_caching=True,
            )
            logger.info(f"Critical analysis response: {response}")
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(f"Error generating critical analysis response: {e}")
            raise e

        # Parse the response to extract final observation lists
        try:
            # Find and parse the JSON response between <response> tags
            try:
                json_str = response[response.find("<response>")+10:response.find("</response>")].strip()
                revised_observations = json.loads(json_str)
            except (ValueError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse critical analysis response: {e}")
                logger.error(f"Response: {response}")
                return ensure_context_structure({}), response
            
            # Ensure we have the expected structure
            final_observations = ensure_context_structure(revised_observations)
            
            return final_observations, response
            
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse critical analysis response: {e}")
            return ensure_context_structure({}), response

    async def recursive_reason_with_trace(self, context: dict, history: str, new_turn: str, message_id: str, session_id: str | None = None, current_time: str | None = None) -> tuple[dict, dict]:
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
            final_observations = await self.recursive_reason(context, history, new_turn, message_id, session_id, current_time)
            
            # Finalize trace
            total_duration_ms = int((time.time() - start_time) * 1000)
            convergence_reason = "completed"
            self._finalize_trace(final_observations, convergence_reason, total_duration_ms)
            
            # Ensure trace is not None before returning
            return final_observations, self.trace or {}
            
        except Exception as e:
            # Finalize trace even on error
            total_duration_ms = int((time.time() - start_time) * 1000)
            convergence_reason = f"error: {str(e)}"
            self._finalize_trace({}, convergence_reason, total_duration_ms)
            
            raise e

    async def recursive_reason(self, context: dict, history: str, new_turn: str, message_id: str, session_id: str | None = None, current_time: str | None = None) -> dict:
        """
        Main recursive reasoning function that critically analyzes and revises understanding.
        Continues recursion only if the LLM makes changes (is "surprised").
        """
        # Check if we've hit the maximum recursion depth
        if self.current_depth >= self.max_recursion_depth:
            logger.warning(f"Maximum recursion depth reached.\nSession ID: {session_id}\nMessage ID: {message_id}")
            return context

        # Increment the recursion depth counter
        self.current_depth += 1

        try:
            iteration_start = time.time()
            
            # Perform critical analysis to get revised observation lists
            revised_observations, raw_response = await self.derive_new_insights(context, history, new_turn, message_id, current_time)

            logger.debug(f"Critical analysis result: {revised_observations}")
            
            # Compare input context with output to detect changes (surprise)
            # Apply depth-based conservatism - require more significant changes at deeper levels
            effective_threshold = self.significance_threshold * (1 + 0.5 * self.current_depth)
            if self.current_depth > 0:
                logger.info(f"Depth {self.current_depth}: Using higher significance threshold: {effective_threshold:.1%}")
            
            # Use the unified change analysis function
            result = analyze_observation_changes(
                context, revised_observations, effective_threshold, include_details=True
            )
            
            # Unpack the result - it should be a tuple when include_details=True
            if isinstance(result, tuple) and len(result) == 3:
                has_changes, changes_detected, significance_score = result
            else:
                # Fallback if something went wrong
                has_changes = False
                changes_detected = {}
                significance_score = 0.0
                logger.error(f"Unexpected result from analyze_observation_changes: {result}")
            
            # Calculate iteration duration
            iteration_duration_ms = int((time.time() - iteration_start) * 1000)
            
            # Capture this iteration in trace
            self._capture_iteration(
                depth=self.current_depth,
                input_context=context,
                raw_response=raw_response,
                output_observations=revised_observations,
                changes_detected=changes_detected or {},  # Ensure it's always a dict
                significance_score=significance_score,
                threshold_met=has_changes,
                continue_reasoning=has_changes,
                duration_ms=iteration_duration_ms
            )
            
            # If no changes were made, the LLM wasn't surprised - exit recursion
            if not has_changes:
                if self.current_depth > 0:
                    logger.info(f"No significant changes detected at depth {self.current_depth} - LLM stabilized. Exiting recursion.")
                else:
                    logger.info("No changes detected - LLM was not surprised. Exiting recursion.")
                return context

            logger.info("Changes detected - LLM was surprised. Continuing analysis.")
            
            # Save only the NEW observations that weren't in the original context
            await self._save_new_observations(context, revised_observations, message_id, session_id)
            
            # Pass the revised observations directly to the next iteration
            # The LLM has already curated the most relevant observations in revised_observations
            logger.debug("Passing revised observations directly to next recursive iteration")
            for level in REASONING_LEVELS:
                level_observations = revised_observations.get(level, [])
                logger.debug(f"Passing {level}: {len(level_observations)} revised observations to next iteration")

            # Recursively analyze with the revised observations from the LLM
            return await self.recursive_reason(revised_observations, history, new_turn, message_id, session_id, current_time)

        finally:
            # Decrement the recursion depth counter when we're done
            self.current_depth -= 1

    async def _save_new_observations(self, original_context: dict, revised_observations: dict, message_id: str, session_id: str | None = None):
        """Save only the observations that are new compared to the original context."""
        if not self.embedding_store:
            return
        
        # Use the utility function to find new observations
        new_observations_by_level = find_new_observations(original_context, revised_observations)
        
        for level, new_observations in new_observations_by_level.items():
            if new_observations:
                logger.debug(f"Saving {len(new_observations)} new {level} observations: {new_observations}")
                await self._save_structured_observations(
                    new_observations,
                    message_id=message_id,
                    level=level,
                    session_id=session_id
                )
            else:
                logger.debug(f"No new observations to save for {level} level")

    async def _save_structured_observations(self, observations: list, message_id: str, level: str, session_id: str | None = None):
        """Save observations with proper handling of structured data including premises."""
        if not self.embedding_store:
            return
            
        for observation in observations:
            if isinstance(observation, dict) and 'conclusion' in observation and 'premises' in observation:
                # This is a structured observation with premises
                conclusion = observation['conclusion']
                premises = observation.get('premises', [])
                
                # Save the conclusion with premises in metadata
                await self.embedding_store.save_observations(
                    [conclusion],  # Only save the conclusion as content
                    message_id=message_id,
                    level=level,
                    session_id=session_id,
                    premises=premises  # Pass premises directly
                )
                
                logger.debug(f"Saved structured observation: conclusion='{conclusion}' with {len(premises)} premises in metadata")
                        
            else:
                # Simple observation (string or dict without structure)
                observation_content = extract_observation_content(observation)
                
                # Save simple observations normally (no premises)
                await self.embedding_store.save_observations(
                    [observation_content],
                    message_id=message_id,
                    level=level,
                    session_id=session_id
                )



