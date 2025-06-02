import json
import logging
import time
from typing import Any

import sentry_sdk
from langfuse.decorators import langfuse_context, observe
from sentry_sdk.ai.monitoring import ai_track

from src.utils.model_client import ModelClient, ModelProvider
from src.agent import parse_xml_content

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

# Constants for reasoning levels and their labels
REASONING_LEVELS = ['abductive', 'inductive', 'deductive']
LEVEL_LABELS = {
    'abductive': 'Abductive (High-level hypotheses about personality, motivations, identity)',
    'inductive': 'Inductive (Behavioral patterns and tendencies observed over time)', 
    'deductive': 'Deductive (Specific facts and explicit information)'
}

def _extract_fact_content(fact) -> str:
    """Extract content string from a fact (dict or string)."""
    if isinstance(fact, dict):
        return fact.get('content', str(fact))
    return str(fact)

def _ensure_context_structure(context: dict) -> dict:
    """Ensure context has all reasoning levels with empty lists as defaults."""
    return {level: context.get(level, []) for level in REASONING_LEVELS}

def format_new_turn_with_timestamp(new_turn: str, current_time: str) -> str:
    """
    Format new turn message with optional timestamp.
    
    Args:
        new_turn: The message content
        current_time: Timestamp string or "unknown"
        
    Returns:
        Formatted string like "2023-05-08 13:56:00 user: hello" or "user: hello"
    """
    if current_time and current_time != "unknown":
        return f"{current_time} user: {new_turn}"
    else:
        return f"user: {new_turn}"


def format_context_for_prompt(context: dict) -> str:
    """
    Format context dictionary into a clean, readable string for LLM prompts.
    
    Args:
        context: Dictionary with reasoning levels as keys and fact lists as values
                Facts can be strings or dicts - will be normalized
    
    Returns:
        Formatted string with clear sections and bullet points including temporal metadata
    """
    if not context:
        return "No context available."
    
    formatted_sections = []
    
    # Process each level in a consistent order
    for level in REASONING_LEVELS:
        facts = context.get(level, [])
        if not facts:
            continue
            
        label = LEVEL_LABELS.get(level, level.title())
        formatted_sections.append(f"{label}:")
        
        # Format facts with temporal metadata when available
        for fact in facts:
            fact_content = _extract_fact_content(fact)
            
            # Check if fact is a dict with temporal metadata
            if isinstance(fact, dict) and 'access_count' in fact:
                temporal_info = _format_temporal_metadata(fact)
                formatted_sections.append(f"  • {fact_content}{temporal_info}")
            else:
                formatted_sections.append(f"  • {fact_content}")
        
        formatted_sections.append("")  # Blank line between sections
    
    # Remove trailing blank line if exists
    if formatted_sections and formatted_sections[-1] == "":
        formatted_sections.pop()
    
    return "\n".join(formatted_sections) if formatted_sections else "No relevant context available."


def _format_temporal_metadata(fact: dict) -> str:
    """
    Format temporal metadata for display in reasoning prompts.
    Shows both fact genesis (when derived) and ongoing relevance (access patterns).
    
    Args:
        fact: Fact dictionary containing temporal metadata
        
    Returns:
        Formatted temporal context string showing inception and access patterns
    """
    temporal_parts = []
    
    # Fact inception - when originally derived
    created_at = fact.get('created_at')
    if created_at:
        if hasattr(created_at, 'isoformat'):
            temporal_parts.append(f"derived {created_at.isoformat()}")
        else:
            temporal_parts.append(f"derived {created_at}")
    
    # Access frequency and session spread
    access_count = fact.get('access_count', 0)
    accessed_sessions = fact.get('accessed_sessions', [])
    
    if access_count > 0:
        if accessed_sessions:
            session_count = len(accessed_sessions)
            if session_count > 1:
                temporal_parts.append(f"accessed {access_count}x across {session_count} sessions")
            else:
                temporal_parts.append(f"accessed {access_count}x in 1 session")
        else:
            temporal_parts.append(f"accessed {access_count}x")
    
    # Last accessed timestamp
    last_accessed = fact.get('last_accessed')
    if last_accessed:
        temporal_parts.append(f"last accessed {last_accessed}")
    
    if temporal_parts:
        return f" [{', '.join(temporal_parts)}]"
    else:
        return ""

PROVIDER = ModelProvider.ANTHROPIC
MODEL = "claude-3-7-sonnet-20250219"

REASONING_SYSTEM_PROMPT = """
You are an expert at critically analyzing user understanding while maintaining appropriate stability. Your goal is to IMPROVE understanding through careful analysis, not to change things unnecessarily.

**REASONING LEVELS:**
- **Deductive**: Specific, verifiable facts explicitly stated or directly observable
- **Inductive**: Behavioral patterns and tendencies supported by multiple observations
- **Abductive**: High-level hypotheses about personality, motivations, and identity

**CONSERVATIVE ANALYSIS PRINCIPLE:**
Be thoughtfully critical but not overly eager to change. The current understanding may already be accurate and well-founded. Only make changes when there is COMPELLING evidence that clearly warrants revision.

**FLUID MOVEMENT BETWEEN LEVELS:**
- **Abductive → Inductive**: Only when hypothesis has STRONG supporting evidence across multiple instances
- **Inductive → Deductive**: Only when pattern becomes explicitly confirmed or directly stated
- **Deductive + Inductive → New Abductive**: Only when combination clearly reveals new personality insights

**HIGH EVIDENCE STANDARDS:**
1. **Falsification**: Require CLEAR contradictory evidence, not just alternative interpretations
2. **Promotion**: Need SUBSTANTIAL evidence across multiple contexts before promoting insights
3. **Synthesis**: Only synthesize when the combination clearly points to something genuinely new
4. **Revision**: Only revise when new evidence significantly changes the understanding
5. **Stability**: When in doubt, maintain the current understanding

**TEMPORAL CONTEXT AWARENESS:**
Facts may include temporal metadata showing access patterns:
- **High access count across multiple sessions**: Indicates core, stable traits
- **Recent access**: Shows immediately relevant or evolving patterns  
- **Cross-session consistency**: Suggests reliable behavioral patterns
- **Low/single access**: May indicate situational or outdated information

**QUALITY OVER CHANGE:**
It's better to maintain accurate existing insights than to make unnecessary modifications. Change should improve understanding, not just demonstrate analysis.
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
Think through these critical analysis steps, but ONLY make changes when there is STRONG, COMPELLING evidence:

1. **CHALLENGE** existing hypotheses - but only reject/revise them if there's STRONG contradictory evidence
2. **PROMOTE** insights only when there's SUBSTANTIAL supporting evidence across multiple instances
3. **SYNTHESIZE** new insights only when the combination of facts/patterns clearly points to something new
4. **REVISE** beliefs only when the new evidence significantly changes the understanding
5. **FALSIFY** only when evidence clearly contradicts existing beliefs

**STABILITY CRITERIA:**
- Prefer STABILITY over change - the current understanding may already be accurate
- Make changes only when evidence is COMPELLING, not just suggestive
- Minor adjustments or weak evidence should NOT trigger changes
- If you're uncertain whether a change is warranted, DON'T make it

<thinking>
For each reasoning level, ask yourself:
- Is there STRONG evidence that contradicts existing beliefs? (Not just weak hints)
- Do I have SUBSTANTIAL evidence to promote hypotheses to higher certainty levels?
- Can I clearly synthesize NEW insights from existing facts and patterns?
- Would keeping things the same be just as reasonable as changing them?

Remember: STABILITY is valuable. Only change what needs changing based on strong evidence.
</thinking>

Format your response as follows:
<response>
{{
    "deductive": ["revised and final list of specific facts"],
    "inductive": ["revised and final list of behavioral patterns"], 
    "abductive": ["revised and final list of identity hypotheses"]
}}
</response>

If the evidence isn't compelling enough to warrant changes, return the SAME lists you received. Stability is preferable to unnecessary churn.
"""


class SurpriseReasoner:
    def __init__(self, embedding_store=None):
        self.max_recursion_depth = 3  # Reduced from 5 to prevent excessive recursion
        self.current_depth = 0
        self.embedding_store = embedding_store
        # Minimum percentage of facts that must change to continue recursion
        self.significance_threshold = 0.2  # 20% of facts must change
        # Trace capture for analysis
        self.trace = None
    
    def _init_trace(self, message_id: str, session_id: str, user_message: str, initial_context: dict):
        """Initialize trace capture for this reasoning session."""
        self.trace = {
            "message_id": message_id,
            "session_id": session_id or "unknown",
            "user_message": user_message,
            "timestamp": time.time(),
            "initial_context": self._format_context_for_trace(initial_context),
            "reasoning_iterations": [],
            "final_facts": {},
            "saved_documents": {},
            "summary": {}
        }
    
    def _format_context_for_trace(self, context: dict) -> dict:
        """Format context with document IDs and similarity scores for trace."""
        formatted = {}
        for level in REASONING_LEVELS:
            facts = context.get(level, [])
            formatted[level] = []
            for fact in facts:
                if isinstance(fact, dict):
                    formatted[level].append({
                        "content": _extract_fact_content(fact),
                        "document_id": fact.get("document_id", "unknown"),
                        "similarity_score": fact.get("similarity_score", 0.0)
                    })
                else:
                    formatted[level].append({
                        "content": str(fact),
                        "document_id": "unknown",
                        "similarity_score": 0.0
                    })
        return formatted
    
    def _capture_iteration(self, depth: int, input_context: dict, raw_response: str, 
                          output_facts: dict, changes_detected: dict, significance_score: float, 
                          threshold_met: bool, continue_reasoning: bool, duration_ms: int):
        """Capture data for a single reasoning iteration."""
        if not self.trace:
            return
            
        # Extract reasoning trace from <thinking> tags
        reasoning_trace = parse_xml_content(raw_response, "thinking") or "No thinking content found"
        
        iteration = {
            "depth": depth,
            "input_context": self._format_simple_context_for_trace(input_context),
            "reasoning_trace": reasoning_trace,
            "output_facts": output_facts,
            "changes_detected": changes_detected,
            "significance_score": significance_score,
            "threshold_met": threshold_met,
            "continue_reasoning": continue_reasoning,
            "iteration_duration_ms": duration_ms
        }
        
        self.trace["reasoning_iterations"].append(iteration)
    
    def _format_simple_context_for_trace(self, context: dict) -> dict:
        """Format context with document IDs for trace (no similarity scores)."""
        formatted = {}
        for level in REASONING_LEVELS:
            facts = context.get(level, [])
            formatted[level] = []
            for fact in facts:
                if isinstance(fact, dict):
                    formatted[level].append({
                        "content": _extract_fact_content(fact),
                        "document_id": fact.get("document_id", "unknown")
                    })
                else:
                    formatted[level].append({
                        "content": str(fact),
                        "document_id": "unknown"
                    })
        return formatted
    
    def _finalize_trace(self, final_facts: dict, convergence_reason: str, total_duration_ms: int):
        """Finalize the trace with summary information."""
        if not self.trace:
            return
            
        self.trace["final_facts"] = final_facts
        self.trace["summary"] = {
            "total_iterations": len(self.trace["reasoning_iterations"]),
            "max_depth_reached": max([it["depth"] for it in self.trace["reasoning_iterations"]], default=0),
            "total_facts_added": sum([
                len(it["changes_detected"].get(level, {}).get("added", []))
                for it in self.trace["reasoning_iterations"]
                for level in REASONING_LEVELS
            ]),
            "total_facts_modified": sum([
                len(it["changes_detected"].get(level, {}).get("modified", []))
                for it in self.trace["reasoning_iterations"]
                for level in REASONING_LEVELS
            ]),
            "total_duration_ms": total_duration_ms,
            "convergence_reason": convergence_reason
        }

    def _adjust_context_for_depth(self):
        """
        Adjust the embedding store's fact retrieval counts based on recursion depth
        to progressively focus on the most relevant facts.
        """
        if not self.embedding_store:
            return
            
        # Progressively reduce context size as depth increases
        depth_factor = max(0.4, 1.0 - (self.current_depth * 0.15))
        
        # Calculate adjusted counts (minimum of 1 for each level)
        adjusted_abductive = max(1, int(2 * depth_factor))
        adjusted_inductive = max(2, int(4 * depth_factor))  
        adjusted_deductive = max(3, int(6 * depth_factor))
        
        logger.debug(f"Depth {self.current_depth}: Adjusting context to {adjusted_abductive}/{adjusted_inductive}/{adjusted_deductive} facts")
        
        self.embedding_store.set_fact_counts(
            abductive=adjusted_abductive,
            inductive=adjusted_inductive,
            deductive=adjusted_deductive
        )

    @observe()
    @ai_track("Derive New Insights")
    async def derive_new_insights(self, context, history, new_turn, message_id: str, current_time: str) -> tuple[dict[str, list[str]], str]:
        """
        Critically analyzes and revises understanding, returning final fact lists at each level.
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

        # Parse the response to extract final fact lists
        try:
            # Find and parse the JSON response between <response> tags
            try:
                json_str = response[response.find("<response>")+10:response.find("</response>")].strip()
                revised_facts = json.loads(json_str)
            except (ValueError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse critical analysis response: {e}")
                logger.error(f"Response: {response}")
                return _ensure_context_structure({}), response
            
            # Ensure we have the expected structure
            final_facts = _ensure_context_structure(revised_facts)
            
            return final_facts, response
            
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse critical analysis response: {e}")
            return _ensure_context_structure({}), response

    async def recursive_reason_with_trace(self, context: dict, history: str, new_turn: str, message_id: str, session_id: str | None = None, current_time: str | None = None) -> tuple[dict, dict]:
        """
        Main entry point for recursive reasoning with full trace capture.
        Returns final facts and complete trace.
        """
        start_time = time.time()
        
        # Initialize trace capture
        self._init_trace(message_id, session_id, new_turn, context)
        
        # Reset depth counter
        self.current_depth = 0
        
        try:
            # Run recursive reasoning
            final_facts = await self.recursive_reason(context, history, new_turn, message_id, session_id, current_time)
            
            # Finalize trace
            total_duration_ms = int((time.time() - start_time) * 1000)
            convergence_reason = "completed"
            self._finalize_trace(final_facts, convergence_reason, total_duration_ms)
            
            return final_facts, self.trace
            
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
            
            # Perform critical analysis to get revised fact lists
            revised_facts, raw_response = await self.derive_new_insights(context, history, new_turn, message_id, current_time)

            logger.debug(f"Critical analysis result: {revised_facts}")
            
            # Compare input context with output to detect changes (surprise)
            # Apply depth-based conservatism - require more significant changes at deeper levels
            effective_threshold = self.significance_threshold * (1 + 0.5 * self.current_depth)
            if self.current_depth > 0:
                logger.info(f"Depth {self.current_depth}: Using higher significance threshold: {effective_threshold:.1%}")
                
            # Temporarily adjust threshold for this depth
            original_threshold = self.significance_threshold
            self.significance_threshold = effective_threshold
            has_changes, changes_detected, significance_score = self._detect_changes_with_details(context, revised_facts)
            self.significance_threshold = original_threshold  # Restore original
            
            # Calculate iteration duration
            iteration_duration_ms = int((time.time() - iteration_start) * 1000)
            
            # Capture this iteration in trace
            self._capture_iteration(
                depth=self.current_depth,
                input_context=context,
                raw_response=raw_response,
                output_facts=revised_facts,
                changes_detected=changes_detected,
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
            
            # Save only the NEW facts that weren't in the original context
            await self._save_new_facts(context, revised_facts, message_id, session_id)
            
            # Pass the revised facts directly to the next iteration
            # The LLM has already curated the most relevant facts in revised_facts
            logger.debug("Passing revised facts directly to next recursive iteration")
            for level in REASONING_LEVELS:
                level_facts = revised_facts.get(level, [])
                logger.debug(f"Passing {level}: {len(level_facts)} revised facts to next iteration")

            # Recursively analyze with the revised facts from the LLM
            return await self.recursive_reason(revised_facts, history, new_turn, message_id, session_id, current_time)

        finally:
            # Decrement the recursion depth counter when we're done
            self.current_depth -= 1

    def _detect_changes(self, original_context: dict, revised_facts: dict) -> bool:
        """
        Compare original context with revised facts to detect if LLM made significant changes.
        Returns True only if changes are substantial enough to warrant continued recursion.
        """
        total_original_facts = 0
        total_changed_facts = 0
        
        for level in REASONING_LEVELS:
            # Get normalized fact sets for comparison
            original_facts = self._normalize_facts_for_comparison(original_context.get(level, []))
            revised_facts_list = self._normalize_facts_for_comparison(revised_facts.get(level, []))
            
            level_original_count = len(original_facts)
            total_original_facts += level_original_count
            
            # Count significant changes (additions, removals, modifications)
            added_facts = revised_facts_list - original_facts
            removed_facts = original_facts - revised_facts_list
            level_changes = len(added_facts) + len(removed_facts)
            total_changed_facts += level_changes
            
            if level_changes > 0:
                logger.debug(f"Changes in {level} level:")
                logger.debug(f"  Original: {level_original_count} facts")
                logger.debug(f"  Added: {len(added_facts)} facts")
                logger.debug(f"  Removed: {len(removed_facts)} facts")
                if added_facts:
                    logger.debug(f"  New facts: {list(added_facts)[:3]}...")  # Show first 3
                if removed_facts:
                    logger.debug(f"  Removed facts: {list(removed_facts)[:3]}...")  # Show first 3
        
        # Calculate change percentage
        if total_original_facts == 0:
            change_percentage = 1.0 if total_changed_facts > 0 else 0.0
        else:
            change_percentage = total_changed_facts / total_original_facts
        
        # Check if changes meet significance threshold
        is_significant = change_percentage >= self.significance_threshold
        
        logger.debug(f"Change analysis: {total_changed_facts}/{total_original_facts} facts changed ({change_percentage:.1%})")
        logger.debug(f"Significance threshold: {self.significance_threshold:.1%} - {'MET' if is_significant else 'NOT MET'}")
        
        if not is_significant and total_changed_facts > 0:
            logger.debug("Changes detected but not significant enough to continue recursion")
        
        return is_significant
    
    def _detect_changes_with_details(self, original_context: dict, revised_facts: dict) -> tuple[bool, dict, float]:
        """
        Detailed version of _detect_changes that returns change details for trace capture.
        Returns (has_changes, changes_detected, significance_score)
        """
        total_original_facts = 0
        total_changed_facts = 0
        changes_detected = {}
        
        for level in REASONING_LEVELS:
            # Get normalized fact sets for comparison
            original_facts = self._normalize_facts_for_comparison(original_context.get(level, []))
            revised_facts_list = self._normalize_facts_for_comparison(revised_facts.get(level, []))
            
            level_original_count = len(original_facts)
            total_original_facts += level_original_count
            
            # Count significant changes (additions, removals, modifications)
            added_facts = revised_facts_list - original_facts
            removed_facts = original_facts - revised_facts_list
            level_changes = len(added_facts) + len(removed_facts)
            total_changed_facts += level_changes
            
            # Format changes for trace
            changes_detected[level] = {
                "added": list(added_facts),
                "removed": [{"content": fact, "document_id": "unknown"} for fact in removed_facts],
                "modified": []  # Could enhance this to detect modifications vs additions/removals
            }
            
            if level_changes > 0:
                logger.debug(f"Changes in {level} level:")
                logger.debug(f"  Original: {level_original_count} facts")
                logger.debug(f"  Added: {len(added_facts)} facts")
                logger.debug(f"  Removed: {len(removed_facts)} facts")
                if added_facts:
                    logger.debug(f"  New facts: {list(added_facts)[:3]}...")  # Show first 3
                if removed_facts:
                    logger.debug(f"  Removed facts: {list(removed_facts)[:3]}...")  # Show first 3
        
        # Calculate change percentage
        if total_original_facts == 0:
            change_percentage = 1.0 if total_changed_facts > 0 else 0.0
        else:
            change_percentage = total_changed_facts / total_original_facts
        
        # Check if changes meet significance threshold
        is_significant = change_percentage >= self.significance_threshold
        
        logger.debug(f"Change analysis: {total_changed_facts}/{total_original_facts} facts changed ({change_percentage:.1%})")
        logger.debug(f"Significance threshold: {self.significance_threshold:.1%} - {'MET' if is_significant else 'NOT MET'}")
        
        if not is_significant and total_changed_facts > 0:
            logger.debug("Changes detected but not significant enough to continue recursion")
        
        return is_significant, changes_detected, change_percentage
    
    def _normalize_facts_for_comparison(self, facts: list) -> set:
        """Convert facts to normalized strings for comparison."""
        normalized = set()
        for fact in facts:
            fact_content = _extract_fact_content(fact)
            normalized.add(fact_content.strip().lower())
        return normalized
    
    async def _save_new_facts(self, original_context: dict, revised_facts: dict, message_id: str, session_id: str | None = None):
        """Save only the facts that are new compared to the original context."""
        if not self.embedding_store:
            return
            
        for level in REASONING_LEVELS:
            original_facts = self._normalize_facts_for_comparison(original_context.get(level, []))
            revised_list = revised_facts.get(level, [])
            
            # Find genuinely new facts
            new_facts = []
            for fact in revised_list:
                normalized_fact = _extract_fact_content(fact).strip().lower()
                if normalized_fact not in original_facts:
                    new_facts.append(fact)
            
            # Save new facts to embedding store
            if new_facts:
                logger.debug(f"Saving {len(new_facts)} new {level} facts: {new_facts}")
                await self.embedding_store.save_facts(
                    new_facts,
                    message_id=message_id,
                    level=level,
                    session_id=session_id
                )
            else:
                logger.debug(f"No new facts to save for {level} level")



