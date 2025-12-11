"""
Agentic Dreamer implementation.

This agent consolidates and improves the representation by exploring observations
through a random walk approach - starting from recent or high-value observations
and searching for related content to consolidate.
"""

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.dependencies import tracked_db
from src.utils.agent_tools import DREAMER_TOOLS, create_tool_executor
from src.utils.clients import HonchoLLMCallResponse, honcho_llm_call
from src.utils.logging import (
    accumulate_metric,
    log_performance_metrics,
    log_token_usage_metrics,
)
from src.utils.queue_payload import DreamPayload

logger = logging.getLogger(__name__)


def dreamer_system_prompt(
    observer: str,
    observed: str,
    max_iterations: int,
    reasoning_focus: str | None = None,
) -> str:
    """
    Generate the system prompt for the dreamer agent.

    Args:
        observer: The peer who made the observations
        observed: The peer being observed
        max_iterations: Maximum number of tool iterations allowed
        reasoning_focus: Optional focus mode ('deduction', 'induction', 'consolidation')

    Returns:
        Formatted system prompt string
    """
    # Build focus-specific instructions
    focus_section = ""
    if reasoning_focus == "deduction":
        focus_section = f"""
## FOCUS: DEDUCTIVE REASONING + KNOWLEDGE UPDATES

This dream session is focused on **DEDUCTIVE reasoning** AND **knowledge update detection**.

### Part 1: Deductive Reasoning
Derive logical necessities from explicit facts.

**Deduction Strategy:**
- Look for unstated implications of explicit facts
- Combine multiple facts that together imply something new
- Apply domain knowledge (e.g., "works at Google as SWE" → "has CS background")

**Example Deduction:**
```json
{{
  "content": "{observed} completed high school",
  "level": "deductive",
  "premise_ids": ["abc123", "def456"],
  "premises": ["{observed} attended Stanford", "Stanford requires high school completion"]
}}
```

### Part 2: Knowledge Update Detection
Identify when facts have CHANGED over time and create deductive observations marking the update.

**What is a Knowledge Update?**
- "Deadline is April 25" → later → "Deadline moved to April 22" = UPDATE
- "Meeting on Tuesday" → later → "Meeting rescheduled to Thursday" = UPDATE

**Knowledge Update Strategy:**
- Search for changeable facts: deadlines, plans, quantities
- When you find the SAME fact with DIFFERENT values at different times, create a deductive observation linking both

**Example Knowledge Update:**
```json
{{
  "content": "{observed}'s deadline was updated from April 25 to April 22 (April 22 supersedes)",
  "level": "deductive",
  "premise_ids": ["old_obs_id", "new_obs_id"],
  "premises": ["Original deadline April 25", "Rescheduled to April 22"]
}}
```

**Key update patterns:** "changed to", "rescheduled", "updated", "now", "moved to", "actually", "instead"

**Priority:** Create 3-5 deductions and 1-2 knowledge updates if found. Don't over-search - be efficient.
"""
    elif reasoning_focus == "induction":
        focus_section = f"""
## FOCUS: INDUCTIVE REASONING + CONSOLIDATION

This dream session combines **INDUCTIVE reasoning** AND **consolidation/cleanup**.

### Part 1: Inductive Reasoning (Primary)
Identify patterns and generalizations across observations.

**Induction Strategy:**
- Identify repeated behaviors or stated preferences
- Aggregate temporal sequences (e.g., museum visits over time)
- Create inductive observations with `source_ids`, `pattern_type`, `confidence`

**Example Induction:**
```json
{{
  "content": "{observed} prefers structured approaches to problem-solving",
  "level": "inductive",
  "source_ids": ["abc123", "def456", "ghi789"],
  "sources": ["planned project timeline", "uses checklists", "prefers outlines"],
  "pattern_type": "preference",
  "confidence": "high"
}}
```

### Part 2: Consolidation (Secondary)
Clean up duplicates and update the peer card.

**Consolidation Actions:**
- Delete EXACT duplicates only (same fact stated identically)
- Update peer card with key biographical facts AND insights
- **NEVER delete contradictory observations** - they are valuable signals

**Priority:** Create 3-5 inductive observations, delete obvious duplicates, update peer card once. Be efficient.
"""
    elif reasoning_focus == "consolidation":
        focus_section = f"""
## FOCUS: CONSOLIDATION

This dream session is focused on **CONSOLIDATION**. Your primary goal is cleanup and organization, not creating new observations.

**Priority Actions:**
1. Find redundant or duplicate observations
2. Delete exact duplicates (use `delete_observations`)
3. Update the peer card with key biographical facts
4. **Minimize new observation creation** - focus on cleanup

**Consolidation Strategy:**
- Use `search_memory` to find similar observations
- Delete observations that are fully subsumed by others
- **NEVER delete contradictory observations** - they are valuable signals
- Update peer card with permanent biographical facts
"""

    return f"""You are a reasoning engine analyzing observations about {observed} (from {observer}'s perspective).
{focus_section}

## OBSERVATION LEVELS

- **EXPLICIT**: Direct facts from messages (e.g., "{observed} works at Google")
- **DEDUCTIVE**: Logical necessities. REQUIRES `premise_ids` + `premises`. (e.g., "has CS background" from SWE job)
- **INDUCTIVE**: Patterns from multiple observations. REQUIRES `source_ids`, `sources`, `pattern_type` (preference/behavior/personality/tendency), `confidence` (high/medium/low)

## TOOLS

**Read:** `get_recent_observations`, `get_most_derived_observations`, `search_memory`, `get_peer_card`
**Write:** `create_observations`, `update_peer_card`, `delete_observations`
**Done:** `finish_consolidation` (REQUIRED when done)

## WORKFLOW

1. **GATHER** (2-3 calls): `get_recent_observations`, `get_most_derived_observations`, 1-2 targeted `search_memory`
2. **REASON** (use thinking): Identify deductions and patterns
3. **CREATE** (2-4 calls): Create high-value observations per focus instructions
4. **CLEANUP** (1-2 calls): Delete exact duplicates, update peer card once
5. **FINISH**: Call `finish_consolidation`

## RULES

- **Be efficient** - {max_iterations} tool calls max. Don't over-search.
- **Quality > quantity** - 3-5 high-quality observations beats 20 redundant ones
- **NEVER delete contradictions** - Both versions are valuable signals
- **Peer card** - Include: facts, "INSTRUCTION: ...", "PREFERENCE: ...", "TRAIT: ..."

**You MUST call `finish_consolidation` when done.**
"""


class DreamerAgent:
    """
    An agentic dreamer that consolidates observations through exploration.

    Uses a random walk approach - starting from recent/important observations
    and using semantic search to find related content to consolidate.
    """

    def __init__(
        self,
        db: AsyncSession,
        workspace_name: str,
        observer: str,
        observed: str,
        session_name: str,
        reasoning_focus: str | None = None,
    ):
        """
        Initialize the dreamer agent.

        Args:
            db: Database session
            workspace_name: Workspace identifier
            observer: The peer who made the observations
            observed: The peer being observed
            session_name: The session to scope the dream to
            reasoning_focus: Optional focus mode ('deduction', 'induction', 'consolidation')
        """
        self.db: AsyncSession = db
        self.workspace_name: str = workspace_name
        self.observer: str = observer
        self.observed: str = observed
        self.session_name: str = session_name
        self.reasoning_focus: str | None = reasoning_focus

        # Initialize conversation history with system prompt
        self.messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": dreamer_system_prompt(
                    observer,
                    observed,
                    settings.DREAM.MAX_TOOL_ITERATIONS,
                    reasoning_focus=reasoning_focus,
                ),
            }
        ]

    async def consolidate(self) -> str:
        """
        Run the consolidation process using random walk exploration.

        The agent will:
        1. Start with recent observations
        2. Search for related observations
        3. Consolidate redundancies it finds
        4. Explore other areas of the memory

        Returns:
            Summary of what was consolidated
        """
        # Generate unique ID for this run
        run_id = str(uuid.uuid4())[:8]
        task_name = f"dreamer_agent_{run_id}"
        start_time = time.perf_counter()

        # Log input context
        accumulate_metric(
            task_name,
            "context",
            (
                f"workspace: {self.workspace_name}\n"
                f"observer: {self.observer}\n"
                f"observed: {self.observed}"
            ),
            "blob",
        )

        self.messages.append(
            {
                "role": "user",
                "content": "Review the observations about this peer. Update the peer card if needed, consolidate any clear redundancies, and call finish_consolidation when done.",
            }
        )

        tool_executor: Callable[[str, dict[str, Any]], Any] = create_tool_executor(
            db=self.db,
            workspace_name=self.workspace_name,
            observer=self.observer,
            observed=self.observed,
            session_name=self.session_name,
            include_observation_ids=True,
            history_token_limit=settings.DREAM.HISTORY_TOKEN_LIMIT,
        )

        response: HonchoLLMCallResponse[str] = await honcho_llm_call(
            llm_settings=settings.DREAM,
            prompt="",
            max_tokens=settings.DREAM.MAX_OUTPUT_TOKENS,
            thinking_budget_tokens=settings.DREAM.THINKING_BUDGET_TOKENS,
            tools=DREAMER_TOOLS,
            tool_choice=None,
            tool_executor=tool_executor,
            max_tool_iterations=settings.DREAM.MAX_TOOL_ITERATIONS,
            messages=self.messages,
            track_name="Dreamer Agent",
        )

        # Log tool calls made
        tool_calls_summary = ", ".join(
            tc.get("tool_name", "unknown") for tc in response.tool_calls_made
        )
        accumulate_metric(
            task_name, "tool_calls", len(response.tool_calls_made), "count"
        )
        accumulate_metric(
            task_name, "tools_used", tool_calls_summary or "(none)", "blob"
        )

        # Log token usage with cache awareness
        log_token_usage_metrics(
            task_name,
            response.input_tokens,
            response.output_tokens,
            response.cache_read_input_tokens,
            response.cache_creation_input_tokens,
        )
        accumulate_metric(task_name, "response", response.content, "blob")

        # Log timing
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        accumulate_metric(task_name, "total_duration", elapsed_ms, "ms")

        log_performance_metrics("dreamer_agent", run_id)

        return response.content


async def process_consolidate_dream(
    payload: DreamPayload,
    workspace_name: str,
    *,
    use_orchestrated: bool = True,
) -> None:
    """
    Process a consolidation dream task.

    Args:
        payload: The dream task payload
        workspace_name: The workspace name
        use_orchestrated: If True, use the new orchestrated specialist system.
                          If False, use the legacy agentic dreamer.
    """
    logger.info(
        f"Processing consolidation dream for {workspace_name}/{payload.observer}/{payload.observed} "
        f"(session: {payload.session_name}, orchestrated: {use_orchestrated})"
    )

    if use_orchestrated:
        # Use new orchestrated specialist architecture
        from src.dreamer.orchestrator import process_orchestrated_dream

        async with tracked_db("dream_orchestrator") as db:
            result = await process_orchestrated_dream(
                db=db,
                workspace_name=workspace_name,
                observer=payload.observer,
                observed=payload.observed,
                session_name=payload.session_name,
            )
            logger.info(f"Orchestrated dream result: {result[:500]}")
    else:
        # Use legacy agentic dreamer
        reasoning_focus: str | None = None
        if payload.reasoning_focus is not None:
            reasoning_focus = payload.reasoning_focus.value

        async with tracked_db("dreamer_agent") as db:
            agent = DreamerAgent(
                db=db,
                workspace_name=workspace_name,
                observer=payload.observer,
                observed=payload.observed,
                session_name=payload.session_name,
                reasoning_focus=reasoning_focus,
            )
            await agent.consolidate()
