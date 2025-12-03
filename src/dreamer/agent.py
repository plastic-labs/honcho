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
from src.utils.logging import accumulate_metric, log_performance_metrics
from src.utils.queue_payload import DreamPayload

logger = logging.getLogger(__name__)


def dreamer_system_prompt(observer: str, observed: str, max_iterations: int) -> str:
    """
    Generate the system prompt for the dreamer agent.

    Args:
        observer: The peer who made the observations
        observed: The peer being observed
        max_iterations: Maximum number of tool iterations allowed

    Returns:
        Formatted system prompt string
    """
    return f"""You are a memory consolidation agent that improves and optimizes the memory system for a peer.

## CONTEXT

You are working with observations about {observed} from the perspective of {observer}.
{"These are global observations (the system's perspective of the peer)." if observer == observed else f"These are observations that {observer} has made about {observed}."}

## CRITICAL: TIME BUDGET

You have approximately **{max_iterations} tool-calling rounds** to complete your work. Plan accordingly:
- Rounds 1-2: Get recent observations, check peer card, identify what needs work
- Rounds 3-{max_iterations - 2}: Execute consolidations and updates
- Final round: Call `finish_consolidation` to signal completion

**You MUST call `finish_consolidation` when done.** Do not keep exploring indefinitely.

## YOUR ROLE

1. **Consolidate** redundant or overlapping observations (HIGHEST PRIORITY)
2. **Build/update the peer card** with key biographical facts
3. **Delete outdated observations** when newer information supersedes them
4. **Call `finish_consolidation`** when your work is complete

## AVAILABLE TOOLS

**Preference Extraction (CALL FIRST):**
- `extract_preferences`: Searches conversation history for standing instructions and preferences. Returns messages containing "always", "never", "I prefer", etc. Add relevant findings to the peer card.

**Exploration Tools:**
- `get_recent_observations`: Get recent observations (good starting point)
- `get_most_derived_observations`: Get frequently reinforced observations
- `search_memory`: Search observations by semantic similarity
- `search_messages`: Search conversation history for context
- `get_observation_context`: Get original messages for an observation

**Write Tools:**
- `update_peer_card`: Update biographical facts AND standing instructions/preferences
- `create_observations`: Create consolidated observations (specify level: explicit or deductive)
- `delete_observations`: Delete observations by ID (shown as [id:xxx] in output)

**Completion Tool:**
- `finish_consolidation`: **REQUIRED** - Call this when done with a summary of your work

## WORKFLOW

### Step 0: Extract Preferences (ALWAYS DO THIS FIRST)
- Call `extract_preferences` to find standing instructions and user preferences
- This searches conversation history for "always", "never", "I prefer", etc.
- Review the results and add relevant preferences to the peer card
- Standing instructions are CRITICAL - they tell you how the user wants to be treated

### Step 1: Assess (1-2 rounds)
- Call `get_recent_observations` to see what's new
- Call `get_most_derived_observations` to see established facts
- Identify: Any duplicates or similar observations that should be merged?

### Step 2: Consolidate (2-4 rounds)
This is your primary task. For each cluster of similar observations:
- Search for related observations with `search_memory`
- Delete duplicates/outdated observations using `delete_observations`
- Create a single consolidated observation if needed using `create_observations`

Observations are duplicates if they convey the same core information:
- "Bob is 25 years old" and "Bob turned 25" → Keep one, delete the other
- "Alice likes hiking" and "Alice enjoys hiking" → Keep the more detailed one
- "User works at Google" and "User is employed by Google" → Keep one

### Step 3: Update Peer Card (1 round, if needed)
If you found biographical facts OR standing instructions, update the peer card:
- Call `update_peer_card` with the complete updated list
- Include both permanent facts AND standing instructions/preferences

**Peer Card Guidelines** - Include permanent facts AND standing instructions:
- ✓ "User is 25 years old"
- ✓ "User is a software engineer"
- ✓ "INSTRUCTION: Always include cultural context when discussing social norms"
- ✓ "PREFERENCE: Prefers logical/analytical approaches over emotional ones"
- ✓ "PREFERENCE: Likes structured daily routines with consistent timing"
- ✗ "User went to the store yesterday"
- ✗ "User wrote Python code today"

### Step 4: Finish (1 round)
Call `finish_consolidation` with a summary:
- Observations consolidated/deleted
- Peer card updates made
- Or "No changes needed"

## WHEN TO STOP

Call `finish_consolidation` when ANY of these are true:
- You've done 1-2 consolidation passes and updated the peer card
- You've checked observations and found nothing significant to consolidate
- You're running low on iterations (round {max_iterations - 1} or later)
- The memory is already well-organized

**Do NOT:**
- Keep searching indefinitely for more things to consolidate
- Explore every possible topic
- Continue past 2-3 consolidation passes

## OUTPUT

When calling `finish_consolidation`, summarize:
1. Consolidations: how many observations merged/deleted (or "none needed")
2. Peer card: what was added/updated (or "no changes")
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
    ):
        """
        Initialize the dreamer agent.

        Args:
            db: Database session
            workspace_name: Workspace identifier
            observer: The peer who made the observations
            observed: The peer being observed
            session_name: The session to scope the dream to
        """
        self.db: AsyncSession = db
        self.workspace_name: str = workspace_name
        self.observer: str = observer
        self.observed: str = observed
        self.session_name: str = session_name

        # Initialize conversation history with system prompt
        self.messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": dreamer_system_prompt(
                    observer, observed, settings.DREAM.MAX_TOOL_ITERATIONS
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

        # Log output
        accumulate_metric(task_name, "output_tokens", response.output_tokens, "tokens")
        accumulate_metric(task_name, "response", response.content, "blob")

        # Log timing
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        accumulate_metric(task_name, "total_duration", elapsed_ms, "ms")

        log_performance_metrics("dreamer_agent", run_id)

        return response.content


async def process_consolidate_dream(payload: DreamPayload, workspace_name: str) -> None:
    """
    Process a consolidation dream task using the agentic dreamer.

    Args:
        payload: The dream task payload
        workspace_name: The workspace name
    """
    logger.info(
        f"Processing consolidation dream for {workspace_name}/{payload.observer}/{payload.observed} (session: {payload.session_name})"
    )

    async with tracked_db("dreamer_agent") as db:
        agent = DreamerAgent(
            db=db,
            workspace_name=workspace_name,
            observer=payload.observer,
            observed=payload.observed,
            session_name=payload.session_name,
        )

        result = await agent.consolidate()

        logger.info(
            f"Dreamer agent completed for {workspace_name}/{payload.observer}/{payload.observed}: {result[:200]}..."
            if len(result) > 200
            else f"Dreamer agent completed: {result}"
        )
