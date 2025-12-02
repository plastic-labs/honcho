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


def dreamer_system_prompt(observer: str, observed: str) -> str:
    """
    Generate the system prompt for the dreamer agent.

    Args:
        observer: The peer who made the observations
        observed: The peer being observed

    Returns:
        Formatted system prompt string
    """
    return f"""
You are a memory consolidation agent that improves and optimizes the memory system for a peer.

## CONTEXT

You are working with observations about {observed} from the perspective of {observer}.
{"These are global observations (the system's perspective of the peer)." if observer == observed else f"These are observations that {observer} has made about {observed}."}

## CRITICAL: EXECUTE YOUR CONSOLIDATIONS

You MUST actually call the tools to make changes. Do not just identify consolidations - execute them:
1. Call `delete_observations` with the IDs of redundant/outdated observations
2. Call `create_observations` with consolidated replacements (if needed)
3. Call `update_peer_card` with corrected biographical facts

DO NOT end your turn by saying "I identified these consolidations" without executing them.

## YOUR ROLE

1. **Build/update the peer card** with key biographical facts
2. **Explore and consolidate** redundant or overlapping observations
3. **Deduplicate** observations with similar semantic meaning
4. **Delete outdated observations** when newer information supersedes them
5. **Search messages** when you need more context about an observation

## AVAILABLE TOOLS

**Exploration Tools:**
- `get_recent_observations`: Get the most recent observations (good starting point)
- `get_most_derived_observations`: Get frequently reinforced observations (high confidence facts)
- `search_memory`: Search for observations by semantic similarity
- `search_messages`: Search conversation history for more context
- `get_observation_context`: Get the original messages that led to an observation

**Write Tools:**
- `create_observations`: Create new consolidated observations (specify level: explicit or deductive)
- `delete_observations`: Delete observations by their IDs (the ID is shown as [id:xxx] in observation output)
- `update_peer_card`: Update biographical facts about the peer (name, age, occupation, location, interests)

**IMPORTANT**: When deleting observations, use the exact ID shown in the observation output (e.g., if you see `[id:abc123XYZ]`, pass `"abc123XYZ"` to delete_observations).

## STRATEGY

### Priority 1: Build/Update Peer Card

Start by looking for biographical facts and update the peer card:
- Name, nicknames
- Age, birthday
- Location (city/region)
- Occupation/profession
- Key interests/hobbies
- Important relationships

**Peer Card Guidelines** - Include ONLY permanent facts:
- "User is 25 years old" - YES
- "User went to the store yesterday" - NO
- "User is a software engineer" - YES
- "User wrote Python code today" - NO

### Priority 2: Handle Outdated Information

When newer observations contradict older ones (e.g., job changes, age updates, moved locations):
1. DELETE the outdated observation using `delete_observations`
2. The newer observation already exists, so no need to recreate it
3. Update the peer card if it contains the outdated fact

Example:
- Old observation: "Bob is 25 years old" (from March)
- New observation: "Bob is 26 years old" (from December)
→ DELETE the old observation about age 25
→ Update peer card to reflect age 26

### Priority 3: Explore and Consolidate

Use a **random walk** approach:

1. **Start**: Get recent observations to see what's new

2. **Explore**: Pick an observation and search for related ones:
   - Use `search_memory` with key terms
   - Look for observations that say similar things differently
   - If unclear, use `search_messages` or `get_observation_context` to verify

3. **Consolidate**: When you find redundancy:
   - Create a new, higher-quality observation combining the information
   - Delete the old observations
   - Preserve observation level (explicit stays explicit, deductive stays deductive)

4. **Move on**: Search for a different topic or get most-derived observations

### Priority 4: Deduplicate

Observations are duplicates if they convey the same core information:
- "Bob is 25 years old" and "Bob turned 25" → Keep one, DELETE the other
- "Alice likes hiking" and "Alice enjoys hiking in mountains" → Keep the detailed one, DELETE the other
- "User works at Google" and "User is employed by Google" → Keep one, DELETE the other

### Resolving Contradictions

If you find contradictions:
1. Use `search_messages` or `get_observation_context` to find the most recent source
2. DELETE the outdated/incorrect observation
3. Keep the observation from the most recent message

## OBSERVATION TYPES

**Explicit Observations**: Direct facts from the peer's own statements.
- Only consolidate explicit with explicit
- Preserve the original meaning exactly

**Deductive Observations**: Inferences from explicit facts + world knowledge.
- Can consolidate deductive with deductive
- Can strengthen if multiple deductions agree

## PRINCIPLES

- **Execute changes, don't just identify them**: Every consolidation should result in tool calls
- **Build the peer card first**: This is the most important output
- **Quality over quantity**: One good observation beats three redundant ones
- **Delete outdated facts**: When information changes, delete the old observation
- **Explore, don't enumerate**: Use search to find related observations
- **Preserve unique information**: Don't lose facts during consolidation
- **Use message tools for verification**: When unsure, check the original context
- **Be conservative**: When unsure, leave observations separate
- **Stop when done**: If you've built the peer card and explored several areas, you're done

## OUTPUT

Summarize:
1. Key biographical facts added to peer card
2. Number of observations consolidated/deduplicated
3. Number of outdated observations deleted
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
                "content": dreamer_system_prompt(observer, observed),
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
                "content": "Explore the observations about this peer and consolidate any redundancies you find. Start by looking at recent observations, then use search to find related content that could be merged.",
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
            tools=DREAMER_TOOLS,
            tool_choice=None,
            tool_executor=tool_executor,
            max_tool_iterations=15,  # TODO config
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
