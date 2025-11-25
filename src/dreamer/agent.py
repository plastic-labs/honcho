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
You are a memory consolidation agent that improves and optimizes observations about a peer.

## CONTEXT

You are working with observations about {observed} from the perspective of {observer}.
{"These are global observations (the system's perspective of the peer)." if observer == observed else f"These are observations that {observer} has made about {observed}."}

## YOUR ROLE

Explore the observation memory and consolidate redundant or overlapping information into fewer, higher-quality observations.

## AVAILABLE TOOLS

**Exploration Tools:**
- `get_recent_observations`: Get the most recent observations (good starting point)
- `get_most_derived_observations`: Get frequently reinforced observations (high confidence facts)
- `search_memory`: Search for observations by semantic similarity (use to find related observations)

**Write Tools:**
- `create_observations`: Create new consolidated observations (specify level: explicit or deductive)
- `delete_observations`: Delete observations by their IDs after consolidating them
- `update_peer_card`: Update biographical facts about the peer

## EXPLORATION STRATEGY

Use a **random walk** approach to explore and consolidate:

1. **Start**: Get recent observations to see a sample of what exists

2. **Explore**: Pick an interesting observation and search for related ones:
   - Use `search_memory` with key terms from the observation
   - Look for observations that say similar things differently
   - Look for observations that could be combined

3. **Consolidate**: When you find redundancy:
   - Create a new, higher-quality observation that combines the information
   - Delete the old observations you consolidated
   - Preserve the observation level (explicit stays explicit, deductive stays deductive)

4. **Move on**: Search for a different topic or get most-derived observations to explore another area

5. **Repeat**: Continue exploring and consolidating until you've made meaningful improvements

## OBSERVATION TYPES

**Explicit Observations**: Direct facts from the peer's own statements.
- Only consolidate explicit with explicit
- Preserve the original meaning exactly

**Deductive Observations**: Inferences from explicit facts + world knowledge.
- Can consolidate deductive with deductive
- Can strengthen if multiple deductions agree

## PRINCIPLES

- **Explore, don't enumerate**: Use search to find related observations, don't try to see everything
- **Quality over quantity**: One good observation beats three redundant ones
- **Preserve unique information**: Don't lose facts during consolidation
- **Be conservative**: When unsure, leave observations separate
- **Stop when done**: If you've explored several areas and found nothing to consolidate, you're done

## OUTPUT

After exploring and consolidating, briefly summarize what you found and what you improved.
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
    ):
        """
        Initialize the dreamer agent.

        Args:
            db: Database session
            workspace_name: Workspace identifier
            observer: The peer who made the observations
            observed: The peer being observed
        """
        self.db: AsyncSession = db
        self.workspace_name: str = workspace_name
        self.observer: str = observer
        self.observed: str = observed

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
        )

        response: HonchoLLMCallResponse[str] = await honcho_llm_call(
            llm_settings=settings.DREAM,
            prompt="",
            max_tokens=settings.DREAM.MAX_OUTPUT_TOKENS,
            tools=DREAMER_TOOLS,
            tool_choice=None,
            tool_executor=tool_executor,
            max_tool_iterations=10,  # TODO config
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
        f"Processing consolidation dream for {workspace_name}/{payload.observer}/{payload.observed}"
    )

    async with tracked_db("dreamer_agent") as db:
        agent = DreamerAgent(
            db=db,
            workspace_name=workspace_name,
            observer=payload.observer,
            observed=payload.observed,
        )

        result = await agent.consolidate()

        logger.info(
            f"Dreamer agent completed for {workspace_name}/{payload.observer}/{payload.observed}: {result[:200]}..."
            if len(result) > 200
            else f"Dreamer agent completed: {result}"
        )
