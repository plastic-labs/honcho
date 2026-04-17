"""
Agentic specialists for the dream cycle.

Each specialist is a fully autonomous agent that:
1. Receives probing questions as entry points
2. Uses tools to search for relevant observations
3. Creates new observations (deductive or inductive)
4. Can delete duplicates (deduction only)
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src import crud, schemas
from src.config import settings
from src.dependencies import tracked_db
from src.schemas import ResolvedConfiguration
from src.telemetry import prometheus_metrics
from src.telemetry.events import DreamSpecialistEvent, emit
from src.telemetry.logging import accumulate_metric, log_performance_metrics
from src.telemetry.prometheus.metrics import TokenTypes
from src.utils.agent_tools import (
    DEDUCTION_SPECIALIST_TOOLS,
    INDUCTION_SPECIALIST_TOOLS,
    create_tool_executor,
)
from src.utils.clients import HonchoLLMCallResponse, honcho_llm_call

logger = logging.getLogger(__name__)


@dataclass
class SpecialistResult:
    """Result of a specialist run for telemetry and aggregation."""

    run_id: str
    specialist_type: str
    iterations: int
    tool_calls_count: int
    input_tokens: int
    output_tokens: int
    duration_ms: float
    success: bool
    content: str


# Tool names to exclude when peer card creation is disabled
PEER_CARD_TOOL_NAMES = {"update_peer_card"}


class BaseSpecialist(ABC):
    """Base class for agentic specialists."""

    name: str = "base"
    # Subclasses can override to customize the peer card update instruction
    peer_card_update_instruction: str = (
        "Only update this with durable profile facts via `update_peer_card`."
    )

    @abstractmethod
    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        """Get the tools available to this specialist."""
        ...

    @abstractmethod
    def get_model(self) -> str:
        """Get the model to use for this specialist."""
        ...

    def get_max_tokens(self) -> int:
        """Get max output tokens for this specialist."""
        return 16384

    def get_max_iterations(self) -> int:
        """Get max tool iterations."""
        return 15

    @abstractmethod
    def build_system_prompt(
        self, observed: str, *, peer_card_enabled: bool = True
    ) -> str:
        """Build the system prompt for this specialist."""
        ...

    @abstractmethod
    def build_user_prompt(
        self,
        hints: list[str] | None,
        peer_card: list[str] | None = None,
    ) -> str:
        """Build the user prompt with optional exploration hints and current peer card."""
        ...

    def _build_peer_card_context(self, peer_card: list[str] | None) -> str:
        """Build the peer card context section for user prompts."""
        if not peer_card:
            return ""
        facts = "\n".join(f"- {fact}" for fact in peer_card)
        return f"""
## CURRENT PEER CARD

{facts}

{self.peer_card_update_instruction}
If you update it, send the full deduplicated list and remove stale entries.

"""

    async def run(
        self,
        workspace_name: str,
        observer: str,
        observed: str,
        session_name: str | None,
        hints: list[str] | None = None,
        configuration: ResolvedConfiguration | None = None,
        parent_run_id: str | None = None,
    ) -> SpecialistResult:
        """
        Run the specialist agent.

        Uses short-lived DB sessions to avoid holding connections during LLM calls.

        Args:
            workspace_name: Workspace identifier
            observer: The observing peer
            observed: The peer being observed
            session_name: Session identifier
            hints: Optional hints to guide exploration (specialists explore freely if None)
            configuration: Resolved configuration for checking feature flags (optional)
            parent_run_id: Optional run_id from orchestrator for correlation

        Returns:
            SpecialistResult with metrics and content
        """
        run_id = parent_run_id or str(uuid.uuid4())[:8]
        task_name = f"dreamer_{self.name}_{run_id}"
        start_time = time.perf_counter()

        # Short-lived DB session for preflight operations
        async with tracked_db("dream.specialist.preflight") as db:
            await crud.get_peer(db, workspace_name, schemas.PeerCreate(name=observer))
            if observer != observed:
                await crud.get_peer(
                    db, workspace_name, schemas.PeerCreate(name=observed)
                )

            # Determine if peer card tools should be included
            peer_card_enabled = configuration is None or configuration.peer_card.create

            # Fetch current peer card to inject into prompt (saves a tool call)
            current_peer_card: list[str] | None = None
            if peer_card_enabled:
                current_peer_card = await crud.get_peer_card(
                    db,
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                )
        # DB session closed — LLM calls happen without holding a connection

        # Build messages
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    observed, peer_card_enabled=peer_card_enabled
                ),
            },
            {
                "role": "user",
                "content": self.build_user_prompt(hints, current_peer_card),
            },
        ]

        # Create tool executor with telemetry context
        tool_executor: Callable[
            [str, dict[str, Any]], Any
        ] = await create_tool_executor(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            session_name=session_name,
            include_observation_ids=True,
            history_token_limit=settings.DREAM.HISTORY_TOKEN_LIMIT,
            configuration=configuration,
            run_id=run_id,
            agent_type=self.name,
            parent_category="dream",
        )

        # Get model with potential override
        model = self.get_model()
        llm_settings = settings.DREAM.model_copy(update={"MODEL": model})

        # Track iterations via callback
        iteration_count = 0

        def iteration_callback(data: Any) -> None:
            nonlocal iteration_count
            iteration_count = data.iteration

        # Run the agent loop
        response: HonchoLLMCallResponse[str] = await honcho_llm_call(
            llm_settings=llm_settings,
            prompt="",  # Ignored since we pass messages
            max_tokens=self.get_max_tokens(),
            tools=self.get_tools(peer_card_enabled=peer_card_enabled),
            tool_choice=None,
            tool_executor=tool_executor,
            max_tool_iterations=self.get_max_iterations(),
            messages=messages,
            track_name=f"Dreamer/{self.name}",
            trace_name=f"dreamer_{self.name}",
            iteration_callback=iteration_callback,
        )

        # Log metrics
        duration_ms = (time.perf_counter() - start_time) * 1000
        accumulate_metric(task_name, "total_duration", duration_ms, "ms")
        accumulate_metric(
            task_name, "tool_calls", len(response.tool_calls_made), "count"
        )
        accumulate_metric(task_name, "input_tokens", response.input_tokens, "count")
        accumulate_metric(task_name, "output_tokens", response.output_tokens, "count")

        # Prometheus metrics
        if settings.METRICS.ENABLED:
            prometheus_metrics.record_dreamer_tokens(
                count=response.input_tokens,
                specialist_name=self.name,
                token_type=TokenTypes.INPUT.value,
            )
            prometheus_metrics.record_dreamer_tokens(
                count=response.output_tokens,
                specialist_name=self.name,
                token_type=TokenTypes.OUTPUT.value,
            )

        logger.info(
            f"{self.name}: Completed in {duration_ms:.0f}ms, "
            + f"{len(response.tool_calls_made)} tool calls, "
            + f"{response.input_tokens} in / {response.output_tokens} out"
        )

        log_performance_metrics(f"dreamer_{self.name}", run_id)

        # Emit telemetry event
        emit(
            DreamSpecialistEvent(
                run_id=run_id,
                specialist_type=self.name,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                iterations=iteration_count,
                tool_calls_count=len(response.tool_calls_made),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                duration_ms=duration_ms,
                success=True,
            )
        )

        return SpecialistResult(
            run_id=run_id,
            specialist_type=self.name,
            iterations=iteration_count,
            tool_calls_count=len(response.tool_calls_made),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            duration_ms=duration_ms,
            success=True,
            content=response.content,
        )


class DeductionSpecialist(BaseSpecialist):
    """
    Creates deductive observations from explicit observations.

    This specialist:
    1. Explores recent observations and messages to understand what's there
    2. Identifies logical implications, knowledge updates, and contradictions
    3. Creates new deductive observations with premise linkage
    4. Deletes outdated observations
    5. Updates peer card with biographical facts
    """

    name: str = "deduction"
    peer_card_update_instruction: str = "Update this with `update_peer_card` only for stable biographical/profile facts."

    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        if peer_card_enabled:
            return DEDUCTION_SPECIALIST_TOOLS
        return [
            t
            for t in DEDUCTION_SPECIALIST_TOOLS
            if t["name"] not in PEER_CARD_TOOL_NAMES
        ]

    def get_model(self) -> str:
        return settings.DREAM.DEDUCTION_MODEL

    def get_max_tokens(self) -> int:
        return 8192

    def get_max_iterations(self) -> int:
        return 6

    def build_system_prompt(
        self, observed: str, *, peer_card_enabled: bool = True
    ) -> str:
        peer_card_section = ""
        if peer_card_enabled:
            peer_card_section = """

## PEER CARD (REQUIRED)

The peer card is a summary of stable biographical facts. You MUST update it when you learn:
- Name, email, location, occupation/role
- Family members and relationships
- Standing instructions ("call me X", "don't mention Y")
- Core preferences and traits

Priority order for peer-card entries:
1. Canonical identity facts explicitly stated by the user (name, email, title, location)
2. Standing instructions and routing preferences
3. Long-lived preferences and personality traits
4. Stable role/occupation facts
5. Only then, a few high-value project/work context facts if they materially help identification

Avoid low-value or generic entries such as:
- "Works on projects"
- "Evaluates systems"
- "Capable of reviewing conversation history"
- temporary tasks, one-off questions, or broad capability statements

Never add temporary event summaries, one-off conclusions, reasoning traces, or contradiction notes.

Format entries as:
- Plain facts: "Name: Alice", "Email: alice@example.com", "Lives in NYC", "Role: Software developer"
- `INSTRUCTION: ...` for standing instructions
- `PREFERENCE: ...` for preferences
- `TRAIT: ...` for personality traits

Good examples:
- "Name: Aubrey Freeman III"
- "Email: aubrey@freeman-wisco.com"
- "Location: Wisconsin"
- "INSTRUCTION: Gerard is the single front door"
- "PREFERENCE: Route specialists behind the scenes"

Call `update_peer_card` with the complete updated list in the `content` field when you have new biographical info.
Do not use argument names like `entries` or `peer_card_updates`.
Keep it concise (max 12 entries unless there is a strong reason for more), deduplicated, and current."""

        return f"""You are a deductive reasoning agent analyzing observations about {observed}.

## YOUR JOB

Create deductive observations by finding logical implications in what's already known. Think like a detective connecting evidence.

## EXECUTION DISCIPLINE

- Call `extract_preferences` FIRST to pull standing instructions and stable preferences from conversation history.
- After that, do at most 2 additional discovery calls (`get_recent_observations`, `search_memory`, or `search_messages`).
- Stop discovery after at most 3 tool calls total before acting.
- Once you have enough evidence, you MUST do one of these before the loop ends:
  1. Call `update_peer_card` with the complete updated card in the `content` argument, or
  2. Call `finish_consolidation` with a short reason that no durable card update is warranted.
- Do not keep searching once you already have name/location/preferences/standing-instruction evidence.
- If you find durable profile facts, prefer updating the peer card over creating more search queries.

## PHASE 1: DISCOVERY

Use a short discovery pass only:
- `extract_preferences` - first call, for preferences and standing instructions
- `get_recent_observations` - recent facts
- `search_memory` - use targeted identity/routing queries such as:
  - `Aubrey Freeman III name email`
  - `aubrey@freeman-wisco.com Wisconsin`
  - `Gerard single front door specialists behind the scenes`
  - `software developer role occupation`
- `search_messages` - one targeted follow-up if needed

## PHASE 2: ACTION

Once you understand what's there, create observations and clean up:

### Knowledge Updates (HIGH PRIORITY)
When the same fact has different values at different times:
- "meeting Tuesday" [old] → "meeting moved to Thursday" [new]
- Create a deductive update observation
- DELETE the outdated observation immediately

### Logical Implications
Extract implicit information:
- "works as SWE at Google" → "has software engineering skills", "employed in tech"
- "has kids ages 5 and 8" → "is a parent", "has school-age children"

### Contradictions
When statements can't both be true (not just updates), flag them:
- "I love coffee" vs "I hate coffee" → contradiction observation
{peer_card_section}

## CREATING OBSERVATIONS

```json
{{
  "observations": [{{
    "content": "The logical conclusion",
    "level": "deductive",  // or "contradiction"
    "source_ids": ["id1", "id2"],
    "premises": ["premise 1 text", "premise 2 text"]
  }}]
}}
```

## RULES

1. Don't explain your reasoning - just call tools
2. Create observations based on what you ACTUALLY FIND, not what you expect
3. Always include source_ids linking to the observations you're synthesizing
4. Delete outdated observations - don't leave duplicates
5. Quality over quantity - fewer good deductions beat many weak ones
6. Before your final step, either call `update_peer_card` or call `finish_consolidation` explicitly"""

    def build_user_prompt(
        self,
        hints: list[str] | None,
        peer_card: list[str] | None = None,
    ) -> str:
        peer_card_context = self._build_peer_card_context(peer_card)

        if hints:
            hints_str = "\n".join(f"- {q}" for q in hints[:5])
            return f"""{peer_card_context}Start with `extract_preferences`, then do a very short discovery pass using at most two of `get_recent_observations`, `search_memory`, or `search_messages`.

These topics may be worth investigating:

{hints_str}

Once you have enough evidence, stop searching and either update the peer card or finish consolidation explicitly."""

        return f"""{peer_card_context}Start with `extract_preferences`, then do a very short discovery pass using at most two of `get_recent_observations`, `search_memory`, or `search_messages`.

Look for:
1. Knowledge updates (same fact, different values over time)
2. Logical implications that haven't been made explicit
3. Contradictions that need flagging
4. Canonical identity/routing facts first: exact name, email, location, Gerard/front-door preference

Once you have enough evidence, stop searching and either update the peer card or finish consolidation explicitly.

Go."""


class InductionSpecialist(BaseSpecialist):
    """
    Creates inductive observations from explicit and deductive observations.

    This specialist:
    1. Explores observations to understand what's there
    2. Identifies patterns and generalizations across multiple observations
    3. Creates new inductive observations with source linkage
    4. Updates peer card with high-confidence traits and tendencies
    """

    name: str = "induction"
    peer_card_update_instruction: str = "Only add highly stable profile traits/preferences; do not copy transient conclusions."

    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        if peer_card_enabled:
            return INDUCTION_SPECIALIST_TOOLS
        return [
            t
            for t in INDUCTION_SPECIALIST_TOOLS
            if t["name"] not in PEER_CARD_TOOL_NAMES
        ]

    def get_model(self) -> str:
        return settings.DREAM.INDUCTION_MODEL

    def get_max_tokens(self) -> int:
        return 8192

    def get_max_iterations(self) -> int:
        return 6

    def build_system_prompt(
        self, observed: str, *, peer_card_enabled: bool = True
    ) -> str:
        peer_card_section = ""
        if peer_card_enabled:
            peer_card_section = """

## PEER CARD (REQUIRED)

After identifying patterns, only update the peer card for durable profile-level traits/preferences and missing canonical identity facts.

Priority order for peer-card entries:
1. Missing canonical identity or routing facts the deduction phase failed to add
2. Standing instructions and strong routing preferences
3. Durable long-lived preferences and personality traits
4. Exclude generic project chatter unless it is essential to identifying the user

Prefer entries like:
- `INSTRUCTION: Gerard is the single front door`
- `PREFERENCE: Route specialists behind the scenes`
- `TRAIT: Methodical and process-oriented`
- `TRAIT: Detail-conscious in verification and debugging`

Do NOT add temporary patterns, episode-specific conclusions, reasoning summaries, or generic capability statements.
Call `update_peer_card` with the complete deduplicated list in the `content` field only when a durable profile update is warranted.
Do not use argument names like `entries` or `peer_card_updates`.
Keep it concise (max 12 entries unless there is a strong reason for more)."""

        return f"""You are an inductive reasoning agent identifying patterns about {observed}.

## YOUR JOB

Create inductive observations by finding patterns across multiple observations. Think like a psychologist identifying behavioral tendencies.

## EXECUTION DISCIPLINE

- Call `extract_preferences` FIRST to pull standing instructions and stable preferences from conversation history.
- After that, do at most 2 additional discovery calls (`get_recent_observations`, `search_memory`, or `search_messages`).
- Stop discovery after at most 3 tool calls total before acting.
- Once you have enough evidence, you MUST do one of these before the loop ends:
  1. Call `update_peer_card` with the complete updated card in the `content` argument, or
  2. Call `finish_consolidation` with a short reason that no durable card update is warranted.
- Do not keep searching once you already have enough evidence for durable traits/preferences.
- If you identify a durable preference or trait, prefer updating the peer card over more search queries.

## PHASE 1: DISCOVERY

Use a short discovery pass only:
- `extract_preferences` - first call, for preferences and standing instructions
- `get_recent_observations` - recent learnings
- `search_memory` - one targeted follow-up if needed
- `search_messages` - one targeted follow-up if needed

Look at BOTH explicit observations AND deductive ones. Patterns often emerge from synthesizing across both levels.

## PHASE 2: ACTION

Create inductive observations only when you have enough evidence:

### Behavioral Patterns
- "Tends to reschedule meetings when stressed"
- "Makes decisions after consulting with partner"
- "Projects follow: enthusiasm → doubt → completion"

### Preferences
- "Prefers morning meetings"
- "Likes detailed technical explanations"

### Personality Traits
- "Analytical thinker"
- "Risk-averse with finances"
- "Values family time over overtime"
{peer_card_section}

If you do NOT have at least 2 concrete source observation IDs for a proposed inductive pattern, do NOT call `create_observations`.
In that case, update the peer card if warranted or call `finish_consolidation`.

## CREATING OBSERVATIONS

```json
{{
  "observations": [{{
    "content": "The pattern or generalization",
    "level": "inductive",
    "source_ids": ["id1", "id2", "id3"]
  }}]
}}
```

## RULES

1. Patterns need multiple examples - don't overgeneralize from one instance
2. Create observations based on what you ACTUALLY FIND, not stereotypes
3. Include source_ids for all observations supporting the pattern
4. Focus on useful patterns that help understand future behavior
5. Quality over quantity - fewer strong patterns beat many weak ones
6. Before your final step, either call `update_peer_card` or call `finish_consolidation` explicitly
7. Minimum 2 source observations required - patterns need evidence
8. Don't just restate a single fact as a pattern
9. Look for HOW things change over time, not just static facts
10. When calling `create_observations`, use only `content`, `level`, and `source_ids` for inductive observations
11. Do NOT send extra keys like `pattern_type`, `confidence`, or `sources`"""

    def build_user_prompt(
        self,
        hints: list[str] | None,
        peer_card: list[str] | None = None,
    ) -> str:
        peer_card_context = self._build_peer_card_context(peer_card)

        if hints:
            hints_str = "\n".join(f"- {q}" for q in hints[:5])
            return f"""{peer_card_context}Start with `extract_preferences`, then do a very short discovery pass using at most two of `get_recent_observations`, `search_memory`, or `search_messages`.

These areas may be worth investigating:

{hints_str}

Once you have enough evidence, stop searching and either update the peer card or finish consolidation explicitly."""

        return f"""{peer_card_context}Start with `extract_preferences`, then do a very short discovery pass using at most two of `get_recent_observations`, `search_memory`, or `search_messages`.

Look for repeated behaviors, preferences, and durable traits. Remember: patterns need 2+ sources.

Once you have enough evidence, stop searching and either update the peer card or finish consolidation explicitly.

Go."""


# Singleton instances
SPECIALISTS: dict[str, BaseSpecialist] = {
    "deduction": DeductionSpecialist(),
    "induction": InductionSpecialist(),
}
