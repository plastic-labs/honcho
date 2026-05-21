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
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from nanoid import generate as generate_nanoid

from src import crud, schemas
from src.config import ConfiguredModelSettings, settings
from src.dependencies import tracked_db
from src.exceptions import ValidationException
from src.llm import HonchoLLMCallResponse, honcho_llm_call
from src.llm.types import LLMTelemetryContext
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

logger = logging.getLogger(__name__)


def _require_specialist_model_config(
    model_config: ConfiguredModelSettings | None,
    *,
    specialist_name: str,
) -> ConfiguredModelSettings:
    if model_config is None:
        raise ValidationException(
            f"{specialist_name} MODEL_CONFIG must be resolved before use"
        )
    return model_config


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
    # Whether this specialist is allowed to write to the peer card. Defaults to True;
    # specialists that should never touch the card (e.g., induction) override to False.
    can_update_peer_card: bool = True
    # Subclasses can override to customize the peer card update instruction
    peer_card_update_instruction: str = (
        "Only update this with durable identity markers via `update_peer_card`."
    )

    @abstractmethod
    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        """Get the tools available to this specialist."""
        ...

    @abstractmethod
    def get_model_config(self) -> ConfiguredModelSettings:
        """Get the configured model to use for this specialist."""
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
        run_id = parent_run_id or generate_nanoid()
        task_name = f"dreamer_{self.name}_{run_id}"
        start_time = time.perf_counter()

        # Telemetry state initialized BEFORE the try so the finally block can
        # always read consistent values. Without this, a failure in preflight
        # (peer lookup, peer-card preload, create_tool_executor, get_model_config,
        # prompt construction) would bypass the finally entirely and the run
        # would disappear from failure-path analytics — orphaning the
        # downstream DreamRunEvent.
        specialist_success = False
        specialist_error_class: str | None = None
        response: HonchoLLMCallResponse[str] | None = None

        # Rollups initialized here so they're accessible from the finally
        # block on the failure path (where they stay at defaults).
        created_observation_count = 0
        deleted_observation_count = 0
        peer_card_updated = False
        search_tool_calls_count = 0
        duration_ms = 0.0
        # Per-level rollups — accumulated from each create/delete_observations
        # tool call's metadata.levels list. Counter rather than list[str] so
        # the emitted dict stays compact even when the specialist produces
        # many observations.
        created_counts_by_level: Counter[str] = Counter()
        deleted_counts_by_level: Counter[str] = Counter()

        try:
            # Short-lived DB session for preflight operations
            async with tracked_db("dream.specialist.preflight") as db:
                await crud.get_peer(
                    db, workspace_name, schemas.PeerCreate(name=observer)
                )
                if observer != observed:
                    await crud.get_peer(
                        db, workspace_name, schemas.PeerCreate(name=observed)
                    )

                # Determine if peer card tools should be included. Specialists that
                # cannot write to the peer card (e.g., induction) skip the fetch and
                # the prompt section entirely.
                peer_card_enabled = self.can_update_peer_card and (
                    configuration is None or configuration.peer_card.create
                )

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

            model_config = self.get_model_config()

            # Respect operator-configured max_output_tokens on the specialist's
            # ModelConfig (e.g. DREAM_DEDUCTION_MODEL_CONFIG__MAX_OUTPUT_TOKENS).
            # Only fall back to the specialist's hardcoded default when the
            # config leaves max_output_tokens unset or non-positive.
            configured_max = model_config.max_output_tokens
            effective_max_tokens = (
                configured_max
                if configured_max and configured_max > 0
                else self.get_max_tokens()
            )

            # call_purpose maps "deduction"/"induction" specialist names onto the
            # closed CallPurpose enum slugs without importing the enum here.
            call_purpose_slug = f"dream.{self.name}"

            # Run the agent loop
            response = await honcho_llm_call(
                model_config=model_config,
                prompt="",  # Ignored since we pass messages
                max_tokens=effective_max_tokens,
                tools=self.get_tools(peer_card_enabled=peer_card_enabled),
                tool_choice=None,
                tool_executor=tool_executor,
                max_tool_iterations=self.get_max_iterations(),
                messages=messages,
                track_name=f"Dreamer/{self.name}",
                telemetry=LLMTelemetryContext(
                    workspace_name=workspace_name,
                    call_purpose=call_purpose_slug,
                    parent_category="dream",
                    agent_type=self.name,
                    run_id=run_id,
                    observer=observer,
                    observed=observed,
                ),
            )

            # Log metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            accumulate_metric(task_name, "total_duration", duration_ms, "ms")
            accumulate_metric(
                task_name, "tool_calls", len(response.tool_calls_made), "count"
            )
            accumulate_metric(task_name, "input_tokens", response.input_tokens, "count")
            accumulate_metric(
                task_name, "output_tokens", response.output_tokens, "count"
            )

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

            # count actual observations created/deleted from the
            # ToolResult.metadata that stashed on `all_tool_calls[i]`.
            # Counting tool-name occurrences would mis-attribute: a single
            # create_observations call can produce N (or zero) observations. The
            # truth lives in the handler's returned metadata.
            _search_tools = {
                "search_memory",
                "search_messages",
                "search_messages_temporal",
            }
            for tc in response.tool_calls_made:
                tool_name_any: Any = tc.get("tool_name") or tc.get("name")
                meta_any: Any = tc.get("tool_result_metadata") or {}
                if tool_name_any in _search_tools:
                    search_tool_calls_count += 1
                if isinstance(meta_any, dict):
                    # `meta_any` is `dict[Unknown, Unknown]` after the isinstance
                    # narrow because tool_calls_made is typed list[dict[str, Any]].
                    # Cast to the expected dict shape to silence the partial-known
                    # warning without losing runtime safety.
                    meta_dict = cast(dict[str, Any], meta_any)
                    created_val: Any = meta_dict.get("created_count") or 0
                    deleted_val: Any = meta_dict.get("deleted_count") or 0
                    created_observation_count += int(created_val)
                    deleted_observation_count += int(deleted_val)
                    if meta_dict.get("peer_card_updated"):
                        peer_card_updated = True
                    # Accumulate per-level counts from create/delete observations.
                    # Both handlers stash `{"levels": ["explicit", "deductive", ...]}`
                    # in metadata (agent_tools.py:1373 + agent_tools.py:2011).
                    levels_any: Any = meta_dict.get("levels")
                    if isinstance(levels_any, list):
                        levels_list = cast(list[Any], levels_any)
                        level_strs = [
                            str(level) for level in levels_list if level is not None
                        ]
                        # Tool-name dispatch decides which counter to update —
                        # create_observations metadata has `created_count`,
                        # delete_observations has `deleted_count`.
                        if "created_count" in meta_dict:
                            created_counts_by_level.update(level_strs)
                        elif "deleted_count" in meta_dict:
                            deleted_counts_by_level.update(level_strs)

            specialist_success = True

            return SpecialistResult(
                run_id=run_id,
                specialist_type=self.name,
                iterations=response.iterations,
                tool_calls_count=len(response.tool_calls_made),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                duration_ms=duration_ms,
                success=True,
                content=response.content,
            )
        except BaseException as e:
            # BaseException (not Exception) — asyncio.CancelledError doesn't
            # inherit from Exception in py3.8+, and we want the failure
            # telemetry populated for cancellations too (worker shutdown,
            # client disconnect). `raise` preserves cancellation semantics.
            specialist_error_class = type(e).__name__
            if duration_ms == 0.0:
                duration_ms = (time.perf_counter() - start_time) * 1000
            raise
        finally:
            # Emit DreamSpecialistEvent unconditionally so the success=False
            # path of the schema is actually populated. Telemetry must not
            # raise from inside finally during exception propagation; the
            # emitter itself swallows errors but we add a defensive try
            # in case event construction fails (e.g. schema validation).
            try:
                tool_calls_count = (
                    len(response.tool_calls_made) if response is not None else 0
                )
                input_tokens = response.input_tokens if response is not None else 0
                output_tokens = response.output_tokens if response is not None else 0
                iterations = response.iterations if response is not None else 0
                emit(
                    DreamSpecialistEvent(
                        run_id=run_id,
                        specialist_type=self.name,
                        workspace_name=workspace_name,
                        observer=observer,
                        observed=observed,
                        iterations=iterations,
                        tool_calls_count=tool_calls_count,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        duration_ms=duration_ms,
                        success=specialist_success,
                        error_class=specialist_error_class,
                        # denormalized rollups (all 0 on the failure path)
                        created_observation_count=created_observation_count,
                        deleted_observation_count=deleted_observation_count,
                        peer_card_updated=peer_card_updated,
                        search_tool_calls_count=search_tool_calls_count,
                        # Per-level breakdowns — `dict(Counter)` keeps the
                        # serialized event compact (zero-count levels are
                        # omitted, not enumerated).
                        created_counts_by_level=dict(created_counts_by_level),
                        deleted_counts_by_level=dict(deleted_counts_by_level),
                    )
                )
            except Exception:  # pragma: no cover - telemetry must not raise
                logger.debug("Failed to emit DreamSpecialistEvent", exc_info=True)


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
    peer_card_update_instruction: str = "Update this with `update_peer_card` only for stable identity markers. See the PEER CARD section in the system prompt for the allowed entry kinds and rules."

    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        if peer_card_enabled:
            return DEDUCTION_SPECIALIST_TOOLS
        return [
            t
            for t in DEDUCTION_SPECIALIST_TOOLS
            if t["name"] not in PEER_CARD_TOOL_NAMES
        ]

    def get_model_config(self) -> ConfiguredModelSettings:
        return _require_specialist_model_config(
            settings.DREAM.DEDUCTION_MODEL_CONFIG,
            specialist_name="DREAM DEDUCTION",
        )

    def get_max_tokens(self) -> int:
        return 8192

    def get_max_iterations(self) -> int:
        return 12

    def build_system_prompt(
        self, observed: str, *, peer_card_enabled: bool = True
    ) -> str:
        peer_card_section = ""
        if peer_card_enabled:
            peer_card_section = f"""

## PEER CARD (REQUIRED)

The peer card is {observed}'s identity store: stable identity markers that distinguish this entity from others and persist across interactions. Behavior, tendencies, transient state, and episodic facts belong in observations, not on the peer card.

A peer can be anything with identity that changes over time — a human, an agent, a codebase, a team, an organization. Do not assume {observed} is human. Do not require any field; empty is the correct output when evidence is absent.

### Allowed entry kinds

Each entry must start with one of these four prefixes (exact case, followed by a space):

- `IDENTITY: ...` — canonical name, kind, aliases, IDs
  - `IDENTITY: Name: Alice`
  - `IDENTITY: Kind: Python monorepo`
  - `IDENTITY: Version: 4.2`
  - `IDENTITY: Aliases: alice@example.com`
- `ATTRIBUTE: ...` — stable durable property of the entity (including explicitly stated standing preferences)
  - `ATTRIBUTE: Location: NYC`
  - `ATTRIBUTE: Language: Python`
  - `ATTRIBUTE: Prefers tea`
  - `ATTRIBUTE: Charter: ship Honcho infrastructure`
- `RELATIONSHIP: ...` — durable link to another entity
  - `RELATIONSHIP: Spouse: Bob`
  - `RELATIONSHIP: Maintainer: vineeth`
  - `RELATIONSHIP: Members: vineeth, rajat`
- `INSTRUCTION: ...` — standing rule of engagement that {observed} has explicitly stated (do/don't for the observer). Only when explicit; never inferred from behavior.
  - `INSTRUCTION: Call me Vee`
  - `INSTRUCTION: Never push to main without review`

### Rules

1. **Stable.** If the value plausibly changes within six months absent a deliberate announcement, it does not belong on the card. Prefer leaving the card empty over filling it with volatile content.
2. **Subject is {observed}.** Every entry must be a fact about {observed}, not about another participant in the session. Never write facts about co-occurring peers into the card, no matter how frequently they appear in the messages.
3. **Evidence-grounded.** Only write what {observed} has explicitly stated, or what another participant has explicitly stated about {observed} with {observed}'s assent. No "general knowledge" inferences (`"co-founder"` does not imply an age; mentioning a colleague does not imply a family relationship).
4. **Type-agnostic.** {observed} may not be human. Do not require name/age/location/family/occupation fields.
5. **No behavioral content.** TRAITs, behavioral tendencies, patterns, and inferred preferences belong in observations, not on the peer card. Do not write `TRAIT:` entries or behavioral `PREFERENCE:` entries — they will be rejected.
6. **No evidence bundles.** Each entry is one concise fact. No `e.g.` clauses, no parenthetical example lists, no semicolon-separated value dumps.

### Migrating an existing peer card

The CURRENT PEER CARD shown in the user message may contain entries from an older format that do not start with an allowed prefix (e.g. `Name: Alice`, `Lives in NYC`, `TRAIT: Analytical`, `PREFERENCE: Detailed explanations`). When you call `update_peer_card`, you are responsible for re-emitting the entries you want to keep — entries you omit are dropped, and entries without an allowed prefix are silently rejected.

For each legacy entry:

- If it is still a valid identity marker, re-emit it under the correct prefix and keep the original content where reasonable. Examples:
  - `Name: Alice` → `IDENTITY: Name: Alice`
  - `Lives in NYC` → `ATTRIBUTE: Location: NYC`
  - `Works at Google` → `ATTRIBUTE: Employer: Google`
  - `INSTRUCTION: Call me Vee` → keep as is (already correctly prefixed)
- Drop entries that violate the rules above: behavioral `TRAIT:` lines, inferred behavioral `PREFERENCE:` lines, one-off events, transient state, evidence bundles. Do not re-prefix them — they are not identity markers.

When in doubt about a specific legacy entry, prefer migrating it (so valid info isn't lost) over dropping it. Splitting one dense legacy entry into multiple correctly-prefixed entries is fine and encouraged (e.g. a semicolon-separated `Tech Stack:` dump can become several `ATTRIBUTE:` lines, one per durable tool/platform).

Call `update_peer_card` with the complete deduplicated list when there is a durable identity update to record, or when the existing card needs migration. Entries that do not start with one of the four allowed prefixes will be rejected. Keep concise (max 40 entries)."""

        return f"""You are a deductive reasoning agent analyzing observations about {observed}.

## YOUR JOB

Create deductive observations by finding logical implications in what's already known. Think like a detective connecting evidence.

## PHASE 1: DISCOVERY

Explore what's actually in memory. Use these tools freely:
- `get_recent_observations` - See what's been learned recently
- `search_memory` - Search for specific topics
- `search_messages` - See actual conversation content

Spend a few tool calls understanding the landscape before creating anything.

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

Use `create_observations_deductive`.

```json
{{
  "observations": [{{
    "content": "The logical conclusion",
    "source_ids": ["id1", "id2"],
    "premises": ["premise 1 text", "premise 2 text"]
  }}]
}}
```

## RULES

1. Don't explain your reasoning - just call tools
2. Create observations based on what you ACTUALLY FIND, not what you expect
3. Always include source_ids linking to the observations you're synthesizing
4. Empty or missing source_ids will be rejected
5. Delete outdated observations - don't leave duplicates
6. Quality over quantity - fewer good deductions beat many weak ones"""

    def build_user_prompt(
        self,
        hints: list[str] | None,
        peer_card: list[str] | None = None,
    ) -> str:
        peer_card_context = self._build_peer_card_context(peer_card)

        if hints:
            hints_str = "\n".join(f"- {q}" for q in hints[:5])
            return f"""{peer_card_context}Start by exploring recent observations and messages. These topics may be worth investigating:

{hints_str}

But follow the evidence - if you find something more interesting, pursue that instead.

Begin with `get_recent_observations` to see what's there."""

        return f"""{peer_card_context}Explore the observation space and create deductive observations.

Start with `get_recent_observations` to see what's been learned recently, then investigate whatever seems most promising.

Look for:
1. Knowledge updates (same fact, different values over time)
2. Logical implications that haven't been made explicit
3. Contradictions that need flagging

Go."""


class InductionSpecialist(BaseSpecialist):
    """
    Creates inductive observations from explicit and deductive observations.

    This specialist:
    1. Explores observations to understand what's there
    2. Identifies patterns and generalizations across multiple observations
    3. Creates new inductive observations with source linkage

    Does not write to the peer card — the peer card stores stable identity markers,
    which is deduction's responsibility. Inductive patterns and tendencies stay as
    observations.
    """

    name: str = "induction"
    # Induction never writes to the peer card; behavioral patterns are observations.
    can_update_peer_card: bool = False

    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        _ = peer_card_enabled
        return INDUCTION_SPECIALIST_TOOLS

    def get_model_config(self) -> ConfiguredModelSettings:
        return _require_specialist_model_config(
            settings.DREAM.INDUCTION_MODEL_CONFIG,
            specialist_name="DREAM INDUCTION",
        )

    def get_max_tokens(self) -> int:
        return 8192

    def get_max_iterations(self) -> int:
        return 10

    def build_system_prompt(
        self, observed: str, *, peer_card_enabled: bool = True
    ) -> str:
        _ = peer_card_enabled
        return f"""You are an inductive reasoning agent identifying patterns about {observed}.

## YOUR JOB

Create inductive observations by finding patterns across multiple observations. Think like a psychologist identifying behavioral tendencies.

## PHASE 1: DISCOVERY

Explore broadly to find patterns. Use these tools:
- `get_recent_observations` - Recent learnings
- `search_memory` - Topic-specific search
- `search_messages` - Actual conversation content

Look at BOTH explicit observations AND deductive ones. Patterns often emerge from synthesizing across both levels.

## PHASE 2: ACTION

Create inductive observations when you see patterns:

### Behavioral Patterns
- "Tends to reschedule meetings when stressed"
- "Makes decisions after consulting with partner"
- "Projects follow: enthusiasm → doubt → completion"

### Preferences
- "Prefers morning meetings"
- "Likes detailed technical explanations"

### Personality Traits
- "Generally optimistic about outcomes"
- "Detail-oriented in planning"

### Temporal Patterns
- "Career goals have remained consistent"
- "Living situation changes frequently"

## CREATING OBSERVATIONS

Use `create_observations_inductive`.

```json
{{
  "observations": [{{
    "content": "The pattern or generalization",
    "source_ids": ["id1", "id2", "id3"],
    "sources": ["evidence 1", "evidence 2"],
    "pattern_type": "tendency", // preference|behavior|personality|tendency|correlation
    "confidence": "medium" // low (2 sources), medium (3-4), high (5+)
  }}]
}}
```

## RULES

1. Minimum 2 source observations required - patterns need evidence
2. Don't just restate a single fact as a pattern
3. Confidence based on evidence count: 2=low, 3-4=medium, 5+=high
4. Look for HOW things change over time, not just static facts
5. Include source_ids - always link back to evidence
6. Empty or missing source_ids will be rejected"""

    def build_user_prompt(
        self,
        hints: list[str] | None,
        peer_card: list[str] | None = None,
    ) -> str:
        # Induction does not consume peer card context — it produces inductive
        # observations, not identity-marker updates.
        _ = peer_card

        if hints:
            hints_str = "\n".join(f"- {q}" for q in hints[:5])
            return f"""Explore and find patterns. These areas may be worth investigating:

{hints_str}

But follow the evidence - if you find patterns elsewhere, pursue those.

Start with `get_recent_observations`."""

        return """Explore the observation space and identify patterns.

Remember: patterns need 2+ sources. Look for tendencies, preferences, and behavioral regularities.

Go."""


# Singleton instances
SPECIALISTS: dict[str, BaseSpecialist] = {
    "deduction": DeductionSpecialist(),
    "induction": InductionSpecialist(),
}
