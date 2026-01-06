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
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src import prometheus
from src.config import settings
from src.utils.agent_tools import (
    DEDUCTION_SPECIALIST_TOOLS,
    INDUCTION_SPECIALIST_TOOLS,
    create_tool_executor,
)
from src.utils.clients import HonchoLLMCallResponse, honcho_llm_call
from src.utils.logging import accumulate_metric, log_performance_metrics

logger = logging.getLogger(__name__)


class BaseSpecialist(ABC):
    """Base class for agentic specialists."""

    name: str = "base"

    @abstractmethod
    def get_tools(self) -> list[dict[str, Any]]:
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
    def build_system_prompt(self, observed: str) -> str:
        """Build the system prompt for this specialist."""
        ...

    @abstractmethod
    def build_user_prompt(self, probing_questions: list[str]) -> str:
        """Build the user prompt with probing questions."""
        ...

    async def run(
        self,
        db: AsyncSession,
        workspace_name: str,
        observer: str,
        observed: str,
        session_name: str,
        probing_questions: list[str],
    ) -> str:
        """
        Run the specialist agent.

        Args:
            db: Database session
            workspace_name: Workspace identifier
            observer: The observing peer
            observed: The peer being observed
            session_name: Session identifier
            probing_questions: Entry point questions to guide exploration

        Returns:
            Summary of work done
        """
        run_id = str(uuid.uuid4())[:8]
        task_name = f"dreamer_{self.name}_{run_id}"
        start_time = time.perf_counter()

        # Build messages
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.build_system_prompt(observed)},
            {"role": "user", "content": self.build_user_prompt(probing_questions)},
        ]

        # Create tool executor
        tool_executor: Callable[
            [str, dict[str, Any]], Any
        ] = await create_tool_executor(
            db=db,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            session_name=session_name,
            include_observation_ids=True,
            history_token_limit=settings.DREAM.HISTORY_TOKEN_LIMIT,
        )

        # Get model with potential override
        model = self.get_model()
        llm_settings = settings.DREAM.model_copy(update={"MODEL": model})

        # Run the agent loop
        response: HonchoLLMCallResponse[str] = await honcho_llm_call(
            llm_settings=llm_settings,
            prompt="",  # Ignored since we pass messages
            max_tokens=self.get_max_tokens(),
            tools=self.get_tools(),
            tool_choice=None,
            tool_executor=tool_executor,
            max_tool_iterations=self.get_max_iterations(),
            messages=messages,
            track_name=f"Dreamer/{self.name}",
        )

        # Log metrics
        duration_ms = (time.perf_counter() - start_time) * 1000
        accumulate_metric(task_name, "total_duration", duration_ms, "ms")
        accumulate_metric(
            task_name, "tool_calls", len(response.tool_calls_made), "count"
        )
        accumulate_metric(task_name, "input_tokens", response.input_tokens, "count")
        accumulate_metric(task_name, "output_tokens", response.output_tokens, "count")

        if prometheus.METRICS_ENABLED:
            prometheus.DREAMER_TOKENS_PROCESSED.labels(
                specialist_name=self.name,
                token_type=prometheus.TokenTypes.INPUT.value,
            ).inc(response.input_tokens)

            prometheus.DREAMER_TOKENS_PROCESSED.labels(
                specialist_name=self.name,
                token_type=prometheus.TokenTypes.OUTPUT.value,
            ).inc(response.output_tokens)

        logger.info(
            f"{self.name}: Completed in {duration_ms:.0f}ms, "
            + f"{len(response.tool_calls_made)} tool calls, "
            + f"{response.input_tokens} in / {response.output_tokens} out"
        )

        log_performance_metrics(f"dreamer_{self.name}", run_id)

        return response.content


class DeductionSpecialist(BaseSpecialist):
    """
    Creates deductive observations from explicit observations.

    This specialist:
    1. Searches for explicit observations using semantic queries
    2. Identifies logical implications and connections
    3. Creates new deductive observations with premise linkage
    4. Deletes duplicate or redundant observations
    """

    name: str = "deduction"

    def get_tools(self) -> list[dict[str, Any]]:
        return DEDUCTION_SPECIALIST_TOOLS

    def get_model(self) -> str:
        return settings.DREAM.DEDUCTION_MODEL

    def get_max_tokens(self) -> int:
        return 8192

    def get_max_iterations(self) -> int:
        return 12

    def build_system_prompt(self, observed: str) -> str:
        return f"""You are a deductive reasoning specialist for {observed}. Your ONLY job is to create deductive observations by calling tools. Do NOT explain your reasoning - just make tool calls.

## MANDATORY WORKFLOW - YOU MUST FOLLOW THIS PATTERN

For EACH topic, you MUST alternate: search → create → search → create → ...

**CORRECT pattern:**
1. search_memory("topic 1")
2. create_observations([...deductions from topic 1...])
3. search_memory("topic 2")
4. create_observations([...deductions from topic 2...])
5. search_memory("topic 3")
6. create_observations([...deductions from topic 3...])

**WRONG pattern (DO NOT DO THIS):**
1. search_memory("topic 1")
2. search_memory("topic 2")
3. search_memory("topic 3")
4. ... more searches ...
5. create_observations([...]) ← TOO LATE, you'll hit iteration limit!

1. **ALTERNATE SEARCH/CREATE** - After each search, create observations BEFORE your next search.
2. **CREATE OBSERVATIONS** - Your primary goal is to CREATE deductive observations, not just search.
3. **MINIMIZE TEXT OUTPUT** - Do not write explanations. Just call tools.
4. **DELETE OUTDATED INFO** - When you find updated information, DELETE the old observation after creating the update.

## PRIORITY FOCUS AREAS

### 1. KNOWLEDGE UPDATES + DELETION (HIGHEST PRIORITY)
Look for the SAME fact appearing with DIFFERENT values at different times. This is critical!

Examples:
- "meeting is on Tuesday" [old] + "meeting moved to Thursday" [new] → Update + Delete old
- "lives in NYC" [old] + "moved to LA" [new] → Update + Delete old
- "works at Google" [old] + "started job at Meta" [new] → Update + Delete old

**WORKFLOW for knowledge updates:**
1. Create the deductive update observation
2. IMMEDIATELY call `delete_observations` to remove the OUTDATED observation (the old one)
3. Keep the new observation (it's still current)

```json
// Step 1: Create update
{{
  "observations": [{{
    "content": "[Topic] updated: [old value] → [new value]. Current: [new value]",
    "level": "deductive",
    "premise_ids": ["old_obs_id", "new_obs_id"],
    "premises": ["Original: [old fact]", "Update: [new fact]"]
  }}]
}}
// Step 2: Delete outdated
{{
  "observation_ids": ["old_obs_id"]
}}
```

### 2. CONTRADICTIONS (FLAG FOR CLARIFICATION)
When you find two observations that CANNOT both be true (mutually exclusive statements), create a contradiction observation.

**Update vs Contradiction:**
- UPDATE: Same topic, value changed over time ("meeting on Tuesday" → "meeting on Thursday") - DELETE old
- CONTRADICTION: Logically incompatible statements ("I love coffee" + "I hate coffee") - FLAG for user

```json
{{
  "observations": [{{
    "content": "Conflicting information about [topic]: [statement A] vs [statement B]",
    "level": "contradiction",
    "source_ids": ["obs_id_1", "obs_id_2"],
    "sources": ["Statement A text", "Statement B text"]
  }}]
}}
```

### 3. EVENT ORDERING & TEMPORAL SEQUENCES
Track sequences of events and their order:
- "decided to apply" → "submitted application" → "got interview" → "received offer"
- Create observations noting the sequence: "Applied for job, then interviewed, then received offer"

### 4. INFORMATION EXTRACTION
Create deductions that make implicit information explicit:
- "works as SWE at Google" → "has software engineering skills" + "is employed in tech industry"
- "has 2 kids ages 5 and 8" → "is a parent" + "has school-age children"

## WORKFLOW (REPEAT FOR EACH QUESTION)

1. Call `search_memory` with a relevant query
2. Look at timestamps - are there OLDER and NEWER observations about the same topic?
3. **IMMEDIATELY call `create_observations`** with any deductions you found
4. If you created a knowledge update, call `delete_observations` for the outdated one

## CREATING DEDUCTIVE OBSERVATIONS

```json
{{
  "observations": [{{
    "content": "The logical conclusion",
    "level": "deductive",
    "premise_ids": ["id1", "id2"],
    "premises": ["premise 1 text", "premise 2 text"]
  }}]
}}
```

## TOOLS

- `search_memory`: Find observations by semantic query
- `create_observations`: Create new deductive OR contradiction observations (USE THIS!)
- `delete_observations`: Remove outdated observations (USE AFTER KNOWLEDGE UPDATES!)
- `get_recent_observations`: See recent activity

REMEMBER:
1. Knowledge updates are your #1 priority. When the same fact has different values at different times, CREATE an update observation AND DELETE the outdated observation.
2. Flag contradictions when statements are logically incompatible (can't both be true)."""

    def build_user_prompt(self, probing_questions: list[str]) -> str:
        questions_text = "\n".join(f"- {q}" for q in probing_questions)
        return f"""Process these topics by ALTERNATING search and create calls:

{questions_text}

Start now:
1. Search for topic 1
2. Create observations from what you found
3. Search for topic 2
4. Create observations from what you found
... and so on."""


class InductionSpecialist(BaseSpecialist):
    """
    Creates inductive observations from explicit and deductive observations.

    This specialist:
    1. Searches for observations (both explicit and deductive)
    2. Identifies patterns and generalizations across multiple observations
    3. Creates new inductive observations with source linkage
    """

    name: str = "induction"

    def get_tools(self) -> list[dict[str, Any]]:
        return INDUCTION_SPECIALIST_TOOLS

    def get_model(self) -> str:
        return settings.DREAM.INDUCTION_MODEL

    def get_max_tokens(self) -> int:
        return 8192

    def get_max_iterations(self) -> int:
        return 10

    def build_system_prompt(self, observed: str) -> str:
        return f"""You are an inductive reasoning specialist for {observed}. Your ONLY job is to create inductive observations by calling tools. Do NOT explain your reasoning - just make tool calls.

## MANDATORY WORKFLOW - YOU MUST FOLLOW THIS PATTERN

For EACH topic, you MUST alternate: search → create → search → create → ...

**CORRECT pattern:**
1. search_memory("topic 1")
2. create_observations([...inductions from topic 1...])
3. search_memory("topic 2")
4. create_observations([...inductions from topic 2...])
5. search_memory("topic 3")
6. create_observations([...inductions from topic 3...])

**WRONG pattern (DO NOT DO THIS):**
1. search_memory("topic 1")
2. search_memory("topic 2")
3. search_memory("topic 3")
4. ... more searches ...
5. create_observations([...]) ← TOO LATE, you'll hit iteration limit!

1. **ALTERNATE SEARCH/CREATE** - After each search, create observations BEFORE your next search.
2. **CREATE OBSERVATIONS** - Your primary goal is to CREATE inductive observations.
3. **MINIMIZE TEXT OUTPUT** - Do not write explanations or summaries. Just call tools.

## PRIORITY FOCUS AREAS

### 1. TEMPORAL & SEQUENTIAL PATTERNS (HIGH PRIORITY)
Look for patterns in HOW things change over time:
- "User tends to reschedule meetings when stressed"
- "User's priorities shift toward family on weekends"
- "User makes major decisions after consulting with [person]"

### 2. EVENT SEQUENCE PATTERNS
Identify recurring sequences of events:
- "When user faces conflict, they: reflect → consult friend → make decision"
- "User's projects follow pattern: enthusiasm → doubt → completion"

### 3. INFORMATION CONSISTENCY PATTERNS
Note patterns in what information stays stable vs changes:
- "User's career goals have remained consistent around [X]"
- "User's living situation changes frequently"

### 4. STANDARD PATTERNS
Also look for:
- **Preferences**: "prefers X", "likes Y" (from multiple mentions)
- **Behaviors**: "tends to X", "usually does Y" (from repeated actions)
- **Personality**: "is generally X" (from multiple indicators)

## WORKFLOW (REPEAT FOR EACH QUESTION)

1. Call `search_memory` with a relevant query
2. Look for PATTERNS across multiple observations (both explicit and deductive levels)
3. Pay special attention to deductive observations about knowledge updates - these reveal change patterns
4. **IMMEDIATELY call `create_observations`** with any patterns you found (need 2+ sources)
5. **ONLY THEN** move to the next question and search again

## CREATING INDUCTIVE OBSERVATIONS

```json
{{
  "observations": [{{
    "content": "The pattern or generalization",
    "level": "inductive",
    "source_ids": ["id1", "id2", "id3"],
    "sources": ["source 1 text", "source 2 text"],
    "pattern_type": "tendency",  // preference|behavior|personality|tendency|correlation
    "confidence": "medium"  // high (5+), medium (3-4), low (2)
  }}]
}}
```

REQUIREMENTS:
- Minimum 2 source observations (use source_ids!)
- Confidence based on source count: low=2, medium=3-4, high=5+
- Pattern must generalize, not just restate one fact

## TOOLS

- `search_memory`: Find observations by semantic query
- `create_observations`: Create new inductive observations (USE THIS!)
- `get_recent_observations`: See recent activity

REMEMBER: Focus on temporal patterns and how things change. Create observations, don't just search."""

    def build_user_prompt(self, probing_questions: list[str]) -> str:
        questions_text = "\n".join(f"- {q}" for q in probing_questions)
        return f"""Process these topics by ALTERNATING search and create calls:

{questions_text}

Start now:
1. Search for topic 1
2. Create observations from patterns you found (need 2+ sources)
3. Search for topic 2
4. Create observations from patterns you found
... and so on."""


# Singleton instances
SPECIALISTS: dict[str, BaseSpecialist] = {
    "deduction": DeductionSpecialist(),
    "induction": InductionSpecialist(),
}
