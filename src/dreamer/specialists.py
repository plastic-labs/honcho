"""
Specialist batch processors for exhaustive dreaming.

Each specialist focuses on ONE reasoning type and processes ALL provided data.
No exploration tools - just pure reasoning over pre-computed context.

If data exceeds context limits, specialists process in batches and aggregate results.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from src.config import settings
from src.utils.agent_tools import TOOLS
from src.utils.clients import HonchoLLMCallResponse, honcho_llm_call

if TYPE_CHECKING:
    from src.dreamer.prescan import DreamContext

logger = logging.getLogger(__name__)


def _get_settings_with_model(model: str):
    """Create a copy of DREAM settings with a different model."""
    # Create a new settings object with the overridden model
    # We use model_copy to clone and override
    return settings.DREAM.model_copy(update={"MODEL": model})


# Tool sets for each specialist - minimal, focused
KNOWLEDGE_UPDATE_TOOLS = [TOOLS["create_observations"]]
DEDUCTION_TOOLS = [TOOLS["create_observations"]]
INDUCTION_TOOLS = [TOOLS["create_observations"]]
CONSOLIDATION_TOOLS = [TOOLS["delete_observations"], TOOLS["update_peer_card"]]

# Approximate tokens per observation (content + ID + formatting)
TOKENS_PER_OBSERVATION = 50
# Leave room for system prompt, response, etc.
MAX_CONTEXT_TOKENS = 180_000
OBSERVATIONS_PER_BATCH = MAX_CONTEXT_TOKENS // TOKENS_PER_OBSERVATION


class BaseSpecialist(ABC):
    """Base class for specialist batch processors."""

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
        """Get max tool iterations per batch."""
        return 10

    @abstractmethod
    def build_batch_prompt(
        self,
        context: DreamContext,
        observed: str,
        batch_num: int,
        total_batches: int,
        batch_data: Any,
    ) -> str:
        """Build prompt for a single batch."""
        ...

    @abstractmethod
    def get_batches(self, context: DreamContext) -> list[Any]:
        """Split context into batches for processing."""
        ...

    async def _process_batch(
        self,
        context: DreamContext,
        observed: str,
        batch_num: int,
        total_batches: int,
        batch_data: Any,
        tool_executor: Callable[[str, dict[str, Any]], Any],
    ) -> tuple[int, str, int, int]:
        """Process a single batch and return results."""
        prompt = self.build_batch_prompt(
            context, observed, batch_num, total_batches, batch_data
        )

        prompt_len = len(prompt)
        model = self.get_model()
        logger.info(
            f"{self.name}: Processing batch {batch_num}/{total_batches} "
            f"(prompt: {prompt_len} chars, model: {model})"
        )

        # Use specialist's model instead of default DREAM model
        llm_settings = _get_settings_with_model(model)

        response: HonchoLLMCallResponse[str] = await honcho_llm_call(
            llm_settings=llm_settings,
            prompt=prompt,
            max_tokens=self.get_max_tokens(),
            tools=self.get_tools(),
            tool_executor=tool_executor,
            max_tool_iterations=self.get_max_iterations(),
            track_name=f"Dreamer/{self.name}/batch_{batch_num}",
        )

        logger.info(
            f"{self.name} batch {batch_num}: {len(response.tool_calls_made)} tool calls, "
            f"{response.input_tokens} in, {response.output_tokens} out"
        )

        return (
            batch_num,
            response.content,
            len(response.tool_calls_made),
            response.input_tokens,
        )

    async def run(
        self,
        context: DreamContext,
        observed: str,
        tool_executor: Callable[[str, dict[str, Any]], Any],
    ) -> str:
        """
        Process ALL data exhaustively, running batches in parallel.

        Args:
            context: Complete pre-scanned dream context
            observed: The peer being observed
            tool_executor: Function to execute tools

        Returns:
            Summary of all work done across all batches
        """
        batches = self.get_batches(context)

        if not batches:
            logger.info(f"{self.name}: No data to process")
            return f"{self.name}: No data to process"

        total_batches = len(batches)
        logger.info(f"{self.name}: Processing {total_batches} batch(es) in parallel")

        # Create tasks for all batches
        tasks = [
            self._process_batch(
                context, observed, batch_num, total_batches, batch_data, tool_executor
            )
            for batch_num, batch_data in enumerate(batches, start=1)
        ]

        # Run all batches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        total_tool_calls = 0
        all_results: list[str] = []

        for result in results:
            if isinstance(result, BaseException):
                logger.error(f"{self.name}: Batch failed with error: {result}")
                all_results.append(f"ERROR: {result}")
            else:
                batch_num, content, tool_calls, _ = result
                total_tool_calls += tool_calls
                all_results.append(content)

        summary = (
            f"{self.name}: Processed {total_batches} batch(es) in parallel, "
            f"{total_tool_calls} total tool calls"
        )
        logger.info(summary)

        return summary


class KnowledgeUpdateSpecialist(BaseSpecialist):
    """Processes ALL knowledge update candidates."""

    name: str = "knowledge_update"

    def get_tools(self) -> list[dict[str, Any]]:
        return KNOWLEDGE_UPDATE_TOOLS

    def get_model(self) -> str:
        return "anthropic/claude-haiku-4.5"  # Simple verification task

    def get_max_iterations(self) -> int:
        return 2  # Usually just 1 tool call needed

    def get_batches(self, context: DreamContext) -> list[Any]:
        """Each batch is a list of knowledge update candidates."""
        candidates = context.knowledge_update_candidates
        if not candidates:
            return []

        # Batch by number of candidates (each takes ~200 tokens)
        batch_size = 50
        return [
            candidates[i : i + batch_size]
            for i in range(0, len(candidates), batch_size)
        ]

    def build_batch_prompt(
        self,
        context: DreamContext,
        observed: str,
        batch_num: int,
        total_batches: int,
        batch_data: Any,
    ) -> str:
        candidates = batch_data

        candidates_text = "\n\n".join(
            f"**Candidate {i + 1}:**\n"
            f"- OLD [id:{c.old_observation.id}]: {c.old_observation.content}\n"
            f"  (Created: {c.old_observation.created_at})\n"
            f"- NEW [id:{c.new_observation.id}]: {c.new_observation.content}\n"
            f"  (Created: {c.new_observation.created_at})\n"
            f"- Topic: {c.topic}\n"
            f"- Similarity: {c.similarity:.2f}"
            for i, c in enumerate(candidates)
        )

        batch_info = (
            f"[Batch {batch_num}/{total_batches}] " if total_batches > 1 else ""
        )

        return f"""{batch_info}You verify knowledge updates about {observed}.

Each candidate below shows an OLD observation and a NEW observation about the same topic.
Your job is to identify TRUE knowledge updates where a fact has CHANGED.

## Candidates ({len(candidates)} in this batch, {len(context.knowledge_update_candidates)} total)
{candidates_text}

## Task
For each TRUE knowledge update (same fact, different value, later timestamp):
1. Create a DEDUCTIVE observation noting the update
2. Include premise_ids linking to BOTH old and new observations
3. Content format: "[Topic] was updated from [old value] to [new value] ([new value] supersedes)"

Only create observations for TRUE updates. Skip if:
- Different facts (not actually the same topic)
- Same value (no actual change)
- Uncertain which is newer

Process ALL {len(candidates)} candidates in this batch. Create observations for every valid update."""


class DeductionSpecialist(BaseSpecialist):
    """Processes ALL explicit observations to create deductions."""

    name: str = "deduction"

    def get_tools(self) -> list[dict[str, Any]]:
        return DEDUCTION_TOOLS

    def get_model(self) -> str:
        return "anthropic/claude-haiku-4.5"

    def get_max_tokens(self) -> int:
        return 4096  # Limit output to force conciseness

    def get_max_iterations(self) -> int:
        return 3  # Each batch is small, 1-2 tool calls expected

    def get_batches(self, context: DreamContext) -> list[Any]:
        """Each batch is a subset of explicit observations."""
        explicit = context.explicit_observations
        if not explicit:
            return []

        # Batch explicit observations - 50 per batch for faster processing
        # Each observation ~50-100 tokens, so 50 obs = ~2.5-5K tokens input
        batch_size = 50
        batches = []
        for i in range(0, len(explicit), batch_size):
            batches.append(
                {
                    "explicit": explicit[i : i + batch_size],
                    "existing_deductive": context.deductive_observations,
                }
            )
        return batches

    def build_batch_prompt(
        self,
        context: DreamContext,
        observed: str,
        batch_num: int,
        total_batches: int,
        batch_data: Any,
    ) -> str:
        explicit = batch_data["explicit"]
        existing = batch_data["existing_deductive"]

        explicit_text = "\n".join(f"- [id:{doc.id}] {doc.content}" for doc in explicit)

        existing_text = (
            "\n".join(f"- {doc.content}" for doc in existing)
            if existing
            else "(none yet)"
        )

        batch_info = (
            f"[Batch {batch_num}/{total_batches}] " if total_batches > 1 else ""
        )

        return f"""{batch_info}Create DEDUCTIVE observations about {observed}.

## Explicit Observations
{explicit_text}

## Existing Deductive (DO NOT duplicate)
{existing_text}

## Task
Create deductive observations - logical inferences from the explicit facts.
- Include premise_ids linking to source observations
- Include premises as human-readable text
- Focus on HIGH-VALUE inferences only (skills, background, relationships, preferences)
- Skip trivial or speculative deductions

Call create_observations once with all deductions."""


class InductionSpecialist(BaseSpecialist):
    """Processes top pattern clusters to create inductive observations."""

    name: str = "induction"

    # Limit to top N clusters (sorted by size in prescan)
    MAX_CLUSTERS = 10

    def get_tools(self) -> list[dict[str, Any]]:
        return INDUCTION_TOOLS

    def get_model(self) -> str:
        return "anthropic/claude-haiku-4.5"  # Pattern matching task

    def get_max_iterations(self) -> int:
        return 2  # Usually just 1 tool call needed

    def get_batches(self, context: DreamContext) -> list[Any]:
        """Single batch with top 10 clusters."""
        clusters = context.pattern_clusters
        if not clusters:
            return []

        # Take only top N clusters (already sorted by size in prescan)
        top_clusters = clusters[: self.MAX_CLUSTERS]

        # Single batch with all top clusters
        return [
            {
                "clusters": top_clusters,
                "existing_inductive": context.inductive_observations,
            }
        ]

    def build_batch_prompt(
        self,
        context: DreamContext,
        observed: str,
        batch_num: int,
        total_batches: int,
        batch_data: Any,
    ) -> str:
        clusters = batch_data["clusters"]
        existing = batch_data["existing_inductive"]

        clusters_text = "\n\n".join(
            f"**Cluster: '{c.theme}' ({c.count} observations)**\n"
            + "\n".join(f"  - [id:{doc.id}] {doc.content}" for doc in c.observations)
            for c in clusters
        )

        existing_text = (
            "\n".join(f"- {doc.content}" for doc in existing)
            if existing
            else "(none yet)"
        )

        batch_info = (
            f"[Batch {batch_num}/{total_batches}] " if total_batches > 1 else ""
        )

        return f"""{batch_info}You create INDUCTIVE observations about {observed}.

Inductions are patterns and generalizations from multiple observations.
Your job is to identify meaningful patterns in the top clusters.

## Top Pattern Clusters ({len(clusters)} clusters)
{clusters_text}

## Existing Inductive Observations ({len(existing)} total - DO NOT duplicate these)
{existing_text}

## Task
For each cluster, identify if there's a generalizable pattern:
- **Preferences**: "prefers X", "likes Y", "favors Z"
- **Behaviors**: "tends to Z", "usually does W", "often X"
- **Personality**: "is methodical", "values structure", "approaches problems analytically"
- **Tendencies**: "frequently mentions", "regularly discusses", "commonly references"

For each inductive observation, include (ALL REQUIRED):
- content: The pattern description
- level: "inductive"
- source_ids: IDs of observations that support the pattern
- sources: Human-readable source text
- pattern_type: preference/behavior/personality/tendency
- confidence: high (5+ sources), medium (3-4 sources), low (2 sources)

Create one inductive observation per valid pattern found."""


class ConsolidationSpecialist(BaseSpecialist):
    """Processes ALL duplicate candidates and updates peer card."""

    name: str = "consolidation"

    def get_tools(self) -> list[dict[str, Any]]:
        return CONSOLIDATION_TOOLS

    def get_model(self) -> str:
        return "anthropic/claude-haiku-4.5"

    def get_max_iterations(self) -> int:
        return 15

    def get_batches(self, context: DreamContext) -> list[Any]:
        """
        First batch: All duplicates to delete
        Last batch: Peer card update with all key facts
        """
        batches = []

        # Batch duplicates
        duplicates = context.duplicate_candidates
        if duplicates:
            batch_size = 50
            for i in range(0, len(duplicates), batch_size):
                batches.append(
                    {
                        "type": "duplicates",
                        "duplicates": duplicates[i : i + batch_size],
                    }
                )

        # Always add peer card update as final batch
        batches.append(
            {
                "type": "peer_card",
                "context": context,
            }
        )

        return batches

    def build_batch_prompt(
        self,
        context: DreamContext,
        observed: str,
        batch_num: int,
        total_batches: int,
        batch_data: Any,
    ) -> str:
        batch_info = (
            f"[Batch {batch_num}/{total_batches}] " if total_batches > 1 else ""
        )

        if batch_data["type"] == "duplicates":
            duplicates = batch_data["duplicates"]
            duplicates_text = "\n".join(
                f'- [id:{d.doc_a.id}] "{d.doc_a.content}"\n'
                f'  vs [id:{d.doc_b.id}] "{d.doc_b.content}"\n'
                f"  (similarity: {d.similarity:.3f})"
                for d in duplicates
            )

            return f"""{batch_info}You clean up duplicate observations about {observed}.

## Duplicate Candidates ({len(duplicates)} in this batch, {len(context.duplicate_candidates)} total)
{duplicates_text}

## Task
Delete EXACT duplicates - observations that express the SAME meaning.

Rules:
- Only delete if both observations say essentially the same thing
- When deleting, keep the MORE SPECIFIC or MORE RECENT observation
- NEVER delete contradictory observations - they represent real changes over time
- Use delete_observations with the ID(s) to delete

Process ALL {len(duplicates)} candidates. Delete every true duplicate."""

        else:  # peer_card
            peer_card_text = (
                "\n".join(f"- {fact}" for fact in context.peer_card)
                if context.peer_card
                else "(empty)"
            )
            key_facts = self._extract_key_facts(context)

            return f"""{batch_info}You update the peer card for {observed}.

The peer card is a summary of the most important facts about this person.

## Current Peer Card
{peer_card_text}

## Key Facts from Observations
{key_facts}

## Task
Update the peer card with:
1. Key biographical facts (name, job, location, family, etc.)
2. Standing instructions the user has given (format: "INSTRUCTION: ...")
3. Important preferences (format: "PREFERENCE: ...")
4. Notable personality traits (format: "TRAIT: ...")

Keep it concise but comprehensive. Include everything important about {observed}."""

    def _extract_key_facts(self, context: DreamContext) -> str:
        """Extract key biographical facts from ALL observations."""
        key_facts: list[str] = []

        bio_keywords = [
            "name",
            "age",
            "born",
            "birthday",
            "lives",
            "location",
            "city",
            "country",
            "works",
            "job",
            "occupation",
            "career",
            "company",
            "role",
            "position",
            "married",
            "wife",
            "husband",
            "partner",
            "spouse",
            "children",
            "daughter",
            "son",
            "family",
            "parents",
            "mother",
            "father",
            "sibling",
            "brother",
            "sister",
            "education",
            "degree",
            "university",
            "college",
            "school",
            "studied",
            "hobby",
            "hobbies",
            "interests",
            "enjoys",
            "loves",
            "hates",
            "prefers",
            "always",
            "never",
            "usually",
            "typically",
        ]

        # Check explicit observations
        for doc in context.explicit_observations:
            content_lower = doc.content.lower()
            if any(kw in content_lower for kw in bio_keywords):
                key_facts.append(f"- {doc.content}")

        # Check inductive observations (patterns)
        for doc in context.inductive_observations:
            key_facts.append(f"- [pattern] {doc.content}")

        # Limit to prevent prompt overflow, but include a lot
        return "\n".join(key_facts[:100]) if key_facts else "(no key facts found)"


# Singleton instances
SPECIALISTS: dict[str, BaseSpecialist] = {
    "knowledge_update": KnowledgeUpdateSpecialist(),
    "deduction": DeductionSpecialist(),
    "induction": InductionSpecialist(),
    "consolidation": ConsolidationSpecialist(),
}
