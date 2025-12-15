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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.utils.agent_tools import TOOLS
from src.utils.clients import HonchoLLMCallResponse, honcho_llm_call

if TYPE_CHECKING:
    from src.dreamer.prescan import DreamContext


@dataclass
class ToolExecutorParams:
    """Parameters for creating tool executors - used by consolidation specialist."""

    db: AsyncSession
    workspace_name: str
    observer: str
    observed: str
    session_name: str
    history_token_limit: int


logger = logging.getLogger(__name__)


def _get_settings_with_model(model: str):
    """Create a copy of DREAM settings with a different model."""
    # Create a new settings object with the overridden model
    # We use model_copy to clone and override
    return settings.DREAM.model_copy(update={"MODEL": model})


# Tool sets for each specialist - minimal, focused
DEDUCTION_TOOLS = [TOOLS["create_observations"]]
INDUCTION_TOOLS = [TOOLS["create_observations"]]
CONSOLIDATION_TOOLS = [TOOLS["create_vignette"]]


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
            + f"(prompt: {prompt_len} chars, model: {model})"
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
            + f"{response.input_tokens} in, {response.output_tokens} out"
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
                _, content, tool_calls, _ = result
                total_tool_calls += tool_calls
                all_results.append(content)

        summary = (
            f"{self.name}: Processed {total_batches} batch(es) in parallel, "
            f"{total_tool_calls} total tool calls"
        )
        logger.info(summary)

        return summary


class DeductionSpecialist(BaseSpecialist):
    """
    Processes explicit observations to create deductions AND handle temporal reasoning.

    This specialist performs two key functions:
    1. LOGICAL INFERENCE: Derive new facts from combinations of explicit observations
    2. TEMPORAL REASONING: Detect when facts have changed over time and create
       knowledge update observations that note the supersession

    Uses semantic clustering (from prescan) to group related observations,
    then examines each cluster with full message context to enable the LLM
    to understand temporal relationships naturally.
    """

    name: str = "deduction"

    def get_tools(self) -> list[dict[str, Any]]:
        return DEDUCTION_TOOLS

    def get_model(self) -> str:
        return settings.DREAM.DEDUCTION_MODEL

    def get_max_tokens(self) -> int:
        return 8192  # More output for detailed reasoning

    def get_max_iterations(self) -> int:
        return 5  # May need multiple tool calls for complex clusters

    def get_batches(self, context: DreamContext) -> list[Any]:
        """
        Create batches from semantic clusters of observations with context.

        Each batch contains a cluster of semantically related observations
        along with their source message context for temporal reasoning.
        """
        # Use pattern clusters as our semantic groupings
        clusters = context.pattern_clusters
        if not clusters:
            # Fall back to processing all explicit observations as one batch
            if not context.explicit_with_context:
                return []
            return [
                {
                    "observations_with_context": context.explicit_with_context,
                    "cluster_theme": "all observations",
                    "existing_deductive": context.deductive_observations,
                }
            ]

        # Build a lookup from observation ID to ObservationWithContext
        obs_context_lookup: dict[str, Any] = {
            owc.observation.id: owc for owc in context.explicit_with_context
        }

        batches: list[Any] = []
        for cluster in clusters:
            # Get ObservationWithContext for each observation in the cluster
            cluster_with_context = [
                obs_context_lookup.get(obs.id)
                for obs in cluster.observations
                if obs.id in obs_context_lookup
            ]
            # Filter out None values
            cluster_with_context = [
                owc for owc in cluster_with_context if owc is not None
            ]

            if cluster_with_context:
                batches.append(
                    {
                        "observations_with_context": cluster_with_context,
                        "cluster_theme": cluster.theme,
                        "existing_deductive": context.deductive_observations,
                    }
                )

        return batches

    def _format_observation_with_context(self, owc: Any) -> str:
        """Format a single observation with its message context."""
        from src.dreamer.prescan import ObservationWithContext

        if not isinstance(owc, ObservationWithContext):
            return f"- [id:{owc.id}] {owc.content}"

        obs = owc.observation
        lines = [f"**Observation [id:{obs.id}]:**"]
        lines.append(f"  Content: {obs.content}")
        lines.append(f"  Timestamp: {obs.created_at}")

        if owc.source_message:
            msg = owc.source_message
            lines.append(
                f'  Source message [{msg.peer_name}]: "{msg.content[:500]}{"..." if len(msg.content) > 500 else ""}"'
            )

        if owc.preceding_message:
            msg = owc.preceding_message
            lines.append(
                f'  Preceding message [{msg.peer_name}]: "{msg.content[:300]}{"..." if len(msg.content) > 300 else ""}"'
            )

        return "\n".join(lines)

    def build_batch_prompt(
        self,
        context: DreamContext,
        observed: str,
        batch_num: int,
        total_batches: int,
        batch_data: Any,
    ) -> str:
        observations_with_context = batch_data["observations_with_context"]
        cluster_theme = batch_data["cluster_theme"]
        existing = batch_data["existing_deductive"]

        # Format observations with full context
        observations_text = "\n\n".join(
            self._format_observation_with_context(owc)
            for owc in observations_with_context
        )

        existing_text = (
            "\n".join(
                f"- {doc.content}" for doc in existing[:20]
            )  # Limit existing to avoid overflow
            if existing
            else "(none yet)"
        )

        batch_info = (
            f"[Batch {batch_num}/{total_batches}] " if total_batches > 1 else ""
        )

        return f"""{batch_info}Create DEDUCTIVE observations about {observed}.

You are analyzing a cluster of semantically related observations about "{cluster_theme}".
Each observation includes its source message context and timestamp for temporal reasoning.

## Observations in this Cluster ({len(observations_with_context)} observations)

{observations_text}

## Existing Deductive Observations (DO NOT duplicate)
{existing_text}

## Your Tasks

### 1. TEMPORAL REASONING (Knowledge Updates)
Look for observations in this cluster that describe the SAME attribute/fact with DIFFERENT values at DIFFERENT times.

When you find a temporal update:
- The LATER observation SUPERSEDES the earlier one
- Create a deductive observation noting the update
- Format: "[Topic] changed: [old value] → [new value] (as of [newer timestamp])"
- Include `premise_ids` linking BOTH the old and new observations

### 2. LOGICAL INFERENCE
Create deductions from combinations of facts - things that logically follow but weren't stated directly.

Focus on HIGH-VALUE inferences:
- Skills and expertise (e.g., "works as SWE at Google" → "has software engineering skills")
- Background implications (e.g., "walked her dog" → "has a pet dog")
- Relationship dynamics
- Preference patterns

### Output Requirements

For EACH deductive observation, include:
- `content`: The deductive statement
- `level`: "deductive"
- `premise_ids`: Array of observation IDs this is derived from (REQUIRED)
- `premises`: Human-readable source text

Call create_observations once with all deductions."""


class ConsolidationSpecialist(BaseSpecialist):
    """
    Consolidates clusters of explicit observations into single vignettes.

    This specialist runs FIRST in the dream cycle (hardcoded, not selected by orchestrator).
    For each cluster of semantically related explicit observations, it creates a single
    vignette that contains ALL the explicit information in a coherent narrative,
    then deletes the source observations.

    This reduces the number of observations while preserving all information.

    NOTE: This specialist uses a custom run() method that accepts ToolExecutorParams
    instead of a pre-built tool_executor, so it can create batch-specific executors
    with the source_ids for each cluster.
    """

    name: str = "consolidation"

    def get_tools(self) -> list[dict[str, Any]]:
        return CONSOLIDATION_TOOLS

    def get_model(self) -> str:
        # Use the same model as deduction for consistency
        return settings.DREAM.DEDUCTION_MODEL

    def get_max_tokens(self) -> int:
        return 4096  # Vignettes are concise narratives

    def get_max_iterations(self) -> int:
        return 1  # Single tool call per batch

    def get_batches(self, context: DreamContext) -> list[Any]:
        """
        Create one batch per pattern cluster.

        Each batch contains a cluster of semantically related explicit observations
        to be consolidated into a single vignette.
        """
        clusters = context.pattern_clusters
        if not clusters:
            return []

        batches: list[Any] = []
        for cluster in clusters:
            # Only consolidate clusters with explicit observations
            explicit_obs = [
                obs for obs in cluster.observations if obs.level == "explicit"
            ]
            if len(explicit_obs) >= 2:  # Only consolidate if 2+ observations
                batches.append(
                    {
                        "observations": explicit_obs,
                        "cluster_theme": cluster.theme,
                        "source_ids": [obs.id for obs in explicit_obs],
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
        observations = batch_data["observations"]
        cluster_theme = batch_data["cluster_theme"]

        # Format observations (no need to show IDs since they're handled automatically)
        observations_text = "\n".join(f"- {obs.content}" for obs in observations)

        batch_info = (
            f"[Batch {batch_num}/{total_batches}] " if total_batches > 1 else ""
        )

        return f"""{batch_info}Consolidate these explicit observations about {observed} into a single vignette.

## Cluster Theme: "{cluster_theme}"

## Observations to Consolidate ({len(observations)} observations)
{observations_text}

## Task
Create a single vignette that:
1. Contains ALL the explicit facts from these observations
2. Presents them as a coherent narrative or organized summary
3. Does NOT add any inferences or interpretations - only explicit facts
4. Preserves important details like dates, names, and specific information

Call `create_vignette` with your consolidated content. The source observations will be automatically deleted.

The vignette should be comprehensive but concise. Every fact from the source observations must be included."""

    async def run_with_params(
        self,
        context: DreamContext,
        observed: str,
        executor_params: ToolExecutorParams,
    ) -> str:
        """
        Process ALL data exhaustively, creating batch-specific tool executors.

        This override allows us to inject source_ids into each batch's tool executor.

        Args:
            context: Complete pre-scanned dream context
            observed: The peer being observed
            executor_params: Parameters to create tool executors for each batch

        Returns:
            Summary of all work done across all batches
        """
        from src.utils.agent_tools import create_tool_executor

        batches = self.get_batches(context)

        if not batches:
            logger.info(f"{self.name}: No data to process")
            return f"{self.name}: No data to process"

        total_batches = len(batches)
        logger.info(f"{self.name}: Processing {total_batches} batch(es) in parallel")

        # Create tasks for all batches, each with its own tool executor
        async def process_batch_with_source_ids(
            batch_num: int, batch_data: Any
        ) -> tuple[int, str, int, int]:
            # Create a batch-specific tool executor with source_ids
            tool_executor = create_tool_executor(
                db=executor_params.db,
                workspace_name=executor_params.workspace_name,
                observer=executor_params.observer,
                observed=executor_params.observed,
                session_name=executor_params.session_name,
                include_observation_ids=True,
                history_token_limit=executor_params.history_token_limit,
                vignette_source_ids=batch_data["source_ids"],
            )
            return await self._process_batch(
                context, observed, batch_num, total_batches, batch_data, tool_executor
            )

        tasks = [
            process_batch_with_source_ids(batch_num, batch_data)
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
                _, content, tool_calls, _ = result
                total_tool_calls += tool_calls
                all_results.append(content)

        summary = (
            f"{self.name}: Processed {total_batches} batch(es) in parallel, "
            f"{total_tool_calls} total tool calls"
        )
        logger.info(summary)

        return summary


class InductionSpecialist(BaseSpecialist):
    """Processes top pattern clusters to create inductive observations."""

    name: str = "induction"

    # Limit to top N clusters (sorted by size in prescan)
    MAX_CLUSTERS: int = 10

    def get_tools(self) -> list[dict[str, Any]]:
        return INDUCTION_TOOLS

    def get_model(self) -> str:
        return settings.DREAM.INDUCTION_MODEL

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
- **Tendencies**: "frequently mentions", "regularly discusses", "commonly references"

For each inductive observation, include (ALL REQUIRED):
- content: The pattern description
- level: "inductive"
- source_ids: IDs of observations that support the pattern
- sources: Human-readable source text
- pattern_type: preference/behavior/personality/tendency
- confidence: high (5+ sources), medium (3-4 sources), low (2 sources)

Create one inductive observation per valid pattern found."""


# Singleton instances
SPECIALISTS: dict[str, BaseSpecialist] = {
    "consolidation": ConsolidationSpecialist(),
    "deduction": DeductionSpecialist(),
    "induction": InductionSpecialist(),
}
