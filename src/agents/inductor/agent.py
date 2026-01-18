"""Inductor agent for extracting patterns from unfalsified predictions."""

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.agents.inductor.config import InductorConfig
from src.agents.inductor.prompts import INDUCTOR_SYSTEM_PROMPT, INDUCTOR_TASK_PROMPT
from src.agents.shared.base_agent import BaseAgent
from src.config import settings
from src.utils.clients import HonchoLLMCallResponse, honcho_llm_call

logger = logging.getLogger(__name__)


class InductorAgent(BaseAgent):
    """Agent for extracting patterns from unfalsified predictions through induction."""

    def __init__(
        self,
        db: AsyncSession,
        config: InductorConfig | None = None,
    ):
        """Initialize the Inductor agent.

        Args:
            db: Database session
            config: Optional configuration override
        """
        super().__init__(db)
        self.config: InductorConfig = config or InductorConfig()
        logger.info(
            f"Initialized InductorAgent with min_predictions={self.config.min_predictions_per_pattern}"
        )

    def validate_input(self, input_data: dict[str, Any]) -> bool:
        """Validate input data for pattern extraction.

        Args:
            input_data: Dictionary with workspace_name, observer, observed

        Returns:
            True if valid

        Raises:
            ValueError: If required fields are missing or invalid
        """
        required_fields = ["workspace_name", "observer", "observed"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(input_data[field], str):
                raise ValueError(f"{field} must be a string")

        return True

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute pattern extraction from unfalsified predictions.

        Args:
            input_data: Dictionary with workspace_name, observer, observed

        Returns:
            Dictionary with:
                - inductions_created: Number of patterns extracted
                - induction_ids: List of Induction IDs created
                - predictions_analyzed: Number of predictions analyzed
                - clusters_found: Number of prediction clusters identified
                - reason: Optional reason if no inductions created
        """
        workspace_name = input_data["workspace_name"]
        observer = input_data["observer"]
        observed = input_data["observed"]

        # Step 1: Retrieve unfalsified predictions
        predictions = await self._retrieve_unfalsified_predictions(
            workspace_name, observer, observed
        )

        if len(predictions) < self.config.min_predictions_per_pattern:
            logger.info(
                f"Insufficient predictions ({len(predictions)}) for pattern extraction (min: {self.config.min_predictions_per_pattern})"
            )
            return {
                "inductions_created": 0,
                "induction_ids": [],
                "predictions_analyzed": len(predictions),
                "clusters_found": 0,
                "reason": "insufficient_predictions",
            }

        # Step 2: Cluster predictions by similarity
        clusters = await self._cluster_predictions(predictions)

        if not clusters:
            logger.info("No clusters found meeting minimum size requirement")
            return {
                "inductions_created": 0,
                "induction_ids": [],
                "predictions_analyzed": len(predictions),
                "clusters_found": 0,
                "reason": "no_clusters",
            }

        # Step 3: Extract patterns from clusters
        induction_ids = await self._extract_patterns(
            clusters, workspace_name, observer, observed
        )

        return {
            "inductions_created": len(induction_ids),
            "induction_ids": induction_ids,
            "predictions_analyzed": len(predictions),
            "clusters_found": len(clusters),
        }

    async def _retrieve_unfalsified_predictions(
        self,
        workspace_name: str,
        _observer: str,
        _observed: str,
    ) -> list[models.Prediction]:
        """Retrieve unfalsified predictions for pattern extraction.

        Args:
            workspace_name: Workspace name
            _observer: Observer peer name (unused - TODO: filter by observer)
            _observed: Observed peer name (unused - TODO: filter by observed)

        Returns:
            List of Prediction models with status="unfalsified"
        """
        # Get unfalsified predictions (predictions that survived falsification)
        stmt = await crud.prediction.list_predictions(
            workspace_name=workspace_name, status="unfalsified"
        )

        result = await self.db.execute(stmt)
        predictions = list(result.scalars().all())

        # TODO: Add observer/observed filtering to CRUD layer
        # For now, return limited number
        limited = predictions[: self.config.max_predictions_retrieval]

        logger.info(
            f"Retrieved {len(limited)} unfalsified predictions for pattern extraction"
        )
        return limited

    async def _cluster_predictions(
        self, predictions: list[models.Prediction]
    ) -> list[list[models.Prediction]]:
        """Cluster predictions by semantic similarity.

        Args:
            predictions: List of Prediction models

        Returns:
            List of prediction clusters (each cluster is a list of predictions)
        """
        # Simple clustering based on embedding similarity
        # TODO: Implement more sophisticated clustering (e.g., HNSW, K-means)

        clusters: list[list[models.Prediction]] = []
        used_indices: set[int] = set()

        for i, pred in enumerate(predictions):
            if i in used_indices:
                continue

            # Start new cluster with this prediction
            cluster = [pred]
            used_indices.add(i)

            # Find similar predictions
            for j, other_pred in enumerate(predictions):
                if j in used_indices or j <= i:
                    continue

                # Calculate cosine similarity between embeddings
                similarity = self._cosine_similarity(
                    pred.embedding, other_pred.embedding
                )

                if similarity >= self.config.similarity_threshold:
                    cluster.append(other_pred)
                    used_indices.add(j)

            # Only keep clusters meeting minimum size
            if len(cluster) >= self.config.min_predictions_per_pattern:
                clusters.append(cluster)

        logger.info(
            f"Found {len(clusters)} clusters meeting minimum size of {self.config.min_predictions_per_pattern}"
        )
        return clusters

    def _cosine_similarity(
        self, vec1: list[float], vec2: list[float]
    ) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    async def _extract_patterns(
        self,
        clusters: list[list[models.Prediction]],
        workspace_name: str,
        observer: str,
        observed: str,
    ) -> list[str]:
        """Extract inductive patterns from prediction clusters.

        Args:
            clusters: List of prediction clusters
            workspace_name: Workspace name
            observer: Observer peer name
            observed: Observed peer name

        Returns:
            List of Induction IDs created
        """
        induction_ids: list[str] = []

        # Process clusters up to max_inductions_per_run
        for cluster in clusters[: self.config.max_inductions_per_run]:
            logger.info(f"Extracting pattern from cluster of {len(cluster)} predictions")

            # Get hypothesis IDs for context
            hypothesis_ids = list({pred.hypothesis_id for pred in cluster})

            # Retrieve hypotheses for context
            hypotheses = await self._retrieve_hypotheses(workspace_name, hypothesis_ids)

            # Generate pattern using LLM
            pattern_data = await self._generate_pattern(
                cluster, hypotheses, workspace_name, observer, observed
            )

            if pattern_data and pattern_data.get("stability", 0) >= self.config.stability_score_threshold:
                # Convert numeric stability to confidence level string
                stability = pattern_data["stability"]
                if stability >= 0.8:
                    confidence = "high"
                elif stability >= 0.6:
                    confidence = "medium"
                else:
                    confidence = "low"

                # Store induction
                induction = await crud.induction.create_induction(
                    self.db,
                    schemas.InductionCreate(
                        content=pattern_data["content"],
                        observer=observer,
                        observed=observed,
                        pattern_type=pattern_data["pattern_type"],
                        confidence=confidence,
                        source_prediction_ids=[pred.id for pred in cluster],
                        source_premise_ids=pattern_data.get("source_premise_ids", []),
                        stability_score=pattern_data["stability"],
                    ),
                    workspace_name,
                )
                induction_ids.append(induction.id)
                logger.info(f"Created induction {induction.id}: {pattern_data['content']}")
            else:
                stability = pattern_data.get("stability", 0) if pattern_data else 0
                logger.info(
                    f"Skipped pattern with insufficient stability: {stability}"
                )

        return induction_ids

    async def _retrieve_hypotheses(
        self, _workspace_name: str, hypothesis_ids: list[str]
    ) -> list[models.Hypothesis]:
        """Retrieve hypotheses by IDs.

        Args:
            _workspace_name: Workspace name (unused in current implementation)
            hypothesis_ids: List of hypothesis IDs

        Returns:
            List of Hypothesis models
        """
        if not hypothesis_ids:
            return []

        stmt = (
            select(models.Hypothesis)
            .where(models.Hypothesis.workspace_name == _workspace_name)
            .where(models.Hypothesis.id.in_(hypothesis_ids))
        )

        result = await self.db.execute(stmt)
        hypotheses = list(result.scalars().all())

        logger.info(f"Retrieved {len(hypotheses)} hypotheses for context")
        return hypotheses

    async def _generate_pattern(
        self,
        cluster: list[models.Prediction],
        hypotheses: list[models.Hypothesis],
        _workspace_name: str,
        observer: str,
        observed: str,
    ) -> dict[str, Any] | None:
        """Generate an inductive pattern from a cluster of predictions.

        Args:
            cluster: List of similar predictions
            hypotheses: Related hypotheses for context
            workspace_name: Workspace name
            observer: Observer peer name
            observed: Observed peer name

        Returns:
            Dictionary with pattern data or None if extraction failed
        """
        # State for pattern extraction
        pattern_data: dict[str, Any] = {}

        # Tool executor closure
        def tool_executor(tool_name: str, tool_input: dict[str, Any]) -> str:
            nonlocal pattern_data

            if tool_name == "create_induction":
                content = tool_input.get("content", "")
                pattern_type = tool_input.get("pattern_type", "")
                stability = tool_input.get("stability", 0.0)
                rationale = tool_input.get("rationale", "")

                if not content:
                    return "ERROR: Pattern content cannot be empty"
                if pattern_type not in self.config.pattern_types:
                    return f"ERROR: Invalid pattern type. Must be one of: {self.config.pattern_types}"
                if stability < 0 or stability > 1:
                    return "ERROR: Stability score must be between 0 and 1"

                # Collect all source premise IDs from predictions
                source_premise_ids: list[str] = []
                for pred in cluster:
                    # Get hypothesis for this prediction
                    hyp = next((h for h in hypotheses if h.id == pred.hypothesis_id), None)
                    if hyp and hyp.source_premise_ids:
                        source_premise_ids.extend(hyp.source_premise_ids)

                pattern_data = {
                    "content": content,
                    "pattern_type": pattern_type,
                    "stability": stability,
                    "rationale": rationale,
                    "source_premise_ids": list(set(source_premise_ids)),  # Deduplicate
                }

                return f"Pattern recorded: '{content}' (type: {pattern_type}, stability: {stability})"

            return f"ERROR: Unknown tool: {tool_name}"

        # Prepare predictions summary
        predictions_summary = "\n".join(
            [f"{i+1}. [{pred.id}] {pred.content}" for i, pred in enumerate(cluster)]
        )

        # Prepare task prompt
        task_prompt = INDUCTOR_TASK_PROMPT.format(
            predictions_summary=predictions_summary,
            total_predictions=len(cluster),
            observer=observer,
            observed=observed,
            min_predictions=self.config.min_predictions_per_pattern,
            similarity_threshold=self.config.similarity_threshold,
            pattern_types=", ".join(self.config.pattern_types),
            stability_threshold=self.config.stability_score_threshold,
            max_inductions=1,  # One pattern per cluster
        )

        messages = [
            {"role": "system", "content": INDUCTOR_SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt},
        ]

        # Execute LLM call
        try:
            _response: HonchoLLMCallResponse[str] = await honcho_llm_call(
                llm_settings=settings.DIALECTIC.LEVELS["medium"],
                prompt="",
                max_tokens=4000,
                messages=messages,
                tools=self._get_inductor_tools(),
                tool_executor=tool_executor,
                max_tool_iterations=5,
                track_name="Inductor Agent",
            )

            logger.info(f"Pattern extraction completed for cluster of {len(cluster)} predictions")
        except Exception as e:
            logger.error(f"Error in LLM call for pattern extraction: {e}")
            return None

        return pattern_data if pattern_data else None

    def _get_inductor_tools(self) -> list[dict[str, Any]]:
        """Get tool definitions for inductor agent.

        Returns:
            List of tool schemas for LLM
        """
        return [
            {
                "name": "create_induction",
                "description": "Create an inductive pattern extracted from similar predictions",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The general pattern extracted from the predictions (concise, actionable statement)",
                        },
                        "pattern_type": {
                            "type": "string",
                            "enum": self.config.pattern_types,
                            "description": f"Type of pattern: {', '.join(self.config.pattern_types)}",
                        },
                        "stability": {
                            "type": "number",
                            "description": "Stability score (0.0 to 1.0) indicating confidence in the pattern",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of why this pattern was extracted and how stability was calculated",
                        },
                    },
                    "required": ["content", "pattern_type", "stability", "rationale"],
                },
            },
        ]
