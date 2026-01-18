"""Predictor agent implementation for generating blind predictions."""

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.agents.shared import BaseAgent
from src.config import settings
from src.utils.clients import HonchoLLMCallResponse, honcho_llm_call

from .config import PredictorConfig
from .prompts import PREDICTOR_SYSTEM_PROMPT, PREDICTOR_TASK_PROMPT

logger = logging.getLogger(__name__)


class PredictorAgent(BaseAgent):
    """Agent for generating blind predictions from hypotheses.

    The Predictor follows Popperian falsification principles to generate
    testable predictions that can be verified or refuted through observation.
    """

    def __init__(
        self,
        db: AsyncSession,
        config: PredictorConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize the Predictor agent.

        Args:
            db: Database session for CRUD operations
            config: Agent configuration (uses defaults if not provided)
            **kwargs: Additional arguments passed to BaseAgent
        """
        config_obj = config or PredictorConfig()
        super().__init__(db, config_obj, **kwargs)  # type: ignore[reportUnknownMemberType]
        self.config: PredictorConfig = config_obj

    def validate_input(self, input_data: dict[str, Any]) -> bool:
        """Validate input data for prediction generation.

        Args:
            input_data: Must contain:
                - workspace_name: str
                - observer: str
                - observed: str
                Optional:
                - hypothesis_id: str (to generate predictions for specific hypothesis)

        Returns:
            True if valid

        Raises:
            ValueError: If required fields are missing
        """
        required_fields = ["workspace_name", "observer", "observed"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")

        # Validate types
        if not isinstance(input_data["workspace_name"], str):
            raise ValueError("workspace_name must be a string")
        if not isinstance(input_data["observer"], str):
            raise ValueError("observer must be a string")
        if not isinstance(input_data["observed"], str):
            raise ValueError("observed must be a string")

        # Validate optional hypothesis_id
        if "hypothesis_id" in input_data and not isinstance(
            input_data["hypothesis_id"], str
        ):
            raise ValueError("hypothesis_id must be a string")

        return True

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Generate predictions from hypotheses.

        Args:
            input_data: Contains workspace_name, observer, observed, optional hypothesis_id

        Returns:
            Dictionary containing:
                - predictions_created: int
                - prediction_ids: list[str]
                - execution_time: float
        """
        workspace_name = input_data["workspace_name"]
        observer = input_data["observer"]
        observed = input_data["observed"]
        hypothesis_id = input_data.get("hypothesis_id")

        logger.info(
            f"Starting prediction generation for {observer} observing {observed} in workspace {workspace_name}"
        )

        # Step 1: Retrieve hypotheses to generate predictions from
        hypotheses = await self._retrieve_hypotheses(
            workspace_name, observer, observed, hypothesis_id
        )

        if not hypotheses:
            logger.info("No hypotheses found for prediction generation")
            return {
                "predictions_created": 0,
                "prediction_ids": [],
                "reason": "no_hypotheses",
            }

        # Step 2: Generate predictions for each hypothesis
        total_predictions_created = 0
        all_prediction_ids: list[str] = []

        for hypothesis in hypotheses:
            # Retrieve existing predictions for this hypothesis
            existing_predictions = await self._retrieve_existing_predictions(
                workspace_name, hypothesis.id
            )

            # Retrieve source premises for context
            source_premises = await self._retrieve_source_premises(
                workspace_name, hypothesis.source_premise_ids or []
            )

            # Generate predictions using LLM
            predictions = await self._generate_predictions(
                hypothesis, source_premises, existing_predictions
            )

            # Store predictions in database
            prediction_ids = await self._store_predictions(
                predictions, hypothesis.id, workspace_name
            )

            total_predictions_created += len(prediction_ids)
            all_prediction_ids.extend(prediction_ids)

        logger.info(f"Generated {total_predictions_created} predictions")

        return {
            "predictions_created": total_predictions_created,
            "prediction_ids": all_prediction_ids,
        }

    async def _retrieve_hypotheses(
        self,
        workspace_name: str,
        observer: str,
        observed: str,
        hypothesis_id: str | None = None,
    ) -> list[models.Hypothesis]:
        """Retrieve hypotheses for prediction generation.

        Args:
            workspace_name: Workspace identifier
            observer: Observer peer name
            observed: Observed peer name
            hypothesis_id: Optional specific hypothesis ID

        Returns:
            List of Hypothesis models
        """
        if hypothesis_id:
            # Retrieve specific hypothesis
            hypothesis = await crud.hypothesis.get_hypothesis(
                self.db, workspace_name, hypothesis_id
            )
            if hypothesis and hypothesis.status == "active":
                return [hypothesis]
            return []

        # Retrieve active hypotheses with sufficient confidence
        stmt = await crud.hypothesis.list_hypotheses(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            status="active",
        )

        result = await self.db.execute(stmt)
        hypotheses = list(result.scalars().all())

        # Filter by confidence threshold
        filtered_hypotheses = [
            h
            for h in hypotheses
            if h.confidence >= self.config.min_confidence_threshold
        ]

        # Sort by tier (lower tier = less tested = higher priority)
        # then by confidence (higher = more promising)
        filtered_hypotheses.sort(key=lambda h: (h.tier, -h.confidence))

        # Limit retrieval
        filtered_hypotheses = filtered_hypotheses[
            : self.config.max_hypothesis_retrieval
        ]

        logger.info(f"Retrieved {len(filtered_hypotheses)} hypotheses for prediction generation")
        return filtered_hypotheses

    async def _retrieve_existing_predictions(
        self, workspace_name: str, hypothesis_id: str
    ) -> list[models.Prediction]:
        """Retrieve existing predictions for a hypothesis.

        Args:
            workspace_name: Workspace identifier
            hypothesis_id: Hypothesis ID

        Returns:
            List of existing Prediction models
        """
        stmt = await crud.prediction.list_predictions(
            workspace_name=workspace_name, hypothesis_id=hypothesis_id
        )

        result = await self.db.execute(stmt)
        predictions = list(result.scalars().all())

        logger.info(f"Found {len(predictions)} existing predictions for hypothesis {hypothesis_id}")
        return predictions

    async def _retrieve_source_premises(
        self, workspace_name: str, premise_ids: list[str]
    ) -> list[models.Document]:
        """Retrieve source premises (documents) by IDs.

        Args:
            workspace_name: Workspace identifier
            premise_ids: List of document IDs

        Returns:
            List of Document models
        """
        if not premise_ids:
            return []

        stmt = (
            select(models.Document)
            .where(models.Document.workspace_name == workspace_name)
            .where(models.Document.id.in_(premise_ids))
        )

        result = await self.db.execute(stmt)
        premises = list(result.scalars().all())

        logger.info(f"Retrieved {len(premises)} source premises")
        return premises

    async def _generate_predictions(
        self,
        hypothesis: models.Hypothesis,
        source_premises: list[models.Document],
        existing_predictions: list[models.Prediction],
    ) -> list[dict[str, Any]]:
        """Generate predictions using LLM.

        Args:
            hypothesis: Hypothesis to generate predictions from
            source_premises: Supporting observations for hypothesis
            existing_predictions: Existing predictions to avoid duplication

        Returns:
            List of prediction dictionaries with:
                - content: str
                - specificity: float
                - rationale: str
        """
        # Format source premises for prompt
        premises_text = "\n".join(
            [f"- {p.content}" for p in source_premises[:10]]  # Limit to 10 for context
        ) or "No explicit premises available"

        # Format existing predictions
        existing_text = "\n".join(
            [f"- {p.content} (specificity: {getattr(p, 'specificity', 'N/A')})" for p in existing_predictions[:10]]
        ) or "None"

        # Prepare tools for LLM
        tools = self._get_predictor_tools()

        # Format task prompt
        task_prompt = PREDICTOR_TASK_PROMPT.format(
            hypothesis_content=hypothesis.content,
            source_premises=premises_text,
            hypothesis_confidence=hypothesis.confidence,
            hypothesis_tier=hypothesis.tier,
            hypothesis_created=hypothesis.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            existing_predictions=existing_text,
            max_predictions=self.config.predictions_per_hypothesis,
            specificity_threshold=self.config.specificity_threshold,
        )

        # Call LLM with tools
        predictions_created: list[dict[str, Any]] = []

        def tool_executor(tool_name: str, tool_input: dict[str, Any]) -> str:
            """Handle tool calls from LLM."""
            if tool_name == "create_prediction":
                # Validate and collect prediction data
                content = tool_input.get("content", "")
                specificity = tool_input.get("specificity", 0.5)
                rationale = tool_input.get("rationale", "")

                if not content:
                    return "ERROR: Prediction content cannot be empty"
                if specificity < self.config.specificity_threshold:
                    return f"ERROR: Specificity {specificity} below threshold {self.config.specificity_threshold}"

                # Store prediction data for later database insertion
                predictions_created.append({
                    "content": content,
                    "specificity": specificity,
                    "rationale": rationale,
                })

                return f"Prediction recorded: '{content}' (specificity: {specificity})"

            return f"ERROR: Unknown tool: {tool_name}"

        # Execute LLM call
        try:
            messages = [
                {"role": "system", "content": PREDICTOR_SYSTEM_PROMPT},
                {"role": "user", "content": task_prompt},
            ]

            response: HonchoLLMCallResponse[str] = await honcho_llm_call(
                llm_settings=settings.DIALECTIC.LEVELS["medium"],
                prompt="",
                max_tokens=4000,
                messages=messages,
                tools=tools,
                tool_executor=tool_executor,
                max_tool_iterations=10,
                track_name="Predictor Agent",
            )

            logger.info(f"LLM completed with {len(response.tool_calls_made)} tool calls for hypothesis {hypothesis.id}")
        except Exception as e:
            logger.error(f"Error in LLM call for prediction generation: {e}")
            raise

        logger.info(f"LLM generated {len(predictions_created)} prediction candidates for hypothesis {hypothesis.id}")
        return predictions_created

    async def _store_predictions(
        self,
        predictions: list[dict[str, Any]],
        hypothesis_id: str,
        workspace_name: str,
    ) -> list[str]:
        """Store generated predictions in database.

        Args:
            predictions: List of prediction data dictionaries
            hypothesis_id: Associated hypothesis ID
            workspace_name: Workspace identifier

        Returns:
            List of created prediction IDs
        """
        prediction_ids: list[str] = []

        for pred_data in predictions:
            try:
                # Check for duplicates using vector similarity
                is_novel = await self._check_novelty(
                    workspace_name, hypothesis_id, pred_data["content"]
                )

                if not is_novel:
                    logger.info(f"Skipping duplicate prediction: {pred_data['content'][:50]}...")
                    continue

                # Create prediction schema
                prediction_create = schemas.PredictionCreate(
                    content=pred_data["content"],
                    hypothesis_id=hypothesis_id,
                    status="untested",
                    is_blind=self.config.is_blind,
                )

                # Store in database
                prediction = await crud.prediction.create_prediction(
                    self.db, prediction_create, workspace_name
                )

                prediction_ids.append(prediction.id)
                logger.info(f"Created prediction {prediction.id}: {prediction.content[:50]}...")

            except Exception as e:
                logger.error(f"Error storing prediction: {e}")
                continue

        return prediction_ids

    async def _check_novelty(
        self, workspace_name: str, hypothesis_id: str, content: str
    ) -> bool:
        """Check if prediction is novel (not too similar to existing predictions).

        Args:
            workspace_name: Workspace identifier
            hypothesis_id: Hypothesis ID
            content: Prediction content text

        Returns:
            True if novel (dissimilar enough from existing predictions)
        """
        # Search for similar predictions using text query
        similar_predictions = await crud.prediction.search_predictions(
            self.db,
            workspace_name,
            content,
            hypothesis_id=hypothesis_id,
            limit=1,
        )

        if not similar_predictions:
            return True

        # For now, accept all predictions as novel
        # TODO: Implement proper similarity scoring when search_predictions
        # returns distance/similarity metrics
        return True

    def _get_predictor_tools(self) -> list[dict[str, Any]]:
        """Get tool definitions for the Predictor agent.

        Returns:
            List of tool definition dictionaries
        """
        return [
            {
                "name": "create_prediction",
                "description": "Create a testable prediction derived from the hypothesis. The prediction must be specific, falsifiable, and blind (no future knowledge).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The prediction statement (preferably in conditional form: 'When X, then Y')",
                        },
                        "specificity": {
                            "type": "number",
                            "description": "Specificity score from 0.0 to 1.0 based on how concrete and measurable the prediction is",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Brief explanation of how this prediction tests the hypothesis",
                        },
                    },
                    "required": ["content", "specificity", "rationale"],
                },
            }
        ]
