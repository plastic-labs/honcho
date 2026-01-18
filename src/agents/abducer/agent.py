"""Abducer agent implementation for hypothesis generation."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.agents.shared import BaseAgent
from src.config import settings
from src.utils.clients import HonchoLLMCallResponse, honcho_llm_call

from .config import AbducerConfig
from .prompts import ABDUCER_SYSTEM_PROMPT, ABDUCER_TASK_PROMPT

logger = logging.getLogger(__name__)


class AbducerAgent(BaseAgent):
    """Agent for generating explanatory hypotheses from observations.

    The Abducer follows abductive reasoning principles to generate hypotheses
    that explain observed patterns in user behavior and preferences.
    """

    def __init__(
        self,
        db: AsyncSession,
        config: AbducerConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize the Abducer agent.

        Args:
            db: Database session for CRUD operations
            config: Agent configuration (uses defaults if not provided)
            **kwargs: Additional arguments passed to BaseAgent
        """
        config_obj = config or AbducerConfig()
        super().__init__(db, config_obj, **kwargs)  # type: ignore[reportUnknownMemberType]
        self.config: AbducerConfig = config_obj

    def validate_input(self, input_data: dict[str, Any]) -> bool:
        """Validate input data for hypothesis generation.

        Args:
            input_data: Must contain:
                - workspace_name: str
                - observer: str
                - observed: str

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

        return True

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Generate hypotheses from recent observations.

        Args:
            input_data: Contains workspace_name, observer, observed

        Returns:
            Dictionary containing:
                - hypotheses_created: int
                - hypothesis_ids: list[str]
                - execution_time: float
        """
        workspace_name = input_data["workspace_name"]
        observer = input_data["observer"]
        observed = input_data["observed"]

        logger.info(
            f"Starting hypothesis generation for {observer} observing {observed} in workspace {workspace_name}"
        )

        # Step 1: Retrieve recent premises (explicit observations)
        premises = await self._retrieve_premises(workspace_name, observer, observed)

        if len(premises) < self.config.min_premise_count:
            logger.info(
                f"Insufficient premises ({len(premises)} < {self.config.min_premise_count}), skipping hypothesis generation"
            )
            return {
                "hypotheses_created": 0,
                "hypothesis_ids": [],
                "reason": "insufficient_premises",
            }

        # Step 2: Retrieve existing hypotheses to avoid duplicates
        existing_hypotheses = await self._retrieve_existing_hypotheses(
            workspace_name, observer, observed
        )

        # Step 3: Generate hypotheses using LLM
        hypotheses = await self._generate_hypotheses(
            premises, existing_hypotheses, workspace_name, observer, observed
        )

        # Step 4: Store hypotheses in database
        hypothesis_ids = await self._store_hypotheses(
            hypotheses, workspace_name, observer, observed
        )

        logger.info(f"Generated {len(hypothesis_ids)} hypotheses")

        return {
            "hypotheses_created": len(hypothesis_ids),
            "hypothesis_ids": hypothesis_ids,
        }

    async def _retrieve_premises(
        self, workspace_name: str, observer: str, observed: str
    ) -> list[models.Document]:
        """Retrieve recent explicit observations to use as premises.

        Args:
            workspace_name: Workspace identifier
            observer: Observer peer name
            observed: Observed peer name

        Returns:
            List of Document models (explicit level only)
        """
        # Calculate lookback date
        lookback_date = datetime.now(timezone.utc) - timedelta(
            days=self.config.lookback_days
        )

        # Query for explicit observations within lookback period
        stmt = (
            select(models.Document)
            .where(models.Document.workspace_name == workspace_name)
            .where(models.Document.observer == observer)
            .where(models.Document.observed == observed)
            .where(models.Document.level == "explicit")
            .where(models.Document.created_at >= lookback_date)
            .order_by(models.Document.created_at.desc())
            .limit(self.config.max_premise_retrieval)
        )

        result = await self.db.execute(stmt)
        premises = list(result.scalars().all())

        logger.info(f"Retrieved {len(premises)} premises for hypothesis generation")
        return premises

    async def _retrieve_existing_hypotheses(
        self, workspace_name: str, observer: str, observed: str
    ) -> list[models.Hypothesis]:
        """Retrieve existing active hypotheses to avoid duplicates.

        Args:
            workspace_name: Workspace identifier
            observer: Observer peer name
            observed: Observed peer name

        Returns:
            List of existing active Hypothesis models
        """
        stmt = await crud.hypothesis.list_hypotheses(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            status="active",
        )

        result = await self.db.execute(stmt)
        hypotheses = list(result.scalars().all())

        logger.info(f"Found {len(hypotheses)} existing active hypotheses")
        return hypotheses

    async def _generate_hypotheses(
        self,
        premises: list[models.Document],
        existing_hypotheses: list[models.Hypothesis],
        _workspace_name: str,
        _observer: str,
        _observed: str,
    ) -> list[dict[str, Any]]:
        """Generate hypotheses using LLM with abductive reasoning.

        Args:
            premises: List of observations to explain
            existing_hypotheses: Existing hypotheses to avoid duplication
            _workspace_name: Workspace identifier (unused)
            _observer: Observer peer name (unused)
            _observed: Observed peer name (unused)

        Returns:
            List of hypothesis dictionaries with:
                - content: str
                - source_premise_ids: list[str]
                - confidence: float
                - tier: int
        """
        # Format premises for prompt
        premises_text = "\n".join(
            [f"- [id:{p.id}] {p.content}" for p in premises[:50]]  # Limit to 50 for context
        )

        # Format existing hypotheses
        existing_text = "\n".join(
            [f"- {h.content} (confidence: {h.confidence})" for h in existing_hypotheses[:20]]
        ) or "None"

        # Prepare tools for LLM
        tools = self._get_abducer_tools()

        # Format task prompt
        task_prompt = ABDUCER_TASK_PROMPT.format(
            premises=premises_text,
            existing_hypotheses=existing_text,
            max_hypotheses=self.config.max_hypotheses_per_batch,
            confidence_threshold=self.config.confidence_threshold,
        )

        # Call LLM with tools
        hypotheses_created: list[dict[str, Any]] = []

        def tool_executor(tool_name: str, tool_input: dict[str, Any]) -> str:
            """Handle tool calls from LLM."""
            if tool_name == "create_hypothesis":
                # Validate and collect hypothesis data
                content = tool_input.get("content", "")
                source_premise_ids = tool_input.get("source_premise_ids", [])
                confidence = tool_input.get("confidence", 0.5)
                tier = tool_input.get("tier", 0)

                if not content:
                    return "ERROR: Hypothesis content cannot be empty"
                if not source_premise_ids:
                    return "ERROR: source_premise_ids must contain at least one premise ID"
                if confidence < self.config.confidence_threshold:
                    return f"ERROR: Confidence {confidence} below threshold {self.config.confidence_threshold}"

                # Store hypothesis data for later database insertion
                hypotheses_created.append({
                    "content": content,
                    "source_premise_ids": source_premise_ids,
                    "confidence": confidence,
                    "tier": tier,
                })

                return f"Hypothesis recorded: '{content}' (confidence: {confidence})"

            return f"ERROR: Unknown tool: {tool_name}"

        # Execute LLM call
        try:
            messages = [
                {"role": "system", "content": ABDUCER_SYSTEM_PROMPT},
                {"role": "user", "content": task_prompt},
            ]

            response: HonchoLLMCallResponse[str] = await honcho_llm_call(
                llm_settings=settings.DIALECTIC.LEVELS["medium"],  # Use medium reasoning level
                prompt="",  # Ignored since we pass messages
                max_tokens=4000,
                messages=messages,
                tools=tools,
                tool_executor=tool_executor,
                max_tool_iterations=10,
                track_name="Abducer Agent",
            )

            logger.info(f"LLM completed with {len(response.tool_calls_made)} tool calls")
        except Exception as e:
            logger.error(f"Error in LLM call for hypothesis generation: {e}")
            raise

        logger.info(f"LLM generated {len(hypotheses_created)} hypothesis candidates")
        return hypotheses_created

    async def _store_hypotheses(
        self,
        hypotheses: list[dict[str, Any]],
        workspace_name: str,
        observer: str,
        observed: str,
    ) -> list[str]:
        """Store generated hypotheses in database.

        Args:
            hypotheses: List of hypothesis data dictionaries
            workspace_name: Workspace identifier
            observer: Observer peer name
            observed: Observed peer name

        Returns:
            List of created hypothesis IDs
        """
        hypothesis_ids: list[str] = []

        for hyp_data in hypotheses:
            try:
                # Create hypothesis schema
                hypothesis_create = schemas.HypothesisCreate(
                    content=hyp_data["content"],
                    observer=observer,
                    observed=observed,
                    status="active",
                    confidence=hyp_data["confidence"],
                    source_premise_ids=hyp_data["source_premise_ids"],
                    tier=hyp_data.get("tier", 0),
                )

                # Store in database
                hypothesis = await crud.hypothesis.create_hypothesis(
                    self.db, hypothesis_create, workspace_name
                )

                hypothesis_ids.append(hypothesis.id)
                logger.info(f"Created hypothesis {hypothesis.id}: {hypothesis.content[:50]}...")

            except Exception as e:
                logger.error(f"Error storing hypothesis: {e}")
                continue

        return hypothesis_ids

    def _get_abducer_tools(self) -> list[dict[str, Any]]:
        """Get tool definitions for the Abducer agent.

        Returns:
            List of tool definition dictionaries
        """
        return [
            {
                "name": "create_hypothesis",
                "description": "Create a new hypothesis that explains the given observations. The hypothesis should be specific, falsifiable, and explain multiple premises.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The hypothesis statement explaining the observations",
                        },
                        "source_premise_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Document IDs of the premises this hypothesis explains",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score from 0.0 to 1.0 based on explanatory power",
                        },
                        "tier": {
                            "type": "integer",
                            "description": "Tier level (0 for new untested hypotheses)",
                        },
                    },
                    "required": ["content", "source_premise_ids", "confidence"],
                },
            }
        ]
