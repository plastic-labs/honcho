"""Falsifier agent for testing predictions through contradiction search."""

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.agents.falsifier.config import FalsifierConfig
from src.agents.falsifier.prompts import (
    FALSIFIER_SYSTEM_PROMPT,
    FALSIFIER_TASK_PROMPT,
)
from src.agents.shared.base_agent import BaseAgent
from src.config import settings
from src.utils.clients import HonchoLLMCallResponse, honcho_llm_call

logger = logging.getLogger(__name__)


class FalsifierAgent(BaseAgent):
    """Agent for testing predictions through systematic contradiction search."""

    def __init__(
        self,
        db: AsyncSession,
        config: FalsifierConfig | None = None,
    ):
        """Initialize the Falsifier agent.

        Args:
            db: Database session
            config: Optional configuration override
        """
        super().__init__(db)
        self.config: FalsifierConfig = config or FalsifierConfig()
        logger.info(
            f"Initialized FalsifierAgent with max_iterations={self.config.max_search_iterations}"
        )

    def validate_input(self, input_data: dict[str, Any]) -> bool:
        """Validate input data for falsification.

        Args:
            input_data: Dictionary with workspace_name, observer, observed, prediction_id (optional)

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

        # prediction_id is optional (if provided, test specific prediction)
        if "prediction_id" in input_data and not isinstance(
            input_data["prediction_id"], str
        ):
            raise ValueError("prediction_id must be a string")

        return True

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute falsification testing on predictions.

        Args:
            input_data: Dictionary with workspace_name, observer, observed, prediction_id (optional)

        Returns:
            Dictionary with:
                - predictions_tested: Number of predictions tested
                - predictions_falsified: Number marked as falsified
                - predictions_unfalsified: Number marked as unfalsified
                - predictions_inconclusive: Number remaining untested
                - trace_ids: List of FalsificationTrace IDs created
                - reason: Optional reason if no predictions tested
        """
        workspace_name = input_data["workspace_name"]
        observer = input_data["observer"]
        observed = input_data["observed"]
        prediction_id = input_data.get("prediction_id")

        # Step 1: Retrieve untested predictions
        predictions = await self._retrieve_predictions(
            workspace_name, observer, observed, prediction_id
        )

        if not predictions:
            logger.info("No untested predictions found")
            return {
                "predictions_tested": 0,
                "predictions_falsified": 0,
                "predictions_unfalsified": 0,
                "predictions_inconclusive": 0,
                "trace_ids": [],
                "reason": "no_predictions",
            }

        # Step 2: Test each prediction
        results: dict[str, int | list[str]] = {
            "predictions_tested": 0,
            "predictions_falsified": 0,
            "predictions_unfalsified": 0,
            "predictions_inconclusive": 0,
            "trace_ids": [],
        }

        for prediction in predictions[: self.config.max_predictions_per_run]:
            logger.info(f"Testing prediction {prediction.id}: {prediction.content}")

            # Retrieve hypothesis for context
            hypothesis = await crud.hypothesis.get_hypothesis(
                self.db, workspace_name, prediction.hypothesis_id
            )

            if not hypothesis:
                logger.warning(
                    f"Hypothesis {prediction.hypothesis_id} not found for prediction {prediction.id}"
                )
                continue

            # Execute falsification search
            trace_data = await self._test_prediction(
                prediction, hypothesis, workspace_name, observer, observed
            )

            # Store trace
            trace = await crud.trace.create_trace(
                self.db,
                schemas.FalsificationTraceCreate(
                    prediction_id=prediction.id,
                    search_queries=trace_data["search_queries"],
                    contradicting_premise_ids=trace_data["contradicting_premise_ids"],
                    reasoning_chain=trace_data["reasoning_chain"],
                    final_status=trace_data["final_status"],
                    search_count=trace_data["search_count"],
                    search_efficiency_score=trace_data["search_efficiency_score"],
                ),
                workspace_name,
            )

            # Update prediction status
            await crud.prediction.update_prediction(
                self.db,
                workspace_name,
                prediction.id,
                schemas.PredictionUpdate(status=trace_data["final_status"]),
            )

            # Phase 4.4: Update hypothesis confidence based on falsification results
            await self._update_hypothesis_confidence(
                workspace_name, hypothesis, trace_data["final_status"]
            )

            # Update results
            tested = results["predictions_tested"]
            assert isinstance(tested, int)
            results["predictions_tested"] = tested + 1

            trace_ids_list = results["trace_ids"]
            assert isinstance(trace_ids_list, list)
            trace_ids_list.append(trace.id)

            if trace_data["final_status"] == "falsified":
                falsified = results["predictions_falsified"]
                assert isinstance(falsified, int)
                results["predictions_falsified"] = falsified + 1
            elif trace_data["final_status"] == "unfalsified":
                unfalsified = results["predictions_unfalsified"]
                assert isinstance(unfalsified, int)
                results["predictions_unfalsified"] = unfalsified + 1
            else:
                inconclusive = results["predictions_inconclusive"]
                assert isinstance(inconclusive, int)
                results["predictions_inconclusive"] = inconclusive + 1

        return results

    async def _retrieve_predictions(
        self,
        workspace_name: str,
        _observer: str,
        _observed: str,
        prediction_id: str | None = None,
    ) -> list[models.Prediction]:
        """Retrieve untested predictions to falsify.

        Args:
            workspace_name: Workspace name
            _observer: Observer peer name (unused - TODO: filter by observer)
            _observed: Observed peer name (unused - TODO: filter by observed)
            prediction_id: Optional specific prediction to test

        Returns:
            List of Prediction models with status="untested"
        """
        if prediction_id:
            # Test specific prediction
            prediction = await crud.prediction.get_prediction(
                self.db, workspace_name, prediction_id
            )
            return [prediction] if prediction and prediction.status == "untested" else []

        # Get all untested predictions for this workspace
        stmt = await crud.prediction.list_predictions(
            workspace_name=workspace_name, status="untested"
        )

        result = await self.db.execute(stmt)
        predictions = list(result.scalars().all())

        # Return limited number
        return predictions[: self.config.max_predictions_per_run]

    async def _test_prediction(
        self,
        prediction: Any,
        hypothesis: Any,
        workspace_name: str,
        observer: str,
        observed: str,
    ) -> dict[str, Any]:
        """Test a prediction by searching for contradictions.

        Args:
            prediction: Prediction model to test
            hypothesis: Parent Hypothesis model
            workspace_name: Workspace name
            observer: Observer peer name
            observed: Observed peer name

        Returns:
            Dictionary with trace data:
                - search_queries: List of search queries executed
                - contradicting_premise_ids: List of contradicting document IDs
                - reasoning_chain: Explanation of evidence evaluation
                - final_status: "falsified", "unfalsified", or "untested"
                - search_count: Number of searches performed
                - search_efficiency_score: Ratio of useful searches
        """
        # Initialize search state
        search_queries: list[str] = []
        contradicting_premise_ids: list[str] = []
        reasoning_chain: dict[str, Any] = {"iterations": []}
        search_results_history: list[str] = []
        useful_searches = 0

        # Initialize result structure
        result: dict[str, Any] = {
            "query": None,
            "contradictions": [],
            "reasoning": "",
            "search_summary": "",
            "status": "untested",
        }

        # Generate search queries and evaluate evidence
        for iteration in range(self.config.max_search_iterations):
            logger.info(f"Falsification iteration {iteration + 1}/{self.config.max_search_iterations}")

            # Prepare context for LLM
            previous_results = "\n".join(search_results_history) if search_results_history else "None yet"

            # Generate next search query and evaluate
            result = await self._execute_search_iteration(
                prediction,
                hypothesis,
                workspace_name,
                observer,
                observed,
                iteration,
                previous_results,
            )

            if result["query"]:
                search_queries.append(result["query"])

            if result["contradictions"]:
                contradicting_premise_ids.extend(result["contradictions"])
                useful_searches += 1

            if result["reasoning"]:
                reasoning_chain["iterations"].append({
                    "iteration": iteration + 1,
                    "query": result["query"],
                    "reasoning": result["reasoning"],
                    "search_summary": result["search_summary"],
                    "status": result["status"],
                })

            if result["search_summary"]:
                search_results_history.append(result["search_summary"])

            # Check if determination made
            if result["status"] != "untested":
                break

        # Calculate efficiency
        search_efficiency = (
            useful_searches / len(search_queries) if search_queries else 0.0
        )

        # Determine final status
        final_status = result.get("status", "untested")
        if not contradicting_premise_ids and len(search_queries) >= self.config.max_search_iterations:
            # Thorough search found no contradictions = unfalsified
            final_status = "unfalsified"

        reasoning_chain["final_status"] = final_status
        reasoning_chain["total_iterations"] = len(search_queries)
        reasoning_chain["useful_searches"] = useful_searches

        return {
            "search_queries": search_queries,
            "contradicting_premise_ids": contradicting_premise_ids,
            "reasoning_chain": reasoning_chain,
            "final_status": final_status,
            "search_count": len(search_queries),
            "search_efficiency_score": search_efficiency,
        }

    async def _execute_search_iteration(
        self,
        prediction: Any,
        hypothesis: Any,
        workspace_name: str,
        observer: str,
        observed: str,
        iteration: int,
        previous_results: str,
    ) -> dict[str, Any]:
        """Execute one iteration of search and evaluation.

        Args:
            prediction: Prediction being tested
            hypothesis: Parent hypothesis
            workspace_name: Workspace name
            observer: Observer peer name
            observed: Observed peer name
            iteration: Current iteration number
            previous_results: Summary of previous search results

        Returns:
            Dictionary with:
                - query: Search query executed (or None)
                - contradictions: List of contradicting document IDs
                - reasoning: Reasoning about this iteration
                - search_summary: Summary of search results
                - status: Current determination ("untested", "falsified", "unfalsified")
        """
        # State for this iteration
        current_query: str | None = None
        search_strategy: str = ""
        contradictions: list[str] = []
        reasoning: str = ""
        search_summary: str = ""
        status: str = "untested"

        # Tool executor closure
        def tool_executor(tool_name: str, tool_input: dict[str, Any]) -> str:
            nonlocal current_query, search_strategy, contradictions, reasoning, search_summary, status

            if tool_name == "generate_search_query":
                query = tool_input.get("query", "")
                strategy = tool_input.get("strategy", "")
                current_query = query
                search_strategy = strategy

                # Return acknowledgment - actual search happens after tool executor
                return f"Search query recorded: '{query}'"

            elif tool_name == "evaluate_prediction":
                evidence_summary = tool_input.get("evidence_summary", "")
                confidence = tool_input.get("confidence", 0.0)
                determination = tool_input.get("determination", "untested")

                reasoning = f"Evidence: {evidence_summary}\nConfidence: {confidence}\nDetermination: {determination}"

                # Validate determination
                if determination == "falsified" and confidence >= self.config.contradiction_confidence_threshold:
                    status = "falsified"
                elif determination == "unfalsified" and confidence >= self.config.unfalsified_confidence_threshold:
                    status = "unfalsified"
                else:
                    status = "untested"

                return f"Evaluation recorded. Status: {status}"

            elif tool_name == "stop_search":
                reason = tool_input.get("reason", "Search complete")
                reasoning = f"Search stopped: {reason}"
                return "Search stopped."

            return f"Unknown tool: {tool_name}"

        # Prepare task prompt
        task_prompt = FALSIFIER_TASK_PROMPT.format(
            prediction_content=prediction.content,
            hypothesis_content=hypothesis.content,
            prediction_id=prediction.id,
            max_iterations=self.config.max_search_iterations,
            result_limit=self.config.search_result_limit,
            current_iteration=iteration + 1,
            previous_results=previous_results,
            contradiction_threshold=self.config.contradiction_confidence_threshold,
            unfalsified_threshold=self.config.unfalsified_confidence_threshold,
        )

        messages = [
            {"role": "system", "content": FALSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt},
        ]

        # Execute LLM call with tools
        _response: HonchoLLMCallResponse[str] = await honcho_llm_call(
            llm_settings=settings.DIALECTIC.LEVELS["medium"],
            prompt="",
            max_tokens=4000,
            messages=messages,
            tools=self._get_falsifier_tools(),
            tool_executor=tool_executor,
            max_tool_iterations=3,  # Allow LLM to make multiple tool calls
            track_name="Falsifier Agent",
        )

        # Execute search if query was generated
        if current_query:
            search_results = await self._search_observations(
                workspace_name, observer, observed, current_query
            )

            # Format results
            if search_results:
                result_text = f"Found {len(search_results)} observations:\n"
                for doc in search_results:
                    result_text += f"- [{doc.id}] {doc.content}\n"
                    contradictions.append(doc.id)
                search_summary = f"Query: {current_query}\nStrategy: {search_strategy}\n{result_text}"
            else:
                search_summary = f"Query: {current_query}\nStrategy: {search_strategy}\nNo results found."

        return {
            "query": current_query,
            "contradictions": contradictions,
            "reasoning": reasoning,
            "search_summary": search_summary,
            "status": status,
        }

    async def _search_observations(
        self,
        workspace_name: str,
        observer: str,
        observed: str,
        query: str,
    ) -> list[models.Document]:
        """Search observations for potential contradictions.

        Args:
            workspace_name: Workspace name
            observer: Observer peer name
            observed: Observed peer name
            query: Search query text

        Returns:
            List of Document models matching query
        """
        # Use CRUD semantic search
        results = await crud.document.query_documents(
            self.db,
            workspace_name,
            query,
            observer=observer,
            observed=observed,
            top_k=self.config.search_result_limit,
        )

        return list(results)

    def _get_falsifier_tools(self) -> list[dict[str, Any]]:
        """Get tool definitions for falsifier agent.

        Returns:
            List of tool schemas for LLM
        """
        return [
            {
                "name": "generate_search_query",
                "description": "Generate a search query to find observations that might contradict the prediction",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query text targeting specific aspects of the prediction",
                        },
                        "strategy": {
                            "type": "string",
                            "description": "Explanation of the search strategy and what you're looking for",
                        },
                    },
                    "required": ["query", "strategy"],
                },
            },
            {
                "name": "evaluate_prediction",
                "description": "Evaluate the prediction based on gathered evidence and make a determination",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "evidence_summary": {
                            "type": "string",
                            "description": "Summary of all evidence found (supporting and contradicting)",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence level in the determination (0.0 to 1.0)",
                        },
                        "determination": {
                            "type": "string",
                            "enum": ["falsified", "unfalsified", "untested"],
                            "description": "Final determination: falsified (contradicted), unfalsified (no contradictions), untested (inconclusive)",
                        },
                    },
                    "required": ["evidence_summary", "confidence", "determination"],
                },
            },
            {
                "name": "stop_search",
                "description": "Stop the search process early (e.g., if sufficient evidence found or no more angles to explore)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Reason for stopping the search",
                        },
                    },
                    "required": ["reason"],
                },
            },
        ]

    async def _update_hypothesis_confidence(
        self,
        workspace_name: str,
        hypothesis: models.Hypothesis,
        prediction_status: str,
    ) -> None:
        """Update hypothesis confidence based on prediction falsification results.

        Args:
            workspace_name: Workspace name
            hypothesis: Hypothesis being tested
            prediction_status: Status of the prediction (falsified/unfalsified/untested)
        """
        # Skip updates for inconclusive results
        if prediction_status == "untested":
            return

        # Get all predictions for this hypothesis
        from sqlalchemy import func, select

        stmt = select(
            func.count().filter(models.Prediction.status == "falsified").label("falsified_count"),
            func.count().filter(models.Prediction.status == "unfalsified").label("unfalsified_count"),
            func.count().label("total_count"),
        ).where(
            models.Prediction.workspace_name == workspace_name,
            models.Prediction.hypothesis_id == hypothesis.id,
            models.Prediction.status.in_(["falsified", "unfalsified"]),
        )

        result = await self.db.execute(stmt)
        row = result.one()
        falsified_count = row.falsified_count or 0
        unfalsified_count = row.unfalsified_count or 0
        total_tested = row.total_count or 0

        if total_tested == 0:
            return

        # Calculate new confidence based on success rate
        # Confidence = (unfalsified predictions) / (total tested predictions)
        success_rate = unfalsified_count / total_tested
        new_confidence = round(success_rate, 2)

        # Determine new tier based on confidence and test count
        # Tier levels: 0=exploratory, 1=working, 2=candidate
        new_tier = hypothesis.tier  # Default to current tier
        if total_tested >= 5:  # Require minimum tests for tier progression
            if new_confidence >= 0.8:
                new_tier = 2  # candidate
            elif new_confidence >= 0.5:
                new_tier = 1  # working
            else:
                new_tier = 0  # exploratory

        # Determine status
        new_status = hypothesis.status
        if new_confidence < 0.3 and total_tested >= 3:
            # Low confidence with multiple tests = superseded
            new_status = "superseded"
        elif total_tested >= 10 and new_confidence >= 0.9:
            # High confidence with many tests = keep active
            new_status = "active"

        # Update hypothesis if changed
        if (
            hypothesis.confidence != new_confidence
            or hypothesis.tier != new_tier
            or hypothesis.status != new_status
        ):
            await crud.hypothesis.update_hypothesis(
                self.db,
                workspace_name,
                hypothesis.id,
                schemas.HypothesisUpdate(
                    confidence=new_confidence,
                    tier=new_tier,
                    status=new_status,
                ),
            )

            logger.info(
                "Updated hypothesis %s: confidence %s->%s, tier %s->%s, status %s->%s (tested %d, falsified %d, unfalsified %d)",
                hypothesis.id,
                hypothesis.confidence,
                new_confidence,
                hypothesis.tier,
                new_tier,
                hypothesis.status,
                new_status,
                total_tested,
                falsified_count,
                unfalsified_count,
            )
