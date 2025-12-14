"""
Enhanced Self-Play Exploration Mode with Reflection and Meta-Learning.

During memory preparation, the agent explores training tasks to:
1. Experiment with different transformation approaches using multiple strategies
2. Discover new patterns and primitives through exploration
3. Reflect on exploration outcomes using Honcho's dialectic API
4. Learn meta-strategies from past explorations
5. Build a rich library of transformations with insights
6. Continuously improve exploration strategies based on reflection

This creates a sophisticated feedback loop where the agent:
- Explores creatively with multiple strategies
- Reflects on what works and what doesn't
- Learns from reflection to improve future exploration
- Stores insights in Honcho for long-term learning
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import winning approaches
try:
    from code_generation import CodeGenerationStrategy
    from airv_augmentation import AIRVAugmentation, ShapePredictor
    CODE_GEN_AVAILABLE = True
except ImportError:
    CODE_GEN_AVAILABLE = False
    logging.warning("Code generation enhancements not available. Install required dependencies.")

Grid = List[List[int]]


class SelfPlayExplorer:
    """
    Self-play exploration system for discovering primitives.

    During memory preparation, instead of just storing task descriptions,
    the agent actively tries to solve training tasks and learns from its
    experiments.
    """

    def __init__(
        self,
        solver,
        primitive_discovery,
        max_experiments_per_task: int = None,  # Now dynamically determined
        exploration_time_limit: int = 60,
        enable_reflection: bool = True,
        enable_dynamic_exploration: bool = True,
    ):
        """
        Initialize enhanced self-play explorer with autonomous decision-making.

        Args:
            solver: ARCSolver instance
            primitive_discovery: PrimitiveDiscoverySystem instance
            max_experiments_per_task: Max solving attempts (None = dynamic)
            exploration_time_limit: Time limit per task in seconds
            enable_reflection: Enable reflection using dialectic API
            enable_dynamic_exploration: Let agent decide exploration depth
        """
        self.solver = solver
        self.primitive_discovery = primitive_discovery
        self.max_experiments = max_experiments_per_task  # Can be None
        self.time_limit = exploration_time_limit
        self.enable_reflection = enable_reflection
        self.enable_dynamic_exploration = enable_dynamic_exploration

        # Statistics
        self.tasks_explored = 0
        self.primitives_discovered = 0
        self.successful_explorations = 0

        # Meta-learning cache
        self.strategy_performance = {}  # Track which strategies work best
        self.pattern_insights = {}  # Insights for specific patterns

        # Dialectic peer for reflection
        self._reflection_peer = None

        # Enhanced strategies (winning approaches)
        self._code_gen_strategy = None
        self._airv_augmentation = None
        self._shape_predictor = None
        if CODE_GEN_AVAILABLE:
            try:
                self._code_gen_strategy = CodeGenerationStrategy(solver=solver, reflection_peer=None)
                self._airv_augmentation = AIRVAugmentation(reflection_peer=None)
                self._shape_predictor = ShapePredictor(reflection_peer=None)
                logging.info("Enhanced strategies (code generation, AIRV) initialized successfully")
            except Exception as e:
                logging.warning(f"Could not initialize enhanced strategies: {e}")

    async def explore_task(
        self, task_id: str, task_data: Dict, tui=None
    ) -> Dict:
        """
        Explore a training task to discover new primitives.

        Args:
            task_id: Task identifier
            task_data: Task data with train/test examples
            tui: Optional TUI for visualization

        Returns:
            Exploration results including discovered primitives
        """
        if tui:
            tui.add_agent_log(
                "self_play", f"[Self-Play] Exploring task {task_id}..."
            )

            # Show the test input with training examples in the TUI
            train_examples = task_data.get("train", [])
            test_examples = task_data.get("test", [])
            if test_examples and isinstance(test_examples, list) and len(test_examples) > 0:
                first_test = test_examples[0]
                # Ensure first_test is a dictionary
                if isinstance(first_test, dict):
                    test_input = first_test.get("input", [])
                    test_output = first_test.get("output")  # May be None for evaluation tasks
                    tui.update_task(
                        task_id=f"[Exploring] {task_id}",
                        input_grid=test_input,
                        expected_output=test_output,
                        training_examples=train_examples  # Pass all training examples
                    )

        logging.info(f"Self-play exploration of task {task_id}")

        results = {
            "task_id": task_id,
            "solved": False,
            "attempts": 0,
            "primitives_discovered": [],
            "successful_transformations": [],
            "insights": [],
        }

        # Analyze the task
        analysis = await self._analyze_task_for_exploration(task_data)
        results["insights"].append(f"Identified patterns: {', '.join(analysis.get('patterns', []))}")

        # INGEST initial analysis thoughts
        await self._ingest_thought(
            task_id=task_id,
            thought_type="initial_analysis",
            content=f"Analyzing task {task_id}: detected patterns {analysis.get('patterns', [])}",
            metadata={"patterns": analysis.get("patterns", [])},
        )

        # QUESTION THE PUZZLE using internal dialogue (dialectic)
        if self.enable_reflection:
            if tui:
                tui.add_agent_log(
                    "self_play",
                    "[Self-Play] 🤔 Questioning the puzzle through internal dialogue...",
                )
            puzzle_understanding = await self._question_puzzle(task_id, task_data, analysis, tui)
            if puzzle_understanding:
                results["insights"].append(f"Puzzle questions: {puzzle_understanding['key_questions'][:2]}")

                # Create rich natural language description for memory storage
                transformation_desc = puzzle_understanding.get("transformation_description", "")
                transformation_name = puzzle_understanding.get("transformation_name", "unknown transformation")
                input_desc = puzzle_understanding.get("input_description", "")
                output_desc = puzzle_understanding.get("output_description", "")

                rich_content = f"""Puzzle Analysis for {task_id}:

TRANSFORMATION: {transformation_name}

INPUT PATTERNS:
{input_desc}

OUTPUT PATTERNS:
{output_desc}

TRANSFORMATION RULE:
{transformation_desc}

KEY HYPOTHESES:
{chr(10).join(f"  - {h}" for h in puzzle_understanding.get('hypotheses', [])[:3])}

RECOMMENDED APPROACH:
{puzzle_understanding.get('recommended_approach', 'TBD')}
"""

                # Display the rich natural language understanding in TUI
                if tui:
                    tui.add_agent_log(
                        "self_play",
                        f"[Self-Play] 🎯 Transformation: {transformation_name}",
                    )
                    if transformation_desc:
                        # Show first 150 chars of transformation description
                        desc_preview = transformation_desc[:150]
                        if len(transformation_desc) > 150:
                            desc_preview += "..."
                        tui.add_agent_log(
                            "self_play",
                            f"[Self-Play] 📝 Rule: {desc_preview}",
                        )

                # INGEST puzzle questioning with rich natural language description
                await self._ingest_thought(
                    task_id=task_id,
                    thought_type="puzzle_questioning",
                    content=rich_content,
                    metadata={
                        "transformation_name": transformation_name,
                        "input_description": input_desc,
                        "output_description": output_desc,
                        "transformation_description": transformation_desc,
                        "questions": puzzle_understanding.get("key_questions", []),
                        "hypotheses": puzzle_understanding.get("hypotheses", []),
                    },
                    tui=tui,
                )

        # META-STRATEGY PLANNING - Think deeply about HOW to approach this puzzle
        meta_strategy = None
        if self.enable_reflection:
            if tui:
                tui.add_agent_log(
                    "self_play",
                    "[Self-Play] 🧩 META-STRATEGY: Thinking about how to think about this puzzle...",
                )
            meta_strategy = await self._meta_strategy_planning(
                task_id=task_id,
                task_data=task_data,
                analysis=analysis,
                puzzle_understanding=puzzle_understanding if puzzle_understanding else {},
                tui=tui
            )
            if meta_strategy:
                results["insights"].append(f"Meta-strategy: {meta_strategy.get('approach_type', 'adaptive')}")
                if tui and meta_strategy.get("thinking_strategy"):
                    tui.add_agent_log(
                        "self_play",
                        f"[Self-Play] 🎯 Strategy: {meta_strategy['thinking_strategy'][:120]}",
                    )
                if tui and meta_strategy.get("memory_query_strategy"):
                    tui.add_agent_log(
                        "self_play",
                        f"[Self-Play] 💾 Memory approach: {meta_strategy['memory_query_strategy'][:120]}",
                    )
                curiosity_questions = meta_strategy.get("curiosity_questions", [])
                if tui and curiosity_questions and isinstance(curiosity_questions, list):
                    for q in curiosity_questions[:2]:
                        tui.add_agent_log(
                            "self_play",
                            f"[Self-Play] 🔍 Curious: {q[:100]}",
                        )
                # Store meta-strategy in memory
                await self._ingest_thought(
                    task_id=task_id,
                    thought_type="meta_strategy",
                    content=f"Meta-strategy for {task_id}: {meta_strategy}",
                    metadata={
                        "approach_type": meta_strategy.get("approach_type", "unknown"),
                        "thinking_strategy": meta_strategy.get("thinking_strategy", ""),
                    },
                    tui=tui,
                )

        # REFLECT on past experiences with similar patterns (memory introspection)
        # NOW ADAPTIVE: Uses meta-strategy to query memory differently
        if self.enable_reflection:
            if tui:
                memory_approach = meta_strategy.get("memory_query_strategy", "standard queries") if meta_strategy else "standard queries"
                tui.add_agent_log(
                    "self_play",
                    f"[Self-Play] 🧠 Reflecting on past experiences using: {memory_approach[:80]}...",
                )
            # Pass meta-strategy to adapt memory queries
            memory_reflection = await self._adaptive_memory_reflection(
                analysis=analysis,
                meta_strategy=meta_strategy,
                tui=tui
            )
            if memory_reflection:
                results["insights"].append(f"Memory reflection: {memory_reflection['summary'][:100]}")
                if tui:
                    tui.add_agent_log(
                        "self_play",
                        f"[Self-Play] 💭 Memory: {memory_reflection['summary'][:120]}...",
                    )
                    # Show learnings from failures
                    failure_learnings = memory_reflection.get("failure_learnings", [])
                    if failure_learnings and isinstance(failure_learnings, list):
                        for learning in failure_learnings[:2]:
                            learning_str = str(learning) if not isinstance(learning, str) else learning
                            tui.add_agent_log(
                                "self_play",
                                f"[Self-Play] 📚 From failures: {learning_str[:80]}",
                            )
                    # Show untried ideas
                    untried_ideas = memory_reflection.get("untried_ideas", [])
                    if untried_ideas and isinstance(untried_ideas, list):
                        for idea in untried_ideas[:2]:
                            idea_str = str(idea) if not isinstance(idea, str) else idea
                            tui.add_agent_log(
                                "self_play",
                                f"[Self-Play] 💡 Untried: {idea_str[:80]}",
                            )
                    # Show memory adaptation notes if available (from adaptive queries)
                    memory_adaptation = memory_reflection.get("memory_adaptation_notes", "")
                    if memory_adaptation:
                        tui.add_agent_log(
                            "self_play",
                            f"[Self-Play] 🔧 Memory adapted: {memory_adaptation[:100]}",
                        )
                # INGEST memory reflection
                await self._ingest_thought(
                    task_id=task_id,
                    thought_type="memory_reflection",
                    content=f"Memory reflection: {memory_reflection}",
                    metadata={"confidence": memory_reflection.get("confidence", "unknown")},
                    tui=tui,
                )

        # ANALYZE FAILURE PATTERNS to learn from mistakes
        if self.enable_reflection:
            if tui:
                tui.add_agent_log(
                    "self_play",
                    "[Self-Play] 🔍 Analyzing past failures to learn what NOT to do...",
                )
            failure_analysis = await self._analyze_failure_patterns(analysis, tui)
            if failure_analysis:
                creative_insights = failure_analysis.get('creative_insights', [])
                insight_preview = str(creative_insights[:1]) if creative_insights else "[]"
                results["insights"].append(f"Failure insights: {insight_preview}")
                # INGEST failure analysis
                await self._ingest_thought(
                    task_id=task_id,
                    thought_type="failure_analysis",
                    content=f"Failure pattern analysis: {failure_analysis}",
                    metadata={
                        "anti_patterns": failure_analysis.get("anti_patterns", []),
                        "curiosity_directions": failure_analysis.get("curiosity_directions", []),
                    },
                    tui=tui,
                )

        # DYNAMICALLY DECIDE exploration depth based on memory and context
        if self.enable_dynamic_exploration:
            if tui:
                tui.add_agent_log(
                    "self_play",
                    "[Self-Play] 🎯 Deciding exploration depth based on memory...",
                )
            decided_max_experiments = await self._decide_exploration_depth(
                task_id=task_id,
                analysis=analysis,
                memory_reflection=memory_reflection if self.enable_reflection else None,
                tui=tui,
            )
            max_attempts = decided_max_experiments
            if tui:
                tui.add_agent_log(
                    "self_play",
                    f"[Self-Play] 📊 Decision: Will try up to {max_attempts} strategies",
                )
        else:
            max_attempts = self.max_experiments or 6

        # Store task analysis in Honcho (full memory ingestion)
        await self._ingest_task_analysis(task_id, analysis, tui)

        # Get contextual guidance from dialectic before starting
        if self.enable_reflection and tui:
            tui.add_agent_log(
                "self_play",
                "[Self-Play] 🧠 Consulting memory for exploration guidance...",
            )
            guidance = await self._get_exploration_guidance(task_id, analysis)
            if guidance:
                results["insights"].append(f"Guidance: {guidance[:100]}")
                if tui:
                    tui.add_agent_log(
                        "self_play",
                        f"[Self-Play] 💡 Insight: {guidance[:150]}...",
                    )

        # Enhanced exploration strategies with more creative approaches
        strategies = [
            self._explore_with_code_generation,
            self._explore_with_primitive_combinations,
            self._explore_with_pattern_matching,
            self._explore_with_code_mutation,  # NEW: Mutate successful code
            self._explore_with_hybrid_approach,  # NEW: Combine multiple strategies
            self._explore_with_creative_combinations,  # NEW: More creative combinations
        ]

        # REASON about strategy effectiveness using dialectic (if we have history)
        if self.enable_reflection and self.tasks_explored > 2:
            if tui:
                tui.add_agent_log(
                    "self_play",
                    "[Self-Play] 🤔 Reasoning about strategy effectiveness via dialectic...",
                )
            # Reason about the first strategy to see if we should use it
            first_strategy_name = strategies[0].__name__.replace("_explore_with_", "").replace("_", " ").title()
            strategy_reasoning = await self._reason_about_strategy_effectiveness(
                first_strategy_name,
                analysis.get("patterns", [])
            )
            if strategy_reasoning and tui:
                recommendation = strategy_reasoning.get("current_recommendation", "maybe")
                if recommendation == "no":
                    tui.add_agent_log(
                        "self_play",
                        f"[Self-Play] ⚠️ Theory suggests avoiding {first_strategy_name}: {strategy_reasoning.get('reasoning', '')[:80]}",
                    )
                elif recommendation == "yes":
                    tui.add_agent_log(
                        "self_play",
                        f"[Self-Play] ✓ Theory suggests using {first_strategy_name}: {strategy_reasoning.get('reasoning', '')[:80]}",
                    )

        # Track strategy results for reflection
        strategy_results = []

        for strategy_idx, strategy in enumerate(strategies):
            # Check dynamic limit
            if results["attempts"] >= max_attempts:
                if tui:
                    tui.add_agent_log(
                        "self_play",
                        f"[Self-Play] Reached exploration limit ({max_attempts} attempts)",
                    )
                break

            strategy_name = strategy.__name__.replace("_explore_with_", "").replace("_", " ").title()
            if tui:
                tui.add_agent_log(
                    "self_play",
                    f"[Self-Play] Strategy {strategy_idx + 1}/{len(strategies)}: {strategy_name}",
                )

            # INGEST thought before strategy
            await self._ingest_thought(
                task_id=task_id,
                thought_type="strategy_decision",
                content=f"Decided to try strategy: {strategy_name}. Reason: exploring different approaches",
                metadata={"strategy": strategy_name, "attempt_number": results["attempts"] + 1},
                tui=tui,
            )

            strategy_result = await strategy(task_id, task_data, analysis, tui)
            results["attempts"] += 1

            # Log result clearly in TUI
            if tui:
                if strategy_result["success"]:
                    tui.add_agent_log(
                        "success",
                        f"[Self-Play] ✅ {strategy_name}: CORRECT - Solved the puzzle!",
                    )
                else:
                    tui.add_agent_log(
                        "error",
                        f"[Self-Play] ❌ {strategy_name}: INCORRECT - Did not solve",
                    )

                    # CURIOSITY: After failure, ask why and what alternative interpretations exist
                    if self.enable_reflection and meta_strategy:
                        curiosity_reflection = await self._curiosity_driven_reflection(
                            task_id=task_id,
                            failed_strategy=strategy_name,
                            strategy_result=strategy_result,
                            meta_strategy=meta_strategy,
                            tui=tui
                        )
                        if curiosity_reflection and tui:
                            alternative_views = curiosity_reflection.get("alternative_interpretations", [])
                            if alternative_views and isinstance(alternative_views, list):
                                for alt in alternative_views[:2]:
                                    tui.add_agent_log(
                                        "self_play",
                                        f"[Self-Play] 🤔 What if: {alt[:100]}",
                                    )

            # INGEST strategy result thought
            await self._ingest_thought(
                task_id=task_id,
                thought_type="strategy_result",
                content=f"Strategy {strategy_name} result: {'SUCCESS' if strategy_result['success'] else 'FAILED'}",
                metadata={"strategy": strategy_name, "success": strategy_result["success"]},
                tui=tui,
            )

            # INGEST strategy attempt into Honcho memory (full ingestion)
            await self._ingest_strategy_attempt(
                task_id=task_id,
                strategy_name=strategy_name,
                strategy_result=strategy_result,
                analysis=analysis,
            )

            # Track for reflection
            strategy_results.append({
                "strategy": strategy_name,
                "result": strategy_result,
                "success": strategy_result["success"],
            })

            if strategy_result["success"]:
                results["solved"] = True
                results["successful_transformations"].append(
                    strategy_result["transformation"]
                )

                # Discover primitive from successful transformation
                if strategy_result.get("code"):
                    if tui:
                        tui.add_agent_log(
                            "self_play",
                            "[Self-Play] 🔍 Analyzing successful code for primitive discovery...",
                        )

                    primitive = await self.primitive_discovery.discover_from_code(
                        code=strategy_result["code"],
                        task_id=task_id,
                        task_data=task_data,
                    )

                    if primitive:
                        results["primitives_discovered"].append(primitive.name)
                        self.primitives_discovered += 1

                        if tui:
                            tui.add_agent_log(
                                "self_play",
                                f"[Self-Play] ✨ Discovered primitive: {primitive.name}",
                            )
                            tui.add_agent_log(
                                "self_play",
                                f"  Type: {primitive.metadata.get('structure_type', 'unknown')}",
                            )
                            tui.add_agent_log(
                                "self_play",
                                f"  Success Rate: {primitive.avg_success_rate:.1%}",
                            )
                            tui.add_agent_log(
                                "self_play",
                                f"  Description: {primitive.description[:80]}...",
                            )
                    else:
                        if tui:
                            tui.add_agent_log(
                                "self_play",
                                "[Self-Play] ℹ Code did not qualify as a new primitive (duplicate or low quality)",
                            )
            else:
                # Strategy failed - do deep failure reflection if we have failure details
                failure_details = strategy_result.get("failure_details")
                if failure_details and self.enable_reflection:
                    # Check if we have the necessary data for reflection
                    if all(k in failure_details for k in ["wrong_output", "correct_output", "input_grid"]):
                        if tui:
                            tui.add_agent_log(
                                "self_play",
                                "[Self-Play] 🔍 Analyzing failure to understand what went wrong...",
                            )

                        # Perform deep failure reflection using dialectic
                        failure_insights = await self._reflect_on_wrong_answer(
                            task_id=task_id,
                            strategy_name=strategy_name,
                            wrong_output=failure_details["wrong_output"],
                            correct_output=failure_details["correct_output"],
                            input_grid=failure_details["input_grid"],
                            task_data=task_data,
                            tui=tui,
                        )

                        # Store failure insights in strategy result for meta-learning
                        if failure_insights:
                            strategy_result["failure_insights"] = failure_insights

        # Store strategy results for meta-learning
        results["strategy_results"] = strategy_results

        # Update statistics
        self.tasks_explored += 1
        if results["solved"]:
            self.successful_explorations += 1

        # REFLECT on exploration using dialectic API
        if self.enable_reflection:
            if tui:
                tui.add_agent_log(
                    "self_play",
                    "\n[Self-Play] 🤔 Reflecting on exploration...",
                )

            reflection = await self._reflect_on_exploration(
                task_id=task_id,
                analysis=analysis,
                results=results,
                strategy_results=strategy_results,
            )

            if reflection:
                results["reflection"] = reflection
                if tui:
                    tui.add_agent_log(
                        "self_play",
                        f"[Self-Play] 💭 Reflection: {reflection['summary'][:150]}...",
                    )
                    if reflection.get("key_learnings"):
                        for learning in reflection["key_learnings"][:3]:
                            tui.add_agent_log(
                                "self_play",
                                f"  - {learning}",
                            )

        # GENERATIVE INVENTION: If task unsolved and we have patterns, try inventing a new primitive
        if not results["solved"] and self.enable_reflection:
            patterns = analysis.get("patterns", [])
            if patterns:
                if tui:
                    tui.add_agent_log(
                        "self_play",
                        "\n[Self-Play] 💡 Task unsolved - attempting generative primitive invention...",
                    )

                # INGEST invention decision thought
                await self._ingest_thought(
                    task_id=task_id,
                    thought_type="invention_decision",
                    content=f"Decided to attempt generative invention for patterns: {patterns}",
                    metadata={"patterns": patterns, "reason": "all_strategies_failed"},
                    tui=tui,
                )

                # Try inventing a primitive for the most prominent pattern
                target_pattern = patterns[0]
                invented_primitive = await self.primitive_discovery.invent_new_primitive(
                    target_pattern=target_pattern,
                    task_examples=task_data.get("train", [])[:3],  # Use first 3 examples
                )

                if invented_primitive:
                    results["primitives_discovered"].append(invented_primitive.name)
                    self.primitives_discovered += 1

                    if tui:
                        tui.add_agent_log(
                            "self_play",
                            f"[Self-Play] ✨ INVENTED new primitive: {invented_primitive.name}",
                        )
                        tui.add_agent_log(
                            "self_play",
                            f"  Target Pattern: {target_pattern}",
                        )
                        tui.add_agent_log(
                            "self_play",
                            f"  Description: {invented_primitive.description[:80]}...",
                        )

                    # INGEST successful invention
                    await self._ingest_thought(
                        task_id=task_id,
                        thought_type="invention_success",
                        content=f"Successfully invented primitive: {invented_primitive.name} for pattern {target_pattern}",
                        metadata={"primitive_name": invented_primitive.name, "pattern": target_pattern},
                        tui=tui,
                    )
                else:
                    if tui:
                        tui.add_agent_log(
                            "self_play",
                            "[Self-Play] ℹ Could not invent a viable primitive this time",
                        )

                    # INGEST failed invention
                    await self._ingest_thought(
                        task_id=task_id,
                        thought_type="invention_failure",
                        content=f"Attempted to invent primitive for {target_pattern} but did not succeed",
                        metadata={"pattern": target_pattern},
                        tui=tui,
                    )

        # Display task exploration summary with clear success/failure indicator
        if tui:
            tui.add_agent_log(
                "self_play",
                f"\n{'='*50}",
            )

            if results['solved']:
                tui.add_agent_log(
                    "success",
                    f"[Self-Play] 🎉 TASK {task_id[:8]} - SOLVED ✅",
                )
            else:
                tui.add_agent_log(
                    "error",
                    f"[Self-Play] ❌ TASK {task_id[:8]} - UNSOLVED",
                )

            tui.add_agent_log(
                "self_play",
                f"  • Attempts: {results['attempts']} strategies tried",
            )
            tui.add_agent_log(
                "self_play",
                f"  • Primitives Discovered: {len(results['primitives_discovered'])}",
            )
            if results['primitives_discovered']:
                tui.add_agent_log(
                    "self_play",
                    f"  • New Primitives: {', '.join(results['primitives_discovered'])}",
                )

            # Show which strategies succeeded/failed
            if strategy_results:
                success_count = sum(1 for sr in strategy_results if sr["success"])
                tui.add_agent_log(
                    "self_play",
                    f"  • Success Rate: {success_count}/{len(strategy_results)} strategies",
                )

            tui.add_agent_log(
                "self_play",
                f"{'='*50}\n",
            )

        # Store exploration results in Honcho (with reflection and ACTUAL examples)
        await self._store_exploration_results(task_id, results, task_data, tui)

        # Update meta-learning from reflection
        if self.enable_reflection and results.get("reflection"):
            await self._update_meta_learning(analysis, strategy_results, results["reflection"])

        # CONTINUOUS LEARNING via dialectic synthesis
        if self.enable_reflection:
            # Periodically synthesize learnings across all tasks (every 5 tasks)
            if self.tasks_explored % 5 == 0 and self.tasks_explored > 0:
                if tui:
                    tui.add_agent_log(
                        "self_play",
                        "\n[Self-Play] 🔄 Triggering continuous learning synthesis...",
                    )
                synthesis = await self._synthesize_learnings_via_dialectic(tui)
                if synthesis:
                    results["learning_synthesis"] = synthesis

            # Meta-reflect on learning progress (every 10 tasks)
            if self.tasks_explored % 10 == 0 and self.tasks_explored > 0:
                if tui:
                    tui.add_agent_log(
                        "self_play",
                        "\n[Self-Play] 🎓 Meta-reflecting on learning progress...",
                    )
                meta_reflection = await self._meta_reflect_on_learning_progress(tui)
                if meta_reflection:
                    results["meta_reflection"] = meta_reflection

        return results

    async def _analyze_task_for_exploration(self, task_data: Dict) -> Dict:
        """
        Analyze task to guide exploration strategies.

        Returns insights about what patterns to look for.
        """
        analysis = {
            "patterns": [],
            "characteristics": {},
            "suggested_approaches": [],
        }

        train_examples = task_data.get("train", [])
        if not train_examples:
            return analysis

        # Analyze first example
        example = train_examples[0]
        # Ensure example is a dictionary
        if not isinstance(example, dict):
            return analysis
        input_grid = np.array(example.get("input", []))
        output_grid = np.array(example.get("output", []))

        # Shape analysis
        input_shape = input_grid.shape
        output_shape = output_grid.shape

        if input_shape == output_shape:
            analysis["patterns"].append("shape_preserving")
            analysis["suggested_approaches"].append("in_place_transformation")
        elif output_shape[0] > input_shape[0] or output_shape[1] > input_shape[1]:
            analysis["patterns"].append("expansion")
            analysis["suggested_approaches"].append("tiling_or_repetition")
        else:
            analysis["patterns"].append("reduction")
            analysis["suggested_approaches"].append("extraction_or_filtering")

        # Color analysis
        input_colors = set(input_grid.flatten())
        output_colors = set(output_grid.flatten())

        if input_colors == output_colors:
            analysis["patterns"].append("color_preserving")
            analysis["suggested_approaches"].append("geometric_transformation")
        else:
            analysis["patterns"].append("color_changing")
            analysis["suggested_approaches"].append("color_mapping")

        # Symmetry analysis
        if np.array_equal(input_grid, np.fliplr(input_grid)):
            analysis["characteristics"]["horizontal_symmetry"] = True
        if np.array_equal(input_grid, np.flipud(input_grid)):
            analysis["characteristics"]["vertical_symmetry"] = True

        # Check for object-based patterns
        num_objects_input = len(np.unique(input_grid)) - 1  # Exclude background
        num_objects_output = len(np.unique(output_grid)) - 1

        if num_objects_input > 1 or num_objects_output > 1:
            analysis["patterns"].append("multi_object")
            analysis["suggested_approaches"].append("object_manipulation")

        return analysis

    async def _explore_with_code_generation(
        self, task_id: str, task_data: Dict, analysis: Dict, tui=None
    ) -> Dict:
        """
        Exploration strategy: Generate transformation code with LLM.

        ENHANCED: Uses winning approach from Poetiq with:
        - Memory-guided code generation (retrieves similar successful code)
        - Iterative refinement with rich feedback
        - Sandbox execution for safety
        - Stores successful patterns in Honcho
        """
        logging.debug("Exploring with enhanced code generation strategy")

        result = {
            "strategy": "code_generation",
            "success": False,
            "code": None,
            "transformation": None,
        }

        # Use enhanced strategy if available
        if self._code_gen_strategy and CODE_GEN_AVAILABLE:
            try:
                # Update reflection peer connection
                if self._reflection_peer:
                    self._code_gen_strategy._reflection_peer = self._reflection_peer

                if tui:
                    tui.add_agent_log(
                        "self_play",
                        "  → Using ENHANCED code generation (Poetiq-style)..."
                    )

                # Call enhanced code generation with iterative refinement
                enhanced_result = await self._code_gen_strategy.generate_and_test_code(
                    task_id=task_id,
                    task_data=task_data,
                    analysis=analysis,
                    tui=tui,
                    max_iterations=5
                )

                # Merge results
                result.update(enhanced_result)
                result["strategy"] = "code_generation_enhanced"

                return result

            except Exception as e:
                logging.warning(f"Enhanced code generation failed, falling back to basic: {e}")
                if tui:
                    tui.add_agent_log("error", f"  ⚠ Enhanced failed: {str(e)[:60]}, trying basic...")

        # Fallback to basic code generation
        if tui:
            tui.add_agent_log("self_play", "  → Generating transformation code with LLM (basic)...")

        # Build enhanced exploration prompt with context
        prompt = await self._build_exploration_prompt(task_data, analysis)

        try:
            # Generate code using Anthropic API
            response = await self.solver.llm_client.messages.create(
                model=self.solver.config.llm_model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract code from response
            response_text = response.content[0].text
            code = self._extract_code_from_response(response_text)
            if not code:
                if tui:
                    tui.add_agent_log("self_play", "  ✗ No code extracted from LLM response")
                return result

            result["code"] = code

            if tui:
                tui.add_agent_log("self_play", "  ✓ Code generated, testing on training examples...")

            # Test the code with visualization
            test_result = await self._test_transformation(code, task_data, tui)

            # Handle both old bool and new dict return formats
            if isinstance(test_result, bool):
                success = test_result
                result["success"] = success
            elif isinstance(test_result, dict):
                success = test_result.get("success", False)
                result["success"] = success
                # Store failure details for potential reflection
                if not success:
                    result["failure_details"] = test_result
            else:
                success = False
                result["success"] = False

            if success:
                if tui:
                    tui.add_agent_log("self_play", "  ✓✓ SUCCESS! Code works on all training examples")
                result["transformation"] = {
                    "type": "generated_code",
                    "code": code,
                }
            else:
                if tui:
                    tui.add_agent_log("self_play", "  ✗ Code failed on training examples")

        except Exception as e:
            logging.debug(f"Code generation exploration failed: {e}")
            if tui:
                tui.add_agent_log("self_play", f"  ✗ Error: {str(e)[:100]}")

        return result

    async def _explore_with_primitive_combinations(
        self, task_id: str, task_data: Dict, analysis: Dict, tui=None
    ) -> Dict:
        """
        Exploration strategy: Try combinations of existing primitives.

        Only uses simple primitives that take a single grid parameter.
        """
        logging.debug("Exploring with primitive combinations")

        result = {
            "strategy": "primitive_combinations",
            "success": False,
            "code": None,
            "transformation": None,
        }

        # Get suggested primitives based on analysis
        suggested_primitives = self._suggest_primitives_for_patterns(
            analysis.get("patterns", [])
        )

        # Filter to only simple primitives (no additional parameters required)
        simple_primitives = [
            p for p in suggested_primitives
            if p in ["rotate_90", "rotate_180", "rotate_270", "flip_horizontal",
                     "flip_vertical", "transpose", "invert_colors",
                     "extract_largest_object", "compress_grid", "fill_background"]
        ]

        if not simple_primitives:
            if tui:
                tui.add_agent_log("self_play", "  ℹ No simple primitives available for combinations")
            return result

        if tui:
            tui.add_agent_log("self_play", f"  → Trying combinations of: {', '.join(simple_primitives[:3])}")

        # Try combinations
        combination_count = 0
        for prim1 in simple_primitives[:3]:
            for prim2 in simple_primitives[:3]:
                if prim1 == prim2:
                    continue

                combination_count += 1
                if tui:
                    tui.add_agent_log("self_play", f"  → Testing: {prim1} + {prim2}")

                # Create composition code
                code = f"""def transform(grid):
    import numpy as np
    from arceus.primitives import ARCPrimitives

    # Convert input to numpy array
    grid = np.array(grid)

    # First transformation
    result = ARCPrimitives.{prim1}(grid)

    # Second transformation
    result = ARCPrimitives.{prim2}(result)

    return result
"""

                # Test composition with visualization
                try:
                    test_result = await self._test_transformation(code, task_data, tui)
                    success = test_result.get("success", False) if isinstance(test_result, dict) else bool(test_result)
                    if success:
                        if tui:
                            tui.add_agent_log("self_play", f"  ✓✓ SUCCESS! {prim1} + {prim2} works!")
                        result["success"] = True
                        result["code"] = code
                        result["transformation"] = {
                            "type": "primitive_composition",
                            "primitives": [prim1, prim2],
                        }
                        return result
                except Exception as e:
                    logging.debug(f"Primitive combination {prim1} + {prim2} failed: {e}")
                    if tui:
                        tui.add_agent_log("self_play", f"  ✗ {prim1} + {prim2}: {str(e)[:30]}")

        if tui and combination_count > 0:
            tui.add_agent_log("self_play", f"  ✗ Tried {combination_count} combinations, none worked")

        return result

    async def _explore_with_pattern_matching(
        self, task_id: str, task_data: Dict, analysis: Dict, tui=None
    ) -> Dict:
        """
        Exploration strategy: Match against known patterns from discovered primitives.
        """
        logging.debug("Exploring with pattern matching")

        result = {
            "strategy": "pattern_matching",
            "success": False,
            "code": None,
            "transformation": None,
        }

        if tui:
            tui.add_agent_log("self_play", "  → Searching for relevant discovered primitives...")

        # Retrieve relevant discovered primitives
        relevant_primitives = await self.primitive_discovery.retrieve_relevant_primitives(
            task_analysis=analysis,
            limit=5,
        )

        if not relevant_primitives:
            if tui:
                tui.add_agent_log("self_play", "  ✗ No discovered primitives found yet")
            return result

        if tui:
            tui.add_agent_log("self_play", f"  → Found {len(relevant_primitives)} relevant primitives, testing...")

        # Try each relevant primitive
        for idx, primitive in enumerate(relevant_primitives, 1):
            try:
                if tui:
                    tui.add_agent_log("self_play", f"  → Testing primitive {idx}/{len(relevant_primitives)}: {primitive.name}")

                # Test the primitive's code with visualization
                test_result = await self._test_transformation(primitive.code, task_data, tui)
                success = test_result.get("success", False) if isinstance(test_result, dict) else bool(test_result)
                if success:
                    if tui:
                        tui.add_agent_log("self_play", f"  ✓✓ SUCCESS! Primitive {primitive.name} works!")
                    result["success"] = True
                    result["code"] = primitive.code
                    result["transformation"] = {
                        "type": "discovered_primitive",
                        "primitive_name": primitive.name,
                    }
                    return result

            except Exception as e:
                logging.debug(f"Pattern matching with {primitive.name} failed: {e}")
                if tui:
                    tui.add_agent_log("self_play", f"  ✗ Primitive {primitive.name} failed")
                continue

        if tui:
            tui.add_agent_log("self_play", f"  ✗ None of the {len(relevant_primitives)} primitives worked")

        return result

    async def _build_exploration_prompt(self, task_data: Dict, analysis: Dict) -> str:
        """
        Build enhanced LLM prompt for exploration.

        Uses Honcho context to inform the prompt with past learnings.
        """
        examples = task_data.get("train", [])
        if not examples:
            return ""

        # Get insights from meta-learning
        patterns = analysis.get("patterns", [])
        insights = []

        for pattern in patterns:
            if pattern in self.pattern_insights:
                pattern_data = self.pattern_insights[pattern]
                if pattern_data.get("learnings"):
                    insights.extend(pattern_data["learnings"][:2])  # Top 2 learnings

        # Build enhanced prompt with context
        prompt = f"""I am exploring an ARC puzzle to discover new transformation patterns.

Task Analysis:
- Patterns detected: {', '.join(patterns)}
- Suggested approaches: {', '.join(analysis.get('suggested_approaches', []))}
"""

        if insights:
            prompt += f"""
Past Learnings for Similar Patterns:
{chr(10).join('- ' + insight for insight in insights[:3])}
"""

        prompt += """
Training Examples:
"""

        for idx, ex in enumerate(examples[:2], 1):  # Show first 2 examples
            # Ensure ex is a dictionary
            if not isinstance(ex, dict):
                continue
            prompt += f"\nExample {idx}:\n"
            ex_input = ex.get('input', [])
            ex_output = ex.get('output', [])
            prompt += f"Input shape: {np.array(ex_input).shape}\n"
            prompt += f"Output shape: {np.array(ex_output).shape}\n"
            input_colors = len(set(np.array(ex_input).flatten()))
            output_colors = len(set(np.array(ex_output).flatten()))
            prompt += f"Input colors: {input_colors}, Output colors: {output_colors}\n"

        prompt += """
Generate a Python function called 'transform(grid)' that implements this transformation.
The function should:
1. Take a grid (list of lists of integers) as input
2. **IMPORTANT**: Convert input to numpy array first: grid = np.array(grid)
3. Return the transformed grid as a list (use .tolist() if returning numpy array)
4. Use numpy for array operations
5. Be generalizable to work on different grid sizes
6. Consider the detected patterns and past learnings above

Be creative and try novel approaches. Focus on discovering reusable transformation patterns.

Your response should contain only the Python code inside ```python code blocks.
"""

        return prompt

    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        import re

        # Look for code block
        pattern = r"```(?:python)?\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        return None

    async def _test_transformation(self, code: str, task_data: Dict, tui=None) -> Dict:
        """
        Test if transformation code works on training examples.
        Shows visual grid display in TUI for each test.

        Returns:
            Dict with 'success' (bool) and failure details if applicable:
            - 'wrong_output': The incorrect output produced
            - 'correct_output': The expected correct output
            - 'input_grid': The input that was transformed
            - 'failed_example_idx': Which example failed (1-indexed)
        """
        import asyncio

        train_examples = task_data.get("train", [])
        if not train_examples:
            return {"success": False}

        if tui:
            tui.add_agent_log("self_play", f"  → Testing on {len(train_examples)} training examples...")

        # Test on all training examples
        for idx, example in enumerate(train_examples, 1):
            try:
                # Ensure example is a dictionary
                if not isinstance(example, dict):
                    continue
                input_grid = example.get("input", [])
                expected_output = example.get("output", [])

                if tui:
                    input_shape = (len(input_grid), len(input_grid[0]) if input_grid else 0)
                    output_shape = (len(expected_output), len(expected_output[0]) if expected_output else 0)
                    tui.add_agent_log("self_play", f"    Example {idx}/{len(train_examples)}: {input_shape} → {output_shape}")

                    # Show the input and expected output grids visually with training examples
                    tui.update_task(
                        task_id=f"Training Example {idx}",
                        input_grid=input_grid,
                        expected_output=expected_output,
                        training_examples=train_examples[:2]  # Show first 2 training examples
                    )
                    # Give TUI time to update
                    await asyncio.sleep(0.3)

                # Execute code
                local_vars = {"grid": input_grid, "np": np}
                exec(code, local_vars)

                if "transform" in local_vars:
                    result = local_vars["transform"](input_grid)
                else:
                    if tui:
                        tui.add_agent_log("self_play", f"    ✗ Example {idx}: No transform function found")
                    return {"success": False}

                # Show the transformation attempt visually
                if tui:
                    tui.update_transformation_attempt(
                        iteration=idx,
                        transformation="Testing generated transformation",
                        result_grid=result
                    )
                    # Give TUI time to show the result
                    await asyncio.sleep(0.5)

                # Compare
                result_array = np.array(result)
                expected_array = np.array(expected_output)

                if not np.array_equal(result_array, expected_array):
                    if tui:
                        result_shape = result_array.shape
                        expected_shape = expected_array.shape
                        tui.add_agent_log("self_play", f"    ✗ Example {idx}: Output mismatch (got {result_shape}, expected {expected_shape})")
                        # Show correct solution in TUI
                        tui.mark_failed_and_show_solution()
                        await asyncio.sleep(0.5)

                    # Return failure details for reflection
                    return {
                        "success": False,
                        "wrong_output": result_array.tolist(),
                        "correct_output": expected_array.tolist(),
                        "input_grid": input_grid,
                        "failed_example_idx": idx,
                    }
                else:
                    if tui:
                        tui.add_agent_log("self_play", f"    ✓ Example {idx}: Match!")
                        await asyncio.sleep(0.3)

            except Exception as e:
                logging.debug(f"Transformation test failed: {e}")
                if tui:
                    tui.add_agent_log("self_play", f"    ✗ Example {idx}: Error - {str(e)[:50]}")
                    await asyncio.sleep(0.3)
                return {"success": False, "error": str(e)}

        if tui:
            tui.add_agent_log("self_play", f"  ✓ All {len(train_examples)} examples passed!")

        return {"success": True}  # All examples passed

    async def _explore_with_code_mutation(
        self, task_id: str, task_data: Dict, analysis: Dict, tui=None
    ) -> Dict:
        """
        NEW Exploration strategy: Mutate previously successful code.

        Takes successful code from Honcho memory and tries variations.
        """
        logging.debug("Exploring with code mutation strategy")

        result = {
            "strategy": "code_mutation",
            "success": False,
            "code": None,
            "transformation": None,
        }

        if tui:
            tui.add_agent_log("self_play", "  → Searching for code to mutate...")

        try:
            # Get past successful explorations from Honcho
            if not hasattr(self.solver, "active_session") or not self.solver.active_session:
                return result

            session = self.solver.active_session

            # Get all messages (Honcho doesn't support nested metadata filtering)
            messages_page = await session.get_messages()

            # Convert to list and filter client-side for successful explorations
            all_messages = [msg async for msg in messages_page]

            # Filter for solved tasks only
            messages = [
                msg for msg in all_messages
                if msg.metadata and msg.metadata.get("solved") == True
            ]

            if not messages or len(messages) == 0:
                if tui:
                    tui.add_agent_log("self_play", "  ℹ No past successful code found")
                return result

            # Try mutating the first successful code
            base_code = messages[0].content if hasattr(messages[0], 'content') else str(messages[0])

            # Extract code from content if it's wrapped in a message
            import re
            code_match = re.search(r"```python\n(.*?)```", base_code, re.DOTALL)
            if code_match:
                base_code = code_match.group(1)

            if tui:
                tui.add_agent_log("self_play", "  → Mutating successful code...")

            # Use dialectic to generate mutations
            mutation_prompt = f"""I have code that solved a similar ARC puzzle. Generate 3 variations of this code that might work for the current task.

Current task patterns: {', '.join(analysis.get('patterns', []))}

Base code:
```python
{base_code}
```

Generate creative mutations that:
1. Change transformation order
2. Add or remove steps
3. Modify parameters
4. Combine with other operations

Provide 3 code variations, each in a separate ```python code block."""

            mutation_response = await self.solver.llm_client.messages.create(
                model=self.solver.config.llm_model,
                max_tokens=1500,
                messages=[{"role": "user", "content": mutation_prompt}],
            )

            mutation_text = mutation_response.content[0].text
            mutations = re.findall(r"```python\n(.*?)```", mutation_text, re.DOTALL)

            # Test each mutation
            for idx, mutated_code in enumerate(mutations, 1):
                if tui:
                    tui.add_agent_log("self_play", f"  → Testing mutation {idx}/{len(mutations)}...")

                test_result = await self._test_transformation(mutated_code, task_data, tui)
                success = test_result.get("success", False) if isinstance(test_result, dict) else bool(test_result)
                if success:
                    if tui:
                        tui.add_agent_log("self_play", f"  ✓✓ SUCCESS! Mutation {idx} works!")
                    result["success"] = True
                    result["code"] = mutated_code
                    result["transformation"] = {
                        "type": "code_mutation",
                        "base_code": base_code[:100],
                        "mutation_index": idx,
                    }
                    return result

            if tui:
                tui.add_agent_log("self_play", f"  ✗ Tried {len(mutations)} mutations, none worked")

        except Exception as e:
            logging.debug(f"Code mutation exploration failed: {e}")
            if tui:
                tui.add_agent_log("self_play", f"  ✗ Error: {str(e)[:80]}")

        return result

    async def _explore_with_hybrid_approach(
        self, task_id: str, task_data: Dict, analysis: Dict, tui=None
    ) -> Dict:
        """
        NEW Exploration strategy: Hybrid approach combining LLM + primitives.

        Uses dialectic to suggest a combination of LLM-generated code and primitives.
        """
        logging.debug("Exploring with hybrid approach")

        result = {
            "strategy": "hybrid_approach",
            "success": False,
            "code": None,
            "transformation": None,
        }

        if tui:
            tui.add_agent_log("self_play", "  → Using hybrid approach (LLM + primitives)...")

        try:
            # Get relevant primitives
            relevant_primitives = await self.primitive_discovery.retrieve_relevant_primitives(
                task_analysis=analysis,
                limit=3,
            )

            primitive_names = [p.name for p in relevant_primitives]
            primitive_descriptions = [f"{p.name}: {p.description[:60]}" for p in relevant_primitives]

            # Use LLM to generate code that uses these primitives
            hybrid_prompt = f"""Generate transformation code for this ARC puzzle using a combination of custom logic and available primitives.

Task patterns: {', '.join(analysis.get('patterns', []))}

Available primitives to use:
{chr(10).join('- ' + desc for desc in primitive_descriptions)}

Generate a transform(grid) function that:
1. Uses creative custom logic
2. Incorporates one or more of the available primitives
3. Combines them in novel ways

Respond with only Python code in ```python blocks."""

            response = await self.solver.llm_client.messages.create(
                model=self.solver.config.llm_model,
                max_tokens=1200,
                messages=[{"role": "user", "content": hybrid_prompt}],
            )

            code = self._extract_code_from_response(response.content[0].text)
            if not code:
                return result

            result["code"] = code

            if tui:
                tui.add_agent_log("self_play", "  ✓ Hybrid code generated, testing...")

            # Test the hybrid code
            test_result = await self._test_transformation(code, task_data, tui)
            success = test_result.get("success", False) if isinstance(test_result, dict) else bool(test_result)
            if success:
                if tui:
                    tui.add_agent_log("self_play", "  ✓✓ SUCCESS! Hybrid approach works!")
                result["success"] = True
                result["transformation"] = {
                    "type": "hybrid_approach",
                    "primitives_used": primitive_names,
                }
                return result
            else:
                if tui:
                    tui.add_agent_log("self_play", "  ✗ Hybrid code failed")

        except Exception as e:
            logging.debug(f"Hybrid approach failed: {e}")
            if tui:
                tui.add_agent_log("self_play", f"  ✗ Error: {str(e)[:80]}")

        return result

    async def _explore_with_creative_combinations(
        self, task_id: str, task_data: Dict, analysis: Dict, tui=None
    ) -> Dict:
        """
        NEW Exploration strategy: Creative combinations of 3+ primitives.

        Tries longer chains and more creative primitive combinations.
        """
        logging.debug("Exploring with creative combinations")

        result = {
            "strategy": "creative_combinations",
            "success": False,
            "code": None,
            "transformation": None,
        }

        # Get suggested primitives
        suggested_primitives = self._suggest_primitives_for_patterns(
            analysis.get("patterns", [])
        )

        # Filter to simple primitives
        simple_primitives = [
            p for p in suggested_primitives
            if p in ["rotate_90", "rotate_180", "rotate_270", "flip_horizontal",
                     "flip_vertical", "transpose", "invert_colors",
                     "extract_largest_object", "compress_grid", "fill_background"]
        ]

        if len(simple_primitives) < 2:
            return result

        if tui:
            tui.add_agent_log("self_play", f"  → Trying creative 3-step combinations from: {', '.join(simple_primitives[:5])}")

        import random

        # Try several random 3-step combinations
        for attempt in range(min(6, len(simple_primitives))):
            # Randomly select 3 primitives
            if len(simple_primitives) >= 3:
                combo = random.sample(simple_primitives, 3)
            else:
                combo = random.choices(simple_primitives, k=3)

            if tui:
                tui.add_agent_log("self_play", f"  → Testing: {' → '.join(combo)}")

            # Create 3-step composition
            code = f"""def transform(grid):
    import numpy as np
    from arceus.primitives import ARCPrimitives

    # Convert input to numpy array
    grid = np.array(grid)

    result = ARCPrimitives.{combo[0]}(grid)
    result = ARCPrimitives.{combo[1]}(result)
    result = ARCPrimitives.{combo[2]}(result)

    return result
"""

            try:
                test_result = await self._test_transformation(code, task_data, tui)
                success = test_result.get("success", False) if isinstance(test_result, dict) else bool(test_result)
                if success:
                    if tui:
                        tui.add_agent_log("self_play", f"  ✓✓ SUCCESS! {' → '.join(combo)} works!")
                    result["success"] = True
                    result["code"] = code
                    result["transformation"] = {
                        "type": "creative_combination",
                        "primitives": combo,
                    }
                    return result
            except Exception as e:
                logging.debug(f"Creative combination {combo} failed: {e}")
                continue

        if tui:
            tui.add_agent_log("self_play", "  ✗ No creative combinations worked")

        return result

    def _suggest_primitives_for_patterns(self, patterns: List[str]) -> List[str]:
        """Suggest primitive names based on detected patterns.

        Only suggests primitives that actually exist in ARCPrimitives class.
        """
        suggestions = []

        pattern_to_primitives = {
            "shape_preserving": ["rotate_90", "rotate_180", "flip_horizontal", "flip_vertical"],
            "expansion": ["tile_grid", "scale_grid", "repeat_pattern"],
            "reduction": ["extract_largest_object", "compress_grid"],
            "color_preserving": ["rotate_90", "flip_horizontal", "transpose"],
            "color_changing": ["replace_color", "invert_colors", "fill_background"],
            "multi_object": ["extract_objects", "extract_largest_object", "apply_to_each_object"],
        }

        for pattern in patterns:
            if pattern in pattern_to_primitives:
                suggestions.extend(pattern_to_primitives[pattern])

        return list(set(suggestions))  # Deduplicate

    async def _store_exploration_results(self, task_id: str, results: Dict, task_data: Dict = None, tui=None):
        """Store exploration results in Honcho memory with ACTUAL examples."""
        try:
            if not hasattr(self.solver, "active_session"):
                return

            session = self.solver.active_session

            if not session:
                return

            # Include actual training examples so dialectic can reference them later
            examples_summary = ""
            if task_data and task_data.get("train"):
                train_examples = task_data.get("train", [])[:2]  # Store first 2 examples
                examples_summary = "\n\nTraining Examples:"
                for idx, ex in enumerate(train_examples, 1):
                    # Ensure ex is a dictionary
                    if not isinstance(ex, dict):
                        continue
                    input_grid = np.array(ex.get("input", []))
                    output_grid = np.array(ex.get("output", []))
                    examples_summary += f"""
Example {idx}:
  Input: {input_grid.shape} - {input_grid.tolist()[:100]}  # Truncate if large
  Output: {output_grid.shape} - {output_grid.tolist()[:100]}
"""

            # Create message with examples
            content = f"""SELF-PLAY EXPLORATION: {task_id}

Result: {'✓ SOLVED' if results['solved'] else '✗ UNSOLVED'}
Attempts: {results['attempts']}
Primitives Discovered: {len(results['primitives_discovered'])}

Insights:
{chr(10).join('- ' + insight for insight in results['insights'])}

Successful Transformations:
{chr(10).join('- ' + str(t.get('type', 'unknown')) for t in results['successful_transformations'])}

Discovered Primitives: {', '.join(results['primitives_discovered'])}
{examples_summary}
"""

            await session.add_messages([{
                "peer_id": "task_analyst",
                "content": content,
                "metadata": {
                    "type": "self_play_exploration",
                    "task_id": task_id,
                    "solved": results["solved"],
                    "primitives_discovered": results["primitives_discovered"],
                },
            }])

            # Log to TUI
            if tui:
                result_status = "SOLVED" if results["solved"] else "UNSOLVED"
                tui.add_memory_operation(
                    operation="Store Exploration",
                    details=f"{task_id[:12]}: {result_status}, {len(results['primitives_discovered'])} prims",
                    num_results=1
                )

        except Exception as e:
            logging.error(f"Error storing exploration results: {e}")

    async def _adaptive_memory_reflection(
        self,
        analysis: Dict,
        meta_strategy: Optional[Dict],
        tui=None
    ) -> Optional[Dict]:
        """
        ADAPTIVE memory reflection that changes based on meta-strategy.

        Instead of generic queries, this adapts:
        - WHAT to query (based on problem type)
        - HOW to query (based on thinking strategy)
        - WHICH memories to prioritize (based on approach type)
        - HOW to combine insights (compositional vs holistic)
        """
        try:
            if not self._reflection_peer and self.solver.honcho_client:
                try:
                    self._reflection_peer = await self.solver.honcho_client.peer(
                        "exploration_guide",
                        metadata={"role": "exploration_strategist"}
                    )
                except Exception:
                    pass

            if not self._reflection_peer or not hasattr(self.solver, "active_session"):
                return None

            session = self.solver.active_session
            if not session:
                return None

            patterns_str = ", ".join(analysis.get("patterns", []))

            # ADAPTIVE QUERY CONSTRUCTION based on meta-strategy
            if meta_strategy:
                reflection_query = self._build_adaptive_memory_query(
                    patterns_str=patterns_str,
                    meta_strategy=meta_strategy,
                    analysis=analysis
                )
            else:
                # Fallback to generic query if no meta-strategy
                reflection_query = f"""Reflect on my accumulated memory about tasks with these patterns: {patterns_str}

Based on ALL past explorations stored in memory (both successful AND failed attempts):

SUCCESSES:
1. What strategies WORKED on these patterns?
2. Why did they work?
3. What conditions led to success?

FAILURES:
4. What strategies FAILED on these patterns?
5. Why did they fail?
6. What did I learn from the failures?

CURIOSITY:
7. What haven't I tried yet that might work?
8. What creative combinations could I explore?

Provide a JSON response with:
- summary: Brief reflection including lessons from BOTH successes and failures
- encounter_count: Estimated number of similar tasks
- confidence: low/medium/high
- successful_strategies: List of strategies that worked
- failed_strategies: List of strategies that failed
- failure_learnings: What I learned from failures (2-3 points)
- key_insights: List of 2-3 insights from both successes and failures
- untried_ideas: 1-2 creative approaches I haven't tried yet

Format as JSON."""

            reflection_response = await self._reflection_peer.chat(
                reflection_query
            )

            # Log dialectic query to TUI
            if tui:
                tui.add_memory_operation(
                    operation="Dialectic Query",
                    details="Memory reflection",
                    num_results=1 if reflection_response else 0
                )

            if not reflection_response:
                return None

            content = reflection_response.content if hasattr(reflection_response, 'content') else str(reflection_response)

            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                reflection_data = json.loads(json_match.group(0))
                return reflection_data
            else:
                # Fallback
                return {
                    "summary": content[:200],
                    "encounter_count": 0,
                    "confidence": "unknown",
                    "key_insights": []
                }

        except Exception as e:
            logging.debug(f"Error reflecting on memory: {e}")
            return None

    def _build_adaptive_memory_query(
        self,
        patterns_str: str,
        meta_strategy: Dict,
        analysis: Dict
    ) -> str:
        """
        Build a memory query that adapts based on meta-strategy.

        Different problem types, thinking strategies, and approaches
        require different kinds of memory queries.
        """
        problem_type = meta_strategy.get("problem_type", "unknown")
        approach_type = meta_strategy.get("approach_type", "analytical")
        thinking_strategy = meta_strategy.get("thinking_strategy", "")
        memory_query_strategy = meta_strategy.get("memory_query_strategy", "")

        # Base context
        base_query = f"""ADAPTIVE MEMORY QUERY - Guided by meta-strategy

CURRENT PUZZLE:
- Patterns: {patterns_str}
- Problem Type: {problem_type}
- Approach: {approach_type}
- Meta-guidance: {memory_query_strategy}

"""

        # Adapt query based on PROBLEM TYPE
        if "pattern recognition" in problem_type.lower():
            base_query += """PATTERN RECOGNITION FOCUS:
From memory, I need to know:
1. What similar VISUAL PATTERNS have I seen before?
2. Which pattern matching strategies worked vs failed?
3. What subtle pattern variations caused failures?
4. What patterns look similar but aren't?

"""
        elif "spatial reasoning" in problem_type.lower():
            base_query += """SPATIAL REASONING FOCUS:
From memory, I need to know:
1. What SPATIAL TRANSFORMATIONS worked on similar layouts?
2. How did object relationships matter in past spatial tasks?
3. What spatial assumptions led to failures?
4. Which spatial primitives (rotate, flip, translate) were effective?

"""
        elif "logical" in problem_type.lower() or "rule" in problem_type.lower():
            base_query += """LOGICAL RULE DISCOVERY FOCUS:
From memory, I need to know:
1. What LOGICAL RULES applied in similar contexts?
2. How did I discover rules from examples before?
3. What rule assumptions were wrong?
4. What conditional/contextual rules did I miss?

"""
        elif "compositional" in problem_type.lower():
            base_query += """COMPOSITIONAL PROBLEM FOCUS:
From memory, I need to know:
1. How did I DECOMPOSE similar complex problems?
2. What sub-problems appeared and how did I solve them?
3. Which composition strategies (sequence, parallel) worked?
4. What compositional assumptions failed?

"""
        else:
            base_query += """GENERAL FOCUS:
From memory, I need to know:
1. What strategies worked on similar patterns?
2. What strategies failed and why?
3. What approaches I haven't tried?

"""

        # Adapt query based on APPROACH TYPE
        if approach_type == "analytical":
            base_query += """ANALYTICAL APPROACH - Memory queries:
- Prioritize: Systematic strategies, step-by-step methods
- Focus on: WHY things worked (causal understanding)
- Look for: Logical decomposition successes
- Avoid: Intuitive leaps without justification

"""
        elif approach_type == "intuitive":
            base_query += """INTUITIVE APPROACH - Memory queries:
- Prioritize: Pattern recognition, visual similarities
- Focus on: WHAT worked (empirical patterns)
- Look for: Quick heuristics, rule-of-thumb successes
- Embrace: Analogical reasoning from similar cases

"""
        elif approach_type == "experimental":
            base_query += """EXPERIMENTAL APPROACH - Memory queries:
- Prioritize: Trial-and-error learnings, exploration paths
- Focus on: What experiments revealed insights
- Look for: Unexpected discoveries, pivotal moments
- Value: Failures that led to breakthroughs

"""
        elif approach_type == "compositional":
            base_query += """COMPOSITIONAL APPROACH - Memory queries:
- Prioritize: Multi-step solutions, sub-problem patterns
- Focus on: How problems decomposed successfully
- Look for: Modular solutions, reusable components
- Consider: Interaction effects between components

"""
        elif approach_type == "analogical":
            base_query += """ANALOGICAL APPROACH - Memory queries:
- Prioritize: Structurally similar problems (not just pattern-similar)
- Focus on: Cross-domain analogies that worked
- Look for: Abstract similarities, transferable insights
- Explore: How seemingly different problems shared solutions

"""

        # Add specific guidance from meta-strategy
        if thinking_strategy:
            base_query += f"""THINKING STRATEGY GUIDANCE:
{thinking_strategy}

Based on this thinking strategy, emphasize relevant memories.

"""

        # Request structured response
        base_query += """Provide a JSON response with:
- summary: Reflection adapted to the problem type and approach (2-3 sentences)
- encounter_count: Estimated similar tasks
- confidence: low/medium/high
- successful_strategies: Strategies that worked (prioritized by approach type)
- failed_strategies: Strategies that failed (with context)
- failure_learnings: What failures taught (2-3 points, relevant to problem type)
- key_insights: Insights (2-3, tailored to approach type)
- untried_ideas: Novel approaches (1-2, aligned with meta-strategy)
- memory_adaptation_notes: How this query differs from generic queries (meta-comment)

Format as JSON."""

        return base_query

    async def _decide_exploration_depth(
        self,
        task_id: str,
        analysis: Dict,
        memory_reflection: Optional[Dict],
        tui=None,
    ) -> int:
        """
        Use dialectic and memory to dynamically decide how much to explore.

        The agent decides based on:
        - Past experience with similar patterns
        - Confidence level
        - Complexity of the task
        - Current success rates
        """
        try:
            if not self._reflection_peer or not hasattr(self.solver, "active_session"):
                return 6  # Default fallback

            session = self.solver.active_session
            if not session:
                return 6

            patterns_str = ", ".join(analysis.get("patterns", []))

            # Build decision prompt
            decision_prompt = f"""Based on my accumulated memory and experience, decide how many exploration strategies I should try for this task.

Task Patterns: {patterns_str}
Task Characteristics: {', '.join(analysis.get('suggested_approaches', []))}
"""

            if memory_reflection:
                decision_prompt += f"""
My Memory Reflection:
- Confidence with these patterns: {memory_reflection.get('confidence', 'unknown')}
- Past encounters: {memory_reflection.get('encounter_count', 0)}
- Key insights: {', '.join(memory_reflection.get('key_insights', [])[:2])}
"""

            # Add strategy performance data if available
            if self.strategy_performance:
                success_rates = []
                for strategy, data in self.strategy_performance.items():
                    rate = data['successes'] / data['attempts'] if data['attempts'] > 0 else 0
                    if rate > 0:
                        success_rates.append(f"{strategy}: {rate:.1%}")

                if success_rates:
                    decision_prompt += f"""
Strategy Success Rates:
{chr(10).join('- ' + rate for rate in success_rates[:5])}
"""

            decision_prompt += """
Decide how many strategies to try (1-10):
- If confidence is HIGH and I've seen this before → Try FEWER (2-4)
- If confidence is MEDIUM → Try MODERATE (4-6)
- If confidence is LOW or patterns are NOVEL → Try MORE (6-10)
- If task seems COMPLEX → Try MORE

Respond with just a number between 1 and 10, and a brief 1-sentence explanation.
Format: NUMBER: explanation"""

            # Use dialectic for decision
            decision_response = await self._reflection_peer.chat(
                decision_prompt
            )

            # Log dialectic query to TUI
            if tui:
                tui.add_memory_operation(
                    operation="Dialectic Query",
                    details="Exploration depth",
                    num_results=1 if decision_response else 0
                )

            if not decision_response:
                return 6

            content = decision_response.content if hasattr(decision_response, 'content') else str(decision_response)

            # Parse response
            import re
            number_match = re.search(r'(\d+)', content)
            if number_match:
                decided_number = int(number_match.group(1))
                # Clamp to reasonable range
                decided_number = max(1, min(10, decided_number))

                # Extract explanation
                explanation_match = re.search(r':\s*(.+)', content)
                if explanation_match and tui:
                    tui.add_agent_log(
                        "self_play",
                        f"[Self-Play] 💡 Reasoning: {explanation_match.group(1)[:100]}",
                    )

                return decided_number
            else:
                return 6

        except Exception as e:
            logging.debug(f"Error deciding exploration depth: {e}")
            return 6  # Safe default

    async def _ingest_task_analysis(self, task_id: str, analysis: Dict, tui=None):
        """
        Store task analysis in Honcho memory for full ingestion.
        """
        try:
            if not hasattr(self.solver, "active_session") or not self.solver.active_session:
                return

            session = self.solver.active_session

            content = f"""TASK ANALYSIS: {task_id}

Detected Patterns:
{chr(10).join('- ' + p for p in analysis.get('patterns', []))}

Suggested Approaches:
{chr(10).join('- ' + a for a in analysis.get('suggested_approaches', []))}

Characteristics:
{json.dumps(analysis.get('characteristics', {}), indent=2)}
"""

            await session.add_messages([{
                "peer_id": "exploration_guide",
                "content": content,
                "metadata": {
                    "type": "task_analysis",
                    "task_id": task_id,
                    "patterns": analysis.get("patterns", []),
                    "timestamp": datetime.now().isoformat(),
                },
            }])

            if tui:
                tui.add_agent_log(
                    "self_play",
                    "[Self-Play] 💾 Task analysis stored in memory",
                )

        except Exception as e:
            logging.debug(f"Error ingesting task analysis: {e}")

    async def _ingest_strategy_attempt(
        self,
        task_id: str,
        strategy_name: str,
        strategy_result: Dict,
        analysis: Dict,
    ):
        """
        Store each strategy attempt in Honcho for complete memory ingestion.
        """
        try:
            if not hasattr(self.solver, "active_session") or not self.solver.active_session:
                return

            session = self.solver.active_session

            status = "SUCCESS" if strategy_result["success"] else "FAILED"

            content = f"""STRATEGY ATTEMPT: {strategy_name}

Task: {task_id}
Status: {status}
Patterns: {', '.join(analysis.get('patterns', []))}

Result Details:
- Strategy Type: {strategy_result.get('strategy', 'unknown')}
- Transformation: {strategy_result.get('transformation', {})}
"""

            if strategy_result.get("code"):
                content += f"""
Generated Code:
```python
{strategy_result['code'][:300]}...
```
"""

            await session.add_messages([{
                "peer_id": "exploration_guide",
                "content": content,
                "metadata": {
                    "type": "strategy_attempt",
                    "task_id": task_id,
                    "strategy": strategy_name,
                    "success": strategy_result["success"],
                    "patterns": analysis.get("patterns", []),
                    "timestamp": datetime.now().isoformat(),
                },
            }])

        except Exception as e:
            logging.debug(f"Error ingesting strategy attempt: {e}")

    async def _question_puzzle(
        self,
        task_id: str,
        task_data: Dict,
        analysis: Dict,
        tui=None,
    ) -> Optional[Dict]:
        """
        Use dialectic for internal dialogue - question the puzzle to understand it better.

        This creates an internal reasoning process where the agent asks itself questions
        about the puzzle and uses its memory to inform answers.
        """
        try:
            if not self._reflection_peer or not hasattr(self.solver, "active_session"):
                return None

            session = self.solver.active_session
            if not session:
                return None

            # Get training examples for concrete analysis
            train_examples = task_data.get("train", [])
            if not train_examples:
                return None

            patterns_str = ", ".join(analysis.get("patterns", []))

            # Format training examples for dialectic to see actual data
            examples_text = []
            for idx, ex in enumerate(train_examples[:3], 1):  # Show up to 3 examples
                # Ensure ex is a dictionary
                if not isinstance(ex, dict):
                    continue
                input_grid = np.array(ex.get("input", []))
                output_grid = np.array(ex.get("output", []))

                # Create readable representation
                examples_text.append(f"""
Example {idx}:
  Input shape: {input_grid.shape}
  Output shape: {output_grid.shape}
  Input grid:
{input_grid.tolist()}
  Output grid:
{output_grid.tolist()}
  Colors in input: {set(input_grid.flatten().tolist())}
  Colors in output: {set(output_grid.flatten().tolist())}
""")

            examples_str = "\n".join(examples_text)

            # Use dialectic to question and reason about the puzzle with CURIOSITY
            # NOW WITH ACTUAL EXAMPLES!
            questioning_prompt = f"""I'm looking at a puzzle with these detected patterns: {patterns_str}

Here are the ACTUAL training examples:
{examples_str}

DESCRIBE THE PUZZLE IN NATURAL LANGUAGE:
First, describe what you observe in plain English:

1. INPUT PATTERNS: Describe the input grids in natural language
   - What objects, shapes, or structures do you see?
   - How are elements arranged spatially?
   - What colors appear and where?
   - Are there any recurring motifs or structures?

2. OUTPUT PATTERNS: Describe the output grids in natural language
   - What objects, shapes, or structures appear in the outputs?
   - How is the spatial arrangement different from inputs?
   - What colors appear and how do they relate to the inputs?
   - What's new, what's removed, what's transformed?

3. TRANSFORMATION DESCRIPTION: In 2-4 sentences, describe the transformation
   - State clearly: "The transformation takes [input description] and produces [output description]"
   - Describe the transformation process in plain English
   - What is the core rule or operation being applied?
   - Give it a name if possible (e.g., "rotation and color inversion", "object mirroring")

TECHNICAL OBSERVATIONS:
4. Looking at the ACTUAL examples above, what is the transformation?
   - What happens to the grid shape?
   - What happens to the colors?
   - What spatial transformations do you see?

5. What patterns are CONSISTENT across all examples?
   - What stays the same in all examples?
   - What changes in the same way?

CURIOUS EXPLORATION:
3. What transformation might be happening?
   - Looking at the actual grids, what are 3-4 different possibilities?
   - What's the MOST creative explanation that fits ALL examples?
   - Can I state the rule in one sentence?

4. What have I learned from SIMILAR SUCCESSES in my memory?
   - Looking at similar transformations I've seen before
   - What strategies worked on patterns like these?
   - Why did they work?

5. What have I learned from SIMILAR FAILURES in my memory?
   - What strategies FAILED on patterns like these?
   - Why did they fail? What was I missing?
   - What did the failures teach me about these types of puzzles?

6. What HAVEN'T I tried yet that might work?
   - Based on these ACTUAL examples, what unconventional approaches could work?
   - What creative combinations haven't I explored?

"WHAT IF" SCENARIOS (looking at the actual grids):
7. What if I'm seeing the pattern wrong?
   - Looking at the actual examples again, what alternative interpretations exist?
8. What if it requires multiple steps?
   - Can I break this transformation into 2-3 simpler steps?
9. What if it's simpler than I think?
   - What's the simplest rule that explains ALL examples?

DECISION (based on actual examples + memory):
10. What should I try first and why?
11. What should I avoid and why (based on past failures)?
12. What should I be curious about during exploration?

Provide insights as JSON with:
- input_description: Natural language description of what you see in the input grids (2-3 sentences)
- output_description: Natural language description of what you see in the output grids (2-3 sentences)
- transformation_description: Natural language description of the transformation (2-4 sentences explaining the rule)
- transformation_name: A short name for this transformation (e.g., "rotation and color swap")
- key_questions: List of 5-7 deep, curious questions
- hypotheses: List of 3-5 transformation hypotheses (from obvious to creative)
- past_successes: What worked before on similar patterns
- past_failures: What failed before and why
- failure_lessons: What failures taught me
- untried_approaches: 2-3 creative things I haven't tried
- what_if_scenarios: 2-3 alternative interpretations
- recommended_approach: What to try first (informed by both successes and failures)
- avoid: What to avoid (based on past failures)
- curiosity_focus: What to pay special attention to
- confidence: How confident am I (low/medium/high)

Format as JSON."""

            # Internal dialogue via dialectic
            response = await self._reflection_peer.chat(
                questioning_prompt
            )

            # Log dialectic query to TUI
            if tui:
                tui.add_memory_operation(
                    operation="Dialectic Query",
                    details="Puzzle questioning",
                    num_results=1 if response else 0
                )

            if not response:
                return None

            content = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                understanding = json.loads(json_match.group(0))

                # Display in TUI with curious mindset
                if tui:
                    # Show hypotheses
                    hypotheses = understanding.get("hypotheses", [])
                    if hypotheses and isinstance(hypotheses, list):
                        for i, hyp in enumerate(hypotheses[:3], 1):
                            tui.add_agent_log(
                                "self_play",
                                f"[Self-Play] 💡 Hypothesis {i}: {hyp[:80]}",
                            )
                    # Show lessons from past failures
                    failure_lessons = understanding.get("failure_lessons", [])
                    if failure_lessons and isinstance(failure_lessons, list):
                        for lesson in failure_lessons[:2]:
                            tui.add_agent_log(
                                "self_play",
                                f"[Self-Play] 📚 Failure lesson: {lesson[:80]}",
                            )
                    # Show untried approaches
                    untried_approaches = understanding.get("untried_approaches", [])
                    if untried_approaches and isinstance(untried_approaches, list):
                        for approach in untried_approaches[:2]:
                            tui.add_agent_log(
                                "self_play",
                                f"[Self-Play] 🎯 Could try: {approach[:80]}",
                            )
                    # Show what-if scenarios
                    what_if_scenarios = understanding.get("what_if_scenarios", [])
                    if what_if_scenarios and isinstance(what_if_scenarios, list):
                        tui.add_agent_log(
                            "self_play",
                            f"[Self-Play] 🤔 What if: {what_if_scenarios[0][:80]}",
                        )

                return understanding
            else:
                return {
                    "key_questions": ["What transformation is happening?"],
                    "hypotheses": [content[:200]],
                    "confidence": "unknown"
                }

        except Exception as e:
            logging.debug(f"Error questioning puzzle: {e}")
            return None

    async def _ingest_thought(
        self,
        task_id: str,
        thought_type: str,
        content: str,
        metadata: Dict = None,
        tui=None,
    ):
        """
        Ingest EVERY thought/reasoning step into Honcho memory.

        This creates a complete thought stream of the agent's reasoning process.
        """
        try:
            if not hasattr(self.solver, "active_session") or not self.solver.active_session:
                return

            session = self.solver.active_session

            thought_content = f"""THOUGHT: {thought_type.upper()}

Task: {task_id}
Timestamp: {datetime.now().isoformat()}

{content}
"""

            meta = {
                "type": "agent_thought",
                "thought_type": thought_type,
                "task_id": task_id,
                "timestamp": datetime.now().isoformat(),
            }
            if metadata:
                meta.update(metadata)

            await session.add_messages([{
                "peer_id": "exploration_guide",
                "content": thought_content,
                "metadata": meta,
            }])

            # Log to TUI
            if tui:
                tui.add_memory_operation(
                    operation=f"Store Thought",
                    details=f"{thought_type}: {task_id[:16]}",
                    num_results=1
                )

        except Exception as e:
            logging.debug(f"Error ingesting thought: {e}")

    async def _retrieve_past_thoughts(
        self,
        task_id: str = None,
        thought_type: str = None,
        limit: int = 10,
        tui=None,
    ) -> List[Dict]:
        """
        Retrieve past thoughts from memory to inform current reasoning.
        """
        try:
            if not hasattr(self.solver, "active_session") or not self.solver.active_session:
                return []

            session = self.solver.active_session

            # Get all messages and filter client-side
            messages_page = await session.get_messages()
            all_messages = [msg async for msg in messages_page]

            # Filter for agent thoughts
            thoughts = []
            for msg in all_messages:
                if not msg.metadata or msg.metadata.get("type") != "agent_thought":
                    continue

                # Optional filtering
                if thought_type and msg.metadata.get("thought_type") != thought_type:
                    continue

                if task_id and msg.metadata.get("task_id") != task_id:
                    continue

                thoughts.append({
                    "content": msg.content,
                    "thought_type": msg.metadata.get("thought_type"),
                    "task_id": msg.metadata.get("task_id"),
                    "timestamp": msg.metadata.get("timestamp"),
                })

            # Return most recent
            result = thoughts[:limit]

            # Log retrieval to TUI
            if tui:
                filter_desc = f"type={thought_type}" if thought_type else "all thoughts"
                tui.add_memory_operation(
                    operation="Retrieve Thoughts",
                    details=f"{filter_desc}: {len(result)} found",
                    num_results=len(result)
                )

            return result

        except Exception as e:
            logging.debug(f"Error retrieving past thoughts: {e}")
            return []

    async def _analyze_failure_patterns(
        self,
        analysis: Dict,
        tui=None,
    ) -> Optional[Dict]:
        """
        Explicitly analyze failure patterns to learn what NOT to do.

        This creates a curious, learning-from-mistakes mindset.
        """
        try:
            if not self._reflection_peer or not hasattr(self.solver, "active_session"):
                return None

            session = self.solver.active_session
            if not session:
                return None

            patterns_str = ", ".join(analysis.get("patterns", []))

            # Query specifically about failures
            failure_query = f"""Let me analyze my FAILED attempts on tasks with patterns: {patterns_str}

I want to learn deeply from my mistakes and failures:

1. What strategies have I tried that FAILED on these patterns?
   - List specific failed strategies
   - Why did each one fail?

2. What was I MISSING in my failed approaches?
   - What did I misunderstand?
   - What aspects of the problem did I overlook?

3. What PATTERNS exist in my failures?
   - Do certain strategies always fail with certain patterns?
   - Are there common mistakes I keep making?

4. What creative insights can I gain from these failures?
   - What do the failures tell me about what MIGHT work?
   - How can I transform these failures into success?

5. What should I be CURIOUS about based on these failures?
   - What questions do my failures raise?
   - What unexplored directions do they suggest?

Provide a JSON response with:
- failed_strategies: List of strategies that failed with brief why
- missing_elements: What I was missing (2-3 points)
- failure_patterns: Common patterns in failures
- creative_insights: What failures suggest I should try (2-3 ideas)
- curiosity_directions: What to be curious about (2-3 directions)
- anti_patterns: Specific things to avoid

Format as JSON."""

            failure_response = await self._reflection_peer.chat(
                failure_query
            )

            # Log dialectic query to TUI
            if tui:
                tui.add_memory_operation(
                    operation="Dialectic Query",
                    details="Failure patterns",
                    num_results=1 if failure_response else 0
                )

            if not failure_response:
                return None

            content = failure_response.content if hasattr(failure_response, 'content') else str(failure_response)

            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                failure_analysis = json.loads(json_match.group(0))

                # Display in TUI
                creative_insights = failure_analysis.get("creative_insights", [])
                if tui and creative_insights and isinstance(creative_insights, list):
                    for insight in creative_insights[:2]:
                        tui.add_agent_log(
                            "self_play",
                            f"[Self-Play] 💡 From failures: {insight[:80]}",
                        )

                return failure_analysis

        except Exception as e:
            logging.debug(f"Error analyzing failure patterns: {e}")
            return None

    async def _meta_strategy_planning(
        self,
        task_id: str,
        task_data: Dict,
        analysis: Dict,
        puzzle_understanding: Dict,
        tui=None
    ) -> Optional[Dict]:
        """
        Deep meta-cognitive planning about HOW to approach this puzzle.

        This goes beyond "what to try" and asks "how should I think?"
        - What type of problem is this fundamentally?
        - How should I adapt my thinking strategy?
        - How should I use memory differently for this type of puzzle?
        - What questions should I be asking that I'm not asking?
        - How can I be more curious and creative?
        """
        try:
            if not self._reflection_peer or not hasattr(self.solver, "active_session"):
                return None

            session = self.solver.active_session
            if not session:
                return None

            patterns_str = ", ".join(analysis.get("patterns", []))
            transformation_name = puzzle_understanding.get("transformation_name", "unknown")

            # Get training examples for concrete grounding
            train_examples = task_data.get("train", [])
            example_summaries = []
            for idx, ex in enumerate(train_examples[:2], 1):
                if not isinstance(ex, dict):
                    continue
                input_grid = np.array(ex.get("input", []))
                output_grid = np.array(ex.get("output", []))
                example_summaries.append(f"  Example {idx}: {input_grid.shape} → {output_grid.shape}")

            meta_query = f"""I need to think deeply about HOW to approach this puzzle, not just what strategies to try.

PUZZLE CONTEXT:
- Task ID: {task_id}
- Detected patterns: {patterns_str}
- Transformation type: {transformation_name}
- Examples:
{chr(10).join(example_summaries)}

META-COGNITIVE QUESTIONS I need to answer:

1. **PROBLEM TYPE**: What TYPE of cognitive challenge is this fundamentally?
   - Is this a pattern recognition problem?
   - Is this a spatial reasoning problem?
   - Is this a logical rule discovery problem?
   - Is this a compositional/decomposition problem?
   - What does this tell me about HOW I should think?

2. **THINKING STRATEGY**: How should I ADAPT my thinking for this specific puzzle?
   - Should I think bottom-up (from examples to rules)?
   - Should I think top-down (from hypotheses to verification)?
   - Should I think compositionally (break into sub-problems)?
   - Should I think analogically (find similar past puzzles)?
   - What mental model should I use?

3. **MEMORY STRATEGY**: How should I USE my memory system differently for this?
   - What KIND of past experiences should I retrieve?
   - Should I query for similar patterns, or similar problem types?
   - Should I look for successes, failures, or both?
   - How can I use memory to challenge my assumptions?
   - What questions should I ask my memory that I'm not asking?

4. **CURIOSITY AMPLIFICATION**: What questions should I be asking that I'm NOT?
   - What am I taking for granted?
   - What assumptions am I making unconsciously?
   - What "obvious" interpretations might be wrong?
   - What would a completely different perspective reveal?
   - If this puzzle is trying to trick me, how?

5. **EXPLORATION STRATEGY**: How should I SEQUENCE my exploration attempts?
   - Should I start broad or narrow?
   - Should I try simple solutions first or complex?
   - When should I pivot vs persist?
   - How will I know if I'm on the wrong track?
   - What would make me reconsider my approach?

6. **BEYOND THIS PUZZLE**: What can I learn about my thinking PROCESS?
   - What patterns in my own thinking should I notice?
   - Am I falling into habitual patterns?
   - How can I be more creative and less mechanical?
   - What would make me think more deeply next time?

Provide a JSON response with:
- problem_type: Fundamental type of cognitive challenge (string)
- thinking_strategy: How I should adapt my thinking (string, 2-3 sentences)
- mental_model: What mental model to use (string)
- memory_query_strategy: How to query memory differently (string, 2-3 sentences)
- curiosity_questions: Deep questions I should ask (list of 3-5 strings)
- exploration_sequence: How to sequence attempts (string)
- assumptions_to_challenge: What I might be taking for granted (list of 2-3 strings)
- meta_insight: Insight about my thinking process (string)
- approach_type: Overall approach (one of: analytical, intuitive, experimental, compositional, analogical)

Format as JSON."""

            meta_response = await self._reflection_peer.chat(meta_query)

            # Log dialectic query to TUI
            if tui:
                tui.add_memory_operation(
                    operation="Dialectic Query",
                    details="Meta-strategy planning",
                    num_results=1 if meta_response else 0
                )

            if not meta_response:
                return None

            content = meta_response.content if hasattr(meta_response, 'content') else str(meta_response)

            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                meta_strategy = json.loads(json_match.group(0))

                # Display key insights in TUI
                if tui and meta_strategy.get("meta_insight"):
                    tui.add_agent_log(
                        "self_play",
                        f"[Self-Play] 💡 Meta-insight: {meta_strategy['meta_insight'][:100]}",
                    )

                assumptions = meta_strategy.get("assumptions_to_challenge", [])
                if tui and assumptions and isinstance(assumptions, list):
                    for assumption in assumptions[:2]:
                        tui.add_agent_log(
                            "self_play",
                            f"[Self-Play] ⚠️ Challenge: {assumption[:100]}",
                        )

                return meta_strategy

        except Exception as e:
            logging.debug(f"Error in meta-strategy planning: {e}")
            return None

    async def _curiosity_driven_reflection(
        self,
        task_id: str,
        failed_strategy: str,
        strategy_result: Dict,
        meta_strategy: Dict,
        tui=None
    ) -> Optional[Dict]:
        """
        After a strategy fails, be deeply curious about WHY and explore alternatives.

        This amplifies curiosity by:
        - Asking "Why did this fail?"
        - Exploring "What if I'm wrong about my interpretation?"
        - Generating alternative perspectives
        - Challenging assumptions actively
        """
        try:
            if not self._reflection_peer or not hasattr(self.solver, "active_session"):
                return None

            session = self.solver.active_session
            if not session:
                return None

            problem_type = meta_strategy.get("problem_type", "unknown")
            assumptions_to_challenge = meta_strategy.get("assumptions_to_challenge", [])

            curiosity_query = f"""Strategy "{failed_strategy}" just failed on {task_id}.

CONTEXT:
- Problem type: {problem_type}
- Assumptions I had: {', '.join(assumptions_to_challenge)}
- Meta-strategy approach: {meta_strategy.get('approach_type', 'unknown')}

I need to be DEEPLY CURIOUS about this failure:

1. **WHY DID THIS FAIL?**
   - What was my implicit assumption that was wrong?
   - What did I think was true that isn't?
   - What did I overlook or miss entirely?
   - Was I solving the RIGHT problem?

2. **ALTERNATIVE INTERPRETATIONS**
   - What if my entire interpretation of the puzzle is wrong?
   - What if the pattern I'm seeing isn't THE pattern?
   - What other ways could I interpret the same examples?
   - What would happen if I flip my assumptions?

3. **WHAT AM I NOT SEEING?**
   - What's hiding in plain sight?
   - What patterns am I blind to?
   - What connections am I missing?
   - If I were completely wrong, what would be right?

4. **DEEPER QUESTIONS**
   - Is this actually multiple problems disguised as one?
   - Am I overthinking or underthinking?
   - What would a child notice that I don't?
   - What would an expert in a different field see?

5. **GENERATIVE CURIOSITY**
   - What experiment could test my assumption?
   - What question would completely change my approach?
   - What's the simplest explanation I'm ignoring?
   - What's the most complex explanation that actually fits?

Provide a JSON response with:
- why_failed: Root cause analysis (string)
- wrong_assumption: What assumption was incorrect (string)
- alternative_interpretations: Different ways to see the puzzle (list of 2-3 strings)
- blind_spots: What I'm not seeing (list of 2-3 strings)
- experiment_ideas: Tests to validate/invalidate assumptions (list of 1-2 strings)
- paradigm_shift: Fundamentally different way to think about this (string)
- curiosity_insight: What being curious reveals (string)

Format as JSON."""

            curiosity_response = await self._reflection_peer.chat(curiosity_query)

            # Log dialectic query to TUI
            if tui:
                tui.add_memory_operation(
                    operation="Dialectic Query",
                    details="Curiosity reflection",
                    num_results=1 if curiosity_response else 0
                )

            if not curiosity_response:
                return None

            content = curiosity_response.content if hasattr(curiosity_response, 'content') else str(curiosity_response)

            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                curiosity_reflection = json.loads(json_match.group(0))

                # Display curiosity insights
                if tui and curiosity_reflection.get("curiosity_insight"):
                    tui.add_agent_log(
                        "self_play",
                        f"[Self-Play] 🎓 Curiosity reveals: {curiosity_reflection['curiosity_insight'][:100]}",
                    )

                if tui and curiosity_reflection.get("paradigm_shift"):
                    tui.add_agent_log(
                        "self_play",
                        f"[Self-Play] 🔄 Paradigm shift: {curiosity_reflection['paradigm_shift'][:100]}",
                    )

                return curiosity_reflection

        except Exception as e:
            logging.debug(f"Error in curiosity-driven reflection: {e}")
            return None

    async def _get_exploration_guidance(self, task_id: str, analysis: Dict) -> Optional[str]:
        """
        Use Honcho's dialectic API to get contextual guidance for exploration.

        This queries the agent's memory for insights about similar tasks and patterns.
        """
        try:
            # Ensure we have a reflection peer
            if not self._reflection_peer and self.solver.honcho_client:
                # Create or get peer for reflection
                try:
                    self._reflection_peer = await self.solver.honcho_client.peer(
                        "exploration_guide",
                        metadata={"role": "exploration_strategist"}
                    )
                except Exception:
                    # Peer might already exist
                    pass

            if not self._reflection_peer or not hasattr(self.solver, "active_session"):
                return None

            session = self.solver.active_session
            if not session:
                return None

            # Use dialectic chat to get guidance
            patterns_str = ", ".join(analysis.get("patterns", []))
            guidance_query = f"""Based on past exploration experiences, what strategies should I try for a task with these patterns: {patterns_str}?

Consider:
- Which exploration strategies worked well for similar patterns
- Common pitfalls to avoid
- Creative approaches to try

Provide concise, actionable guidance (2-3 sentences)."""

            # Use peer's chat method (dialectic API)
            chat_response = await self._reflection_peer.chat(
                guidance_query
            )

            # Log dialectic query to TUI
            if tui:
                tui.add_memory_operation(
                    operation="Dialectic Query",
                    details="Exploration guidance",
                    num_results=1 if chat_response else 0
                )

            if chat_response and hasattr(chat_response, 'content'):
                return chat_response.content
            elif isinstance(chat_response, str):
                return chat_response

            return None

        except Exception as e:
            logging.debug(f"Error getting exploration guidance: {e}")
            return None

    async def _reflect_on_exploration(
        self,
        task_id: str,
        analysis: Dict,
        results: Dict,
        strategy_results: List[Dict],
    ) -> Optional[Dict]:
        """
        Use Honcho's dialectic API to reflect on exploration outcomes.

        Analyzes what worked, what didn't, and why. Stores insights for future use.
        """
        try:
            if not self._reflection_peer or not hasattr(self.solver, "active_session"):
                return None

            session = self.solver.active_session
            if not session:
                return None

            # Build reflection prompt
            patterns_str = ", ".join(analysis.get("patterns", []))
            strategies_summary = []
            for sr in strategy_results:
                status = "✓ SUCCESS" if sr["success"] else "✗ FAILED"
                strategies_summary.append(f"- {sr['strategy']}: {status}")

            reflection_prompt = f"""Reflect on this exploration attempt for task {task_id}:

Task Patterns: {patterns_str}
Final Result: {'SOLVED' if results['solved'] else 'UNSOLVED'}
Attempts: {results['attempts']}
Primitives Discovered: {len(results['primitives_discovered'])}

Strategy Results:
{chr(10).join(strategies_summary)}

Provide a reflection that includes:
1. Summary: What happened overall (1 sentence)
2. Key Learnings: 3 specific insights about what worked or didn't work
3. Future Recommendations: What to try next time for similar patterns

Format as JSON with keys: summary, key_learnings (list), recommendations (list)"""

            # Use dialectic chat for reflection
            reflection_response = await self._reflection_peer.chat(
                reflection_prompt
            )

            # Log dialectic query to TUI
            if tui:
                tui.add_memory_operation(
                    operation="Dialectic Query",
                    details="Memory reflection",
                    num_results=1 if reflection_response else 0
                )

            if not reflection_response:
                return None

            content = reflection_response.content if hasattr(reflection_response, 'content') else str(reflection_response)

            # Try to parse JSON response
            import json
            import re

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                reflection_data = json.loads(json_match.group(0))
                return reflection_data
            else:
                # Fallback to structured text parsing
                return {
                    "summary": content[:200],
                    "key_learnings": ["Reflection generated but not in expected format"],
                    "recommendations": [],
                }

        except Exception as e:
            logging.debug(f"Error reflecting on exploration: {e}")
            return None

    async def _reflect_on_wrong_answer(
        self,
        task_id: str,
        strategy_name: str,
        wrong_output: Grid,
        correct_output: Grid,
        input_grid: Grid,
        task_data: Dict = None,
        tui=None
    ) -> Optional[Dict]:
        """
        Use dialectic to deeply analyze WHY the agent got the answer wrong.

        Compares wrong answer to correct answer and extracts insights about:
        - What the agent misunderstood about the transformation
        - Visual differences between wrong and correct outputs
        - What the correct transformation actually does
        - What insights can be learned from this failure

        Args:
            task_id: Task identifier
            strategy_name: Which strategy produced the wrong answer
            wrong_output: The incorrect output grid produced by the agent
            correct_output: The correct expected output grid
            input_grid: The input grid that was transformed
            task_data: Full task data including training examples
            tui: Optional TUI for visualization

        Returns:
            Dict with failure insights, or None if reflection fails
        """
        try:
            if not self._reflection_peer or not hasattr(self.solver, "active_session"):
                return None

            session = self.solver.active_session
            if not session:
                return None

            # Convert grids to lists for JSON serialization
            wrong_output_list = np.array(wrong_output).tolist()
            correct_output_list = np.array(correct_output).tolist()
            input_grid_list = np.array(input_grid).tolist()

            # Get grid shapes
            input_shape = np.array(input_grid).shape
            wrong_shape = np.array(wrong_output).shape
            correct_shape = np.array(correct_output).shape

            # Get color information
            input_colors = set(np.array(input_grid).flatten().tolist())
            wrong_colors = set(np.array(wrong_output).flatten().tolist())
            correct_colors = set(np.array(correct_output).flatten().tolist())

            # Build reflection prompt with actual grids
            reflection_prompt = f"""I got a puzzle WRONG and need to understand why.

Task: {task_id}
Strategy Used: {strategy_name}

INPUT GRID:
  Shape: {input_shape}
  Colors: {sorted(input_colors)}
  Grid: {input_grid_list}

MY WRONG ANSWER:
  Shape: {wrong_shape}
  Colors: {sorted(wrong_colors)}
  Grid: {wrong_output_list}

CORRECT ANSWER:
  Shape: {correct_shape}
  Colors: {sorted(correct_colors)}
  Grid: {correct_output_list}

"""

            # Add training examples if available
            if task_data and task_data.get("train"):
                reflection_prompt += "\nTRAINING EXAMPLES (Correct transformations):\n"
                for idx, ex in enumerate(task_data["train"][:2], 1):
                    # Ensure ex is a dictionary
                    if not isinstance(ex, dict):
                        continue
                    ex_input = np.array(ex.get("input", []))
                    ex_output = np.array(ex.get("output", []))
                    reflection_prompt += f"""
Example {idx}:
  Input: {ex_input.shape} - {ex_input.tolist()}
  Output: {ex_output.shape} - {ex_output.tolist()}
"""

            reflection_prompt += """

DEEP ANALYSIS - Help me understand my failure:

1. DESCRIBE IN NATURAL LANGUAGE:
   - Describe the INPUT in plain English (what objects/patterns/structures are present)
   - Describe the CORRECT OUTPUT in plain English (what it should look like)
   - Describe my WRONG OUTPUT in plain English (what I produced instead)
   - In 2-4 sentences, explain what the CORRECT transformation rule is

2. WHAT WENT WRONG:
   - What is the CORRECT transformation rule? (Be specific, looking at the actual grids)
   - What did my wrong answer do instead?
   - What did I MISUNDERSTAND about the transformation?

3. VISUAL COMPARISON:
   - What are the KEY visual differences between my wrong answer and the correct answer?
   - What patterns did I miss in the correct answer?
   - What patterns did I incorrectly add in my wrong answer?

4. ROOT CAUSE:
   - Why did this strategy fail? What is the fundamental reason?
   - What aspect of the transformation does this strategy struggle with?

5. INSIGHTS FOR LEARNING:
   - What should I learn from this failure?
   - How can I avoid this mistake in the future?
   - What new capability or understanding do I need?

Provide insights as JSON with keys:
- input_description: Natural language description of the input (2-3 sentences)
- correct_output_description: Natural language description of what the correct output looks like (2-3 sentences)
- wrong_output_description: Natural language description of what my wrong output looks like (2-3 sentences)
- correct_rule: Clear natural language description of what the correct transformation does (2-4 sentences)
- correct_rule_name: A short name for the correct transformation (e.g., "object rotation and recoloring")
- my_mistake: What my wrong answer did wrong
- visual_differences: List of key visual differences
- root_cause: Why this strategy failed fundamentally
- learnings: List of 2-3 actionable insights for future
- missing_capability: What capability/understanding I lack"""

            # Use dialectic chat for deep failure analysis
            reflection_response = await self._reflection_peer.chat(
                reflection_prompt
            )

            # Log dialectic query to TUI
            if tui:
                tui.add_memory_operation(
                    operation="Dialectic Query",
                    details="Memory reflection",
                    num_results=1 if reflection_response else 0
                )

            if not reflection_response:
                return None

            content = reflection_response.content if hasattr(reflection_response, 'content') else str(reflection_response)

            # Try to parse JSON response
            import json
            import re

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                failure_insights = json.loads(json_match.group(0))

                # Create rich natural language failure analysis for memory
                input_desc = failure_insights.get("input_description", "")
                correct_output_desc = failure_insights.get("correct_output_description", "")
                wrong_output_desc = failure_insights.get("wrong_output_description", "")
                correct_rule = failure_insights.get("correct_rule", "N/A")
                correct_rule_name = failure_insights.get("correct_rule_name", "unknown transformation")

                rich_failure_content = f"""Deep Failure Analysis - {strategy_name} on {task_id}:

CORRECT TRANSFORMATION: {correct_rule_name}

INPUT:
{input_desc}

WHAT I PRODUCED (WRONG):
{wrong_output_desc}

WHAT IT SHOULD BE (CORRECT):
{correct_output_desc}

CORRECT TRANSFORMATION RULE:
{correct_rule}

MY MISTAKE:
{failure_insights.get('my_mistake', 'N/A')}

ROOT CAUSE:
{failure_insights.get('root_cause', 'N/A')}

KEY LEARNINGS:
{chr(10).join(f"  - {learning}" for learning in failure_insights.get('learnings', []))}

MISSING CAPABILITY:
{failure_insights.get('missing_capability', 'N/A')}
"""

                # Store this failure reflection in memory with rich natural language
                await self._ingest_thought(
                    task_id=task_id,
                    thought_type="failure_reflection",
                    content=rich_failure_content,
                    metadata={
                        "strategy": strategy_name,
                        "correct_rule_name": correct_rule_name,
                        "input_description": input_desc,
                        "correct_output_description": correct_output_desc,
                        "wrong_output_description": wrong_output_desc,
                        "correct_rule": correct_rule,
                        "root_cause": failure_insights.get("root_cause", ""),
                        "learnings": failure_insights.get("learnings", []),
                        "missing_capability": failure_insights.get("missing_capability", ""),
                    },
                    tui=tui,
                )

                if tui:
                    # Show the correct transformation name
                    tui.add_agent_log(
                        "self_play",
                        f"\n[Self-Play] 🎯 Correct transformation: {correct_rule_name}",
                    )
                    # Show what went wrong
                    tui.add_agent_log(
                        "self_play",
                        f"[Self-Play] 🔍 Failure Analysis: {failure_insights.get('my_mistake', '')[:100]}",
                    )
                    # Show key learnings
                    for learning in failure_insights.get("learnings", [])[:2]:
                        tui.add_agent_log(
                            "self_play",
                            f"  💡 Learn: {learning}",
                        )

                return failure_insights
            else:
                # Fallback to unstructured insights
                return {
                    "correct_rule": "Could not parse",
                    "my_mistake": content[:200],
                    "visual_differences": [],
                    "root_cause": "Analysis did not return expected format",
                    "learnings": ["Deep reflection generated but format unclear"],
                    "missing_capability": "Unknown"
                }

        except Exception as e:
            logging.debug(f"Error reflecting on wrong answer: {e}")
            if tui:
                tui.add_agent_log("self_play", f"  ⚠️ Error in failure reflection: {str(e)[:80]}")
            return None

    async def _update_meta_learning(
        self,
        analysis: Dict,
        strategy_results: List[Dict],
        reflection: Dict,
    ):
        """
        Update meta-learning cache based on reflection insights.

        Stores BOTH successes AND failures - learns from both!
        """
        try:
            patterns = analysis.get("patterns", [])

            # Update strategy performance tracking (successes AND failures)
            for sr in strategy_results:
                strategy_name = sr["strategy"]
                success = sr["success"]

                if strategy_name not in self.strategy_performance:
                    self.strategy_performance[strategy_name] = {
                        "attempts": 0,
                        "successes": 0,
                        "failures": 0,
                        "patterns_worked": [],
                        "patterns_failed": [],  # NEW: Track what patterns it fails on
                    }

                self.strategy_performance[strategy_name]["attempts"] += 1
                if success:
                    self.strategy_performance[strategy_name]["successes"] += 1
                    self.strategy_performance[strategy_name]["patterns_worked"].extend(patterns)
                else:
                    # TRACK FAILURES explicitly
                    self.strategy_performance[strategy_name]["failures"] += 1
                    self.strategy_performance[strategy_name]["patterns_failed"].extend(patterns)

            # Store pattern insights (successes AND failures)
            for pattern in patterns:
                if pattern not in self.pattern_insights:
                    self.pattern_insights[pattern] = {
                        "successful_strategies": [],
                        "failed_strategies": [],  # NEW: Track what fails
                        "learnings": [],
                        "failure_learnings": [],  # NEW: Learn from failures
                    }

                # Add learnings from reflection
                if reflection.get("key_learnings"):
                    self.pattern_insights[pattern]["learnings"].extend(
                        reflection["key_learnings"]
                    )

                # Track which strategies worked AND failed
                for sr in strategy_results:
                    if sr["success"]:
                        self.pattern_insights[pattern]["successful_strategies"].append(
                            sr["strategy"]
                        )
                    else:
                        # TRACK FAILURES
                        self.pattern_insights[pattern]["failed_strategies"].append(
                            sr["strategy"]
                        )

            # Store meta-learning insights in Honcho
            if hasattr(self.solver, "active_session") and self.solver.active_session:
                session = self.solver.active_session

                # Separate successes and failures for clarity
                successful_strats = [sr["strategy"] for sr in strategy_results if sr["success"]]
                failed_strats = [sr["strategy"] for sr in strategy_results if not sr["success"]]

                meta_content = f"""META-LEARNING UPDATE (Learning from BOTH successes AND failures)

Pattern Insights (including failure patterns):
{json.dumps(self.pattern_insights, indent=2)}

Strategy Performance (successes AND failures):
{json.dumps(self.strategy_performance, indent=2)}

Recent Reflection:
{reflection.get('summary', 'N/A')}

Key Learnings from SUCCESSES:
{chr(10).join('- ' + learning for learning in reflection.get('key_learnings', []))}

Insights from FAILURES:
{chr(10).join('- ' + insight for insight in reflection.get('failure_insights', [])[:3])}

What worked on this task: {', '.join(successful_strats) if successful_strats else 'Nothing yet'}
What failed on this task: {', '.join(failed_strats) if failed_strats else 'N/A'}
"""

                await session.add_messages([{
                    "peer_id": "exploration_guide",
                    "content": meta_content,
                    "metadata": {
                        "type": "meta_learning_update",
                        "patterns": patterns,
                        "strategies_used": [sr["strategy"] for sr in strategy_results],
                        "successful_strategies": successful_strats,
                        "failed_strategies": failed_strats,  # NEW: Track failures
                    },
                }])

                logging.info("Updated meta-learning insights in Honcho")

        except Exception as e:
            logging.debug(f"Error updating meta-learning: {e}")

    async def _synthesize_learnings_via_dialectic(
        self,
        tui=None,
    ) -> Optional[Dict]:
        """
        Use Honcho's dialectic to SYNTHESIZE patterns across ALL past experiences.

        This is continuous learning - the agent reasons about its accumulated
        knowledge to build deeper understanding.
        """
        try:
            if not self._reflection_peer or not hasattr(self.solver, "active_session"):
                return None

            session = self.solver.active_session
            if not session:
                return None

            if tui:
                tui.add_agent_log(
                    "self_play",
                    "[Self-Play] 🧠 Synthesizing learnings across all experiences via dialectic...",
                )

            # Use dialectic to REASON about patterns in the agent's history
            synthesis_query = f"""I want to synthesize and reason about ALL my accumulated experiences.

Look across my ENTIRE memory of explorations - all successes, all failures, all attempts.
IMPORTANT: You can see the ACTUAL puzzle examples (input/output grids) that were stored in memory.

PATTERN SYNTHESIS:
1. What patterns do I see across my successful explorations?
   - Looking at the actual examples that succeeded, what strategies consistently work?
   - What visual/structural patterns in the grids lead to success?
   - What conditions lead to success?
   - Are there "rules of thumb" I can extract from the actual examples?

2. What patterns do I see across my FAILURES?
   - Looking at the actual examples that failed, what strategies consistently fail?
   - What visual/structural patterns in the grids cause failure?
   - What conditions lead to failure?
   - What mistakes do I keep making when I look at the actual failed examples?

3. What THEORIES can I build about problem-solving?
   - Based on actual examples in memory: When should I use strategy X vs strategy Y?
   - What types of grid transformations need what approaches?
   - What principles guide successful exploration based on real examples?

CONTINUOUS LEARNING:
4. How has my performance changed over time?
   - Am I getting better? In what ways?
   - What have I learned that I didn't know before?
   - What gaps remain in my knowledge?

5. What should I focus on learning next?
   - What patterns do I struggle with?
   - What strategies need more practice?
   - What new approaches should I develop?

META-REASONING:
6. How effective is my learning process itself?
   - Am I learning from failures effectively?
   - Are my theories accurate?
   - How can I learn better?

Based on ALL my accumulated memory, provide a deep synthesis as JSON:
- success_patterns: List of 3-5 patterns that lead to success
- failure_patterns: List of 3-5 patterns that lead to failure
- theories: List of 2-3 theories about problem-solving (when X works, when Y fails)
- learning_trajectory: How I've improved over time
- knowledge_gaps: What I still need to learn
- focus_areas: What to practice/develop next (2-3 areas)
- meta_insights: Insights about my learning process itself

Format as JSON."""

            synthesis_response = await self._reflection_peer.chat(
                synthesis_query
            )

            # Log dialectic query to TUI
            if tui:
                tui.add_memory_operation(
                    operation="Dialectic Query",
                    details="Learning synthesis",
                    num_results=1 if synthesis_response else 0
                )

            if not synthesis_response:
                return None

            content = synthesis_response.content if hasattr(synthesis_response, 'content') else str(synthesis_response)

            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                synthesis = json.loads(json_match.group(0))

                # Display key insights in TUI
                if tui:
                    if synthesis.get("theories"):
                        tui.add_agent_log(
                            "self_play",
                            f"[Self-Play] 💡 Theory: {synthesis['theories'][0][:100]}",
                        )
                    if synthesis.get("learning_trajectory"):
                        tui.add_agent_log(
                            "self_play",
                            f"[Self-Play] 📈 Progress: {synthesis['learning_trajectory'][:100]}",
                        )
                    if synthesis.get("focus_areas"):
                        tui.add_agent_log(
                            "self_play",
                            f"[Self-Play] 🎯 Focus on: {synthesis['focus_areas'][0][:80]}",
                        )

                # STORE this synthesis back into memory for future reasoning
                await self._ingest_thought(
                    task_id="continuous_learning",
                    thought_type="learning_synthesis",
                    content=f"Synthesized learnings: {synthesis}",
                    metadata={
                        "synthesis_type": "cross_task_patterns",
                        "theories": synthesis.get("theories", [],
            tui=tui,
        ),
                        "focus_areas": synthesis.get("focus_areas", []),
                    },
                )

                return synthesis

        except Exception as e:
            logging.debug(f"Error synthesizing learnings: {e}")
            return None

    async def _reason_about_strategy_effectiveness(
        self,
        strategy_name: str,
        patterns: List[str],
    ) -> Optional[Dict]:
        """
        Use dialectic to REASON deeply about why a strategy works or fails.

        This builds theories about strategy effectiveness through dialectic reasoning.
        """
        try:
            if not self._reflection_peer or not hasattr(self.solver, "active_session"):
                return None

            session = self.solver.active_session
            if not session:
                return None

            # Get strategy performance data
            strategy_data = self.strategy_performance.get(strategy_name, {})
            successes = strategy_data.get("successes", 0)
            failures = strategy_data.get("failures", 0)
            total = successes + failures

            if total == 0:
                return None

            success_rate = successes / total if total > 0 else 0

            patterns_str = ", ".join(patterns)

            # Use dialectic to REASON about why this strategy succeeds/fails
            reasoning_query = f"""Let me reason deeply about the strategy: {strategy_name}

CURRENT DATA:
- Success rate: {success_rate:.1%} ({successes}/{total} attempts)
- Current patterns: {patterns_str}
- Patterns it worked on: {strategy_data.get('patterns_worked', [])}
- Patterns it failed on: {strategy_data.get('patterns_failed', [])}

DEEP REASONING:
1. WHY does this strategy succeed when it succeeds?
   - What characteristics of the problem make it work?
   - What conditions are necessary for success?

2. WHY does this strategy fail when it fails?
   - What characteristics cause failure?
   - What is this strategy unable to handle?

3. Build a THEORY about when to use this strategy:
   - What types of problems is it good for?
   - What types should it avoid?
   - What patterns suggest using it?

4. How should I IMPROVE this strategy?
   - What modifications would increase success rate?
   - What complementary strategies should I combine with it?

5. For the CURRENT patterns ({patterns_str}):
   - Should I use this strategy? Why or why not?
   - What's my confidence level?
   - What adaptations would help?

Provide deep reasoning as JSON:
- success_theory: Why it succeeds (2-3 sentences)
- failure_theory: Why it fails (2-3 sentences)
- usage_rules: When to use vs avoid (2-3 rules)
- improvement_suggestions: How to improve (2-3 ideas)
- current_recommendation: For these patterns, should I use it? (yes/no/maybe)
- confidence: How confident (low/medium/high)
- reasoning: Why this recommendation

Format as JSON."""

            reasoning_response = await self._reflection_peer.chat(
                reasoning_query
            )

            # Log dialectic query to TUI
            if tui:
                tui.add_memory_operation(
                    operation="Dialectic Query",
                    details="Strategy reasoning",
                    num_results=1 if reasoning_response else 0
                )

            if not reasoning_response:
                return None

            content = reasoning_response.content if hasattr(reasoning_response, 'content') else str(reasoning_response)

            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                reasoning = json.loads(json_match.group(0))

                # STORE this reasoning back into memory - continuous learning!
                await self._ingest_thought(
                    task_id="strategy_reasoning",
                    thought_type="strategy_theory",
                    content=f"Theory about {strategy_name}: {reasoning}",
                    metadata={
                        "strategy": strategy_name,
                        "success_rate": success_rate,
                        "theory_type": "strategy_effectiveness",
                    },
                    tui=tui,
                )

                return reasoning

        except Exception as e:
            logging.debug(f"Error reasoning about strategy: {e}")
            return None

    async def _meta_reflect_on_learning_progress(
        self,
        tui=None,
    ) -> Optional[Dict]:
        """
        Use dialectic for META-REFLECTION: reasoning about the learning process itself.

        The agent reasons about how well it's learning, not just what it's learned.
        """
        try:
            if not self._reflection_peer or not hasattr(self.solver, "active_session"):
                return None

            session = self.solver.active_session
            if not session:
                return None

            if tui:
                tui.add_agent_log(
                    "self_play",
                    "[Self-Play] 🤔 Meta-reflecting on my learning process...",
                )

            # Get current statistics
            stats = self.get_exploration_statistics()

            # Use dialectic to META-REASON about learning
            meta_query = f"""Let me reflect on my LEARNING PROCESS itself - not just what I've learned, but HOW WELL I'm learning.

CURRENT STATE:
- Tasks explored: {stats.get('tasks_explored', 0)}
- Success rate: {stats.get('success_rate', 0):.1%}
- Primitives discovered: {stats.get('primitives_discovered', 0)}

Looking at ALL my accumulated experiences and learning history:

META-LEARNING ANALYSIS:
1. How effective is my learning process?
   - Am I improving over time?
   - Is my success rate increasing?
   - Am I learning from failures effectively?

2. What's working well in my learning approach?
   - What strategies for learning are effective?
   - What reflection practices help?
   - What memory patterns lead to insights?

3. What's NOT working in my learning?
   - Where am I stuck?
   - What mistakes do I keep repeating?
   - What blind spots do I have?

4. How can I learn BETTER?
   - What should I change about my learning process?
   - What new reflection practices should I try?
   - How can I better integrate past experiences?

5. What's my learning trajectory?
   - Am I on track?
   - What's my learning rate?
   - What bottlenecks exist?

CONTINUOUS IMPROVEMENT:
6. What experiments should I run on my learning process?
   - What new approaches to try?
   - What hypotheses to test?

Provide meta-reflection as JSON:
- learning_effectiveness: How well I'm learning (poor/fair/good/excellent)
- what_works: What's working in my learning (2-3 points)
- what_doesnt: What's not working (2-3 points)
- learning_trajectory: Am I improving? How fast?
- improvement_suggestions: How to learn better (3-5 concrete suggestions)
- experiments: Learning experiments to try (2-3)
- meta_insights: Deeper insights about learning itself

Format as JSON."""

            meta_response = await self._reflection_peer.chat(
                meta_query
            )

            # Log dialectic query to TUI
            if tui:
                tui.add_memory_operation(
                    operation="Dialectic Query",
                    details="Meta-learning",
                    num_results=1 if meta_response else 0
                )

            if not meta_response:
                return None

            content = meta_response.content if hasattr(meta_response, 'content') else str(meta_response)

            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                meta_reflection = json.loads(json_match.group(0))

                # Display in TUI
                if tui:
                    if meta_reflection.get("learning_effectiveness"):
                        tui.add_agent_log(
                            "self_play",
                            f"[Self-Play] 📊 Learning quality: {meta_reflection['learning_effectiveness']}",
                        )
                    if meta_reflection.get("improvement_suggestions"):
                        for suggestion in meta_reflection["improvement_suggestions"][:2]:
                            tui.add_agent_log(
                                "self_play",
                                f"[Self-Play] 💡 Improve: {suggestion[:80]}",
                            )

                # STORE meta-reflection back into memory - learning about learning!
                await self._ingest_thought(
                    task_id="meta_learning",
                    thought_type="learning_meta_reflection",
                    content=f"Meta-reflection on learning: {meta_reflection}",
                    metadata={
                        "learning_effectiveness": meta_reflection.get("learning_effectiveness",
            tui=tui,
        ),
                        "reflection_type": "meta_learning",
                    },
                )

                return meta_reflection

        except Exception as e:
            logging.debug(f"Error in meta-reflection: {e}")
            return None

    def get_exploration_statistics(self) -> Dict:
        """Get statistics about self-play exploration including meta-learning."""
        base_stats = {
            "tasks_explored": self.tasks_explored,
            "successful_explorations": self.successful_explorations,
            "success_rate": (
                self.successful_explorations / self.tasks_explored
                if self.tasks_explored > 0
                else 0.0
            ),
            "primitives_discovered": self.primitives_discovered,
        }

        # Add meta-learning stats
        if self.strategy_performance:
            strategy_stats = {}
            for strategy, data in self.strategy_performance.items():
                strategy_stats[strategy] = {
                    "attempts": data["attempts"],
                    "successes": data["successes"],
                    "success_rate": data["successes"] / data["attempts"] if data["attempts"] > 0 else 0.0,
                }
            base_stats["strategy_performance"] = strategy_stats

        if self.pattern_insights:
            base_stats["patterns_learned"] = len(self.pattern_insights)

        return base_stats
