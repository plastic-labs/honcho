"""
Core ARC-AGI-2 solver with proper Honcho memory integration.

This implements the architecture specified in the PDF documentation:
1. Memory Preparation Phase - Ingest training tasks into Honcho
2. Multi-Peer Architecture - task_analyst, solution_generator, verifier
3. Dialectic API - Natural language queries to memory
4. Semantic Search - Find similar tasks using Honcho's hybrid search
5. LLM Integration - Generate solutions with memory context
6. Iterative Refinement - Store attempts and feedback
7. Solution Storage - Store successful solutions back to Honcho
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import numpy as np

# Add Honcho SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "sdks" / "python" / "src"))

from honcho import AsyncHoncho, AsyncPeer, AsyncSession
from honcho.types import PeerContext

from .code_executor import SafeCodeExecutor
from .cognitive_layers import CognitiveLayers
from .config import ArceusConfig
from .logger import JSONTraceLogger
from .metrics import SolverMetrics
from .primitives import ARCPrimitives, PRIMITIVE_FUNCTIONS, Grid


class ARCSolver:
    """Memory-augmented ARC-AGI-2 solver using Honcho SDK."""

    def __init__(self, config: ArceusConfig):
        self.config = config
        self.honcho_client: Optional[AsyncHoncho] = None
        self.llm_client = None
        self.primitives = ARCPrimitives()
        self.code_executor = SafeCodeExecutor()

        # Peers for multi-agent architecture
        self.task_analyst_peer: Optional[AsyncPeer] = None
        self.solution_generator_peer: Optional[AsyncPeer] = None
        self.verifier_peer: Optional[AsyncPeer] = None

        # Medium-term enhancements (lazy loaded)
        self._ensemble_solver = None
        self._test_time_trainer = None
        self._pattern_learner = None

        # Primitive discovery system (lazy loaded)
        self._primitive_discovery = None
        self._self_play_explorer = None

        # Initialize LLM client
        if config.llm_provider == "anthropic":
            self.llm_client = anthropic.AsyncAnthropic(api_key=config.llm_api_key)

    @property
    def ensemble_solver(self):
        """Lazy load ensemble solver."""
        if self._ensemble_solver is None:
            from .ensemble import EnsembleSolver
            self._ensemble_solver = EnsembleSolver(self)
        return self._ensemble_solver

    @property
    def test_time_trainer(self):
        """Lazy load test-time trainer."""
        if self._test_time_trainer is None:
            from .test_time_training import TestTimeTrainer
            self._test_time_trainer = TestTimeTrainer(self, self.config.trace_output_dir)
        return self._test_time_trainer

    @property
    def pattern_learner(self):
        """Lazy load pattern learner."""
        if self._pattern_learner is None:
            from .pattern_learning import PatternLearner
            self._pattern_learner = PatternLearner(self.config.trace_output_dir)
        return self._pattern_learner

    @property
    def primitive_discovery(self):
        """Lazy load primitive discovery system (Honcho-backed)."""
        if self._primitive_discovery is None:
            from .primitive_discovery import PrimitiveDiscoverySystem
            self._primitive_discovery = PrimitiveDiscoverySystem(
                solver=self,
                # All storage handled by Honcho - no local files
            )
        return self._primitive_discovery

    @property
    def self_play_explorer(self):
        """Lazy load self-play explorer."""
        if self._self_play_explorer is None:
            from .self_play import SelfPlayExplorer
            self._self_play_explorer = SelfPlayExplorer(
                solver=self,
                primitive_discovery=self.primitive_discovery,
            )
        return self._self_play_explorer

    async def initialize(self):
        """Initialize the solver and Honcho connection."""
        if not self.config.enable_memory:
            return

        # Initialize Honcho client using the SDK
        self.honcho_client = AsyncHoncho(
            base_url=self.config.honcho_url,
            workspace_id=self.config.workspace_id,
        )

        # Create peers with proper metadata
        # When passing metadata, peer() is async and needs to be awaited
        self.task_analyst_peer = await self.honcho_client.peer(
            "task_analyst",
            metadata={
                "role": "task_analyst",
                "purpose": "Analyzes ARC task structure and identifies patterns",
                "capabilities": ["pattern_detection", "structure_analysis", "feature_extraction"],
            },
        )

        self.solution_generator_peer = await self.honcho_client.peer(
            "solution_generator",
            metadata={
                "role": "solution_generator",
                "purpose": "Generates candidate transformation solutions",
                "capabilities": ["transformation_synthesis", "code_generation", "hypothesis_formation"],
            },
        )

        self.verifier_peer = await self.honcho_client.peer(
            "verifier",
            metadata={
                "role": "verifier",
                "purpose": "Validates solutions against training examples",
                "capabilities": ["solution_verification", "error_analysis", "feedback_generation"],
            },
        )

    async def prepare_memory_phase(self, training_data_path: Path, tui=None):
        """
        Memory Preparation Phase: Ingest all training tasks into Honcho.

        This implements Phase 1 from the PDF specification:
        - Ingest all 1000 training tasks
        - Store raw JSON, natural language descriptions, solutions
        - Use LLM to explain each task
        - Derive primitives and tags
        - Build primitive library with explicit documentation
        """
        if not self.honcho_client:
            return

        if tui:
            tui.add_agent_log("memory_prep", "Starting memory preparation phase...")

        training_files = list(training_data_path.glob("*.json"))

        for idx, task_file in enumerate(training_files[:10]):  # Start with first 10 for testing
            task_id = task_file.stem

            if tui:
                tui.add_agent_log("memory_prep", f"Ingesting task {idx + 1}/{len(training_files[:10])}: {task_id}")

            # Load task data
            with open(task_file) as f:
                task_data = json.load(f)

            # Create a session for this task
            task_session = await self.honcho_client.session(
                f"training_task_{task_id}",
                metadata={
                    "type": "training_task",
                    "task_id": task_id,
                    "num_examples": len(task_data["train"]),
                },
            )

            # Add peers to session
            await task_session.add_peers([
                self.task_analyst_peer,
                self.solution_generator_peer,
                self.verifier_peer,
            ])

            # Store task analysis as messages
            analysis = await self._analyze_task_structure(task_data)

            # Add analysis message from task_analyst
            await task_session.add_messages([{
                "peer_id": "task_analyst",
                "content": json.dumps({
                    "task_id": task_id,
                    "analysis": analysis,
                    "raw_task": task_data,
                }),
                "metadata": {
                    "type": "task_analysis",
                    "task_id": task_id,
                },
            }])

            # Generate natural language description using LLM
            description = await self._generate_task_description(task_data, analysis)

            # Store description
            await task_session.add_messages([{
                "peer_id": "task_analyst",
                "content": f"Task Description: {description}",
                "metadata": {
                    "type": "task_description",
                    "task_id": task_id,
                },
            }])

            # Identify and store primitives
            primitives = self._identify_primitives(task_data)
            await task_session.add_messages([{
                "peer_id": "solution_generator",
                "content": f"Identified primitives: {', '.join(primitives)}",
                "metadata": {
                    "type": "primitives",
                    "task_id": task_id,
                    "primitives": primitives,
                },
            }])

            # Wait for deriver to process
            await asyncio.sleep(0.5)

        if tui:
            tui.add_agent_log("memory_prep", f"Memory preparation complete! Ingested {len(training_files[:10])} tasks.")

    async def prepare_memory_with_self_play(
        self, training_data_path: Path, limit: int = None, tui=None
    ):
        """
        Memory Preparation with Self-Play Mode.

        Instead of just ingesting task descriptions, the agent actively:
        1. Explores training tasks by attempting to solve them
        2. Discovers new transformation primitives
        3. Learns which primitives work for which patterns
        4. Stores discovered primitives in Honcho memory

        This builds the agent's transformation vocabulary dynamically.
        """
        if not self.honcho_client:
            import logging
            logging.warning("Honcho client not initialized, skipping self-play")
            return

        if tui:
            tui.add_agent_log("self_play", "🎮 Starting self-play exploration mode...")

        training_files = list(training_data_path.glob("*.json"))

        # Determine how many tasks to explore
        num_tasks = limit if limit else len(training_files)
        tasks_to_explore = training_files[:num_tasks]

        if tui:
            tui.add_agent_log(
                "self_play", f"Will explore {len(tasks_to_explore)} training tasks"
            )

        # Statistics
        total_primitives_discovered = 0
        successful_explorations = 0

        # Create a session for self-play exploration
        self.active_session = await self.honcho_client.session(
            "self_play_exploration",
            metadata={
                "type": "self_play_exploration",
                "num_tasks": len(tasks_to_explore),
            },
        )

        # Add peers to session
        await self.active_session.add_peers([
            self.task_analyst_peer,
            self.solution_generator_peer,
            self.verifier_peer,
        ])

        # Explore each task
        for idx, task_file in enumerate(tasks_to_explore, 1):
            task_id = task_file.stem

            if tui:
                tui.add_agent_log(
                    "self_play",
                    f"[{idx}/{len(tasks_to_explore)}] Exploring task {task_id}...",
                )

            import logging
            logging.info(f"Self-play exploration {idx}/{len(tasks_to_explore)}: {task_id}")

            # Load task data
            with open(task_file) as f:
                task_data = json.load(f)

            # Explore the task
            try:
                result = await self.self_play_explorer.explore_task(
                    task_id, task_data, tui
                )

                if result["solved"]:
                    successful_explorations += 1

                total_primitives_discovered += len(result["primitives_discovered"])

                if tui and result["primitives_discovered"]:
                    tui.add_agent_log(
                        "self_play",
                        f"✓ Task {task_id}: Discovered {len(result['primitives_discovered'])} primitives",
                    )

            except Exception as e:
                import logging
                logging.error(f"Error exploring task {task_id}: {e}")
                import traceback
                traceback.print_exc()
                if tui:
                    tui.add_agent_log("self_play", f"✗ Error exploring {task_id}: {str(e)}")

            # Small delay between tasks
            await asyncio.sleep(0.1)

        # Print statistics
        exploration_stats = self.self_play_explorer.get_exploration_statistics()
        primitive_stats = self.primitive_discovery.get_primitive_statistics()

        if tui:
            tui.add_agent_log("self_play", "")
            tui.add_agent_log("self_play", "=== Self-Play Exploration Complete ===")
            tui.add_agent_log(
                "self_play",
                f"Tasks explored: {exploration_stats['tasks_explored']}",
            )
            tui.add_agent_log(
                "self_play",
                f"Successful: {exploration_stats['successful_explorations']} ({exploration_stats['success_rate']:.1%})",
            )
            tui.add_agent_log(
                "self_play",
                f"Primitives discovered: {total_primitives_discovered}",
            )
            tui.add_agent_log(
                "self_play",
                f"Total primitives in library: {primitive_stats['total']}",
            )

        import logging
        logging.info("=== Self-Play Exploration Statistics ===")
        logging.info(f"Tasks explored: {exploration_stats['tasks_explored']}")
        logging.info(
            f"Successful: {exploration_stats['successful_explorations']} ({exploration_stats['success_rate']:.1%})"
        )
        logging.info(f"Primitives discovered: {total_primitives_discovered}")
        logging.info(f"Total primitives in library: {primitive_stats['total']}")
        logging.info(f"Primitive statistics: {json.dumps(primitive_stats, indent=2)}")

    async def _analyze_task_structure(self, task_data: Dict) -> Dict[str, Any]:
        """Analyze the task structure and extract features."""
        train_examples = task_data["train"]

        analysis = {
            "num_examples": len(train_examples),
            "input_shapes": [
                (len(ex["input"]), len(ex["input"][0])) for ex in train_examples
            ],
            "output_shapes": [
                (len(ex["output"]), len(ex["output"][0])) for ex in train_examples
            ],
            "colors_used": set(),
            "patterns": {},
            "symmetries": [],
        }

        # Analyze each example
        for example in train_examples:
            input_grid = example["input"]
            output_grid = example["output"]

            input_pattern = ARCPrimitives.detect_pattern(input_grid)
            output_pattern = ARCPrimitives.detect_pattern(output_grid)

            analysis["colors_used"].update(input_pattern["colors"].keys())
            analysis["colors_used"].update(output_pattern["colors"].keys())

            # Check symmetries
            input_symmetry = ARCPrimitives.get_symmetry(input_grid)
            if input_symmetry not in analysis["symmetries"]:
                analysis["symmetries"].append(input_symmetry)

        analysis["colors_used"] = list(analysis["colors_used"])

        return analysis

    async def _generate_task_description(self, task_data: Dict, analysis: Dict) -> str:
        """Generate a natural language description of the task using LLM."""
        if not self.llm_client:
            return "Task analysis not available (no LLM configured)"

        prompt = f"""Analyze this ARC-AGI-2 task and provide a concise natural language description of the transformation rule.

TASK STRUCTURE:
- Number of training examples: {analysis['num_examples']}
- Input shapes: {analysis['input_shapes']}
- Output shapes: {analysis['output_shapes']}
- Colors used: {analysis['colors_used']}
- Symmetries detected: {analysis['symmetries']}

TRAINING EXAMPLES:
"""
        for idx, example in enumerate(task_data["train"][:2]):  # Show first 2 examples
            # Ensure example is a dictionary
            if not isinstance(example, dict):
                continue
            prompt += f"\nExample {idx + 1}:\nInput:\n{self._format_grid(example['input'])}\n"
            prompt += f"Output:\n{self._format_grid(example['output'])}\n"

        prompt += "\n\nProvide a 1-2 sentence description of the transformation rule."

        try:
            response = await self.llm_client.messages.create(
                model=self.config.llm_model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            return f"Error generating description: {str(e)}"

    def _identify_primitives(self, task_data: Dict) -> List[str]:
        """Identify which primitive operations might be relevant for this task."""
        primitives = []

        # Try each primitive on first training example to see if it might be relevant
        first_example = task_data["train"][0]
        input_grid = first_example["input"]

        for prim_name, prim_func in PRIMITIVE_FUNCTIONS.items():
            try:
                result = prim_func(input_grid)
                if result is not None:
                    primitives.append(prim_name)
            except Exception:
                pass

        return primitives if primitives else ["rotate_90", "flip_horizontal", "flip_vertical"]

    async def query_memory_dialectic(
        self, query: str, peer: AsyncPeer, logger: JSONTraceLogger, metrics: SolverMetrics, tui=None
    ) -> Optional[str]:
        """
        Use Dialectic API to query memory conversationally.

        This implements the Dialectic API from the PDF specification.
        Example: "Have I solved a puzzle like this before?"
        """
        if not self.honcho_client or not peer:
            return None

        if tui:
            tui.add_agent_log("dialectic", f"Querying memory: {query[:100]}...")

        start_time = time.time()
        try:
            response = await peer.chat(query)
            query_time_ms = (time.time() - start_time) * 1000

            metrics.add_memory_query(query_time_ms, 1 if response else 0)
            logger.log_memory_query("dialectic", query, response)

            if tui and response:
                tui.add_memory_operation("Dialectic Query", f"Query successful ({query_time_ms:.0f}ms)", 1)

            return response
        except Exception as e:
            logger.log_error("dialectic_error", str(e))
            return None

    async def get_peer_context(
        self, peer: AsyncPeer, search_query: str, logger: JSONTraceLogger, metrics: SolverMetrics, tui=None
    ) -> Optional[PeerContext]:
        """
        Get peer context using get_context() API.

        This implements the get_context() API from the PDF specification.
        """
        if not peer:
            return None

        if tui:
            tui.add_agent_log("context", f"Retrieving peer context with query: {search_query[:100]}...")

        start_time = time.time()
        try:
            context = await peer.get_context(
                search_query=search_query,
                search_top_k=10,
                include_most_derived=True,
            )
            query_time_ms = (time.time() - start_time) * 1000

            metrics.add_memory_query(query_time_ms, len(context.representation.observations) if context.representation else 0)

            if tui and context:
                tui.add_memory_operation(
                    "Context Retrieval",
                    f"Retrieved context ({query_time_ms:.0f}ms)",
                    len(context.representation.observations) if context.representation else 0,
                )

            return context
        except Exception as e:
            logger.log_error("context_error", str(e))
            return None

    async def generate_solution_with_memory(
        self,
        task_data: Dict,
        analysis: Dict,
        iteration: int,
        logger: JSONTraceLogger,
        metrics: SolverMetrics,
        tui=None,
        solving_context=None,
    ) -> Dict[str, Any]:
        """
        Generate solution using LLM with memory context.

        This implements the LLM integration with memory context from the PDF specification.
        """
        if not self.llm_client:
            return {"error": "No LLM client configured"}

        metrics.num_hypotheses_generated += 1

        # Get actionable solution guidance from memory
        solution_guidance = None
        if solving_context and self.solution_generator_peer:
            from .cognitive_layers import CognitiveLayers
            cognitive = CognitiveLayers(self.solution_generator_peer, self.honcho_client, tui)

            if tui:
                tui.add_agent_log("memory", "🧠 Querying memory for 'HOW TO SOLVE' guidance...")

            # Get meta-strategy if available
            meta_strategy = solving_context.get('meta_strategy') if solving_context else None

            solution_guidance = await cognitive.get_solution_guidance_from_memory(
                task_id=task_data.get('task_id', 'unknown'),
                task_analysis=analysis,
                meta_strategy=meta_strategy,
                tui_label="solving"
            )

        # Fallback: Query memory for similar tasks using Dialectic API (if no guidance available)
        memory_response = None
        if not solution_guidance and self.solution_generator_peer:
            dialectic_query = f"""Have I solved a task similar to this before?
Task has {analysis['num_examples']} examples.
Input shapes: {analysis['input_shapes']}.
Output shapes: {analysis['output_shapes']}.
Colors used: {analysis['colors_used']}."""

            memory_response = await self.query_memory_dialectic(
                dialectic_query,
                self.solution_generator_peer,
                logger,
                metrics,
                tui,
            )

        # Get peer context for relevant facts
        context = None
        if self.solution_generator_peer:
            search_query = f"ARC task transformation with shapes {analysis['input_shapes']} to {analysis['output_shapes']}"
            context = await self.get_peer_context(
                self.solution_generator_peer,
                search_query,
                logger,
                metrics,
                tui,
            )

        # Build prompt with memory context and solution guidance
        prompt = self._build_solution_prompt(task_data, analysis, memory_response, context, solution_guidance)

        if tui:
            tui.add_agent_log("llm", f"Generating solution with memory context (iteration {iteration + 1})...")

        start_time = time.time()
        try:
            response = await self.llm_client.messages.create(
                model=self.config.llm_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            call_time_ms = (time.time() - start_time) * 1000

            content = response.content[0].text
            tokens_used = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            }
            metrics.add_llm_call(call_time_ms, tokens_used)
            metrics.calculate_api_cost()  # Update cost after each LLM call

            hypothesis = self._parse_llm_response(content)
            logger.log_reasoning_step(
                iteration, "solution_generation", content, hypothesis.get("confidence", 0.5)
            )

            # Attach solution guidance to hypothesis for later storage
            if solution_guidance:
                hypothesis["_solution_guidance"] = solution_guidance

            if tui:
                tui.add_agent_log("llm", f"Generated solution (confidence: {hypothesis.get('confidence', 0.5):.2f})")

            return hypothesis

        except Exception as e:
            logger.log_error("llm_error", str(e))
            if tui:
                tui.add_agent_log("error", f"LLM error: {str(e)}")
            return {"error": str(e)}

    def _build_solution_prompt(
        self,
        task_data: Dict,
        analysis: Dict,
        memory_response: Optional[str],
        context: Optional[PeerContext],
        solution_guidance: Optional[Dict] = None
    ) -> str:
        """Build the prompt for solution generation with memory context and actionable guidance."""
        train_examples = task_data["train"]

        prompt = f"""You are an expert at solving ARC-AGI-2 (Abstraction and Reasoning Corpus) puzzles.

TASK ANALYSIS:
- Number of training examples: {analysis['num_examples']}
- Input shapes: {analysis['input_shapes']}
- Output shapes: {analysis['output_shapes']}
- Colors used: {analysis['colors_used']}
"""

        # Add solution guidance from memory (HIGH PRIORITY - specific actionable advice)
        if solution_guidance and solution_guidance.get('has_experience'):
            confidence = solution_guidance.get('confidence', 0.0)
            confidence_label = "HIGH" if confidence > 0.7 else "MEDIUM" if confidence > 0.4 else "LOW"

            prompt += f"\n\n{'='*70}\n🎯 MEMORY GUIDANCE: HOW TO SOLVE (Confidence: {confidence_label} - {confidence:.2f})\n{'='*70}\n"

            # Solving strategy
            if solution_guidance.get('solving_strategy'):
                prompt += f"\n📋 STRATEGY:\n{solution_guidance['solving_strategy']}\n"

            # Code approach
            if solution_guidance.get('code_approach'):
                prompt += f"\n🔧 CODE APPROACH:\n{solution_guidance['code_approach']}\n"

            # Step-by-step instructions
            if solution_guidance.get('step_by_step'):
                prompt += "\n📝 STEP-BY-STEP INSTRUCTIONS:\n"
                for idx, step in enumerate(solution_guidance['step_by_step'], 1):
                    prompt += f"{idx}. {step}\n"

            # Recommended primitives
            if solution_guidance.get('primitives_to_use'):
                primitives_str = ", ".join(solution_guidance['primitives_to_use'])
                prompt += f"\n🛠️  RECOMMENDED PRIMITIVES:\n{primitives_str}\n"

            # Pitfalls to avoid
            if solution_guidance.get('pitfalls_to_avoid'):
                prompt += "\n⚠️  PITFALLS TO AVOID:\n"
                for idx, pitfall in enumerate(solution_guidance['pitfalls_to_avoid'], 1):
                    prompt += f"- {pitfall}\n"

            # Example code pattern
            if solution_guidance.get('example_code_pattern'):
                prompt += f"\n💡 EXAMPLE CODE PATTERN FROM MEMORY:\n{solution_guidance['example_code_pattern']}\n"

            # Reasoning
            if solution_guidance.get('reasoning'):
                prompt += f"\n🧠 WHY THIS SHOULD WORK:\n{solution_guidance['reasoning']}\n"

            prompt += f"\n{'='*70}\n\n"
            prompt += "IMPORTANT: Use the above guidance to inform your solution. Translate the natural language strategy into concrete Python code.\n"

        # Add generic memory context if available (fallback)
        elif memory_response:
            prompt += f"\n\nMEMORY CONTEXT:\n{memory_response}\n"

        if context and context.representation and context.representation.observations:
            prompt += "\n\nRELEVANT FACTS FROM MEMORY:\n"
            for idx, obs in enumerate(context.representation.observations[:5], 1):
                prompt += f"{idx}. {obs.content}\n"

        prompt += "\n\nTRAINING EXAMPLES:\n"

        for idx, example in enumerate(train_examples[:3]):  # Show first 3 examples
            prompt += f"\nExample {idx + 1}:\nInput:\n{self._format_grid(example['input'])}\n"
            prompt += f"Output:\n{self._format_grid(example['output'])}\n"

        prompt += """

AVAILABLE PRIMITIVES:
You can use these primitive functions in your code:
- rotate_90, rotate_180, rotate_270, flip_horizontal, flip_vertical
- transpose, tile_grid, scale_grid, crop_grid, pad_grid
- replace_color, fill_background, invert_colors
- extract_objects, extract_largest_object, compress_grid
- overlay_grids, apply_to_each_object, gravity_down
- draw_border, repeat_pattern

TASK:
Analyze the transformation rule that converts inputs to outputs. Generate Python code to perform the transformation.

Provide your analysis in this format:

HYPOTHESIS: [Describe the transformation rule in 1-2 sentences]

CODE:
```python
def transform(grid):
    # Your transformation code here
    # grid is a list of lists of integers
    # Return the transformed grid

    import numpy as np

    # Example: rotate the grid 90 degrees
    arr = np.array(grid)
    result = np.rot90(arr, k=-1)
    return result.tolist()
```

CONFIDENCE: [0.0 to 1.0]

IMPORTANT:
- Write actual executable Python code
- The function must be named 'transform' and take 'grid' as parameter
- Use numpy operations or the primitive functions listed above
- Focus on what changes between input and output
- Use insights from memory if available
- Be specific and precise"""

        return prompt

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data."""
        hypothesis = {
            "raw_response": response,
            "transformation_rule": "",
            "code": "",
            "primitives": [],
            "approach": "",
            "confidence": 0.5,
        }

        # Parse hypothesis
        if "HYPOTHESIS:" in response:
            if "CODE:" in response:
                hypothesis["transformation_rule"] = response.split("HYPOTHESIS:")[1].split("CODE:")[0].strip()
            elif "CONFIDENCE:" in response:
                hypothesis["transformation_rule"] = response.split("HYPOTHESIS:")[1].split("CONFIDENCE:")[0].strip()
            else:
                hypothesis["transformation_rule"] = response.split("HYPOTHESIS:")[1].strip()

        # Parse CODE block
        if "```python" in response:
            # Extract code between ```python and ```
            parts = response.split("```python")
            if len(parts) > 1:
                code_part = parts[1].split("```")[0]
                hypothesis["code"] = code_part.strip()
        elif "CODE:" in response:
            # Try without backticks
            parts = response.split("CODE:")[1].split("CONFIDENCE:")
            hypothesis["code"] = parts[0].strip()

        # Also extract primitives mentioned (for fallback)
        for prim_name in PRIMITIVE_FUNCTIONS.keys():
            if prim_name in response:
                hypothesis["primitives"].append(prim_name)

        # Parse confidence
        if "CONFIDENCE:" in response:
            try:
                conf_text = response.split("CONFIDENCE:")[1].strip().split()[0]
                hypothesis["confidence"] = float(conf_text)
            except (ValueError, IndexError):
                pass

        return hypothesis

    def _format_grid(self, grid: Grid) -> str:
        """Format a grid for display."""
        return "\n".join(["  " + " ".join(str(cell) for cell in row) for row in grid])

    async def store_solution(
        self,
        task_id: str,
        solution: Dict,
        success: bool,
        logger: JSONTraceLogger,
        tui=None,
        metrics=None,
        solving_context=None,
        solution_guidance=None
    ):
        """
        Store solution back to Honcho for future retrieval.

        This implements solution storage from the PDF specification.
        Now also stores the solution guidance that was used and whether it was helpful.
        """
        if not self.honcho_client or not self.solution_generator_peer:
            return

        if tui:
            tui.add_agent_log("storage", f"Storing solution for task {task_id}...")

        # Create session for solution
        solution_session = await self.honcho_client.session(
            f"solution_{task_id}",
            metadata={
                "type": "solution",
                "task_id": task_id,
                "success": success,
            },
        )

        # Add peers
        await solution_session.add_peers([self.solution_generator_peer, self.verifier_peer])

        # Build comprehensive solution record including guidance used
        solution_record = {
            "task_id": task_id,
            "solution": solution,
            "success": success,
        }

        # Add guidance information if available
        if solution_guidance:
            solution_record["guidance_used"] = {
                "had_memory_guidance": solution_guidance.get('has_experience', False),
                "confidence": solution_guidance.get('confidence', 0.0),
                "solving_strategy": solution_guidance.get('solving_strategy', ''),
                "code_approach": solution_guidance.get('code_approach', ''),
                "primitives_recommended": solution_guidance.get('primitives_to_use', []),
                "guidance_was_helpful": success,  # If we succeeded, guidance was helpful
            }

        # Add solving context if available
        if solving_context:
            if solving_context.get('meta_strategy'):
                solution_record["meta_strategy_used"] = {
                    "problem_type": solving_context['meta_strategy'].get('problem_type', ''),
                    "approach_type": solving_context['meta_strategy'].get('approach_type', ''),
                }

        # Store solution details
        await solution_session.add_messages([{
            "peer_id": "solution_generator",
            "content": json.dumps(solution_record),
            "metadata": {
                "type": "solution_record",
                "task_id": task_id,
                "success": success,
                "had_guidance": bool(solution_guidance and solution_guidance.get('has_experience')),
            },
        }])
        if metrics:
            metrics.num_messages_ingested += 1
            metrics.add_peer_fact("solution_generator")

        logger.log_event("solution_stored", {"task_id": task_id, "success": success})

        if tui:
            tui.add_agent_log("storage", f"Solution stored successfully")

    async def verify_solution(
        self, solution_grid: Grid, expected_grid: Grid, logger: JSONTraceLogger
    ) -> bool:
        """Verify if a solution matches the expected output."""
        if len(solution_grid) != len(expected_grid):
            return False
        if len(solution_grid[0]) != len(expected_grid[0]):
            return False

        for i in range(len(solution_grid)):
            for j in range(len(solution_grid[0])):
                if solution_grid[i][j] != expected_grid[i][j]:
                    return False

        return True

    def _try_primitive_transformation(
        self, input_grid: Grid, primitive_name: str, params: Dict = None
    ) -> Optional[Grid]:
        """Try applying a primitive transformation."""
        try:
            primitive_func = PRIMITIVE_FUNCTIONS.get(primitive_name)
            if not primitive_func:
                return None

            if params:
                result = primitive_func(input_grid, **params)
            else:
                result = primitive_func(input_grid)

            # Validate that the result is actually a Grid (list of lists of integers)
            # Some primitives return other structures like dicts or tuples
            if not isinstance(result, list):
                return None

            if len(result) == 0:
                return None

            # Check if it's a proper 2D grid
            if not isinstance(result[0], list):
                return None

            # Check if cells are integers (not lists or other types)
            for row in result:
                if not isinstance(row, list):
                    return None
                for cell in row:
                    if isinstance(cell, list):
                        return None
                    if not isinstance(cell, (int, float, np.integer)):
                        return None

            return result
        except Exception:
            return None

    async def solve_task(
        self, task_id: str, task_data: Dict, logger: JSONTraceLogger, metrics: SolverMetrics, tui=None
    ) -> Optional[Grid]:
        """
        Solve an ARC task using memory-augmented reasoning.

        Implements the complete solving workflow from the PDF specification:
        1. Task Input - Create session for task
        2. Memory Retrieval - Use semantic search + Dialectic API
        3. LLM Reasoning - Generate code/solution with memory context
        4. Execution & Feedback - Verify on training examples
        5. Iterative Refinement - Store feedback, retry with updated context
        6. Solution Finalization - Store successful solution in Honcho

        Args:
            task_id: Task identifier
            task_data: Task data with train and test examples
            logger: JSON trace logger
            metrics: Metrics tracker
            tui: Optional TUI for visualization

        Returns:
            Predicted output grid or None if unsolved
        """
        metrics.task_id = task_id
        metrics.model_name = self.config.llm_model  # Track model for cost calculation
        logger.log_event("task_start", {"task_id": task_id})

        # Create session for this solving attempt
        solving_session = None
        if self.honcho_client:
            solving_session = await self.honcho_client.session(
                f"solve_{task_id}",
                metadata={"type": "solving_session", "task_id": task_id},
            )
            metrics.num_sessions_created += 1  # Track session creation
            await solving_session.add_peers([
                self.task_analyst_peer,
                self.solution_generator_peer,
                self.verifier_peer,
            ])

        # Phase 1: Analyze task
        if tui:
            tui.add_agent_log("analysis", "Analyzing task structure and patterns...")
        analysis = await self._analyze_task_structure(task_data)
        metrics.num_reasoning_steps += 1

        logger.log_event("task_analysis", analysis)

        # Get test input for visualization
        first_test = task_data["test"][0] if task_data.get("test") else {}
        # Ensure first_test is a dictionary
        if isinstance(first_test, dict):
            test_input = first_test.get("input", [])
            test_expected = first_test.get("output")  # May not exist
        else:
            test_input = []
            test_expected = None

        # Phase 1.5: Cognitive Layers - Strategy Selection and Meta-Cognitive Planning
        solving_context = None
        if self.honcho_client and self.task_analyst_peer:
            cognitive = CognitiveLayers(self.task_analyst_peer, self.honcho_client, tui)

            # Strategy Selection Layer: Should we use deep exploration?
            if tui:
                tui.add_agent_log("strategy_selection", "🤔 Consulting memory: Should I use deep exploration for this task?")

            strategy_decision = await cognitive.should_use_deep_exploration(
                task_id=task_id,
                task_analysis=analysis,
                tui_label="solving"
            )

            use_deep_exploration = strategy_decision.get('use_deep_exploration', True)  # Default True

            if tui:
                decision_text = "YES - Using deep exploration" if use_deep_exploration else "NO - Using fast baseline"
                tui.add_agent_log("strategy_selection", f"🎯 Decision: {decision_text}")
                reasoning = strategy_decision.get('reasoning', 'N/A')
                tui.add_agent_log("strategy_selection", f"💡 Reasoning: {reasoning[:80]}")

            # Only engage three-layer system if memory suggests it's beneficial
            if use_deep_exploration:
                # Layer 1: Meta-strategy planning
                if tui:
                    tui.add_agent_log("meta_strategy", "🧩 META-STRATEGY: Planning how to think about this puzzle...")

                meta_strategy = await cognitive.meta_strategy_planning(
                    task_id=task_id,
                    task_patterns=analysis,
                    tui_label="solving"
                )

                if meta_strategy and tui:
                    tui.add_agent_log("meta_strategy", f"Problem type: {meta_strategy.get('problem_type', 'unknown')}")
                    tui.add_agent_log("meta_strategy", f"Approach: {meta_strategy.get('approach_type', 'analytical')}")

                # Layer 2: Adaptive memory reflection
                if meta_strategy:
                    if tui:
                        tui.add_agent_log("memory", "🧠 MEMORY: Querying memory adaptively based on problem type...")

                    memory_reflection = await cognitive.adaptive_memory_reflection(
                        task_id=task_id,
                        meta_strategy=meta_strategy,
                        tui_label="solving"
                    )

                    if memory_reflection and tui:
                        insights = memory_reflection.get('key_insights', [])
                        if insights and isinstance(insights, list):
                            for insight in insights[:2]:
                                insight_str = str(insight) if not isinstance(insight, str) else insight
                                tui.add_agent_log("memory", f"💭 Insight: {insight_str[:80]}")

                    # Store context for iteration loop
                    solving_context = {
                        'use_deep_exploration': True,
                        'meta_strategy': meta_strategy,
                        'memory_reflection': memory_reflection,
                        'strategy_decision': strategy_decision,
                    }
                else:
                    # Meta-strategy failed, use fast baseline
                    solving_context = {
                        'use_deep_exploration': False,
                        'strategy_decision': strategy_decision,
                    }
            else:
                # Fast baseline solving - no deep exploration
                solving_context = {
                    'use_deep_exploration': False,
                    'strategy_decision': strategy_decision,
                }

        # Phase 2: Iterative solving with memory-augmented generation
        for iteration in range(self.config.max_iterations):
            metrics.num_iterations += 1

            if tui:
                tui.add_agent_log("iteration", f"Starting iteration {iteration + 1}/{self.config.max_iterations}")

            # Generate solution using LLM with memory context
            hypothesis = await self.generate_solution_with_memory(
                task_data, analysis, iteration, logger, metrics, tui, solving_context
            )

            if "error" in hypothesis:
                if tui:
                    tui.add_agent_log("error", f"Failed to generate solution: {hypothesis.get('error', 'Unknown error')}")
                continue

            if tui:
                tui.add_agent_log("hypothesis", f"Generated hypothesis with confidence {hypothesis.get('confidence', 0):.2f}")
                if hypothesis.get("code"):
                    tui.add_agent_log("code", "Generated transformation code")
                elif hypothesis.get("primitives"):
                    tui.add_agent_log("primitives", f"Primitives to try: {', '.join(hypothesis.get('primitives', []))}")

            # Try generated code first if available
            solution_found = False

            if hypothesis.get("code"):
                if tui:
                    tui.add_agent_log("execution", "Executing generated transformation code...")
                    tui.update_transformation_attempt(iteration + 1, "Generated Code", None)

                await asyncio.sleep(0.5)

                # Try code on first training example
                first_train = task_data["train"][0]
                # Ensure first_train is a dictionary
                if not isinstance(first_train, dict):
                    continue
                results = self.code_executor.try_multiple_variations(hypothesis["code"], first_train["input"])

                if results:
                    if tui:
                        tui.add_agent_log("execution", f"Code executed successfully, got {len(results)} result(s)")

                    # Try each result variation
                    for var_idx, result in enumerate(results):
                        if tui:
                            tui.update_transformation_attempt(iteration + 1, f"Generated Code (variant {var_idx + 1})", result)
                            await asyncio.sleep(0.5)

                        metrics.add_transformation_attempt("generated_code")
                        logger.log_transformation_attempt("generated_code", first_train["input"], result, True)

                        # Verify on all training examples
                        all_passed = True
                        for ex_idx, example in enumerate(task_data["train"]):
                            # Ensure example is a dictionary
                            if not isinstance(example, dict):
                                continue
                            attempt_result = self.code_executor.execute_transformation(hypothesis["code"], example["input"])
                            metrics.num_verifications += 1

                            if attempt_result and await self.verify_solution(attempt_result, example["output"], logger):
                                if tui:
                                    tui.add_agent_log("verify", f"✓ Passed training example {ex_idx + 1}")
                                logger.log_verification(ex_idx, example["output"], attempt_result, True, None)
                            else:
                                metrics.num_failed_verifications += 1
                                all_passed = False
                                if tui:
                                    tui.add_agent_log("verify", f"✗ Failed training example {ex_idx + 1}")
                                logger.log_verification(ex_idx, example["output"], attempt_result, False, "Output mismatch")

                                # Store feedback for iterative refinement
                                if solving_session and self.verifier_peer:
                                    await solving_session.add_messages([{
                                        "peer_id": "verifier",
                                        "content": f"Generated code failed on training example {ex_idx + 1}",
                                        "metadata": {
                                            "type": "feedback",
                                            "approach": "generated_code",
                                            "example_idx": ex_idx,
                                            "success": False,
                                        },
                                    }])
                                    metrics.num_messages_ingested += 1
                                    metrics.add_peer_fact("verifier")
                                break

                        if all_passed:
                            # Apply to test case
                            test_result = self.code_executor.execute_transformation(hypothesis["code"], test_input)
                            if test_result:
                                solution_found = True
                                if tui:
                                    tui.update_output(test_result)
                                    tui.add_agent_log("success", "✓ SOLVED using generated code!")

                                logger.log_event("task_solved", {"task_id": task_id, "iteration": iteration, "approach": "generated_code"})
                                metrics.mark_complete(solved=True)

                                # Store successful solution
                                await self.store_solution(
                                    task_id,
                                    {"code": hypothesis["code"], "result": test_result},
                                    True,
                                    logger,
                                    tui,
                                    metrics,
                                    solving_context,
                                    hypothesis.get("_solution_guidance"),
                                )

                                # Attempt to discover a new primitive from this successful code
                                try:
                                    if hypothesis.get("code"):
                                        discovered = await self.primitive_discovery.discover_from_code(
                                            code=hypothesis["code"],
                                            task_id=task_id,
                                            task_data=task_data,
                                        )
                                        if discovered and tui:
                                            tui.add_agent_log(
                                                "discovery",
                                                f"✨ Discovered new primitive: {discovered.name}",
                                            )
                                except Exception as e:
                                    import logging
                                    logging.debug(f"Primitive discovery failed: {e}")

                                return test_result
                            break  # Break out of variant loop if solution found

                    if solution_found:
                        break  # Break out of iteration loop

            # Fallback: Try primitives if code didn't work
            if not solution_found and hypothesis.get("primitives"):
                primitives_to_try = hypothesis.get("primitives", [])
                if tui:
                    tui.add_agent_log("primitives", f"Trying {len(primitives_to_try)} primitive(s)...")

                for prim_idx, primitive in enumerate(primitives_to_try):
                    if tui:
                        tui.update_transformation_attempt(
                            iteration + 1,
                            f"{primitive} (attempt {prim_idx + 1}/{len(primitives_to_try)})",
                            None
                        )
                        tui.add_agent_log("transform", f"Trying: {primitive}")

                    await asyncio.sleep(0.5)

                    # Try applying to first training example
                    first_train = task_data["train"][0]
                    # Ensure first_train is a dictionary
                    if not isinstance(first_train, dict):
                        continue
                    result = self._try_primitive_transformation(first_train["input"], primitive)

                    if result and tui:
                        tui.update_transformation_attempt(iteration + 1, primitive, result)
                        await asyncio.sleep(0.5)

                    metrics.add_transformation_attempt(primitive)
                    logger.log_transformation_attempt(primitive, first_train["input"], result, result is not None)

                    # Verify on training examples
                    if result:
                        all_passed = True
                        for ex_idx, example in enumerate(task_data["train"]):
                            # Ensure example is a dictionary
                            if not isinstance(example, dict):
                                continue
                            attempt_result = self._try_primitive_transformation(example["input"], primitive)
                            metrics.num_verifications += 1

                            if attempt_result and await self.verify_solution(attempt_result, example["output"], logger):
                                if tui:
                                    tui.add_agent_log("verify", f"✓ Passed training example {ex_idx + 1}")
                                logger.log_verification(ex_idx, example["output"], attempt_result, True, None)
                            else:
                                metrics.num_failed_verifications += 1
                                all_passed = False
                                if tui:
                                    tui.add_agent_log("verify", f"✗ Failed training example {ex_idx + 1}")
                                logger.log_verification(ex_idx, example["output"], attempt_result, False, "Output mismatch")

                                # Store feedback
                                if solving_session and self.verifier_peer:
                                    await solving_session.add_messages([{
                                        "peer_id": "verifier",
                                        "content": f"Primitive {primitive} failed on training example {ex_idx + 1}",
                                        "metadata": {
                                            "type": "feedback",
                                            "primitive": primitive,
                                            "example_idx": ex_idx,
                                            "success": False,
                                        },
                                    }])
                                    metrics.num_messages_ingested += 1
                                    metrics.add_peer_fact("verifier")
                                break

                        if all_passed:
                            # Apply to test case
                            test_result = self._try_primitive_transformation(test_input, primitive)
                            if test_result:
                                solution_found = True
                                if tui:
                                    tui.update_output(test_result)
                                    tui.add_agent_log("success", f"✓ SOLVED using {primitive}!")

                                logger.log_event("task_solved", {"task_id": task_id, "iteration": iteration, "primitive": primitive})
                                metrics.mark_complete(solved=True)

                                # Store successful solution
                                await self.store_solution(
                                    task_id,
                                    {"primitive": primitive, "result": test_result},
                                    True,
                                    logger,
                                    tui,
                                    metrics,
                                    solving_context,
                                    hypothesis.get("_solution_guidance"),
                                )

                                return test_result

                    if tui:
                        tui.clear_attempt()
                        await asyncio.sleep(0.3)

            if solution_found:
                break  # Break out of iteration loop

            if tui:
                tui.add_agent_log("iteration", f"Iteration {iteration + 1} complete - no solution found")

        # Failed to solve - store failure
        if tui:
            tui.add_agent_log("failure", f"Could not solve after {self.config.max_iterations} iterations")

        logger.log_event("task_unsolved", {"task_id": task_id})
        metrics.mark_complete(solved=False)

        await self.store_solution(
            task_id,
            {"attempts": metrics.num_iterations},
            False,
            logger,
            tui,
            metrics,
            solving_context,
            None,  # No specific solution guidance to store for failed attempts
        )

        return None

    async def solve_with_ensemble(
        self, task_id: str, task_data: Dict, logger: JSONTraceLogger, metrics: SolverMetrics, tui=None
    ) -> Optional[Grid]:
        """
        Solve using ensemble of multiple strategies.

        This is a medium-term enhancement that runs multiple solving strategies
        and votes on the best solution.

        Args:
            task_id: Task identifier
            task_data: Task data with train/test examples
            logger: Trace logger
            metrics: Metrics tracker
            tui: Optional TUI for visualization

        Returns:
            Best solution from ensemble or None
        """
        if tui:
            tui.add_agent_log("enhancement", "Using ENSEMBLE mode with multiple strategies")

        # Use ensemble solver
        result = await self.ensemble_solver.solve_with_ensemble(
            task_id, task_data, logger, metrics, tui, parallel=False
        )

        return result

    async def solve_with_test_time_training(
        self, task_id: str, task_data: Dict, logger: JSONTraceLogger, metrics: SolverMetrics, tui=None
    ) -> Optional[Grid]:
        """
        Solve using test-time training guidance.

        This is a medium-term enhancement that uses similar solved tasks
        to guide the solving process.

        Args:
            task_id: Task identifier
            task_data: Task data with train/test examples
            logger: Trace logger
            metrics: Metrics tracker
            tui: Optional TUI for visualization

        Returns:
            Solution or None
        """
        if tui:
            tui.add_agent_log("enhancement", "Using TEST-TIME TRAINING mode")

        # Get test-time training guidance
        ttt_guidance = await self.test_time_trainer.apply_test_time_training(
            task_data, logger, metrics, tui
        )

        # Store guidance for use in prompt enhancement
        self._ttt_guidance = ttt_guidance

        # Solve normally (but prompts will be enhanced with TTT guidance)
        result = await self.solve_task(task_id, task_data, logger, metrics, tui)

        return result

    async def solve_with_all_enhancements(
        self, task_id: str, task_data: Dict, logger: JSONTraceLogger, metrics: SolverMetrics, tui=None
    ) -> Optional[Grid]:
        """
        Solve using ALL medium-term enhancements:
        - Test-time training
        - Ensemble methods
        - Pattern learning

        Args:
            task_id: Task identifier
            task_data: Task data with train/test examples
            logger: Trace logger
            metrics: Metrics tracker
            tui: Optional TUI for visualization

        Returns:
            Best solution or None
        """
        if tui:
            tui.add_agent_log("enhancement", "Using ALL ENHANCEMENTS mode (TTT + Ensemble + Patterns)")

        # 1. Apply test-time training
        ttt_guidance = await self.test_time_trainer.apply_test_time_training(
            task_data, logger, metrics, tui
        )
        self._ttt_guidance = ttt_guidance

        # 2. Learn patterns from successful solutions
        learned_prims = self.pattern_learner.analyze_successful_solutions(min_frequency=2)
        if tui and learned_prims:
            tui.add_agent_log("enhancement", f"Loaded {len(learned_prims)} learned primitives")

        # 3. Use ensemble solver
        result = await self.ensemble_solver.solve_with_ensemble(
            task_id, task_data, logger, metrics, tui, parallel=False
        )

        return result

    async def close(self):
        """Clean up resources."""
        # Honcho SDK handles cleanup automatically
        pass
