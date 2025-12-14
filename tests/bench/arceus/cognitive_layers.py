"""
Reusable cognitive layer implementations for both self-play and solving.

This module implements the three-layer cognitive architecture:
- Layer 1: Meta-Strategy Planning - Decides HOW to think
- Layer 2: Adaptive Memory Reflection - Queries memory based on problem type
- Layer 3: Curiosity-Driven Reflection - Explores alternative interpretations after failures

Plus a Strategy Selection Layer that decides when to use deep exploration.
Plus Solution Guidance that extracts actionable "how to solve" instructions from memory.
"""

import json
import logging
import re
from typing import Dict, Optional


class CognitiveLayers:
    """Implements the three-layer cognitive architecture."""

    def __init__(self, reflection_peer, honcho_client, tui=None):
        """
        Initialize cognitive layers.

        Args:
            reflection_peer: Honcho peer for dialectic queries
            honcho_client: Honcho client instance
            tui: Optional TUI for visualization
        """
        self.reflection_peer = reflection_peer
        self.honcho_client = honcho_client
        self.tui = tui

    async def should_use_deep_exploration(
        self,
        task_id: str,
        task_analysis: dict,
        tui_label: str = "solving"
    ) -> dict:
        """
        Strategy Selection Layer: Consult memory to decide whether to use
        deep exploration (three-layer) or fast baseline solving.

        Uses dialectic to ask:
        - Have I seen tasks like this before?
        - Did deep exploration help on similar tasks?
        - Is this task type worth the extra cognitive overhead?
        - What does my experience suggest?

        Returns: {
            'use_deep_exploration': bool,
            'confidence': float,
            'reasoning': str,
            'estimated_benefit': str
        }
        """
        try:
            if not self.reflection_peer:
                # Default to deep exploration if no memory available
                return {
                    'use_deep_exploration': True,
                    'confidence': 0.5,
                    'reasoning': 'No memory available - defaulting to deep exploration for learning',
                    'estimated_benefit': 'unknown'
                }

            num_examples = task_analysis.get('num_examples', 0)
            input_shapes = task_analysis.get('input_shapes', [])
            output_shapes = task_analysis.get('output_shapes', [])
            colors = task_analysis.get('colors_used', [])

            query = f"""Based on my experience with similar ARC-AGI tasks, should I use DEEP EXPLORATION or FAST SOLVING?

CURRENT TASK:
- Task ID: {task_id}
- Examples: {num_examples}
- Input shapes: {input_shapes}
- Output shapes: {output_shapes}
- Colors used: {colors}

DEEP EXPLORATION means:
- Layer 1: Meta-strategy planning (HOW to think)
- Layer 2: Adaptive memory queries (problem-type specific)
- Layer 3: Curiosity reflection after failures
- Takes ~30-60 seconds longer but learns more

FAST SOLVING means:
- Standard memory query
- Direct strategy execution
- Faster but less learning

Based on tasks like this in my memory:
1. Did deep exploration significantly help solve similar tasks?
2. Or was fast solving sufficient?
3. What's the benefit-cost ratio for this task type?
4. What does my experience suggest?

Provide a JSON response with:
- use_deep_exploration: true or false (boolean)
- confidence: 0.0 to 1.0 (float)
- reasoning: Why this decision (string, 1-2 sentences)
- estimated_benefit: What I expect to gain (string)

Format as JSON."""

            response = await self.reflection_peer.chat(query=query)

            if self.tui:
                self.tui.add_memory_operation(
                    operation="Strategy Selection",
                    details="Should use deep exploration?",
                    num_results=1 if response else 0
                )

            if not response:
                # Default to deep exploration (learning mode)
                return {
                    'use_deep_exploration': True,
                    'confidence': 0.5,
                    'reasoning': 'No response from memory - defaulting to deep exploration',
                    'estimated_benefit': 'Learning opportunity'
                }

            content = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group(0))
                return decision
            else:
                # Default to deep exploration
                return {
                    'use_deep_exploration': True,
                    'confidence': 0.5,
                    'reasoning': 'Could not parse response - defaulting to deep exploration',
                    'estimated_benefit': 'Safe default'
                }

        except Exception as e:
            logging.debug(f"Error in strategy selection: {e}")
            # Default to deep exploration on error
            return {
                'use_deep_exploration': True,
                'confidence': 0.3,
                'reasoning': f'Error occurred - defaulting to deep exploration: {str(e)}',
                'estimated_benefit': 'Fallback to learning mode'
            }

    async def meta_strategy_planning(
        self,
        task_id: str,
        task_patterns: dict,
        tui_label: str = "solving"
    ) -> Optional[Dict]:
        """
        Layer 1: Plan HOW to think about this problem.

        Returns: {
            'problem_type': str,
            'thinking_strategy': str,
            'mental_model': str,
            'memory_query_strategy': str,
            'curiosity_questions': [str],
            'exploration_sequence': str,
            'assumptions_to_challenge': [str],
            'meta_insight': str,
            'approach_type': str  # analytical|intuitive|experimental|compositional|analogical
        }
        """
        try:
            if not self.reflection_peer:
                return None

            patterns_str = ", ".join(task_patterns.get("patterns", []))
            num_examples = task_patterns.get("num_examples", 0)
            input_shapes = task_patterns.get("input_shapes", [])
            output_shapes = task_patterns.get("output_shapes", [])

            meta_query = f"""Deep meta-cognitive planning: HOW should I approach this puzzle?

PUZZLE CONTEXT:
- Task ID: {task_id}
- Patterns: {patterns_str}
- Examples: {num_examples}
- Input shapes: {input_shapes} → Output shapes: {output_shapes}

META-COGNITIVE QUESTIONS:

1. **PROBLEM TYPE**: What cognitive challenge is this?
   - Pattern recognition? Spatial reasoning? Logical rules? Compositional?

2. **THINKING STRATEGY**: How should I adapt my thinking?
   - Bottom-up (examples to rules) or top-down (hypotheses to verification)?
   - What mental model should I use?

3. **MEMORY STRATEGY**: How to query memory for this problem type?
   - What past experiences are most relevant?
   - Should I look for successes, failures, or both?

4. **CURIOSITY**: What assumptions am I making unconsciously?
   - What am I taking for granted?
   - What "obvious" interpretations might be wrong?

5. **EXPLORATION**: How to sequence attempts?
   - Start simple or complex?
   - When to pivot vs persist?

Provide a JSON response with:
- problem_type: Type of challenge (string)
- thinking_strategy: How to think (string, 2-3 sentences)
- mental_model: What mental model (string)
- memory_query_strategy: How to query memory (string, 2-3 sentences)
- curiosity_questions: Questions to ask (list of 3-5 strings)
- exploration_sequence: How to sequence attempts (string)
- assumptions_to_challenge: What I'm taking for granted (list of 2-3 strings)
- meta_insight: Insight about thinking process (string)
- approach_type: Overall approach (analytical|intuitive|experimental|compositional|analogical)

Format as JSON."""

            meta_response = await self.reflection_peer.chat(query=meta_query)

            if self.tui:
                self.tui.add_memory_operation(
                    operation="Meta-Strategy",
                    details="Planning how to think",
                    num_results=1 if meta_response else 0
                )

            if not meta_response:
                return None

            content = meta_response.content if hasattr(meta_response, 'content') else str(meta_response)

            # Parse JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                meta_strategy = json.loads(json_match.group(0))

                # Display insights in TUI
                if self.tui and meta_strategy.get("meta_insight"):
                    self.tui.add_agent_log(
                        tui_label,
                        f"[{tui_label.title()}] 💡 Meta-insight: {meta_strategy['meta_insight'][:80]}"
                    )

                return meta_strategy

        except Exception as e:
            logging.debug(f"Error in meta-strategy planning: {e}")
            return None

    async def adaptive_memory_reflection(
        self,
        task_id: str,
        meta_strategy: dict,
        tui_label: str = "solving"
    ) -> Optional[Dict]:
        """
        Layer 2: Query memory ADAPTIVELY based on problem type.

        Returns: {
            'summary': str,
            'encounter_count': int,
            'confidence': str,
            'successful_strategies': [str],
            'failed_strategies': [str],
            'failure_learnings': [str],
            'key_insights': [str],
            'untried_ideas': [str],
            'memory_adaptation_notes': str
        }
        """
        try:
            if not self.reflection_peer:
                return None

            problem_type = meta_strategy.get('problem_type', 'unknown')
            approach_type = meta_strategy.get('approach_type', 'analytical')
            memory_query_strategy = meta_strategy.get('memory_query_strategy', '')

            # Build adaptive query based on problem type
            reflection_query = f"""ADAPTIVE MEMORY QUERY

CURRENT PUZZLE:
- Task: {task_id}
- Problem Type: {problem_type}
- Approach: {approach_type}
- Memory Strategy: {memory_query_strategy}

"""

            # Adapt query based on problem type
            if "pattern" in problem_type.lower():
                reflection_query += """PATTERN RECOGNITION FOCUS:
- What visual patterns and transformations worked before?
- What pattern matching strategies succeeded vs failed?
- What subtle variations caused failures?

"""
            elif "spatial" in problem_type.lower():
                reflection_query += """SPATIAL REASONING FOCUS:
- What spatial transformations worked before?
- What object relationship strategies succeeded?
- What spatial primitives were effective?

"""
            elif "logical" in problem_type.lower() or "rule" in problem_type.lower():
                reflection_query += """LOGICAL/RULE DISCOVERY FOCUS:
- What rule discovery methods worked before?
- What logical reasoning strategies succeeded?
- What conditional patterns were effective?

"""
            elif "compositional" in problem_type.lower():
                reflection_query += """COMPOSITIONAL FOCUS:
- What problem decomposition strategies worked?
- How were sub-problems effectively combined?
- What composition patterns succeeded?

"""
            else:
                reflection_query += """GENERAL FOCUS:
- What strategies worked on similar problems?
- What approaches failed and why?

"""

            # Adapt based on approach type
            if approach_type == "analytical":
                reflection_query += """ANALYTICAL PRIORITY:
- Systematic strategies and logical decomposition
- Causal understanding and step-by-step methods

"""
            elif approach_type == "intuitive":
                reflection_query += """INTUITIVE PRIORITY:
- Pattern recognition and empirical methods
- Quick heuristics and visual insights

"""
            elif approach_type == "experimental":
                reflection_query += """EXPERIMENTAL PRIORITY:
- Trial-and-error learnings
- Unexpected discoveries and creative attempts

"""

            reflection_query += """
Based on accumulated memory:

SUCCESSES:
1. What strategies WORKED on this problem type?
2. Why did they work?

FAILURES:
3. What strategies FAILED on this problem type?
4. What did failures teach?

NEW IDEAS:
5. What haven't I tried that might work?

Provide a JSON response with:
- summary: Brief reflection (2-3 sentences)
- encounter_count: Estimated similar tasks (int)
- confidence: low|medium|high
- successful_strategies: Strategies that worked (list)
- failed_strategies: Strategies that failed (list)
- failure_learnings: What failures taught (list of 2-3 points)
- key_insights: Insights tailored to this approach (list of 2-3)
- untried_ideas: Novel approaches (list of 1-2)
- memory_adaptation_notes: How this query differs from generic (string)

Format as JSON."""

            reflection_response = await self.reflection_peer.chat(query=reflection_query)

            if self.tui:
                self.tui.add_memory_operation(
                    operation="Adaptive Memory",
                    details=f"Query adapted to {problem_type}",
                    num_results=1 if reflection_response else 0
                )

            if not reflection_response:
                return None

            content = reflection_response.content if hasattr(reflection_response, 'content') else str(reflection_response)

            # Parse JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                reflection_data = json.loads(json_match.group(0))

                # Display insights in TUI
                if self.tui:
                    insights = reflection_data.get('key_insights', [])
                    if insights and isinstance(insights, list):
                        for insight in insights[:2]:
                            insight_str = str(insight) if not isinstance(insight, str) else insight
                            self.tui.add_agent_log(
                                tui_label,
                                f"[{tui_label.title()}] 💭 Memory insight: {insight_str[:80]}"
                            )

                return reflection_data
            else:
                return {
                    'summary': content[:200],
                    'encounter_count': 0,
                    'confidence': 'unknown',
                    'key_insights': []
                }

        except Exception as e:
            logging.debug(f"Error in adaptive memory reflection: {e}")
            return None

    async def curiosity_driven_reflection(
        self,
        failed_attempt: dict,
        meta_strategy: dict,
        problem_type: str,
        tui_label: str = "solving"
    ) -> Optional[Dict]:
        """
        Layer 3: After failure, ask WHY and explore alternatives.

        Args:
            failed_attempt: {'strategy': str, 'hypothesis': dict, 'iteration': int}
            meta_strategy: Meta-strategy from Layer 1
            problem_type: Type of problem

        Returns: {
            'why_failed': str,
            'wrong_assumption': str,
            'alternative_interpretations': [str],
            'blind_spots': [str],
            'experiment_ideas': [str],
            'paradigm_shift': str,
            'curiosity_insight': str
        }
        """
        try:
            if not self.reflection_peer:
                return None

            failed_strategy = failed_attempt.get('strategy', 'unknown')
            iteration = failed_attempt.get('iteration', 0)
            assumptions = meta_strategy.get('assumptions_to_challenge', [])

            curiosity_query = f"""Strategy "{failed_strategy}" failed on iteration {iteration}.

CONTEXT:
- Problem type: {problem_type}
- Assumptions I had: {', '.join(assumptions) if assumptions else 'unknown'}
- Approach: {meta_strategy.get('approach_type', 'unknown')}

DEEP CURIOSITY QUESTIONS:

1. **WHY DID THIS FAIL?**
   - What assumption was wrong?
   - What did I overlook?
   - Was I solving the RIGHT problem?

2. **ALTERNATIVE INTERPRETATIONS**
   - What if my entire interpretation is wrong?
   - What other ways could I see this puzzle?
   - What if I flip my assumptions?

3. **WHAT AM I NOT SEEING?**
   - What's hiding in plain sight?
   - What patterns am I blind to?
   - If I were completely wrong, what would be right?

4. **GENERATIVE CURIOSITY**
   - What experiment could test my assumption?
   - What's the simplest explanation I'm ignoring?
   - What question would completely change my approach?

Provide a JSON response with:
- why_failed: Root cause analysis (string)
- wrong_assumption: What assumption was incorrect (string)
- alternative_interpretations: Different ways to see this (list of 2-3 strings)
- blind_spots: What I'm not seeing (list of 2-3 strings)
- experiment_ideas: Tests to validate/invalidate (list of 1-2 strings)
- paradigm_shift: Fundamentally different way to think (string)
- curiosity_insight: What curiosity reveals (string)

Format as JSON."""

            curiosity_response = await self.reflection_peer.chat(query=curiosity_query)

            if self.tui:
                self.tui.add_memory_operation(
                    operation="Curiosity Reflection",
                    details="Why did this fail?",
                    num_results=1 if curiosity_response else 0
                )

            if not curiosity_response:
                return None

            content = curiosity_response.content if hasattr(curiosity_response, 'content') else str(curiosity_response)

            # Parse JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                curiosity_reflection = json.loads(json_match.group(0))

                # Display curiosity insights in TUI
                if self.tui and curiosity_reflection.get("curiosity_insight"):
                    self.tui.add_agent_log(
                        tui_label,
                        f"[{tui_label.title()}] 🎓 Curiosity reveals: {curiosity_reflection['curiosity_insight'][:80]}"
                    )

                if self.tui and curiosity_reflection.get("paradigm_shift"):
                    self.tui.add_agent_log(
                        tui_label,
                        f"[{tui_label.title()}] 🔄 Paradigm shift: {curiosity_reflection['paradigm_shift'][:80]}"
                    )

                return curiosity_reflection

        except Exception as e:
            logging.debug(f"Error in curiosity-driven reflection: {e}")
            return None

    async def get_solution_guidance_from_memory(
        self,
        task_id: str,
        task_analysis: dict,
        meta_strategy: Optional[dict] = None,
        tui_label: str = "solving"
    ) -> Optional[Dict]:
        """
        Extract actionable "HOW TO SOLVE" guidance from memory for similar tasks.

        This queries memory to get:
        - Specific solving strategies that worked before
        - Code patterns and primitives that were successful
        - Step-by-step approach based on past experience
        - Common pitfalls to avoid

        The natural language guidance is then translated into concrete code strategies.

        Args:
            task_id: Current task identifier
            task_analysis: Analysis of current task (shapes, colors, patterns)
            meta_strategy: Optional meta-strategy from Layer 1
            tui_label: Label for TUI logging

        Returns: {
            'has_experience': bool,
            'confidence': float,  # How confident memory is about this guidance
            'solving_strategy': str,  # Natural language description of how to solve
            'code_approach': str,  # High-level code approach (e.g., "rotate then filter objects")
            'primitives_to_use': [str],  # Specific primitive functions that worked
            'step_by_step': [str],  # Step-by-step solving instructions
            'pitfalls_to_avoid': [str],  # Common mistakes on similar tasks
            'example_code_pattern': str,  # Optional: pseudo-code or code snippet from memory
            'reasoning': str  # Why this approach should work
        }
        """
        try:
            if not self.reflection_peer:
                return None

            num_examples = task_analysis.get('num_examples', 0)
            input_shapes = task_analysis.get('input_shapes', [])
            output_shapes = task_analysis.get('output_shapes', [])
            colors = task_analysis.get('colors_used', [])
            patterns = task_analysis.get('patterns', [])

            # Build problem type context if available
            problem_context = ""
            if meta_strategy:
                problem_type = meta_strategy.get('problem_type', 'unknown')
                approach_type = meta_strategy.get('approach_type', 'analytical')
                problem_context = f"""
PROBLEM TYPE ANALYSIS:
- Problem Type: {problem_type}
- Recommended Approach: {approach_type}
"""

            guidance_query = f"""I need SPECIFIC, ACTIONABLE guidance on HOW TO SOLVE this ARC-AGI puzzle based on my past experience.

CURRENT TASK:
- Task ID: {task_id}
- Examples: {num_examples}
- Input shapes: {input_shapes} → Output shapes: {output_shapes}
- Colors: {colors}
- Patterns detected: {', '.join(patterns) if patterns else 'analyzing...'}
{problem_context}

CRITICAL: Based on my memory of SIMILAR tasks I've solved before:

1. **SOLVING STRATEGY**: What specific approach worked on similar transformations?
   - If I've seen similar input→output patterns, HOW did I solve them?
   - What was the key insight that unlocked the solution?

2. **CODE APPROACH**: What high-level code strategy should I use?
   - Examples: "rotate 90° then extract largest object", "filter by color then tile", etc.
   - Be specific about the sequence of operations

3. **PRIMITIVES TO USE**: Which primitive functions were most effective?
   - From: rotate_90, flip_horizontal, extract_objects, replace_color, tile_grid, etc.
   - List 2-5 specific primitives that worked on similar tasks

4. **STEP-BY-STEP INSTRUCTIONS**: Break down the solving process
   - Step 1: [specific action]
   - Step 2: [specific action]
   - Step 3: [specific action]

5. **PITFALLS TO AVOID**: What mistakes did I make on similar tasks?
   - What approaches FAILED?
   - What assumptions were wrong?

6. **EXAMPLE CODE PATTERN** (if available): Pseudo-code or code snippet that worked
   - Even partial patterns help (e.g., "used np.rot90(arr, k=1) with color filtering")

7. **CONFIDENCE & REASONING**: How confident am I that this guidance applies?
   - Do I have strong experience with this task type?
   - Why should this approach work?

Provide a JSON response with:
- has_experience: true if I've solved similar tasks, false if uncertain (boolean)
- confidence: 0.0 to 1.0 (float) - how confident this guidance is
- solving_strategy: High-level strategy description (string, 2-3 sentences)
- code_approach: Specific code approach (string, e.g., "rotate 90° clockwise then filter objects by size")
- primitives_to_use: List of primitive function names (list of strings)
- step_by_step: Ordered solving steps (list of strings, 3-5 steps)
- pitfalls_to_avoid: Common mistakes (list of 2-3 strings)
- example_code_pattern: Code snippet or pattern if available, empty string if not (string)
- reasoning: Why this approach should work (string, 1-2 sentences)

Format as JSON."""

            response = await self.reflection_peer.chat(query=guidance_query)

            if self.tui:
                self.tui.add_memory_operation(
                    operation="Solution Guidance",
                    details="Extracting how-to-solve from memory",
                    num_results=1 if response else 0
                )

            if not response:
                return {
                    'has_experience': False,
                    'confidence': 0.0,
                    'solving_strategy': 'No memory guidance available',
                    'code_approach': '',
                    'primitives_to_use': [],
                    'step_by_step': [],
                    'pitfalls_to_avoid': [],
                    'example_code_pattern': '',
                    'reasoning': 'First time encountering this task type'
                }

            content = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                guidance = json.loads(json_match.group(0))

                # Display guidance in TUI
                if self.tui and guidance.get('has_experience'):
                    confidence = guidance.get('confidence', 0.0)
                    confidence_icon = "🎯" if confidence > 0.7 else "💡" if confidence > 0.4 else "🤔"

                    self.tui.add_agent_log(
                        tui_label,
                        f"[{tui_label.title()}] {confidence_icon} Memory guidance (confidence: {confidence:.2f}): {guidance.get('solving_strategy', '')[:80]}"
                    )

                    # Show code approach
                    if guidance.get('code_approach'):
                        self.tui.add_agent_log(
                            tui_label,
                            f"[{tui_label.title()}] 🔧 Code approach: {guidance['code_approach'][:80]}"
                        )

                    # Show primitives
                    if guidance.get('primitives_to_use'):
                        primitives_str = ", ".join(guidance['primitives_to_use'][:3])
                        self.tui.add_agent_log(
                            tui_label,
                            f"[{tui_label.title()}] 🛠️  Recommended primitives: {primitives_str}"
                        )

                return guidance
            else:
                # Fallback: parse as natural language
                return {
                    'has_experience': True,
                    'confidence': 0.3,
                    'solving_strategy': content[:200],
                    'code_approach': 'Parse from natural language response',
                    'primitives_to_use': [],
                    'step_by_step': [],
                    'pitfalls_to_avoid': [],
                    'example_code_pattern': '',
                    'reasoning': 'Extracted from unstructured memory response'
                }

        except Exception as e:
            logging.debug(f"Error getting solution guidance from memory: {e}")
            return None
