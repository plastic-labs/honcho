"""
Dynamic Primitive Discovery System with Honcho Memory Integration.

This module enables the agent to:
1. Discover new transformation primitives from successful solutions
2. Store primitives in Honcho memory (NOT local files)
3. Retrieve primitives semantically using Honcho's search
4. Learn and expand its transformation vocabulary dynamically

All primitives are stored in Honcho's memory system with rich metadata,
enabling semantic search, peer access, and unlimited context.
"""

import ast
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Grid = List[List[int]]


@dataclass
class DiscoveredPrimitive:
    """A primitive discovered and learned by the agent."""

    name: str
    description: str
    code: str
    signature: str  # Hash of the normalized code structure
    discovered_at: str  # ISO timestamp
    task_id: str  # Task where it was discovered
    success_count: int = 0
    usage_count: int = 0
    avg_success_rate: float = 0.0
    applicable_patterns: List[str] = field(default_factory=list)  # When to use it
    metadata: Dict[str, Any] = field(default_factory=dict)
    honcho_message_id: Optional[str] = None  # ID in Honcho

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "description": self.description,
            "code": self.code,
            "signature": self.signature,
            "discovered_at": self.discovered_at,
            "task_id": self.task_id,
            "success_count": self.success_count,
            "usage_count": self.usage_count,
            "avg_success_rate": self.avg_success_rate,
            "applicable_patterns": self.applicable_patterns,
            "metadata": self.metadata,
            "honcho_message_id": self.honcho_message_id,
        }

    @staticmethod
    def from_dict(data: Dict) -> "DiscoveredPrimitive":
        """Create from dictionary."""
        return DiscoveredPrimitive(
            name=data["name"],
            description=data["description"],
            code=data["code"],
            signature=data["signature"],
            discovered_at=data["discovered_at"],
            task_id=data["task_id"],
            success_count=data.get("success_count", 0),
            usage_count=data.get("usage_count", 0),
            avg_success_rate=data.get("avg_success_rate", 0.0),
            applicable_patterns=data.get("applicable_patterns", []),
            metadata=data.get("metadata", {}),
            honcho_message_id=data.get("honcho_message_id"),
        )


class PrimitiveDiscoverySystem:
    """
    System for dynamically discovering and learning new transformation primitives.

    Uses Honcho memory as the ONLY storage mechanism:
    - Primitives stored as messages with rich metadata
    - Semantic search via Honcho's get_context()
    - Pattern filtering via Honcho's metadata filters
    - Unlimited context via Honcho's summarization
    """

    def __init__(
        self,
        solver,
        min_success_rate: float = 0.6,
    ):
        """
        Initialize primitive discovery system.

        Args:
            solver: ARCSolver instance for Honcho access
            min_success_rate: Minimum success rate to keep a primitive
        """
        self.solver = solver
        self.min_success_rate = min_success_rate

        # In-memory cache of discovered primitives (loaded from Honcho)
        self.primitives: Dict[str, DiscoveredPrimitive] = {}
        self._primitives_loaded = False

        # Dedicated Honcho session for primitives
        self._primitives_session = None

        logging.info("Primitive discovery system initialized (Honcho-backed)")

    async def _ensure_primitives_session(self):
        """Ensure we have a dedicated session for primitive storage."""
        if self._primitives_session is not None:
            return

        if not self.solver.honcho_client:
            logging.warning("Honcho client not available, primitive discovery disabled")
            return

        # Create dedicated session for primitive library
        self._primitives_session = await self.solver.honcho_client.session(
            "primitive_library",
            metadata={
                "type": "primitive_library",
                "purpose": "Storage for dynamically discovered transformation primitives",
                "version": "1.0",
            },
        )

        # Add solution_generator peer to session
        if self.solver.solution_generator_peer:
            await self._primitives_session.add_peers([self.solver.solution_generator_peer])

        logging.info("Created Honcho session for primitive library")

    async def _load_primitives_from_honcho(self):
        """Load all discovered primitives from Honcho memory."""
        if self._primitives_loaded:
            return

        await self._ensure_primitives_session()

        if not self._primitives_session:
            return

        try:
            # Query all messages from primitive library session
            # Note: Honcho doesn't support nested metadata filtering
            messages_page = await self._primitives_session.get_messages()

            # Convert to list and filter client-side
            all_messages = [msg async for msg in messages_page]

            # Filter for discovered primitives
            messages = [
                msg for msg in all_messages
                if msg.metadata and msg.metadata.get("type") == "discovered_primitive"
            ]

            for msg in messages:
                try:
                    # Parse primitive from message metadata
                    if msg.metadata and "primitive_data" in msg.metadata:
                        primitive_data = json.loads(msg.metadata["primitive_data"])
                        primitive = DiscoveredPrimitive.from_dict(primitive_data)
                        primitive.honcho_message_id = msg.id
                        self.primitives[primitive.name] = primitive
                except Exception as e:
                    logging.debug(f"Error loading primitive from message {msg.id}: {e}")
                    continue

            self._primitives_loaded = True
            logging.info(f"Loaded {len(self.primitives)} primitives from Honcho")

        except Exception as e:
            logging.error(f"Error loading primitives from Honcho: {e}")

    async def discover_from_code(
        self,
        code: str,
        task_id: str,
        task_data: Dict,
        test_grids: List[Tuple[Grid, Grid]] = None,
    ) -> Optional[DiscoveredPrimitive]:
        """
        Discover a new primitive from successful transformation code.

        Args:
            code: The transformation code that worked
            task_id: ID of the task where it was discovered
            task_data: Full task data for testing
            test_grids: Optional additional test cases

        Returns:
            DiscoveredPrimitive if a valid new primitive was found
        """
        logging.info(f"Analyzing code from task {task_id} for primitive discovery")

        # Ensure primitives loaded and session created
        await self._load_primitives_from_honcho()
        await self._ensure_primitives_session()

        if not self._primitives_session:
            logging.warning("Honcho session not available, skipping discovery")
            return None

        # INGEST the code analysis attempt into memory
        await self._ingest_code_analysis(code, task_id, "analyzing")

        # 1. Extract and analyze the code structure
        structure = self._analyze_code_structure(code)
        if not structure:
            await self._ingest_code_analysis(code, task_id, "rejected_no_structure")
            return None

        # 2. Generate signature for deduplication
        signature = self._generate_signature(code)

        # Check if we already have this primitive
        for prim in self.primitives.values():
            if prim.signature == signature:
                logging.debug(f"Primitive {prim.name} already exists (signature match)")
                await self._ingest_code_analysis(code, task_id, "duplicate_found")
                # Update usage statistics in Honcho
                await self._update_primitive_stats(prim, success=True)
                return prim

        # 3. Generalize the code
        generalized_code = self._generalize_code(code, structure)
        await self._ingest_code_analysis(generalized_code, task_id, "generalized")

        # 4. Test the generalized code
        success_rate = await self._test_primitive(
            generalized_code, task_data, test_grids
        )

        await self._ingest_code_analysis(
            generalized_code, task_id, f"tested_success_rate_{success_rate:.2f}"
        )

        if success_rate < self.min_success_rate:
            logging.info(
                f"Primitive success rate {success_rate:.2%} below threshold {self.min_success_rate:.2%}"
            )
            await self._ingest_code_analysis(code, task_id, "rejected_low_success_rate")
            return None

        # 5. Use dialectic to reason about the primitive before generating description
        reasoning = await self._reason_about_primitive(generalized_code, structure, task_id)

        # 6. Generate description using LLM (or dialectic if available)
        description = await self._generate_description(generalized_code, structure)

        # 6. Create the primitive
        prim_name = f"discovered_{len(self.primitives) + 1}_{structure['type']}"
        primitive = DiscoveredPrimitive(
            name=prim_name,
            description=description,
            code=generalized_code,
            signature=signature,
            discovered_at=datetime.now().isoformat(),
            task_id=task_id,
            success_count=1,
            usage_count=1,
            avg_success_rate=success_rate,
            applicable_patterns=structure.get("patterns", []),
            metadata={
                "structure_type": structure["type"],
                "complexity": structure.get("complexity", "unknown"),
                "operations": structure.get("operations", []),
            },
        )

        # 7. Store in Honcho (primary storage)
        await self._store_in_honcho(primitive)

        # 8. Add to in-memory cache
        self.primitives[prim_name] = primitive

        logging.info(
            f"Discovered new primitive: {prim_name} (success_rate={success_rate:.2%})"
        )

        return primitive

    def _analyze_code_structure(self, code: str) -> Optional[Dict]:
        """
        Analyze code structure to understand transformation type.

        Returns dict with:
        - type: transformation type (geometric, color, object, composite)
        - operations: list of operations used
        - patterns: applicable patterns
        - complexity: simple/medium/complex
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None

        structure = {
            "type": "unknown",
            "operations": [],
            "patterns": [],
            "complexity": "simple",
        }

        # Analyze AST
        for node in ast.walk(tree):
            # Check for numpy operations
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    attr = node.func.attr
                    if attr in ["rot90", "flip", "fliplr", "flipud"]:
                        structure["operations"].append("geometric")
                        structure["patterns"].append("rotation_or_flip")
                    elif attr in ["tile", "repeat"]:
                        structure["operations"].append("tiling")
                        structure["patterns"].append("repetition")
                    elif attr in ["where", "select"]:
                        structure["operations"].append("conditional")
                        structure["patterns"].append("color_mapping")

            # Check for loops
            if isinstance(node, (ast.For, ast.While)):
                structure["complexity"] = "medium"
                structure["operations"].append("iteration")

            # Check for nested loops
            if isinstance(node, ast.For):
                for child in ast.walk(node):
                    if child != node and isinstance(child, ast.For):
                        structure["complexity"] = "complex"
                        structure["operations"].append("double_iteration")
                        break

        # Determine primary type
        if "geometric" in structure["operations"]:
            structure["type"] = "geometric"
        elif "tiling" in structure["operations"]:
            structure["type"] = "tiling"
        elif "conditional" in structure["operations"]:
            structure["type"] = "color_mapping"
        elif "iteration" in structure["operations"]:
            structure["type"] = "object_manipulation"
        else:
            structure["type"] = "composite"

        return structure

    def _generate_signature(self, code: str) -> str:
        """Generate unique signature for code based on structure."""
        # Normalize code (remove whitespace, comments, variable names)
        normalized = re.sub(r"#.*", "", code)  # Remove comments
        normalized = re.sub(r"\s+", " ", normalized)  # Normalize whitespace
        normalized = re.sub(
            r"\b[a-z_]\w*\b", "VAR", normalized
        )  # Replace variable names

        # Hash the normalized code
        return hashlib.md5(normalized.encode()).hexdigest()

    def _generalize_code(self, code: str, structure: Dict) -> str:
        """
        Generalize code by parameterizing constants and making it reusable.
        """
        # Extract function if it's inline code
        if "def transform" not in code:
            code = f"def transform(grid):\n" + "\n".join(
                f"    {line}" for line in code.split("\n")
            )

        # Ensure it returns a result
        if "return" not in code:
            code += "\n    return result"

        return code

    async def _test_primitive(
        self,
        code: str,
        task_data: Dict,
        test_grids: List[Tuple[Grid, Grid]] = None,
    ) -> float:
        """
        Test the primitive on task examples.

        Returns success rate (0.0 to 1.0)
        """
        test_cases = test_grids or []

        # Add task training examples
        for example in task_data.get("train", []):
            # Ensure example is a dictionary
            if not isinstance(example, dict):
                continue
            test_cases.append((example["input"], example["output"]))

        if not test_cases:
            return 0.0

        success_count = 0
        for input_grid, expected_output in test_cases:
            try:
                # Execute the code
                local_vars = {"grid": input_grid, "np": np}
                exec(code, local_vars)

                if "transform" in local_vars:
                    result = local_vars["transform"](input_grid)
                else:
                    result = local_vars.get("result")

                # Compare with expected
                if result == expected_output or np.array_equal(
                    np.array(result), np.array(expected_output)
                ):
                    success_count += 1

            except Exception as e:
                logging.debug(f"Primitive test failed: {e}")
                continue

        return success_count / len(test_cases)

    async def _generate_description(self, code: str, structure: Dict) -> str:
        """
        Generate natural language description of the primitive.

        Uses dialectic API to create contextual descriptions informed by past discoveries.
        """
        try:
            # Try using dialectic if available for more contextual descriptions
            if (hasattr(self.solver, "solution_generator_peer") and
                self.solver.solution_generator_peer and
                self._primitives_session):

                # Use dialectic chat to get description with context
                dialectic_prompt = f"""Analyze this transformation code and provide a concise description (1-2 sentences) that captures its essence and when to use it:

Code:
```python
{code}
```

Structure: {structure['type']} transformation
Operations: {', '.join(structure['operations'])}

Based on similar primitives we've discovered, describe:
1. What this transformation does
2. When it's useful (which patterns)

Keep it concise and actionable."""

                chat_response = await self.solver.solution_generator_peer.chat(
                    query=dialectic_prompt,
                )

                if chat_response and hasattr(chat_response, 'content'):
                    return chat_response.content.strip()
                elif isinstance(chat_response, str):
                    return chat_response.strip()

            # Fallback to direct LLM if dialectic not available
            prompt = f"""Analyze this transformation code and provide a concise description (1-2 sentences):

Code:
```python
{code}
```

Structure: {structure['type']} transformation
Operations: {', '.join(structure['operations'])}

Provide a description that explains what this transformation does in simple terms."""

            response = await self.solver.llm_client.messages.create(
                model=self.solver.config.llm_model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()

        except Exception as e:
            logging.error(f"Error generating description: {e}")
            return f"{structure['type']} transformation using {', '.join(structure['operations'])}"

    async def _store_in_honcho(self, primitive: DiscoveredPrimitive):
        """Store discovered primitive in Honcho memory."""
        try:
            await self._ensure_primitives_session()

            if not self._primitives_session or not self.solver.solution_generator_peer:
                logging.warning("Cannot store primitive: Honcho session not available")
                return

            # Create comprehensive message content
            content = f"""DISCOVERED PRIMITIVE: {primitive.name}

Description: {primitive.description}

Code:
```python
{primitive.code}
```

Type: {primitive.metadata.get('structure_type', 'unknown')}
Complexity: {primitive.metadata.get('complexity', 'unknown')}
Success Rate: {primitive.avg_success_rate:.2%}
Discovered in Task: {primitive.task_id}
Discovered at: {primitive.discovered_at}

Applicable Patterns: {', '.join(primitive.applicable_patterns)}
Operations: {', '.join(primitive.metadata.get('operations', []))}

This primitive can be used when encountering tasks with patterns: {', '.join(primitive.applicable_patterns)}
"""

            # Store as message in primitive library session
            messages = await self._primitives_session.add_messages([{
                "peer_id": "solution_generator",
                "content": content,
                "metadata": {
                    "type": "discovered_primitive",
                    "primitive_name": primitive.name,
                    "primitive_data": json.dumps(primitive.to_dict()),
                    "task_id": primitive.task_id,
                    "success_rate": primitive.avg_success_rate,
                    "signature": primitive.signature,
                    "applicable_patterns": primitive.applicable_patterns,
                    "structure_type": primitive.metadata.get("structure_type"),
                    "operations": primitive.metadata.get("operations", []),
                    "complexity": primitive.metadata.get("complexity"),
                },
            }])

            # Store message ID for future updates
            if messages and len(messages) > 0:
                primitive.honcho_message_id = messages[0].id

            logging.info(f"Stored primitive {primitive.name} in Honcho")

        except Exception as e:
            logging.error(f"Error storing primitive in Honcho: {e}")
            import traceback
            traceback.print_exc()

    async def _update_primitive_stats(self, primitive: DiscoveredPrimitive, success: bool):
        """Update primitive usage statistics in Honcho."""
        try:
            primitive.usage_count += 1
            if success:
                primitive.success_count += 1

            # Recalculate average success rate
            primitive.avg_success_rate = primitive.success_count / primitive.usage_count

            # Update in Honcho by creating a new message
            await self._ensure_primitives_session()

            if not self._primitives_session or not self.solver.solution_generator_peer:
                return

            update_content = f"""PRIMITIVE USAGE UPDATE: {primitive.name}

Usage Count: {primitive.usage_count}
Success Count: {primitive.success_count}
Current Success Rate: {primitive.avg_success_rate:.2%}
Updated at: {datetime.now().isoformat()}
"""

            await self._primitives_session.add_messages([{
                "peer_id": "solution_generator",
                "content": update_content,
                "metadata": {
                    "type": "primitive_usage_update",
                    "primitive_name": primitive.name,
                    "usage_count": primitive.usage_count,
                    "success_count": primitive.success_count,
                    "avg_success_rate": primitive.avg_success_rate,
                },
            }])

            logging.debug(f"Updated stats for primitive {primitive.name} in Honcho")

        except Exception as e:
            logging.debug(f"Error updating primitive stats: {e}")

    async def retrieve_relevant_primitives(
        self, task_analysis: Dict, limit: int = 5
    ) -> List[DiscoveredPrimitive]:
        """
        Retrieve primitives relevant to the current task using Honcho's semantic search.

        Args:
            task_analysis: Analysis of the current task
            limit: Maximum number of primitives to return

        Returns:
            List of relevant primitives sorted by relevance
        """
        # Ensure primitives loaded
        await self._load_primitives_from_honcho()

        if not self._primitives_session:
            return []

        relevant = []
        task_patterns = task_analysis.get("patterns", [])

        try:
            # Build semantic query from task patterns
            query_parts = []
            if task_patterns:
                query_parts.append(f"transformation primitives for patterns: {', '.join(task_patterns)}")

            # Add task characteristics
            if "colors" in task_analysis:
                query_parts.append(f"using colors {task_analysis['colors']}")

            if "shape_change" in task_analysis:
                if task_analysis["shape_change"]:
                    query_parts.append("with shape changing")
                else:
                    query_parts.append("with shape preserving")

            query = " ".join(query_parts) if query_parts else "transformation primitives"

            # Get all messages from primitives session
            # Note: Honcho doesn't support nested metadata filtering, so we filter client-side
            messages_page = await self._primitives_session.get_messages()

            # Convert to list and filter client-side by metadata.type
            all_messages = [msg async for msg in messages_page]

            # Filter for discovered primitives only
            messages = [
                msg for msg in all_messages
                if msg.metadata and msg.metadata.get("type") == "discovered_primitive"
            ]

            # Take the most recent ones (already sorted by creation time)
            for msg in messages[:limit * 2]:
                try:
                    if msg.metadata and "primitive_data" in msg.metadata:
                        primitive_data = json.loads(msg.metadata["primitive_data"])
                        primitive = DiscoveredPrimitive.from_dict(primitive_data)

                        # Check success rate threshold
                        if primitive.avg_success_rate >= self.min_success_rate:
                            relevant.append(primitive)

                except Exception as e:
                    logging.debug(f"Error parsing primitive from context: {e}")
                    continue

            # Also check local cache for pattern matches
            for prim in self.primitives.values():
                if prim.avg_success_rate >= self.min_success_rate:
                    overlap = set(prim.applicable_patterns) & set(task_patterns)
                    if overlap and prim not in relevant:
                        relevant.append(prim)

            # Sort by success rate and limit
            relevant.sort(key=lambda p: p.avg_success_rate, reverse=True)
            return relevant[:limit]

        except Exception as e:
            logging.error(f"Error retrieving primitives from Honcho: {e}")
            import traceback
            traceback.print_exc()

            # Fallback to local pattern matching
            for prim in self.primitives.values():
                if prim.avg_success_rate >= self.min_success_rate:
                    overlap = set(prim.applicable_patterns) & set(task_patterns)
                    if overlap:
                        relevant.append((len(overlap), prim))

            relevant.sort(key=lambda x: x[0], reverse=True)
            return [prim for _, prim in relevant[:limit]]

    async def get_all_primitives(self) -> List[DiscoveredPrimitive]:
        """Get all discovered primitives from Honcho."""
        await self._load_primitives_from_honcho()
        return list(self.primitives.values())

    async def get_primitive_statistics(self) -> Dict:
        """Get statistics about discovered primitives."""
        await self._load_primitives_from_honcho()

        if not self.primitives:
            return {"total": 0}

        success_rates = [p.avg_success_rate for p in self.primitives.values()]
        usage_counts = [p.usage_count for p in self.primitives.values()]

        return {
            "total": len(self.primitives),
            "avg_success_rate": np.mean(success_rates),
            "total_usage": sum(usage_counts),
            "by_type": self._group_by_type(),
        }

    def _group_by_type(self) -> Dict[str, int]:
        """Group primitives by type."""
        by_type = {}
        for prim in self.primitives.values():
            prim_type = prim.metadata.get("structure_type", "unknown")
            by_type[prim_type] = by_type.get(prim_type, 0) + 1
        return by_type

    async def _ingest_code_analysis(self, code: str, task_id: str, status: str):
        """
        Store every code analysis interaction in Honcho for full memory ingestion.
        """
        try:
            await self._ensure_primitives_session()

            if not self._primitives_session or not self.solver.solution_generator_peer:
                return

            content = f"""CODE ANALYSIS: {status.upper()}

Task: {task_id}
Status: {status}

Code Snippet:
```python
{code[:500]}...
```

Timestamp: {datetime.now().isoformat()}
"""

            await self._primitives_session.add_messages([{
                "peer_id": "solution_generator",
                "content": content,
                "metadata": {
                    "type": "code_analysis",
                    "task_id": task_id,
                    "status": status,
                    "timestamp": datetime.now().isoformat(),
                },
            }])

        except Exception as e:
            logging.debug(f"Error ingesting code analysis: {e}")

    async def _reason_about_primitive(
        self,
        code: str,
        structure: Dict,
        task_id: str,
    ) -> Optional[Dict]:
        """
        Use dialectic to reason about a discovered primitive before storing it.

        This creates an internal dialogue about what makes this primitive useful.
        """
        try:
            await self._ensure_primitives_session()

            if not self._primitives_session or not self.solver.solution_generator_peer:
                return None

            # Use dialectic to reason
            reasoning_prompt = f"""I've discovered a potential new primitive transformation.

Type: {structure['type']}
Operations: {', '.join(structure['operations'])}
Complexity: {structure['complexity']}

Code snippet:
```python
{code[:300]}...
```

Let me reason about this primitive:

1. What makes this transformation useful?
2. When should this primitive be used?
3. What patterns does it solve well?
4. How does it differ from existing primitives?
5. What are its limitations?

Provide insights as JSON:
- usefulness: Why is this useful (1-2 sentences)
- use_cases: List of 2-3 specific use cases
- related_patterns: List of patterns this solves
- limitations: List of limitations
- uniqueness: How it differs from existing primitives

Format as JSON."""

            response = await self.solver.solution_generator_peer.chat(
                query=reasoning_prompt,
            )

            if not response:
                return None

            content = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                reasoning = json.loads(json_match.group(0))

                # INGEST the reasoning
                await self._ingest_code_analysis(
                    code=f"Reasoning about primitive: {reasoning}",
                    task_id=task_id,
                    status="primitive_reasoning"
                )

                return reasoning

        except Exception as e:
            logging.debug(f"Error reasoning about primitive: {e}")
            return None

    async def invent_new_primitive(
        self,
        target_pattern: str,
        task_examples: List[Dict] = None,
    ) -> Optional[DiscoveredPrimitive]:
        """
        GENERATIVELY INVENT a completely new primitive from scratch.

        Uses dialectic to brainstorm and generate novel transformation code
        that doesn't exist in the current primitive library.
        """
        try:
            await self._ensure_primitives_session()

            if not self._primitives_session or not self.solver.solution_generator_peer:
                return None

            logging.info(f"Attempting to generatively invent primitive for pattern: {target_pattern}")

            # INGEST invention attempt
            await self._ingest_code_analysis(
                code=f"Inventing new primitive for pattern: {target_pattern}",
                task_id="generative_invention",
                status="invention_attempt"
            )

            # Use dialectic to brainstorm what primitive we need
            brainstorm_prompt = f"""I need to invent a NEW transformation primitive for pattern: {target_pattern}

Looking at my current primitive library, I want to create something novel that doesn't exist yet.

What new transformation would be useful for {target_pattern} tasks?

Think creatively about:
1. Transformations that combine multiple operations in novel ways
2. Pattern-specific transformations (e.g., for symmetry, repetition, object manipulation)
3. Transformations that fill gaps in my current library

Provide 2-3 creative primitive ideas as JSON:
- ideas: List of ideas, each with:
  - name: Descriptive name
  - description: What it does
  - approach: High-level approach
  - novelty: Why it's different/useful

Format as JSON."""

            brainstorm_response = await self.solver.solution_generator_peer.chat(
                query=brainstorm_prompt,
            )

            if not brainstorm_response:
                return None

            content = brainstorm_response.content if hasattr(brainstorm_response, 'content') else str(brainstorm_response)

            # Parse ideas
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                return None

            brainstorm = json.loads(json_match.group(0))
            ideas = brainstorm.get("ideas", [])

            if not ideas:
                return None

            # Pick first idea and generate code for it
            first_idea = ideas[0]

            # Use dialectic to generate actual code
            code_gen_prompt = f"""Generate Python code for this new primitive:

Name: {first_idea.get('name', 'new_primitive')}
Description: {first_idea.get('description', '')}
Approach: {first_idea.get('approach', '')}

Generate a complete 'transform(grid)' function that:
1. Takes a grid (2D list of integers) as input
2. Returns the transformed grid
3. Uses numpy for array operations
4. Implements the transformation described above

Make it creative and novel!

Provide only the Python code in a ```python code block."""

            code_response = await self.solver.solution_generator_peer.chat(
                query=code_gen_prompt,
            )

            if not code_response:
                return None

            code_content = code_response.content if hasattr(code_response, 'content') else str(code_response)

            # Extract code
            code_match = re.search(r"```python\n(.*?)```", code_content, re.DOTALL)
            if not code_match:
                return None

            generated_code = code_match.group(1).strip()

            # INGEST generated code
            await self._ingest_code_analysis(
                code=generated_code,
                task_id="generative_invention",
                status="code_generated"
            )

            # Now process this like any discovered primitive
            # Test it on provided examples if any
            if task_examples:
                fake_task_data = {"train": task_examples}
                success_rate = await self._test_primitive(generated_code, fake_task_data, None)

                if success_rate >= self.min_success_rate:
                    # Create and store the invented primitive
                    structure = self._analyze_code_structure(generated_code)
                    signature = self._generate_signature(generated_code)

                    prim_name = f"invented_{len(self.primitives) + 1}_{first_idea.get('name', 'new').replace(' ', '_')}"

                    primitive = DiscoveredPrimitive(
                        name=prim_name,
                        description=first_idea.get('description', 'Generatively invented primitive'),
                        code=generated_code,
                        signature=signature,
                        discovered_at=datetime.now().isoformat(),
                        task_id="generative_invention",
                        success_count=1,
                        usage_count=1,
                        avg_success_rate=success_rate,
                        applicable_patterns=[target_pattern],
                        metadata={
                            "structure_type": structure.get("type", "generative"),
                            "invention_method": "dialectic_generation",
                            "idea_source": first_idea,
                        },
                    )

                    await self._store_in_honcho(primitive)
                    self.primitives[prim_name] = primitive

                    logging.info(f"Successfully invented new primitive: {prim_name}")
                    return primitive

        except Exception as e:
            logging.error(f"Error inventing new primitive: {e}")
            import traceback
            traceback.print_exc()

        return None

    async def reflect_on_primitive_library(self, query: str = None) -> Optional[Dict]:
        """
        Use dialectic to reflect on the accumulated primitive library.

        Allows the discovery system to introspect on what it has learned.
        """
        try:
            await self._ensure_primitives_session()

            if not self._primitives_session or not self.solver.solution_generator_peer:
                return None

            # Default query if none provided
            if not query:
                query = """Reflect on the primitive library I've accumulated:

1. How many primitives have I discovered?
2. What types of transformations do I know well?
3. What patterns am I confident with?
4. What gaps exist in my primitive library?
5. What should I focus on discovering next?

Provide insights as JSON with:
- primitive_count: number
- strong_areas: list of transformation types I'm good at
- weak_areas: list of gaps to fill
- confidence_level: overall confidence (low/medium/high)
- recommendations: what to focus on next

Format as JSON."""

            # Use dialectic chat for reflection
            reflection_response = await self.solver.solution_generator_peer.chat(
                query=query,
            )

            if not reflection_response:
                return None

            content = reflection_response.content if hasattr(reflection_response, 'content') else str(reflection_response)

            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                reflection_data = json.loads(json_match.group(0))
                return reflection_data
            else:
                return {
                    "primitive_count": len(self.primitives),
                    "summary": content[:300],
                    "confidence_level": "unknown"
                }

        except Exception as e:
            logging.debug(f"Error reflecting on primitive library: {e}")
            return None
