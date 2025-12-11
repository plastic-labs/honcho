"""
Explicit Derivation Judge System

Implements a 3-stage validation pipeline for evaluating atomic propositions
extracted from peer messages.

Stage 1: Structural Validation (rule-based)
Stage 2: NLI Validation (model-based entailment)
Stage 3: LLM Judge (escalation + coverage assessment)
"""

import json
import logging
import os
import re
from typing import Any, cast

import numpy as np
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from .explicit_common import (
    ExtractionResult,
    JudgeResult,
    Proposition,
    PropositionJudgment,
    calculate_f2_score,
)

logger = logging.getLogger(__name__)


# Configuration parameters
COMPOUND_MARKERS = [
    " and ",
    " but ",
    " or ",
    "because",
    "since",
    "so ",
    "although",
    "however",
    "while",
    "whereas",
    " if ",
    "unless",
    "when ",
]

MIN_WORD_COUNT = 4
REDUNDANCY_THRESHOLD = 0.9
NLI_ENTAILMENT_THRESHOLD = 0.8
NLI_CONTRADICTION_THRESHOLD = 0.7
MIN_QUALITY_SCORE = 0.8


class ExplicitJudge:
    """
    3-stage judge system for validating explicit derivations.
    """

    def __init__(
        self,
        llm_client: AsyncAnthropic | AsyncOpenAI | None = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        mode: str = "eval",
        verbose: bool = False,
        use_openai_embeddings: bool = True,
        stages_to_run: list[int] | None = None,
    ):
        """
        Initialize the judge system.

        Args:
            llm_client: Client for LLM calls (Anthropic or OpenAI)
            embedding_model_name: Sentence transformer model name (default: all-MiniLM-L6-v2)
            mode: Operating mode (eval, filter, production)
            verbose: Enable verbose logging for debugging
            use_openai_embeddings: Use OpenAI embeddings instead of local model (default: True)
            stages_to_run: List of stages to run (1, 2, 3). Default: [1, 2, 3]
        """
        self.llm_client = llm_client
        self.mode = mode
        self.verbose = verbose
        self.use_openai_embeddings = use_openai_embeddings
        self.stages_to_run = stages_to_run if stages_to_run else [1, 2, 3]

        # Set logger level based on verbose flag
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

        # Load embedding model or OpenAI client
        if use_openai_embeddings:
            logger.info("Using OpenAI embeddings (text-embedding-3-small)")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.warning(
                    "OPENAI_API_KEY not found, falling back to local embeddings"
                )
                self.use_openai_embeddings = False
                logger.info(f"Loading local embedding model: {embedding_model_name}")
                self.embedding_model = SentenceTransformer(embedding_model_name)
            else:
                self.openai_embed_client = AsyncOpenAI(api_key=openai_api_key)
                self.embedding_model = None  # type: ignore
        else:
            logger.info(f"Loading local embedding model: {embedding_model_name}")
            self.embedding_model = SentenceTransformer(embedding_model_name)

    async def _llm_structural_check(
        self, proposition: Proposition, peer_name: str
    ) -> dict[str, Any]:
        """
        LLM-based structural validation.

        Uses LLM to evaluate:
        - Has subject (does it have a grammatical subject?)
        - Has predicate (does it have a verb/predicate?)
        - Atomicity (is this truly ONE claim?)
        - Naming convention (does it properly reference the peer?)
        - Context sufficiency (is this standalone and clear?)

        Args:
            proposition: The proposition to validate
            peer_name: Name of the peer

        Returns:
            Dict with LLM validation results
        """
        if not self.llm_client:
            # Fallback: accept all if no LLM available
            return {
                "has_subject": True,
                "has_predicate": True,
                "atomicity": True,
                "naming_convention": True,
                "context_sufficiency": True,
                "issues": [],
            }

        text = proposition["text"]

        system_prompt = f"""You are a structural validator for atomic propositions extracted from conversation data.

Your task is to evaluate whether a proposition meets structural quality standards.

## Evaluation Criteria:

1. **Has Subject**: The proposition has a clear grammatical subject (a person, thing, or entity that the statement is about).
   - Examples with subject: "Sarah likes coffee", "The meeting was cancelled", "It is raining"
   - Examples without subject: "Likes coffee" (fragment), "Yesterday" (fragment)

2. **Has Predicate**: The proposition has a predicate (verb or statement about the subject).
   - Examples with predicate: "Sarah likes coffee" (likes), "The book is interesting" (is)
   - Examples without predicate: "Sarah" (just a noun), "The tall building" (noun phrase only)

3. **Atomicity**: The proposition contains exactly ONE factual claim. It should not contain:
   - Multiple claims joined by "and", "but", "or"
   - Compound statements with "because", "since", "so"
   - Conditional statements with "if", "unless", "when"
   - Contrasts with "although", "however", "while"

4. **Naming Convention**: The proposition properly identifies the peer by name:
   - Should start with or prominently feature "{peer_name}" (case-insensitive)
   - Pronouns alone (he/she/they) without context are insufficient
   - The peer's identity should be clear

5. **Context Sufficiency**: The proposition is standalone and understandable:
   - No dangling references that need external context
   - Includes necessary temporal, spatial, or relational context
   - Reader can understand the claim without seeing the source message

Use the `validate_structure` tool to submit your evaluation."""

        user_prompt = f"""## PEER NAME: {peer_name}

## PROPOSITION TO VALIDATE:
"{text}"

Evaluate this proposition against all five structural criteria: has subject, has predicate, atomicity, naming convention, and context sufficiency."""

        # Define tool schema (Anthropic format)
        anthropic_tool_definition = {
            "name": "validate_structure",
            "description": "Submit structural validation results",
            "input_schema": {
                "type": "object",
                "properties": {
                    "has_subject": {
                        "type": "boolean",
                        "description": "True if proposition has a clear grammatical subject",
                    },
                    "has_predicate": {
                        "type": "boolean",
                        "description": "True if proposition has a predicate (verb or statement about subject)",
                    },
                    "atomicity": {
                        "type": "boolean",
                        "description": "True if proposition contains exactly one claim",
                    },
                    "naming_convention": {
                        "type": "boolean",
                        "description": "True if peer is properly identified by name",
                    },
                    "context_sufficiency": {
                        "type": "boolean",
                        "description": "True if proposition is standalone and clear",
                    },
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of specific issues found (empty if all pass)",
                    },
                },
                "required": ["has_subject", "has_predicate", "atomicity", "naming_convention", "context_sufficiency", "issues"],
            },
        }

        # OpenAI format
        openai_tool_definition = {
            "type": "function",
            "function": {
                "name": "validate_structure",
                "description": "Submit structural validation results",
                "parameters": anthropic_tool_definition["input_schema"],
            },
        }

        if self.verbose:
            logger.debug(f"\n--- LLM Structural Check ---")
            logger.debug(f"Calling LLM for nuanced structural validation...")

        try:
            import asyncio

            if isinstance(self.llm_client, AsyncAnthropic):
                response = await asyncio.wait_for(
                    self.llm_client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1000,
                        temperature=0.0,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                        tools=[anthropic_tool_definition],  # type: ignore
                        tool_choice={"type": "tool", "name": "validate_structure"},
                    ),
                    timeout=30.0,
                )

                # Extract tool use
                tool_use = None
                for block in response.content:
                    if block.type == "tool_use" and block.name == "validate_structure":
                        tool_use = block
                        break

                if not tool_use:
                    raise ValueError("No tool use found in LLM response")

                result = tool_use.input

            elif isinstance(self.llm_client, AsyncOpenAI):
                model = os.getenv("OPENAI_MODEL", "gpt-4o")
                messages_list = cast(
                    list[dict[str, Any]],
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )

                response = await asyncio.wait_for(
                    self.llm_client.chat.completions.create(
                        model=model,
                        max_tokens=1000,
                        temperature=0.0,
                        messages=cast(Any, messages_list),  # type: ignore
                        tools=cast(Any, [openai_tool_definition]),  # type: ignore
                        tool_choice={
                            "type": "function",
                            "function": {"name": "validate_structure"},
                        },
                    ),
                    timeout=30.0,
                )

                if not response.choices or not response.choices[0].message.tool_calls:
                    raise ValueError("No tool calls in OpenAI response")

                tool_call = response.choices[0].message.tool_calls[0]
                result = json.loads(tool_call.function.arguments)  # type: ignore

            else:
                raise ValueError(f"Unsupported LLM client type: {type(self.llm_client)}")

            if self.verbose:
                logger.debug(f"LLM validation: {result}")

            return cast(dict[str, Any], result)

        except Exception as e:
            logger.warning(f"LLM structural check failed: {e}. Falling back to permissive mode.")
            # Fallback: accept all on error
            return {
                "atomicity": True,
                "naming_convention": True,
                "context_sufficiency": True,
                "issues": [f"LLM check failed: {str(e)}"],
            }

    async def stage_1_structural_validation(
        self, proposition: Proposition, peer_name: str
    ) -> dict[str, Any]:
        """
        Stage 1: LLM-based structural validation.

        Basic check (fast, deterministic):
        - Word count (≥4 words)

        LLM checks (comprehensive evaluation):
        - Has subject (grammatical subject?)
        - Has predicate (verb or statement?)
        - Atomicity (truly one claim?)
        - Naming convention (properly identifies peer?)
        - Context sufficiency (standalone and clear?)

        Args:
            proposition: The proposition to validate
            peer_name: Name of the peer

        Returns:
            Dict with pass/fail status and violations
        """
        text = proposition["text"]
        violations: list[str] = []

        if self.verbose:
            logger.debug(f"\n{'='*60}")
            logger.debug(f"STAGE 1: LLM-Based Structural Validation")
            logger.debug(f"{'='*60}")
            logger.debug(f"Proposition: {text}")
            logger.debug(f"Peer Name: {peer_name}")

        # === WORD COUNT CHECK (Fast) ===
        if self.verbose:
            logger.debug(f"\n--- Word Count Check ---")

        # Check word count
        word_count = len(text.split())
        if word_count < MIN_WORD_COUNT:
            violations.append(f"too_short: Only {word_count} words (min: {MIN_WORD_COUNT})")

        if self.verbose:
            logger.debug(f"Word count: {word_count} (min: {MIN_WORD_COUNT})")

        # === LLM CHECKS (Comprehensive) ===
        # Only call LLM if word count passes
        llm_result = None
        if not violations and self.llm_client:
            if self.verbose:
                logger.debug(f"\n--- LLM Structural Checks ---")
            llm_result = await self._llm_structural_check(proposition, peer_name)

            # Add LLM-detected issues to violations
            if not llm_result["has_subject"]:
                violations.append("missing_subject: No clear grammatical subject")
            if not llm_result["has_predicate"]:
                violations.append("missing_predicate: No predicate/verb")
            if not llm_result["atomicity"]:
                violations.append("compound_statement: LLM detected multiple claims")
            if not llm_result["naming_convention"]:
                violations.append(f"wrong_naming: Peer '{peer_name}' not properly identified")
            if not llm_result["context_sufficiency"]:
                violations.append("missing_context: Lacks necessary context to be standalone")

            # Add specific issues from LLM
            for issue in llm_result.get("issues", []):
                if issue and not any(issue in v for v in violations):
                    violations.append(f"llm_issue: {issue}")

        passed = len(violations) == 0

        result = {
            "stage": "structural",
            "passed": passed,
            "violations": violations,
            "checks": {
                "word_count": word_count >= MIN_WORD_COUNT,
                "has_subject": llm_result["has_subject"] if llm_result else True,
                "has_predicate": llm_result["has_predicate"] if llm_result else True,
                "atomicity": llm_result["atomicity"] if llm_result else True,
                "naming_convention": llm_result["naming_convention"] if llm_result else True,
                "context_sufficiency": llm_result["context_sufficiency"] if llm_result else True,
            },
            "llm_result": llm_result,
        }

        if self.verbose:
            logger.debug(f"\n--- Final Results ---")
            logger.debug(f"Checks: {result['checks']}")
            logger.debug(f"Violations: {violations}")
            logger.debug(f"PASSED: {passed}")

        return result

    async def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Get embeddings for texts using OpenAI or local model.

        Args:
            texts: List of texts to embed

        Returns:
            numpy array of embeddings
        """
        if self.use_openai_embeddings and self.openai_embed_client:
            # Use OpenAI embeddings
            response = await self.openai_embed_client.embeddings.create(
                model="text-embedding-3-small", input=texts
            )
            embeddings = np.array([item.embedding for item in response.data])
            return embeddings
        else:
            # Use local model (synchronous)
            return self.embedding_model.encode(texts)

    async def check_redundancy(
        self, propositions: list[Proposition]
    ) -> list[tuple[int, int]]:
        """
        Check for redundant propositions using embedding similarity.

        Args:
            propositions: List of propositions to check

        Returns:
            List of (index1, index2) tuples for redundant pairs
        """
        if len(propositions) < 2:
            return []

        # Get embeddings
        texts = [p["text"] for p in propositions]
        embeddings = await self._get_embeddings(texts)

        # Find similar pairs
        redundant_pairs: list[tuple[int, int]] = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Calculate cosine similarity
                sim = float(
                    np.dot(embeddings[i], embeddings[j])
                    / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                )
                if sim > REDUNDANCY_THRESHOLD:
                    redundant_pairs.append((i, j))

        return redundant_pairs

    async def stage_2_llm_entailment_batch(
        self, propositions: list[Proposition], message: str
    ) -> list[dict[str, Any]]:
        """
        Stage 2: LLM-based batch entailment validation.

        Uses LLM to evaluate whether multiple propositions are entailed by the source message.
        Processes all propositions in a single LLM call for efficiency.

        Args:
            propositions: List of propositions to validate
            message: The source message

        Returns:
            List of dicts with entailment categories and decisions for each proposition
        """
        if not self.llm_client or not propositions:
            # Fallback: mark all as needing escalation
            return [
                {
                    "stage": "entailment",
                    "decision": "escalate",
                    "entailment_category": "neutral",
                    "error_hint": None,
                    "reasoning": "No LLM client available",
                }
                for _ in propositions
            ]

        if self.verbose:
            logger.debug(f"\n{'='*60}")
            logger.debug(f"STAGE 2: LLM-Based Batch Entailment Validation")
            logger.debug(f"{'='*60}")
            logger.debug(f"Evaluating {len(propositions)} propositions")
            logger.debug(f"Message: {message[:200]}..." if len(message) > 200 else f"Message: {message}")

        # Build prompt
        props_text = "\n".join([f"{i+1}. {prop['text']}" for i, prop in enumerate(propositions)])

        system_prompt = """You are an expert at natural language inference and factual entailment.

Your task is to determine whether each proposition is entailed by the source message.

## Entailment Scoring:

For each proposition, assign:
- **Entailment score** (0.0-1.0): How strongly the message supports this claim
  - 1.0 = Directly stated or necessarily true given the message
  - 0.7-0.9 = Strongly implied or very likely true
  - 0.4-0.6 = Possibly true but uncertain
  - 0.1-0.3 = Weakly supported or unlikely
  - 0.0 = No support in the message

- **Contradiction score** (0.0-1.0): How strongly the message contradicts this claim
  - 1.0 = Directly contradicts or impossible given the message
  - 0.5-0.9 = Partially contradicts or unlikely to be true
  - 0.0 = No contradiction

- **Error type** (if entailment < 0.3 or contradiction > 0.5):
  - negation_error: Negation removed or inverted
  - quantity_error: Quantity/amount altered
  - temporal_error: Tense or time changed
  - hedge_removal: Uncertainty qualifier removed
  - hallucination: No basis in message
  - other: Other error

Use the `evaluate_entailment` tool to submit your evaluations."""

        user_prompt = f"""## SOURCE MESSAGE:
"{message}"

## PROPOSITIONS TO EVALUATE:
{props_text}

For each proposition, determine:
1. Entailment category: Does the message support, contradict, or remain neutral to this claim?
2. Error type (if applicable): What kind of error occurred?
3. Reasoning: Brief explanation of your judgment."""

        # Define tool schema
        anthropic_tool_definition = {
            "name": "evaluate_entailment",
            "description": "Submit entailment evaluations for all propositions",
            "input_schema": {
                "type": "object",
                "properties": {
                    "evaluations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "proposition_index": {
                                    "type": "integer",
                                    "description": "1-based index of the proposition",
                                },
                                "entailment_category": {
                                    "type": "string",
                                    "enum": [
                                        "strongly_entailed",
                                        "entailed",
                                        "weakly_entailed",
                                        "neutral",
                                        "weakly_contradicted",
                                        "contradicted",
                                        "strongly_contradicted",
                                    ],
                                    "description": "Categorical judgment of how the message relates to this proposition",
                                },
                                "error_type": {
                                    "type": "string",
                                    "enum": [
                                        "none",
                                        "negation_error",
                                        "quantity_error",
                                        "temporal_error",
                                        "hedge_removal",
                                        "hallucination",
                                        "other",
                                    ],
                                    "description": "Type of error if entailment is weak or contradicted",
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Brief explanation of the entailment judgment",
                                },
                            },
                            "required": [
                                "proposition_index",
                                "entailment_category",
                                "error_type",
                                "reasoning",
                            ],
                        },
                    }
                },
                "required": ["evaluations"],
            },
        }

        openai_tool_definition = {
            "type": "function",
            "function": {
                "name": "evaluate_entailment",
                "description": "Submit entailment evaluations for all propositions",
                "parameters": anthropic_tool_definition["input_schema"],
            },
        }

        try:
            import asyncio

            if isinstance(self.llm_client, AsyncAnthropic):
                response = await asyncio.wait_for(
                    self.llm_client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=2000,
                        temperature=0.0,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                        tools=[anthropic_tool_definition],  # type: ignore
                        tool_choice={"type": "tool", "name": "evaluate_entailment"},
                    ),
                    timeout=60.0,
                )

                # Extract tool use
                tool_use = None
                for block in response.content:
                    if block.type == "tool_use" and block.name == "evaluate_entailment":
                        tool_use = block
                        break

                if not tool_use:
                    raise ValueError("No tool use found in LLM response")

                evaluations = tool_use.input["evaluations"]

            elif isinstance(self.llm_client, AsyncOpenAI):
                model = os.getenv("OPENAI_MODEL", "gpt-4o")
                messages_list = cast(
                    list[dict[str, Any]],
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )

                response = await asyncio.wait_for(
                    self.llm_client.chat.completions.create(
                        model=model,
                        messages=messages_list,
                        tools=[openai_tool_definition],
                        tool_choice={"type": "function", "function": {"name": "evaluate_entailment"}},
                        temperature=0.0,
                        max_tokens=2000,
                    ),
                    timeout=60.0,
                )

                # Extract tool call
                if not response.choices[0].message.tool_calls:
                    raise ValueError("No tool calls in OpenAI response")

                tool_call = response.choices[0].message.tool_calls[0]
                evaluations = json.loads(tool_call.function.arguments)["evaluations"]

            else:
                raise ValueError(f"Unsupported LLM client type: {type(self.llm_client)}")

            # Process evaluations and build results
            results = []
            for eval_data in evaluations:
                idx = eval_data["proposition_index"] - 1  # Convert to 0-based
                if idx < 0 or idx >= len(propositions):
                    logger.warning(f"Invalid proposition index: {idx + 1}")
                    continue

                category = eval_data["entailment_category"]

                # Map categorical judgment to decision
                # strongly_entailed, entailed -> valid
                # weakly_entailed, neutral -> escalate
                # weakly_contradicted, contradicted, strongly_contradicted -> invalid
                if category in ["strongly_contradicted", "contradicted", "weakly_contradicted"]:
                    decision = "invalid"
                    error_hint = eval_data["error_type"] if eval_data["error_type"] != "none" else "contradiction"
                elif category in ["strongly_entailed", "entailed"]:
                    decision = "valid"
                    error_hint = None
                else:  # weakly_entailed, neutral
                    decision = "escalate"
                    error_hint = None

                result = {
                    "stage": "entailment",
                    "decision": decision,
                    "entailment_category": category,
                    "error_hint": error_hint,
                    "reasoning": eval_data.get("reasoning", ""),
                }

                results.append(result)

                if self.verbose:
                    prop_text = propositions[idx]["text"]
                    logger.debug(f"\nProposition {idx+1}: {prop_text}")
                    logger.debug(f"Entailment Category: {category}")
                    logger.debug(f"Decision: {decision}")
                    if error_hint:
                        logger.debug(f"Error: {error_hint}")

            # Ensure we have results for all propositions
            if len(results) < len(propositions):
                logger.warning(f"LLM returned {len(results)} evaluations for {len(propositions)} propositions")
                # Fill missing with escalate
                for i in range(len(results), len(propositions)):
                    results.append({
                        "stage": "entailment",
                        "decision": "escalate",
                        "entailment_category": "neutral",
                        "error_hint": None,
                        "reasoning": "Missing evaluation",
                    })

            return results

        except Exception as e:
            logger.error(f"LLM entailment evaluation failed: {e}")
            # Fallback: escalate all to Stage 3
            return [
                {
                    "stage": "entailment",
                    "decision": "escalate",
                    "entailment_category": "neutral",
                    "error_hint": None,
                    "reasoning": f"Evaluation error: {str(e)}",
                }
                for _ in propositions
            ]

    def _detect_error_type(self, message: str, proposition: str) -> str:
        """
        Attempt to classify the error type for invalid propositions.

        Args:
            message: Source message
            proposition: Invalid proposition

        Returns:
            Error type hint
        """
        message_lower = message.lower()
        prop_lower = proposition.lower()

        # Check for negation errors
        negation_words = ["not", "no", "never", "don't", "doesn't", "didn't", "won't", "can't"]
        message_has_negation = any(word in message_lower for word in negation_words)
        prop_has_negation = any(word in prop_lower for word in negation_words)

        if message_has_negation and not prop_has_negation:
            return "negation_error"

        # Check for temporal errors
        past_words = ["used to", "previously", "formerly", "was", "were", "had"]
        message_has_past = any(word in message_lower for word in past_words)
        present_words = ["is", "am", "are", "has", "have", "currently"]
        prop_has_present = any(word in prop_lower for word in present_words)

        if message_has_past and prop_has_present:
            return "temporal_error"

        # Check for hedge removal
        hedge_words = ["might", "maybe", "perhaps", "possibly", "probably", "considering"]
        message_has_hedge = any(word in message_lower for word in hedge_words)
        if message_has_hedge and not any(word in prop_lower for word in hedge_words):
            return "hedge_removal"

        # Default
        return "semantic_error"

    async def stage_3_llm_judge(
        self,
        escalated_propositions: list[tuple[Proposition, dict[str, Any]]],
        message: str,
        peer_name: str,
        all_propositions: list[Proposition],
    ) -> dict[str, Any]:
        """
        Stage 3: LLM-based escalation resolution and coverage assessment.

        Args:
            escalated_propositions: List of (proposition, stage2_result) tuples
            message: Source message
            peer_name: Name of the peer
            all_propositions: All extracted propositions (for coverage check)

        Returns:
            Dict with judgments for escalated props, coverage assessment, and error counts
        """
        if not self.llm_client:
            # Fallback: mark all escalated as invalid
            return {
                "escalated_judgments": [
                    {
                        "text": prop["text"],
                        "is_valid": False,
                        "error_type": "escalated_no_llm",
                        "reasoning": "No LLM client available for escalation",
                    }
                    for prop, _ in escalated_propositions
                ],
                "missing_propositions": [],
                "hallucinations": [],
                "error_counts": {"escalated_no_llm": len(escalated_propositions)},
            }

        # Build prompt for LLM judge
        system_prompt = f"""You are an expert evaluator assessing the quality of atomic propositions extracted from a peer's message.

Your task is to:
1. Judge whether each ESCALATED proposition is valid (supported by the message)
2. Identify what information was MISSING from the extraction
3. Detect any HALLUCINATIONS (propositions with no basis in the message)
4. Classify error types for invalid propositions

## Proposition Requirements:

**Atomicity**: ONE claim only, no compound statements
**Context Sufficiency**: Standalone, includes who/what/when context
**Naming Convention**: Starts with peer name "{peer_name}" (case-insensitive - "sarah", "Sarah", or "SARAH" are all acceptable)
**Semantic Preservation**: Preserve negations, quantities, hedges, tense exactly
**Grounding**: Directly stated OR immediately entailed (single logical step)

## Error Types:

- negation_error: Negation present in message but not in proposition
- quantity_error: Quantity altered (e.g., "some" → "many")
- temporal_error: Tense changed (e.g., "used to" → "currently")
- hedge_removal: Uncertainty removed (e.g., "might" → "is")
- ownership_assumption: Infers ownership not stated
- missing_context: Lacks necessary context to be standalone
- compound_statement: Multiple claims in one proposition
- hallucination: No basis in message

Use the `judge_propositions` tool to submit your evaluation."""

        # Format escalated propositions
        if escalated_propositions:
            escalated_text = "\n".join(
                [f"{i+1}. {prop['text']}" for i, (prop, _) in enumerate(escalated_propositions)]
            )
        else:
            escalated_text = "(None - all propositions passed or failed earlier stages)"

        # Format all propositions for hallucination check
        if all_propositions:
            all_props_text = "\n".join(
                [f"{i+1}. {p['text']}" for i, p in enumerate(all_propositions)]
            )
        else:
            all_props_text = "(No propositions extracted)"

        user_prompt = f"""## SOURCE MESSAGE:
"{message}"

## PEER NAME: {peer_name}

## ESCALATED PROPOSITIONS (uncertain from NLI):
{escalated_text}

## ALL EXTRACTED PROPOSITIONS (check for hallucinations):
{all_props_text}

Evaluate:
1. For each escalated proposition: valid or invalid? If invalid, what error type?
2. What factual information in the message was NOT extracted?
3. Are there any hallucinated propositions (no basis in the message)?
4. Count errors by type across all propositions."""

        # Define tool schema (Anthropic format)
        anthropic_tool_definition = {
            "name": "judge_propositions",
            "description": "Submit evaluation results for propositions",
            "input_schema": {
                "type": "object",
                "properties": {
                    "escalated_judgments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "is_valid": {"type": "boolean"},
                                "error_type": {
                                    "type": "string",
                                    "enum": [
                                        "none",
                                        "negation_error",
                                        "quantity_error",
                                        "temporal_error",
                                        "hedge_removal",
                                        "ownership_assumption",
                                        "missing_context",
                                        "compound_statement",
                                        "hallucination",
                                        "other",
                                    ],
                                },
                                "reasoning": {"type": "string"},
                            },
                            "required": ["text", "is_valid", "error_type", "reasoning"],
                        },
                    },
                    "missing_propositions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Factual information from the message that was not extracted",
                    },
                    "hallucinations": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Indices (1-based) of hallucinated propositions from ALL EXTRACTED",
                    },
                    "error_counts": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                        "description": "Count of each error type across all propositions",
                    },
                },
                "required": [
                    "escalated_judgments",
                    "missing_propositions",
                    "hallucinations",
                    "error_counts",
                ],
            },
        }

        # OpenAI format (nested under "function")
        openai_tool_definition = {
            "type": "function",
            "function": {
                "name": "judge_propositions",
                "description": "Submit evaluation results for propositions",
                "parameters": anthropic_tool_definition["input_schema"],
            },
        }

        if self.verbose:
            logger.debug(f"\n{'='*60}")
            logger.debug(f"STAGE 3: LLM Judge")
            logger.debug(f"{'='*60}")
            logger.debug(f"Escalated Propositions: {len(escalated_propositions)}")
            logger.debug(f"\n--- SYSTEM PROMPT ---")
            logger.debug(system_prompt)
            logger.debug(f"\n--- USER PROMPT ---")
            logger.debug(user_prompt)
        else:
            # Show progress indicator for non-verbose mode
            logger.info(
                f"Stage 3: Calling LLM judge for {len(escalated_propositions)} escalated propositions..."
            )

        try:
            if isinstance(self.llm_client, AsyncAnthropic):
                logger.info("Sending request to Anthropic Claude API...")
                if self.verbose:
                    logger.debug(f"Model: claude-sonnet-4-20250514")
                    logger.debug(f"Max tokens: 4000")

                # Add timeout to prevent hanging
                import asyncio

                try:
                    response = await asyncio.wait_for(
                        self.llm_client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=4000,
                            temperature=0.0,
                            system=system_prompt,
                            messages=[{"role": "user", "content": user_prompt}],
                            tools=[anthropic_tool_definition],  # type: ignore
                            tool_choice={"type": "tool", "name": "judge_propositions"},
                        ),
                        timeout=120.0,  # 2 minute timeout
                    )
                    logger.info("Received response from Anthropic Claude")
                except asyncio.TimeoutError:
                    logger.error("Anthropic API call timed out after 120 seconds")
                    raise TimeoutError("LLM judge request timed out")

                # Extract tool use
                if not response.content:
                    raise ValueError("Empty response from Anthropic")

                tool_use = None
                for block in response.content:
                    if block.type == "tool_use" and block.name == "judge_propositions":
                        tool_use = block
                        break

                if not tool_use:
                    raise ValueError("No tool use found in response")

                result = tool_use.input

                if self.verbose:
                    logger.debug(f"\n--- LLM RESPONSE ---")
                    logger.debug(f"Tool use received: {tool_use.name}")
                    logger.debug(f"Escalated judgments: {len(result.get('escalated_judgments', []))}")
                    logger.debug(f"Missing propositions: {len(result.get('missing_propositions', []))}")
                    logger.debug(f"Hallucinations: {len(result.get('hallucinations', []))}")

            elif isinstance(self.llm_client, AsyncOpenAI):
                logger.info("Sending request to OpenAI API...")
                model = os.getenv("OPENAI_MODEL", "gpt-4o")
                if self.verbose:
                    logger.debug(f"Model: {model}")
                    logger.debug(f"Max tokens: 4000")

                messages_list = cast(
                    list[dict[str, Any]],
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )

                import asyncio

                try:
                    response = await asyncio.wait_for(
                        self.llm_client.chat.completions.create(
                            model=model,
                            max_tokens=4000,
                            temperature=0.0,
                            messages=cast(Any, messages_list),  # type: ignore
                            tools=cast(Any, [openai_tool_definition]),  # type: ignore
                            tool_choice={
                                "type": "function",
                                "function": {"name": "judge_propositions"},
                            },
                        ),
                        timeout=120.0,  # 2 minute timeout
                    )
                    logger.info("Received response from OpenAI")
                except asyncio.TimeoutError:
                    logger.error("OpenAI API call timed out after 120 seconds")
                    raise TimeoutError("LLM judge request timed out")

                if not response.choices or not response.choices[0].message.tool_calls:
                    raise ValueError("No tool calls in OpenAI response")

                tool_call = response.choices[0].message.tool_calls[0]
                result = json.loads(tool_call.function.arguments)  # type: ignore

                if self.verbose:
                    logger.debug(f"\n--- LLM RESPONSE ---")
                    logger.debug(f"Tool call received: {tool_call.function.name}")  # type: ignore
                    logger.debug(f"Escalated judgments: {len(result.get('escalated_judgments', []))}")
                    logger.debug(f"Missing propositions: {len(result.get('missing_propositions', []))}")
                    logger.debug(f"Hallucinations: {len(result.get('hallucinations', []))}")

            else:
                raise ValueError(f"Unsupported LLM client type: {type(self.llm_client)}")

            if self.verbose:
                logger.debug(f"\n--- DETAILED RESULTS ---")
                logger.debug(json.dumps(result, indent=2))

            return cast(dict[str, Any], result)

        except TimeoutError as e:
            logger.error(f"Timeout in LLM judge: {e}")
            # Fallback for timeout
            return {
                "escalated_judgments": [
                    {
                        "text": prop["text"],
                        "is_valid": False,
                        "error_type": "judge_timeout",
                        "reasoning": f"LLM judge timed out after 120 seconds",
                    }
                    for prop, _ in escalated_propositions
                ],
                "missing_propositions": [],
                "hallucinations": [],
                "error_counts": {"judge_timeout": len(escalated_propositions)},
            }
        except Exception as e:
            logger.error(f"Error in LLM judge: {e}")
            import traceback

            traceback.print_exc()
            # Fallback for other errors
            return {
                "escalated_judgments": [
                    {
                        "text": prop["text"],
                        "is_valid": False,
                        "error_type": "judge_error",
                        "reasoning": f"LLM judge failed: {e}",
                    }
                    for prop, _ in escalated_propositions
                ],
                "missing_propositions": [],
                "hallucinations": [],
                "error_counts": {"judge_error": len(escalated_propositions)},
            }

    def calculate_quality_rating(
        self, structural_result: dict[str, Any], nli_result: dict[str, Any]
    ) -> tuple[str, str]:
        """
        Calculate a quality rating for a proposition based on validation results.

        Returns ordinal categorical rating with justification.

        Args:
            structural_result: Result from stage 1
            nli_result: Result from stage 2

        Returns:
            Tuple of (rating, justification) where rating is one of:
            - "excellent": Perfect structural form, very high entailment
            - "good": Good structural form, high entailment
            - "fair": Acceptable structure and entailment, but some concerns
            - "poor": Weak structure or low entailment
            - "invalid": Failed validation
        """
        structural_passed = structural_result["passed"]
        entailment_category = nli_result.get("entailment_category", "neutral")

        # Build justification components
        justification_parts = []

        # Structural assessment
        if structural_passed:
            justification_parts.append("structurally sound")
        else:
            violations = structural_result.get("violations", [])
            justification_parts.append(f"structural issues: {', '.join(violations[:2])}")

        # Entailment assessment (categorical)
        category_descriptions = {
            "strongly_entailed": "strongly entailed by source",
            "entailed": "entailed by source",
            "weakly_entailed": "weakly entailed by source",
            "neutral": "neutral relationship to source",
            "weakly_contradicted": "weakly contradicted by source",
            "contradicted": "contradicted by source",
            "strongly_contradicted": "strongly contradicted by source",
        }
        justification_parts.append(category_descriptions.get(entailment_category, entailment_category))

        justification = "; ".join(justification_parts)

        # Determine rating
        if not structural_passed:
            return "invalid", justification

        # Check if Stage 2 was skipped (no entailment_category)
        if "entailment_category" not in nli_result:
            return "fair", "structurally sound; entailment validation skipped"

        # Map categorical entailment to quality rating
        if entailment_category == "strongly_entailed":
            return "excellent", justification
        elif entailment_category == "entailed":
            return "good", justification
        elif entailment_category in ["weakly_entailed", "neutral"]:
            return "fair", justification
        else:  # weakly_contradicted, contradicted, strongly_contradicted
            return "poor", justification

    async def judge_extraction(
        self,
        propositions: list[Proposition],
        message: str,
        peer_name: str,
        ground_truth: list[str] | None = None,
    ) -> JudgeResult:
        """
        Run the complete 3-stage judge pipeline on extracted propositions.

        Args:
            propositions: Extracted propositions to judge
            message: Source message
            peer_name: Name of the peer
            ground_truth: Optional ground truth propositions for eval mode

        Returns:
            Complete judgment result
        """
        judgments: list[PropositionJudgment] = []
        escalated: list[tuple[Proposition, dict[str, Any]]] = []

        # Stage 1: Structural validation for all propositions
        stage_1_passed: list[tuple[Proposition, dict[str, Any]]] = []

        for prop in propositions:
            # Run stage 1 if included
            if 1 in self.stages_to_run:
                structural_result = await self.stage_1_structural_validation(prop, peer_name)
            else:
                # Skip stage 1 - create a passing result
                structural_result = {
                    "stage": "structural",
                    "passed": True,
                    "violations": [],
                    "checks": {
                        "word_count": True,
                        "has_subject": True,
                        "has_predicate": True,
                        "atomicity": True,
                        "naming_convention": True,
                        "context_sufficiency": True,
                    },
                    "llm_result": None,
                }

            if not structural_result["passed"]:
                # Failed structural validation
                violations_str = "; ".join(structural_result["violations"])
                judgment: PropositionJudgment = {
                    "text": prop["text"],
                    "is_valid": False,
                    "quality_rating": "invalid",
                    "quality_justification": f"Failed structural validation: {violations_str}",
                    "error_type": "structural_failure",
                    "error_reason": violations_str,
                    "stage_results": {"stage_1": structural_result},
                }
                judgments.append(judgment)
            else:
                # Passed Stage 1, collect for batch Stage 2 processing
                stage_1_passed.append((prop, structural_result))

        # Stage 2: Batch entailment validation for all Stage 1 passed propositions
        if 2 in self.stages_to_run and stage_1_passed:
            # Extract just the propositions for batch processing
            props_to_validate = [prop for prop, _ in stage_1_passed]

            # Batch process all propositions in one LLM call
            stage_2_results = await self.stage_2_llm_entailment_batch(props_to_validate, message)

            # Process results for each proposition
            for (prop, structural_result), nli_result in zip(stage_1_passed, stage_2_results):
                quality_rating, quality_justification = self.calculate_quality_rating(
                    structural_result, nli_result
                )

                if nli_result["decision"] == "invalid":
                    # Failed entailment validation
                    category = nli_result.get("entailment_category", "unknown")
                    judgment = {
                        "text": prop["text"],
                        "is_valid": False,
                        "quality_rating": quality_rating,
                        "quality_justification": quality_justification,
                        "error_type": nli_result["error_hint"] or "entailment_failure",
                        "error_reason": nli_result.get("reasoning", f"Entailment category: {category}"),
                        "stage_results": {
                            "stage_1": structural_result,
                            "stage_2": nli_result,
                        },
                    }
                    judgments.append(judgment)
                elif nli_result["decision"] == "valid":
                    # Passed both stages
                    judgment = {
                        "text": prop["text"],
                        "is_valid": True,
                        "quality_rating": quality_rating,
                        "quality_justification": quality_justification,
                        "error_type": None,
                        "error_reason": None,
                        "stage_results": {
                            "stage_1": structural_result,
                            "stage_2": nli_result,
                        },
                    }
                    judgments.append(judgment)
                else:
                    # Escalate to Stage 3
                    escalated.append(
                        (
                            prop,
                            {
                                "stage_1": structural_result,
                                "stage_2": nli_result,
                                "quality_rating": quality_rating,
                                "quality_justification": quality_justification,
                            },
                        )
                    )
        elif stage_1_passed:
            # Stage 2 skipped - handle all Stage 1 passed propositions
            for prop, structural_result in stage_1_passed:
                nli_result = {
                    "stage": "entailment",
                    "decision": "skipped",
                    "error_hint": None,
                    # No entailment_category key - signals that Stage 2 was skipped
                }
                quality_rating, quality_justification = self.calculate_quality_rating(
                    structural_result, nli_result
                )

                # If Stage 3 is included, escalate all props to Stage 3
                # Otherwise, mark them as valid
                if 3 in self.stages_to_run:
                    # Escalate to Stage 3 for evaluation
                    escalated.append(
                        (
                            prop,
                            {
                                "stage_1": structural_result,
                                "stage_2": nli_result,
                                "quality_rating": quality_rating,
                                "quality_justification": quality_justification,
                            },
                        )
                    )
                else:
                    # No Stage 3 either, accept as valid
                    judgment = {
                        "text": prop["text"],
                        "is_valid": True,
                        "quality_rating": quality_rating,
                        "quality_justification": quality_justification,
                        "error_type": None,
                        "error_reason": None,
                        "stage_results": {
                            "stage_1": structural_result,
                            "stage_2": nli_result,
                        },
                    }
                    judgments.append(judgment)

        # Check for redundancy across all propositions
        redundant_pairs = await self.check_redundancy(propositions)

        # Stage 3: LLM judge for escalated cases and coverage
        stage3_result: dict[str, Any] | None = None
        if 3 in self.stages_to_run:
            # Run stage 3 if included
            if escalated or self.mode == "eval":
                stage3_result = await self.stage_3_llm_judge(
                    escalated, message, peer_name, propositions
                )

                # Process escalated judgments
                for escalated_judgment in stage3_result["escalated_judgments"]:
                    # Find the matching proposition
                    matching_prop = None
                    matching_results = None
                    for prop, results in escalated:
                        if prop["text"] == escalated_judgment["text"]:
                            matching_prop = prop
                            matching_results = results
                            break

                    if matching_prop and matching_results:
                        judgment = {
                            "text": matching_prop["text"],
                            "is_valid": escalated_judgment["is_valid"],
                            "quality_rating": matching_results["quality_rating"],
                            "quality_justification": matching_results["quality_justification"],
                            "error_type": escalated_judgment["error_type"]
                            if not escalated_judgment["is_valid"]
                            else None,
                            "error_reason": escalated_judgment["reasoning"]
                            if not escalated_judgment["is_valid"]
                            else None,
                            "stage_results": {
                                **matching_results,
                                "stage_3": escalated_judgment,
                            },
                        }
                        judgments.append(judgment)
        else:
            # Skip stage 3 - accept all escalated propositions
            for prop, results in escalated:
                judgment = {
                    "text": prop["text"],
                    "is_valid": True,
                    "quality_rating": results["quality_rating"],
                    "quality_justification": results["quality_justification"],
                    "error_type": None,
                    "error_reason": None,
                    "stage_results": {
                        **results,
                        "stage_3": {
                            "text": prop["text"],
                            "is_valid": True,
                            "error_type": "none",
                            "reasoning": "Stage 3 skipped",
                        },
                    },
                }
                judgments.append(judgment)

        # Calculate metrics
        logger.info("Calculating final metrics...")
        valid_count = sum(1 for j in judgments if j["is_valid"])
        total_count = len(judgments)
        precision = valid_count / total_count if total_count > 0 else 0.0

        # Calculate recall if ground truth provided
        coverage = 0.0
        missing_propositions: list[str] = []
        if ground_truth and stage3_result:
            logger.info("Processing coverage from ground truth...")
            missing_propositions = stage3_result.get("missing_propositions", [])
            captured_count = total_count - len(missing_propositions)
            total_capturable = len(ground_truth)
            coverage = captured_count / total_capturable if total_capturable > 0 else 0.0

        # Error breakdown
        logger.info("Building error breakdown...")
        error_breakdown: dict[str, int] = {}
        for judgment in judgments:
            if not judgment["is_valid"] and judgment["error_type"]:
                error_type = judgment["error_type"]
                error_breakdown[error_type] = error_breakdown.get(error_type, 0) + 1

        # Add redundancy errors
        if redundant_pairs:
            error_breakdown["redundant_propositions"] = len(redundant_pairs)

        # Calculate stage-by-stage metrics (only for stages that actually ran)
        logger.info("Calculating stage metrics...")

        from .explicit_common import StageMetrics

        # Stage 1 metrics (only if Stage 1 ran)
        if 1 in self.stages_to_run:
            stage_1_total = len(propositions)
            stage_1_passed = sum(
                1
                for j in judgments
                if "stage_1" in j["stage_results"] and j["stage_results"]["stage_1"]["passed"]
            )
            stage_1_pass_rate = stage_1_passed / stage_1_total if stage_1_total > 0 else 0.0
        else:
            # Stage 1 was skipped
            stage_1_total = 0
            stage_1_passed = 0
            stage_1_pass_rate = 0.0

        # Stage 2 metrics (only if Stage 2 ran)
        if 2 in self.stages_to_run:
            # Only props that passed Stage 1 (or all props if Stage 1 was skipped) enter Stage 2
            stage_2_total = stage_1_passed if 1 in self.stages_to_run else len(propositions)
            stage_2_passed = sum(
                1
                for j in judgments
                if "stage_2" in j["stage_results"]
                and j["stage_results"]["stage_2"]["decision"] == "valid"
            )
            stage_2_escalated = sum(
                1
                for j in judgments
                if "stage_2" in j["stage_results"]
                and j["stage_results"]["stage_2"]["decision"] == "escalate"
            )
            stage_2_pass_rate = stage_2_passed / stage_2_total if stage_2_total > 0 else 0.0
        else:
            # Stage 2 was skipped
            stage_2_total = 0
            stage_2_passed = 0
            stage_2_escalated = 0
            stage_2_pass_rate = 0.0

        # Stage 3 metrics (only if Stage 3 ran)
        if 3 in self.stages_to_run:
            stage_3_total = len(escalated)  # Props that were escalated from Stage 2
            stage_3_validated = sum(
                1 for j in judgments
                if "stage_3" in j["stage_results"]
                and j["is_valid"]
                and j["stage_results"]["stage_3"].get("reasoning") != "Stage 3 skipped"
            )
            stage_3_invalidated = sum(
                1 for j in judgments
                if "stage_3" in j["stage_results"]
                and not j["is_valid"]
                and j["stage_results"]["stage_3"].get("reasoning") != "Stage 3 skipped"
            )
        else:
            # Stage 3 was skipped
            stage_3_total = 0
            stage_3_validated = 0
            stage_3_invalidated = 0

        final_valid = valid_count
        final_invalid = total_count - valid_count

        stage_metrics: StageMetrics = {
            "stage_1_total": stage_1_total,
            "stage_1_passed": stage_1_passed,
            "stage_1_pass_rate": stage_1_pass_rate,
            "stage_2_total": stage_2_total,
            "stage_2_passed": stage_2_passed,
            "stage_2_pass_rate": stage_2_pass_rate,
            "stage_2_escalated": stage_2_escalated,
            "stage_3_total": stage_3_total,
            "stage_3_validated": stage_3_validated,
            "stage_3_invalidated": stage_3_invalidated,
            "final_valid": final_valid,
            "final_invalid": final_invalid,
        }

        logger.info("Judge extraction complete")
        return {
            "judgments": judgments,
            "precision": precision,
            "coverage": coverage,
            "error_breakdown": error_breakdown,
            "missing_propositions": missing_propositions,
            "stage_metrics": stage_metrics,
        }

    async def match_propositions_to_ground_truth(
        self, extracted: list[Proposition], ground_truth: list[str]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Match extracted propositions to ground truth using LLM-based semantic matching.

        Args:
            extracted: List of extracted propositions
            ground_truth: List of ground truth proposition strings

        Returns:
            Tuple of (matches, unmatched_extracted_indices, unmatched_gt_indices)
            where matches is list of (extracted_idx, gt_idx) tuples
        """
        if not extracted or not ground_truth:
            return [], list(range(len(extracted))), list(range(len(ground_truth)))

        if not self.llm_client:
            logger.warning("No LLM client available for matching. Using fallback: no matches.")
            return [], list(range(len(extracted))), list(range(len(ground_truth)))

        logger.info(f"Using LLM to match {len(extracted)} extracted propositions to {len(ground_truth)} ground truth...")

        extracted_texts = [p["text"] for p in extracted]

        # Create matching prompt
        system_prompt = """You are an expert at determining semantic equivalence between propositions.

Your task is to match extracted propositions to ground truth propositions based on semantic meaning.

Two propositions match if they express the same factual claim, even if worded differently:
- **Match**: "Sarah has a dog" ≈ "Sarah owns a dog" ≈ "Sarah is a dog owner"
- **Match**: "John lives in NYC" ≈ "John resides in New York City"
- **No match**: "Sarah likes dogs" ≠ "Sarah has a dog" (liking vs owning)
- **No match**: "Sarah might get a dog" ≠ "Sarah has a dog" (uncertain vs factual)

Important matching rules:
1. **Semantic equivalence**: Match if the core meaning is the same
2. **Paraphrasing**: Different words expressing the same fact should match
3. **Entity references**: "Sarah" and "she" may refer to the same person
4. **Temporal precision**: "June 2025" and "June 26, 2025" should match if both refer to the same event
5. **No hallucination**: Only match if the facts are truly equivalent

Use the `report_matches` tool to submit your matching results."""

        user_prompt = f"""## EXTRACTED PROPOSITIONS:
{chr(10).join(f"[{i}] {text}" for i, text in enumerate(extracted_texts))}

## GROUND TRUTH PROPOSITIONS:
{chr(10).join(f"[{j}] {text}" for j, text in enumerate(ground_truth))}

Identify all semantic matches between extracted and ground truth propositions. Each match should be a pair of indices [extracted_index, ground_truth_index]."""

        # Define tool schema
        anthropic_tool_definition = {
            "name": "report_matches",
            "description": "Report the matched pairs of extracted and ground truth propositions",
            "input_schema": {
                "type": "object",
                "properties": {
                    "matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "extracted_index": {
                                    "type": "integer",
                                    "description": "Index of the extracted proposition",
                                },
                                "ground_truth_index": {
                                    "type": "integer",
                                    "description": "Index of the ground truth proposition",
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Brief explanation of why these match",
                                },
                            },
                            "required": ["extracted_index", "ground_truth_index", "reasoning"],
                        },
                        "description": "List of matched pairs",
                    }
                },
                "required": ["matches"],
            },
        }

        openai_tool_definition = {
            "type": "function",
            "function": {
                "name": "report_matches",
                "description": "Report the matched pairs of extracted and ground truth propositions",
                "parameters": anthropic_tool_definition["input_schema"],
            },
        }

        if self.verbose:
            logger.debug("Calling LLM for proposition matching...")

        try:
            import asyncio

            if isinstance(self.llm_client, AsyncAnthropic):
                response = await asyncio.wait_for(
                    self.llm_client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=4000,
                        temperature=0.0,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                        tools=[anthropic_tool_definition],  # type: ignore
                        tool_choice={"type": "tool", "name": "report_matches"},
                    ),
                    timeout=60.0,
                )

                tool_use = None
                for block in response.content:
                    if block.type == "tool_use" and block.name == "report_matches":
                        tool_use = block
                        break

                if not tool_use:
                    raise ValueError("No tool use found in LLM response")

                result = tool_use.input

            elif isinstance(self.llm_client, AsyncOpenAI):
                model = os.getenv("OPENAI_MODEL", "gpt-4o")
                messages_list = cast(
                    list[dict[str, Any]],
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )

                response = await asyncio.wait_for(
                    self.llm_client.chat.completions.create(
                        model=model,
                        max_tokens=4000,
                        temperature=0.0,
                        messages=cast(Any, messages_list),  # type: ignore
                        tools=cast(Any, [openai_tool_definition]),  # type: ignore
                        tool_choice={
                            "type": "function",
                            "function": {"name": "report_matches"},
                        },
                    ),
                    timeout=60.0,
                )

                if not response.choices or not response.choices[0].message.tool_calls:
                    raise ValueError("No tool calls in OpenAI response")

                tool_call = response.choices[0].message.tool_calls[0]
                result = json.loads(tool_call.function.arguments)  # type: ignore

            else:
                raise ValueError(f"Unsupported LLM client type: {type(self.llm_client)}")

            # Process matches
            matches: list[tuple[int, int]] = []
            for match in result.get("matches", []):
                ext_idx = match["extracted_index"]
                gt_idx = match["ground_truth_index"]
                reasoning = match.get("reasoning", "")

                # Validate indices
                if 0 <= ext_idx < len(extracted) and 0 <= gt_idx < len(ground_truth):
                    matches.append((ext_idx, gt_idx))
                    if self.verbose:
                        logger.debug(
                            f"Match: extracted[{ext_idx}] ↔ GT[{gt_idx}] - {reasoning}"
                        )
                else:
                    logger.warning(f"Invalid match indices: {ext_idx}, {gt_idx}")

            # Find unmatched indices
            matched_extracted = {i for i, _ in matches}
            matched_gt = {j for _, j in matches}
            unmatched_extracted = [i for i in range(len(extracted)) if i not in matched_extracted]
            unmatched_gt = [j for j in range(len(ground_truth)) if j not in matched_gt]

            logger.info(
                f"LLM matching complete: {len(matches)} matches, "
                f"{len(unmatched_extracted)} unmatched extracted, "
                f"{len(unmatched_gt)} unmatched GT"
            )
            return matches, unmatched_extracted, unmatched_gt

        except Exception as e:
            logger.error(f"LLM matching failed: {e}. Falling back to no matches.")
            return [], list(range(len(extracted))), list(range(len(ground_truth)))

    async def calculate_matched_count(
        self,
        extracted: list[Proposition],
        ground_truth: list[str],
        judgment: JudgeResult,
    ) -> tuple[int, int]:
        """
        Calculate matching metrics for precision and recall.

        Only valid propositions (those that passed all judge stages) are considered for matching.

        Args:
            extracted: All extracted propositions
            ground_truth: Ground truth propositions
            judgment: Judgment result containing validity information

        Returns:
            Tuple of (matched_extracted_count, matched_ground_truth_count) where:
            - matched_extracted_count: Number of unique valid extracted propositions that matched any ground truth
            - matched_ground_truth_count: Number of unique ground truth propositions that were matched
        """
        # Filter to only valid propositions
        valid_props = []
        valid_indices_map = {}  # Map from filtered index to original index
        for i, (extracted_prop, judgment_result) in enumerate(zip(extracted, judgment["judgments"])):
            if judgment_result["is_valid"]:
                valid_indices_map[len(valid_props)] = i
                valid_props.append(extracted_prop)

        logger.info(
            f"Calculating matched count: {len(valid_props)} valid extracted, {len(ground_truth)} ground truth"
        )

        if not valid_props or not ground_truth:
            logger.info("No valid propositions or ground truth to match")
            return 0, 0

        # Match only valid propositions
        logger.info("Matching valid propositions to ground truth (this may take a moment)...")
        matches, unmatched_extracted, unmatched_gt = await self.match_propositions_to_ground_truth(
            valid_props, ground_truth
        )

        # Count unique extracted indices (in valid props) and unique ground truth indices
        unique_extracted = len(set(m[0] for m in matches))
        unique_ground_truth = len(set(m[1] for m in matches))

        logger.info(
            f"Found {len(matches)} total matches: {unique_extracted} unique valid extracted, "
            f"{unique_ground_truth} unique ground truth"
        )
        return unique_extracted, unique_ground_truth
