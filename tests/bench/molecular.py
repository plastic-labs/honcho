"""
MolecularBench - Molecular Facts Evaluation Benchmark
Author: 3un01a (3un01a@plasticlabs.ai)

Based on "Molecular Facts: Desiderata for Decontextualization in LLM Fact Verification"
(Gunjal & Durrett, EMNLP 2024)

## Core Principle

Atomic facts are insufficient for verification because they lose context.
Molecular facts balance two competing criteria:

1. **Decontextuality**: The claim can be fully interpreted without the original context
2. **Minimality**: The minimum information needed to achieve decontextuality

## Evaluation Pipeline

For each proposition, we evaluate:

### Stage 1: Ambiguity Detection
Identify what makes the proposition context-dependent:
- Unresolved references (pronouns, demonstratives)
- Ambiguous entities (names that could refer to multiple people)
- Missing anchors (temporal, spatial, comparative)
- Implicit knowledge requirements

### Stage 2: Decontextuality Assessment
Can a reader understand this claim in isolation?
- Entity identification: Can we uniquely identify WHO/WHAT?
- Event grounding: Can we identify WHEN/WHERE/HOW?
- Relation clarity: Are relationships explicit?

### Stage 3: Minimality Assessment
Does the proposition avoid unnecessary elaboration?
- Information density: Is every word necessary?
- Claim independence: Does it contain only ONE verifiable claim?
- Elaboration check: Are qualifiers essential or excessive?

## Scoring

- Decontextuality ∈ [0, 1]: Higher = more standalone
- Minimality ∈ [0, 1]: Higher = more concise
- Molecular Score = √(Decontextuality * Minimality): Geometric mean captures balance

## Usage

python -m tests.bench.molecular --traces path/to/traces.jsonl
"""

import argparse
import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, cast

import anthropic
import openai
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .runner_common import (
    configure_logging,
    create_anthropic_client,
    create_openai_client,
    extract_conversation_id,
    extract_messages,
    extract_peer_name,
    extract_propositions,
    format_duration,
    load_traces,
)

# Load .env from bench directory
bench_dir = Path(__file__).parent
load_dotenv(bench_dir / ".env")

logger = configure_logging(level=logging.INFO, name=__name__)


# =============================================================================
# AMBIGUITY TYPES (Stage 1)
# =============================================================================


class AmbiguityType(Enum):
    """Types of ambiguity that prevent standalone interpretation."""

    # Reference ambiguities
    UNRESOLVED_PRONOUN = "unresolved_pronoun"  # he, she, they, it
    UNRESOLVED_DEMONSTRATIVE = "unresolved_demonstrative"  # this, that, these
    ANAPHORIC_REFERENCE = "anaphoric_reference"  # "the project", "the meeting"
    ELLIPSIS = "ellipsis"  # Incomplete structure relying on context

    # Entity ambiguities
    AMBIGUOUS_PERSON = "ambiguous_person"  # Common name without disambiguation
    AMBIGUOUS_ORGANIZATION = "ambiguous_organization"  # Company name collision
    AMBIGUOUS_LOCATION = "ambiguous_location"  # City/place name collision
    AMBIGUOUS_EVENT = "ambiguous_event"  # Event without unique identification

    # Anchor ambiguities
    RELATIVE_TIME = "relative_time"  # "recently", "last week", "soon"
    RELATIVE_LOCATION = "relative_location"  # "here", "there", "nearby"
    IMPLICIT_COMPARISON = "implicit_comparison"  # "better", "more", "preferred"

    # Knowledge ambiguities
    DOMAIN_JARGON = "domain_jargon"  # Unexpanded acronyms, technical terms
    IMPLICIT_CONTEXT = "implicit_context"  # Requires world knowledge to interpret


class DecontextualityIssue(Enum):
    """Issues that prevent a proposition from standing alone."""

    UNIDENTIFIABLE_SUBJECT = "unidentifiable_subject"  # Can't tell WHO
    UNIDENTIFIABLE_OBJECT = "unidentifiable_object"  # Can't tell WHAT
    UNIDENTIFIABLE_TIME = "unidentifiable_time"  # Can't tell WHEN
    UNIDENTIFIABLE_PLACE = "unidentifiable_place"  # Can't tell WHERE
    UNCLEAR_RELATION = "unclear_relation"  # Can't tell the relationship
    VERIFICATION_IMPOSSIBLE = "verification_impossible"  # Can't fact-check this


class MinimalityIssue(Enum):
    """Issues where a proposition contains unnecessary information."""

    NESTED_CLAIM = "nested_claim"  # Contains multiple verifiable facts
    EXCESSIVE_QUALIFIER = "excessive_qualifier"  # Unnecessary identifying info
    REDUNDANT_CONTEXT = "redundant_context"  # Context that doesn't aid verification
    EXPLANATORY_ADDITION = "explanatory_addition"  # Added explanation not in source
    TEMPORAL_OVER_PRECISION = "temporal_over_precision"  # Unnecessary date detail
    BIOGRAPHICAL_OVERLOAD = "biographical_overload"  # Too much background info


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class AmbiguityAnalysis:
    """Result of ambiguity detection for a proposition."""

    proposition: str
    ambiguities: list[tuple[AmbiguityType, str, str]] = field(default_factory=list)
    # Each tuple: (type, problematic_span, explanation)
    ambiguity_count: int = 0
    primary_subject: str = ""  # Who/what is this about?
    requires_context: bool = False
    disambiguation_needed: list[str] = field(default_factory=list)


@dataclass
class DecontextualityAnalysis:
    """Result of decontextuality assessment."""

    proposition: str
    score: float  # 0-1, higher = more decontextualized
    is_standalone: bool
    issues: list[tuple[DecontextualityIssue, str]] = field(default_factory=list)

    # Specific checks
    subject_identifiable: bool = True
    object_identifiable: bool = True
    time_anchored: bool = True
    place_anchored: bool = True
    relation_clear: bool = True

    # Suggested fix
    suggested_decontextualization: str = ""


@dataclass
class MinimalityAnalysis:
    """Result of minimality assessment."""

    proposition: str
    score: float  # 0-1, higher = more minimal
    is_minimal: bool
    issues: list[tuple[MinimalityIssue, str]] = field(default_factory=list)

    # Metrics
    token_count: int = 0
    claim_count: int = 1  # Should be 1 for molecular
    excess_tokens: int = 0

    # Nested claims found
    nested_claims: list[str] = field(default_factory=list)

    # Suggested reduction
    suggested_reduction: str = ""


@dataclass
class MolecularAnalysis:
    """Complete molecular analysis for a single proposition."""

    proposition: str
    ambiguity: AmbiguityAnalysis
    decontextuality: DecontextualityAnalysis
    minimality: MinimalityAnalysis

    # Molecular score: geometric mean of decontextuality and minimality
    molecular_score: float = 0.0

    # Classification
    is_molecular: bool = False  # True if both decontextualized AND minimal
    classification: str = ""  # "molecular", "too_atomic", "too_verbose", "problematic"

    def compute_molecular_score(self) -> float:
        """Compute the molecular score as geometric mean."""
        d = self.decontextuality.score
        m = self.minimality.score
        self.molecular_score = (d * m) ** 0.5

        # Classification based on scores
        if d >= 0.8 and m >= 0.8:
            self.is_molecular = True
            self.classification = "molecular"
        elif d < 0.5:
            self.classification = "too_atomic"  # Needs more context
        elif m < 0.5:
            self.classification = "too_verbose"  # Over-elaborated
        else:
            self.classification = "borderline"

        return self.molecular_score


@dataclass
class MolecularReport:
    """Aggregate report for all propositions in a trace."""

    conversation_id: str
    peer_name: str
    proposition_count: int
    source_message_count: int

    # Aggregate scores
    avg_decontextuality: float = 0.0
    avg_minimality: float = 0.0
    avg_molecular: float = 0.0

    # Classification counts
    molecular_count: int = 0
    too_atomic_count: int = 0
    too_verbose_count: int = 0
    borderline_count: int = 0

    # Common issues
    ambiguity_distribution: dict[str, int] = field(default_factory=dict)
    decontextuality_issues: dict[str, int] = field(default_factory=dict)
    minimality_issues: dict[str, int] = field(default_factory=dict)

    # Detailed results
    analyses: list[MolecularAnalysis] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "peer_name": self.peer_name,
            "proposition_count": self.proposition_count,
            "source_message_count": self.source_message_count,
            "scores": {
                "molecular": round(self.avg_molecular, 4),
                "decontextuality": round(self.avg_decontextuality, 4),
                "minimality": round(self.avg_minimality, 4),
            },
            "classification": {
                "molecular": self.molecular_count,
                "too_atomic": self.too_atomic_count,
                "too_verbose": self.too_verbose_count,
                "borderline": self.borderline_count,
                "molecular_rate": round(
                    self.molecular_count / self.proposition_count, 3
                )
                if self.proposition_count > 0
                else 0,
            },
            "issues": {
                "ambiguity_types": self.ambiguity_distribution,
                "decontextuality_issues": self.decontextuality_issues,
                "minimality_issues": self.minimality_issues,
            },
            "propositions": [
                {
                    "text": a.proposition,
                    "molecular_score": round(a.molecular_score, 3),
                    "decontextuality": round(a.decontextuality.score, 3),
                    "minimality": round(a.minimality.score, 3),
                    "classification": a.classification,
                    "is_molecular": a.is_molecular,
                }
                for a in self.analyses
            ],
        }


# =============================================================================
# EVALUATION PROMPTS
# =============================================================================


AMBIGUITY_DETECTION_PROMPT = """You are analyzing propositions for ambiguity - elements that prevent standalone interpretation.

## Ambiguity Types

**Reference Ambiguities** (require prior context to resolve):
- UNRESOLVED_PRONOUN: "he", "she", "they", "it" without clear antecedent
- UNRESOLVED_DEMONSTRATIVE: "this", "that", "these" without clear referent
- ANAPHORIC_REFERENCE: "the project", "the meeting" - definite reference without introduction
- ELLIPSIS: Grammatically incomplete, relies on context

**Entity Ambiguities** (multiple possible referents):
- AMBIGUOUS_PERSON: Common name (e.g., "John Smith", "Maria Garcia") without disambiguation
- AMBIGUOUS_ORGANIZATION: Company name that could refer to multiple entities
- AMBIGUOUS_LOCATION: Place name collision (e.g., "Springfield", "Cambridge")
- AMBIGUOUS_EVENT: Event without unique identification

**Anchor Ambiguities** (relative without absolute reference):
- RELATIVE_TIME: "recently", "last week", "tomorrow" without date anchor
- RELATIVE_LOCATION: "here", "there", "nearby" without location anchor
- IMPLICIT_COMPARISON: "better", "more", "preferred" without baseline

**Knowledge Ambiguities** (require external knowledge):
- DOMAIN_JARGON: Unexpanded acronyms, technical terms
- IMPLICIT_CONTEXT: Requires unstated world knowledge

## Your Task

For each proposition, identify ALL ambiguities that would prevent a reader from fully understanding the claim in isolation (without the original conversation).

For each ambiguity, specify:
1. The type (from the enum above)
2. The problematic span (the actual text causing ambiguity)
3. Brief explanation of why it's ambiguous

Also identify:
- The primary subject (who/what is this proposition about?)
- What disambiguation would be needed to make it standalone"""


DECONTEXTUALITY_PROMPT = """You are evaluating whether propositions can stand alone without their original context.

## Decontextuality Criteria

A proposition is DECONTEXTUALIZED if a reader can fully understand it WITHOUT:
- Reading the original conversation
- Knowing who the speakers are
- Having any prior context

### Checks

**Subject Identifiable**: Can we uniquely identify WHO the proposition is about?
- Bad: "He is a software engineer" (who is "he"?)
- Good: "Victor Chen is a software engineer"
- Good: "The user is a software engineer" (if "user" is the established convention)

**Object Identifiable**: Can we uniquely identify WHAT is being described?
- Bad: "User likes it" (what is "it"?)
- Good: "User likes Python programming"

**Time Anchored**: If temporal, can we identify WHEN?
- Bad: "User started recently" (when is recently?)
- Good: "User started in January 2024"
- Acceptable: "User currently works at X" (present tense is self-anchoring)

**Place Anchored**: If spatial, can we identify WHERE?
- Bad: "User lives there" (where is there?)
- Good: "User lives in San Francisco"

**Relation Clear**: Is the relationship/predicate unambiguous?
- Bad: "User has a connection to Google" (what kind of connection?)
- Good: "User works at Google"

### Scoring Guide

- 1.0: Fully standalone, no context needed
- 0.8: Minor ambiguity but interpretable
- 0.6: Some context would help but core meaning clear
- 0.4: Significant ambiguity, meaning partially unclear
- 0.2: Major context needed
- 0.0: Unintelligible without context

### Issue Types
- UNIDENTIFIABLE_SUBJECT: Can't determine who/what this is about
- UNIDENTIFIABLE_OBJECT: Can't determine the object/target
- UNIDENTIFIABLE_TIME: Temporal reference unclear
- UNIDENTIFIABLE_PLACE: Spatial reference unclear
- UNCLEAR_RELATION: The relationship/action is ambiguous
- VERIFICATION_IMPOSSIBLE: Cannot be fact-checked as written"""


MINIMALITY_PROMPT = """You are evaluating whether propositions are MINIMAL - containing only the information necessary for standalone interpretation.

## Minimality Criteria

A proposition is MINIMAL if:
1. It contains exactly ONE verifiable claim
2. Every word contributes to that claim's interpretability
3. No unnecessary elaboration or explanation is added

### The Molecular Sweet Spot

Too Atomic (needs more context):
- "Ann won a medal" → Which Ann? What medal? When?

Too Verbose (excessive elaboration):
- "Ann Jansson, the Swedish footballer born on 6 May 1957 who played for Hammarby IF DFF and later became a coach, won a bronze medal at the 1984 European Athletics Championships, an event organized by the European Athletics Association held in Prague, Czechoslovakia"

Molecular (just right):
- "Ann Jansson, the Swedish footballer born 1957, won a medal at the 1984 European Athletics Championships"

### Issue Types

**NESTED_CLAIM**: Contains multiple independently verifiable facts
- "User's sister, who is a doctor, lives in Boston" (3 claims: has sister, sister is doctor, sister lives in Boston)

**EXCESSIVE_QUALIFIER**: Unnecessary identifying information
- "John Smith, the 45-year-old Caucasian male software engineer" (when just "John Smith, software engineer at X" would suffice for disambiguation)

**REDUNDANT_CONTEXT**: Context that doesn't aid verification
- "User, who we've been discussing cooking with, likes Italian food"

**EXPLANATORY_ADDITION**: Added explanation not from source
- "User likes Python, which is a popular programming language"

**TEMPORAL_OVER_PRECISION**: Unnecessary time detail
- "On Tuesday, March 5th, 2024 at 3:47 PM Pacific Standard Time, user mentioned..."

**BIOGRAPHICAL_OVERLOAD**: Excessive background for simple claims
- "Victor Chen, born in 1990 in Shanghai, educated at MIT, currently residing in San Francisco, enjoys hiking"

### Metrics

- Token count: Ideal molecular range is 8-25 tokens
- Claim count: Should be exactly 1
- Excess tokens: Tokens that could be removed without losing meaning

### Scoring Guide

- 1.0: Perfectly minimal, one claim, no excess
- 0.8: Slight excess but acceptable
- 0.6: Some unnecessary information
- 0.4: Multiple claims or significant bloat
- 0.2: Heavy over-elaboration
- 0.0: Massively verbose, many nested claims"""


# =============================================================================
# MOLECULAR JUDGE
# =============================================================================


class MolecularJudge:
    """
    Evaluates propositions against the Molecular Facts criteria.

    Based on Gunjal & Durrett (2024): The key insight is that purely atomic
    facts lose context needed for verification, but over-decontextualization
    adds verifiable claims that could be wrong. Molecular facts are the sweet spot.
    """

    def __init__(
        self,
        llm_client: AsyncAnthropic | AsyncOpenAI,
        model: str = "claude-sonnet-4-20250514",
        verbose: bool = False,
        provider: str = "anthropic",
    ):
        self.llm_client: AsyncAnthropic | AsyncOpenAI = llm_client
        self.model: str = model
        self.verbose: bool = verbose
        self.provider: str = provider
        if verbose:
            logger.setLevel(logging.DEBUG)

    async def _call_llm(
        self, system: str, user: str, response_schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Call LLM without tool use, requesting JSON response directly."""
        try:
            # Add JSON schema instruction to user prompt
            schema_instruction = f"\n\nRespond with a JSON object matching this schema:\n```json\n{json.dumps(response_schema, indent=2)}\n```"
            full_user_prompt = user + schema_instruction

            if isinstance(self.llm_client, AsyncAnthropic):
                resp = await asyncio.wait_for(
                    self.llm_client.messages.create(
                        model=self.model,
                        max_tokens=4096,
                        temperature=0.0,
                        system=system,
                        messages=[{"role": "user", "content": full_user_prompt}],
                    ),
                    timeout=180.0,
                )
                # Extract text content
                text_content = ""
                for block in resp.content:
                    if isinstance(block, TextBlock):
                        text_content += block.text

                # Try to extract JSON from the response
                return self._extract_json(text_content)
            else:
                resp = await asyncio.wait_for(
                    self.llm_client.chat.completions.create(
                        model=self.model,
                        max_tokens=4096,
                        temperature=0.0,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": full_user_prompt},
                        ],
                        response_format={"type": "json_object"},
                    ),
                    timeout=180.0,
                )
                if resp.choices and resp.choices[0].message.content:
                    return json.loads(resp.choices[0].message.content)
                return {}
        except TimeoutError:
            logger.exception(
                "LLM call timed out (model=%s, provider=%s)",
                self.model,
                self.provider,
            )
            return {}
        except (anthropic.APIError, openai.APIError):
            logger.exception(
                "LLM API error (model=%s, provider=%s)",
                self.model,
                self.provider,
            )
            return {}
        except json.JSONDecodeError:
            logger.exception(
                "Failed to parse LLM response as JSON (model=%s, provider=%s)",
                self.model,
                self.provider,
            )
            return {}
        except Exception:
            logger.exception(
                "Unexpected error in LLM call (model=%s, provider=%s)",
                self.model,
                self.provider,
            )
            raise

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract JSON from text response, handling markdown code blocks."""
        # Try direct JSON parsing first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        import re

        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object in text
        brace_start = text.find("{")
        if brace_start >= 0:
            try:
                return json.loads(text[brace_start:])
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to extract JSON from response: {text[:200]}")
        return {}

    async def analyze_ambiguity(
        self, propositions: list[str], source_context: str
    ) -> list[AmbiguityAnalysis]:
        """Stage 1: Detect ambiguities in each proposition."""

        if not propositions:
            return []

        response_schema = {
            "type": "object",
            "properties": {
                "analyses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "ambiguities": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "enum": [t.value for t in AmbiguityType],
                                        },
                                        "span": {"type": "string"},
                                        "explanation": {"type": "string"},
                                    },
                                    "required": ["type", "span", "explanation"],
                                },
                            },
                            "primary_subject": {"type": "string"},
                            "requires_context": {"type": "boolean"},
                            "disambiguation_needed": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "index",
                            "ambiguities",
                            "primary_subject",
                            "requires_context",
                        ],
                    },
                }
            },
            "required": ["analyses"],
        }

        props_text = "\n".join(f'{i+1}. "{p}"' for i, p in enumerate(propositions))

        result = await self._call_llm(
            AMBIGUITY_DETECTION_PROMPT,
            (
                f"SOURCE CONTEXT (for reference only):\n{source_context}\n\n"
                + f"PROPOSITIONS TO ANALYZE:\n{props_text}\n\n"
                + "Identify all ambiguities in each proposition."
            ),
            response_schema,
        )

        analyses: list[AmbiguityAnalysis] = []
        for prop in propositions:
            analyses.append(AmbiguityAnalysis(proposition=prop))

        for item in cast(list[dict[str, Any]], result.get("analyses", [])):
            idx: int = int(item.get("index", 0)) - 1
            if 0 <= idx < len(analyses):
                a: AmbiguityAnalysis = analyses[idx]
                for amb in cast(list[dict[str, Any]], item.get("ambiguities", [])):
                    try:
                        atype = AmbiguityType(amb["type"])
                        a.ambiguities.append(
                            (atype, amb.get("span", ""), amb.get("explanation", ""))
                        )
                    except (ValueError, KeyError):
                        continue
                a.ambiguity_count = len(a.ambiguities)
                a.primary_subject = item.get("primary_subject", "")
                a.requires_context = item.get("requires_context", False)
                a.disambiguation_needed = item.get("disambiguation_needed", [])

        return analyses

    async def analyze_decontextuality(
        self, propositions: list[str], source_context: str
    ) -> list[DecontextualityAnalysis]:
        """Stage 2: Assess how well each proposition stands alone."""

        if not propositions:
            return []

        response_schema = {
            "type": "object",
            "properties": {
                "analyses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "score": {"type": "number", "minimum": 0, "maximum": 1},
                            "is_standalone": {"type": "boolean"},
                            "issues": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "enum": [
                                                t.value for t in DecontextualityIssue
                                            ],
                                        },
                                        "description": {"type": "string"},
                                    },
                                },
                            },
                            "subject_identifiable": {"type": "boolean"},
                            "object_identifiable": {"type": "boolean"},
                            "time_anchored": {"type": "boolean"},
                            "place_anchored": {"type": "boolean"},
                            "relation_clear": {"type": "boolean"},
                            "suggested_decontextualization": {"type": "string"},
                        },
                        "required": ["index", "score", "is_standalone"],
                    },
                }
            },
            "required": ["analyses"],
        }

        props_text = "\n".join(f'{i+1}. "{p}"' for i, p in enumerate(propositions))

        result = await self._call_llm(
            DECONTEXTUALITY_PROMPT,
            (
                f"SOURCE CONTEXT (the original conversation - propositions should be understandable WITHOUT this):\n{source_context}\n\n"
                + f"PROPOSITIONS TO EVALUATE:\n{props_text}\n\n"
                + "For each proposition, assess whether it can be understood in COMPLETE ISOLATION."
            ),
            response_schema,
        )

        analyses: list[DecontextualityAnalysis] = []
        for prop in propositions:
            analyses.append(
                DecontextualityAnalysis(
                    proposition=prop, score=0.5, is_standalone=False
                )
            )

        for item in cast(list[dict[str, Any]], result.get("analyses", [])):
            idx: int = int(item.get("index", 0)) - 1
            if 0 <= idx < len(analyses):
                a: DecontextualityAnalysis = analyses[idx]
                a.score = item.get("score", 0.5)
                a.is_standalone = item.get("is_standalone", False)
                for issue in cast(list[dict[str, Any]], item.get("issues", [])):
                    try:
                        itype = DecontextualityIssue(issue["type"])
                        a.issues.append((itype, issue.get("description", "")))
                    except (ValueError, KeyError):
                        continue
                a.subject_identifiable = item.get("subject_identifiable", True)
                a.object_identifiable = item.get("object_identifiable", True)
                a.time_anchored = item.get("time_anchored", True)
                a.place_anchored = item.get("place_anchored", True)
                a.relation_clear = item.get("relation_clear", True)
                a.suggested_decontextualization = item.get(
                    "suggested_decontextualization", ""
                )

        return analyses

    async def analyze_minimality(
        self, propositions: list[str], source_context: str
    ) -> list[MinimalityAnalysis]:
        """Stage 3: Assess whether each proposition avoids excessive elaboration."""

        if not propositions:
            return []

        response_schema = {
            "type": "object",
            "properties": {
                "analyses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "score": {"type": "number", "minimum": 0, "maximum": 1},
                            "is_minimal": {"type": "boolean"},
                            "issues": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "enum": [t.value for t in MinimalityIssue],
                                        },
                                        "description": {"type": "string"},
                                    },
                                },
                            },
                            "token_count": {"type": "integer"},
                            "claim_count": {"type": "integer"},
                            "excess_tokens": {"type": "integer"},
                            "nested_claims": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "suggested_reduction": {"type": "string"},
                        },
                        "required": [
                            "index",
                            "score",
                            "is_minimal",
                            "token_count",
                            "claim_count",
                        ],
                    },
                }
            },
            "required": ["analyses"],
        }

        props_text = "\n".join(f'{i+1}. "{p}"' for i, p in enumerate(propositions))

        result = await self._call_llm(
            MINIMALITY_PROMPT,
            (
                f"SOURCE CONTEXT (for comparison):\n{source_context}\n\n"
                + f"PROPOSITIONS TO EVALUATE:\n{props_text}\n\n"
                + "For each proposition, assess whether it is MINIMAL (one claim, no excess)."
            ),
            response_schema,
        )

        analyses: list[MinimalityAnalysis] = []
        for prop in propositions:
            analyses.append(
                MinimalityAnalysis(proposition=prop, score=0.5, is_minimal=False)
            )

        for item in cast(list[dict[str, Any]], result.get("analyses", [])):
            idx: int = int(item.get("index", 0)) - 1
            if 0 <= idx < len(analyses):
                a: MinimalityAnalysis = analyses[idx]
                a.score = item.get("score", 0.5)
                a.is_minimal = item.get("is_minimal", False)
                for issue in cast(list[dict[str, Any]], item.get("issues", [])):
                    try:
                        itype = MinimalityIssue(issue["type"])
                        a.issues.append((itype, issue.get("description", "")))
                    except (ValueError, KeyError):
                        continue
                a.token_count = item.get("token_count", 0)
                a.claim_count = item.get("claim_count", 1)
                a.excess_tokens = item.get("excess_tokens", 0)
                a.nested_claims = item.get("nested_claims", [])
                a.suggested_reduction = item.get("suggested_reduction", "")

        return analyses

    async def evaluate(
        self,
        propositions: list[str],
        messages: list[dict[str, Any]],
        peer_name: str,
        conversation_id: str = "",
    ) -> MolecularReport:
        """Run full molecular evaluation pipeline.

        Args:
            propositions: Extracted fact strings to evaluate for molecular
                quality (decontextuality, minimality, and ambiguity).
            messages: Raw conversation messages used as source context.
                Each dict should contain ``speaker`` and ``text`` keys.
            peer_name: Name of the peer whose propositions are being evaluated.
            conversation_id: Optional identifier for the conversation trace.
                Defaults to an empty string.

        Returns:
            A ``MolecularReport`` containing per-proposition molecular
            analyses (ambiguity, decontextuality, minimality scores and
            issues), aggregate scores, classification distribution, and
            common issues found across propositions.
        """

        if not propositions:
            return MolecularReport(
                conversation_id=conversation_id,
                peer_name=peer_name,
                proposition_count=0,
                source_message_count=len(messages),
            )

        # Build source context string
        user_msgs = [m for m in messages if m.get("speaker", "user") == "user"]
        source_context = "\n".join(f"- {m.get('text', '')}" for m in user_msgs)

        logger.info(
            f"Evaluating {len(propositions)} propositions (3-stage molecular pipeline)..."
        )

        # Run all three stages in parallel
        ambiguity_results, decontext_results, minimal_results = await asyncio.gather(
            self.analyze_ambiguity(propositions, source_context),
            self.analyze_decontextuality(propositions, source_context),
            self.analyze_minimality(propositions, source_context),
        )

        # Combine into molecular analyses
        analyses: list[MolecularAnalysis] = []
        for i, prop in enumerate(propositions):
            analysis = MolecularAnalysis(
                proposition=prop,
                ambiguity=ambiguity_results[i]
                if i < len(ambiguity_results)
                else AmbiguityAnalysis(proposition=prop),
                decontextuality=decontext_results[i]
                if i < len(decontext_results)
                else DecontextualityAnalysis(
                    proposition=prop, score=0.5, is_standalone=False
                ),
                minimality=minimal_results[i]
                if i < len(minimal_results)
                else MinimalityAnalysis(proposition=prop, score=0.5, is_minimal=False),
            )
            analysis.compute_molecular_score()
            analyses.append(analysis)

        # Build aggregate report
        report = MolecularReport(
            conversation_id=conversation_id,
            peer_name=peer_name,
            proposition_count=len(propositions),
            source_message_count=len(user_msgs),
            analyses=analyses,
        )

        # Compute averages
        report.avg_decontextuality = sum(
            a.decontextuality.score for a in analyses
        ) / len(analyses)
        report.avg_minimality = sum(a.minimality.score for a in analyses) / len(
            analyses
        )
        report.avg_molecular = sum(a.molecular_score for a in analyses) / len(analyses)

        # Count classifications
        for a in analyses:
            if a.classification == "molecular":
                report.molecular_count += 1
            elif a.classification == "too_atomic":
                report.too_atomic_count += 1
            elif a.classification == "too_verbose":
                report.too_verbose_count += 1
            else:
                report.borderline_count += 1

        # Aggregate issue distributions
        for a in analyses:
            for amb_type, _, _ in a.ambiguity.ambiguities:
                report.ambiguity_distribution[amb_type.value] = (
                    report.ambiguity_distribution.get(amb_type.value, 0) + 1
                )
            for issue_type, _ in a.decontextuality.issues:
                report.decontextuality_issues[issue_type.value] = (
                    report.decontextuality_issues.get(issue_type.value, 0) + 1
                )
            for issue_type, _ in a.minimality.issues:
                report.minimality_issues[issue_type.value] = (
                    report.minimality_issues.get(issue_type.value, 0) + 1
                )

        logger.info(
            f"Evaluation complete. Molecular: {report.avg_molecular:.2%} "
            + f"(D: {report.avg_decontextuality:.2%}, M: {report.avg_minimality:.2%})"
        )

        return report


# =============================================================================
# OUTPUT
# =============================================================================


def print_report(report: MolecularReport) -> None:
    """Print formatted molecular report."""

    print("\n" + "=" * 70)
    print(f"MOLECULAR ANALYSIS: {report.conversation_id}")
    print("=" * 70)
    print(
        f"Peer: {report.peer_name} | Propositions: {report.proposition_count} | Messages: {report.source_message_count}"
    )

    print(f"\n{'MOLECULAR SCORE:':<20} {report.avg_molecular:.1%}")
    print("-" * 40)
    print(f"{'Decontextuality:':<20} {report.avg_decontextuality:.1%}")
    print(f"{'Minimality:':<20} {report.avg_minimality:.1%}")

    print("\n" + "-" * 40)
    print("CLASSIFICATION DISTRIBUTION:")
    total = report.proposition_count
    print(
        f"  ✓ Molecular:    {report.molecular_count:3d} ({report.molecular_count/total*100:5.1f}%)"
    )
    print(
        f"  ⚠ Too Atomic:   {report.too_atomic_count:3d} ({report.too_atomic_count/total*100:5.1f}%)"
    )
    print(
        f"  ⚠ Too Verbose:  {report.too_verbose_count:3d} ({report.too_verbose_count/total*100:5.1f}%)"
    )
    print(
        f"  ~ Borderline:   {report.borderline_count:3d} ({report.borderline_count/total*100:5.1f}%)"
    )

    # Show top issues
    if report.ambiguity_distribution:
        print("\nTop Ambiguity Types:")
        for k, v in sorted(report.ambiguity_distribution.items(), key=lambda x: -x[1])[
            :3
        ]:
            print(f"  - {k}: {v}")

    if report.decontextuality_issues:
        print("\nTop Decontextuality Issues:")
        for k, v in sorted(report.decontextuality_issues.items(), key=lambda x: -x[1])[
            :3
        ]:
            print(f"  - {k}: {v}")

    if report.minimality_issues:
        print("\nTop Minimality Issues:")
        for k, v in sorted(report.minimality_issues.items(), key=lambda x: -x[1])[:3]:
            print(f"  - {k}: {v}")

    # Show example problems
    too_atomic = [a for a in report.analyses if a.classification == "too_atomic"]
    if too_atomic:
        print("\nExample Too Atomic (needs more context):")
        for a in too_atomic[:2]:
            print(f'  - "{a.proposition[:60]}..."')
            if a.decontextuality.suggested_decontextualization:
                print(
                    f'    → Suggest: "{a.decontextuality.suggested_decontextualization[:60]}..."'
                )

    too_verbose = [a for a in report.analyses if a.classification == "too_verbose"]
    if too_verbose:
        print("\nExample Too Verbose (over-elaborated):")
        for a in too_verbose[:2]:
            print(
                f'  - "{a.proposition[:60]}..." [{a.minimality.token_count} tokens, {a.minimality.claim_count} claims]'
            )
            if a.minimality.suggested_reduction:
                print(f'    → Suggest: "{a.minimality.suggested_reduction[:60]}..."')

    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================


async def main():
    parser = argparse.ArgumentParser(
        description="MolecularBench - Molecular Facts Evaluation"
    )
    parser.add_argument("--traces", type=Path, help="JSON/JSONL trace file")
    parser.add_argument("--trace-dir", type=Path, help="Directory of trace files")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tests/bench/molecular_results")
    )
    parser.add_argument(
        "--provider", choices=["anthropic", "openai", "openrouter"], default="anthropic"
    )
    parser.add_argument("--api-key", type=str, help="API key")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--limit", type=int, help="Limit number of traces")
    parser.add_argument("--sample", type=float, help="Sample fraction (0-1)")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Concurrent batch size"
    )
    args = parser.parse_args()

    # Initialize client using runner_common helpers
    if args.provider == "anthropic":
        client: AsyncAnthropic | AsyncOpenAI = create_anthropic_client(
            api_key=args.api_key
        )
    elif args.provider == "openrouter":
        client = create_openai_client(
            api_key=args.api_key,
            base_url="https://openrouter.ai/api/v1",
            env_key_name="OPENROUTER_API_KEY",
        )
    else:
        client = create_openai_client(api_key=args.api_key)

    judge = MolecularJudge(client, args.model, args.verbose, args.provider)

    # Load traces
    all_traces: list[tuple[dict[str, Any], str]] = []
    if args.traces:
        traces = load_traces(args.traces)
        all_traces.extend((t, args.traces.name) for t in traces)
    elif args.trace_dir:
        for f in list(args.trace_dir.glob("*.json")) + list(
            args.trace_dir.glob("*.jsonl")
        ):
            traces = load_traces(f)
            all_traces.extend((t, f.name) for t in traces)
    else:
        parser.error("Specify --traces or --trace-dir")

    print(f"Loaded {len(all_traces)} traces")

    if args.sample and 0 < args.sample < 1:
        all_traces = random.sample(
            all_traces, max(1, int(len(all_traces) * args.sample))
        )
    if args.limit:
        all_traces = all_traces[: args.limit]

    print(f"\nEvaluating {len(all_traces)} traces with Molecular Facts criteria...\n")

    overall_start = time.time()
    results: list[MolecularReport] = []

    async def process_trace(
        idx: int, trace: dict[str, Any], _source: str
    ) -> MolecularReport | None:
        props = extract_propositions(trace)
        if not props:
            logger.warning(f"Trace {idx} has no propositions")
            return None

        msgs = extract_messages(trace)
        peer = extract_peer_name(trace)
        conv_id = extract_conversation_id(trace, idx)

        print(f"[{idx+1}/{len(all_traces)}] {conv_id} ({len(props)} props)...")

        try:
            report = await judge.evaluate(props, msgs, peer, conv_id)
            print_report(report)
            return report
        except Exception as e:
            logger.error(f"Failed {conv_id}: {e}")
            return None

    # Process
    if args.batch_size > 1:
        for i in range(0, len(all_traces), args.batch_size):
            batch = all_traces[i : i + args.batch_size]
            batch_results = await asyncio.gather(
                *[process_trace(i + j, t, s) for j, (t, s) in enumerate(batch)],
                return_exceptions=True,
            )
            for r in batch_results:
                if isinstance(r, MolecularReport):
                    results.append(r)
    else:
        for idx, (trace, source) in enumerate(all_traces):
            r = await process_trace(idx, trace, source)
            if r:
                results.append(r)

    total_duration = time.time() - overall_start

    # Save results
    if results:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_file = args.output_dir / f"molecular_{ts}.json"

        agg: dict[str, Any] = {
            "timestamp": ts,
            "model": args.model,
            "count": len(results),
            "duration": format_duration(total_duration),
            "averages": {
                "molecular": sum(r.avg_molecular for r in results) / len(results),
                "decontextuality": sum(r.avg_decontextuality for r in results)
                / len(results),
                "minimality": sum(r.avg_minimality for r in results) / len(results),
            },
            "classification_totals": {
                "molecular": sum(r.molecular_count for r in results),
                "too_atomic": sum(r.too_atomic_count for r in results),
                "too_verbose": sum(r.too_verbose_count for r in results),
                "borderline": sum(r.borderline_count for r in results),
            },
            "results": [r.to_dict() for r in results],
        }

        with open(out_file, "w") as f:
            json.dump(agg, f, indent=2)

        print(f"\nSaved to {out_file}")

        if len(results) > 1:
            print("\n" + "=" * 70)
            print("AGGREGATE RESULTS")
            print("=" * 70)
            print(
                f"Traces: {len(results)} | Duration: {format_duration(total_duration)}"
            )
            print(f"\nMolecular Score:     {agg['averages']['molecular']:.1%}")
            print(f"Decontextuality:     {agg['averages']['decontextuality']:.1%}")
            print(f"Minimality:          {agg['averages']['minimality']:.1%}")
            print("\nClassification:")
            total_props = sum(r.proposition_count for r in results)
            for k, v in agg["classification_totals"].items():
                print(f"  {k:<15} {v:4d} ({v/total_props*100:5.1f}%)")
        else:
            print(f"Duration: {format_duration(total_duration)}")
    else:
        print("\nNo traces evaluated")
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
