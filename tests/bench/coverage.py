"""
CoverageBench - Information Recall Benchmark for Fact Extraction
Author: Based on FActScore, SAFE, and QuestEval methodologies

## Core Principle

Measures how much extractable information from the source was actually captured.
Complements MolecularBench (quality) with recall metrics (quantity/coverage).

## Evaluation Pipeline

### Stage 1: Gold Fact Extraction
Extract ALL facts that could reasonably be derived from source messages.
Categories:
- Explicit facts (directly stated)
- Implicit facts (clearly implied by context)
- Relational facts (relationships between entities)

### Stage 2: Coverage Matching
For each gold fact, determine if it's covered by the extraction:
- COVERED: Semantically equivalent fact exists in extraction
- PARTIAL: Core information present but incomplete
- MISSING: Not present in extraction

### Stage 3: QA Verification (Optional)
Generate questions from source; verify answerability from extraction.

## Metrics

- Recall = covered_facts / gold_facts
- Partial Recall = (covered + 0.5 * partial) / gold_facts
- QA Coverage = answerable_questions / total_questions
- Density = extracted_facts / source_tokens
- F1 = harmonic_mean(molecular_quality, coverage_recall)

## Usage

python -m bench.coverage --traces path/to/traces.jsonl
"""

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================


class FactCategory(Enum):
    """Categories of extractable facts."""
    EXPLICIT = "explicit"           # Directly stated: "I work at Google"
    IMPLICIT = "implicit"           # Clearly implied: "my commute to Mountain View" → lives near MV
    RELATIONAL = "relational"       # Relationships: "my sister's husband" → has sister, sister is married
    TEMPORAL = "temporal"           # Time-bound facts: "started last year"
    PREFERENCE = "preference"       # Likes/dislikes: "I love hiking"
    BIOGRAPHICAL = "biographical"   # Personal details: name, age, location
    BEHAVIORAL = "behavioral"       # Habits/patterns: "I usually wake up early"


class CoverageStatus(Enum):
    """How well a gold fact is covered by extraction."""
    COVERED = "covered"             # Fully present (possibly rephrased)
    PARTIAL = "partial"             # Core info present but incomplete
    MISSING = "missing"             # Not present at all
    OVERCLAIMED = "overclaimed"     # Extraction claims more than source supports


class ImportanceLevel(Enum):
    """Importance weighting for facts (Pyramid-inspired)."""
    CRITICAL = "critical"           # Core identifying information
    IMPORTANT = "important"         # Significant details
    MINOR = "minor"                 # Nice-to-have details
    TRIVIAL = "trivial"             # Marginal information


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class GoldFact:
    """A fact that should be extractable from the source."""
    content: str
    category: FactCategory
    importance: ImportanceLevel
    source_span: str = ""           # The text that supports this fact
    requires_inference: bool = False # Whether extraction requires reasoning


@dataclass
class CoverageMatch:
    """Result of matching a gold fact against extraction."""
    gold_fact: GoldFact
    status: CoverageStatus
    matched_extraction: str = ""    # Which extracted fact covers this (if any)
    match_quality: float = 0.0      # 0-1, how well it matches
    explanation: str = ""


@dataclass 
class ExtractionAnalysis:
    """Analysis of an extracted fact."""
    content: str
    is_grounded: bool = True        # Supported by source
    is_hallucinated: bool = False   # Claims something not in source
    matched_gold: list[str] = field(default_factory=list)  # Which gold facts it covers


@dataclass
class CoverageReport:
    """Complete coverage analysis for a trace."""
    conversation_id: str
    source_message_count: int
    source_token_count: int
    
    # Gold facts
    gold_facts: list[GoldFact] = field(default_factory=list)
    gold_fact_count: int = 0
    
    # Extracted facts
    extracted_facts: list[str] = field(default_factory=list)
    extracted_count: int = 0
    
    # Coverage matching
    matches: list[CoverageMatch] = field(default_factory=list)
    
    # Core metrics
    recall: float = 0.0             # covered / gold
    partial_recall: float = 0.0     # (covered + 0.5*partial) / gold
    weighted_recall: float = 0.0    # importance-weighted recall
    precision: float = 0.0          # grounded / extracted
    f1: float = 0.0                 # harmonic mean
    
    # Detailed metrics
    coverage_by_category: dict[str, float] = field(default_factory=dict)
    coverage_by_importance: dict[str, float] = field(default_factory=dict)
    
    # Density metrics
    extraction_density: float = 0.0  # extracted / source_tokens
    gold_density: float = 0.0        # gold / source_tokens
    density_ratio: float = 0.0       # extraction_density / gold_density
    
    # QA verification (optional)
    qa_questions: list[str] = field(default_factory=list)
    qa_answerable: int = 0
    qa_coverage: float = 0.0
    
    # Issues
    missing_critical: list[str] = field(default_factory=list)
    hallucinations: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "source_messages": self.source_message_count,
            "source_tokens": self.source_token_count,
            "counts": {
                "gold_facts": self.gold_fact_count,
                "extracted_facts": self.extracted_count,
                "covered": sum(1 for m in self.matches if m.status == CoverageStatus.COVERED),
                "partial": sum(1 for m in self.matches if m.status == CoverageStatus.PARTIAL),
                "missing": sum(1 for m in self.matches if m.status == CoverageStatus.MISSING),
            },
            "scores": {
                "recall": round(self.recall, 4),
                "partial_recall": round(self.partial_recall, 4),
                "weighted_recall": round(self.weighted_recall, 4),
                "precision": round(self.precision, 4),
                "f1": round(self.f1, 4),
                "qa_coverage": round(self.qa_coverage, 4) if self.qa_questions else None,
            },
            "density": {
                "extraction_density": round(self.extraction_density, 4),
                "gold_density": round(self.gold_density, 4),
                "density_ratio": round(self.density_ratio, 4),
            },
            "coverage_by_category": {k: round(v, 3) for k, v in self.coverage_by_category.items()},
            "coverage_by_importance": {k: round(v, 3) for k, v in self.coverage_by_importance.items()},
            "issues": {
                "missing_critical": self.missing_critical[:5],  # Top 5
                "hallucinations": self.hallucinations[:5],
            },
            "gold_facts": [
                {
                    "content": gf.content,
                    "category": gf.category.value,
                    "importance": gf.importance.value,
                }
                for gf in self.gold_facts
            ],
            "matches": [
                {
                    "gold": m.gold_fact.content,
                    "status": m.status.value,
                    "matched_to": m.matched_extraction,
                    "quality": round(m.match_quality, 2),
                }
                for m in self.matches
            ],
        }


# =============================================================================
# PROMPTS
# =============================================================================


GOLD_EXTRACTION_PROMPT = """You are extracting ALL facts that could reasonably be derived from conversation messages.

## Your Task

Given messages from a conversation, extract every piece of factual information about the speaker.
Be EXHAUSTIVE - capture everything that a careful reader could learn about the speaker.

## Fact Categories

**EXPLICIT**: Directly stated facts
- "I work at Google" → "Speaker works at Google"

**IMPLICIT**: Clearly implied by context (not speculation)
- "My commute to the Googleplex is 30 minutes" → "Speaker lives within 30 minutes of Googleplex"

**RELATIONAL**: Facts about relationships
- "My sister's wedding was beautiful" → "Speaker has a sister", "Speaker's sister is married"

**TEMPORAL**: Time-bound information
- "I started this job last year" → "Speaker started current job within the past year"

**PREFERENCE**: Likes, dislikes, opinions
- "I love hiking on weekends" → "Speaker enjoys hiking"

**BIOGRAPHICAL**: Personal details
- "I'm 32 years old" → "Speaker is 32 years old"

**BEHAVIORAL**: Habits and patterns
- "I usually wake up at 6am" → "Speaker typically wakes up at 6am"

## Importance Levels

**CRITICAL**: Core identifying info (name, occupation, location, key relationships)
**IMPORTANT**: Significant details that build a clear picture
**MINOR**: Nice-to-have details
**TRIVIAL**: Marginal information that adds little value

## Rules

1. Extract ATOMIC facts (one claim per fact)
2. Resolve all pronouns and references to be standalone
3. Include the source span that supports each fact
4. Mark whether inference was required
5. Do NOT speculate beyond what the text clearly implies
6. Do NOT include facts about the assistant/system, only about the human speaker"""


COVERAGE_MATCHING_PROMPT = """You are evaluating how well extracted facts cover the gold standard facts.

## Your Task

For each GOLD FACT (what should have been extracted), determine if the EXTRACTED FACTS cover it.

## Coverage Statuses

**COVERED**: The gold fact is fully represented in the extraction
- Gold: "User works at Google"
- Extracted: "User is employed at Google as an engineer"
- Status: COVERED (employment at Google is captured, even with added detail)

**PARTIAL**: Core information present but incomplete
- Gold: "User has a sister who lives in Boston"
- Extracted: "User has a sister"
- Status: PARTIAL (sister exists, but location missing)

**MISSING**: The gold fact is not present at all
- Gold: "User enjoys hiking"
- Extracted: [no mention of hiking or outdoor activities]
- Status: MISSING

## Matching Rules

1. Semantic equivalence counts as COVERED (rephrasing is fine)
2. If extraction is MORE specific, still COVERED
3. If extraction is LESS specific, mark as PARTIAL
4. Consider entailment: If extracted fact entails gold fact, it's COVERED
5. Don't penalize for organization/grouping differences"""


QA_GENERATION_PROMPT = """Generate questions that should be answerable if information was fully extracted.

## Your Task

Given the source messages, generate questions about the speaker that a complete extraction would answer.

## Question Types

1. **Identity Questions**: Who/What is the speaker?
   - "What is the speaker's occupation?"
   - "Where does the speaker live?"

2. **Relationship Questions**: Who are the key people in their life?
   - "Does the speaker have siblings?"
   - "Is the speaker married?"

3. **Preference Questions**: What do they like/dislike?
   - "What are the speaker's hobbies?"
   - "What is the speaker's favorite food?"

4. **Temporal Questions**: When did events occur?
   - "When did the speaker start their current job?"
   - "How long has the speaker lived in their current city?"

5. **Behavioral Questions**: What are their habits?
   - "What time does the speaker typically wake up?"
   - "How does the speaker commute to work?"

## Rules

1. Only ask questions answerable from the source
2. Vary difficulty (some obvious, some requiring inference)
3. Cover different fact categories
4. Make questions specific enough to have clear answers"""


QA_VERIFICATION_PROMPT = """Verify which questions can be answered from the extracted facts alone.

## Your Task

For each question, determine if the EXTRACTED FACTS (not the source!) provide enough information to answer it.

## Answerability Levels

**ANSWERABLE**: The extracted facts directly answer or clearly imply the answer
**PARTIAL**: Some relevant information exists but answer is incomplete
**UNANSWERABLE**: The extracted facts don't contain relevant information

## Rules

1. Only consider the EXTRACTED FACTS, not your world knowledge
2. The answer must be derivable from the extraction alone
3. If the question requires combining multiple extracted facts, that's fine
4. Partial credit for incomplete answers"""


# =============================================================================
# COVERAGE JUDGE
# =============================================================================


class CoverageJudge:
    """
    Evaluates information recall/coverage in fact extraction.
    
    Based on:
    - FActScore: Atomic fact decomposition
    - SAFE: F1 scoring with precision and recall
    - QuestEval: QA-based coverage verification
    - Pyramid: Importance weighting
    """
    
    def __init__(
        self,
        llm_client: AsyncAnthropic | AsyncOpenAI,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        use_qa_verification: bool = True,
        verbose: bool = False,
    ):
        self.llm_client = llm_client
        self.model = model
        self.provider = provider
        self.use_qa_verification = use_qa_verification
        self.verbose = verbose
        
        if verbose:
            logger.setLevel(logging.DEBUG)
    
    async def _call_llm(
        self, 
        system: str, 
        user: str, 
        tool_def: dict[str, Any]
    ) -> dict[str, Any]:
        """Call LLM with structured tool output."""
        try:
            if isinstance(self.llm_client, AsyncAnthropic):
                resp = await asyncio.wait_for(
                    self.llm_client.messages.create(
                        model=self.model,
                        max_tokens=8192,
                        temperature=0.0,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                        tools=[tool_def],
                        tool_choice={"type": "tool", "name": tool_def["name"]},
                    ),
                    timeout=300.0,
                )
                for block in resp.content:
                    if block.type == "tool_use":
                        return dict(block.input) if isinstance(block.input, dict) else {}
                return {}
            else:
                # OpenAI-compatible
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool_def["name"],
                        "description": tool_def.get("description", ""),
                        "parameters": tool_def["input_schema"],
                    },
                }
                resp = await asyncio.wait_for(
                    self.llm_client.chat.completions.create(
                        model=self.model,
                        max_tokens=8192,
                        temperature=0.0,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        tools=[openai_tool],
                        tool_choice={"type": "function", "function": {"name": tool_def["name"]}},
                    ),
                    timeout=300.0,
                )
                if resp.choices and resp.choices[0].message.tool_calls:
                    func = resp.choices[0].message.tool_calls[0].function
                    if func.arguments:
                        return json.loads(func.arguments)
                return {}
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {}

    async def extract_gold_facts(self, source_messages: list[dict[str, Any]]) -> list[GoldFact]:
        """
        Stage 1: Extract all facts that SHOULD be extractable from source.
        
        This defines the "gold standard" for recall measurement.
        """
        
        tool_def = {
            "name": "submit_gold_facts",
            "description": "Submit all extractable facts from source messages",
            "input_schema": {
                "type": "object",
                "properties": {
                    "facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "The fact as a standalone statement"
                                },
                                "category": {
                                    "type": "string",
                                    "enum": [c.value for c in FactCategory]
                                },
                                "importance": {
                                    "type": "string",
                                    "enum": [i.value for i in ImportanceLevel]
                                },
                                "source_span": {
                                    "type": "string",
                                    "description": "The text that supports this fact"
                                },
                                "requires_inference": {
                                    "type": "boolean",
                                    "description": "Whether extracting this requires reasoning beyond literal text"
                                }
                            },
                            "required": ["content", "category", "importance"]
                        }
                    }
                },
                "required": ["facts"]
            }
        }
        
        # Format messages for LLM
        messages_text = "\n".join(
            f"[{msg.get('speaker', 'user')}]: {msg.get('text', '')}"
            for msg in source_messages
            if msg.get('speaker', 'user') == 'user'  # Only user messages
        )
        
        result = await self._call_llm(
            GOLD_EXTRACTION_PROMPT,
            f"Extract all facts from these messages:\n\n{messages_text}",
            tool_def
        )
        
        gold_facts = []
        for item in result.get("facts", []):
            try:
                gold_facts.append(GoldFact(
                    content=item.get("content", ""),
                    category=FactCategory(item.get("category", "explicit")),
                    importance=ImportanceLevel(item.get("importance", "important")),
                    source_span=item.get("source_span", ""),
                    requires_inference=item.get("requires_inference", False),
                ))
            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping malformed gold fact: {e}")
                continue
        
        logger.info(f"Extracted {len(gold_facts)} gold facts from source")
        return gold_facts

    async def match_coverage(
        self,
        gold_facts: list[GoldFact],
        extracted_facts: list[str]
    ) -> list[CoverageMatch]:
        """
        Stage 2: Match gold facts against extraction to measure coverage.
        """
        
        if not gold_facts:
            return []
        
        tool_def = {
            "name": "submit_coverage_matches",
            "description": "Submit coverage matching results",
            "input_schema": {
                "type": "object",
                "properties": {
                    "matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "gold_index": {"type": "integer"},
                                "status": {
                                    "type": "string",
                                    "enum": [s.value for s in CoverageStatus]
                                },
                                "matched_extraction": {
                                    "type": "string",
                                    "description": "The extracted fact that covers this (if any)"
                                },
                                "match_quality": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                    "description": "How well it matches (1.0 = perfect)"
                                },
                                "explanation": {"type": "string"}
                            },
                            "required": ["gold_index", "status"]
                        }
                    }
                },
                "required": ["matches"]
            }
        }
        
        # Format for LLM
        gold_text = "\n".join(
            f"{i+1}. [{gf.importance.value.upper()}] {gf.content}"
            for i, gf in enumerate(gold_facts)
        )
        
        extracted_text = "\n".join(
            f"- {fact}" for fact in extracted_facts
        ) if extracted_facts else "(No facts extracted)"
        
        result = await self._call_llm(
            COVERAGE_MATCHING_PROMPT,
            f"GOLD FACTS (what should be extracted):\n{gold_text}\n\n"
            f"EXTRACTED FACTS (what was actually extracted):\n{extracted_text}\n\n"
            f"For each gold fact, determine its coverage status.",
            tool_def
        )
        
        matches = []
        for item in result.get("matches", []):
            idx = item.get("gold_index", 1) - 1
            if 0 <= idx < len(gold_facts):
                try:
                    matches.append(CoverageMatch(
                        gold_fact=gold_facts[idx],
                        status=CoverageStatus(item.get("status", "missing")),
                        matched_extraction=item.get("matched_extraction", ""),
                        match_quality=item.get("match_quality", 0.0),
                        explanation=item.get("explanation", ""),
                    ))
                except ValueError:
                    matches.append(CoverageMatch(
                        gold_fact=gold_facts[idx],
                        status=CoverageStatus.MISSING,
                    ))
        
        # Ensure all gold facts have a match result
        matched_indices = {gold_facts.index(m.gold_fact) for m in matches if m.gold_fact in gold_facts}
        for i, gf in enumerate(gold_facts):
            if i not in matched_indices:
                matches.append(CoverageMatch(
                    gold_fact=gf,
                    status=CoverageStatus.MISSING,
                    explanation="No match result returned"
                ))
        
        return matches

    async def generate_qa_pairs(
        self,
        source_messages: list[dict[str, Any]]
    ) -> list[str]:
        """
        Stage 3a: Generate questions that should be answerable from complete extraction.
        """
        
        tool_def = {
            "name": "submit_questions",
            "description": "Submit questions for QA-based coverage verification",
            "input_schema": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "expected_answer": {"type": "string"},
                                "difficulty": {
                                    "type": "string",
                                    "enum": ["easy", "medium", "hard"]
                                }
                            },
                            "required": ["question"]
                        }
                    }
                },
                "required": ["questions"]
            }
        }
        
        messages_text = "\n".join(
            f"[{msg.get('speaker', 'user')}]: {msg.get('text', '')}"
            for msg in source_messages
            if msg.get('speaker', 'user') == 'user'
        )
        
        result = await self._call_llm(
            QA_GENERATION_PROMPT,
            f"Generate questions from these messages:\n\n{messages_text}",
            tool_def
        )
        
        questions = [
            item.get("question", "")
            for item in result.get("questions", [])
            if item.get("question")
        ]
        
        logger.info(f"Generated {len(questions)} QA questions")
        return questions

    async def verify_qa_coverage(
        self,
        questions: list[str],
        extracted_facts: list[str]
    ) -> tuple[int, int]:
        """
        Stage 3b: Verify how many questions can be answered from extraction alone.
        
        Returns: (answerable_count, total_count)
        """
        
        if not questions:
            return 0, 0
        
        tool_def = {
            "name": "submit_qa_results",
            "description": "Submit QA verification results",
            "input_schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question_index": {"type": "integer"},
                                "answerable": {
                                    "type": "string",
                                    "enum": ["yes", "partial", "no"]
                                },
                                "answer_from_extraction": {"type": "string"},
                                "explanation": {"type": "string"}
                            },
                            "required": ["question_index", "answerable"]
                        }
                    }
                },
                "required": ["results"]
            }
        }
        
        questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        extracted_text = "\n".join(f"- {f}" for f in extracted_facts) if extracted_facts else "(No facts)"
        
        result = await self._call_llm(
            QA_VERIFICATION_PROMPT,
            f"QUESTIONS:\n{questions_text}\n\n"
            f"EXTRACTED FACTS:\n{extracted_text}\n\n"
            f"For each question, determine if it can be answered from the extracted facts alone.",
            tool_def
        )
        
        answerable = 0
        for item in result.get("results", []):
            status = item.get("answerable", "no")
            if status == "yes":
                answerable += 1
            elif status == "partial":
                answerable += 0.5
        
        return int(answerable), len(questions)

    async def analyze_extraction_quality(
        self,
        extracted_facts: list[str],
        source_messages: list[dict[str, Any]]
    ) -> list[ExtractionAnalysis]:
        """
        Analyze each extracted fact for grounding and hallucination.
        """
        
        tool_def = {
            "name": "submit_extraction_analysis",
            "description": "Analyze extracted facts for grounding",
            "input_schema": {
                "type": "object",
                "properties": {
                    "analyses": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "is_grounded": {"type": "boolean"},
                                "is_hallucinated": {"type": "boolean"},
                                "explanation": {"type": "string"}
                            },
                            "required": ["index", "is_grounded"]
                        }
                    }
                },
                "required": ["analyses"]
            }
        }
        
        messages_text = "\n".join(
            f"[{msg.get('speaker', 'user')}]: {msg.get('text', '')}"
            for msg in source_messages
            if msg.get('speaker', 'user') == 'user'
        )
        
        extracted_text = "\n".join(f"{i+1}. {f}" for i, f in enumerate(extracted_facts))
        
        result = await self._call_llm(
            "Verify each extracted fact is grounded in the source messages. "
            "A fact is GROUNDED if the source text supports it. "
            "A fact is HALLUCINATED if it claims something not supported by the source.",
            f"SOURCE MESSAGES:\n{messages_text}\n\n"
            f"EXTRACTED FACTS:\n{extracted_text}\n\n"
            f"Analyze each extracted fact.",
            tool_def
        )
        
        analyses = []
        for i, fact in enumerate(extracted_facts):
            analyses.append(ExtractionAnalysis(content=fact))
        
        for item in result.get("analyses", []):
            idx = item.get("index", 1) - 1
            if 0 <= idx < len(analyses):
                analyses[idx].is_grounded = item.get("is_grounded", True)
                analyses[idx].is_hallucinated = item.get("is_hallucinated", False)
        
        return analyses

    def compute_metrics(self, report: CoverageReport) -> None:
        """Compute all coverage metrics from matches."""
        
        if not report.gold_facts:
            return
        
        n_gold = len(report.gold_facts)
        
        # Count by status
        covered = sum(1 for m in report.matches if m.status == CoverageStatus.COVERED)
        partial = sum(1 for m in report.matches if m.status == CoverageStatus.PARTIAL)
        
        # Basic recall
        report.recall = covered / n_gold if n_gold > 0 else 0
        report.partial_recall = (covered + 0.5 * partial) / n_gold if n_gold > 0 else 0
        
        # Weighted recall (by importance)
        importance_weights = {
            ImportanceLevel.CRITICAL: 3.0,
            ImportanceLevel.IMPORTANT: 2.0,
            ImportanceLevel.MINOR: 1.0,
            ImportanceLevel.TRIVIAL: 0.5,
        }
        
        total_weight = sum(importance_weights[gf.importance] for gf in report.gold_facts)
        covered_weight = sum(
            importance_weights[m.gold_fact.importance]
            for m in report.matches
            if m.status == CoverageStatus.COVERED
        )
        partial_weight = sum(
            importance_weights[m.gold_fact.importance] * 0.5
            for m in report.matches
            if m.status == CoverageStatus.PARTIAL
        )
        
        report.weighted_recall = (covered_weight + partial_weight) / total_weight if total_weight > 0 else 0
        
        # Precision (grounded extractions / total extractions)
        if report.extracted_count > 0:
            # This would need extraction analysis, simplified here
            report.precision = 1.0  # Assume all grounded unless analyzed
        
        # F1
        if report.recall + report.precision > 0:
            report.f1 = 2 * report.recall * report.precision / (report.recall + report.precision)
        
        # Coverage by category
        for category in FactCategory:
            cat_facts = [m for m in report.matches if m.gold_fact.category == category]
            if cat_facts:
                cat_covered = sum(1 for m in cat_facts if m.status == CoverageStatus.COVERED)
                report.coverage_by_category[category.value] = cat_covered / len(cat_facts)
        
        # Coverage by importance
        for importance in ImportanceLevel:
            imp_facts = [m for m in report.matches if m.gold_fact.importance == importance]
            if imp_facts:
                imp_covered = sum(1 for m in imp_facts if m.status == CoverageStatus.COVERED)
                report.coverage_by_importance[importance.value] = imp_covered / len(imp_facts)
        
        # Density metrics
        if report.source_token_count > 0:
            report.extraction_density = report.extracted_count / report.source_token_count
            report.gold_density = n_gold / report.source_token_count
            report.density_ratio = report.extraction_density / report.gold_density if report.gold_density > 0 else 0
        
        # Track missing critical facts
        report.missing_critical = [
            m.gold_fact.content
            for m in report.matches
            if m.status == CoverageStatus.MISSING and m.gold_fact.importance == ImportanceLevel.CRITICAL
        ]

    async def evaluate(
        self,
        extracted_facts: list[str],
        source_messages: list[dict[str, Any]],
        conversation_id: str = "",
    ) -> CoverageReport:
        """
        Run full coverage evaluation pipeline.
        """
        
        # Estimate token count
        source_text = " ".join(
            msg.get("text", "") for msg in source_messages
            if msg.get("speaker", "user") == "user"
        )
        source_tokens = len(source_text.split())  # Rough estimate
        
        report = CoverageReport(
            conversation_id=conversation_id,
            source_message_count=len([m for m in source_messages if m.get("speaker") == "user"]),
            source_token_count=source_tokens,
            extracted_facts=extracted_facts,
            extracted_count=len(extracted_facts),
        )
        
        logger.info(f"Evaluating coverage for {conversation_id}...")
        
        # Stage 1: Extract gold facts
        logger.info("Stage 1: Extracting gold facts...")
        report.gold_facts = await self.extract_gold_facts(source_messages)
        report.gold_fact_count = len(report.gold_facts)
        
        if not report.gold_facts:
            logger.warning(f"No gold facts extracted for {conversation_id}")
            return report
        
        # Stage 2: Match coverage
        logger.info(f"Stage 2: Matching {len(extracted_facts)} extracted against {len(report.gold_facts)} gold facts...")
        report.matches = await self.match_coverage(report.gold_facts, extracted_facts)
        
        # Stage 3: QA verification (optional)
        if self.use_qa_verification:
            logger.info("Stage 3: QA-based verification...")
            report.qa_questions = await self.generate_qa_pairs(source_messages)
            if report.qa_questions:
                answerable, total = await self.verify_qa_coverage(report.qa_questions, extracted_facts)
                report.qa_answerable = answerable
                report.qa_coverage = answerable / total if total > 0 else 0
        
        # Compute all metrics
        self.compute_metrics(report)
        
        logger.info(
            f"Coverage complete: recall={report.recall:.1%}, "
            f"partial_recall={report.partial_recall:.1%}, "
            f"weighted_recall={report.weighted_recall:.1%}"
        )
        
        return report


# =============================================================================
# COMBINED SCORER (Molecular + Coverage)
# =============================================================================


@dataclass
class CombinedScore:
    """Combined molecular quality + coverage recall score."""
    
    # Individual scores
    molecular: float = 0.0          # From MolecularBench
    decontextuality: float = 0.0
    minimality: float = 0.0
    
    coverage: float = 0.0           # From CoverageBench
    weighted_coverage: float = 0.0
    qa_coverage: float = 0.0
    
    # Combined scores
    f1: float = 0.0                 # Harmonic mean of quality and recall
    f2: float = 0.0                 # F2 weights recall higher
    f05: float = 0.0                # F0.5 weights precision higher
    
    # For training data filtering
    passes_threshold: bool = False
    
    def compute_combined(
        self,
        quality_weight: float = 1.0,
        coverage_weight: float = 1.5,  # Weight coverage higher by default
    ):
        """Compute F-beta scores."""
        
        quality = self.molecular  # Use molecular as quality proxy
        recall = self.weighted_coverage or self.coverage
        
        if quality + recall == 0:
            return
        
        # F1 (balanced)
        self.f1 = 2 * quality * recall / (quality + recall)
        
        # F2 (recall-weighted)
        beta = 2
        self.f2 = (1 + beta**2) * quality * recall / (beta**2 * quality + recall)
        
        # F0.5 (precision-weighted)
        beta = 0.5
        self.f05 = (1 + beta**2) * quality * recall / (beta**2 * quality + recall)


def compute_combined_score(
    molecular_score: float,
    decontextuality: float,
    minimality: float,
    coverage: float,
    weighted_coverage: float = None,
    qa_coverage: float = None,
    min_coverage: float = 0.5,
    min_molecular: float = 0.8,
) -> CombinedScore:
    """
    Combine molecular quality with coverage recall.
    
    Args:
        molecular_score: From MolecularBench
        decontextuality: Decontextuality score
        minimality: Minimality score
        coverage: Basic recall score
        weighted_coverage: Importance-weighted recall
        qa_coverage: QA-based coverage
        min_coverage: Minimum coverage threshold
        min_molecular: Minimum molecular quality threshold
    """
    
    score = CombinedScore(
        molecular=molecular_score,
        decontextuality=decontextuality,
        minimality=minimality,
        coverage=coverage,
        weighted_coverage=weighted_coverage or coverage,
        qa_coverage=qa_coverage or 0.0,
    )
    
    score.compute_combined()
    
    # Check if passes both thresholds
    score.passes_threshold = (
        score.molecular >= min_molecular and
        score.coverage >= min_coverage
    )
    
    return score


# =============================================================================
# TRACE PARSING
# =============================================================================


def extract_propositions(trace: dict[str, Any]) -> list[str]:
    """Extract propositions from trace output."""
    output = trace.get("output", {})
    if not isinstance(output, dict):
        return []
    content = output.get("content", {})
    if not isinstance(content, dict):
        return []
    explicit = content.get("explicit", [])
    if not isinstance(explicit, list):
        return []
    return [
        item["content"] 
        for item in explicit 
        if isinstance(item, dict) and "content" in item
    ]


def extract_messages(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract source messages from trace input."""
    input_data = trace.get("input", {})
    if not isinstance(input_data, dict):
        return []
    prompt = input_data.get("prompt", "")
    if not isinstance(prompt, str):
        return []
    
    messages = []
    if "<messages>" in prompt:
        section = prompt.split("<messages>")[1].split("</messages>")[0]
        for line in section.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 3)
            if len(parts) >= 3:
                speaker = parts[2].rstrip(":")
                text = parts[3] if len(parts) > 3 else ""
                if "->->" in text:
                    text = text.split("->->")[0].strip()
                messages.append({"speaker": speaker, "text": text})
    return messages


def load_traces(path: Path) -> list[dict[str, Any]]:
    """Load traces from JSON or JSONL file."""
    traces = []
    
    try:
        with open(path) as f:
            first_line = f.readline().strip()
            if first_line and not first_line.startswith("["):
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            traces.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                if traces:
                    return traces
    except Exception:
        pass
    
    with open(path) as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    return []


# =============================================================================
# OUTPUT
# =============================================================================


def print_report(report: CoverageReport) -> None:
    """Print formatted coverage report."""
    
    print("\n" + "=" * 70)
    print(f"COVERAGE ANALYSIS: {report.conversation_id}")
    print("=" * 70)
    
    print(f"\nSource: {report.source_message_count} messages, ~{report.source_token_count} tokens")
    print(f"Gold Facts: {report.gold_fact_count} | Extracted: {report.extracted_count}")
    
    # Coverage counts
    covered = sum(1 for m in report.matches if m.status == CoverageStatus.COVERED)
    partial = sum(1 for m in report.matches if m.status == CoverageStatus.PARTIAL)
    missing = sum(1 for m in report.matches if m.status == CoverageStatus.MISSING)
    
    print(f"\nCoverage: {covered} covered, {partial} partial, {missing} missing")
    
    # Scores
    print(f"\n{'RECALL SCORES'}")
    print("-" * 40)
    print(f"{'Basic Recall:':<25} {report.recall:.1%}")
    print(f"{'Partial Recall:':<25} {report.partial_recall:.1%}")
    print(f"{'Weighted Recall:':<25} {report.weighted_recall:.1%}")
    if report.qa_coverage > 0:
        print(f"{'QA Coverage:':<25} {report.qa_coverage:.1%} ({report.qa_answerable}/{len(report.qa_questions)})")
    
    # Density
    print(f"\n{'DENSITY'}")
    print("-" * 40)
    print(f"{'Extraction Density:':<25} {report.extraction_density:.4f} facts/token")
    print(f"{'Gold Density:':<25} {report.gold_density:.4f} facts/token")
    print(f"{'Density Ratio:':<25} {report.density_ratio:.1%}")
    
    # By category
    if report.coverage_by_category:
        print(f"\n{'COVERAGE BY CATEGORY'}")
        print("-" * 40)
        for cat, cov in sorted(report.coverage_by_category.items(), key=lambda x: -x[1]):
            print(f"  {cat:<20} {cov:.1%}")
    
    # By importance
    if report.coverage_by_importance:
        print(f"\n{'COVERAGE BY IMPORTANCE'}")
        print("-" * 40)
        for imp, cov in sorted(report.coverage_by_importance.items(), key=lambda x: -x[1]):
            print(f"  {imp:<20} {cov:.1%}")
    
    # Missing critical
    if report.missing_critical:
        print(f"\n⚠️ MISSING CRITICAL FACTS:")
        for fact in report.missing_critical[:5]:
            print(f"  • {fact[:60]}...")
    
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================


async def main():
    parser = argparse.ArgumentParser(description="CoverageBench - Information Recall Evaluation")
    parser.add_argument("--traces", type=Path, help="JSON/JSONL trace file")
    parser.add_argument("--trace-dir", type=Path, help="Directory of trace files")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/coverage_results"))
    parser.add_argument("--provider", choices=["anthropic", "openai", "openrouter"], default="anthropic")
    parser.add_argument("--api-key", type=str, help="API key")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--no-qa", action="store_true", help="Skip QA verification")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--limit", type=int, help="Limit number of traces")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent evaluations")
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv(f"{args.provider.upper()}_API_KEY")
    if not api_key and args.provider == "anthropic":
        api_key = os.getenv("LLM_ANTHROPIC_API_KEY")
    if not api_key:
        logger.error(f"Set {args.provider.upper()}_API_KEY or use --api-key")
        return 1
    
    # Initialize client
    if args.provider == "anthropic":
        client = AsyncAnthropic(api_key=api_key)
    else:
        base_url = "https://openrouter.ai/api/v1" if args.provider == "openrouter" else None
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    judge = CoverageJudge(
        client, 
        args.model, 
        args.provider,
        use_qa_verification=not args.no_qa,
        verbose=args.verbose
    )
    
    # Load traces
    all_traces = []
    if args.traces:
        traces = load_traces(args.traces)
        all_traces.extend((t, args.traces.name) for t in traces)
    elif args.trace_dir:
        for f in list(args.trace_dir.glob("*.json")) + list(args.trace_dir.glob("*.jsonl")):
            traces = load_traces(f)
            all_traces.extend((t, f.name) for t in traces)
    else:
        parser.error("Specify --traces or --trace-dir")
    
    if args.limit:
        all_traces = all_traces[:args.limit]
    
    print(f"Evaluating {len(all_traces)} traces for coverage with concurrency={args.concurrency}...\n")

    # Semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(args.concurrency)

    async def evaluate_trace(idx: int, trace: dict, source: str) -> tuple[int, CoverageReport | None]:
        """Evaluate a single trace with concurrency control."""
        async with semaphore:
            props = extract_propositions(trace)
            msgs = extract_messages(trace)
            conv_id = trace.get("conversation_id", f"trace_{idx:04d}")

            try:
                report = await judge.evaluate(props, msgs, conv_id)
                return idx, report
            except Exception as e:
                logger.error(f"Failed {conv_id}: {e}")
                return idx, None

    # Process all traces concurrently with progress bar
    tasks = [evaluate_trace(idx, trace, source) for idx, (trace, source) in enumerate(all_traces)]
    results_raw = []

    for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Evaluating traces"):
        idx, report = await coro
        if report:
            results_raw.append((idx, report))

    # Sort by original index and extract reports
    results_raw.sort(key=lambda x: x[0])
    results: list[CoverageReport] = [report for _, report in results_raw]

    # Print detailed reports
    if args.verbose:
        print("\n" + "=" * 70)
        print("DETAILED REPORTS")
        print("=" * 70)
        for report in results:
            print_report(report)
    
    # Save results
    if results:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_file = args.output_dir / f"coverage_{ts}.json"
        
        agg = {
            "timestamp": ts,
            "model": args.model,
            "count": len(results),
            "averages": {
                "recall": sum(r.recall for r in results) / len(results),
                "partial_recall": sum(r.partial_recall for r in results) / len(results),
                "weighted_recall": sum(r.weighted_recall for r in results) / len(results),
                "qa_coverage": sum(r.qa_coverage for r in results) / len(results),
                "density_ratio": sum(r.density_ratio for r in results) / len(results),
            },
            "totals": {
                "gold_facts": sum(r.gold_fact_count for r in results),
                "extracted_facts": sum(r.extracted_count for r in results),
            },
            "results": [r.to_dict() for r in results],
        }
        
        with open(out_file, "w") as f:
            json.dump(agg, f, indent=2)
        
        print(f"\n✅ Saved to {out_file}")
        
        # Aggregate summary
        print("\n" + "=" * 70)
        print("AGGREGATE RESULTS")
        print("=" * 70)
        print(f"Traces: {len(results)}")
        print(f"Total Gold Facts: {agg['totals']['gold_facts']}")
        print(f"Total Extracted: {agg['totals']['extracted_facts']}")
        print(f"\nRecall:          {agg['averages']['recall']:.1%}")
        print(f"Partial Recall:  {agg['averages']['partial_recall']:.1%}")
        print(f"Weighted Recall: {agg['averages']['weighted_recall']:.1%}")
        print(f"QA Coverage:     {agg['averages']['qa_coverage']:.1%}")
        print(f"Density Ratio:   {agg['averages']['density_ratio']:.1%}")
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))