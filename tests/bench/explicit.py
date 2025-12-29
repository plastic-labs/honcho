"""
ExplicitBench - Explicit Derivation Benchmark
Author: 3un01a (3un01a@plasticlabs.ai)

A single-file implementation of the 5-axis evaluation system for explicit derivations.

## To Run:

1. Save this file to: tests/bench/explicit.py

2. Set your API key (or pass via --api-key):
   export ANTHROPIC_API_KEY=your_key_here
   # or
   export OPENAI_API_KEY=your_key_here
   # or
   export OPENROUTER_API_KEY=your_key_here

3. Run against a JSON or JSONL file of traces:
   python -m tests.bench.explicit --traces path/to/traces.json
   python -m tests.bench.explicit --traces path/to/traces.jsonl

4. Run against a directory of JSON/JSONL files:
   python -m tests.bench.explicit --trace-dir path/to/traces/

5. Optional flags:
   --provider anthropic  # Provider: anthropic, openai, or openrouter (default: anthropic)
   --api-key your_key_here  # API key (overrides environment variable)
   --model claude-sonnet-4-20250514  # Model for evaluation (default)
   --output-dir tests/bench/eval_results  # Where to save results
   --verbose  # Enable detailed logging
   --weights '{"coverage": 0.35, "atomicity": 0.15}'  # Custom score weights
   --limit 10  # Only evaluate first N traces from the file
   --sample 0.1  # Randomly sample 10% of traces
   --batch-size 5  # Process N traces concurrently (default: 1)

## Input Format (Trace JSON/JSONL):

The script accepts two formats:

1. JSON array:
[
  {"input": {"prompt": "...<messages>...</messages>..."}, "output": {"content": {"explicit": [{"content": "prop1"}, ...]}}},
  {"input": {"prompt": "...<messages>...</messages>..."}, "output": {"content": {"explicit": [{"content": "prop1"}, ...]}}},
  ...
]

2. JSONL (one JSON object per line):
{"input": {"prompt": "...<messages>...</messages>..."}, "output": {"content": {"explicit": [{"content": "prop1"}, ...]}}}
{"input": {"prompt": "...<messages>...</messages>..."}, "output": {"content": {"explicit": [{"content": "prop1"}, ...]}}}
...

## Output:

- Console summary with scores and notable issues
- JSON file with detailed evaluation results
"""

import argparse
import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, cast

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class TemporalType(Enum):
    STATIC = "static"
    DYNAMIC_STATE = "dynamic"
    EVENT = "event"
    DURATION = "duration"
    HABITUAL = "habitual"
    UNKNOWN = "unknown"


class AtomicityViolation(Enum):
    CONJUNCTION = "conjunction"
    DISJUNCTION = "disjunction"
    CONDITIONAL = "conditional"
    EMBEDDED_QUOTE = "embedded_quote"
    COMPOUND_PREDICATE = "compound_predicate"
    CAUSAL_CHAIN = "causal_chain"
    TEMPORAL_SEQUENCE = "temporal_sequence"


class FidelityViolation(Enum):
    HEDGE_REMOVAL = "hedge_removal"
    NEGATION_FLIP = "negation_flip"
    TEMPORAL_SHIFT = "temporal_shift"
    QUANTITY_CHANGE = "quantity_change"
    ATTRIBUTION_ERROR = "attribution_error"
    OVERGENERALIZATION = "overgeneralization"
    OVERSPECIFICATION = "overspecification"
    INFERENCE_AS_EXPLICIT = "inference_as_explicit"


class PremiseSuitability(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    MARGINAL = "marginal"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class AtomicityResult:
    proposition: str
    is_atomic: bool
    violations: list[AtomicityViolation] = field(default_factory=list)
    suggested_decomposition: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class CoverageGap:
    missing_claim: str
    source_quote: str
    source_message_id: str
    severity: str
    reasoning: str = ""


@dataclass
class RedundancyCluster:
    propositions: list[str]
    canonical_form: str
    redundancy_type: str


@dataclass
class FidelityResult:
    proposition: str
    is_faithful: bool
    violations: list[tuple[FidelityViolation, str]] = field(default_factory=list)
    source_quote: str = ""
    severity: str = "none"


@dataclass
class DownstreamUtilityResult:
    proposition: str
    suitability: PremiseSuitability
    issues: list[str] = field(default_factory=list)
    has_clear_subject: bool = True
    has_clear_predicate: bool = True
    is_contextually_complete: bool = True
    has_stable_truth_value: bool = True
    is_composable: bool = True
    temporal_handling: str = "appropriate"
    reasoning: str = ""


@dataclass
class AtomicityReport:
    total_propositions: int
    atomic_count: int
    score: float
    violations_by_type: dict[str, int] = field(default_factory=dict)
    detailed_results: list[AtomicityResult] = field(default_factory=list)
    decomposition_suggestions: int = 0
    estimated_atomic_propositions: int = 0


@dataclass
class CoverageReport:
    total_source_claims: int
    extracted_claims: int
    coverage_score: float
    gaps: list[CoverageGap] = field(default_factory=list)
    gaps_by_severity: dict[str, int] = field(default_factory=dict)
    source_message_count: int = 0
    propositions_per_message: float = 0.0


@dataclass
class FidelityReport:
    total_propositions: int
    faithful_count: int
    fidelity_score: float
    violations_by_type: dict[str, int] = field(default_factory=dict)
    violations_by_severity: dict[str, int] = field(default_factory=dict)
    detailed_results: list[FidelityResult] = field(default_factory=list)


@dataclass
class EfficiencyReport:
    total_propositions: int
    unique_propositions: int
    efficiency_score: float
    redundancy_clusters: list[RedundancyCluster] = field(default_factory=list)
    exact_duplicates: int = 0
    near_duplicates: int = 0
    subsumptions: int = 0


@dataclass
class DownstreamUtilityReport:
    total_propositions: int
    suitability_distribution: dict[str, int] = field(default_factory=dict)
    clarity_score: float = 0.0
    completeness_score: float = 0.0
    stability_score: float = 0.0
    composability_score: float = 0.0
    temporal_score: float = 0.0
    utility_score: float = 0.0
    detailed_results: list[DownstreamUtilityResult] = field(default_factory=list)


@dataclass
class EvaluationResult:
    conversation_id: str
    peer_name: str
    proposition_count: int
    source_message_count: int
    atomicity: AtomicityReport
    coverage: CoverageReport
    fidelity: FidelityReport
    efficiency: EfficiencyReport
    downstream_utility: DownstreamUtilityReport
    overall_score: float = 0.0
    
    def compute_overall_score(self, weights: dict[str, float] | None = None) -> float:
        w = weights or {
            "atomicity": 0.15,
            "coverage": 0.35,
            "fidelity": 0.20,
            "efficiency": 0.10,
            "utility": 0.20,
        }
        self.overall_score = (
            self.atomicity.score * w.get("atomicity", 0.15) +
            self.coverage.coverage_score * w.get("coverage", 0.35) +
            self.fidelity.fidelity_score * w.get("fidelity", 0.20) +
            self.efficiency.efficiency_score * w.get("efficiency", 0.10) +
            self.downstream_utility.utility_score * w.get("utility", 0.20)
        )
        return self.overall_score
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "peer_name": self.peer_name,
            "proposition_count": self.proposition_count,
            "source_message_count": self.source_message_count,
            "scores": {
                "overall": round(self.overall_score, 4),
                "atomicity": round(self.atomicity.score, 4),
                "coverage": round(self.coverage.coverage_score, 4),
                "fidelity": round(self.fidelity.fidelity_score, 4),
                "efficiency": round(self.efficiency.efficiency_score, 4),
                "downstream_utility": round(self.downstream_utility.utility_score, 4),
            },
            "atomicity_details": {
                "atomic_count": self.atomicity.atomic_count,
                "total": self.atomicity.total_propositions,
                "violations_by_type": self.atomicity.violations_by_type,
            },
            "coverage_details": {
                "estimated_total": self.coverage.total_source_claims,
                "gaps_count": len(self.coverage.gaps),
                "gaps_by_severity": self.coverage.gaps_by_severity,
                "propositions_per_message": round(self.coverage.propositions_per_message, 2),
                "gaps": [
                    {"claim": g.missing_claim, "severity": g.severity}
                    for g in self.coverage.gaps
                ],
            },
            "fidelity_details": {
                "faithful_count": self.fidelity.faithful_count,
                "violations_by_type": self.fidelity.violations_by_type,
            },
            "efficiency_details": {
                "unique_propositions": self.efficiency.unique_propositions,
                "redundancy_clusters": len(self.efficiency.redundancy_clusters),
            },
            "utility_details": {
                "suitability_distribution": self.downstream_utility.suitability_distribution,
                "component_scores": {
                    "clarity": round(self.downstream_utility.clarity_score, 4),
                    "completeness": round(self.downstream_utility.completeness_score, 4),
                    "stability": round(self.downstream_utility.stability_score, 4),
                    "composability": round(self.downstream_utility.composability_score, 4),
                    "temporal": round(self.downstream_utility.temporal_score, 4),
                },
            },
        }


# =============================================================================
# EVALUATION PROMPTS
# =============================================================================

ATOMICITY_CRITERIA = """## Atomicity Evaluation

A proposition is ATOMIC if it contains exactly ONE claim with ONE truth value.

### Violations:
- CONJUNCTION: Multiple claims joined by "and" ("User has a dog and lives in NYC")
- DISJUNCTION: Alternatives with "or" ("User works at Google or Microsoft")  
- CONDITIONAL: If/then structure ("If user gets the job, they will move")
- EMBEDDED_QUOTE: Contains quoted multi-claim content
- COMPOUND_PREDICATE: Multiple predicates ("User studied and worked in Paris")
- CAUSAL_CHAIN: Because/since linking claims ("User is tired because they worked late")
- TEMPORAL_SEQUENCE: Multiple events in sequence

### Test: Can part of this proposition be false while another part remains true?
If YES → Not atomic, needs decomposition"""

COVERAGE_CRITERIA = """## Coverage Evaluation

Coverage measures whether ALL extractable information from source messages is captured.

### Extract:
- Explicit statements: Direct claims made by the speaker
- Embedded facts: Facts within larger statements ("I walked my dog" → has dog, walked dog)
- Relational info: Relationships between entities
- Temporal info: When things happened or states began
- Quantitative info: Numbers, amounts, frequencies

### Gap Severity:
- CRITICAL: Core identity info missed (name, location, key relationships)
- IMPORTANT: Significant facts (job, major events, goals)
- MINOR: Supporting details (preferences, minor temporal info)"""

FIDELITY_CRITERIA = """## Fidelity Evaluation

Fidelity measures whether propositions faithfully represent source semantics.

### Violations:
- HEDGE_REMOVAL: "I might get a dog" → "User will get a dog"
- NEGATION_FLIP: "I don't like coffee" → "User likes coffee"
- TEMPORAL_SHIFT: "I used to work at Google" → "User works at Google"
- QUANTITY_CHANGE: "I sometimes go running" → "User runs regularly"
- ATTRIBUTION_ERROR: "My sister loves jazz" → "User loves jazz"
- OVERGENERALIZATION: "I enjoyed that restaurant" → "User enjoys Italian food"
- OVERSPECIFICATION: "I have a pet" → "User has a dog"
- INFERENCE_AS_EXPLICIT: Implied → stated as fact

### Severity: critical (changes meaning), major (significant), minor (slight imprecision)"""

UTILITY_CRITERIA = """## Downstream Utility Evaluation

Evaluates whether propositions can serve as valid logical premises.

### Good Premise Requirements:
- CLEAR SUBJECT: Unambiguous who/what ("They are excited" fails)
- CLEAR PREDICATE: Unambiguous claim
- CONTEXTUALLY COMPLETE: Standalone ("User is nervous" → about what?)
- STABLE TRUTH VALUE: Definitively T/F ("kind of likes" is fuzzy)
- COMPOSABLE: Can participate in syllogisms (no embedded complexity)
- TEMPORAL CLARITY: When states/events apply

### Suitability Ratings:
- EXCELLENT: Perfect premise, ideal for reasoning
- GOOD: Minor issues, usable
- MARGINAL: May cause ambiguity
- POOR: Significant issues
- UNUSABLE: Cannot serve as premise"""


# =============================================================================
# JUDGE IMPLEMENTATION
# =============================================================================

class ExplicitJudge:
    llm_client: AsyncAnthropic | AsyncOpenAI
    model: str
    verbose: bool
    provider: str

    def __init__(
        self,
        llm_client: AsyncAnthropic | AsyncOpenAI,
        model: str = "claude-sonnet-4-20250514",
        verbose: bool = False,
        provider: str = "anthropic",
    ):
        self.llm_client = llm_client
        self.model = model
        self.verbose = verbose
        self.provider = provider
        if verbose:
            logger.setLevel(logging.DEBUG)

    async def _call_llm(
        self,
        system: str,
        user: str,
        tool_def: dict[str, Any],
    ) -> dict[str, Any]:
        """Call LLM with tool use."""
        try:
            if isinstance(self.llm_client, AsyncAnthropic):
                resp = await asyncio.wait_for(
                    self.llm_client.messages.create(
                        model=self.model,
                        max_tokens=4000,
                        temperature=0.0,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                        tools=[tool_def],  # pyright: ignore[reportArgumentType]
                        tool_choice={"type": "tool", "name": tool_def["name"]},  # pyright: ignore[reportArgumentType]
                    ),
                    timeout=120.0,
                )
                for block in resp.content:
                    if block.type == "tool_use":
                        # block.input is typed as object, but we know it's a dict
                        return block.input if isinstance(block.input, dict) else {}  # type: ignore[return-value]
                return {}

            elif isinstance(self.llm_client, AsyncOpenAI):
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
                        max_tokens=4000,
                        temperature=0.0,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        tools=[openai_tool],  # pyright: ignore[reportArgumentType]
                        tool_choice={"type": "function", "function": {"name": tool_def["name"]}},  # pyright: ignore[reportArgumentType]
                    ),
                    timeout=120.0,
                )
                if resp.choices and resp.choices[0].message.tool_calls:
                    tool_call = resp.choices[0].message.tool_calls[0]
                    # Check if tool_call has function attribute
                    if hasattr(tool_call, 'function'):
                        func = tool_call.function  # pyright: ignore[reportAttributeAccessIssue]
                        if hasattr(func, 'arguments'):
                            return json.loads(func.arguments)
                return {}
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {}

    async def evaluate_atomicity(self, propositions: list[str]) -> AtomicityReport:
        if not propositions:
            return AtomicityReport(0, 0, 1.0)
        
        tool_def = {
            "name": "evaluate_atomicity",
            "description": "Submit atomicity evaluation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "evaluations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "is_atomic": {"type": "boolean"},
                                "violation_types": {
                                    "type": "array",
                                    "items": {"type": "string", "enum": [v.value for v in AtomicityViolation]}
                                },
                                "suggested_decomposition": {"type": "array", "items": {"type": "string"}},
                                "reasoning": {"type": "string"},
                            },
                            "required": ["index", "is_atomic"],
                        },
                    }
                },
                "required": ["evaluations"],
            },
        }
        
        props_text = "\n".join(f'{i+1}. "{p}"' for i, p in enumerate(propositions))
        result = await self._call_llm(
            ATOMICITY_CRITERIA,
            f"Evaluate atomicity:\n\n{props_text}",
            tool_def,
        )
        
        detailed: list[AtomicityResult] = []
        violations_by_type: dict[str, int] = {}
        atomic_count = 0
        total_suggested = 0
        
        for ev in result.get("evaluations", []):
            idx = ev.get("index", 0) - 1
            if idx < 0 or idx >= len(propositions):
                continue
            is_atomic = ev.get("is_atomic", True)
            if is_atomic:
                atomic_count += 1
            violations = []
            for v in ev.get("violation_types", []):
                try:
                    violations.append(AtomicityViolation(v))
                except ValueError as e:
                    logger.warning(f"Skipping invalid AtomicityViolation type: {v} - {e}")
                    continue
            for v in violations:
                violations_by_type[v.value] = violations_by_type.get(v.value, 0) + 1
            decomp = ev.get("suggested_decomposition", [])
            total_suggested += len(decomp)
            detailed.append(AtomicityResult(
                propositions[idx], is_atomic, violations, decomp, ev.get("reasoning", "")
            ))
        
        return AtomicityReport(
            len(propositions), atomic_count,
            atomic_count / len(propositions) if propositions else 1.0,
            violations_by_type, detailed,
            len(propositions) - atomic_count, atomic_count + total_suggested,
        )

    async def evaluate_coverage(
        self, propositions: list[str], messages: list[dict[str, Any]], peer_name: str
    ) -> CoverageReport:
        user_msgs = [m for m in messages if m.get("speaker", "user") == "user"]
        if not user_msgs:
            return CoverageReport(0, len(propositions), 1.0, source_message_count=0)
        
        msgs_text = "\n\n".join(f'[{i+1}] {m.get("text", "")}' for i, m in enumerate(user_msgs))
        props_text = "\n".join(f'{i+1}. "{p}"' for i, p in enumerate(propositions))
        
        tool_def = {
            "name": "evaluate_coverage",
            "description": "Submit coverage evaluation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "estimated_total_claims": {"type": "integer"},
                    "gaps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "missing_claim": {"type": "string"},
                                "source_quote": {"type": "string"},
                                "severity": {"type": "string", "enum": ["critical", "important", "minor"]},
                                "reasoning": {"type": "string"},
                            },
                            "required": ["missing_claim", "severity"],
                        },
                    },
                },
                "required": ["estimated_total_claims", "gaps"],
            },
        }
        
        result = await self._call_llm(
            COVERAGE_CRITERIA + f"\n\nPeer name: {peer_name}",
            f"SOURCE MESSAGES:\n{msgs_text}\n\nEXTRACTED:\n{props_text}\n\nIdentify gaps.",
            tool_def,
        )
        
        est_total = result.get("estimated_total_claims", len(propositions))
        gaps: list[CoverageGap] = []
        gaps_by_sev: dict[str, int] = {"critical": 0, "important": 0, "minor": 0}
        
        for g in result.get("gaps", []):
            sev = g.get("severity", "minor")
            gaps_by_sev[sev] = gaps_by_sev.get(sev, 0) + 1
            gaps.append(CoverageGap(
                g["missing_claim"], g.get("source_quote", ""), "0", sev, g.get("reasoning", "")
            ))
        
        sev_weights = {"critical": 1.0, "important": 0.5, "minor": 0.25}
        weighted_gaps = sum(sev_weights.get(g.severity, 0.25) for g in gaps)
        score = max(0, (est_total - weighted_gaps) / est_total) if est_total > 0 else 1.0
        
        return CoverageReport(
            est_total, len(propositions), score, gaps, gaps_by_sev,
            len(user_msgs), len(propositions) / len(user_msgs) if user_msgs else 0,
        )

    async def evaluate_fidelity(
        self, propositions: list[str], messages: list[dict[str, Any]]
    ) -> FidelityReport:
        if not propositions:
            return FidelityReport(0, 0, 1.0)
        
        user_msgs = [m for m in messages if m.get("speaker", "user") == "user"]
        msgs_text = "\n\n".join(f'[{i+1}] {m.get("text", "")}' for i, m in enumerate(user_msgs))
        props_text = "\n".join(f'{i+1}. "{p}"' for i, p in enumerate(propositions))
        
        tool_def = {
            "name": "evaluate_fidelity",
            "description": "Submit fidelity evaluation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "evaluations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "is_faithful": {"type": "boolean"},
                                "violations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": {"type": "string", "enum": [v.value for v in FidelityViolation]},
                                            "description": {"type": "string"},
                                        },
                                    },
                                },
                                "severity": {"type": "string", "enum": ["none", "minor", "major", "critical"]},
                            },
                            "required": ["index", "is_faithful"],
                        },
                    }
                },
                "required": ["evaluations"],
            },
        }
        
        result = await self._call_llm(
            FIDELITY_CRITERIA,
            f"SOURCE:\n{msgs_text}\n\nPROPOSITIONS:\n{props_text}",
            tool_def,
        )
        
        detailed: list[FidelityResult] = []
        violations_by_type: dict[str, int] = {}
        violations_by_sev: dict[str, int] = {"none": 0, "minor": 0, "major": 0, "critical": 0}
        faithful_count = 0
        
        for ev in result.get("evaluations", []):
            idx = ev.get("index", 0) - 1
            if idx < 0 or idx >= len(propositions):
                continue
            is_faithful = ev.get("is_faithful", True)
            if is_faithful:
                faithful_count += 1
            sev = ev.get("severity", "none")
            violations_by_sev[sev] = violations_by_sev.get(sev, 0) + 1
            violations: list[tuple[FidelityViolation, str]] = []
            for v in ev.get("violations", []):
                try:
                    vtype = FidelityViolation(v["type"])
                    violations_by_type[vtype.value] = violations_by_type.get(vtype.value, 0) + 1
                    violations.append((vtype, v.get("description", "")))
                except ValueError as e:
                    logger.warning(f"Skipping invalid FidelityViolation type: {v.get('type')} - {e}")
                    continue
            detailed.append(FidelityResult(propositions[idx], is_faithful, violations, "", sev))
        
        sev_penalties = {"none": 0, "minor": 0.25, "major": 0.5, "critical": 1.0}
        penalty = sum(sev_penalties.get(r.severity, 0) for r in detailed)
        score = max(0, 1 - penalty / len(propositions)) if propositions else 1.0
        
        return FidelityReport(
            len(propositions), faithful_count, score,
            violations_by_type, violations_by_sev, detailed,
        )

    async def evaluate_efficiency(self, propositions: list[str]) -> EfficiencyReport:
        if len(propositions) <= 1:
            return EfficiencyReport(len(propositions), len(propositions), 1.0)
        
        props_text = "\n".join(f'{i+1}. "{p}"' for i, p in enumerate(propositions))
        
        tool_def = {
            "name": "evaluate_efficiency",
            "description": "Submit redundancy analysis",
            "input_schema": {
                "type": "object",
                "properties": {
                    "redundancy_clusters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "proposition_indices": {"type": "array", "items": {"type": "integer"}},
                                "canonical_form": {"type": "string"},
                                "redundancy_type": {
                                    "type": "string",
                                    "enum": ["exact_duplicate", "near_duplicate", "subsumption", "overlap"],
                                },
                            },
                            "required": ["proposition_indices", "canonical_form", "redundancy_type"],
                        },
                    },
                    "unique_proposition_count": {"type": "integer"},
                },
                "required": ["redundancy_clusters", "unique_proposition_count"],
            },
        }
        
        result = await self._call_llm(
            "Identify redundant propositions (exact duplicates, near duplicates, subsumptions, overlaps).",
            f"PROPOSITIONS:\n{props_text}",
            tool_def,
        )
        
        clusters: list[RedundancyCluster] = []
        exact = near = subs = 0
        
        for c in result.get("redundancy_clusters", []):
            indices = c.get("proposition_indices", [])
            props = [propositions[i-1] for i in indices if 0 < i <= len(propositions)]
            rtype = c.get("redundancy_type", "overlap")
            count = len(props) - 1 if props else 0
            if rtype == "exact_duplicate":
                exact += count
            elif rtype == "near_duplicate":
                near += count
            elif rtype == "subsumption":
                subs += count
            clusters.append(RedundancyCluster(props, c.get("canonical_form", ""), rtype))
        
        unique = result.get("unique_proposition_count", len(propositions))
        
        return EfficiencyReport(
            len(propositions), unique,
            unique / len(propositions) if propositions else 1.0,
            clusters, exact, near, subs,
        )

    async def evaluate_utility(self, propositions: list[str], peer_name: str) -> DownstreamUtilityReport:
        if not propositions:
            return DownstreamUtilityReport(0, utility_score=1.0)
        
        props_text = "\n".join(f'{i+1}. "{p}"' for i, p in enumerate(propositions))
        
        tool_def = {
            "name": "evaluate_utility",
            "description": "Submit utility evaluation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "evaluations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "has_clear_subject": {"type": "boolean"},
                                "has_clear_predicate": {"type": "boolean"},
                                "is_contextually_complete": {"type": "boolean"},
                                "has_stable_truth_value": {"type": "boolean"},
                                "is_composable": {"type": "boolean"},
                                "temporal_handling": {"type": "string", "enum": ["appropriate", "missing", "excessive"]},
                                "suitability": {"type": "string", "enum": [s.value for s in PremiseSuitability]},
                                "issues": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["index", "suitability"],
                        },
                    }
                },
                "required": ["evaluations"],
            },
        }
        
        result = await self._call_llm(
            UTILITY_CRITERIA + f"\n\nPeer: {peer_name}",
            f"PROPOSITIONS:\n{props_text}\n\nEvaluate as logical premises.",
            tool_def,
        )
        
        detailed: list[DownstreamUtilityResult] = []
        dist: dict[str, int] = {s.value: 0 for s in PremiseSuitability}
        clarity = complete = stable = compos = temporal = 0
        
        for ev in result.get("evaluations", []):
            idx = ev.get("index", 0) - 1
            if idx < 0 or idx >= len(propositions):
                continue
            suit = ev.get("suitability", "marginal")
            dist[suit] = dist.get(suit, 0) + 1
            
            subj = ev.get("has_clear_subject", True)
            pred = ev.get("has_clear_predicate", True)
            comp = ev.get("is_contextually_complete", True)
            stab = ev.get("has_stable_truth_value", True)
            comb = ev.get("is_composable", True)
            temp = ev.get("temporal_handling", "appropriate")
            
            if subj and pred:
                clarity += 1
            if comp:
                complete += 1
            if stab:
                stable += 1
            if comb:
                compos += 1
            if temp == "appropriate":
                temporal += 1
            
            detailed.append(DownstreamUtilityResult(
                propositions[idx], PremiseSuitability(suit),
                ev.get("issues", []), subj, pred, comp, stab, comb, temp,
            ))
        
        n = len(propositions)
        weights = {"excellent": 1.0, "good": 0.8, "marginal": 0.5, "poor": 0.2, "unusable": 0.0}
        score = sum(dist[s] * weights[s] for s in dist) / n if n else 1.0
        
        return DownstreamUtilityReport(
            n, dist,
            clarity / n if n else 1.0,
            complete / n if n else 1.0,
            stable / n if n else 1.0,
            compos / n if n else 1.0,
            temporal / n if n else 1.0,
            score, detailed,
        )

    async def evaluate(
        self,
        propositions: list[str],
        messages: list[dict[str, Any]],
        peer_name: str,
        conversation_id: str = "",
        weights: dict[str, float] | None = None,
    ) -> EvaluationResult:
        logger.info(f"Evaluating {len(propositions)} propositions...")
        
        atom, cov, fid, eff, util = await asyncio.gather(
            self.evaluate_atomicity(propositions),
            self.evaluate_coverage(propositions, messages, peer_name),
            self.evaluate_fidelity(propositions, messages),
            self.evaluate_efficiency(propositions),
            self.evaluate_utility(propositions, peer_name),
        )
        
        user_msgs = [m for m in messages if m.get("speaker", "user") == "user"]
        
        result = EvaluationResult(
            conversation_id, peer_name, len(propositions), len(user_msgs),
            atom, cov, fid, eff, util,
        )
        result.compute_overall_score(weights)
        
        logger.info(f"Evaluation complete. Overall: {result.overall_score:.2%}")
        return result


# =============================================================================
# TRACE PARSING
# =============================================================================

def load_traces_from_json(path: Path) -> list[dict[str, Any]]:
    """Load traces from a JSON or JSONL file.

    Supports:
    - JSON array: [{"trace": 1}, {"trace": 2}]
    - JSON object: {"trace": 1}
    - JSONL: One JSON object per line
    """
    traces = []

    # Try JSONL format first (one JSON per line)
    try:
        with open(path) as f:
            first_line = f.readline().strip()
            if first_line and not first_line.startswith('['):
                # Likely JSONL format
                f.seek(0)  # Reset to beginning
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        trace = json.loads(line)
                        traces.append(trace)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at line {line_num} in {path}: {e}")

                if traces:
                    logger.info(f"Loaded {len(traces)} traces from JSONL file: {path}")
                    return traces
    except Exception as e:
        logger.debug(f"Not JSONL format, trying standard JSON: {e}")

    # Try standard JSON format
    try:
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Single trace, wrap in list
            return [data]
        else:
            raise ValueError(f"Unexpected JSON format in {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse {path} as JSON or JSONL: {e}") from e


def extract_propositions(trace: dict[str, Any]) -> list[str]:
    explicit = trace.get("output", {}).get("content", {}).get("explicit", [])
    return [item["content"] for item in explicit if "content" in item]


def extract_messages(trace: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = trace.get("input", {}).get("prompt", "")
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


def extract_conversation_id(trace: dict[str, Any], index: int) -> str:
    """Extract or generate a conversation ID for a trace."""
    # Try common ID fields
    if "conversation_id" in trace:
        return trace["conversation_id"]
    if "id" in trace:
        return trace["id"]
    # Generate from index
    return f"trace_{index:04d}"


def extract_peer_name(trace: dict[str, Any]) -> str:
    """Extract peer name from trace propositions."""
    for prop in extract_propositions(trace):
        prop_lower = prop.lower()
        # Pattern: "user's name is Victor" or "User is named Victor"
        if "name is" in prop_lower:
            parts = prop_lower.split("name is")
            if len(parts) > 1:
                name = parts[1].strip().rstrip(".").split()[0]
                return name.capitalize()
        if "is named" in prop_lower:
            parts = prop_lower.split("is named")
            if len(parts) > 1:
                name = parts[1].strip().rstrip(".").split()[0]
                return name.capitalize()
    return "user"


# =============================================================================
# OUTPUT
# =============================================================================

def print_summary(result: EvaluationResult) -> None:
    print("\n" + "=" * 70)
    print(f"EVALUATION: {result.conversation_id}")
    print("=" * 70)
    print(f"Peer: {result.peer_name} | Props: {result.proposition_count} | Messages: {result.source_message_count}")
    
    print(f"\n{'OVERALL SCORE:':<20} {result.overall_score:.1%}")
    print("-" * 40)
    print(f"{'Atomicity:':<20} {result.atomicity.score:.1%} ({result.atomicity.atomic_count}/{result.atomicity.total_propositions} atomic)")
    print(f"{'Coverage:':<20} {result.coverage.coverage_score:.1%} ({len(result.coverage.gaps)} gaps)")
    print(f"{'Fidelity:':<20} {result.fidelity.fidelity_score:.1%} ({result.fidelity.faithful_count}/{result.fidelity.total_propositions} faithful)")
    print(f"{'Efficiency:':<20} {result.efficiency.efficiency_score:.1%} ({result.efficiency.unique_propositions}/{result.efficiency.total_propositions} unique)")
    print(f"{'Utility:':<20} {result.downstream_utility.utility_score:.1%}")
    
    # Show issues
    if result.coverage.gaps:
        crit = [g for g in result.coverage.gaps if g.severity == "critical"]
        if crit:
            print(f"\n⚠️  Critical coverage gaps ({len(crit)}):")
            for g in crit[:3]:
                print(f"   - {g.missing_claim[:60]}...")
    
    non_atomic = [r for r in result.atomicity.detailed_results if not r.is_atomic]
    if non_atomic:
        print(f"\n⚠️  Non-atomic propositions ({len(non_atomic)}):")
        for r in non_atomic[:2]:
            print(f"   - \"{r.proposition[:50]}...\"")
            if r.suggested_decomposition:
                print(f"     → Split into: {r.suggested_decomposition[:2]}")
    
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Run explicit derivation benchmark")
    parser.add_argument("--traces", type=Path, help="JSON or JSONL file containing traces")
    parser.add_argument("--trace-dir", type=Path, help="Directory of JSON/JSONL trace files")
    parser.add_argument("--output-dir", type=Path, default=Path("tests/bench/eval_results"))
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "openrouter"],
        default="anthropic",
        help="LLM provider to use (default: anthropic)",
    )
    parser.add_argument("--api-key", type=str, help="API key (overrides environment variable)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--weights", type=str, help="JSON string of custom weights")
    parser.add_argument("--limit", type=int, help="Only evaluate first N traces")
    parser.add_argument("--sample", type=float, help="Randomly sample this fraction of traces (0.0-1.0)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of traces to process concurrently (default: 1)",
    )
    args = parser.parse_args()

    # Get API key from argument or environment
    api_key = args.api_key
    if not api_key:
        if args.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_ANTHROPIC_API_KEY")
            if not api_key:
                logger.error("Set ANTHROPIC_API_KEY or LLM_ANTHROPIC_API_KEY, or use --api-key")
                return 1
        elif args.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("Set OPENAI_API_KEY or use --api-key")
                return 1
        elif args.provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                logger.error("Set OPENROUTER_API_KEY or use --api-key")
                return 1

    # Initialize client based on provider
    if args.provider == "anthropic":
        client = AsyncAnthropic(api_key=api_key)
    elif args.provider == "openai":
        client = AsyncOpenAI(api_key=api_key)
    elif args.provider == "openrouter":
        # OpenRouter uses OpenAI-compatible API
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        logger.error(f"Unsupported provider: {args.provider}")
        return 1

    judge = ExplicitJudge(client, args.model, args.verbose, args.provider)
    
    weights = json.loads(args.weights) if args.weights else None
    
    # Collect all traces
    all_traces: list[tuple[dict[str, Any], str]] = []  # (trace, source_file)
    
    if args.traces:
        traces = load_traces_from_json(args.traces)
        all_traces.extend((t, args.traces.name) for t in traces)
    elif args.trace_dir:
        # Support both .json and .jsonl extensions
        for json_file in list(args.trace_dir.glob("*.json")) + list(args.trace_dir.glob("*.jsonl")):
            traces = load_traces_from_json(json_file)
            all_traces.extend((t, json_file.name) for t in traces)
    else:
        parser.error("Specify --traces or --trace-dir")
    
    print(f"Loaded {len(all_traces)} trace(s)")

    # Apply sampling/limiting
    if args.sample and 0 < args.sample < 1:
        sample_size = max(1, int(len(all_traces) * args.sample))
        all_traces = random.sample(all_traces, sample_size)
        print(f"Sampled {len(all_traces)} traces ({args.sample:.0%})")
    
    if args.limit and args.limit < len(all_traces):
        all_traces = all_traces[:args.limit]
        print(f"Limited to first {args.limit} traces")
    
    print(f"\nEvaluating {len(all_traces)} trace(s)...\n")

    # Process traces in batches if batch_size > 1
    results: list[EvaluationResult] = []

    async def process_trace(idx: int, trace: dict[str, Any], source_file: str) -> EvaluationResult | None:
        """Process a single trace and return the result."""
        try:
            props = extract_propositions(trace)
            if not props:
                logger.warning(f"Trace {idx} from {source_file} has no propositions, skipping")
                return None

            msgs = extract_messages(trace)
            peer = extract_peer_name(trace)
            conv_id = extract_conversation_id(trace, idx)

            print(f"[{idx+1}/{len(all_traces)}] Evaluating {conv_id} ({len(props)} props)...")

            result = await judge.evaluate(props, msgs, peer, conv_id, weights)
            print_summary(result)
            return result

        except Exception as e:
            logger.error(f"Failed trace {idx} from {source_file}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return None

    # Process in batches
    if args.batch_size > 1:
        print(f"Processing traces in batches of {args.batch_size}...\n")
        for i in range(0, len(all_traces), args.batch_size):
            batch = all_traces[i:i + args.batch_size]
            batch_results = await asyncio.gather(
                *[process_trace(i + j, trace, source_file) for j, (trace, source_file) in enumerate(batch)],
                return_exceptions=True
            )
            for result in batch_results:
                if isinstance(result, EvaluationResult):
                    results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
    else:
        # Process sequentially
        for idx, (trace, source_file) in enumerate(all_traces):
            result = await process_trace(idx, trace, source_file)
            if result:
                results.append(result)
    
    if results:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_file = args.output_dir / f"explicit_{ts}.json"

        averages: dict[str, float] = {
            "overall": sum(r.overall_score for r in results) / len(results),
            "atomicity": sum(r.atomicity.score for r in results) / len(results),
            "coverage": sum(r.coverage.coverage_score for r in results) / len(results),
            "fidelity": sum(r.fidelity.fidelity_score for r in results) / len(results),
            "efficiency": sum(r.efficiency.efficiency_score for r in results) / len(results),
            "utility": sum(r.downstream_utility.utility_score for r in results) / len(results),
        }

        agg = {
            "timestamp": ts,
            "model": args.model,
            "count": len(results),
            "averages": averages,
            "results": [r.to_dict() for r in results],
        }
        
        with open(out_file, "w") as f:
            json.dump(agg, f, indent=2)
        
        print(f"\n✅ Results saved to {out_file}")
        
        if len(results) > 1:
            print("\n" + "=" * 70)
            print("AGGREGATE RESULTS")
            print("=" * 70)
            print(f"Traces evaluated: {len(results)}")
            print()
            for k, v in averages.items():
                print(f"  {k:<20} {v:.1%}")
    else:
        print("\n⚠️  No traces were successfully evaluated")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))