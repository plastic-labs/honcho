"""Metrics tracking for ARC-AGI-2 solver performance."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SolverMetrics:
    """Metrics for tracking solver performance."""

    # Task information
    task_id: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # Success metrics
    solved: bool = False
    num_iterations: int = 0
    time_to_solution: Optional[float] = None

    # Memory metrics
    num_memory_queries: int = 0
    num_similar_tasks_retrieved: int = 0
    num_primitives_retrieved: int = 0
    memory_query_time_ms: List[float] = field(default_factory=list)

    # Reasoning metrics
    num_reasoning_steps: int = 0
    num_hypotheses_generated: int = 0
    num_verifications: int = 0
    num_failed_verifications: int = 0

    # Token usage and API cost tracking
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_name: str = ""  # Track which model is being used
    api_cost: float = 0.0  # Calculated cost in USD

    # LLM API calls
    num_llm_calls: int = 0
    llm_call_times_ms: List[float] = field(default_factory=list)

    # Honcho memory statistics
    num_facts_stored: int = 0  # Total facts/observations stored across all peers
    num_sessions_created: int = 0  # Sessions created during this task
    num_messages_ingested: int = 0  # Messages added to Honcho
    facts_per_peer: Dict[str, int] = field(default_factory=dict)  # {peer_name: fact_count}

    # Transformation attempts
    transformation_attempts: Dict[str, int] = field(default_factory=dict)

    def mark_complete(self, solved: bool = False):
        """Mark the solving attempt as complete."""
        self.end_time = time.time()
        self.solved = solved
        if solved:
            self.time_to_solution = self.end_time - self.start_time

    def add_memory_query(self, query_time_ms: float, num_results: int = 0):
        """Record a memory query."""
        self.num_memory_queries += 1
        self.memory_query_time_ms.append(query_time_ms)
        self.num_similar_tasks_retrieved += num_results

    def add_llm_call(self, call_time_ms: float, tokens_used: Optional[Dict[str, int]] = None):
        """Record an LLM API call."""
        self.num_llm_calls += 1
        self.llm_call_times_ms.append(call_time_ms)

        if tokens_used:
            self.prompt_tokens += tokens_used.get("prompt_tokens", 0)
            self.completion_tokens += tokens_used.get("completion_tokens", 0)
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def add_transformation_attempt(self, primitive_name: str):
        """Record a transformation attempt."""
        self.transformation_attempts[primitive_name] = (
            self.transformation_attempts.get(primitive_name, 0) + 1
        )

    def get_avg_memory_query_time(self) -> float:
        """Get average memory query time in milliseconds."""
        if not self.memory_query_time_ms:
            return 0.0
        return sum(self.memory_query_time_ms) / len(self.memory_query_time_ms)

    def get_avg_llm_call_time(self) -> float:
        """Get average LLM call time in milliseconds."""
        if not self.llm_call_times_ms:
            return 0.0
        return sum(self.llm_call_times_ms) / len(self.llm_call_times_ms)

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def get_verification_success_rate(self) -> float:
        """Get the percentage of successful verifications."""
        if self.num_verifications == 0:
            return 0.0
        successful = self.num_verifications - self.num_failed_verifications
        return (successful / self.num_verifications) * 100

    def calculate_api_cost(self) -> float:
        """Calculate total API cost based on token usage and model pricing."""
        from .config import MODEL_PRICING

        if not self.model_name:
            return 0.0

        pricing = MODEL_PRICING.get(self.model_name, {"input": 0, "output": 0})
        input_cost = (self.prompt_tokens / 1000) * pricing["input"]
        output_cost = (self.completion_tokens / 1000) * pricing["output"]
        self.api_cost = input_cost + output_cost
        return self.api_cost

    def get_cost_per_token(self) -> float:
        """Get cost per token."""
        if self.total_tokens == 0:
            return 0.0
        return self.api_cost / self.total_tokens

    async def update_peer_facts(self, peer_name: str, peer):
        """Query peer context to count facts/observations."""
        try:
            context = await peer.get_context(search_query="", search_top_k=1000)
            if context and hasattr(context, 'representation') and context.representation:
                if hasattr(context.representation, 'observations') and context.representation.observations:
                    fact_count = len(context.representation.observations)
                    self.facts_per_peer[peer_name] = fact_count
                    self.num_facts_stored = sum(self.facts_per_peer.values())
        except Exception as e:
            import logging
            logging.warning(f"Could not fetch facts for peer {peer_name}: {e}")

    def add_peer_fact(self, peer_name: str):
        """Increment fact count for a specific peer (when storing messages)."""
        self.facts_per_peer[peer_name] = self.facts_per_peer.get(peer_name, 0) + 1
        self.num_facts_stored = sum(self.facts_per_peer.values())

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for display/export."""
        return {
            "task_id": self.task_id,
            "solved": self.solved,
            "elapsed_time_s": self.get_elapsed_time(),
            "time_to_solution_s": self.time_to_solution,
            "num_iterations": self.num_iterations,
            "num_reasoning_steps": self.num_reasoning_steps,
            "num_hypotheses": self.num_hypotheses_generated,
            "num_verifications": self.num_verifications,
            "verification_success_rate": f"{self.get_verification_success_rate():.1f}%",
            "num_memory_queries": self.num_memory_queries,
            "avg_memory_query_ms": f"{self.get_avg_memory_query_time():.2f}",
            "num_llm_calls": self.num_llm_calls,
            "avg_llm_call_ms": f"{self.get_avg_llm_call_time():.2f}",
            "total_tokens": self.total_tokens,
            "api_cost_usd": f"${self.api_cost:.4f}",
            "cost_per_token": f"${self.get_cost_per_token():.6f}",
            "num_sessions": self.num_sessions_created,
            "num_messages": self.num_messages_ingested,
            "num_facts": self.num_facts_stored,
            "facts_per_peer": self.facts_per_peer,
            "transformation_attempts": self.transformation_attempts,
        }


class MetricsAggregator:
    """Aggregates metrics across multiple tasks."""

    def __init__(self):
        self.task_metrics: List[SolverMetrics] = []

    def add_task_metrics(self, metrics: SolverMetrics):
        """Add metrics for a completed task."""
        self.task_metrics.append(metrics)

    def get_overall_accuracy(self) -> float:
        """Get overall solving accuracy."""
        if not self.task_metrics:
            return 0.0
        solved = sum(1 for m in self.task_metrics if m.solved)
        return (solved / len(self.task_metrics)) * 100

    def get_avg_time_to_solution(self) -> float:
        """Get average time to solution for solved tasks."""
        solved_times = [
            m.time_to_solution
            for m in self.task_metrics
            if m.solved and m.time_to_solution
        ]
        if not solved_times:
            return 0.0
        return sum(solved_times) / len(solved_times)

    def get_total_tokens_used(self) -> int:
        """Get total tokens used across all tasks."""
        return sum(m.total_tokens for m in self.task_metrics)

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "total_tasks": len(self.task_metrics),
            "solved_tasks": sum(1 for m in self.task_metrics if m.solved),
            "accuracy": f"{self.get_overall_accuracy():.1f}%",
            "avg_time_to_solution_s": f"{self.get_avg_time_to_solution():.2f}",
            "total_tokens_used": self.get_total_tokens_used(),
            "avg_tokens_per_task": (
                self.get_total_tokens_used() / len(self.task_metrics)
                if self.task_metrics
                else 0
            ),
        }
