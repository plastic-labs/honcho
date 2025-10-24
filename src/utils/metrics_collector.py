"""
Global metrics collector for aggregating performance metrics across benchmark runs.

This module provides functionality to collect, aggregate, and export performance
metrics from deriver and dialectic operations during benchmarking.
"""

import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from typing_extensions import TypedDict

from src.config import settings
from src.utils.representation import Representation

logger = logging.getLogger(__name__)


def save_explicit_trace(
    provider: str,
    model: str,
    max_tokens: int,
    peer_id: str,
    peer_card: list[str] | None,
    message_created_at: datetime,
    working_representation: Representation,
    history: str,
    new_turns: list[str],
    prompt: str,
    response: str,
    thinking: str | None,
    message_metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save the critical analysis call inputs and prompt to trace.jsonl file.
    Uses JSONL format (one JSON object per line) for efficient appending.
    Use convert_trace_to_json() to convert to a proper JSON array format.

    The trace file path can be specified in message_metadata['trace_file_path'].
    If not provided, falls back to environment variable or default setting.
    """
    import os

    from src.config import settings

    # Priority order for trace file path:
    # 1. Message metadata (set by tracer script) - allows per-message trace routing
    # 2. Environment variable (set by tracer process)
    # 3. Settings default (trace.jsonl)
    if message_metadata and "trace_file_path" in message_metadata:
        trace_file_path = message_metadata["trace_file_path"]
    else:
        trace_file_path = os.environ.get("LOCAL_TRACE_FILE", settings.LOCAL_TRACE_FILE)

    trace_file = Path(trace_file_path)

    # Parse response to extract explicit and implicit facts
    response_parsed = None
    try:
        response_data = json.loads(response)
        explicit_facts = response_data.get("explicit", [])
        implicit_facts = response_data.get("implicit", [])

        # Handle both old format (list of objects) and new format (list of strings)
        if explicit_facts and isinstance(explicit_facts[0], dict):
            # Old format: list of {content: "..."}
            explicit_facts = [item.get("content", "") for item in explicit_facts]
        if implicit_facts and isinstance(implicit_facts[0], dict):
            # Old format: list of {conclusion: "...", premises: [...]}
            implicit_facts = [item.get("conclusion", "") for item in implicit_facts]

        response_parsed = {
            "explicit": explicit_facts,
            "implicit": implicit_facts,
        }
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        # If parsing fails, keep response_parsed as None
        pass

    trace_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_type": "explicit",
        "provider": provider,
        "model": model,
        "max_tokens": max_tokens,
        "peer_id": peer_id,
        "peer_card": peer_card,
        "message_created_at": message_created_at.isoformat(),
        "dataset_uuid": message_metadata.get("dataset_uuid") if message_metadata else None,
        "conversation_id": message_metadata.get("conversation_id") if message_metadata else None,
        "message_sequence_id": message_metadata.get("message_sequence_id") if message_metadata else None,
        "total_messages": message_metadata.get("total_messages") if message_metadata else None,
        "dataset": message_metadata.get("dataset") if message_metadata else None,
        "working_representation_input": {
            "explicit": [
                obs.content
                for obs in working_representation.explicit
            ],
            "implicit": [
                obs.content
                for obs in working_representation.implicit
            ],
            "deductive": [
                obs.conclusion
                for obs in working_representation.deductive
            ],
        },
        "history": history,
        "new_turns": new_turns,
        "prompt": prompt,
        "llm_output": response_parsed,
        "llm_output_raw": response,
        "thinking": thinking,
    }

    try:
        with open(trace_file, "a") as f:
            f.write(json.dumps(trace_data) + "\n")
        logger.debug("Saved explicit trace to %s", trace_file)
    except Exception as e:
        logger.warning("Failed to save trace to file: %s", e)


def save_deductive_trace(
    provider: str,
    model: str,
    max_tokens: int,
    peer_id: str,
    peer_card: list[str] | None,
    message_created_at: datetime,
    existing_deductions: list[str],
    atomic_propositions: list[str],
    history: str,
    new_turns: list[str],
    prompt: str,
    response: str,
    thinking: str | None,
    message_metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save the deductive reasoning call inputs and prompt to trace.jsonl file.
    Uses JSONL format (one JSON object per line) for efficient appending.

    The trace file path can be specified in message_metadata['trace_file_path'].
    If not provided, falls back to environment variable or default setting.
    """
    import os

    from src.config import settings

    # Priority order for trace file path:
    # 1. Message metadata (set by tracer script) - allows per-message trace routing
    # 2. Environment variable (set by tracer process)
    # 3. Settings default (trace.jsonl)
    if message_metadata and "trace_file_path" in message_metadata:
        trace_file_path = message_metadata["trace_file_path"]
    else:
        trace_file_path = os.environ.get("LOCAL_TRACE_FILE", settings.LOCAL_TRACE_FILE)

    trace_file = Path(trace_file_path)

    # Parse response to extract deductions
    response_parsed = None
    try:
        response_data = json.loads(response)
        deductions = response_data.get("deductions", [])
        response_parsed = {
            "deductions": deductions,
        }
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        # If parsing fails, keep response_parsed as None
        pass

    trace_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_type": "deductive",
        "provider": provider,
        "model": model,
        "max_tokens": max_tokens,
        "peer_id": peer_id,
        "peer_card": peer_card,
        "message_created_at": message_created_at.isoformat(),
        "dataset_uuid": message_metadata.get("dataset_uuid") if message_metadata else None,
        "conversation_id": message_metadata.get("conversation_id") if message_metadata else None,
        "message_sequence_id": message_metadata.get("message_sequence_id") if message_metadata else None,
        "total_messages": message_metadata.get("total_messages") if message_metadata else None,
        "dataset": message_metadata.get("dataset") if message_metadata else None,
        "deductive_reasoning_input": {
            "existing_deductions": existing_deductions,
            "atomic_propositions": atomic_propositions,
        },
        "history": history,
        "new_turns": new_turns,
        "prompt": prompt,
        "llm_output": response_parsed,
        "llm_output_raw": response,
        "thinking": thinking,
    }

    try:
        with open(trace_file, "a") as f:
            f.write(json.dumps(trace_data) + "\n")
        logger.debug("Saved deductive trace to %s", trace_file)
    except Exception as e:
        logger.warning("Failed to save trace to file: %s", e)


def save_inductive_trace(
    provider: str,
    model: str,
    max_tokens: int,
    peer_id: str,
    peer_card: list[str] | None,
    message_created_at: datetime,
    existing_inductions: list[str],
    explicit_conclusions: list[str],
    deductive_conclusions: list[str],
    history: str,
    new_turns: list[str],
    prompt: str,
    response: str,
    thinking: str | None,
    message_metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save the inductive reasoning call inputs and prompt to trace.jsonl file.
    Uses JSONL format (one JSON object per line) for efficient appending.

    The trace file path can be specified in message_metadata['trace_file_path'].
    If not provided, falls back to environment variable or default setting.
    """
    import os

    from src.config import settings

    # Priority order for trace file path:
    # 1. Message metadata (set by tracer script) - allows per-message trace routing
    # 2. Environment variable (set by tracer process)
    # 3. Settings default (trace.jsonl)
    if message_metadata and "trace_file_path" in message_metadata:
        trace_file_path = message_metadata["trace_file_path"]
    else:
        trace_file_path = os.environ.get("LOCAL_TRACE_FILE", settings.LOCAL_TRACE_FILE)

    trace_file = Path(trace_file_path)

    # Parse response to extract inductions
    response_parsed = None
    try:
        response_data = json.loads(response)
        inductions = response_data.get("inductions", [])
        response_parsed = {
            "inductions": inductions,
        }
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        # If parsing fails, keep response_parsed as None
        pass

    trace_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_type": "inductive",
        "provider": provider,
        "model": model,
        "max_tokens": max_tokens,
        "peer_id": peer_id,
        "peer_card": peer_card,
        "message_created_at": message_created_at.isoformat(),
        "dataset_uuid": message_metadata.get("dataset_uuid") if message_metadata else None,
        "conversation_id": message_metadata.get("conversation_id") if message_metadata else None,
        "message_sequence_id": message_metadata.get("message_sequence_id") if message_metadata else None,
        "total_messages": message_metadata.get("total_messages") if message_metadata else None,
        "dataset": message_metadata.get("dataset") if message_metadata else None,
        "inductive_reasoning_input": {
            "existing_inductions": existing_inductions,
            "explicit_conclusions": explicit_conclusions,
            "deductive_conclusions": deductive_conclusions,
        },
        "history": history,
        "new_turns": new_turns,
        "prompt": prompt,
        "llm_output": response_parsed,
        "llm_output_raw": response,
        "thinking": thinking,
    }

    try:
        with open(trace_file, "a") as f:
            f.write(json.dumps(trace_data) + "\n")
        logger.debug("Saved inductive trace to %s", trace_file)
    except Exception as e:
        logger.warning("Failed to save trace to file: %s", e)


def save_abductive_trace(
    provider: str,
    model: str,
    max_tokens: int,
    peer_id: str,
    peer_card: list[str] | None,
    message_created_at: datetime,
    existing_abductions: list[str],
    explicit_conclusions: list[str],
    deductive_conclusions: list[str],
    inductive_conclusions: list[str],
    history: str,
    new_turns: list[str],
    prompt: str,
    response: str,
    thinking: str | None,
    message_metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save the abductive reasoning call inputs and prompt to trace.jsonl file.
    Uses JSONL format (one JSON object per line) for efficient appending.

    The trace file path can be specified in message_metadata['trace_file_path'].
    If not provided, falls back to environment variable or default setting.
    """
    import os

    from src.config import settings

    # Priority order for trace file path:
    # 1. Message metadata (set by tracer script) - allows per-message trace routing
    # 2. Environment variable (set by tracer process)
    # 3. Settings default (trace.jsonl)
    if message_metadata and "trace_file_path" in message_metadata:
        trace_file_path = message_metadata["trace_file_path"]
    else:
        trace_file_path = os.environ.get("LOCAL_TRACE_FILE", settings.LOCAL_TRACE_FILE)

    trace_file = Path(trace_file_path)

    # Parse response to extract abductions
    response_parsed = None
    try:
        response_data = json.loads(response)
        abductions = response_data.get("abductions", [])
        response_parsed = {
            "abductions": abductions,
        }
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        # If parsing fails, keep response_parsed as None
        pass

    trace_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_type": "abductive",
        "provider": provider,
        "model": model,
        "max_tokens": max_tokens,
        "peer_id": peer_id,
        "peer_card": peer_card,
        "message_created_at": message_created_at.isoformat(),
        "dataset_uuid": message_metadata.get("dataset_uuid") if message_metadata else None,
        "conversation_id": message_metadata.get("conversation_id") if message_metadata else None,
        "message_sequence_id": message_metadata.get("message_sequence_id") if message_metadata else None,
        "total_messages": message_metadata.get("total_messages") if message_metadata else None,
        "dataset": message_metadata.get("dataset") if message_metadata else None,
        "abductive_reasoning_input": {
            "existing_abductions": existing_abductions,
            "explicit_conclusions": explicit_conclusions,
            "deductive_conclusions": deductive_conclusions,
            "inductive_conclusions": inductive_conclusions,
        },
        "history": history,
        "new_turns": new_turns,
        "prompt": prompt,
        "llm_output": response_parsed,
        "llm_output_raw": response,
        "thinking": thinking,
    }

    try:
        with open(trace_file, "a") as f:
            f.write(json.dumps(trace_data) + "\n")
        logger.debug("Saved abductive trace to %s", trace_file)
    except Exception as e:
        logger.warning("Failed to save trace to file: %s", e)


def convert_trace_to_json(
    input_file: str = "trace.jsonl", output_file: str = "trace.json"
) -> None:
    """
    Convert trace.jsonl to a proper JSON array format in trace.json.
    This should be run after data collection is complete.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        logger.warning("Trace file %s does not exist", input_file)
        return

    try:
        traces: list[dict[str, Any]] = []
        with open(input_path) as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line))

        with open(output_path, "w") as f:
            json.dump(traces, f, indent=2)

        logger.info(
            "Converted %d traces from %s to %s", len(traces), input_file, output_file
        )
        # delete the input file
        input_path.unlink()
    except Exception as e:
        logger.error("Failed to convert trace file: %s", e)


class MetricStats(TypedDict):
    """Statistics for a single metric type."""

    count: int
    mean: float
    median: float
    min: float
    max: float
    std_dev: float
    unit: str
    raw_values: list[float]


class MetricsExport(TypedDict):
    """Complete metrics export structure."""

    run_id: str
    start_time: str
    end_time: str
    total_tasks: int
    metrics_by_type: dict[str, list[float]]
    aggregated_stats: dict[str, MetricStats]


class MetricsCollector:
    """
    Collects and aggregates performance metrics across multiple deriver runs.

    This collector is designed to work alongside the existing logging system,
    capturing metrics from individual tasks and providing aggregated statistics
    for benchmarking analysis.
    """

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self.run_id: str | None = None
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.metrics_by_type: dict[str, list[float]] = {}
        self.task_count: int = 0
        self.is_collecting: bool = False

    def start_collection(self, run_id: str) -> None:
        """
        Initialize metrics collection for a new benchmark run.

        Args:
            run_id: Unique identifier for this benchmark run
        """
        self.run_id = run_id
        self.start_time = datetime.now()
        self.end_time = None
        self.metrics_by_type.clear()
        self.task_count = 0
        self.is_collecting = True
        print(f"📊 Started metrics collection for run: {run_id}")

    def collect_metrics(
        self, metrics_list: list[tuple[str, str | int | float, str]]
    ) -> None:
        """
        Collect metrics from a completed task.

        Args:
            metrics_list: List of (metric_name, value, unit) tuples
        """
        if not self.is_collecting:
            return

        self.task_count += 1

        for metric_name, value, unit in metrics_list:
            # Normalize metric names to be consistent
            normalized_name = metric_name.lower().replace(" ", "_")

            # skip metrics whose unit is "id" (more in future possibly)
            if unit in ["id"]:
                continue

            # Convert value to float for aggregation
            try:
                numeric_value = float(value)
            except (ValueError, TypeError):
                continue  # Skip non-numeric metrics

            # Store the metric with its unit
            metric_key = f"{normalized_name}_{unit}"

            if metric_key not in self.metrics_by_type:
                self.metrics_by_type[metric_key] = []

            self.metrics_by_type[metric_key].append(numeric_value)

    def load_from_file(self, filepath: Path) -> None:
        """
        Load metrics from a file into this collector.
        Creates the file if it doesn't exist.

        Args:
            filepath: Path to the metrics file to load
        """
        if not filepath.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.touch()
            with open(filepath, "w") as f:
                f.write("")

        file_metrics = load_metrics_from_file(filepath)
        for _task_name, metrics_list in file_metrics:
            self.collect_metrics(metrics_list)

    def finalize_collection(self) -> None:
        """
        Finalize metrics collection and calculate aggregated statistics.
        """
        if not self.is_collecting:
            return

        # Load any metrics from the file before finalizing
        metrics_file = get_metrics_file_path()
        if metrics_file:
            self.load_from_file(metrics_file)

        self.end_time = datetime.now()
        self.is_collecting = False

        duration = (
            (self.end_time - self.start_time).total_seconds()
            if self.end_time and self.start_time
            else 0
        )
        print(f"📊 Finalized metrics collection for run: {self.run_id}")
        print(f"   Tasks processed: {self.task_count}")
        print(f"   Collection duration: {duration:.2f}s")
        print(f"   Metric types collected: {len(self.metrics_by_type)}")

    def get_aggregated_stats(self) -> dict[str, MetricStats]:
        """
        Calculate aggregated statistics for all collected metrics.

        Returns:
            Dictionary mapping metric names to their statistics
        """
        stats: dict[str, MetricStats] = {}

        for metric_key, values in self.metrics_by_type.items():
            if not values:
                continue

            # Extract unit from metric key (format: metric_name_unit)
            parts = metric_key.rsplit("_", 1)
            if len(parts) == 2:
                metric_name, unit = parts
            else:
                metric_name, unit = metric_key, "unknown"

            # Calculate statistics
            stats[metric_name] = MetricStats(
                count=len(values),
                mean=statistics.mean(values),
                median=statistics.median(values),
                min=min(values),
                max=max(values),
                std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
                unit=unit,
                raw_values=values.copy(),
            )

        return stats

    def export_to_json(self, filepath: Path) -> None:
        """
        Export collected metrics to a JSON file.

        Args:
            filepath: Path where the JSON file should be written
        """
        if not self.run_id or not self.start_time:
            raise ValueError("Cannot export metrics - collection was never started")

        # Prepare raw metrics data (without units in keys for cleaner JSON)
        raw_metrics: dict[str, list[float]] = {}
        for metric_key, values in self.metrics_by_type.items():
            # Remove unit suffix for cleaner JSON keys
            parts = metric_key.rsplit("_", 1)
            clean_name = parts[0] if len(parts) == 2 else metric_key
            raw_metrics[clean_name] = values

        export_data: MetricsExport = {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else "",
            "total_tasks": self.task_count,
            "metrics_by_type": raw_metrics,
            "aggregated_stats": self.get_aggregated_stats(),
        }

        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"📊 Exported metrics to: {filepath}")

    def print_summary(self) -> None:
        """
        Print a summary of collected metrics to the console.
        """
        if not self.metrics_by_type:
            print("📊 No metrics collected")
            return

        stats = self.get_aggregated_stats()

        print(f"\n{'=' * 80}")
        print(f"📊 PERFORMANCE METRICS SUMMARY - {self.run_id}")
        print(f"{'=' * 80}")
        print(f"Tasks processed: {self.task_count}")

        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            print(f"Collection duration: {duration:.2f}s")

        print("\nAggregated Performance Metrics:")
        print(
            f"{'Metric':<40} {'Count':<8} {'Mean':<12} {'Median':<12} {'Min':<12} {'Max':<12} {'Unit'}"
        )
        print(
            f"{'-' * 40} {'-' * 8} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 8}"
        )

        # Sort metrics by name for consistent display
        for metric_name in sorted(stats.keys()):
            stat = stats[metric_name]
            unit_display = stat["unit"]

            # Format values based on unit
            if unit_display == "ms":
                mean_str = f"{stat['mean']:.1f}"
                median_str = f"{stat['median']:.1f}"
                min_str = f"{stat['min']:.1f}"
                max_str = f"{stat['max']:.1f}"
            elif unit_display == "s":
                mean_str = f"{stat['mean']:.3f}"
                median_str = f"{stat['median']:.3f}"
                min_str = f"{stat['min']:.3f}"
                max_str = f"{stat['max']:.3f}"
            else:
                mean_str = f"{stat['mean']:.2f}"
                median_str = f"{stat['median']:.2f}"
                min_str = f"{stat['min']:.2f}"
                max_str = f"{stat['max']:.2f}"

            print(
                f"{metric_name:<40} {stat['count']:<8} {mean_str:<12} {median_str:<12} {min_str:<12} {max_str:<12} {unit_display}"
            )

        print(f"{'=' * 80}")

    def cleanup_collection(self) -> None:
        """
        Cleanup metrics collection.
        """
        self.is_collecting = False
        self.end_time = datetime.now()
        # delete the metrics file
        metrics_file = get_metrics_file_path()
        if metrics_file:
            metrics_file.unlink()


def get_metrics_file_path() -> Path | None:
    """Get the current metrics file path."""
    env_path = settings.LOCAL_METRICS_FILE
    if env_path:
        _metrics_file_path = Path(env_path)
        return _metrics_file_path

    return None


def append_metrics_to_file(
    task_slug: str,
    task_name: str,
    metrics_list: list[tuple[str, str | int | float, str]],
) -> None:
    """
    Append metrics to the shared metrics file for cross-process collection.

    Args:
        task_slug: Slug of the task that generated these metrics
        task_name: Name of the task that generated these metrics
        metrics_list: List of (metric_name, value, unit) tuples
    """
    metrics_file = get_metrics_file_path()
    if not metrics_file:
        return

    import fcntl
    import time

    # Prepare metrics data
    timestamp = time.time()
    metrics_entry = {
        "timestamp": timestamp,
        "task_name": f"{task_slug}_{task_name}",
        "metrics": [
            {"name": f"{task_slug}_{name}", "value": value, "unit": unit}
            for name, value, unit in metrics_list
        ],
    }

    # Use file locking to handle concurrent writes from multiple processes
    with open(metrics_file, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(metrics_entry) + "\n")
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_metrics_from_file(
    filepath: Path,
) -> list[tuple[str, list[tuple[str, str | int | float, str]]]]:
    """
    Load all metrics from the shared metrics file.

    Args:
        filepath: Path to the metrics file

    Returns:
        List of (task_name, metrics_list) tuples
    """
    if not filepath.exists():
        return []

    all_metrics: list[tuple[str, list[tuple[str, str | int | float, str]]]] = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line.strip())
                task_name = entry["task_name"]
                metrics_list = [
                    (m["name"], m["value"], m["unit"]) for m in entry["metrics"]
                ]
                all_metrics.append((task_name, metrics_list))

    return all_metrics