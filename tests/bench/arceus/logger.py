"""JSON trace logging system for debugging and analysis."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class JSONTraceLogger:
    """Logger that outputs structured JSON traces for debugging."""

    def __init__(self, output_dir: Path, task_id: str, enable: bool = True):
        """
        Initialize the JSON trace logger.

        Args:
            output_dir: Directory to write trace files
            task_id: Identifier for the task being solved
            enable: Whether logging is enabled
        """
        self.output_dir = output_dir
        self.task_id = task_id
        self.enable = enable
        self.traces: list[Dict[str, Any]] = []
        self.start_time = datetime.now()

        if self.enable:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.output_dir / f"{task_id}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"

    def log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log an event with structured data.

        Args:
            event_type: Type of event (e.g., 'analysis', 'reasoning', 'memory_query')
            data: Event data
            metadata: Optional metadata
        """
        if not self.enable:
            return

        event = {
            "timestamp": datetime.now().isoformat(),
            "task_id": self.task_id,
            "event_type": event_type,
            "data": data,
            "metadata": metadata or {},
        }

        self.traces.append(event)

    def log_reasoning_step(
        self,
        step_number: int,
        step_type: str,
        content: str,
        confidence: float = 0.0,
    ):
        """Log a reasoning step from the solver."""
        self.log_event(
            "reasoning_step",
            {
                "step_number": step_number,
                "step_type": step_type,
                "content": content,
                "confidence": confidence,
            },
        )

    def log_memory_query(
        self, query_type: str, query: str, results: Any, metadata: Optional[Dict] = None
    ):
        """Log a memory query operation."""
        self.log_event(
            "memory_query",
            {"query_type": query_type, "query": query, "results": results},
            metadata=metadata,
        )

    def log_transformation_attempt(
        self, primitive: str, input_grid: Any, output_grid: Any, success: bool
    ):
        """Log a transformation attempt."""
        self.log_event(
            "transformation_attempt",
            {
                "primitive": primitive,
                "input_shape": (
                    (len(input_grid), len(input_grid[0]))
                    if input_grid
                    else None
                ),
                "output_shape": (
                    (len(output_grid), len(output_grid[0]))
                    if output_grid
                    else None
                ),
                "success": success,
            },
        )

    def log_verification(
        self,
        example_idx: int,
        expected: Any,
        actual: Any,
        passed: bool,
        details: Optional[str] = None,
    ):
        """Log a verification result."""
        self.log_event(
            "verification",
            {
                "example_idx": example_idx,
                "expected_shape": (
                    (len(expected), len(expected[0])) if expected else None
                ),
                "actual_shape": (
                    (len(actual), len(actual[0])) if actual else None
                ),
                "passed": passed,
                "details": details,
            },
        )

    def log_error(self, error_type: str, message: str, stack_trace: Optional[str] = None):
        """Log an error."""
        self.log_event(
            "error", {"error_type": error_type, "message": message, "stack_trace": stack_trace}
        )

    def flush(self):
        """Write all traces to disk."""
        if not self.enable or not self.traces:
            return

        try:
            with open(self.log_file, "w") as f:
                json.dump(
                    {
                        "task_id": self.task_id,
                        "start_time": self.start_time.isoformat(),
                        "end_time": datetime.now().isoformat(),
                        "num_events": len(self.traces),
                        "events": self.traces,
                    },
                    f,
                    indent=2,
                )
            logging.info(f"Trace written to {self.log_file}")
        except Exception as e:
            logging.error(f"Failed to write trace file: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of logged events."""
        if not self.traces:
            return {}

        event_types = {}
        for trace in self.traces:
            event_type = trace["event_type"]
            event_types[event_type] = event_types.get(event_type, 0) + 1

        return {
            "total_events": len(self.traces),
            "event_types": event_types,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
        }
