"""
Common utilities for OOLONG benchmark test runners.

Based on the OOLONG paper:
- OOLONG-synth: Synthetic ICL-based aggregation tasks
- OOLONG-real: Real D&D transcript aggregation tasks
"""

import ast
import json
import logging
import re
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import dateutil.parser
import pyarrow.parquet as pq
import tiktoken
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class SimpleDataset:
    """Simple dataset class that mimics HuggingFace dataset API."""

    data: list[dict[str, Any]]
    column_names: list[str]

    def __init__(self, data: list[dict[str, Any]]):
        """Initialize dataset with list of examples.

        Args:
            data: List of example dictionaries
        """
        self.data = data
        self.column_names = list(data[0].keys()) if data else []

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get example by index."""
        return self.data[idx]

    def filter(self, function: Callable[[dict[str, Any]], bool]) -> "SimpleDataset":
        """Filter dataset using a function.

        Args:
            function: Filter function

        Returns:
            Filtered dataset
        """
        filtered_data = [item for item in self.data if function(item)]
        return SimpleDataset(filtered_data)

    def select(self, indices: Sequence[int]) -> "SimpleDataset":
        """Select examples by indices.

        Args:
            indices: List of indices to select

        Returns:
            Dataset with selected examples
        """
        selected_data = [self.data[i] for i in indices]
        return SimpleDataset(selected_data)


class BaseQueryResult(TypedDict):
    """Base type definition for query execution results."""

    question: str
    expected_answer: str
    actual_response: str
    score: float
    context_length_tokens: int


class BaseTestResult(TypedDict):
    """Base type definition for test execution results."""

    question_id: str
    task_group: str
    dataset: str
    passed: bool
    score: float
    error: str | None
    start_time: float
    end_time: float
    duration_seconds: float
    output_lines: list[str]


def format_duration(total_seconds: float) -> str:
    """Format a duration in seconds into a human-readable string.

    If the duration is at least one minute, this returns a string in the
    form "XmYYs" with zero-padded seconds. Otherwise, it returns the
    duration in seconds with two decimal places, e.g., "12.34s".

    Args:
        total_seconds: The duration in seconds.

    Returns:
        A formatted duration string.
    """
    minutes = int(total_seconds // 60)
    if minutes > 0:
        seconds_rounded = int(round(total_seconds - minutes * 60))
        if seconds_rounded == 60:
            minutes += 1
            seconds_rounded = 0
        return f"{minutes}m{seconds_rounded:02d}s"
    return f"{total_seconds:.2f}s"


def calculate_context_length(text: str) -> int:
    """Calculate token count for context text using tiktoken.

    Args:
        text: Context text to count tokens for

    Returns:
        Number of tokens
    """
    try:
        tokenizer = tiktoken.get_encoding("o200k_base")
        return len(
            tokenizer.encode(
                text,
                disallowed_special=(tokenizer.special_tokens_set - {"<|endoftext|>"}),
            )
        )
    except Exception:
        # Fallback to character-based estimate
        return len(text) // 4


def load_oolong_synth_dataset(
    split: str = "test", data_dir: str | Path | None = None
) -> SimpleDataset:
    """Load the OOLONG-synth dataset from filesystem.

    Args:
        split: Dataset split to load (default: "test")
        data_dir: Path to the oolong-synth dataset directory (must contain a 'data' subdirectory)

    Returns:
        SimpleDataset object

    Raises:
        ValueError: If data_dir is not provided
        FileNotFoundError: If no parquet files found for the split
    """
    if data_dir is None:
        raise ValueError(
            "data_dir parameter is required. Please provide the path to the oolong-synth dataset."
        )

    dataset_path = Path(data_dir) / "data"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Expected synth dataset directory at {dataset_path} (layout: data/*.parquet)"
        )

    # Find all parquet files for the given split
    parquet_files = sorted(dataset_path.glob(f"{split}-*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No {split} parquet files found in {dataset_path}")

    # Load and combine all parquet files
    all_data: list[dict[str, Any]] = []
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file)  # pyright: ignore[reportUnknownVariableType]
        all_data.extend(table.to_pylist())  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

    return SimpleDataset(all_data)


def load_oolong_real_dataset(
    split: str = "test", data_dir: str | Path | None = None
) -> SimpleDataset:
    """Load the OOLONG-real dataset from filesystem.

    Args:
        split: Dataset split to load (default: "test")
        data_dir: Path to the oolong-real dataset directory (must contain a 'dnd' subdirectory)

    Returns:
        SimpleDataset object

    Raises:
        ValueError: If data_dir is not provided
        FileNotFoundError: If JSONL file not found
    """
    if data_dir is None:
        raise ValueError(
            "data_dir parameter is required. Please provide the path to the oolong-real dataset."
        )

    dataset_path = Path(data_dir) / "dnd"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Expected real dataset directory at {dataset_path} (layout: dnd/*.jsonl)"
        )
    jsonl_file = dataset_path / f"{split}.jsonl"

    if not jsonl_file.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")

    # Load JSONL file
    data: list[dict[str, Any]] = []
    with open(jsonl_file) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    return SimpleDataset(data)


def parse_synth_context_messages(context_text: str) -> list[dict[str, Any]]:
    """Parse OOLONG-synth context text into individual messages.

    Context format:
    Date: YYYY-MM-DD || User: user_XYZ || Instance: <text> [label]

    Args:
        context_text: Raw context window text

    Returns:
        List of message dictionaries with content and metadata
    """
    messages: list[dict[str, Any]] = []

    # Split by lines and parse each entry
    lines = context_text.strip().split("\n")
    for line in lines:
        if not line.strip():
            continue

        # Parse: Date: ... || User: ... || Instance: ... || Label: ...
        try:
            parts = line.split(" || ")
            if len(parts) < 3:
                logger.warning(f"Skipping malformed line: {line}")
                continue

            date_part = parts[0].replace("Date: ", "").strip()
            user_part = parts[1].replace("User: ", "").strip()
            instance_part = parts[2].replace("Instance: ", "").strip()

            # Check if there's a 4th part with label
            label = None
            if len(parts) >= 4:
                label_part = parts[3].replace("Label: ", "").strip()
                label = label_part if label_part else None

            # Include label in content so deriver can observe it.
            content = f"{instance_part} [Label: {label}]" if label else instance_part

            msg: dict[str, Any] = {
                "content": content,
                "metadata": {
                    "date": date_part,
                    "user_id": user_part,
                    "label": label,
                },
            }
            messages.append(msg)
        except Exception as e:
            logger.warning(f"Error parsing line: {line}. Error: {e}")
            continue

    return messages


def parse_real_context_messages(context_text: str) -> list[dict[str, Any]]:
    """Parse OOLONG-real D&D transcript into individual messages.

    Context format:
    Speaker: dialogue text
    [multiple lines]

    Args:
        context_text: Raw D&D transcript text

    Returns:
        List of message dictionaries with content and metadata
    """
    messages: list[dict[str, Any]] = []

    # Split by speaker turns (format: "SPEAKER: text")
    lines = context_text.strip().split("\n")
    current_speaker = None
    current_content = []

    for line in lines:
        if not line.strip():
            continue

        # Check if this is a new speaker turn (must start with speaker label)
        # Speaker labels are typically single words or use underscores/hyphens (no spaces)
        speaker_match = re.match(r"^\s*([A-Za-z0-9_-]+):", line)
        if speaker_match:
            # Save previous message if exists
            if current_speaker and current_content:
                # Include speaker in content for deriver visibility
                content_text = " ".join(current_content)
                content = f"[Speaker: {current_speaker}] {content_text}"
                prev_msg: dict[str, Any] = {
                    "content": content,
                    "metadata": {
                        "speaker": current_speaker,
                    },
                }
                messages.append(prev_msg)

            # Parse new speaker from regex match
            current_speaker = speaker_match.group(1).strip()
            # Extract content after the colon
            content_after_colon = line[speaker_match.end() :].strip()
            current_content = [content_after_colon] if content_after_colon else []
        else:
            # Continuation of current speaker's dialogue
            current_content.append(line.strip())

    # Save last message
    if current_speaker and current_content:
        content_text = " ".join(current_content)
        content = f"[Speaker: {current_speaker}] {content_text}"
        last_msg: dict[str, Any] = {
            "content": content,
            "metadata": {
                "speaker": current_speaker,
            },
        }
        messages.append(last_msg)

    return messages


def parse_synth_answer(answer_str: str) -> Any:
    """Parse OOLONG-synth answer string.

    Answers can be:
    - Strings (labels, comparisons, dates)
    - Numbers (counts)
    - Dates (datetime objects)

    Args:
        answer_str: Raw answer string from dataset

    Returns:
        Parsed answer value
    """
    # Handle datetime answers
    if "datetime" in answer_str:
        try:
            # Format: [datetime.date(2023, 5, 15)]
            match = re.search(r"datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)", answer_str)
            if match:
                year, month, day = map(int, match.groups())
                return datetime(year, month, day).date()
        except Exception as e:
            logger.warning(f"Error parsing datetime answer: {answer_str}. Error: {e}")
            return answer_str

    # Try literal eval for lists/primitives
    try:
        parsed = ast.literal_eval(answer_str)
        # If it's a list with one element, return that element
        if isinstance(parsed, list) and len(parsed) == 1:  # pyright: ignore[reportUnknownArgumentType]
            return parsed[0]  # pyright: ignore[reportUnknownVariableType]
        return parsed  # pyright: ignore[reportUnknownVariableType]
    except (ValueError, SyntaxError):
        # Return as-is if can't parse
        return answer_str


def parse_real_answer(answer_str: str) -> int | str | list[str]:
    """Parse OOLONG-real answer string.

    Answers can be:
    - Integers (counts)
    - Strings (spell names, roll types)
    - Lists (comma-separated spells)

    Args:
        answer_str: Raw answer string from dataset

    Returns:
        Parsed answer value
    """
    # Try to convert to int first
    try:
        return int(answer_str)
    except ValueError:
        pass

    # Check if it contains commas (list case)
    if "," in answer_str:
        return [item.strip() for item in answer_str.split(",") if item.strip()]

    # Otherwise return as string
    return answer_str


def score_synth_response(
    gold_answer: Any, model_answer_str: str, answer_type: str
) -> float:
    """Score a response for OOLONG-synth following the paper's scoring rubric.

    Scoring:
    - Exact match: 1.0
    - Numeric: 0.75^|y-ŷ| (partial credit)
    - Other: 0.0

    Args:
        gold_answer: Expected answer
        model_answer_str: Model's response string
        answer_type: Type of answer (ANSWER_TYPE.NUMERIC, ANSWER_TYPE.DATE, etc.)

    Returns:
        Score between 0.0 and 1.0
    """
    # Try to extract answer from response
    # Look for common patterns at start of line or end of response
    model_answer = model_answer_str.strip()

    # Try to extract from common formats (use MULTILINE to match at line start, and $ to find at line end)
    # Look for patterns at the start of a line (final answer format)
    for pattern in [
        r"^[Aa]nswer:\s*(.+)$",  # "Answer: X" on its own line
        r"^[Ll]abel:\s*(.+)$",  # "Label: X" on its own line
        r"\n[Ll]abel:\s*(.+)$",  # "Label: X" after a newline (final answer)
        r"[Aa]nswer:\s*(.+)$",  # "Answer: X" at end of response
        r"[Uu]ser:\s*(.+)$",  # "User: X" at end
        r"[Dd]ate:\s*(.+)$",  # "Date: X" at end
    ]:
        match = re.search(pattern, model_answer, re.MULTILINE)
        if match:
            model_answer = match.group(1).strip()
            break

    # Remove formatting like ** or []
    model_answer = re.sub(r"[\*\[\]]", "", model_answer)

    # If still long, try last significant token
    if len(model_answer) > 50:
        model_answer = model_answer.split()[-1]

    # Exact string match
    if str(model_answer).lower() == str(gold_answer).lower():
        return 1.0

    # Check for comparison answers
    if (
        "more common" in model_answer.lower()
        and "more common" in str(gold_answer).lower()
    ):
        return 1.0
    if (
        "less common" in model_answer.lower()
        and "less common" in str(gold_answer).lower()
    ):
        return 1.0
    if (
        "same frequency" in model_answer.lower()
        and "same frequency" in str(gold_answer).lower()
    ):
        return 1.0

    # Numeric partial credit
    if "NUMERIC" in answer_type.upper():
        try:
            model_num = float(re.sub(r"[^\d.-]", "", model_answer))
            gold_num = float(gold_answer)
            return 0.75 ** abs(gold_num - model_num)
        except (ValueError, TypeError):
            return 0.0

    # Date matching
    if "DATE" in answer_type.upper():
        try:
            model_date = dateutil.parser.parse(model_answer)
            if isinstance(gold_answer, datetime):
                return 1.0 if model_date.date() == gold_answer.date() else 0.0
            elif isinstance(gold_answer, date):
                return 1.0 if model_date.date() == gold_answer else 0.0
            return 0.0
        except (ValueError, TypeError):
            return 0.0

    return 0.0


def score_real_response(
    gold_answer: int | str | list[str], model_answer_str: str
) -> float:
    """Score a response for OOLONG-real following the paper's scoring rubric.

    Scoring:
    - Integer: 0.75^|y-ŷ| (partial credit)
    - String: exact match (1.0 or 0.0)
    - List: set overlap / |gold| (Jaccard-style)

    Args:
        gold_answer: Expected answer
        model_answer_str: Model's response string

    Returns:
        Score between 0.0 and 1.0
    """
    # Extract answer from \boxed{} format if present
    match = re.search(r"\\boxed\{\\text\{([^}]*)\}\}", model_answer_str) or re.search(
        r"\\boxed[\{]+([^}]*)[\}]+", model_answer_str
    )

    if match:
        model_answer_str = match.group(1)

    # Parse model answer
    try:
        model_answer: int | str | list[str] = int(model_answer_str)
    except ValueError:
        if "," in model_answer_str:
            model_answer = [
                item.strip() for item in model_answer_str.split(",") if item.strip()
            ]
        else:
            model_answer = model_answer_str.strip()

    # Score based on type
    if isinstance(gold_answer, int) and isinstance(model_answer, int):
        return 0.75 ** abs(gold_answer - model_answer)
    elif isinstance(gold_answer, str) and isinstance(model_answer, str):
        return 1.0 if gold_answer.lower() == model_answer.lower() else 0.0
    elif isinstance(gold_answer, list) and isinstance(model_answer, list):
        overlap = set(gold_answer) & set(model_answer)
        return len(overlap) / len(gold_answer) if gold_answer else 0.0
    else:
        return 0.0


def filter_dataset(
    dataset: SimpleDataset,
    max_context_len: int | None = None,
    min_context_len: int | None = None,
    max_examples: int | None = None,
    context_window_id: str | None = None,
) -> SimpleDataset:
    """Filter dataset by context length and example count.

    Args:
        dataset: HuggingFace dataset
        max_context_len: Maximum context length in tokens
        min_context_len: Minimum context length in tokens
        max_examples: Maximum number of examples to return
        context_window_id: Specific context window ID to filter to

    Returns:
        Filtered dataset
    """
    if context_window_id is not None:
        dataset = dataset.filter(lambda x: x["context_window_id"] == context_window_id)

    if max_context_len is not None:
        dataset = dataset.filter(
            lambda x: x.get(
                "context_len",
                calculate_context_length(str(x.get("context_window_text", ""))),
            )
            <= max_context_len
        )

    if min_context_len is not None:
        dataset = dataset.filter(
            lambda x: x.get(
                "context_len",
                calculate_context_length(str(x.get("context_window_text", ""))),
            )
            > min_context_len
        )

    if max_examples is not None and max_examples > 0:
        # Get a slice of the dataset
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    return dataset


def calculate_task_statistics(
    results: Sequence[Any],
) -> dict[str, dict[str, int | float]]:
    """Calculate score statistics grouped by task type.

    Args:
        results: List of test results

    Returns:
        Dictionary mapping task type to statistics
    """
    task_stats: dict[str, dict[str, int | float]] = {}

    for result in results:
        task_key = result.get("task_group", "unknown")
        if task_key not in task_stats:
            task_stats[task_key] = {"total": 0, "total_score": 0.0, "perfect_scores": 0}

        task_stats[task_key]["total"] += 1
        score = result.get("score", 0.0)
        task_stats[task_key]["total_score"] += score
        if score >= 0.99:  # Consider >= 0.99 as perfect to account for floating point
            task_stats[task_key]["perfect_scores"] += 1

    # Calculate averages
    for task_key in task_stats:
        stats = task_stats[task_key]
        total = stats["total"]
        stats["average_score"] = stats["total_score"] / total if total > 0 else 0.0
        stats["perfect_score_rate"] = (
            (stats["perfect_scores"] / total) * 100 if total > 0 else 0.0
        )

    return task_stats


def calculate_timing_statistics(
    results: Sequence[Any], total_elapsed_seconds: float
) -> dict[str, Any]:
    """Calculate timing statistics from test results.

    Args:
        results: List of test results
        total_elapsed_seconds: Total elapsed time for all tests

    Returns:
        Dictionary of timing statistics
    """
    durations = [r["duration_seconds"] for r in results]
    return {
        "total_duration_seconds": total_elapsed_seconds,
        "individual_test_durations": {
            "min_seconds": min(durations) if durations else 0,
            "max_seconds": max(durations) if durations else 0,
            "mean_seconds": sum(durations) / len(durations) if durations else 0,
            "median_seconds": sorted(durations)[len(durations) // 2]
            if durations
            else 0,
        },
    }


def write_json_summary(summary: dict[str, Any], output_file: Path) -> None:
    """Write a JSON summary to a file.

    Args:
        summary: Summary dictionary to write
        output_file: Path to output file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nJSON summary written to: {output_file}")
