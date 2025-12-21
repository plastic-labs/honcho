"""
Honcho Trace Ingestion Tool

A script for ingesting conversation data from various benchmark datasets into Honcho
without running the question-answering evaluation. This is useful for:
- Performance testing of message ingestion
- Testing deriver processing at scale
- Pre-loading conversations for later analysis
- Benchmarking trace processing speed

## Supported Datasets

1. **LoCoMo** - Long conversation memory dataset
2. **LongMemEval** - Long-term memory evaluation dataset
3. **BEAM** - Benchmark for long-context memory
4. Custom conversation datasets following the standard format

## Usage

1. Start the harness pool:
```
python -m tests.bench.harness --pool-size 4
```

2. Run trace ingestion:
```
# LoCoMo dataset (single JSON file)
python -m tests.bench.trace --data-dir tests/bench/locomo_data/locomo10.json

# LongMemEval dataset (single file)
python -m tests.bench.trace --data-dir tests/bench/longmemeval_data/longmemeval_single.json

# LongMemEval dataset (directory - requires --test-file)
python -m tests.bench.trace --data-dir tests/bench/longmemeval_data --test-file longmemeval_single.json

# BEAM dataset (directory with 100K, 500K, etc. subdirectories)
python -m tests.bench.trace --data-dir tests/bench/beam_data --context-length 100K --conversation-id 0
```

## Arguments

- `--data-dir`: Path to data file or directory (auto-detects dataset type)
- `--test-file`: Test file name (for LongMemEval, optional)
- `--context-length`: Context length for BEAM (100K, 500K, 1M, 10M, optional)
- `--conversation-id`: Specific conversation ID to process (optional)
- `--base-api-port`: Base port for Honcho API instances (default: 8000)
- `--pool-size`: Number of Honcho instances (default: 1)
- `--batch-size`: Number of conversations to process concurrently (default: 1)
- `--timeout`: Timeout for deriver queue in seconds (default: 600)
- `--cleanup-workspace`: Delete workspace after ingestion (default: False)
- `--json-output`: Path to write JSON summary results
- `--sample-limit`: Limit number of conversations to process
- `--wait-for-deriver`: Wait for deriver queue to empty (default: True)
- `--trigger-dream`: Trigger dream consolidation after ingestion (default: False)
"""

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from honcho import AsyncHoncho
from honcho.async_client.session import SessionPeerConfig
from honcho_core.types.workspaces.sessions.message_create_param import (
    MessageCreateParam,
)

from .beam_common import (
    extract_messages_from_chat_data,
    load_conversation as load_beam_conversation,
    list_conversations as list_beam_conversations,
)
from .locomo_common import (
    calculate_tokens,
    extract_sessions,
    format_duration,
    load_locomo_data,
    parse_locomo_date,
)
from .longmem_common import (
    calculate_total_tokens,
    load_test_file,
    parse_longmemeval_date,
)

# Load .env from bench directory
bench_dir = Path(__file__).parent
load_dotenv(bench_dir / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TraceResult:
    """Result from ingesting a conversation trace."""

    def __init__(self, trace_id: str, dataset: str):
        self.trace_id = trace_id
        self.dataset = dataset
        self.workspace_id: str = ""
        self.total_messages: int = 0
        self.total_tokens: int = 0
        self.total_sessions: int = 0
        self.ingestion_time: float = 0.0
        self.deriver_wait_time: float = 0.0
        self.dream_time: float = 0.0
        self.total_time: float = 0.0
        self.error: str | None = None
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "trace_id": self.trace_id,
            "dataset": self.dataset,
            "workspace_id": self.workspace_id,
            "total_messages": self.total_messages,
            "total_tokens": self.total_tokens,
            "total_sessions": self.total_sessions,
            "ingestion_time_seconds": self.ingestion_time,
            "deriver_wait_time_seconds": self.deriver_wait_time,
            "dream_time_seconds": self.dream_time,
            "total_time_seconds": self.total_time,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class TraceRunner:
    """Runner for ingesting conversation traces into Honcho."""

    def __init__(
        self,
        base_api_port: int = 8000,
        pool_size: int = 1,
        timeout_seconds: int = 600,
        cleanup_workspace: bool = False,
        wait_for_deriver: bool = True,
        trigger_dream: bool = False,
    ):
        """
        Initialize the trace runner.

        Args:
            base_api_port: Base port for Honcho API instances
            pool_size: Number of Honcho instances in the pool
            timeout_seconds: Timeout for deriver queue in seconds
            cleanup_workspace: If True, delete workspace after ingestion
            wait_for_deriver: If True, wait for deriver queue to empty
            trigger_dream: If True, trigger dream consolidation after ingestion
        """
        self.base_api_port = base_api_port
        self.pool_size = pool_size
        self.timeout_seconds = timeout_seconds
        self.cleanup_workspace = cleanup_workspace
        self.wait_for_deriver = wait_for_deriver
        self.trigger_dream = trigger_dream

    def get_honcho_url_for_index(self, index: int) -> str:
        """Get the Honcho URL for a given index using round-robin distribution."""
        instance_id = index % self.pool_size
        port = self.base_api_port + instance_id
        return f"http://localhost:{port}"

    async def create_honcho_client(
        self, workspace_id: str, honcho_url: str
    ) -> AsyncHoncho:
        """Create a Honcho client for a specific workspace."""
        return AsyncHoncho(
            environment="local",
            workspace_id=workspace_id,
            base_url=honcho_url,
        )

    async def wait_for_deriver_queue_empty(
        self, honcho_client: AsyncHoncho, session_id: str | None = None
    ) -> bool:
        """Wait for the deriver queue to be empty."""
        start_time = time.time()
        while True:
            try:
                status = await honcho_client.get_deriver_status(session=session_id)
            except Exception:
                await asyncio.sleep(1)
                elapsed_time = time.time() - start_time
                if elapsed_time >= self.timeout_seconds:
                    return False
                continue

            if status.pending_work_units == 0 and status.in_progress_work_units == 0:
                return True

            elapsed_time = time.time() - start_time
            if elapsed_time >= self.timeout_seconds:
                return False
            await asyncio.sleep(1)

    async def ingest_locomo_trace(
        self,
        conversation_data: dict[str, Any],
        honcho_url: str,
        index: int,
    ) -> TraceResult:
        """
        Ingest a LoCoMo conversation trace.

        Args:
            conversation_data: Dictionary containing conversation data
            honcho_url: URL of the Honcho instance to use
            index: Index for generating unique IDs

        Returns:
            TraceResult with ingestion metrics
        """
        start_time = time.time()
        sample_id = conversation_data.get("sample_id", f"unknown_{index}")
        conversation = conversation_data.get("conversation", {})

        result = TraceResult(trace_id=sample_id, dataset="locomo")
        result.start_time = start_time

        speaker_a = conversation.get("speaker_a", "User")
        speaker_b = conversation.get("speaker_b", "Assistant")

        logger.info(f"[{sample_id}] Ingesting LoCoMo trace: {speaker_a} and {speaker_b}")

        # Create workspace
        workspace_id = f"locomo_trace_{sample_id}_{int(start_time)}"
        result.workspace_id = workspace_id
        honcho_client = await self.create_honcho_client(workspace_id, honcho_url)

        try:
            # Create peers
            peer_a = await honcho_client.peer(id=speaker_a)
            peer_b = await honcho_client.peer(id=speaker_b)

            # Create session
            session_id = f"{workspace_id}_session"
            session = await honcho_client.session(id=session_id)

            # Configure peer observation
            await session.add_peers(
                [
                    (peer_a, SessionPeerConfig(observe_me=True, observe_others=False)),
                    (peer_b, SessionPeerConfig(observe_me=True, observe_others=False)),
                ]
            )

            # Extract and prepare messages
            sessions = extract_sessions(conversation)
            result.total_sessions = len(sessions)

            messages: list[MessageCreateParam] = []
            total_tokens = 0

            for date_str, session_messages in sessions:
                session_date = parse_locomo_date(date_str) if date_str else None

                for msg in session_messages:
                    speaker = msg.get("speaker", "")
                    text = msg.get("text", "")
                    blip_caption = msg.get("blip_caption")

                    # Format content with image caption if present
                    content = (
                        f"{text}\n\n[Image shared: {blip_caption}]"
                        if blip_caption
                        else text
                    )
                    total_tokens += calculate_tokens(content)

                    # Build metadata if image data exists
                    metadata: dict[str, Any] | None = None
                    if blip_caption or msg.get("img_url") or msg.get("query"):
                        metadata = {}
                        if msg.get("img_url"):
                            metadata["img_urls"] = msg["img_url"]
                        if blip_caption:
                            metadata["blip_caption"] = blip_caption
                        if msg.get("query"):
                            metadata["image_query"] = msg["query"]

                    # Map speaker to peer
                    if speaker == speaker_a:
                        messages.append(
                            peer_a.message(
                                content, metadata=metadata, created_at=session_date
                            )
                        )
                    elif speaker == speaker_b:
                        messages.append(
                            peer_b.message(
                                content, metadata=metadata, created_at=session_date
                            )
                        )

            result.total_messages = len(messages)
            result.total_tokens = total_tokens

            # Ingest messages in batches of 100
            ingestion_start = time.time()
            for i in range(0, len(messages), 100):
                batch = messages[i : i + 100]
                await session.add_messages(batch)
                logger.info(
                    f"[{sample_id}] Ingested batch {i // 100 + 1}/{(len(messages) + 99) // 100}"
                )

            result.ingestion_time = time.time() - ingestion_start

            logger.info(
                f"[{sample_id}] Ingested {len(messages)} messages (~{total_tokens:,} tokens) in {format_duration(result.ingestion_time)}"
            )

            # Wait for deriver queue
            if self.wait_for_deriver:
                logger.info(f"[{sample_id}] Waiting for deriver queue to empty...")
                deriver_start = time.time()
                await asyncio.sleep(1)
                queue_empty = await self.wait_for_deriver_queue_empty(
                    honcho_client, session_id
                )
                result.deriver_wait_time = time.time() - deriver_start

                if not queue_empty:
                    result.error = f"Deriver queue timeout after {self.timeout_seconds}s"
                    logger.warning(f"[{sample_id}] {result.error}")
                else:
                    logger.info(
                        f"[{sample_id}] Deriver queue empty after {format_duration(result.deriver_wait_time)}"
                    )

            # Trigger dream if requested
            if self.trigger_dream and not result.error:
                logger.info(f"[{sample_id}] Triggering dream consolidation...")
                dream_start = time.time()
                # Note: Dream triggering logic would go here
                # Simplified for now - would need to import and use trigger_dream_and_wait
                result.dream_time = time.time() - dream_start

            # Cleanup workspace if requested
            if self.cleanup_workspace:
                logger.info(f"[{sample_id}] Cleaning up workspace...")
                try:
                    await honcho_client.delete_workspace()
                except Exception as e:
                    logger.warning(f"[{sample_id}] Failed to cleanup workspace: {e}")

        except Exception as e:
            result.error = str(e)
            logger.error(f"[{sample_id}] Error: {e}", exc_info=True)

        result.end_time = time.time()
        result.total_time = result.end_time - result.start_time
        return result

    async def ingest_longmem_trace(
        self,
        test_data: dict[str, Any],
        honcho_url: str,
        index: int,
    ) -> TraceResult:
        """
        Ingest a LongMemEval conversation trace.

        Args:
            test_data: Dictionary containing test data
            honcho_url: URL of the Honcho instance to use
            index: Index for generating unique IDs

        Returns:
            TraceResult with ingestion metrics
        """
        start_time = time.time()
        question_id = test_data.get("question_id", f"unknown_{index}")

        result = TraceResult(trace_id=question_id, dataset="longmem")
        result.start_time = start_time

        logger.info(f"[{question_id}] Ingesting LongMemEval trace")

        # Create workspace
        workspace_id = f"longmem_trace_{question_id}_{int(start_time)}"
        result.workspace_id = workspace_id
        honcho_client = await self.create_honcho_client(workspace_id, honcho_url)

        try:
            # Create peer (LongMemEval is single user)
            user_peer = await honcho_client.peer(id="user")

            # Create session
            session_id = f"{workspace_id}_session"
            session = await honcho_client.session(id=session_id)

            # Configure peer observation
            await session.add_peers(
                [(user_peer, SessionPeerConfig(observe_me=True, observe_others=False))]
            )

            # Extract haystack sessions
            haystack_sessions = test_data.get("haystack_sessions", [])
            result.total_sessions = len(haystack_sessions)

            messages: list[MessageCreateParam] = []
            total_tokens = calculate_total_tokens(haystack_sessions)

            for session_messages in haystack_sessions:
                # Get timestamp from first message if available
                first_msg_meta = (
                    session_messages[0].get("metadata", {})
                    if session_messages
                    else {}
                )
                timestamp_str = first_msg_meta.get("timestamp")
                timestamp = (
                    parse_longmemeval_date(timestamp_str) if timestamp_str else None
                )

                for msg in session_messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    metadata = msg.get("metadata", {})

                    if role == "user":
                        messages.append(
                            user_peer.message(
                                content, metadata=metadata, created_at=timestamp
                            )
                        )
                    # Skip assistant messages as they're not part of user history

            result.total_messages = len(messages)
            result.total_tokens = total_tokens

            # Ingest messages in batches of 100
            ingestion_start = time.time()
            for i in range(0, len(messages), 100):
                batch = messages[i : i + 100]
                await session.add_messages(batch)
                logger.info(
                    f"[{question_id}] Ingested batch {i // 100 + 1}/{(len(messages) + 99) // 100}"
                )

            result.ingestion_time = time.time() - ingestion_start

            logger.info(
                f"[{question_id}] Ingested {len(messages)} messages (~{total_tokens:,} tokens) in {format_duration(result.ingestion_time)}"
            )

            # Wait for deriver queue
            if self.wait_for_deriver:
                logger.info(f"[{question_id}] Waiting for deriver queue to empty...")
                deriver_start = time.time()
                await asyncio.sleep(1)
                queue_empty = await self.wait_for_deriver_queue_empty(
                    honcho_client, session_id
                )
                result.deriver_wait_time = time.time() - deriver_start

                if not queue_empty:
                    result.error = f"Deriver queue timeout after {self.timeout_seconds}s"
                    logger.warning(f"[{question_id}] {result.error}")
                else:
                    logger.info(
                        f"[{question_id}] Deriver queue empty after {format_duration(result.deriver_wait_time)}"
                    )

            # Cleanup workspace if requested
            if self.cleanup_workspace:
                logger.info(f"[{question_id}] Cleaning up workspace...")
                try:
                    await honcho_client.delete_workspace()
                except Exception as e:
                    logger.warning(
                        f"[{question_id}] Failed to cleanup workspace: {e}"
                    )

        except Exception as e:
            result.error = str(e)
            logger.error(f"[{question_id}] Error: {e}", exc_info=True)

        result.end_time = time.time()
        result.total_time = result.end_time - result.start_time
        return result

    async def ingest_beam_trace(
        self,
        conversation_data: dict[str, Any],
        conversation_id: str,
        context_length: str,
        honcho_url: str,
        index: int,
    ) -> TraceResult:
        """
        Ingest a BEAM conversation trace.

        Args:
            conversation_data: Dictionary containing conversation data
            conversation_id: Conversation ID
            context_length: Context length (100K, 500K, 1M, 10M)
            honcho_url: URL of the Honcho instance to use
            index: Index for generating unique IDs

        Returns:
            TraceResult with ingestion metrics
        """
        start_time = time.time()
        trace_id = f"{context_length}_{conversation_id}"

        result = TraceResult(trace_id=trace_id, dataset="beam")
        result.start_time = start_time

        logger.info(f"[{trace_id}] Ingesting BEAM trace")

        # Create workspace
        workspace_id = f"beam_trace_{trace_id}_{int(start_time)}"
        result.workspace_id = workspace_id
        honcho_client = await self.create_honcho_client(workspace_id, honcho_url)

        try:
            # Create peers (BEAM has user and assistant)
            user_peer = await honcho_client.peer(id="user")
            assistant_peer = await honcho_client.peer(id="assistant")

            # Create session
            session_id = f"{workspace_id}_session"
            session = await honcho_client.session(id=session_id)

            # Configure peer observation
            await session.add_peers(
                [
                    (
                        user_peer,
                        SessionPeerConfig(observe_me=True, observe_others=False),
                    ),
                    (
                        assistant_peer,
                        SessionPeerConfig(observe_me=True, observe_others=False),
                    ),
                ]
            )

            # Extract messages
            chat_data = conversation_data.get("chat", [])
            extracted_messages = extract_messages_from_chat_data(chat_data)
            result.total_sessions = 1  # BEAM conversations are single session

            messages: list[MessageCreateParam] = []
            total_tokens = 0

            for msg in extracted_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                total_tokens += calculate_tokens(content)

                if role == "user":
                    messages.append(user_peer.message(content))
                elif role == "assistant":
                    messages.append(assistant_peer.message(content))

            result.total_messages = len(messages)
            result.total_tokens = total_tokens

            # Ingest messages in batches of 100
            ingestion_start = time.time()
            for i in range(0, len(messages), 100):
                batch = messages[i : i + 100]
                await session.add_messages(batch)
                logger.info(
                    f"[{trace_id}] Ingested batch {i // 100 + 1}/{(len(messages) + 99) // 100}"
                )

            result.ingestion_time = time.time() - ingestion_start

            logger.info(
                f"[{trace_id}] Ingested {len(messages)} messages (~{total_tokens:,} tokens) in {format_duration(result.ingestion_time)}"
            )

            # Wait for deriver queue
            if self.wait_for_deriver:
                logger.info(f"[{trace_id}] Waiting for deriver queue to empty...")
                deriver_start = time.time()
                await asyncio.sleep(1)
                queue_empty = await self.wait_for_deriver_queue_empty(
                    honcho_client, session_id
                )
                result.deriver_wait_time = time.time() - deriver_start

                if not queue_empty:
                    result.error = f"Deriver queue timeout after {self.timeout_seconds}s"
                    logger.warning(f"[{trace_id}] {result.error}")
                else:
                    logger.info(
                        f"[{trace_id}] Deriver queue empty after {format_duration(result.deriver_wait_time)}"
                    )

            # Cleanup workspace if requested
            if self.cleanup_workspace:
                logger.info(f"[{trace_id}] Cleaning up workspace...")
                try:
                    await honcho_client.delete_workspace()
                except Exception as e:
                    logger.warning(f"[{trace_id}] Failed to cleanup workspace: {e}")

        except Exception as e:
            result.error = str(e)
            logger.error(f"[{trace_id}] Error: {e}", exc_info=True)

        result.end_time = time.time()
        result.total_time = result.end_time - result.start_time
        return result


async def run_locomo_traces(
    runner: TraceRunner,
    data_file: Path,
    batch_size: int = 1,
    sample_limit: int | None = None,
) -> list[TraceResult]:
    """Run LoCoMo trace ingestion."""
    logger.info(f"Loading LoCoMo data from {data_file}")
    conversations = load_locomo_data(data_file)

    if sample_limit:
        conversations = conversations[:sample_limit]

    logger.info(f"Processing {len(conversations)} LoCoMo conversations")

    results: list[TraceResult] = []
    for i in range(0, len(conversations), batch_size):
        batch = conversations[i : i + batch_size]
        batch_tasks = [
            runner.ingest_locomo_trace(
                conv,
                runner.get_honcho_url_for_index(i + j),
                i + j,
            )
            for j, conv in enumerate(batch)
        ]
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)

    return results


async def run_longmem_traces(
    runner: TraceRunner,
    data_path: Path,
    test_file: str | None = None,
    batch_size: int = 1,
    sample_limit: int | None = None,
) -> list[TraceResult]:
    """Run LongMemEval trace ingestion.

    Args:
        runner: TraceRunner instance
        data_path: Path to data file or directory
        test_file: Test file name (only used if data_path is a directory)
        batch_size: Number of traces to process concurrently
        sample_limit: Optional limit on number of traces
    """
    # Determine the actual file to load
    if data_path.is_file():
        test_path = data_path
    elif test_file:
        test_path = data_path / test_file
    else:
        raise ValueError("test_file is required when data_path is a directory")

    logger.info(f"Loading LongMemEval data from {test_path}")
    test_data = load_test_file(test_path)

    if sample_limit:
        test_data = test_data[:sample_limit]

    logger.info(f"Processing {len(test_data)} LongMemEval traces")

    results: list[TraceResult] = []
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i : i + batch_size]
        batch_tasks = [
            runner.ingest_longmem_trace(
                test,
                runner.get_honcho_url_for_index(i + j),
                i + j,
            )
            for j, test in enumerate(batch)
        ]
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)

    return results


async def run_beam_traces(
    runner: TraceRunner,
    data_dir: Path,
    context_length: str,
    conversation_ids: list[str] | None = None,
    batch_size: int = 1,
    sample_limit: int | None = None,
) -> list[TraceResult]:
    """Run BEAM trace ingestion."""
    logger.info(f"Loading BEAM conversations from {data_dir}/{context_length}")

    if conversation_ids:
        conv_ids = conversation_ids
    else:
        conv_ids = list_beam_conversations(data_dir, context_length)

    if sample_limit:
        conv_ids = conv_ids[:sample_limit]

    logger.info(f"Processing {len(conv_ids)} BEAM conversations")

    results: list[TraceResult] = []
    for i in range(0, len(conv_ids), batch_size):
        batch = conv_ids[i : i + batch_size]
        batch_tasks = []
        for j, conv_id in enumerate(batch):
            conv_data = load_beam_conversation(data_dir, context_length, conv_id)
            batch_tasks.append(
                runner.ingest_beam_trace(
                    conv_data,
                    conv_id,
                    context_length,
                    runner.get_honcho_url_for_index(i + j),
                    i + j,
                )
            )
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)

    return results


def print_summary(results: list[TraceResult], total_time: float) -> None:
    """Print summary of trace ingestion results."""
    print(f"\n{'=' * 80}")
    print("TRACE INGESTION SUMMARY")
    print(f"{'=' * 80}")

    successful = [r for r in results if not r.error]
    failed = [r for r in results if r.error]

    print(f"Total Traces: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total Time: {format_duration(total_time)}")

    if successful:
        total_messages = sum(r.total_messages for r in successful)
        total_tokens = sum(r.total_tokens for r in successful)
        avg_ingestion = sum(r.ingestion_time for r in successful) / len(successful)
        avg_deriver = sum(r.deriver_wait_time for r in successful) / len(successful)

        print(f"\nTotal Messages Ingested: {total_messages:,}")
        print(f"Total Tokens: {total_tokens:,}")
        print(f"Average Ingestion Time: {format_duration(avg_ingestion)}")
        print(f"Average Deriver Wait Time: {format_duration(avg_deriver)}")

    print(f"{'=' * 80}")


def save_results(results: list[TraceResult], output_file: Path, total_time: float) -> None:
    """Save results to JSON file."""
    summary = {
        "metadata": {
            "execution_timestamp": datetime.now().isoformat(),
            "total_traces": len(results),
            "successful": len([r for r in results if not r.error]),
            "failed": len([r for r in results if r.error]),
            "total_time_seconds": total_time,
        },
        "results": [r.to_dict() for r in results],
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {output_file}")


def detect_dataset_type(data_path: Path) -> tuple[str, dict[str, Any]]:
    """
    Auto-detect the dataset type based on file/directory structure.

    Args:
        data_path: Path to data file or directory

    Returns:
        Tuple of (dataset_type, info_dict) where info_dict contains relevant paths/metadata

    Raises:
        ValueError: If dataset type cannot be determined
    """
    # Check if it's a file
    if data_path.is_file() and data_path.suffix == ".json":
        # Check if it's a LongMemEval file
        if data_path.name.startswith("longmemeval_"):
            logger.info(f"Detected LongMemEval dataset (single file): {data_path}")
            return "longmem", {"data_file": data_path}
        # Otherwise assume LoCoMo
        logger.info(f"Detected LoCoMo dataset (single JSON file): {data_path}")
        return "locomo", {"data_file": data_path}

    # Check if it's a directory
    if not data_path.is_dir():
        raise ValueError(f"Path does not exist or is not a file/directory: {data_path}")

    # Check for BEAM structure (subdirectories named 100K, 500K, 1M, 10M)
    beam_contexts = ["100K", "500K", "1M", "10M"]
    has_beam_structure = any(
        (data_path / context).is_dir() for context in beam_contexts
    )
    if has_beam_structure:
        logger.info(f"Detected BEAM dataset (context subdirectories): {data_path}")
        return "beam", {"data_dir": data_path}

    # Check for LongMemEval structure (longmemeval_*.json or qa_*.json files)
    longmem_files = list(data_path.glob("longmemeval_*.json")) + list(data_path.glob("qa_*.json"))
    if longmem_files:
        logger.info(
            f"Detected LongMemEval dataset ({len(longmem_files)} data files): {data_path}"
        )
        return "longmem", {"data_dir": data_path}

    raise ValueError(
        f"Could not detect dataset type from path: {data_path}\n"
        "Expected:\n"
        "  - LoCoMo: Single .json file (not starting with 'longmemeval_')\n"
        "  - BEAM: Directory with subdirectories named 100K, 500K, 1M, or 10M\n"
        "  - LongMemEval: Single longmemeval_*.json file OR directory with longmemeval_*.json or qa_*.json files"
    )


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest conversation traces into Honcho",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # LoCoMo dataset (auto-detected from .json file)
  %(prog)s --data-dir tests/bench/locomo_data/locomo10.json

  # LongMemEval dataset - single file
  %(prog)s --data-dir tests/bench/longmemeval_data/longmemeval_single.json

  # LongMemEval dataset - directory mode
  %(prog)s --data-dir tests/bench/longmemeval_data --test-file longmemeval_single.json

  # BEAM dataset (auto-detected from 100K/500K/1M/10M subdirs)
  %(prog)s --data-dir tests/bench/beam_data --context-length 100K --conversation-id 0
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to data file or directory (auto-detects dataset type)",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        help="Test file name (for LongMemEval, e.g., qa_single_session_user.json)",
    )
    parser.add_argument(
        "--context-length",
        type=str,
        choices=["100K", "500K", "1M", "10M"],
        help="Context length (for BEAM dataset)",
    )
    parser.add_argument(
        "--conversation-id", type=str, help="Specific conversation ID (for BEAM)"
    )
    parser.add_argument(
        "--base-api-port", type=int, default=8000, help="Base port for Honcho API"
    )
    parser.add_argument(
        "--pool-size", type=int, default=1, help="Number of Honcho instances"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of traces to process concurrently",
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="Timeout for deriver queue (seconds)"
    )
    parser.add_argument(
        "--cleanup-workspace",
        action="store_true",
        help="Delete workspace after ingestion",
    )
    parser.add_argument(
        "--no-wait-deriver",
        action="store_true",
        help="Don't wait for deriver queue to empty",
    )
    parser.add_argument(
        "--trigger-dream",
        action="store_true",
        help="Trigger dream consolidation after ingestion",
    )
    parser.add_argument(
        "--json-output", type=Path, help="Path to write JSON summary results"
    )
    parser.add_argument(
        "--sample-limit", type=int, help="Limit number of traces to process"
    )

    args = parser.parse_args()

    # Auto-detect dataset type
    try:
        dataset_type, dataset_info = detect_dataset_type(args.data_dir)
    except ValueError as e:
        parser.error(str(e))
        return

    # Validate dataset-specific arguments
    if dataset_type == "longmem":
        # If data_file is provided, test_file is not needed (single file mode)
        if "data_file" in dataset_info:
            if args.test_file:
                logger.warning("--test-file ignored when using single file mode")
        # If data_dir is provided, test_file is required (directory mode)
        elif "data_dir" in dataset_info and not args.test_file:
            data_dir = dataset_info["data_dir"]
            test_files = sorted([f.name for f in data_dir.glob("longmemeval_*.json")] +
                               [f.name for f in data_dir.glob("qa_*.json")])
            parser.error(
                f"--test-file is required for LongMemEval directory mode.\n"
                f"Available test files in {data_dir}:\n"
                + "\n".join(f"  - {tf}" for tf in test_files)
            )

    if dataset_type == "beam" and not args.context_length:
        # List available context lengths
        data_dir = dataset_info["data_dir"]
        contexts = [d.name for d in data_dir.iterdir() if d.is_dir() and d.name in ["100K", "500K", "1M", "10M"]]
        parser.error(
            f"--context-length is required for BEAM dataset.\n"
            f"Available context lengths in {data_dir}:\n"
            + "\n".join(f"  - {c}" for c in sorted(contexts))
        )

    # Create runner
    runner = TraceRunner(
        base_api_port=args.base_api_port,
        pool_size=args.pool_size,
        timeout_seconds=args.timeout,
        cleanup_workspace=args.cleanup_workspace,
        wait_for_deriver=not args.no_wait_deriver,
        trigger_dream=args.trigger_dream,
    )

    # Run trace ingestion
    start_time = time.time()
    results: list[TraceResult] = []

    try:
        if dataset_type == "locomo":
            data_file = dataset_info["data_file"]
            results = await run_locomo_traces(
                runner, data_file, args.batch_size, args.sample_limit
            )
        elif dataset_type == "longmem":
            # Handle both single file and directory modes
            data_path = dataset_info.get("data_file") or dataset_info.get("data_dir")
            if data_path is None:
                raise ValueError("No data_file or data_dir in dataset_info")
            results = await run_longmem_traces(
                runner,
                data_path,
                args.test_file,
                args.batch_size,
                args.sample_limit,
            )
        elif dataset_type == "beam":
            data_dir = dataset_info["data_dir"]
            conv_ids = [args.conversation_id] if args.conversation_id else None
            results = await run_beam_traces(
                runner,
                data_dir,
                args.context_length,
                conv_ids,
                args.batch_size,
                args.sample_limit,
            )
    except Exception as e:
        logger.error(f"Error running trace ingestion: {e}", exc_info=True)
        return

    total_time = time.time() - start_time

    # Print summary
    print_summary(results, total_time)

    # Save results if requested
    if args.json_output:
        save_results(results, args.json_output, total_time)


if __name__ == "__main__":
    asyncio.run(main())
