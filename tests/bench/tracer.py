"""
Honcho Trace Generator

A script that processes conversation data and generates trace files by sending
messages through Honcho's representation system. This captures the theory-of-mind
processing traces without evaluation.

This script:
1. Loads conversation data from JSON files
2. Creates a workspace and session for each conversation
3. Processes messages sequentially (preserving context dependencies)
4. Generates trace data showing the representation building process

## To use

0. Set up env:
```
uv sync
source .venv/bin/activate
```

**IMPORTANT**: Ensure peer cards are enabled in your `.env` file:
```
PEER_CARD_ENABLED=true
```
This is required for peer card biographical information to be captured in traces.

1. Run the test harness to start a Honcho instance:
```
python -m tests.bench.harness
```

2. Run this script with a conversation data file:
```
python -m tests.bench.tracer --data-file tests/bench/longmemeval_data/test_data.json
```

Optional arguments:
```
--data-file: Path to conversation data JSON file (required)
--base-api-port: Port for Honcho API instance (default: 8000)
--timeout: Timeout for deriver queue to empty in seconds (default: 10 minutes)
--cleanup-workspace: Delete workspace after processing (default: False)
--limit: Limit number of conversations to process (default: process all)
--concurrency: Number of conversations to process in parallel (default: 1)
--output-json: Path for final JSON trace file (default: trace.json)
--output-jsonl: Path for intermediate JSONL trace file (default: auto-derived from output-json)
```

**Note on Output Files**: The JSONL file is automatically derived from the JSON filename.
For example, `--output-json results/my_trace.json` will automatically use
`results/my_trace.jsonl`. The trace file path is embedded in each message's metadata,
so the deriver knows exactly where to write traces. This ensures that:
- Parallel tracer instances write to separate files without conflicts
- The intermediate JSONL file is never written to the default "trace.jsonl"
- Each tracer run produces uniquely named output files
- Multiple tracer scripts can run simultaneously without interfering with each other

**Note on Concurrency**: Setting `--concurrency` to a value greater than 1 will process
multiple conversations in parallel to speed up processing. Each conversation's messages
are still processed sequentially to maintain context dependencies.

## Input Data Format

The input JSON should be a dictionary where keys are conversation IDs and values contain:
- dataset: Name of the dataset (string)
- peers: List of peer IDs (list of strings)
  - **Note**: Peer IDs with whitespaces or special characters will be automatically
    sanitized (e.g., "user 1" becomes "user_1"). The original IDs are preserved in
    the display, but the sanitized versions are used internally.
- messages: List of message objects with:
  - timestamp: Message timestamp (string in "YYYY-MM-DD HH:MM:SS" format)
  - peer: Peer ID who sent the message (string, can contain whitespaces)
  - content: Message content (string)

Example:
```json
{
    "conversation-1": {
        "dataset": "dailydialog",
        "peers": ["user_1", "assistant_1"],
        "messages": [
            {
                "timestamp": "2023-01-01 00:00:00",
                "peer": "user_1",
                "content": "Hello!"
            },
            {
                "timestamp": "2023-01-01 00:01:00",
                "peer": "assistant_1",
                "content": "Hi there!"
            }
        ]
    }
}
```

## Output

The script generates a trace.json file containing the theory-of-mind processing
traces for each message, including:
- Input representation state
- Message history
- New conversation turns
- Generated prompt
- LLM output (extracted facts)
- Metadata (conversation ID, message sequence, etc.)
"""

import argparse
import asyncio
import json
import logging
import os
import re
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
from tqdm import tqdm
from typing_extensions import TypedDict

from src.config import settings
from src.utils.metrics_collector import convert_trace_to_json

load_dotenv()


def sanitize_peer_id(peer_id: str) -> str:
    """
    Sanitize peer ID to ensure it's safe for use in the system.

    Replaces whitespaces and special characters with underscores,
    removes consecutive underscores, and ensures the ID is not empty.

    Args:
        peer_id: Original peer ID that may contain whitespaces or special characters

    Returns:
        Sanitized peer ID safe for use in the system

    Examples:
        "user 1" -> "user_1"
        "assistant  2" -> "assistant_2"
        "peer@name!" -> "peer_name"
        "   spaces   " -> "spaces"
    """
    # Replace whitespaces and special characters with underscores
    sanitized = re.sub(r'[^\w\-]', '_', peer_id)

    # Replace multiple consecutive underscores with a single one
    sanitized = re.sub(r'_+', '_', sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')

    # Ensure the ID is not empty
    if not sanitized:
        sanitized = "peer"

    return sanitized


class ConversationData(TypedDict):
    """Type definition for conversation data."""

    dataset: str
    peers: list[str]
    messages: list[dict[str, Any]]


class ProcessingResult(TypedDict):
    """Type definition for conversation processing results."""

    conversation_id: str
    dataset: str
    message_count: int
    peer_count: int
    success: bool
    error: str | None
    duration_seconds: float


class TraceGenerator:
    """
    Processes conversation data through Honcho to generate theory-of-mind traces.
    """

    def __init__(
        self,
        api_port: int = 8000,
        timeout_seconds: int | None = None,
        cleanup_workspace: bool = False,
        output_jsonl: Path | None = None,
        output_json: Path | None = None,
        concurrency: int = 1,
    ):
        """
        Initialize the trace generator.

        Args:
            api_port: Port for Honcho API instance (default: 8000)
            timeout_seconds: Timeout for deriver queue in seconds
            cleanup_workspace: If True, delete workspace after processing
            output_jsonl: Path for intermediate JSONL trace file (auto-derived from output_json if not specified)
            output_json: Path for final JSON trace file (default: trace.json)
            concurrency: Number of conversations to process in parallel (default: 1)
        """
        self.api_port: int = api_port
        self.timeout_seconds: int = (
            timeout_seconds if timeout_seconds is not None else 600
        )
        self.cleanup_workspace: bool = cleanup_workspace
        self.output_json: Path = output_json or Path("trace.json")

        # Auto-derive JSONL filename from JSON filename if not explicitly provided
        if output_jsonl is None:
            # Replace .json extension with .jsonl (or append .jsonl if no extension)
            if self.output_json.suffix == ".json":
                self.output_jsonl = self.output_json.with_suffix(".jsonl")
            else:
                self.output_jsonl = Path(str(self.output_json) + ".jsonl")
        else:
            self.output_jsonl = output_jsonl

        self.concurrency: int = concurrency

        # Ensure parent directories exist
        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.output_json.parent.mkdir(parents=True, exist_ok=True)

        # Set the environment variable so the deriver uses our custom trace file path
        os.environ["LOCAL_TRACE_FILE"] = str(self.output_jsonl)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Suppress HTTP request logs from the Honcho SDK
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)

    def get_honcho_url(self) -> str:
        """
        Get the Honcho URL.

        Returns:
            URL of the Honcho instance
        """
        return f"http://localhost:{self.api_port}"

    def _format_duration(self, total_seconds: float) -> str:
        """Format a duration in seconds into a human-readable string.

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

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime.

        Args:
            timestamp_str: Timestamp string in format "YYYY-MM-DD HH:MM:SS"

        Returns:
            Parsed datetime object

        Raises:
            ValueError: If timestamp format is invalid
        """
        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            raise ValueError(
                f"Failed to parse timestamp '{timestamp_str}': {e}"
            ) from e

    def load_conversation_file(self, data_file: Path) -> dict[str, ConversationData]:
        """
        Load conversation data from a JSON file.

        Args:
            data_file: Path to the JSON data file

        Returns:
            Dictionary mapping conversation IDs to conversation data
        """
        with open(data_file) as f:
            return json.load(f)

    async def create_honcho_client(self, workspace_id: str) -> AsyncHoncho:
        """
        Create a Honcho client for a specific workspace.

        Args:
            workspace_id: Workspace ID

        Returns:
            AsyncHoncho client instance
        """
        return AsyncHoncho(
            environment="local",
            workspace_id=workspace_id,
            base_url=self.get_honcho_url(),
        )

    async def wait_for_deriver_queue_empty(
        self, honcho_client: AsyncHoncho, session_id: str | None = None
    ) -> bool:
        """Wait for the deriver queue to empty.

        Args:
            honcho_client: Honcho client instance
            session_id: Optional session ID to check status for

        Returns:
            True if queue emptied successfully, False if timeout
        """
        start_time = time.time()
        while True:
            try:
                status = await honcho_client.get_deriver_status(session_id=session_id)
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

    async def process_conversation(
        self, conversation_id: str, conversation_data: ConversationData
    ) -> ProcessingResult:
        """
        Process a single conversation through Honcho to generate traces.

        Args:
            conversation_id: Unique identifier for the conversation
            conversation_data: Conversation data containing messages and metadata

        Returns:
            Processing result
        """
        start_time = time.time()

        dataset = conversation_data["dataset"]
        peers = conversation_data["peers"]
        messages = conversation_data["messages"]

        self.logger.info(
            f"Processing conversation {conversation_id} from dataset {dataset}"
        )
        self.logger.info(f"  Peers: {peers}")
        self.logger.info(f"  Messages: {len(messages)}")

        # Create workspace for this conversation
        workspace_id = f"trace_{dataset}_{conversation_id}"
        honcho_client = await self.create_honcho_client(workspace_id)

        result: ProcessingResult = {
            "conversation_id": conversation_id,
            "dataset": dataset,
            "message_count": len(messages),
            "peer_count": len(peers),
            "success": False,
            "error": None,
            "duration_seconds": 0.0,
        }

        try:
            # Create session for this conversation
            session_id = f"{conversation_id}_session"
            session = await honcho_client.session(id=session_id)

            # Create peer objects for all participants
            # Sanitize peer IDs to handle whitespaces and special characters
            peer_objects = {}
            peer_id_mapping = {}  # Map original peer_id -> sanitized peer_id

            for original_peer_id in peers:
                sanitized_peer_id = sanitize_peer_id(original_peer_id)
                peer_id_mapping[original_peer_id] = sanitized_peer_id

                # Log if sanitization changed the ID
                if original_peer_id != sanitized_peer_id:
                    self.logger.info(
                        f"  Sanitized peer ID: '{original_peer_id}' -> '{sanitized_peer_id}'"
                    )

                peer_obj = await honcho_client.peer(id=sanitized_peer_id)
                peer_objects[original_peer_id] = peer_obj

            # Add all peers to the session with observation enabled
            # All peers observe themselves by default
            peer_configs = [
                (
                    peer_obj,
                    SessionPeerConfig(observe_me=True, observe_others=False),
                )
                for peer_obj in peer_objects.values()
            ]
            await session.add_peers(peer_configs)

            # Process messages sequentially to maintain context dependencies
            self.logger.info(f"Processing {len(messages)} messages sequentially...")

            # Create progress bar for message processing
            progress_bar = tqdm(
                total=len(messages),
                desc=f"[{conversation_id}] Processing messages",
                unit="msg",
                leave=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )

            try:
                for msg_idx, msg_data in enumerate(messages):
                    peer_id = msg_data["peer"]
                    content = msg_data["content"]
                    timestamp_str = msg_data["timestamp"]

                    # Parse timestamp
                    try:
                        timestamp = self._parse_timestamp(timestamp_str)
                    except ValueError as e:
                        progress_bar.close()
                        self.logger.error(f"Error parsing timestamp: {e}")
                        result["error"] = str(e)
                        return result

                    # Get the peer object
                    peer_obj = peer_objects.get(peer_id)
                    if not peer_obj:
                        progress_bar.close()
                        error_msg = f"Unknown peer ID: {peer_id}"
                        self.logger.error(error_msg)
                        result["error"] = error_msg
                        return result

                    # Create message with metadata for trace tracking
                    # Include trace file path so deriver knows where to write traces
                    message_metadata = {
                        "dataset_uuid": f"{dataset}_{conversation_id}",
                        "conversation_id": conversation_id,
                        "message_sequence_id": msg_idx,
                        "total_messages": len(messages),
                        "dataset": dataset,
                        "trace_file_path": str(self.output_jsonl),  # Tell deriver where to write traces
                    }

                    # Add the message
                    message_param = peer_obj.message(
                        content, created_at=timestamp, metadata=message_metadata
                    )

                    await session.add_messages([message_param])

                    # Update progress bar description with current peer
                    # Show sanitized peer ID if it was changed
                    display_peer_id = peer_id_mapping.get(peer_id, peer_id)
                    if peer_id != display_peer_id:
                        progress_bar.set_description(
                            f"[{conversation_id}] Processing {peer_id} ({display_peer_id})"
                        )
                    else:
                        progress_bar.set_description(
                            f"[{conversation_id}] Processing {peer_id}"
                        )

                    # Wait for deriver to process this message before continuing
                    # This ensures sequential processing and proper context building
                    self.logger.debug(
                        f"  Waiting for deriver to process message {msg_idx + 1}/{len(messages)}..."
                    )

                    # Give a short delay to allow the message to be enqueued
                    await asyncio.sleep(0.5)

                    queue_empty = await self.wait_for_deriver_queue_empty(
                        honcho_client, session_id=session_id
                    )

                    if not queue_empty:
                        progress_bar.close()
                        error_msg = f"Deriver queue timeout on message {msg_idx + 1}"
                        self.logger.error(error_msg)
                        result["error"] = error_msg
                        return result

                    # Update progress
                    progress_bar.update(1)

                    self.logger.debug(
                        f"  Message {msg_idx + 1}/{len(messages)} processed successfully"
                    )

                # Close the progress bar
                progress_bar.close()
            except Exception as e:
                progress_bar.close()
                raise

            # Final wait to ensure all processing is complete
            self.logger.info("Waiting for final deriver queue to empty...")
            await asyncio.sleep(1)
            queue_empty = await self.wait_for_deriver_queue_empty(
                honcho_client, session_id=session_id
            )

            if not queue_empty:
                result["error"] = "Final deriver queue timeout"
                return result

            # Clean up workspace if requested
            if self.cleanup_workspace:
                try:
                    await honcho_client.delete_workspace(workspace_id)
                    self.logger.info(f"Cleaned up workspace {workspace_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete workspace: {e}")

            result["success"] = True
            result["duration_seconds"] = time.time() - start_time

            self.logger.info(
                f"Conversation {conversation_id} completed successfully "
                f"(Duration: {self._format_duration(result['duration_seconds'])})"
            )

        except Exception as e:
            self.logger.error(f"Error processing conversation {conversation_id}: {e}")
            result["error"] = str(e)
            result["success"] = False
            result["duration_seconds"] = time.time() - start_time

        return result

    async def process_all_conversations(
        self, data_file: Path, limit: int | None = None
    ) -> tuple[list[ProcessingResult], float]:
        """
        Process all conversations in a data file with optional parallelism.

        Args:
            data_file: Path to the conversation data JSON file
            limit: Optional limit on number of conversations to process

        Returns:
            Tuple of (list of processing results, total duration)
        """
        conversations = self.load_conversation_file(data_file)

        conversation_ids = list(conversations.keys())
        if limit:
            conversation_ids = conversation_ids[:limit]

        self.logger.info(
            f"Found {len(conversation_ids)} conversations to process in {data_file}"
        )
        if self.concurrency > 1:
            self.logger.info(f"Processing with concurrency level: {self.concurrency}")

        overall_start = time.time()

        # Process conversations with controlled concurrency
        # Messages within each conversation are still processed sequentially
        all_results: list[ProcessingResult] = []

        if self.concurrency == 1:
            # Sequential processing (original behavior)
            for idx, conv_id in enumerate(conversation_ids):
                self.logger.info(
                    f"\n{'=' * 80}\n"
                    f"Processing conversation {idx + 1}/{len(conversation_ids)}: {conv_id}\n"
                    f"{'=' * 80}"
                )

                result = await self.process_conversation(conv_id, conversations[conv_id])
                all_results.append(result)

                # Print result summary
                status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
                self.logger.info(
                    f"{status} - {conv_id} "
                    f"({result['message_count']} messages, "
                    f"{self._format_duration(result['duration_seconds'])})"
                )

                if result["error"]:
                    self.logger.error(f"  Error: {result['error']}")
        else:
            # Parallel processing with semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.concurrency)

            async def process_with_semaphore(
                conv_id: str, conv_data: ConversationData, idx: int
            ) -> ProcessingResult:
                async with semaphore:
                    self.logger.info(
                        f"\n{'=' * 80}\n"
                        f"Processing conversation {idx + 1}/{len(conversation_ids)}: {conv_id}\n"
                        f"{'=' * 80}"
                    )

                    result = await self.process_conversation(conv_id, conv_data)

                    # Print result summary
                    status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
                    self.logger.info(
                        f"{status} - {conv_id} "
                        f"({result['message_count']} messages, "
                        f"{self._format_duration(result['duration_seconds'])})"
                    )

                    if result["error"]:
                        self.logger.error(f"  Error: {result['error']}")

                    return result

            # Create tasks for all conversations
            tasks = [
                process_with_semaphore(conv_id, conversations[conv_id], idx)
                for idx, conv_id in enumerate(conversation_ids)
            ]

            # Run tasks concurrently
            all_results = await asyncio.gather(*tasks)

        overall_end = time.time()
        overall_duration = overall_end - overall_start

        # Convert trace file to JSON format
        self.logger.info(
            f"Converting trace file from {self.output_jsonl} to {self.output_json}..."
        )
        convert_trace_to_json(
            input_file=str(self.output_jsonl), output_file=str(self.output_json)
        )

        return all_results, overall_duration

    def print_summary(
        self, results: list[ProcessingResult], total_elapsed_seconds: float
    ) -> None:
        """
        Print a summary of all processing results.

        Args:
            results: List of processing results
            total_elapsed_seconds: Total elapsed time
        """
        print(f"\n{'=' * 80}")
        print("TRACE GENERATION SUMMARY")
        print(f"{'=' * 80}")

        total_conversations = len(results)
        successful = sum(1 for r in results if r["success"])
        failed = total_conversations - successful
        total_messages = sum(r["message_count"] for r in results)

        print(f"Total Conversations: {total_conversations}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(
            f"Success Rate: {(successful / total_conversations) * 100:.1f}%"
            if total_conversations > 0
            else "N/A"
        )
        print(f"Total Messages Processed: {total_messages}")
        print(f"Total Processing Time: {self._format_duration(total_elapsed_seconds)}")
        print(f"Concurrency Level: {self.concurrency}")

        if successful > 0:
            avg_duration = sum(
                r["duration_seconds"] for r in results if r["success"]
            ) / successful
            print(f"Average Duration per Conversation: {self._format_duration(avg_duration)}")

        print("\nDetailed Results:")
        print(
            f"{'Conversation ID':<30} {'Dataset':<20} {'Messages':<10} {'Status':<10} {'Duration':<12}"
        )
        print(f"{'-' * 30} {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 12}")

        for result in results:
            conv_id = result["conversation_id"]
            dataset = result["dataset"]
            msg_count = result["message_count"]
            status = "SUCCESS" if result["success"] else "FAILED"
            duration = self._format_duration(result["duration_seconds"])

            print(
                f"{conv_id:<30} {dataset:<20} {msg_count:<10} {status:<10} {duration:<12}"
            )

        print(f"{'=' * 80}")

        # Check if trace file was created
        if self.output_json.exists():
            print(f"\n✓ Trace file created: {self.output_json}")
            print(f"  File size: {self.output_json.stat().st_size / 1024:.1f} KB")
        else:
            print(f"\n✗ Warning: {self.output_json} file was not created")


async def main() -> int:
    """
    Main entry point for the trace generator.
    """
    parser = argparse.ArgumentParser(
        description="Generate theory-of-mind traces from conversation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data-file tests/bench/longmemeval_data/test_data.json    # Process all conversations
  %(prog)s --data-file data.json --limit 10                            # Process first 10 conversations
  %(prog)s --data-file data.json --concurrency 5                       # Process 5 conversations in parallel
  %(prog)s --data-file data.json --cleanup-workspace                   # Clean up after processing
  %(prog)s --data-file data.json --output-json results/my_trace.json  # Auto-uses results/my_trace.jsonl
  %(prog)s --data-file data.json --output-json out.json --output-jsonl custom.jsonl  # Override JSONL path
        """,
    )

    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help="Path to conversation data JSON file (required)",
    )

    parser.add_argument(
        "--base-api-port",
        type=int,
        default=8000,
        help="Port for Honcho API instance (default: 8000)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout for deriver queue to empty in seconds (default: 10 minutes)",
    )

    parser.add_argument(
        "--cleanup-workspace",
        action="store_true",
        help="Delete workspace after processing (default: False)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of conversations to process (default: process all)",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of conversations to process in parallel (default: 1)",
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path for final JSON trace file (default: trace.json)",
    )

    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Path for intermediate JSONL trace file (default: auto-derived from output-json)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.data_file.exists():
        print(f"Error: Data file {args.data_file} does not exist")
        return 1

    if args.limit is not None and args.limit <= 0:
        print(f"Error: Limit must be positive, got {args.limit}")
        return 1

    if args.concurrency <= 0:
        print(f"Error: Concurrency must be positive, got {args.concurrency}")
        return 1

    # Create trace generator
    generator = TraceGenerator(
        api_port=args.base_api_port,
        timeout_seconds=args.timeout,
        cleanup_workspace=args.cleanup_workspace,
        output_jsonl=args.output_jsonl,
        output_json=args.output_json,
        concurrency=args.concurrency,
    )

    try:
        # Process all conversations
        results, total_elapsed = await generator.process_all_conversations(
            args.data_file, args.limit
        )
        generator.print_summary(results, total_elapsed_seconds=total_elapsed)

        # Return exit code based on results
        all_successful = all(r["success"] for r in results)
        return 0 if all_successful else 1

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        import traceback

        print(f"Error processing conversations: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
