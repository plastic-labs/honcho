"""
Honcho Trace Data Collector

A script that collects trace data from conversations processed through Honcho.
This script:
1. Loads conversation data from JSON files
2. Creates a workspace and sessions with the conversation messages
3. Processes messages incrementally (one turn at a time)
4. Waits for the deriver queue after each turn to capture traces
5. Outputs trace data showing the theory of mind analysis evolution at each conversation step

## Input Format

The input JSON file should contain conversation data in the following format:
```json
{
    "conversation_id": {
        "dataset": "dataset_name",
        "peers": ["peer1_id", "peer2_id"],
        "messages": [
            {
                "timestamp": "2023-01-01 00:00:00",
                "peer": "peer1_id",
                "content": "message content"
            },
            ...
        ]
    },
    ...
}
```

## To use

0. Set up env:
```
uv sync
source .venv/bin/activate
```

1. Run the tracer script:
```
python tracer.py --conversation-file path/to/conversation_data.json
```

Optional arguments:
```
--base-api-port: Base port for Honcho API instances (default: 8000)
--pool-size: Number of Honcho instances in the pool (default: 1)
--batch-size: Number of conversations to process concurrently in each batch (default: 10)
--timeout: Timeout for deriver queue to empty in seconds (default: 10 minutes)
--workspace-id-prefix: Prefix for workspace IDs (default: auto-generated)
--cleanup-workspace: Delete workspace after processing (default: False)
```

## Output

The script produces a trace.json file containing trace data at each conversation turn:
- Timestamp and model configuration
- **Dataset UUID**: Unique identifier for this conversation/dataset
- **Conversation ID**: Original conversation identifier from the input file
- **Message Sequence ID**: Sequential number (1 to N) indicating the order of this message
- **Total Messages**: Total number of messages in the conversation
- **Dataset**: Name of the source dataset
- Peer information
- **Working representation input**: The representation passed into the analysis (explicit and deductive observations)
- Conversation history and new turns
- Full prompt sent to the LLM
- **LLM output**: Structured response with explicit and implicit facts extracted
- **LLM output raw**: Raw JSON response from the model
- Thinking process (if available)

For a conversation with N messages, this will generate N trace entries showing
how the theory of mind representation evolves as each message is processed.

Each trace entry can be correlated back to the original conversation using the
dataset_uuid and conversation_id fields, and the message_sequence_id shows the
order in which messages were processed.

## Output Schema

Each trace entry follows this schema:
```json
{
  "timestamp": "ISO 8601 timestamp",
  "dataset_uuid": "UUID for this conversation",
  "conversation_id": "Original conversation ID",
  "message_sequence_id": 1,
  "total_messages": 10,
  "dataset": "dataset_name",
  "peer_id": "peer_identifier",
  "message_created_at": "ISO 8601 timestamp",
  "working_representation_input": {
    "explicit": ["explicit fact 1", "explicit fact 2", ...],
    "implicit": ["implicit fact 1", "implicit fact 2", ...]
  },
  "llm_output": {
    "explicit": ["explicit fact 1", "explicit fact 2", ...],
    "implicit": ["implicit fact 1", "implicit fact 2", ...]
  },
  "prompt": "Full prompt text",
  "thinking": "Model thinking trace (if available)"
}
```
"""

import argparse
import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from honcho import AsyncHoncho
from honcho.async_client.session import SessionPeerConfig
from honcho_core.types.workspaces.sessions.message_create_param import (
    MessageCreateParam,
)

from src.config import settings
from src.utils.metrics_collector import MetricsCollector, convert_trace_to_json

load_dotenv()


class TraceCollector:
    """
    Collects trace data from conversation processing through Honcho.
    """

    def __init__(
        self,
        base_api_port: int = 8000,
        pool_size: int = 1,
        timeout_seconds: int | None = None,
        workspace_id_prefix: str | None = None,
        cleanup_workspace: bool = False,
    ):
        """
        Initialize the trace collector.

        Args:
            base_api_port: Base port for Honcho API instances (default: 8000)
            pool_size: Number of Honcho instances in the pool (default: 1)
            timeout_seconds: Timeout for deriver queue in seconds
            workspace_id_prefix: Prefix for workspace IDs (default: auto-generated)
            cleanup_workspace: If True, delete workspace after processing
        """
        self.base_api_port: int = base_api_port
        self.pool_size: int = pool_size
        self.timeout_seconds: int = (
            timeout_seconds if timeout_seconds is not None else 600
        )
        self.workspace_id_prefix: str = workspace_id_prefix or f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.cleanup_workspace: bool = cleanup_workspace

        # Initialize metrics collector
        self.metrics_collector: MetricsCollector = MetricsCollector()
        self.metrics_collector.start_collection(
            f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Configure logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Suppress HTTP request logs from the Honcho SDK
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)

    def get_honcho_url_for_index(self, conversation_index: int) -> str:
        """
        Get the Honcho URL for a given conversation index using round-robin distribution.

        Args:
            conversation_index: Index of the conversation

        Returns:
            URL of the Honcho instance to use for this conversation
        """
        instance_id = conversation_index % self.pool_size
        port = self.base_api_port + instance_id
        return f"http://localhost:{port}"

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime.

        Args:
            timestamp_str: Timestamp string in various formats

        Returns:
            Parsed datetime object

        Raises:
            ValueError: If timestamp format is invalid
        """
        try:
            # Try common formats
            for fmt in [
                "%Y-%m-%d %H:%M:%S",
                "%Y/%m/%d %H:%M",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
            ]:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Unable to parse timestamp: {timestamp_str}")
        except Exception as e:
            raise ValueError(f"Failed to parse timestamp '{timestamp_str}': {e}") from e

    def load_conversation_file(self, conversation_file: Path) -> dict[str, Any]:
        """
        Load conversation data from a JSON file.

        Args:
            conversation_file: Path to the JSON conversation file

        Returns:
            Dictionary containing conversation data
        """
        with open(conversation_file) as f:
            return json.load(f)

    async def create_honcho_client(
        self, workspace_id: str, honcho_url: str
    ) -> AsyncHoncho:
        """
        Create a Honcho client for a specific workspace.

        Args:
            workspace_id: Workspace ID for the conversation
            honcho_url: URL of the Honcho instance

        Returns:
            AsyncHoncho client instance
        """
        return AsyncHoncho(
            environment="local",
            workspace_id=workspace_id,
            base_url=honcho_url,
        )

    async def wait_for_deriver_queue_empty(
        self, honcho_client: AsyncHoncho, session_id: str | None = None
    ) -> bool:
        """
        Wait for the deriver queue to become empty.

        Args:
            honcho_client: Honcho client instance
            session_id: Optional session ID to check

        Returns:
            True if queue became empty, False if timeout
        """
        import time

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
        self, conversation_id: str, conversation_data: dict[str, Any], honcho_url: str
    ) -> bool:
        """
        Process a single conversation and collect trace data.

        Args:
            conversation_id: Unique identifier for this conversation
            conversation_data: Dictionary containing conversation data
            honcho_url: URL of the Honcho instance to use

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Processing conversation: {conversation_id} on {honcho_url}")

        # Generate a dataset UUID for this conversation
        dataset_uuid = str(uuid.uuid4())

        # Extract conversation data
        dataset = conversation_data.get("dataset", "unknown")
        peers_list = conversation_data.get("peers", [])
        messages = conversation_data.get("messages", [])

        if not messages:
            self.logger.warning(f"No messages found in conversation {conversation_id}")
            return False

        # Create workspace ID for this conversation
        workspace_id = f"{self.workspace_id_prefix}_{conversation_id}"

        # Create Honcho client
        honcho_client = await self.create_honcho_client(workspace_id, honcho_url)

        self.logger.info(
            f"Dataset UUID: {dataset_uuid} | Conversation ID: {conversation_id} | Dataset: {dataset}"
        )

        try:
            # Create session with metadata
            session_id = f"{conversation_id}"
            session = await honcho_client.session(
                id=session_id,
                metadata={
                    "dataset_uuid": dataset_uuid,
                    "conversation_id": conversation_id,
                    "dataset": dataset,
                },
            )
            self.logger.info(f"Created session: {session_id}")

            # Create peers and add to session
            peers = {}
            for peer_id in peers_list:
                peer = await honcho_client.peer(id=peer_id)
                peers[peer_id] = peer
                self.logger.info(f"Created peer: {peer_id}")

            # Add peers to session with observation settings
            # First peer observes themselves, others don't
            peer_configs = []
            for idx, peer_id in enumerate(peers_list):
                observe_me = idx == 0  # First peer observes themselves
                peer_configs.append(
                    (
                        peers[peer_id],
                        SessionPeerConfig(observe_me=observe_me, observe_others=False),
                    )
                )
            await session.add_peers(peer_configs)

            # Process messages incrementally to capture traces at each turn
            total_messages = len(messages)
            self.logger.info(
                f"Processing {total_messages} messages incrementally to capture traces at each turn"
            )

            for turn_idx, msg in enumerate(messages, 1):
                peer_id = msg["peer"]
                content = msg["content"]
                timestamp_str = msg["timestamp"]

                if peer_id not in peers:
                    self.logger.warning(
                        f"Peer {peer_id} not found in peers list, skipping message"
                    )
                    continue

                if len(content) == 0:
                    content = "(empty message)"

                # Parse timestamp
                try:
                    message_time = self._parse_timestamp(timestamp_str)
                except ValueError as e:
                    self.logger.warning(f"Error parsing timestamp: {e}")
                    message_time = datetime.now()

                # Prepare messages for this turn (handle long messages)
                # Add metadata with dataset_uuid and message sequence number
                turn_messages: list[MessageCreateParam] = []
                message_metadata = {
                    "dataset_uuid": dataset_uuid,
                    "conversation_id": conversation_id,
                    "message_sequence_id": turn_idx,
                    "total_messages": total_messages,
                    "dataset": dataset,
                }

                # Split message if it exceeds 25000 characters
                if len(content) > 25000:
                    chunks = [
                        content[i : i + 25000] for i in range(0, len(content), 25000)
                    ]
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_metadata = message_metadata.copy()
                        chunk_metadata["chunk_index"] = chunk_idx
                        chunk_metadata["total_chunks"] = len(chunks)
                        turn_messages.append(
                            peers[peer_id].message(
                                chunk, created_at=message_time, metadata=chunk_metadata
                            )
                        )
                else:
                    turn_messages.append(
                        peers[peer_id].message(
                            content, created_at=message_time, metadata=message_metadata
                        )
                    )

                # Add this turn's messages
                await session.add_messages(turn_messages)

                self.logger.info(
                    f"Turn {turn_idx}/{total_messages}: Added message from {peer_id}"
                )

                # Wait for deriver queue to process this turn before adding the next
                self.logger.debug(
                    f"Waiting for deriver to process turn {turn_idx}..."
                )
                await asyncio.sleep(0.5)  # Give time for tasks to be queued

                queue_empty = await self.wait_for_deriver_queue_empty(honcho_client)
                if not queue_empty:
                    self.logger.error(
                        f"Deriver queue timeout at turn {turn_idx} - not all messages processed"
                    )
                    return False

                self.logger.debug(f"Turn {turn_idx} processed successfully")

            self.logger.info(f"Successfully processed conversation {conversation_id}")

            # Clean up workspace if requested
            if self.cleanup_workspace:
                try:
                    await honcho_client.delete_workspace(workspace_id)
                    self.logger.info(f"Cleaned up workspace: {workspace_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete workspace: {e}")

            return True

        except Exception as e:
            self.logger.error(f"Error processing conversation {conversation_id}: {e}")
            return False

    async def run(self, conversation_file: Path, batch_size: int = 10) -> bool:
        """
        Run the trace collector on a conversation file.

        Args:
            conversation_file: Path to the conversation JSON file
            batch_size: Number of conversations to process concurrently in each batch

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load conversation data
            conversations = self.load_conversation_file(conversation_file)
            conversations_list = list(conversations.items())

            self.logger.info(
                f"Loaded {len(conversations_list)} conversation(s) from {conversation_file}"
            )

            if self.pool_size > 1:
                self.logger.info(
                    f"Distributing conversations across {self.pool_size} Honcho instances "
                    f"(ports {self.base_api_port}-{self.base_api_port + self.pool_size - 1})"
                )

            # Process conversations in batches
            success_count = 0

            for i in range(0, len(conversations_list), batch_size):
                batch = conversations_list[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(conversations_list) + batch_size - 1) // batch_size

                self.logger.info(f"\n{'=' * 60}")
                self.logger.info(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch)} conversations)"
                )
                self.logger.info(f"{'=' * 60}")

                # Run conversations in current batch concurrently, distributing via round-robin
                batch_results: list[bool] = await asyncio.gather(
                    *[
                        self.process_conversation(
                            conv_id, conv_data, self.get_honcho_url_for_index(i + idx)
                        )
                        for idx, (conv_id, conv_data) in enumerate(batch)
                    ]
                )

                # Count successes in this batch
                batch_success_count = sum(1 for result in batch_results if result)
                success_count += batch_success_count

                self.logger.info(
                    f"Batch {batch_num} completed: {batch_success_count}/{len(batch)} successful"
                )

            self.logger.info(
                f"\nProcessed {success_count}/{len(conversations_list)} conversations successfully"
            )

            # Finalize metrics collection
            self.metrics_collector.finalize_collection()

            # Convert trace file to JSON
            convert_trace_to_json()

            # Cleanup
            self.metrics_collector.cleanup_collection()

            return success_count == len(conversations_list)

        except Exception as e:
            self.logger.error(f"Error running trace collector: {e}")
            return False


async def main() -> int:
    """
    Main entry point for the trace collector.
    """
    parser = argparse.ArgumentParser(
        description="Collect trace data from Honcho conversation processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --conversation-file tests/bench/longmemeval_data/test_data.json
  %(prog)s --conversation-file data.json --pool-size 4 --batch-size 20
  %(prog)s --conversation-file data.json --workspace-id-prefix my_trace
        """,
    )

    parser.add_argument(
        "--conversation-file",
        type=Path,
        required=True,
        help="Path to conversation JSON file (required)",
    )

    parser.add_argument(
        "--base-api-port",
        type=int,
        default=8000,
        help="Base port for Honcho API instances (default: 8000)",
    )

    parser.add_argument(
        "--pool-size",
        type=int,
        default=1,
        help="Number of Honcho instances in the pool (default: 1)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of conversations to process concurrently in each batch (default: 10)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout for deriver queue in seconds (default: 10 minutes)",
    )

    parser.add_argument(
        "--workspace-id-prefix",
        type=str,
        help="Prefix for workspace IDs (default: auto-generated)",
    )

    parser.add_argument(
        "--cleanup-workspace",
        action="store_true",
        help="Delete workspace after processing (default: False)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.conversation_file.exists():
        print(f"Error: Conversation file {args.conversation_file} does not exist")
        return 1

    if args.batch_size <= 0:
        print(f"Error: Batch size must be positive, got {args.batch_size}")
        return 1

    if args.pool_size <= 0:
        print(f"Error: Pool size must be positive, got {args.pool_size}")
        return 1

    # Create trace collector
    collector = TraceCollector(
        base_api_port=args.base_api_port,
        pool_size=args.pool_size,
        timeout_seconds=args.timeout,
        workspace_id_prefix=args.workspace_id_prefix,
        cleanup_workspace=args.cleanup_workspace,
    )

    try:
        # Run trace collection
        success = await collector.run(args.conversation_file, args.batch_size)
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\nTrace collection interrupted by user")
        return 1
    except Exception as e:
        import traceback

        print(f"Error collecting trace data: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
