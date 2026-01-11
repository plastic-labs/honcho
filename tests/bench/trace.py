"""
Honcho Trace Generation Tool

A high-throughput script for generating explicit derivations from conversation data
at scale using the actual Honcho deriver system. Designed to handle 1M+ conversation
samples with maximum parallelization.

This script uses the Honcho bench harness to start Honcho API instances and deriver
processes, then ingests conversations via the Honcho API to generate explicit
derivations using the same prompts and logic as the production system.

Key Features:
- Uses actual Honcho deriver system (not direct LLM calls)
- Streaming JSONL processing (no full file loading)
- Configurable parallelization with harness pool
- Appending writes to output file (results written as generated)
- Metadata tracking (workspace_id, session_id, conversation_id, peer_id)
- Uses production deriver prompts

## Input Format

JSONL file where each line is a JSON object with:
```json
{
  "conversation_id": "unique-id",
  "dataset": "dataset-name",
  "peers": ["peer1", "peer2"],
  "messages": [
    {"timestamp": "2023-01-01 12:00:00", "peer": "peer1", "content": "..."},
    {"timestamp": "2023-01-01 12:01:00", "peer": "peer2", "content": "..."}
  ]
}
```

## Output Format

JSONL file where each line contains:
```json
{
  "conversation_id": "unique-id",
  "workspace_id": "trace_workspace_xxx",
  "session_id": "trace_session_xxx",
  "peer_id": "peer1",
  "input_prompt": "The prompt used for derivation...",
  "explicit_derivations": ["observation 1", "observation 2"],
  "metadata": {
    "dataset": "dataset-name",
    "message_count": 10,
    "observation_count": 5,
    "processing_time_seconds": 1.234,
    "timestamp": "2026-01-08T12:00:00Z"
  }
}
```

## Usage

### Step 1: Start the Harness Pool

```bash
# Start 4 Honcho instances on ports 8000-8003
python -m tests.bench.harness --pool-size 4
```

### Step 2: Run Trace Generation

```bash
# Basic usage
python -m tests.bench.trace_optimize \\
  --input tests/bench/trace_data/input.jsonl \\
  --output tests/bench/trace_data/output.jsonl

# High throughput with parallelization
python -m tests.bench.trace_optimize \\
  --input tests/bench/trace_data/input.jsonl \\
  --output tests/bench/trace_data/output.jsonl \\
  --concurrency 20 \\
  --pool-size 4 \\
  --base-api-port 8000

# Process only first N conversations
python -m tests.bench.trace_optimize \\
  --input tests/bench/trace_data/input.jsonl \\
  --output tests/bench/trace_data/output.jsonl \\
  --limit 1000
```

## Arguments

- `--input`: Path to input JSONL file
- `--output`: Path to output JSONL file (will be created/appended)
- `--base-api-port`: Base port for Honcho API instances (default: 8000)
- `--pool-size`: Number of Honcho instances (default: 1)
- `--concurrency`: Number of parallel conversations to process (default: 10)
- `--timeout`: Timeout for deriver queue in seconds (default: 600)
- `--limit`: Process only first N conversations (optional)
- `--resume`: Resume from conversation_id (optional)
- `--cleanup-workspace`: Delete workspace after processing (default: False)
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
from tqdm import tqdm

# Load environment variables
bench_dir = Path(__file__).parent
load_dotenv(bench_dir / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TraceGenerator:
    """Generator for creating explicit derivations using Honcho system."""

    def __init__(
        self,
        base_api_port: int = 8000,
        pool_size: int = 1,
        concurrency: int = 10,
        timeout_seconds: int = 600,
        cleanup_workspace: bool = False,
    ):
        """
        Initialize the generator.

        Args:
            base_api_port: Base port for Honcho API instances
            pool_size: Number of Honcho instances in the pool
            concurrency: Maximum number of parallel conversations
            timeout_seconds: Timeout for deriver queue in seconds
            cleanup_workspace: If True, delete workspace after processing
        """
        self.base_api_port = base_api_port
        self.pool_size = pool_size
        self.concurrency = concurrency
        self.timeout_seconds = timeout_seconds
        self.cleanup_workspace = cleanup_workspace
        self.semaphore = asyncio.Semaphore(concurrency)

        # Statistics
        self.total_processed = 0
        self.total_errors = 0
        self.start_time = time.time()
        self.progress_bar = None

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
        self, honcho_client: AsyncHoncho, session_id: str
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

    @staticmethod
    def _write_result(output_file: Path, result: dict[str, Any]) -> None:
        """Write result to file synchronously."""
        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    async def process_conversation(
        self,
        conversation: dict[str, Any],
        output_file: Path,
        index: int,
    ) -> None:
        """
        Process a single conversation through Honcho and write results.

        Args:
            conversation: Conversation data dictionary
            output_file: Path to output JSONL file
            index: Index for round-robin Honcho instance selection
        """
        async with self.semaphore:
            start_time = time.time()
            conversation_id = conversation.get("conversation_id", f"unknown_{index}")
            dataset = conversation.get("dataset", "unknown")
            peers = conversation.get("peers", [])
            messages = conversation.get("messages", [])

            if not peers or not messages:
                logger.warning(f"[{conversation_id}] Skipping: no peers or messages")
                return

            # Generate IDs
            workspace_id = f"trace_{conversation_id}_{int(start_time)}"
            session_id = f"{workspace_id}_session"

            # Get Honcho instance
            honcho_url = self.get_honcho_url_for_index(index)
            honcho_client = await self.create_honcho_client(workspace_id, honcho_url)

            try:
                # Create peers
                peer_objects = {}
                for peer_id in peers:
                    peer = await honcho_client.peer(id=peer_id)
                    peer_objects[peer_id] = peer

                # Create session
                session = await honcho_client.session(id=session_id)

                # Configure peer observation (observe_me=True for self-observation)
                peer_configs = [
                    (peer, SessionPeerConfig(observe_me=True, observe_others=False))
                    for peer in peer_objects.values()
                ]
                await session.add_peers(peer_configs)

                # Ingest messages
                message_params = []
                for msg in messages:
                    peer_id = msg.get("peer")
                    content = msg.get("content", "")
                    timestamp_str = msg.get("timestamp")

                    # Parse timestamp
                    created_at = None
                    if timestamp_str:
                        try:
                            created_at = datetime.fromisoformat(
                                timestamp_str.replace(" ", "T")
                            )
                        except ValueError:
                            pass

                    if peer_id in peer_objects:
                        peer_obj = peer_objects[peer_id]
                        message_params.append(
                            peer_obj.message(content, created_at=created_at)
                        )

                # Batch ingest (100 at a time)
                for i in range(0, len(message_params), 100):
                    batch = message_params[i : i + 100]
                    await session.add_messages(batch)

                # Wait for deriver to process
                logger.info(
                    f"[{conversation_id}] Ingested {len(message_params)} messages, waiting for deriver..."
                )
                await asyncio.sleep(2)  # Give deriver time to pick up work
                queue_empty = await self.wait_for_deriver_queue_empty(
                    honcho_client, session_id
                )

                if not queue_empty:
                    logger.error(
                        f"[{conversation_id}] Deriver timeout after {self.timeout_seconds}s"
                    )
                    self.total_errors += 1
                    return

                # Retrieve observations for each peer
                for peer_id, peer in peer_objects.items():
                    try:
                        # Get observations (explicit derivations) with pagination
                        # API has max page size of 100, so we need to paginate
                        all_observations = []
                        page = 1
                        page_size = 100

                        while True:
                            observations = await peer.observations.list(
                                session=session_id,
                                page=page,
                                size=page_size
                            )

                            if not observations:
                                break

                            all_observations.extend(observations)

                            # If we got fewer than page_size, we're done
                            if len(observations) < page_size:
                                break

                            page += 1

                        # Extract observation contents (explicit derivations)
                        explicit_derivations = [obs.content for obs in all_observations]

                        # For the prompt, we need to reconstruct what was sent to the deriver
                        # This mirrors the minimal_deriver_prompt from src/deriver/prompts.py
                        formatted_messages = "\n".join(
                            f"[{msg.get('timestamp', '')}] {msg.get('peer', '')}: {msg.get('content', '')}"
                            for msg in messages
                        )

                        input_prompt = f"""Analyze messages from {peer_id} to extract **explicit atomic facts** about them.

[EXPLICIT] DEFINITION: Facts about {peer_id} that can be derived directly from their messages.
   - Transform statements into one or multiple conclusions
   - Each conclusion must be self-contained with enough context
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")

RULES:
- Properly attribute observations to the correct subject: if it is about {peer_id}, say so. If {peer_id} is referencing someone or something else, make that clear.
- Observations should make sense on their own. Each observation will be used in the future to better understand {peer_id}.
- Extract ALL observations from {peer_id} messages, using others as context.
- Contextualize each observation sufficiently (e.g. "Ann is nervous about the job interview at the pharmacy" not just "Ann is nervous")

EXAMPLES:
- EXPLICIT: "I just had my 25th birthday last Saturday" → "{peer_id} is 25 years old", "{peer_id}'s birthday is June 21st"
- EXPLICIT: "I took my dog for a walk in NYC" → "{peer_id} has a dog", "{peer_id} lives in NYC"
- EXPLICIT: "{peer_id} attended college" + general knowledge → "{peer_id} completed high school or equivalent"

Messages to analyze:
<messages>
{formatted_messages}
</messages>"""

                        # Create result
                        result = {
                            "conversation_id": conversation_id,
                            "workspace_id": workspace_id,
                            "session_id": session_id,
                            "peer_id": peer_id,
                            "input_prompt": input_prompt,
                            "explicit_derivations": explicit_derivations,
                            "metadata": {
                                "dataset": dataset,
                                "message_count": len(messages),
                                "observation_count": len(explicit_derivations),
                                "processing_time_seconds": time.time() - start_time,
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                            },
                        }

                        # Write result
                        await asyncio.to_thread(self._write_result, output_file, result)

                    except Exception as e:
                        logger.error(
                            f"[{conversation_id}] Error retrieving observations for {peer_id}: {e}"
                        )
                        self.total_errors += 1

                # Cleanup workspace if requested
                if self.cleanup_workspace:
                    try:
                        await honcho_client.delete_workspace()
                    except Exception as e:
                        logger.warning(
                            f"[{conversation_id}] Failed to cleanup workspace: {e}"
                        )

                self.total_processed += 1

                # Update progress bar
                if self.progress_bar:
                    elapsed = time.time() - self.start_time
                    rate = self.total_processed / elapsed if elapsed > 0 else 0
                    self.progress_bar.update(1)
                    self.progress_bar.set_postfix({
                        "rate": f"{rate:.2f} conv/s",
                        "errors": self.total_errors
                    })

            except Exception as e:
                logger.error(f"[{conversation_id}] Error: {e}", exc_info=True)
                self.total_errors += 1

                # Update progress bar even on error
                if self.progress_bar:
                    elapsed = time.time() - self.start_time
                    rate = self.total_processed / elapsed if elapsed > 0 else 0
                    self.progress_bar.update(1)
                    self.progress_bar.set_postfix({
                        "rate": f"{rate:.2f} conv/s",
                        "errors": self.total_errors
                    })

    @staticmethod
    def _count_lines(file_path: Path) -> int:
        """Count total lines in file efficiently."""
        count = 0
        with open(file_path, 'rb') as f:
            for _ in f:
                count += 1
        return count

    async def process_file(
        self,
        input_file: Path,
        output_file: Path,
        limit: int | None = None,
        resume_from: str | None = None,
    ) -> None:
        """
        Process input JSONL file and generate derivations.

        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            limit: Maximum number of conversations to process
            resume_from: Resume from this conversation_id (skip all before)
        """
        logger.info(f"Starting processing: {input_file} -> {output_file}")
        logger.info(f"Base API Port: {self.base_api_port}, Pool Size: {self.pool_size}")
        logger.info(f"Concurrency: {self.concurrency}")

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Count total lines for progress bar
        logger.info("Counting conversations in input file...")
        total_lines = self._count_lines(input_file)
        logger.info(f"Found {total_lines:,} conversations in input file")

        # Track which conversations we've already processed (for resume)
        processed_ids = set()
        if output_file.exists() and resume_from:
            logger.info(f"Resume mode: skipping until {resume_from}")
            with open(output_file, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed_ids.add(data.get("conversation_id"))
                    except json.JSONDecodeError:
                        continue

        # Calculate total to process
        total_to_process = min(limit, total_lines) if limit else total_lines

        # Initialize progress bar
        self.progress_bar = tqdm(
            total=total_to_process,
            desc="Processing conversations",
            unit="conv",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            dynamic_ncols=True,
        )

        # Process conversations in batches
        batch_size = self.concurrency * 2
        batch: list[tuple[dict[str, Any], int]] = []
        count = 0
        skip_until_found = bool(resume_from)

        try:
            with open(input_file, "r") as f:
                for line in f:
                    try:
                        conversation = json.loads(line)
                        conversation_id = conversation.get("conversation_id")

                        # Handle resume logic
                        if skip_until_found:
                            if conversation_id == resume_from:
                                skip_until_found = False
                                logger.info(f"Resumed at {resume_from}")
                            continue

                        # Skip already processed
                        if conversation_id in processed_ids:
                            continue

                        batch.append((conversation, count))
                        count += 1

                        # Process batch when full
                        if len(batch) >= batch_size:
                            tasks = [
                                self.process_conversation(conv, output_file, idx)
                                for conv, idx in batch
                            ]
                            await asyncio.gather(*tasks, return_exceptions=True)
                            batch = []

                        # Check limit
                        if limit and count >= limit:
                            break

                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON line: {e}")
                        continue

                # Process remaining batch
                if batch:
                    tasks = [
                        self.process_conversation(conv, output_file, idx)
                        for conv, idx in batch
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)

        finally:
            # Close progress bar
            if self.progress_bar:
                self.progress_bar.close()

        # Final statistics
        elapsed = time.time() - self.start_time
        rate = self.total_processed / elapsed if elapsed > 0 else 0
        logger.info("=" * 80)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total conversations processed: {self.total_processed:,}")
        logger.info(f"Total errors: {self.total_errors:,}")
        logger.info(f"Total time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
        logger.info(f"Average rate: {rate:.2f} conversations/second")
        if total_to_process > 0:
            success_rate = (self.total_processed / total_to_process) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info("=" * 80)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate explicit derivations using Honcho system at scale",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (requires harness running on port 8000)
  %(prog)s --input data/input.jsonl --output data/output.jsonl

  # High throughput with 4 Honcho instances
  %(prog)s --input data/input.jsonl --output data/output.jsonl \\
    --pool-size 4 --base-api-port 8000 --concurrency 20

  # Process first 1000 conversations
  %(prog)s --input data/input.jsonl --output data/output.jsonl --limit 1000

  # Resume from specific conversation
  %(prog)s --input data/input.jsonl --output data/output.jsonl \\
    --resume conversation-id-123

Note: You must start the harness pool first:
  python -m tests.bench.harness --pool-size 4
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output JSONL file (will be created/appended)",
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
        help="Number of Honcho instances in harness pool (default: 1)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of parallel conversations to process (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout for deriver queue in seconds (default: 600)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only first N conversations (optional)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from conversation_id (skip all before this)",
    )
    parser.add_argument(
        "--cleanup-workspace",
        action="store_true",
        help="Delete workspace after processing (default: False)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.input.exists():
        parser.error(f"Input file does not exist: {args.input}")

    # Create generator
    generator = TraceGenerator(
        base_api_port=args.base_api_port,
        pool_size=args.pool_size,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout,
        cleanup_workspace=args.cleanup_workspace,
    )

    # Process file
    try:
        await generator.process_file(
            input_file=args.input,
            output_file=args.output,
            limit=args.limit,
            resume_from=args.resume,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())