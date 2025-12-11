"""
Honcho Explicit Derivation Benchmark Test Runner

A script that executes explicit derivation benchmark tests against a running Honcho instance.
This script:
1. Loads conversations with ground truth explicit facts
2. Creates a workspace for each conversation
3. Ingests conversation messages
4. Waits for the deriver to extract observations
5. Retrieves extracted observations from Honcho
6. Uses the 3-stage judge to evaluate extraction quality
7. Calculates cumulative precision, recall, F1, and F2 scores across all propositions

## Explicit Derivation Overview

This benchmark tests Honcho's ability to extract atomic facts from conversations.
It evaluates:
- Cumulative Precision: Valid extractions / Total extractions (across all conversations)
- Cumulative Recall: Matched ground truth / Total ground truth (across all conversations)
- Cumulative F1/F2 Score: Harmonic and recall-weighted F-scores
- Stage Metrics: Pass rates for each validation stage
- Error Detection: Classification of common failure modes (negation, temporal, hedge removal, etc.)

## To use

0. Set up env:
```
uv sync
source .venv/bin/activate
```
NOTE: you may create a .env file in this directory to customize honcho config.

1. Run the test harness:
```
python -m tests.bench.harness
```

2. Run this file with a dataset:
```
python -m tests.bench.explicit --data-file tests/bench/explicit_data/dataset.json
```

Optional arguments:
```
--data-file: Path to test dataset JSON file
--timeout: Timeout for deriver queue to empty in seconds (default: 10 minutes)
--base-api-port: Base port for Honcho API instances (default: 8000)
--pool-size: Number of Honcho instances in the pool (default: 1)
--batch-size: Number of conversations to run concurrently in each batch (default: 1)
--json-output: Path to write JSON summary results for analytics
--cleanup-workspace: Delete workspace after executing each conversation (default: False)
--conversation-id: Run only the conversation with this ID (skips all others)
--judge-mode: Judge operating mode (eval, filter, production) (default: eval)
--verbose, -v: Enable verbose logging to see detailed judge prompts and decisions
--stages: Comma-separated stages to run (e.g., '1,2' to skip Stage 3 LLM judge) (default: '1,2,3')
```

## Dataset Format

The dataset should be a JSON file with this structure:
```json
[
  {
    "conversation_id": "conv_001",
    "peer_name": "Sarah",
    "messages": [
      {
        "text": "I just got back from walking my dog in Central Park.",
        "timestamp": "2025-06-26T10:30:00Z"
      },
      {
        "text": "I've been living in NYC for about 3 years now.",
        "timestamp": "2025-06-26T10:31:00Z"
      }
    ],
    "ground_truth": [
      "Sarah has a dog",
      "Sarah went to Central Park on June 26, 2025",
      "Sarah walked her dog",
      "Sarah lives in NYC",
      "Sarah has lived in NYC for approximately 3 years as of June 26, 2025"
    ]
  }
]
```
"""

import argparse
import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from honcho import AsyncHoncho
from honcho.async_client.session import SessionPeerConfig
from honcho_core.types.workspaces.sessions.message_create_param import (
    MessageCreateParam,
)
from openai import AsyncOpenAI

from src.config import settings
from src.utils.metrics_collector import MetricsCollector

from .explicit_common import (
    ConversationResult,
    ExtractionResult,
    Proposition,
    format_duration,
    generate_json_summary,
    load_dataset,
    print_summary,
)
from .explicit_judge import ExplicitJudge

# Load .env from bench directory
bench_dir = Path(__file__).parent
load_dotenv(bench_dir / ".env")


class ExplicitBenchmarkRunner:
    """
    Executes explicit derivation benchmark tests against a Honcho instance.
    """

    def __init__(
        self,
        base_api_port: int = 8000,
        pool_size: int = 1,
        timeout_seconds: int | None = None,
        cleanup_workspace: bool = False,
        judge_mode: str = "eval",
        verbose: bool = False,
        stages_to_run: list[int] | None = None,
    ):
        """
        Initialize the explicit benchmark runner.

        Args:
            base_api_port: Base port for Honcho API instances (default: 8000)
            pool_size: Number of Honcho instances in the pool (default: 1)
            timeout_seconds: Timeout for deriver queue in seconds
            cleanup_workspace: If True, delete workspace after executing conversation
            judge_mode: Operating mode for judge (eval, filter, production)
            verbose: Enable verbose logging for debugging
            stages_to_run: List of stages to run (1, 2, 3). Default: [1, 2, 3]
        """
        self.base_api_port: int = base_api_port
        self.pool_size: int = pool_size
        self.timeout_seconds: int = (
            timeout_seconds if timeout_seconds is not None else 600
        )
        self.cleanup_workspace: bool = cleanup_workspace
        self.judge_mode: str = judge_mode
        self.verbose: bool = verbose
        self.stages_to_run: list[int] = stages_to_run if stages_to_run else [1, 2, 3]

        # Initialize metrics collector
        self.metrics_collector: MetricsCollector = MetricsCollector()
        self.metrics_collector.start_collection(
            f"explicit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Configure logging
        log_level = logging.DEBUG if verbose else logging.WARNING
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger: logging.Logger = logging.getLogger(__name__)

        if verbose:
            self.logger.info("Verbose mode enabled - detailed logging active")

        # Suppress HTTP request logs unless verbose
        if not verbose:
            logging.getLogger("httpx").setLevel(logging.ERROR)
            logging.getLogger("httpcore").setLevel(logging.ERROR)

        # Initialize LLM client for judge
        anthropic_api_key = os.getenv("LLM_ANTHROPIC_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        llm_client: AsyncAnthropic | AsyncOpenAI | None = None
        if anthropic_api_key:
            llm_client = AsyncAnthropic(api_key=anthropic_api_key)
        elif openai_api_key:
            llm_client = AsyncOpenAI(api_key=openai_api_key)
        else:
            self.logger.warning(
                "No LLM API key found. Stage 3 judge will use fallback behavior."
            )

        # Initialize judge system
        print("\n" + "=" * 80)
        print("Initializing Explicit Derivation Judge System")
        print("=" * 80)
        print("\nLoading NLP models (this may take a moment on first run):")
        print("  - NLI Model: facebook/bart-large-mnli (~1.6GB)")
        if openai_api_key:
            print("  - Embedding Model: OpenAI text-embedding-3-small (API)")
        else:
            print("  - Embedding Model: all-MiniLM-L6-v2 (~90MB, local)")
        print("  - spaCy Model: en_core_web_sm (auto-download if missing)")
        print("\nPlease wait...\n")

        self.judge = ExplicitJudge(
            llm_client=llm_client,
            mode=judge_mode,
            verbose=verbose,
            use_openai_embeddings=bool(openai_api_key),
            stages_to_run=self.stages_to_run,
        )

        print("\n✓ All models loaded successfully!")
        if verbose:
            print("  [Verbose mode: detailed logging enabled]")
        if openai_api_key:
            print("  [Using OpenAI embeddings for faster matching]")
        stages_str = ",".join(str(s) for s in self.stages_to_run)
        print(f"  [Running stages: {stages_str}]")
        print("=" * 80 + "\n")

    def get_honcho_url_for_index(self, conversation_index: int) -> str:
        """Get the Honcho URL for a given conversation index using round-robin distribution."""
        instance_id = conversation_index % self.pool_size
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

    async def extract_observations_from_honcho(
        self, honcho_client: AsyncHoncho, peer_id: str
    ) -> list[Proposition]:
        """
        Extract observations from Honcho's representation of a peer.

        Args:
            honcho_client: Honcho client instance
            peer_id: Peer identifier

        Returns:
            List of propositions extracted from observations
        """
        propositions: list[Proposition] = []

        try:
            # Get peer to access observations
            peer = await honcho_client.peer(id=peer_id)

            # Get the peer's working representation (explicit observations)
            representation = await peer.observations.get_representation()

            # Extract explicit observations (the facts extracted from messages)
            for explicit_obs in representation.explicit:
                text = explicit_obs.content.strip()
                if text:
                    # Create proposition
                    # Note: source_quote is not directly available from observations
                    # We could potentially track this via observation metadata if needed
                    propositions.append({"text": text, "source_quote": ""})

            # Note: We focus on explicit observations as this is what the benchmark tests
            # Deductive observations are generated later through the dreamer

        except Exception as e:
            self.logger.error(f"Error extracting observations from Honcho: {e}")
            import traceback

            traceback.print_exc()

        return propositions

    async def execute_conversation(
        self, conversation: dict[str, Any], honcho_url: str
    ) -> ConversationResult:
        """
        Execute explicit benchmark for a single conversation.

        Args:
            conversation: Conversation test case with messages and ground truth
            honcho_url: URL of the Honcho instance to use

        Returns:
            Conversation execution results
        """
        start_time = time.time()
        conversation_id = conversation["conversation_id"]
        peer_name = conversation["peer_name"]
        messages_data = conversation["messages"]
        ground_truth = conversation["ground_truth"]

        print(f"\n{'=' * 80}")
        print(f"Executing conversation {conversation_id} (peer: {peer_name})")
        print(f"{'=' * 80}")

        # Create workspace for this conversation
        workspace_id = f"explicit_{conversation_id}"
        honcho_client = await self.create_honcho_client(workspace_id, honcho_url)

        result: ConversationResult = {
            "conversation_id": conversation_id,
            "workspace_id": workspace_id,
            "total_messages": 0,
            "extracted_propositions": [],
            "ground_truth_propositions": ground_truth,
            "judgment": {
                "judgments": [],
                "precision": 0.0,
                "coverage": 0.0,
                "error_breakdown": {},
                "missing_propositions": [],
                "stage_metrics": {
                    "stage_1_total": 0,
                    "stage_1_passed": 0,
                    "stage_1_pass_rate": 0.0,
                    "stage_2_total": 0,
                    "stage_2_passed": 0,
                    "stage_2_pass_rate": 0.0,
                    "stage_2_escalated": 0,
                    "stage_3_total": 0,
                    "stage_3_validated": 0,
                    "stage_3_invalidated": 0,
                    "final_valid": 0,
                    "final_invalid": 0,
                },
            },
            "matched_count": 0,
            "hallucination_rate": 0.0,
            "stage_metrics": {
                "stage_1_total": 0,
                "stage_1_passed": 0,
                "stage_1_pass_rate": 0.0,
                "stage_2_total": 0,
                "stage_2_passed": 0,
                "stage_2_pass_rate": 0.0,
                "stage_2_escalated": 0,
                "stage_3_total": 0,
                "stage_3_validated": 0,
                "stage_3_invalidated": 0,
                "final_valid": 0,
                "final_invalid": 0,
            },
            "error": None,
            "start_time": start_time,
            "end_time": 0.0,
            "duration_seconds": 0.0,
        }

        try:
            # Create peer
            peer = await honcho_client.peer(id=peer_name.lower())

            # Create session for this conversation
            session_id = f"{workspace_id}_session"
            session = await honcho_client.session(id=session_id)

            # Configure peer observation - observe the peer
            await session.add_peers(
                [
                    (
                        peer,
                        SessionPeerConfig(observe_me=True, observe_others=False),
                    )
                ]
            )

            # Ingest conversation messages
            print(f"[{workspace_id}] Ingesting {len(messages_data)} messages...")
            messages: list[MessageCreateParam] = []

            for msg_data in messages_data:
                text = msg_data["text"]
                # TODO: Use timestamp if needed for temporal resolution
                messages.append(peer.message(text))

            result["total_messages"] = len(messages)

            # Add messages
            await session.add_messages(messages)

            print(
                f"[{workspace_id}] Ingested {result['total_messages']} messages. Waiting for deriver queue..."
            )

            # Wait for deriver queue to empty
            await asyncio.sleep(1)
            queue_empty = await self.wait_for_deriver_queue_empty(honcho_client)
            if not queue_empty:
                result["error"] = "Deriver queue timeout"
                result["end_time"] = time.time()
                result["duration_seconds"] = result["end_time"] - result["start_time"]
                print(
                    f"\n[{workspace_id}] ERROR: Deriver queue timeout after {self.timeout_seconds}s"
                )
                return result

            print(f"[{workspace_id}] Deriver queue empty. Extracting observations...")

            # Extract observations from Honcho
            extracted_propositions = await self.extract_observations_from_honcho(
                honcho_client, peer_name.lower()
            )
            result["extracted_propositions"] = extracted_propositions

            print(
                f"[{workspace_id}] Extracted {len(extracted_propositions)} observations"
            )

            # Judge the extraction
            print(f"[{workspace_id}] Running 3-stage judge pipeline...")
            print(
                f"[{workspace_id}] This may take 1-2 minutes for LLM judge calls..."
            )

            # Combine all messages into a single context for judging
            full_message = "\n".join([msg["text"] for msg in messages_data])

            judgment = await self.judge.judge_extraction(
                propositions=extracted_propositions,
                message=full_message,
                peer_name=peer_name,
                ground_truth=ground_truth,
            )

            print(f"[{workspace_id}] Judge extraction complete")
            result["judgment"] = judgment

            # Calculate matched counts for precision and recall
            print(f"[{workspace_id}] Calculating matched propositions...")
            matched_extracted, matched_ground_truth = await self.judge.calculate_matched_count(
                extracted_propositions, ground_truth, judgment
            )
            result["matched_extracted_count"] = matched_extracted
            result["matched_ground_truth_count"] = matched_ground_truth
            print(
                f"[{workspace_id}] Found {matched_extracted} matched valid propositions, "
                f"{matched_ground_truth} matched ground truth"
            )

            # Copy stage metrics
            result["stage_metrics"] = judgment["stage_metrics"]

            # Calculate hallucination rate
            if extracted_propositions:
                hallucination_count = judgment["error_breakdown"].get("hallucination", 0)
                result["hallucination_rate"] = hallucination_count / len(
                    extracted_propositions
                )

            print(f"[{workspace_id}] Results:")
            print(f"  Extracted: {len(extracted_propositions)}")
            print(f"  Valid: {result['stage_metrics']['final_valid']}")
            print(f"  Ground Truth: {len(ground_truth)}")
            print(f"  Matched (Valid): {matched_extracted}")
            print(f"  Matched (Ground Truth): {matched_ground_truth}")
            print(f"  Hallucination Rate: {result['hallucination_rate']:.3f}")

            # Print stage-by-stage metrics (only for stages that ran)
            print(f"\n  Stage Metrics:")
            sm = result["stage_metrics"]
            if sm['stage_1_total'] > 0:
                print(f"    Stage 1 (Structural): {sm['stage_1_passed']}/{sm['stage_1_total']} passed ({sm['stage_1_pass_rate']:.1%})")
            else:
                print(f"    Stage 1 (Structural): SKIPPED")

            if sm['stage_2_total'] > 0:
                print(f"    Stage 2 (NLI): {sm['stage_2_passed']}/{sm['stage_2_total']} passed ({sm['stage_2_pass_rate']:.1%}), {sm['stage_2_escalated']} escalated")
            else:
                print(f"    Stage 2 (NLI): SKIPPED")

            if sm['stage_3_total'] > 0 or 3 in self.stages_to_run:
                print(f"    Stage 3 (LLM Judge): {sm['stage_3_validated']}/{sm['stage_3_total']} validated, {sm['stage_3_invalidated']} invalidated")
            else:
                print(f"    Stage 3 (LLM Judge): SKIPPED")

            print(f"    Final: {sm['final_valid']} valid, {sm['final_invalid']} invalid")

            if judgment["error_breakdown"]:
                print("\n  Error Breakdown:")
                for error_type, count in judgment["error_breakdown"].items():
                    print(f"    {error_type}: {count}")

            # Cleanup workspace if requested
            if self.cleanup_workspace:
                try:
                    await honcho_client.delete_workspace(workspace_id)
                    print(f"[{workspace_id}] Cleaned up workspace")
                except Exception as e:
                    print(f"Failed to delete workspace: {e}")

            result["end_time"] = time.time()
            result["duration_seconds"] = result["end_time"] - result["start_time"]

            print(
                f"\n[{workspace_id}] Completed in {format_duration(result['duration_seconds'])}"
            )

        except Exception as e:
            self.logger.error(f"Error executing conversation {conversation_id}: {e}")
            import traceback

            traceback.print_exc()
            result["error"] = str(e)
            result["end_time"] = time.time()
            result["duration_seconds"] = result["end_time"] - result["start_time"]

        return result

    async def run_conversations(
        self,
        conversations: list[dict[str, Any]],
        batch_size: int = 1,
    ) -> tuple[list[ConversationResult], float]:
        """
        Run multiple conversations from the benchmark dataset.

        Args:
            conversations: List of conversation test cases
            batch_size: Number of conversations to run concurrently in each batch

        Returns:
            Tuple of (list of conversation results, total duration)
        """
        print(f"Running {len(conversations)} conversations")
        if self.pool_size > 1:
            print(
                f"Distributing conversations across {self.pool_size} Honcho instances"
            )

        overall_start = time.time()
        all_results: list[ConversationResult] = []

        for i in range(0, len(conversations), batch_size):
            batch = conversations[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(conversations) + batch_size - 1) // batch_size

            print(f"\n{'=' * 80}")
            print(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} conversations)"
            )
            print(f"{'=' * 80}")

            # Run conversations in current batch concurrently
            batch_results: list[ConversationResult] = await asyncio.gather(
                *[
                    self.execute_conversation(conv, self.get_honcho_url_for_index(i + idx))
                    for idx, conv in enumerate(batch)
                ]
            )

            all_results.extend(batch_results)

        overall_end = time.time()
        overall_duration = overall_end - overall_start

        # Finalize metrics collection
        self.metrics_collector.finalize_collection()

        return all_results, overall_duration


async def main() -> int:
    """Main entry point for the explicit benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run explicit derivation benchmark tests against a Honcho instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help="Path to test dataset JSON file",
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
        "--timeout",
        type=int,
        default=None,
        help="Timeout for deriver queue to empty in seconds (default: 10 minutes (600s))",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of conversations to run concurrently in each batch (default: 1)",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        help="Path to write JSON summary results for analytics (optional)",
    )

    parser.add_argument(
        "--cleanup-workspace",
        action="store_true",
        help="Delete workspace after executing each conversation (default: False)",
    )

    parser.add_argument(
        "--conversation-id",
        type=str,
        help="Run only the conversation with this ID (skips all others)",
    )

    parser.add_argument(
        "--judge-mode",
        type=str,
        default="eval",
        choices=["eval", "filter", "production"],
        help="Judge operating mode (default: eval)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging to see detailed judge prompts and decisions",
    )

    parser.add_argument(
        "--stages",
        type=str,
        default="1,2,3",
        help="Comma-separated list of stages to run (e.g., '1,2' to skip Stage 3 LLM judge). Default: '1,2,3' (all stages)",
    )

    args = parser.parse_args()

    # Parse stages argument
    try:
        stages_to_run = [int(s.strip()) for s in args.stages.split(",")]
        for stage in stages_to_run:
            if stage not in [1, 2, 3]:
                print(f"Error: Invalid stage number '{stage}'. Must be 1, 2, or 3.")
                return 1
    except ValueError:
        print(f"Error: Invalid stages format '{args.stages}'. Use comma-separated numbers like '1,2,3'")
        return 1

    # Load dataset
    if not args.data_file.exists():
        print(f"Error: Dataset file not found at {args.data_file}")
        return 1

    print(f"Loading dataset from {args.data_file}")
    dataset = load_dataset(args.data_file)

    # Filter by conversation ID if specified
    if args.conversation_id:
        dataset = [c for c in dataset if c["conversation_id"] == args.conversation_id]
        if not dataset:
            print(f"Error: No conversation found with ID {args.conversation_id}")
            return 1

    # Create runner
    runner = ExplicitBenchmarkRunner(
        base_api_port=args.base_api_port,
        pool_size=args.pool_size,
        timeout_seconds=args.timeout,
        cleanup_workspace=args.cleanup_workspace,
        judge_mode=args.judge_mode,
        verbose=args.verbose,
        stages_to_run=stages_to_run,
    )

    try:
        # Run conversations
        results, total_elapsed = await runner.run_conversations(
            dataset, args.batch_size
        )

        print_summary(results, total_elapsed)

        # Generate JSON output
        if args.json_output:
            output_file = args.json_output
        else:
            output_file = Path(
                f"tests/bench/eval_results/explicit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        generate_json_summary(
            results,
            total_elapsed,
            output_file,
            metadata_extra={
                "base_api_port": runner.base_api_port,
                "pool_size": runner.pool_size,
                "timeout_seconds": runner.timeout_seconds,
                "judge_mode": runner.judge_mode,
                "deriver_settings": settings.DERIVER.model_dump(),
            },
        )

        # Export metrics
        metrics_output = Path(
            f"tests/bench/perf_metrics/explicit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        runner.metrics_collector.export_to_json(metrics_output)
        runner.metrics_collector.cleanup_collection()

        return 0

    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
