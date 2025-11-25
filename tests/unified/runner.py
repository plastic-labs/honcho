import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

from anthropic import AsyncAnthropic
from honcho.async_client.session import AsyncSession
from honcho.session_context import SessionContext
from honcho_core.types.deriver_status import DeriverStatus
from pydantic import ValidationError

# Adjust path to allow imports from tests.bench
sys.path.insert(0, str(Path(__file__).parents[2]))

from honcho import AsyncHoncho
from honcho.async_client.peer import AsyncPeer
from honcho.async_client.session import SessionPeerConfig as SDKSessionPeerConfig
from honcho_core.types.workspaces.sessions.message_create_param import (
    Configuration,
    MessageCreateParam,
)

from tests.bench.harness import HonchoHarness
from tests.unified.schema import (
    AddMessageAction,
    AddMessagesAction,
    ContainsAssertion,
    CreateSessionAction,
    ExactMatchAssertion,
    JsonMatchAssertion,
    LLMJudgeAssertion,
    NotContainsAssertion,
    QueryAction,
    SetSessionConfigAction,
    SetWorkspaceConfigAction,
    TestDefinition,
    TriggerDreamAction,
    WaitAction,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

JUDGE_MODEL: str = "claude-sonnet-4-5"


class TestExecutionError(Exception):
    pass


class UnifiedTestExecutor:
    def __init__(
        self, honcho_client: AsyncHoncho, anthropic_client: AsyncAnthropic | None
    ):
        self.client: AsyncHoncho = honcho_client
        self.anthropic: AsyncAnthropic | None = anthropic_client

    async def execute(self, test_def: TestDefinition, test_name: str) -> bool:
        logger.info(f"Starting test: {test_name}")

        # 1. Apply workspace config if present
        if test_def.workspace_config:
            await self.client.set_config(
                test_def.workspace_config.model_dump(exclude_none=True)
            )

        for i, step in enumerate(test_def.steps):
            logger.info(f"Executing step {i + 1}: {step.step_type}")
            try:
                await self.execute_step(step)
            except Exception as e:
                logger.error(f"Step {i + 1} failed: {e}", exc_info=False)
                return False

        logger.info(f"Test {test_name} PASSED")
        return True

    async def execute_step(self, step: Any):
        if isinstance(step, SetWorkspaceConfigAction):
            await self.client.set_config(step.config.model_dump(exclude_none=True))

        elif isinstance(step, SetSessionConfigAction):
            session = await self.client.session(id=step.session_id)
            await session.set_config(step.config.model_dump(exclude_none=True))

        elif isinstance(step, CreateSessionAction):
            session = await self.client.session(
                id=step.session_id,
                config=step.config.model_dump(exclude_none=True)
                if step.config
                else None,
            )

            if step.peer_configs:
                peer_list: list[tuple[str | AsyncPeer, SDKSessionPeerConfig]] = []
                for peer_id, config in step.peer_configs.items():
                    sdk_config = SDKSessionPeerConfig(
                        **config.model_dump(exclude_none=True)
                    )
                    peer_list.append((peer_id, sdk_config))
                await session.add_peers(peer_list)

        elif isinstance(step, AddMessageAction):
            session = await self.client.session(id=step.session_id)
            peer = await self.client.peer(id=step.peer_id)
            # TODO: NOT CURRENTLY RESPECTING MESSAGE CONFIG

            config = (
                cast(
                    Configuration, cast(Any, step.config.model_dump(exclude_none=True))
                )
                if step.config
                else None
            )

            await session.add_messages(
                [peer.message(step.content, created_at=step.created_at, config=config)]
            )

        elif isinstance(step, AddMessagesAction):
            session = await self.client.session(id=step.session_id)
            msgs: list[MessageCreateParam] = []
            for msg_item in step.messages:
                peer = await self.client.peer(id=msg_item.peer_id)
                # TODO: NOT CURRENTLY RESPECTING MESSAGE CONFIG

                config = (
                    cast(
                        Configuration,
                        cast(Any, msg_item.config.model_dump(exclude_none=True)),
                    )
                    if msg_item.config
                    else None
                )

                msgs.append(
                    peer.message(
                        msg_item.content,
                        created_at=msg_item.created_at,
                        config=config,
                    )
                )
            await session.add_messages(msgs)

        elif isinstance(step, WaitAction):
            if step.duration:
                await asyncio.sleep(step.duration)
            if step.target == "queue_empty":
                await self.wait_for_queue(step.timeout)

        elif isinstance(step, TriggerDreamAction):
            # Use the core SDK to trigger a dream
            await self.client.core.workspaces.trigger_dream(
                workspace_id=self.client.workspace_id,
                observer=step.observer,
                observed=step.observed,
                dream_type=step.dream_type.value,
            )

        elif isinstance(step, QueryAction):
            result = await self.perform_query(step)
            for assertion in step.assertions:
                await self.check_assertion(result, assertion)

    async def wait_for_queue(self, timeout: int):
        # Poll deriver status
        # Wait for potential background tasks to enqueue
        await asyncio.sleep(1)
        start = time.time()
        while time.time() - start < timeout:
            status: DeriverStatus = await self.client.get_deriver_status()
            # status structure from schema: DeriverStatus with pending_work_units, in_progress_work_units
            if status.pending_work_units == 0 and status.in_progress_work_units == 0:
                return
            await asyncio.sleep(1)
        raise TimeoutError("Deriver queue did not empty within timeout")

    async def perform_query(self, step: QueryAction) -> Any:
        if step.target == "chat":
            if not step.observer_peer_id:
                raise ValueError("observer_peer_id required for chat")
            if step.input is None:
                raise ValueError("input required for chat")

            peer = await self.client.peer(id=step.observer_peer_id)

            response = await peer.chat(
                step.input, session_id=step.session_id, target=step.observed_peer_id
            )
            return response

        elif step.target == "get_context":
            if not step.session_id:
                raise ValueError("session_id required for get_context")
            session: AsyncSession = await self.client.session(id=step.session_id)
            context: SessionContext = await session.get_context(
                summary=step.summary, tokens=step.max_tokens
            )
            # Return the whole context object
            return context

        elif step.target == "get_peer_card":
            if not step.observer_peer_id:
                raise ValueError("peer_id required for get_peer_card")

            peer = await self.client.peer(id=step.observer_peer_id)
            card = await peer.card(
                step.observed_peer_id
                if step.observed_peer_id
                else step.observer_peer_id
            )
            return {"peer_card": card if card else None}

        elif step.target == "get_representation":
            if not step.observer_peer_id:
                raise ValueError("observer_peer_id required for get_representation")

            peer = await self.client.peer(id=step.observer_peer_id)
            representation = await peer.working_rep(
                step.session_id, target=step.observed_peer_id, search_query=step.input
            )
            return representation

        return None

    async def check_assertion(self, result: Any, assertion: Any):
        result_str = str(result)

        if isinstance(assertion, LLMJudgeAssertion):
            if not self.anthropic:
                raise ValueError("Anthropic client required for LLM judge")

            prompt = f"""
            You are evaluating a test result.

            Task: {assertion.prompt}

            Actual Result:
            {result_str}

            Use the submit_verdict tool to submit your decision.
            """

            resp = await self.anthropic.messages.create(
                model=JUDGE_MODEL,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
                tools=[
                    {
                        "name": "submit_verdict",
                        "description": "Submit the verdict of the test evaluation.",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "passed": {
                                    "type": "boolean",
                                    "description": "Whether the test result meets the requirement.",
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Explanation of why the result passed or failed.",
                                },
                            },
                            "required": ["passed", "reasoning"],
                        },
                    }
                ],
                tool_choice={"type": "tool", "name": "submit_verdict"},
            )

            tool_use = next(
                (block for block in resp.content if block.type == "tool_use"), None
            )

            if not tool_use:
                raise TestExecutionError(
                    f"No tool use in judge response: {resp.content}"
                )

            data: object = tool_use.input
            if not isinstance(data, dict):
                raise TestExecutionError(f"Tool input is not a dict: {data}")

            typed_data = cast(dict[str, Any], data)
            passed: bool = typed_data.get("passed", False)
            if passed != assertion.pass_if:
                raise TestExecutionError(
                    f"LLM Judge failed: {typed_data.get('reasoning')}"
                )

        elif isinstance(assertion, ContainsAssertion):
            text = result_str if assertion.case_sensitive else result_str.lower()
            target = (
                assertion.text if assertion.case_sensitive else assertion.text.lower()
            )
            if target not in text:
                raise TestExecutionError(f"Result did not contain '{assertion.text}'")

        elif isinstance(assertion, NotContainsAssertion):
            text = result_str if assertion.case_sensitive else result_str.lower()
            target = (
                assertion.text if assertion.case_sensitive else assertion.text.lower()
            )
            if target in text:
                raise TestExecutionError(
                    f"Result contained forbidden '{assertion.text}'"
                )

        elif isinstance(assertion, ExactMatchAssertion):
            if result_str != assertion.text:
                raise TestExecutionError(
                    f"Exact match failed. Expected '{assertion.text}', got '{result_str}'"
                )

        elif isinstance(assertion, JsonMatchAssertion):
            # This implies result is a dict or json string
            result_dict: dict[str, Any]
            if isinstance(result, str):
                result_dict = json.loads(result)
            else:
                # Try model_dump if pydantic
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump()
                else:
                    result_dict = result

            if assertion.key_value_pairs:
                for k, v in assertion.key_value_pairs.items():
                    if k not in result_dict:
                        raise TestExecutionError(f"Key '{k}' missing from result")
                    if result_dict[k] != v:
                        raise TestExecutionError(
                            f"Value mismatch for '{k}': expected {v}, got {result_dict[k]}"
                        )


class UnifiedTestRunner:
    def __init__(
        self,
        tests_dir: Path | None = None,
        test_file: Path | None = None,
        honcho_port: int = 9000,
        api_port: int = 9001,
        redis_port: int = 6379,
    ):
        if not tests_dir and not test_file:
            raise ValueError("Either tests_dir or test_file must be provided")
        if tests_dir and test_file:
            raise ValueError("Cannot specify both tests_dir and test_file")

        self.tests_dir: Path | None = tests_dir
        self.test_file: Path | None = test_file
        self.harness: HonchoHarness = HonchoHarness(
            db_port=honcho_port,
            api_port=api_port,
            redis_port=redis_port,
            project_root=Path.cwd(),
        )
        self.api_key: str | None = os.getenv("LLM_ANTHROPIC_API_KEY")
        self.anthropic: AsyncAnthropic | None = (
            AsyncAnthropic(api_key=self.api_key) if self.api_key else None
        )

    async def run(self):
        try:
            # 1. Start Harness
            logger.info("Starting Honcho Harness...")
            self.harness.create_temp_docker_compose()
            # Setup .env for harness
            if (self.harness.project_root / ".env").exists():
                self.harness.backup_env_file()

            if not self.harness.temp_dir:
                raise RuntimeError("Harness temp dir not created")

            temp_env = self.harness.temp_dir / ".env"
            with open(temp_env, "w") as f:
                for k, v in os.environ.items():
                    f.write(f"{k}={v}\n")

            self.harness.start_database()
            self.harness.start_redis()
            if not self.harness.wait_for_database():
                raise RuntimeError("DB failed to start")
            if not self.harness.wait_for_redis():
                raise RuntimeError("Redis failed to start")

            await self.harness.init_cache()
            self.harness.provision_database()
            self.harness.verify_empty_database()

            self.harness.start_fastapi_server()
            if not self.harness.wait_for_fastapi():
                raise RuntimeError("API failed to start")

            self.harness.start_deriver()

            # Start output streaming threads for each process
            for name, process in self.harness.processes:
                thread = threading.Thread(
                    target=self.harness.stream_process_output,
                    args=(name, process),
                    daemon=True,
                )
                thread.start()
                self.harness.output_threads.append(thread)

            # Give services a moment to settle
            await asyncio.sleep(2)

            # 2. Load Tests
            if self.test_file:
                test_files = [self.test_file]
            else:
                if not self.tests_dir:
                    raise ValueError("tests_dir must be set if test_file is not")
                test_files = sorted(list(self.tests_dir.glob("*.json")))

            results: dict[str, tuple[str, float]] = {}

            logger.info(f"Found {len(test_files)} test(s)")

            # 3. Execute Tests
            client = AsyncHoncho(
                base_url=f"http://localhost:{self.harness.api_port}",
                workspace_id="default",  # Will be overridden per test
            )

            executor = UnifiedTestExecutor(client, self.anthropic)

            suite_start_time = time.time()

            for test_file in test_files:
                test_start_time = time.time()
                try:
                    # Use filename (without extension) as test name
                    test_name = test_file.stem

                    with open(test_file) as f:
                        data = json.load(f)
                    test_def = TestDefinition(**data)

                    executor.client = AsyncHoncho(
                        base_url=f"http://localhost:{self.harness.api_port}",
                        workspace_id=f"test_{test_name}_{int(time.time())}",
                    )

                    success = await executor.execute(test_def, test_name)
                    test_duration = time.time() - test_start_time
                    results[test_file.name] = (
                        "PASS" if success else "FAIL",
                        test_duration,
                    )

                except ValidationError as e:
                    logger.error(f"Schema validation failed for {test_file}: {e}")
                    test_duration = time.time() - test_start_time
                    results[test_file.name] = ("INVALID SCHEMA", test_duration)
                except Exception as e:
                    logger.error(
                        f"Test {test_file.name} failed with error: {e}", exc_info=True
                    )
                    test_duration = time.time() - test_start_time
                    results[test_file.name] = (f"ERROR: {str(e)}", test_duration)

            total_suite_time = time.time() - suite_start_time

            # 4. Report
            print("\n" + "=" * 60)
            print("TEST RESULTS")
            print("=" * 60)

            failed_count = 0
            total_count = len(results)

            # Calculate max name length for alignment
            max_name_length = max(len(name) for name in results) if results else 0

            for name, (status, duration) in results.items():
                duration_str = f"({duration:.2f}s)"
                if status == "PASS":
                    print(
                        f"{name:<{max_name_length}} {GREEN}{status:<15}{RESET} {duration_str}"
                    )
                else:
                    print(
                        f"{name:<{max_name_length}} {RED}{status:<15}{RESET} {duration_str}"
                    )
                    failed_count += 1

            print("=" * 60)
            print(f"\n{failed_count} failed / {total_count} total")
            print(f"Total execution time: {total_suite_time:.2f}s")
            print("=" * 60)

        finally:
            # 5. Cleanup
            logger.info("Cleaning up harness...")
            await self.harness.cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, default="tests/unified/test_cases")
    args = parser.parse_args()

    runner = UnifiedTestRunner(Path(args.test_dir))
    asyncio.run(runner.run())
