"""
Honcho Test Runner

A script that executes JSON-formatted tests against a running Honcho instance.
This script:
1. Loads test definitions from JSON files
2. Creates a workspace for each test
3. Adds all messages to sessions
4. Waits for the deriver queue to be empty
5. Executes queries and judges responses using an LLM
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

import tiktoken
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from honcho import Honcho
from honcho.session import SessionPeerConfig

load_dotenv()


class SessionResult(TypedDict):
    """Type definition for session creation results."""

    name: str
    message_count: int


class QueryResult(TypedDict):
    """Type definition for query execution results."""

    query: str
    expected_response: str
    actual_response: str
    session: str | None
    observer: str | None
    target: str | None
    judgment: dict[str, Any]


class TestResult(TypedDict):
    """Type definition for test execution results."""

    test_name: str
    workspace_id: str
    sessions_created: list[SessionResult]
    queries_executed: list[QueryResult]
    passed: bool
    error: str | None
    start_time: float
    end_time: float
    duration_seconds: float


class TestRunner:
    """
    Executes JSON tests against a Honcho instance.
    """

    def __init__(
        self,
        honcho_url: str = "http://localhost:8000",
        anthropic_api_key: str | None = None,
    ):
        """
        Initialize the test runner.

        Args:
            honcho_url: URL of the running Honcho instance
            anthropic_api_key: Anthropic API key for judging responses
        """
        self.honcho_url: str = honcho_url
        self.anthropic_api_key: str | None = anthropic_api_key

        # Configure logging
        logging.basicConfig(
            level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Suppress HTTP request logs from the Honcho SDK
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)

        if self.anthropic_api_key:
            self.anthropic_client: AsyncAnthropic = AsyncAnthropic(
                api_key=self.anthropic_api_key
            )
        else:
            api_key = os.getenv("LLM_ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("LLM_ANTHROPIC_API_KEY is not set")
            self.anthropic_client = AsyncAnthropic(api_key=api_key)

    def load_test_file(self, test_file: Path) -> dict[str, Any]:
        """
        Load a test definition from a JSON file.

        Args:
            test_file: Path to the JSON test file

        Returns:
            Test definition dictionary
        """
        with open(test_file) as f:
            return json.load(f)

    def create_honcho_client(self, workspace_id: str) -> Honcho:
        """
        Create a Honcho client for a specific workspace.

        Args:
            workspace_id: Workspace ID for the test

        Returns:
            Honcho client instance
        """
        return Honcho(
            environment="local", workspace_id=workspace_id, base_url=self.honcho_url
        )

    async def wait_for_deriver_queue_empty(
        self, honcho_client: Honcho, session_id: str | None = None
    ) -> bool:
        """
        Wait for the deriver queue to be empty.

        Args:
            honcho_client: Honcho client instance
            timeout: Maximum time to wait in seconds

        Returns:
            True if queue is empty, False if timeout exceeded
        """
        time.sleep(1)
        try:
            while True:
                status = honcho_client.get_deriver_status(session_id=session_id)
                if (
                    status.in_progress_work_units == 0
                    and status.pending_work_units == 0
                ):
                    break
                time.sleep(0.5)
            return True
        except Exception as e:
            self.logger.warning(f"Error polling deriver status: {e}")
            return False

    async def judge_response(
        self, query: str, expected_response: str, actual_response: str
    ) -> dict[str, Any]:
        """
        Use an LLM to judge if the actual response matches the expected response.

        Args:
            query: The original query
            expected_response: Expected response from the test
            actual_response: Actual response from Honcho

        Returns:
            Judgment result with pass/fail and reasoning
        """
        try:
            system_prompt = """
You are an expert judge evaluating AI responses. Your task is to determine if an actual response contains the core correct information from an expected response.

CRITICAL JUDGING PRINCIPLES:
1. SEMANTIC UNDERSTANDING: Focus on whether the actual response conveys the same core factual information as expected, even if expressed differently
2. FLEXIBLE INTERPRETATION: Accept responses that are longer, more detailed, or use different phrasing as long as they contain the correct core facts
3. CONTEXTUAL REASONING: If the response shows logical reasoning that leads to the correct conclusion, consider it correct even if the path differs
4. CONFLICTING INFORMATION: If a response acknowledges conflicts but correctly identifies the most recent/authoritative information, that should pass
5. IMPLICIT vs EXPLICIT: Accept responses that imply the correct answer through reasoning, not just explicit statements

ONLY FAIL when:
- Core factual information is demonstrably wrong
- The response contradicts the expected information without justification
- Essential information is completely missing with no reasonable inference path

Always respond with valid JSON: {"passed": boolean, "reasoning": "short (1-3 sentences) explanation of why the response is correct or incorrect"}"""

            user_prompt = f"""Query: "{query}"
Expected response: "{expected_response}"
Actual response: "{actual_response}"

Evaluate whether the actual response contains the core correct information from the expected response. Focus on semantic meaning and logical conclusions, not exact phrasing.
"""

            response = await self.anthropic_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=300,
                temperature=0.0,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )

            if not response.content:
                raise ValueError("Anthropic returned empty response")

            content_block = response.content[0]
            judgment_text = getattr(content_block, "text", None)
            if judgment_text is None:
                raise ValueError(
                    f"No text content in response block: {type(content_block)}"
                )

            if judgment_text is None:
                raise ValueError("Anthropic returned empty response")

            # Extract JSON from the response if it's wrapped in markdown
            if "```json" in judgment_text:
                json_start = judgment_text.find("```json") + 7
                json_end = judgment_text.find("```", json_start)
                judgment_text = judgment_text[json_start:json_end].strip()
            elif "```" in judgment_text:
                json_start = judgment_text.find("```") + 3
                json_end = judgment_text.find("```", json_start)
                judgment_text = judgment_text[json_start:json_end].strip()

            judgment = json.loads(judgment_text)

            return judgment

        except Exception as e:
            self.logger.error(f"Error judging response: {e}")
            # Fallback to simple string matching
            is_correct = expected_response.lower() in actual_response.lower()
            return {
                "passed": is_correct,
                "reasoning": f"Fallback string matching due to error: {'Match found' if is_correct else 'No match found'}",
            }

    async def execute_test(self, test_file: Path) -> TestResult:
        """
        Execute a single test file.

        Args:
            test_file: Path to the JSON test file

        Returns:
            Test execution results
        """
        test_name = test_file.stem
        print(f"\033[1mExecuting test {test_name}\033[0m")

        # Load test definition
        test_def = self.load_test_file(test_file)

        # Create workspace for this test
        workspace_id = f"test_{test_name}_{int(time.time())}"
        honcho_client = self.create_honcho_client(workspace_id)

        results: TestResult = {
            "test_name": test_name,
            "workspace_id": workspace_id,
            "sessions_created": [],
            "queries_executed": [],
            "passed": False,
            "error": None,
            "start_time": time.time(),
            "end_time": 0.0,
            "duration_seconds": 0.0,
        }

        try:
            # Step 1: Create sessions and add messages
            sessions = test_def.get("sessions", {})

            # Collect all unique peers from messages and queries
            all_peers: set[str] = set()
            for session_data in sessions.values():
                for msg in session_data.get("messages", []):
                    all_peers.add(msg["peer"])

            # Get queries and collect peers from them
            queries: list[dict[str, Any]] = test_def.get("queries", [])
            for query_dict in queries:
                if "observer" in query_dict:
                    all_peers.add(str(query_dict["observer"]))
                if "target" in query_dict:
                    all_peers.add(str(query_dict["target"]))

            # Identify peers that are observers in queries (need observe_others=True)
            observer_peers: set[str] = set()
            for query_dict in queries:
                if "observer" in query_dict and "target" in query_dict:
                    observer_peers.add(str(query_dict["observer"]))

            # for efficiency, identify peers that are never targets in any query
            # so we can turn off their observe_me flag
            observed_peers: set[str] = set()
            for query_dict in queries:
                if "target" in query_dict:
                    observed_peers.add(str(query_dict["target"]))
                if "observer" in query_dict and "target" not in query_dict:
                    observed_peers.add(str(query_dict["observer"]))

            non_observed_peers: set[str] = all_peers - observed_peers

            # Create all peers first
            peers: dict[str, Any] = {}
            for peer_name in all_peers:
                peers[peer_name] = honcho_client.peer(id=peer_name)

            for session_name, session_data in sessions.items():
                # Create session
                session = honcho_client.session(id=str(session_name))

                print(f"\n  session: {session_name}")

                # Create peer configurations based on requirements
                peer_configs: list[tuple[Any, SessionPeerConfig]] = []
                for peer_name in all_peers:
                    # If this peer is an observer in any *targeted* query, they need to observe others
                    if peer_name in observer_peers:
                        if peer_name in non_observed_peers:
                            config = SessionPeerConfig(
                                observe_me=False, observe_others=True
                            )
                            peer_configs.append((peers[peer_name], config))
                            print(f"    peer config: {peer_name} -> {config}")
                        else:
                            config = SessionPeerConfig(
                                observe_me=True, observe_others=True
                            )
                            peer_configs.append((peers[peer_name], config))
                            print(f"    peer config: {peer_name} -> {config}")
                    elif peer_name in observed_peers:
                        config = SessionPeerConfig(
                            observe_me=True, observe_others=False
                        )
                        peer_configs.append((peers[peer_name], config))
                        print(f"    peer config: {peer_name} -> {config}")
                    else:
                        config = SessionPeerConfig(
                            observe_me=False, observe_others=False
                        )
                        peer_configs.append((peers[peer_name], config))
                        print(f"    peer config: {peer_name} -> {config}")

                session.add_peers(peer_configs)

                # Add messages to session
                messages = session_data.get("messages", [])
                for msg in messages:
                    peer_name: str = msg["peer"]
                    content: str = msg["content"]

                    truncated_content = (
                        content[:140] + "..." if len(content) > 140 else content
                    )
                    print(f"    {peer_name}: {truncated_content}")

                # Add messages to session
                session.add_messages(
                    [peers[msg["peer"]].message(msg["content"]) for msg in messages]
                )

                results["sessions_created"].append(
                    SessionResult(name=str(session_name), message_count=len(messages))
                )

            # Step 2: Execute queries
            all_queries_passed = True

            for i, query_data in enumerate(queries):
                query: str = query_data["query"]
                expected_response: str = query_data["expected_response"]
                session_name: str | None = query_data.get("session")
                observer_name: str | None = query_data.get("observer")
                target_name: str | None = query_data.get("target")

                # Wait for deriver queue to be empty for this session
                queue_empty = await self.wait_for_deriver_queue_empty(
                    honcho_client, session_id=session_name
                )
                if not queue_empty:
                    print(f"Deriver queue never emptied for session {session_name}!!!")
                    sys.exit(1)

                print(f"\n  query {i + 1}: {query}")
                context_parts: list[str] = []
                if session_name:
                    context_parts.append(f"session: {session_name}")
                if observer_name:
                    context_parts.append(f"observer: {observer_name}")
                if target_name:
                    context_parts.append(f"target: {target_name}")
                if context_parts:
                    print("    " + ", ".join(context_parts))

                try:
                    # Determine which peer to use for the query (observer)
                    if observer_name:
                        query_peer = peers[observer_name]
                    else:
                        # Use the first peer from the first session
                        first_session_data = list(sessions.values())[0]
                        first_peer_name: str = first_session_data["messages"][0]["peer"]
                        query_peer = peers[first_peer_name]

                    # Execute chat query
                    if session_name and target_name:
                        response = query_peer.chat(
                            query, session_id=session_name, target=peers[target_name]
                        )
                    elif session_name:
                        response = query_peer.chat(query, session_id=session_name)
                    elif target_name:
                        response = query_peer.chat(query, target=peers[target_name])
                    else:
                        response = query_peer.chat(query)

                    actual_response: str = (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )

                    # Judge the response
                    judgment = await self.judge_response(
                        query, expected_response, actual_response
                    )

                    query_result: QueryResult = {
                        "query": query,
                        "expected_response": expected_response,
                        "actual_response": actual_response,
                        "session": session_name,
                        "observer": observer_name,
                        "target": target_name,
                        "judgment": judgment,
                    }

                    results["queries_executed"].append(query_result)

                    print(
                        "    judgment: \033[1m\033[32mPASS\033[0m"
                        if judgment["passed"]
                        else "    judgment: \033[1m\033[31mFAIL\033[0m"
                    )
                    if not judgment["passed"]:
                        # if failed, print
                        print(f"    got response: \033[3m{actual_response}\033[0m")
                        print(f"    expected: {expected_response}")
                    else:
                        # if passed, just log
                        self.logger.info(
                            f"    got response: \033[3m{actual_response}\033[0m"
                        )
                        self.logger.info(f"    expected: {expected_response}")
                    print(f"    reasoning: {judgment['reasoning']}")

                    # Track if all queries pass
                    if not judgment["passed"]:
                        all_queries_passed = False

                except Exception as e:
                    self.logger.error(f"Error executing query {i + 1}: {e}")
                    query_result = QueryResult(
                        query=query,
                        expected_response=expected_response,
                        actual_response=f"ERROR: {e}",
                        session=session_name,
                        observer=observer_name,
                        target=target_name,
                        judgment={
                            "passed": False,
                            "reasoning": f"Query execution failed: {e}",
                        },
                    )
                    results["queries_executed"].append(query_result)
                    all_queries_passed = False

            # Step 3: Execute get_context calls
            get_context_calls = test_def.get("get_context_calls", [])
            for i, get_context_call in enumerate(get_context_calls):
                print(f"\n  get_context call #{i + 1}")
                session_name = str(get_context_call["session"])
                summary = get_context_call["summary"]
                max_tokens: int | None = get_context_call.get("max_tokens")
                session = honcho_client.session(id=session_name)

                # Wait for deriver queue to be empty for this session
                queue_empty = await self.wait_for_deriver_queue_empty(
                    honcho_client, session_id=session_name
                )
                if not queue_empty:
                    print(f"Deriver queue never emptied for session {session_name}!!!")
                    sys.exit(1)

                session_context = session.get_context(
                    summary=summary, tokens=max_tokens
                )

                tokenizer = tiktoken.get_encoding("cl100k_base")
                summary_tokens = len(tokenizer.encode(session_context.summary))
                print(f"    summary: {session_context.summary}")

                got_tokens = summary_tokens
                for message in session_context.messages:
                    got_tokens += message.token_count
                print(f"    max tokens: {max_tokens}")
                print(
                    f"    got token count: {got_tokens} (summary: {summary_tokens}, messages: {got_tokens - summary_tokens} in {len(session_context.messages)} messages)"
                )
                if max_tokens and got_tokens > max_tokens:
                    all_queries_passed = False

            # Determine if test passed (all queries and get_context calls must pass)
            results["passed"] = all_queries_passed
            results["end_time"] = time.time()
            results["duration_seconds"] = results["end_time"] - results["start_time"]

            print(
                f"\nTest {test_name} completed. Status: {'PASS' if results['passed'] else 'FAIL'} (Duration: {results['duration_seconds']:.2f}s)"
            )

        except Exception as e:
            self.logger.error(f"Error executing test {test_name}: {e}")
            results["error"] = str(e)
            results["passed"] = False
            results["end_time"] = time.time()
            results["duration_seconds"] = results["end_time"] - results["start_time"]

        return results

    async def run_all_tests(self, tests_dir: Path) -> list[TestResult]:
        """
        Run all tests in a directory.

        Args:
            tests_dir: Directory containing JSON test files

        Returns:
            List of test results
        """
        test_files = list(tests_dir.glob("*.json"))
        print(f"found {len(test_files)} test files in {tests_dir}")

        all_results: list[TestResult] = []

        for test_file in test_files:
            result = await self.execute_test(test_file)
            all_results.append(result)

            # Print summary for this test
            print(f"\n{'=' * 60}")
            print(f"Test: {result['test_name']}")
            print(f"Status: {'PASS' if result['passed'] else 'FAIL'}")
            print(f"Duration: {result['duration_seconds']:.2f}s")
            print(f"{'=' * 60}\n")

        return all_results

    def print_summary(self, results: list[TestResult]) -> None:
        """
        Print a summary of all test results.

        Args:
            results: List of test results
        """
        print(f"\n{'=' * 80}")
        print("TEST EXECUTION SUMMARY")
        print(f"{'=' * 80}")

        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.get("passed", False))
        failed_tests = total_tests - passed_tests
        total_test_time = sum(r["duration_seconds"] for r in results)

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests / total_tests) * 100:.1f}%")
        print(f"Total Test Time: {total_test_time:.2f}s")

        print("\nDetailed Results:")
        print(f"{'Test Name':<20} {'Status':<8} {'Duration':<10} {'Workspace ID':<30}")
        print(f"{'-' * 20} {'-' * 8} {'-' * 10} {'-' * 30}")

        for result in results:
            test_name = result["test_name"]
            status = "PASS" if result.get("passed", False) else "FAIL"
            duration = f"{result['duration_seconds']:.2f}s"
            workspace = result["workspace_id"]

            print(f"{test_name:<20} {status:<8} {duration:<10} {workspace:<30}")

        print(f"{'=' * 80}")


async def main() -> int:
    """
    Main entry point for the test runner.
    """
    parser = argparse.ArgumentParser(
        description="Run JSON tests against a Honcho instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --tests-dir tests/bench/tests                    # Run all tests
  %(prog)s --tests-dir tests/bench/tests --test 1.json      # Run specific test
  %(prog)s --honcho-url http://localhost:8000               # Custom Honcho URL
        """,
    )

    parser.add_argument(
        "--tests-dir",
        type=Path,
        default=Path("tests/bench/tests"),
        help="Directory containing JSON test files (default: tests/bench/tests)",
    )

    parser.add_argument(
        "--test",
        type=str,
        help="Run a specific test file by name (e.g., '1.json') (optional)",
    )

    parser.add_argument(
        "--honcho-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the running Honcho instance (default: http://localhost:8000)",
    )

    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        help="Anthropic API key for response judging (optional)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout for deriver queue to empty in seconds (default: 60)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.test:
        test_file_path = args.tests_dir / args.test
        if not test_file_path.exists():
            print(f"Error: Test file {test_file_path} does not exist")
            return 1

    if not args.tests_dir.exists():
        print(f"Error: Tests directory {args.tests_dir} does not exist")
        return 1

    # Create test runner
    runner = TestRunner(
        honcho_url=args.honcho_url, anthropic_api_key=args.anthropic_api_key
    )

    try:
        if args.test:
            # Run single test
            test_file_path = args.tests_dir / args.test
            result = await runner.execute_test(test_file_path)
            runner.print_summary([result])
        else:
            # Run all tests
            results = await runner.run_all_tests(args.tests_dir)
            runner.print_summary(results)

        return 0

    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
