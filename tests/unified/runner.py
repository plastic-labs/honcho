import asyncio
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import httpx
from anthropic import AsyncAnthropic
from honcho.api_types import (
    MessageCreateParams,
    QueueStatusResponse,
)
from honcho.api_types import (
    SessionConfiguration as SDKSessionConfiguration,
)
from honcho.api_types import (
    WorkspaceConfiguration as SDKWorkspaceConfiguration,
)
from honcho.session import Session
from honcho.session_context import SessionContext
from pydantic import ValidationError

# Adjust path to allow imports from tests.bench
sys.path.insert(0, str(Path(__file__).parents[2]))

from honcho import Honcho
from honcho.base import PeerBase
from honcho.session import SessionPeerConfig as SDKSessionPeerConfig

from tests.bench.harness import HonchoHarness
from tests.unified.schema import (
    AddMessageAction,
    AddMessagesAction,
    ContainsAssertion,
    CreateScopeAction,
    CreateSessionAction,
    ExactMatchAssertion,
    JsonMatchAssertion,
    LLMJudgeAssertion,
    NotContainsAssertion,
    QueryAction,
    ScheduleDreamAction,
    SetSessionConfigAction,
    SetWorkspaceConfigAction,
    TestDefinition,
    WaitAction,
)

# Override log level with UNIFIED_TEST_LOG_LEVEL env var if needed (e.g., INFO, DEBUG)
logging.basicConfig(
    level=getattr(
        logging, os.getenv("UNIFIED_TEST_LOG_LEVEL", "WARNING").upper(), logging.WARNING
    ),
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

JUDGE_MODEL: str = "claude-haiku-4-5"


class TestExecutionError(Exception):
    pass


# Discord rejects a webhook payload whose content exceeds this with a 400.
DISCORD_MAX_CONTENT = 2000


def clamp_lines(lines: list[str], limit: int) -> str:
    """Join `lines` within `limit`, dropping the longest ones first if needed.

    Whole lines rather than characters: a presigned URL cut in half is useless
    and renders as broken markdown. Longest-first rather than last-first because
    the only lines that can blow the budget are presigned URLs — dropping one of
    those keeps every short, always-valid link, the Actions run link above all,
    instead of losing them to a long URL that merely came first.
    """
    kept = list(range(len(lines)))

    def size() -> int:
        return sum(len(lines[i]) for i in kept) + max(0, len(kept) - 1)

    while kept and size() > limit:
        kept.remove(max(kept, key=lambda i: len(lines[i])))
    return "\n".join(lines[i] for i in sorted(kept))


async def send_discord_message(webhook_url: str, lines: list[str]) -> None:
    """Send a report to Discord via webhook.

    Clamped to Discord's content limit here rather than at the call site: a
    presigned URL carries an OIDC session token and can run past a thousand
    characters on its own, and a 400 loses the whole notification.
    """
    try:
        async with httpx.AsyncClient() as client:
            content = clamp_lines(lines, DISCORD_MAX_CONTENT)
            response = await client.post(webhook_url, json={"content": content})
            response.raise_for_status()
            logger.info("Discord notification sent successfully")
    except Exception:
        logger.exception("Failed to send Discord notification")


@dataclass
class StepFailure:
    """Why a test stopped: the step that raised, and what it said."""

    step_index: int
    step_type: str
    message: str

    def describe(self) -> str:
        return f"step {self.step_index} ({self.step_type}): {self.message}"


@dataclass
class TestOutcome:
    """One test's result. `failure` carries the reason whenever status isn't PASS."""

    # Not a pytest case despite the name; keeps collection from warning on it.
    __test__: ClassVar[bool] = False

    status: str
    duration: float
    failure: StepFailure | None = None


@dataclass
class RunArtifact:
    """One uploaded file: its S3 key, and a presigned URL when one could be made."""

    key: str
    url: str | None = None


@dataclass
class RunArtifacts:
    """Artifacts published for a run. Any field is None when its upload failed."""

    results: RunArtifact | None = None
    traces: RunArtifact | None = None


# 3 days. Long enough to survive a weekend before someone reads the report.
PRESIGN_EXPIRY_SECONDS = 259200


def presign(s3_client: Any, bucket: str, key: str) -> RunArtifact:
    """Wrap an uploaded key with a presigned URL, or just the key if signing fails."""
    try:
        url: str = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=PRESIGN_EXPIRY_SECONDS,
        )
        return RunArtifact(key=key, url=url)
    except Exception as e:
        logger.warning(f"Could not generate S3 presigned URL for {key}: {e}")
        return RunArtifact(key=key)


def artifact_line(label: str, artifact: RunArtifact | None) -> list[str]:
    """One markdown line for an artifact: a link when presigned, the key otherwise."""
    if artifact is None:
        return []
    if artifact.url:
        return [f"[{label}]({artifact.url}) — `{artifact.key}`"]
    return [f"{label}: `{artifact.key}`"]


def artifact_lines(artifacts: RunArtifacts) -> list[str]:
    """Both uploaded artifacts. The reasoning traces carry the full prompts and
    model outputs for the run, which is what a failure usually needs to diagnose.
    """
    return artifact_line("View Complete Results", artifacts.results) + artifact_line(
        "Reasoning traces", artifacts.traces
    )


def gha_run_lines() -> list[str]:
    """Link to this run's Actions page, which hosts the job summary.

    That summary carries the per-test failure reasons in full, so the Discord
    message can stay short and point at it instead of restating them.
    """
    run_id = os.getenv("GITHUB_RUN_ID")
    repository = os.getenv("GITHUB_REPOSITORY")
    if not run_id or not repository:
        return []
    server = os.getenv("GITHUB_SERVER_URL", "https://github.com")
    return [f"[View GHA]({server}/{repository}/actions/runs/{run_id})"]


def failure_lines(results: dict[str, "TestOutcome"]) -> list[str]:
    """One markdown bullet per failing test, naming the step and the reason."""
    failed = [(name, o) for name, o in results.items() if o.status != "PASS"]
    if not failed:
        return []
    lines = ["", "**Failures**"]
    for name, outcome in failed:
        reason = outcome.failure.describe() if outcome.failure else outcome.status
        lines.append(f"- `{name}` — {reason}")
    return lines


def write_job_summary(lines: list[str]) -> None:
    """Append a markdown block to the GitHub Actions job summary; a no-op locally."""
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    try:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    except OSError as e:
        logger.warning(f"Could not write job summary: {e}")


async def save_results_to_s3(
    results: dict[str, TestOutcome],
    failed_count: int,
    total_count: int,
    execution_time: float,
) -> RunArtifacts:
    """Save comprehensive test results and reasoning traces to S3."""
    try:
        import boto3

        s3_bucket = "honcho-unified-tests"
        s3_prefix = "unified-test-results"
        aws_region = "us-east-1"

        # AWS credentials are configured via OIDC in GitHub Actions
        # Check if boto3 can access credentials (either from environment or OIDC)
        try:
            session = boto3.Session()
            credentials = session.get_credentials()  # pyright: ignore
            if not credentials:
                logger.warning("No AWS credentials available, skipping S3 upload")
                return RunArtifacts()
        except Exception as e:
            logger.warning(f"Could not verify AWS credentials: {e}, skipping S3 upload")
            return RunArtifacts()

        # Create comprehensive results object
        timestamp = datetime.now(UTC).isoformat()
        github_run_id = os.getenv("GITHUB_RUN_ID", "local")
        github_run_attempt = os.getenv("GITHUB_RUN_ATTEMPT", "1")
        github_sha = os.getenv("GITHUB_SHA", "unknown")
        github_ref = os.getenv("GITHUB_REF_NAME", "unknown")

        comprehensive_results = {
            "timestamp": timestamp,
            "summary": {
                "total": total_count,
                "passed": total_count - failed_count,
                "failed": failed_count,
                "execution_time": execution_time,
            },
            "metadata": {
                "github_run_id": github_run_id,
                "github_run_attempt": github_run_attempt,
                "github_sha": github_sha,
                "github_ref": github_ref,
            },
            "tests": [
                {
                    "name": name,
                    "status": outcome.status,
                    "duration": outcome.duration,
                    # The reason a test failed lives only in the job log otherwise,
                    # where secret masking can render it unreadable.
                    "failure": (
                        {
                            "step_index": outcome.failure.step_index,
                            "step_type": outcome.failure.step_type,
                            "message": outcome.failure.message,
                        }
                        if outcome.failure
                        else None
                    ),
                }
                for name, outcome in results.items()
            ],
        }

        # One "folder" per run: <prefix>/<date>/<run>/ holding results.json plus
        # the reasoning-trace file(s), so a run's summary and full LLM I/O live together.
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        sha_short = github_sha[:7] if github_sha != "unknown" else "unknown"
        ref_slug = github_ref.replace("/", "-")  # branch names may contain "/"
        run_slug = f"{ref_slug}-{sha_short}-{github_run_id}-{github_run_attempt}"
        run_prefix = f"{s3_prefix}/{date_str}/{run_slug}"
        results_key = f"{run_prefix}/results.json"

        s3_client = boto3.client("s3", region_name=aws_region)  # pyright: ignore
        s3_client.put_object(  # pyright: ignore
            Bucket=s3_bucket,
            Key=results_key,
            Body=json.dumps(comprehensive_results, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(f"Saved test results to S3 key {results_key}")

        # Upload the reasoning traces (full LLM/deriver I/O) captured this run. The
        # API and deriver both append to REASONING_TRACES_FILE (file-locked). Use
        # upload_file so large trace files stream via multipart instead of buffering.
        traces: RunArtifact | None = None
        traces_path_str = os.getenv("REASONING_TRACES_FILE")
        if traces_path_str:
            traces_path = Path(traces_path_str)
            if traces_path.is_file() and traces_path.stat().st_size > 0:
                traces_key = f"{run_prefix}/{traces_path.name}"
                try:
                    s3_client.upload_file(  # pyright: ignore
                        str(traces_path),
                        s3_bucket,
                        traces_key,
                        ExtraArgs={"ContentType": "application/x-ndjson"},
                    )
                    logger.info(f"Saved reasoning traces to S3 key {traces_key}")
                    traces = presign(s3_client, s3_bucket, traces_key)
                except Exception as e:
                    logger.error(
                        f"Failed to upload reasoning traces: {e}", exc_info=True
                    )
            else:
                logger.warning(
                    f"REASONING_TRACES_FILE={traces_path} is missing or empty; no traces uploaded"
                )

        return RunArtifacts(
            results=presign(s3_client, s3_bucket, results_key), traces=traces
        )

    except Exception as e:
        logger.error(f"Failed to save results to S3: {e}", exc_info=True)
        return RunArtifacts()


class UnifiedTestExecutor:
    def __init__(
        self,
        honcho_client: Honcho,
        anthropic_client: AsyncAnthropic | None,
    ):
        self.client: Honcho = honcho_client
        self.anthropic: AsyncAnthropic | None = anthropic_client

    # --- raw HTTP -----------------------------------------------------------
    # Some surfaces (scopes, the `scope` read option) exist in the API before the
    # published SDK exposes them. Calling them directly also tests the contract
    # the SDK is generated from, so a wrong status or shape surfaces here instead
    # of being masked by client-side validation.

    @property
    def workspace_id(self) -> str:
        workspace_id = getattr(self.client, "workspace_id", None)
        if not workspace_id:
            raise ValueError("Honcho client has no workspace_id")
        return str(workspace_id)

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Call a /v3 workspace-scoped path directly, raising on error status."""
        url = f"{str(self.client.base_url).rstrip('/')}/v3/workspaces/{self.workspace_id}{path}"
        # Carry the same credential the SDK resolved (from `HONCHO_API_KEY`, unless
        # passed explicitly). The harness sets no AUTH vars of its own, so auth is
        # off by default — but it inherits `AUTH_USE_AUTH` from the environment,
        # and these raw calls are the only ones here that would not be authorized.
        headers: dict[str, str] = dict(kwargs.pop("headers", None) or {})
        api_key = getattr(getattr(self.client, "_http", None), "api_key", None)
        if api_key:
            headers.setdefault("Authorization", f"Bearer {api_key}")
        async with httpx.AsyncClient(timeout=120.0) as raw:
            response = await raw.request(method, url, headers=headers, **kwargs)
        if response.is_error:
            raise AssertionError(
                f"{method} {path} failed: {response.status_code} {response.text[:400]}"
            )
        return response

    async def execute(
        self, test_def: TestDefinition, test_name: str
    ) -> StepFailure | None:
        """Run every step. Returns None on success, or the failure that stopped it."""
        logger.info(f"Starting test: {test_name}")

        # 1. Apply workspace config if present
        if test_def.workspace_config:
            sdk_config = SDKWorkspaceConfiguration.model_validate(
                test_def.workspace_config.model_dump(exclude_none=True)
            )
            await self.client.aio.set_configuration(sdk_config)

        for i, step in enumerate(test_def.steps):
            logger.info(f"Executing step {i + 1}: {step.step_type}")
            try:
                await self.execute_step(step)
            except Exception as e:
                logger.error(f"Step {i + 1} failed: {e}", exc_info=False)
                return StepFailure(
                    step_index=i + 1, step_type=step.step_type, message=str(e)
                )

        logger.info(f"Test {test_name} PASSED")
        return None

    async def execute_step(self, step: Any):
        if isinstance(step, SetWorkspaceConfigAction):
            sdk_config = SDKWorkspaceConfiguration.model_validate(
                step.config.model_dump(exclude_none=True)
            )
            await self.client.aio.set_configuration(sdk_config)

        elif isinstance(step, SetSessionConfigAction):
            session = await self.client.aio.session(id=step.session_id)
            sdk_config = SDKSessionConfiguration.model_validate(
                step.config.model_dump(exclude_none=True)
            )
            await session.aio.set_configuration(sdk_config)

        elif isinstance(step, CreateSessionAction):
            sdk_config = (
                SDKSessionConfiguration.model_validate(
                    step.config.model_dump(exclude_none=True)
                )
                if step.config
                else None
            )
            session = await self.client.aio.session(
                id=step.session_id,
                configuration=sdk_config,
            )

            if step.peer_configs:
                peer_list: list[tuple[str | PeerBase, SDKSessionPeerConfig]] = []
                for peer_id, peer_config in step.peer_configs.items():
                    sdk_config = SDKSessionPeerConfig(
                        **peer_config.model_dump(exclude_none=True)
                    )
                    peer_list.append((peer_id, sdk_config))
                await session.aio.add_peers(peer_list)

        elif isinstance(step, AddMessageAction):
            session = await self.client.aio.session(id=step.session_id)
            peer = await self.client.aio.peer(id=step.peer_id)
            # TODO: NOT CURRENTLY RESPECTING MESSAGE CONFIG

            config: dict[str, Any] | None = (
                step.config.model_dump(exclude_none=True) if step.config else None
            )

            await session.aio.add_messages(
                [
                    peer.message(
                        step.content, created_at=step.created_at, configuration=config
                    )
                ]
            )

        elif isinstance(step, AddMessagesAction):
            session = await self.client.aio.session(id=step.session_id)
            msgs: list[MessageCreateParams] = []
            for msg_item in step.messages:
                peer = await self.client.aio.peer(id=msg_item.peer_id)
                # TODO: NOT CURRENTLY RESPECTING MESSAGE CONFIG

                config = (
                    msg_item.config.model_dump(exclude_none=True)
                    if msg_item.config
                    else None
                )

                msgs.append(
                    peer.message(
                        msg_item.content,
                        created_at=msg_item.created_at,
                        configuration=config,
                    )
                )
            await session.aio.add_messages(msgs)

        elif isinstance(step, CreateScopeAction):
            await self._request("POST", "/scopes", json={"id": step.scope_id})
            if step.session_ids:
                await self._request(
                    "POST",
                    f"/scopes/{step.scope_id}/sessions",
                    json={"session_ids": step.session_ids},
                )

        elif isinstance(step, WaitAction):
            if step.duration:
                await asyncio.sleep(step.duration)
            if step.target == "queue_empty":
                # Flush is process-wide, not per-step: the harness starts the
                # deriver with DERIVER_FLUSH_ENABLED=true so batches never wait
                # on the token threshold. See tests/bench/harness.py.
                await self.wait_for_queue(step.timeout)

        elif isinstance(step, ScheduleDreamAction):
            await self.client.aio.schedule_dream(
                observer=step.observer,
                session=step.session_id,
                observed=step.observed,
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
            status: QueueStatusResponse = await self.client.aio.queue_status()
            # status structure from schema: DeriverStatus with pending_work_units, in_progress_work_units
            if status.pending_work_units == 0 and status.in_progress_work_units == 0:
                return
            await asyncio.sleep(1)
        raise TimeoutError("Deriver queue did not empty within timeout")

    async def perform_query(self, step: QueryAction) -> Any:
        if step.target == "workspace_chat":
            if step.input is None:
                raise ValueError("input required for workspace_chat")
            return await self.client.aio.chat(
                step.input,
                session=step.session_id,
                reasoning_level=step.reasoning_level,
                response_format=step.response_format,
                scope=step.scope,
            )

        if step.scope is not None:
            return await self._perform_scoped_query(step)

        if step.target == "chat":
            if not step.observer_peer_id:
                raise ValueError("observer_peer_id required for chat")
            if step.input is None:
                raise ValueError("input required for chat")

            peer = await self.client.aio.peer(id=step.observer_peer_id)

            response = await peer.aio.chat(
                step.input,
                session=step.session_id,
                target=step.observed_peer_id,
                reasoning_level=step.reasoning_level,
                response_format=step.response_format,
            )
            return response

        elif step.target == "get_context":
            if not step.session_id:
                raise ValueError("session_id required for get_context")
            session: Session = await self.client.aio.session(id=step.session_id)
            context: SessionContext = await session.aio.context(
                summary=step.summary, tokens=step.max_tokens
            )
            # Return the whole context object
            return context

        elif step.target == "get_peer_card":
            if not step.observer_peer_id:
                raise ValueError("peer_id required for get_peer_card")

            peer = await self.client.aio.peer(id=step.observer_peer_id)
            card = await peer.aio.get_card(
                step.observed_peer_id
                if step.observed_peer_id
                else step.observer_peer_id
            )
            return {"peer_card": card if card else None}

        elif step.target == "get_representation":
            if not step.observer_peer_id:
                raise ValueError("observer_peer_id required for get_representation")

            peer = await self.client.aio.peer(id=step.observer_peer_id)
            representation = await peer.aio.representation(
                step.session_id, target=step.observed_peer_id, search_query=step.input
            )
            return representation

        return None

    async def _perform_scoped_query(self, step: QueryAction) -> Any:
        """Run a `scope`-confined read over raw HTTP (no SDK parameter for it)."""
        if step.target == "chat":
            if not step.observer_peer_id:
                raise ValueError("observer_peer_id required for chat")
            if step.input is None:
                raise ValueError("input required for chat")
            body: dict[str, Any] = {"query": step.input, "scope": step.scope}
            if step.session_id:
                body["session_id"] = step.session_id
            if step.observed_peer_id:
                body["target"] = step.observed_peer_id
            if step.reasoning_level:
                body["reasoning_level"] = step.reasoning_level
            response = await self._request(
                "POST", f"/peers/{step.observer_peer_id}/chat", json=body
            )
            return response.json()["content"]

        if step.target == "get_representation":
            if not step.observer_peer_id:
                raise ValueError("observer_peer_id required for get_representation")
            body = {"scope": step.scope}
            if step.observed_peer_id:
                body["target"] = step.observed_peer_id
            if step.input:
                body["search_query"] = step.input
            response = await self._request(
                "POST", f"/peers/{step.observer_peer_id}/representation", json=body
            )
            return response.json()["representation"]

        if step.target == "get_context":
            if not step.session_id:
                raise ValueError("session_id required for get_context")
            if not step.observed_peer_id:
                raise ValueError("observed_peer_id required for a scoped get_context")
            # `scope` on session context takes a single scope name.
            if isinstance(step.scope, list):
                raise ValueError("get_context accepts a single scope, not a list")
            params: dict[str, Any] = {
                "scope": step.scope,
                "peer_target": step.observed_peer_id,
                "summary": str(step.summary).lower(),
            }
            if step.max_tokens is not None:
                params["tokens"] = step.max_tokens
            response = await self._request(
                "GET", f"/sessions/{step.session_id}/context", params=params
            )
            return response.json()

        raise ValueError(f"`scope` is not supported for target {step.target!r}")

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

            data = tool_use.input
            passed = bool(data.get("passed", False))
            if passed != assertion.pass_if:
                raise TestExecutionError(f"LLM Judge failed: {data.get('reasoning')}")

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
        redis_port: int = 9002,
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

    async def run(self) -> int:
        """Run the suite and return the number of tests that did not pass."""
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

            results: dict[str, TestOutcome] = {}

            logger.info(f"Found {len(test_files)} test(s)")

            # 3. Execute Tests
            client = Honcho(
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

                    executor.client = Honcho(
                        base_url=f"http://localhost:{self.harness.api_port}",
                        workspace_id=f"test_{test_name}_{int(time.time())}",
                    )

                    failure = await executor.execute(test_def, test_name)
                    test_duration = time.time() - test_start_time
                    results[test_file.name] = TestOutcome(
                        status="PASS" if failure is None else "FAIL",
                        duration=test_duration,
                        failure=failure,
                    )

                except ValidationError as e:
                    logger.error(f"Schema validation failed for {test_file}: {e}")
                    test_duration = time.time() - test_start_time
                    results[test_file.name] = TestOutcome(
                        status="INVALID SCHEMA", duration=test_duration
                    )
                except Exception as e:
                    logger.error(
                        f"Test {test_file.name} failed with error: {e}", exc_info=True
                    )
                    test_duration = time.time() - test_start_time
                    results[test_file.name] = TestOutcome(
                        status=f"ERROR: {str(e)}", duration=test_duration
                    )

            total_suite_time = time.time() - suite_start_time

            # 4. Report
            print("\n" + "=" * 60)
            print("TEST RESULTS")
            print("=" * 60)

            failed_count = 0
            total_count = len(results)

            # Calculate max name length for alignment
            max_name_length = max(len(name) for name in results) if results else 0

            for name, outcome in results.items():
                duration_str = f"({outcome.duration:.2f}s)"
                if outcome.status == "PASS":
                    print(
                        f"{name:<{max_name_length}} {GREEN}{outcome.status:<15}{RESET} {duration_str}"
                    )
                else:
                    print(
                        f"{name:<{max_name_length}} {RED}{outcome.status:<15}{RESET} {duration_str}"
                    )
                    if outcome.failure:
                        print(f"{'':<{max_name_length}} {outcome.failure.describe()}")
                    failed_count += 1

            print("=" * 60)
            print(f"\n{failed_count} failed / {total_count} total")
            print(f"Total execution time: {total_suite_time:.2f}s")
            print("=" * 60)

            # 5. Save results and send notifications
            # Always attempt S3 upload - save_results_to_s3 will check for credentials
            artifacts = await save_results_to_s3(
                results, failed_count, total_count, total_suite_time
            )

            # 6. Report the run: GitHub job summary, then Discord.
            passed_count = total_count - failed_count
            status_emoji = "✅" if failed_count == 0 else "⚠️"
            headline = (
                f"Results: {passed_count}/{total_count} passed, "
                f"{failed_count}/{total_count} failed"
            )

            write_job_summary(
                [
                    f"## {status_emoji} Unified Test Results",
                    "",
                    headline,
                    "",
                    f"Execution time: {total_suite_time:.2f}s",
                    *failure_lines(results),
                    "",
                    *artifact_lines(artifacts),
                ]
            )

            discord_webhook_url = os.getenv("TEST_DISCORD_WEBHOOK_URL")
            if discord_webhook_url:
                message_lines = [
                    f"{status_emoji} **Unified Test Results**",
                    headline,
                    f"Execution time: {total_suite_time:.2f}s",
                    *artifact_line("View Complete Results", artifacts.results),
                    *gha_run_lines(),
                    *(
                        [f"Reasoning traces: `{artifacts.traces.key}`"]
                        if artifacts.traces
                        else []
                    ),
                ]
                await send_discord_message(discord_webhook_url, message_lines)

            return failed_count

        finally:
            # 7. Cleanup
            logger.info("Cleaning up harness...")
            await self.harness.cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, default="tests/unified/test_cases")
    args = parser.parse_args()

    runner = UnifiedTestRunner(Path(args.test_dir))
    sys.exit(1 if asyncio.run(runner.run()) else 0)
