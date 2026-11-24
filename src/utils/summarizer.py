import asyncio
import logging
import time
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import cache
from inspect import cleandoc as c
from typing import Any, TypedDict

from nanoid import generate as generate_nanoid
from sqlalchemy import DateTime, case, cast, func, literal, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from src import schemas
from src.cache.client import cache as cache_client
from src.config import ConfiguredModelSettings, settings
from src.crud.lifecycle import live_session_row
from src.crud.message import observer_scope_clause
from src.crud.session import session_cache_key
from src.dependencies import tracked_db
from src.exceptions import EmptySummaryError, ResourceNotFoundException
from src.llm import HonchoLLMCallResponse, honcho_llm_call
from src.llm.types import (
    LLMTelemetryContext,
    degraded_response_parse_class,
    hit_output_cap,
)
from src.models import Message
from src.telemetry import prometheus_metrics
from src.telemetry.events import AgentToolSummaryCreatedEvent, emit
from src.telemetry.events.llm import CallPurpose
from src.telemetry.logging import accumulate_metric
from src.telemetry.prometheus.metrics import (
    SUMMARY_REJECTION_CONTENT_FILTERED,
    SUMMARY_REJECTION_DEGENERATE,
    DeriverComponents,
    DeriverTaskTypes,
    TokenTypes,
)
from src.utils.formatting import custom_instructions_section, utc_now_iso
from src.utils.retryable_errors import is_retryable_error
from src.utils.sanitization import strip_nul
from src.utils.tokens import estimate_tokens, tokenizer, track_deriver_input_tokens
from src.utils.types import PostCommit

from .. import crud, models

logger = logging.getLogger(__name__)


# TypedDict definitions for summary data
class Summary(TypedDict):
    """
    A summary object. Stored in session metadata and used in a session's get_context.

    Attributes:
        content: The summary text.
        message_id: The primary key ID of the message that this summary covers up to.
        summary_type: The type of summary (short or long).
        created_at: The timestamp of when the summary was created (ISO format string).
        token_count: The number of tokens in the summary text.
    """

    content: str
    message_id: int
    summary_type: str
    created_at: str
    token_count: int
    message_public_id: str


def to_schema_summary(s: Summary) -> schemas.Summary:
    return schemas.Summary(
        content=s["content"],
        message_id=s["message_id"],
        summary_type=s["summary_type"],
        created_at=s["created_at"],
        token_count=s["token_count"],
        message_public_id=s.get("message_public_id", ""),
    )


# Export the public functions
__all__ = [
    "get_summary",
    "get_both_summaries",
    "get_summarized_history",
    "get_session_context",
    "get_session_context_formatted",
    "SummaryType",
    "Summary",
    "to_schema_summary",
]


def _get_summary_model_config() -> ConfiguredModelSettings:
    return settings.SUMMARY.MODEL_CONFIG


# Configuration constants for summaries
MESSAGES_PER_SHORT_SUMMARY = settings.SUMMARY.MESSAGES_PER_SHORT_SUMMARY
MESSAGES_PER_LONG_SUMMARY = settings.SUMMARY.MESSAGES_PER_LONG_SUMMARY

SUMMARIES_KEY = "summaries"


# The prompt asks for this fraction of the per-call cap as a soft target. The rest
# is headroom: a cap equal to the target truncates summaries mid-sentence (#1204).
SUMMARY_TARGET_RATIO = 0.6


# The types of summary to store in the session metadata
class SummaryType(Enum):
    SHORT = "honcho_chat_summary_short"
    LONG = "honcho_chat_summary_long"


# Degeneration guard (#899). A model that loops until the output cap emits non-empty
# text, so the empty-check in _create_summary can't catch it. distinct-4 is chosen over
# zlib compression ratio because it is length-stable: measured 1.00 on clean prose from
# 3.7k-30k chars, while zlib drifts 2.17 -> 2.50 and would cross Whisper's 2.4 threshold
# on perfectly good long text. Against this PR's fixtures the #899 degenerate loop scores
# 0.003 while heavily templated but legitimate prose that also hits the cap scores 0.467,
# so 0.35 leaves the clean case a 1.33x margin and the degenerate case two orders below.
_DEGENERATE_NGRAM_N = 4
_DEGENERATE_DISTINCT_RATIO = 0.35


def _distinct_ngram_ratio(text: str, n: int = _DEGENERATE_NGRAM_N) -> float:
    """Fraction of n-grams in `text` that are unique. 1.0 = no repetition."""
    words = text.split()
    if len(words) < n:
        return 1.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def _basic_fallback_summary(
    message_count: int, last_message_content_preview: str
) -> str:
    """Deterministic one-liner used when the LLM summary is unusable."""
    if message_count <= 0:
        return ""
    return (
        f"Conversation with {message_count} messages about "
        f"{last_message_content_preview}..."
    )


def short_summary_prompt(
    formatted_messages: str,
    target_tokens: int,
    previous_summary_text: str,
    custom_instructions: str | None = None,
) -> str:
    """Generate the short summary prompt."""
    instructions_section = custom_instructions_section(custom_instructions)
    return c(f"""
You are a system that summarizes parts of a conversation to create a concise and accurate summary. Focus on capturing:

1. Key facts and information shared (**Capture as many explicit facts as possible**)
2. User preferences, opinions, and questions
3. Important context and requests
4. Core topics discussed

If there is a previous summary, ALWAYS make your new summary inclusive of both it and the new messages, therefore capturing the ENTIRE conversation. Prioritize key facts across the entire conversation.

Provide a concise, factual summary that captures the essence of the conversation. Your summary should be detailed enough to serve as context for future messages, but brief enough to be helpful. Prefer a thorough chronological narrative over a list of bullet points.

Return only the summary without any explanation or meta-commentary.

{instructions_section}

<previous_summary>
{previous_summary_text}
</previous_summary>

<conversation>
{formatted_messages}
</conversation>

Aim for about {target_tokens} tokens. If needed, drop lower-priority detail to stay near that length.
""")


def long_summary_prompt(
    formatted_messages: str,
    target_tokens: int,
    previous_summary_text: str,
    custom_instructions: str | None = None,
) -> str:
    """Generate the long summary prompt."""
    instructions_section = custom_instructions_section(custom_instructions)
    return c(f"""
You are a system that creates thorough, comprehensive summaries of conversations. Focus on capturing:

1. Key facts and information shared (**Capture as many explicit facts as possible**)
2. User preferences, opinions, and questions
3. Important context and requests
4. Core topics discussed in detail
5. User's apparent emotional state and personality traits
6. Important themes and patterns across the conversation

If there is a previous summary, ALWAYS make your new summary inclusive of both it and the new messages, therefore capturing the ENTIRE conversation. Prioritize key facts across the entire conversation.

Provide a thorough and detailed summary that captures the essence of the conversation. Your summary should serve as a comprehensive record of the important information in this conversation. Prefer an exhaustive chronological narrative over a list of bullet points.

Return only the summary without any explanation or meta-commentary.

{instructions_section}

<previous_summary>
{previous_summary_text}
</previous_summary>

<conversation>
{formatted_messages}
</conversation>

Aim for about {target_tokens} tokens. If needed, drop lower-priority detail to stay near that length.
""")


@cache
def estimate_short_summary_prompt_tokens() -> int:
    """Estimate tokens for the short summary prompt (without messages/previous_summary)."""
    try:
        return estimate_tokens(
            short_summary_prompt(
                formatted_messages="",
                target_tokens=0,
                previous_summary_text="",
            )
        )
    except Exception:
        # Return a rough estimate if estimation fails
        return 200


@cache
def estimate_long_summary_prompt_tokens() -> int:
    """Estimate tokens for the long summary prompt (without messages/previous_summary)."""
    try:
        return estimate_tokens(
            long_summary_prompt(
                formatted_messages="",
                target_tokens=0,
                previous_summary_text="",
            )
        )
    except Exception:
        # Return a rough estimate if estimation fails
        return 200


def _refuse_degraded_empty_summary(
    prompt: str,
) -> Callable[[HonchoLLMCallResponse[Any]], None]:
    """Build a ``validate_response`` hook that refuses a truncated or filtered
    empty summary inside the retried call. The fallback text for an empty
    summary is never saved, so accepting it would consume the boundary."""

    def validate(response: HonchoLLMCallResponse[Any]) -> None:
        # NUL-only output is empty once `_create_summary` strips it.
        if strip_nul(str(response.content)).strip():
            return
        parse_class = degraded_response_parse_class(response)
        if parse_class is not None:
            raise EmptySummaryError.for_prompt(
                prompt,
                parse_class=parse_class,
                finish_reasons=response.finish_reasons,
                output_tokens=response.output_tokens,
            )

    return validate


async def create_short_summary(
    formatted_messages: str,
    input_tokens: int,
    previous_summary: str | None = None,
    custom_instructions: str | None = None,
    *,
    telemetry: LLMTelemetryContext | None = None,
) -> HonchoLLMCallResponse[str]:
    # input_tokens indicates how many tokens the message list + previous summary take up.
    # Short summaries should be smaller than the content being summarized, so the target
    # scales with the input up to the cap.
    target_tokens = int(
        min(input_tokens, settings.SUMMARY.MAX_TOKENS_SHORT) * SUMMARY_TARGET_RATIO
    )

    if previous_summary:
        previous_summary_text = previous_summary
    else:
        previous_summary_text = "There is no previous summary -- the messages are the beginning of the conversation."

    prompt = short_summary_prompt(
        formatted_messages, target_tokens, previous_summary_text, custom_instructions
    )

    # The caller's trace id is the summary's run id; mint one if it has none.
    trace_id = (telemetry.trace_id if telemetry else None) or generate_nanoid()
    return await honcho_llm_call(
        model_config=_get_summary_model_config(),
        prompt=prompt,
        max_tokens=settings.SUMMARY.MAX_TOKENS_SHORT,
        validate_response=_refuse_degraded_empty_summary(prompt),
        telemetry=replace(
            telemetry or LLMTelemetryContext(),
            call_purpose=CallPurpose.SUMMARY_SHORT.value,
            parent_category="summary",
            agent_type="summarizer",
            trace_id=trace_id,
            span_id=trace_id,
            track_name="Short Summary",
        ),
    )


async def create_long_summary(
    formatted_messages: str,
    previous_summary: str | None = None,
    custom_instructions: str | None = None,
    *,
    telemetry: LLMTelemetryContext | None = None,
) -> HonchoLLMCallResponse[str]:
    target_tokens = int(settings.SUMMARY.MAX_TOKENS_LONG * SUMMARY_TARGET_RATIO)

    if previous_summary:
        previous_summary_text = previous_summary
    else:
        previous_summary_text = "There is no previous summary -- the messages are the beginning of the conversation."

    prompt = long_summary_prompt(
        formatted_messages, target_tokens, previous_summary_text, custom_instructions
    )

    # The caller's trace id is the summary's run id; mint one if it has none.
    trace_id = (telemetry.trace_id if telemetry else None) or generate_nanoid()
    return await honcho_llm_call(
        model_config=_get_summary_model_config(),
        prompt=prompt,
        max_tokens=settings.SUMMARY.MAX_TOKENS_LONG,
        validate_response=_refuse_degraded_empty_summary(prompt),
        telemetry=replace(
            telemetry or LLMTelemetryContext(),
            call_purpose=CallPurpose.SUMMARY_LONG.value,
            parent_category="summary",
            agent_type="summarizer",
            trace_id=trace_id,
            span_id=trace_id,
            track_name="Long Summary",
        ),
    )


async def summarize_if_needed(
    workspace_name: str,
    session_name: str,
    message_id: int,
    message_seq_in_session: int,
    message_public_id: str,
    configuration: schemas.ResolvedConfiguration,
    *,
    queue_item_id: int | None = None,
) -> None:
    """
    Create short/long summaries if thresholds met.

    This function checks for both short and long summary needs independently,
    without assuming any relationship between their thresholds.

    Args:
        workspace_name: The workspace name
        session_name: The session name
        message_id: The message ID
        message_seq_in_session: The sequence number of the message in the session
        message_public_id: The public ID of the message
        configuration: The resolved configuration for the message
        queue_item_id: Queue row that triggered this summary, when available.
    """
    if configuration.summary.enabled is False:
        return

    should_create_long: bool = (
        message_seq_in_session % configuration.summary.messages_per_long_summary == 0
    )
    should_create_short: bool = (
        message_seq_in_session % configuration.summary.messages_per_short_summary == 0
    )

    if should_create_long is False and should_create_short is False:
        return

    # Auto workspaces queue history with only the last summaries of each batch,
    # so a summary there may have to catch up (see _create_and_save_summary).
    catch_up = await _in_auto_workspace(workspace_name)

    # If both summaries need to be created, run them in parallel (one after the
    # other when catching up, so the short summary can start from the long one)
    if should_create_long and should_create_short:

        async def create_long_summary_task():
            await _create_and_save_summary(
                workspace_name,
                session_name,
                message_id=message_id,
                message_seq_in_session=message_seq_in_session,
                message_public_id=message_public_id,
                summary_type=SummaryType.LONG,
                configuration=configuration,
                queue_item_id=queue_item_id,
                catch_up=catch_up,
            )
            accumulate_metric(
                f"summary_{workspace_name}_{message_id}",
                "long_summary_up_to_message",
                message_seq_in_session,
                "count",
            )

        async def create_short_summary_task():
            await _create_and_save_summary(
                workspace_name,
                session_name,
                message_id=message_id,
                message_seq_in_session=message_seq_in_session,
                message_public_id=message_public_id,
                summary_type=SummaryType.SHORT,
                configuration=configuration,
                queue_item_id=queue_item_id,
                catch_up=catch_up,
            )
            accumulate_metric(
                f"summary_{workspace_name}_{message_id}",
                "short_summary_up_to_message",
                message_seq_in_session,
                "count",
            )

        # Every long boundary is also a short one under the default cadence, so
        # this is how every long summary runs. Both finish before any failure is
        # raised, so one failing never cancels the other.
        if catch_up:
            long_result = await _outcome(create_long_summary_task())
            short_result = await _outcome(create_short_summary_task())
        else:
            long_result, short_result = await asyncio.gather(
                create_long_summary_task(),
                create_short_summary_task(),
                return_exceptions=True,
            )
        _raise_summary_failures(
            {SummaryType.LONG: long_result, SummaryType.SHORT: short_result},
            workspace_name=workspace_name,
            session_name=session_name,
            message_id=message_id,
        )
    else:
        # If only one summary needs to be created, run individually
        if should_create_long:
            await _create_and_save_summary(
                workspace_name,
                session_name,
                message_id=message_id,
                message_seq_in_session=message_seq_in_session,
                message_public_id=message_public_id,
                summary_type=SummaryType.LONG,
                configuration=configuration,
                queue_item_id=queue_item_id,
                catch_up=catch_up,
            )
            accumulate_metric(
                f"summary_{workspace_name}_{message_id}",
                "long_summary_up_to_message",
                message_seq_in_session,
                "count",
            )
        elif should_create_short:
            await _create_and_save_summary(
                workspace_name,
                session_name,
                message_id=message_id,
                message_seq_in_session=message_seq_in_session,
                message_public_id=message_public_id,
                summary_type=SummaryType.SHORT,
                configuration=configuration,
                queue_item_id=queue_item_id,
                catch_up=catch_up,
            )
            accumulate_metric(
                f"summary_{workspace_name}_{message_id}",
                "short_summary_up_to_message",
                message_seq_in_session,
                "count",
            )


async def _outcome(task: Coroutine[Any, Any, None]) -> BaseException | None:
    """Await a summary task, returning its exception instead of raising it."""
    try:
        await task
    except Exception as error:
        return error
    return None


async def _in_auto_workspace(workspace_name: str) -> bool:
    """Whether the workspace was created as an auto workspace."""
    async with tracked_db("summary.workspace_auto", read_only=True) as db:
        enabled_at = await db.scalar(
            select(models.Workspace.auto_enabled_at).where(
                models.Workspace.tenant_id == models.scoped_tenant_id(),
                models.Workspace.name == workspace_name,
            )
        )
    return enabled_at is not None


def _raise_summary_failures(
    results: dict[SummaryType, BaseException | None],
    *,
    workspace_name: str,
    session_name: str,
    message_id: int,
) -> None:
    """Log every failed summary from a combined boundary, then raise one.

    A failure here used to be dropped unless it was an ``EmptySummaryError``, so
    the queue item completed and the summary for that window was lost for good
    (DEV-3159). Now any failure reaches the queue. A cancellation is re-raised
    first. Otherwise a transient error wins over a terminal one, because only
    transient errors are retried; the retry reruns both summaries, and one that
    was already saved skips itself. Between terminal errors, an
    ``EmptySummaryError`` wins, as before.

    The raised failure is logged at warning, since the queue reports it. Any
    other is logged at error, because it is reported nowhere else.

    Args:
        results: Each summary type's ``gather`` result, None when it succeeded
        workspace_name: Workspace name, for the log line
        session_name: Session name, for the log line
        message_id: The boundary message's ID, for the log line
    """
    failures = {
        summary_type: result
        for summary_type, result in results.items()
        if isinstance(result, BaseException)
    }
    if not failures:
        return
    for failure in failures.values():
        if not isinstance(failure, Exception):
            raise failure
    # Transient first, then a refusal (raised here before DEV-3159), then the rest.
    raised = min(
        failures.values(),
        key=lambda failure: (
            0
            if is_retryable_error(failure)
            else 1
            if isinstance(failure, EmptySummaryError)
            else 2
        ),
    )
    for summary_type, failure in failures.items():
        logger.log(
            logging.WARNING if failure is raised else logging.ERROR,
            "%s summary failed for message %s in session %s (workspace %s)",
            summary_type.name,
            message_id,
            session_name,
            workspace_name,
            exc_info=failure,
        )
    raise raised


class SummaryCatchUpIncomplete(Exception):
    """A catch-up summary stopped partway: a chunk's summary was a fallback."""


@dataclass(frozen=True)
class _Window:
    """The messages one summary call reads, taken while their session is open."""

    formatted: str
    first_seq: int
    last_id: int
    last_seq: int
    last_public_id: str
    last_preview: str
    count: int
    source_ids: list[str]
    # Message tokens as sent: a clipped message is shorter than its stored count.
    tokens: int


def _window(messages: list[Message], formatted: str, clipped: bool) -> _Window:
    return _Window(
        formatted=formatted,
        first_seq=messages[0].seq_in_session,
        last_id=messages[-1].id,
        last_seq=messages[-1].seq_in_session,
        last_public_id=messages[-1].public_id,
        last_preview=messages[-1].content[:30],
        count=len(messages),
        source_ids=[message.public_id for message in messages],
        tokens=_count_prompt_tokens(formatted)
        if clipped
        else sum(message.token_count for message in messages),
    )


# A catch-up summary reads the history it missed in chunks of at most this many
# message tokens, oldest first: one call per chunk instead of one per interval.
_CATCH_UP_CHUNK_TOKENS = 20_000
# Messages read per catch-up chunk: more than a chunk of tokens ever holds.
_CATCH_UP_FETCH = 2_000


async def _create_and_save_summary(
    workspace_name: str,
    session_name: str,
    *,
    message_id: int,
    message_seq_in_session: int,
    message_public_id: str,
    summary_type: SummaryType,
    configuration: schemas.ResolvedConfiguration,
    queue_item_id: int | None = None,
    catch_up: bool = False,
) -> None:
    """
    Create a new summary and save it to the database.
    1. Get the latest summary
    2. Get the messages since the latest summary
    3. Generate a new summary using the messages and the previous summary
    4. Save the new summary to the database

    The summary reads the interval's messages ending at this one. With
    `catch_up` (auto workspaces, whose history is queued with only the last
    summaries of each batch), it reads every message after the latest summary
    instead, however many, in chunks, oldest first, folding each into the
    summary and saving it before the next, so a retry resumes where a failure
    stopped. A short summary behind the long one starts from the long one.
    """

    logger.debug("Creating new %s summary", summary_type.name)
    summary_start = time.perf_counter()

    # Prompt scaffold tokens, without messages or previous summary. Cheap and
    # cached; used for the input budget below and for the telemetry after the
    # call. Custom instructions are counted exactly, since they are caller text.
    if summary_type == SummaryType.SHORT:
        prompt_tokens = estimate_short_summary_prompt_tokens()
    else:
        prompt_tokens = estimate_long_summary_prompt_tokens()
    if configuration.summary.custom_instructions:
        prompt_tokens += _count_prompt_tokens(
            custom_instructions_section(configuration.summary.custom_instructions)
        )
    messages_per_summary = (
        configuration.summary.messages_per_long_summary
        if summary_type == SummaryType.LONG
        else configuration.summary.messages_per_short_summary
    )

    # The single window, when not catching up.
    window: _Window | None = None
    async with tracked_db("summary.fetch_data") as db:
        try:
            session = await crud.get_session(db, session_name, workspace_name)
        except ResourceNotFoundException:
            return
        session_id = session.id
        summaries: dict[str, Summary] = session.internal_metadata.get(SUMMARIES_KEY, {})
        latest_summary = summaries.get(summary_type.value)
        if latest_summary:
            latest_summary_message_id = latest_summary["message_id"]
            # Skip if latest summary already covers message.
            if latest_summary_message_id >= message_id:
                return

        previous_summary_text = latest_summary["content"] if latest_summary else None
        previous_summary_tokens = latest_summary["token_count"] if latest_summary else 0

        catch_up_from: int | None = None
        if catch_up:
            covered = await _covered_seq(
                db, workspace_name, session_name, latest_summary
            )
            # Everything after the latest summary, however far behind it is,
            # so a retry after a crash mid-catch-up resumes where it stopped.
            catch_up_from = covered + 1
            long_summary = summaries.get(SummaryType.LONG.value)
            if summary_type == SummaryType.SHORT and long_summary is not None:
                long_covered = await _covered_seq(
                    db, workspace_name, session_name, long_summary
                )
                if long_covered > covered:
                    # The long summary already holds the history up to its
                    # message: start from it, after it.
                    previous_summary_text = long_summary["content"]
                    previous_summary_tokens = long_summary["token_count"]
                    catch_up_from = long_covered + 1
                    if long_covered >= message_seq_in_session:
                        # Nothing after it: condense it with the interval's
                        # messages, as an ordinary short summary would.
                        catch_up_from = None

        if catch_up_from is None:
            # Calculate the sequence range for messages to summarize
            # We want to get the last N messages where N is the configured summary interval
            start_seq = max(message_seq_in_session - messages_per_summary + 1, 1)

            messages = await crud.get_messages_by_seq_range(
                db,
                workspace_name,
                session_name,
                start_seq=start_seq,
                end_seq=message_seq_in_session,
            )
            if not messages:
                logger.warning("No messages to summarize for message %s", message_id)
                return

            # The window is chosen by message count, so it can outgrow the model's
            # context window. Fit it to the input budget, dropping the oldest
            # messages first.
            window_size = len(messages)
            message_budget = (
                settings.SUMMARY.MAX_INPUT_TOKENS
                - prompt_tokens
                - _count_prompt_tokens(previous_summary_text or "")
                - _PROMPT_SLACK_TOKENS
            )
            messages, formatted_messages, clipped = _fit_messages_to_budget(
                messages, message_budget
            )
            if not messages:
                logger.warning(
                    "No room for messages in the %s summary for message %s: the "
                    + "prompt and previous summary use the whole input budget of %s "
                    + "tokens",
                    summary_type.name,
                    message_id,
                    settings.SUMMARY.MAX_INPUT_TOKENS,
                )
                return
            if len(messages) < window_size:
                logger.warning(
                    "Dropped the oldest %s of %s messages from the %s summary for "
                    + "message %s to stay under the input budget of %s tokens",
                    window_size - len(messages),
                    window_size,
                    summary_type.name,
                    message_id,
                    settings.SUMMARY.MAX_INPUT_TOKENS,
                )
            window = _window(messages, formatted_messages, clipped)

    if window is not None:
        saved = await _summarize_window(
            workspace_name,
            session_name,
            session_id=session_id,
            summary_type=summary_type,
            configuration=configuration,
            prompt_tokens=prompt_tokens,
            window=window,
            previous_summary_text=previous_summary_text,
            previous_summary_tokens=previous_summary_tokens,
            message_public_id=message_public_id,
            message_seq_in_session=message_seq_in_session,
            metric_key=f"summary_{workspace_name}_{message_id}",
            summary_start=summary_start,
            queue_item_id=queue_item_id,
            raise_errors=catch_up,
        )
        if catch_up and saved is None:
            raise SummaryCatchUpIncomplete(
                f"{summary_type.name} summary of session {session_name} wasn't "
                + f"saved at message {message_seq_in_session}"
            )
        return
    if catch_up_from is None:
        return

    # Chunks whose summary came back bad (a refusal, an empty answer). They
    # are skipped, as a regular summary's failed window is lost, so one bad
    # chunk can't hold back the rest of the session; the task then reports them.
    skipped: list[tuple[int, int]] = []

    start = catch_up_from
    while start <= message_seq_in_session:
        budget = (
            settings.SUMMARY.MAX_INPUT_TOKENS
            - prompt_tokens
            - _count_prompt_tokens(previous_summary_text or "")
            - _PROMPT_SLACK_TOKENS
        )
        async with tracked_db("summary.catch_up") as db:
            # The next live messages, however many sequence numbers deleted
            # messages left empty.
            fetched = await crud.get_messages_by_seq_range(
                db,
                workspace_name,
                session_name,
                start_seq=start,
                end_seq=message_seq_in_session,
                limit=_CATCH_UP_FETCH,
            )
            chunk, formatted_chunk, chunk_clipped = _fit_oldest_messages_to_budget(
                fetched, min(_CATCH_UP_CHUNK_TOKENS, budget), budget
            )
            chunk_window = (
                _window(chunk, formatted_chunk, chunk_clipped) if chunk else None
            )
        if chunk_window is None:
            if not fetched:
                break
            raise SummaryCatchUpIncomplete(
                f"No room for messages in the {summary_type.name} summary of "
                + f"session {session_name}: the prompt and previous summary use "
                + "the whole input budget"
            )
        try:
            saved = await _summarize_window(
                workspace_name,
                session_name,
                session_id=session_id,
                summary_type=summary_type,
                configuration=configuration,
                prompt_tokens=prompt_tokens,
                window=chunk_window,
                previous_summary_text=previous_summary_text,
                previous_summary_tokens=previous_summary_tokens,
                message_public_id=chunk_window.last_public_id,
                message_seq_in_session=chunk_window.last_seq,
                metric_key=f"summary_{workspace_name}_{message_id}",
                # Each chunk is timed on its own.
                summary_start=time.perf_counter(),
                queue_item_id=queue_item_id,
                # A failed call reaches the queue and leaves the checkpoint at
                # the last saved chunk, where a retry or the next summary
                # resumes; only a bad answer (a refusal, an empty one) skips.
                raise_errors=True,
            )
        except EmptySummaryError:
            saved = None
        if saved is None:
            skipped.append((chunk_window.first_seq, chunk_window.last_seq))
        else:
            previous_summary_text = saved["content"]
            previous_summary_tokens = saved["token_count"]
        start = chunk_window.last_seq + 1
    if skipped:
        raise SummaryCatchUpIncomplete(
            f"{summary_type.name} summary of session {session_name} skipped "
            + ", ".join(f"messages {a}-{b}" for a, b in skipped)
        )


async def _covered_seq(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    summary: "Summary | None",
) -> int:
    """The sequence number of the last message a summary covers; 0 for none.

    Read as the newest message at or before the summary's message id, so a
    deleted message still places the summary.
    """
    if summary is None:
        return 0
    seq = await db.scalar(
        select(func.max(models.Message.seq_in_session)).where(
            models.Message.tenant_id == models.scoped_tenant_id(),
            models.Message.workspace_name == workspace_name,
            models.Message.session_name == session_name,
            models.Message.id <= summary["message_id"],
        )
    )
    return int(seq or 0)


async def _summarize_window(
    workspace_name: str,
    session_name: str,
    *,
    session_id: str,
    summary_type: SummaryType,
    configuration: schemas.ResolvedConfiguration,
    prompt_tokens: int,
    window: _Window,
    previous_summary_text: str | None,
    previous_summary_tokens: int,
    message_public_id: str,
    message_seq_in_session: int,
    metric_key: str,
    summary_start: float,
    queue_item_id: int | None,
    raise_errors: bool = False,
) -> "Summary | None":
    """Summarize one window of messages onto the previous summary, and save it.

    Returns the saved summary, or None when the result was a fallback, which
    is never saved. With `raise_errors`, a failed call is raised instead of
    becoming a fallback.
    """
    messages_tokens = window.tokens
    input_tokens = messages_tokens + previous_summary_tokens

    # One run id per summary: its LLM calls and its summary-created event share it.
    run_id = generate_nanoid()
    (
        new_summary,
        is_fallback,
        llm_input_tokens,
        llm_output_tokens,
    ) = await _create_summary(
        formatted_messages=window.formatted,
        previous_summary_text=previous_summary_text,
        summary_type=summary_type,
        input_tokens=input_tokens,
        message_public_id=message_public_id,
        last_message_id=window.last_id,
        last_message_content_preview=window.last_preview,
        message_count=window.count,
        custom_instructions=configuration.summary.custom_instructions,
        raise_errors=raise_errors,
        telemetry=LLMTelemetryContext(
            workspace_name=workspace_name,
            session_id=session_id,
            trace_id=run_id,
            source_message_ids=window.source_ids,
            queue_item_ids=[queue_item_id] if queue_item_id is not None else [],
            custom_instructions=configuration.summary.custom_instructions,
        ),
    )

    # Step 3: Save to database with new transaction
    if not is_fallback:
        track_deriver_input_tokens(
            task_type=DeriverTaskTypes.SUMMARY,
            components={
                DeriverComponents.PROMPT: prompt_tokens,
                DeriverComponents.MESSAGES: messages_tokens,
                DeriverComponents.PREVIOUS_SUMMARY: previous_summary_tokens,
            },
        )

        # Track output tokens
        if settings.METRICS.ENABLED:
            prometheus_metrics.record_deriver_tokens(
                count=new_summary["token_count"],
                task_type=DeriverTaskTypes.SUMMARY.value,
                token_type=TokenTypes.OUTPUT.value,
                component=DeriverComponents.OUTPUT_TOTAL.value,
            )

        # Save summary to database with new transaction
        async with tracked_db("summary.save") as db:
            summary_write = await _save_summary(
                db,
                new_summary,
                workspace_name,
                session_name,
            )
            await db.commit()
            await summary_write.post_commit()

    accumulate_metric(
        metric_key,
        f"{summary_type.name}_summary_text",
        new_summary["content"],
        "blob",
    )
    accumulate_metric(
        metric_key,
        f"{summary_type.name}_summary_size",
        new_summary["token_count"],
        "tokens",
    )

    summary_duration = (time.perf_counter() - summary_start) * 1000
    accumulate_metric(
        metric_key,
        f"{summary_type.name}_summary_creation",
        summary_duration,
        "ms",
    )

    # Emit telemetry event (only for non-fallback summaries).
    if not is_fallback:
        emit(
            AgentToolSummaryCreatedEvent(
                run_id=run_id,
                parent_category="deriver",
                agent_type="summarizer",
                workspace_name=workspace_name,
                session_name=session_name,
                message_id=message_public_id,
                message_count=window.count,
                message_seq_in_session=message_seq_in_session,
                summary_type="short" if summary_type == SummaryType.SHORT else "long",
                input_tokens=llm_input_tokens,
                output_tokens=llm_output_tokens,
                # additive token-breakdown fields
                previous_summary_tokens=previous_summary_tokens,
                message_tokens=messages_tokens,
                prompt_scaffold_tokens=prompt_tokens,
                custom_instructions_tokens=estimate_tokens(
                    configuration.summary.custom_instructions
                ),
                custom_instructions_source=configuration.summary.custom_instructions_source,
            )
        )
    return None if is_fallback else new_summary


async def _create_summary(
    formatted_messages: str,
    previous_summary_text: str | None,
    summary_type: SummaryType,
    input_tokens: int,
    message_public_id: str,
    last_message_id: int,
    last_message_content_preview: str,
    message_count: int,
    custom_instructions: str | None = None,
    *,
    raise_errors: bool = False,
    telemetry: LLMTelemetryContext | None = None,
) -> tuple[Summary, bool, int, int]:
    """
    Generate a summary of the provided messages using an LLM.

    Args:
        formatted_messages: Pre-formatted message string
        previous_summary_text: Optional previous summary to provide context
        summary_type: Type of summary to create ("short" or "long")
        input_tokens: Token count for input
        message_public_id: Public ID of the last message
        last_message_id: ID of the last message
        last_message_content_preview: Preview of last message content for fallback
        message_count: Number of messages for fallback
        custom_instructions: Resolved summary custom instructions, if any
        raise_errors: Raise a failed call (a timeout, a provider error)
            instead of falling back; a bad answer still falls back.
        telemetry: Source identity carried through summary generation.

    Returns:
        A tuple of (Summary, is_fallback, llm_input_tokens, llm_output_tokens)
        where is_fallback indicates if the summary was generated using a
        fallback instead of an LLM call, and the token counts are from the LLM call
        (0 if fallback was used)
    """

    response: HonchoLLMCallResponse[str] | None = None
    is_fallback = False
    llm_input_tokens = 0
    llm_output_tokens = 0
    try:
        if summary_type == SummaryType.SHORT:
            response = await create_short_summary(
                formatted_messages,
                input_tokens,
                previous_summary_text,
                custom_instructions,
                telemetry=telemetry,
            )
        else:
            response = await create_long_summary(
                formatted_messages,
                previous_summary_text,
                custom_instructions,
                telemetry=telemetry,
            )

        # Strip NUL before the checks below so they judge the text that gets
        # stored: jsonb rejects NUL, and NUL-only output must count as empty.
        summary_text = strip_nul(response.content)
        summary_tokens = response.output_tokens
        llm_input_tokens = response.input_tokens
        llm_output_tokens = response.output_tokens

        # Detect potential issues with the summary
        if not summary_text.strip():
            logger.error(
                "Generated summary is empty (finish_reasons=%s). Falling back to basic summary.",
                response.finish_reasons,
            )
            is_fallback = True
            summary_text = _basic_fallback_summary(
                message_count, last_message_content_preview
            )
            summary_tokens = estimate_tokens(summary_text) if summary_text else 0
            llm_input_tokens = 0
            llm_output_tokens = 0
        elif hit_output_cap(response.finish_reasons):
            repetition_ratio = _distinct_ngram_ratio(summary_text)
            if repetition_ratio < _DEGENERATE_DISTINCT_RATIO:
                logger.error(
                    (
                        "Generated %s summary is degenerate: hit the output cap "
                        "(finish_reasons=%s) with a distinct-%d ratio of %.3f < %.2f. "
                        "Discarding; the previous summary is retained."
                    ),
                    summary_type.name,
                    response.finish_reasons,
                    _DEGENERATE_NGRAM_N,
                    repetition_ratio,
                    _DEGENERATE_DISTINCT_RATIO,
                )
                is_fallback = True
                summary_text = _basic_fallback_summary(
                    message_count, last_message_content_preview
                )
                summary_tokens = estimate_tokens(summary_text) if summary_text else 0
                llm_input_tokens = 0  # match the documented fallback contract
                llm_output_tokens = 0
                if settings.METRICS.ENABLED:
                    prometheus_metrics.record_summary_rejection(
                        summary_type=summary_type.name.lower(),
                        reason=SUMMARY_REJECTION_DEGENERATE,
                    )
    except EmptySummaryError as e:
        # Not a generation failure to paper over: the fallback below is never
        # saved, so it must reach the queue as an error.
        if e.parse_class == "content_filtered" and settings.METRICS.ENABLED:
            prometheus_metrics.record_summary_rejection(
                summary_type=summary_type.name.lower(),
                reason=SUMMARY_REJECTION_CONTENT_FILTERED,
            )
        raise
    except Exception:
        if raise_errors:
            raise
        logger.exception("Error generating summary!")
        # Fallback to a basic summary in case of error
        summary_text = _basic_fallback_summary(
            message_count, last_message_content_preview
        )
        summary_tokens = 0
        is_fallback = True

    return (
        Summary(
            content=summary_text,
            message_id=last_message_id,
            summary_type=summary_type.value,
            created_at=utc_now_iso(),
            token_count=summary_tokens,
            message_public_id=message_public_id,
        ),
        is_fallback,
        llm_input_tokens,
        llm_output_tokens,
    )


async def _save_summary(
    db: AsyncSession,
    summary: Summary,
    workspace_name: str,
    session_name: str,
) -> PostCommit:
    """
    Save a summary as metadata on a session, in the caller's transaction.

    NUL is stripped from the content because jsonb rejects it. Deciding whether
    the content is worth saving (empty, fallback, degenerate) is the caller's job.
    Does not commit: the caller commits, then runs ``post_commit()`` on the
    result to invalidate the session cache.

    Args:
        db: Database session
        summary: The summary to save
        workspace_name: Workspace name
        session_name: Session name

    Returns:
        The cache work to run after the caller commits. Empty when the session
        no longer exists and nothing was written.
    """
    from src.exceptions import ResourceNotFoundException

    # Get the label value from the enum
    label_value = summary["summary_type"]

    try:
        await crud.get_session(db, session_name, workspace_name)
    except ResourceNotFoundException:
        # If session doesn't exist, we can't save the summary
        logger.warning(
            f"Cannot save summary: session {session_name} not found in workspace {workspace_name}"
        )
        return PostCommit()

    sanitized: Summary = {**summary, "content": strip_nul(summary["content"])}

    # Merge this one entry into the summaries map on the server. Writing back a
    # map read earlier (possibly from cache) would drop or revert a summary saved
    # in between, such as the sibling at a combined short/long boundary (DEV-3160).
    stored = models.Session.internal_metadata[SUMMARIES_KEY]
    summaries = case(
        (func.jsonb_typeof(stored) == "object", stored),
        else_=literal({}, JSONB),
    ).op("||", return_type=JSONB)(
        func.jsonb_build_object(label_value, literal(sanitized, JSONB))
    )
    stmt = (
        update(models.Session)
        .where(models.Session.workspace_name == workspace_name)
        .where(models.Session.name == session_name)
        .values(
            internal_metadata=models.Session.internal_metadata.op(
                "||", return_type=JSONB
            )(func.jsonb_build_object(SUMMARIES_KEY, summaries))
        )
    )

    await db.execute(stmt)

    async def on_commit() -> None:
        await cache_client.delete(session_cache_key(workspace_name, session_name))

    return PostCommit(on_commit)


async def get_summarized_history(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    cutoff: int | None = None,
    summary_type: SummaryType = SummaryType.SHORT,
) -> str:
    """
    Get a summarized version of the chat history by combining the latest summary
    with all messages since that summary.

    Note: history is exclusive of the cutoff message.

    Args:
        db: Database session
        workspace_name: The workspace name
        session_name: The session name
        cutoff: (Optional) message ID to cutoff at
        summary_type: Type of summary to get ("short" or "long")

    Returns:
        A string formatted history text with summary and recent messages
    """
    # Get messages since the latest summary and the summary itself
    summary = await get_summary(db, workspace_name, session_name, summary_type)

    # Check if we have a valid summary with a message_id
    if summary:
        messages = await crud.get_messages_id_range(
            db,
            workspace_name,
            session_name,
            start_id=summary["message_id"],
            end_id=cutoff,
        )
    else:
        messages = await crud.get_messages_id_range(
            db, workspace_name, session_name, end_id=cutoff
        )

    # Format messages
    messages_text = _format_messages(messages)

    if summary:
        # Combine summary with recent messages
        return f"""
<summary>
{summary["content"]}
</summary>
<recent_messages>
{messages_text}
</recent_messages>
"""

    # No summary available, return just the messages
    return messages_text


async def get_summary(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    summary_type: SummaryType = SummaryType.SHORT,
) -> Summary | None:
    """
    Get summary for a given session.

    Args:
        db: Database session
        workspace_name: The workspace name
        session_name: The session name
        summary_type: Type of summary to retrieve ("short" or "long")

    Returns:
        The summary data dictionary, or None if no summary exists
    """
    try:
        session = await crud.get_session(db, session_name, workspace_name)
    except ResourceNotFoundException:
        # If session doesn't exist, there's no summary to retrieve
        return None

    summaries: dict[str, Summary] = session.internal_metadata.get(SUMMARIES_KEY, {})
    if not summaries or summary_type.value not in summaries:
        return None
    return summaries[summary_type.value]


async def get_recent_peer_summaries(
    db: AsyncSession,
    workspace_name: str,
    peer_names: Sequence[str],
    *,
    limit: int,
    offset: int = 0,
) -> list[tuple[str, Summary]]:
    """
    Get the latest summary of the sessions a set of peers most recently shared.

    A session qualifies when every named peer has a membership row in it, current
    or ended, except that a scope peer needs a current one: a session removed from
    a scope stops granting it access, as it does for message search. Each session
    contributes whichever of its summaries reaches the later message: both are
    cumulative, and the short one is rewritten more often, so it is usually the
    more current. On a tie the long one wins, since it is the more detailed. Sessions are ordered by their
    newest summary, most recent first, as a proxy for recent activity that needs
    no message scan.

    Args:
        db: Database session
        workspace_name: The workspace name
        peer_names: Peers that must all be members of a returned session
        limit: Maximum number of sessions to return
        offset: Number of most recent sessions to skip

    Returns:
        (session name, summary) pairs, most recently summarized first
    """
    tenant_id = models.scoped_tenant_id()
    summaries = models.Session.internal_metadata[SUMMARIES_KEY]
    long_summarized_at = cast(
        summaries[SummaryType.LONG.value]["created_at"].astext,
        DateTime(timezone=True),
    )
    short_summarized_at = cast(
        summaries[SummaryType.SHORT.value]["created_at"].astext,
        DateTime(timezone=True),
    )
    last_summarized_at = func.greatest(long_summarized_at, short_summarized_at)

    stmt = (
        select(models.Session.name, summaries)
        .where(models.Session.tenant_id == tenant_id)
        .where(models.Session.workspace_name == workspace_name)
        .where(live_session_row())
        .where(last_summarized_at.is_not(None))
    )
    for peer_name in dict.fromkeys(peer_names):
        stmt = stmt.where(
            observer_scope_clause(workspace_name, peer_name, models.Session.name)
        )
    stmt = (
        stmt.order_by(last_summarized_at.desc(), models.Session.id.asc())
        .limit(limit)
        .offset(offset)
    )

    rows = (await db.execute(stmt)).all()
    results: list[tuple[str, Summary]] = []
    for session_name, session_summaries in rows:
        stored: dict[str, Summary] = session_summaries or {}
        long_summary = stored.get(SummaryType.LONG.value)
        short_summary = stored.get(SummaryType.SHORT.value)
        if long_summary is None:
            summary = short_summary
        elif short_summary is None:
            summary = long_summary
        elif short_summary["message_id"] > long_summary["message_id"]:
            summary = short_summary
        else:
            summary = long_summary
        if summary is not None:
            results.append((session_name, summary))
    return results


async def get_both_summaries(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
) -> tuple[Summary | None, Summary | None]:
    """
    Get both short and long summaries for a given session.

    Args:
        db: Database session
        workspace_name: The workspace name
        session_name: The session name

    Returns:
        A tuple of the short and long summaries, or None if no summary exists
    """
    try:
        session = await crud.get_session(db, session_name, workspace_name)
    except ResourceNotFoundException:
        # If session doesn't exist, there's no summary to retrieve
        return None, None

    summaries: dict[str, Summary] = session.internal_metadata.get(SUMMARIES_KEY, {})
    return summaries.get(SummaryType.SHORT.value), summaries.get(SummaryType.LONG.value)


async def get_session_context(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    token_limit: int,
    *,
    cutoff: int | None = None,
    include_summary: bool = True,
) -> tuple[schemas.Summary | None, list[models.Message]]:
    """
    Get session context similar to the API endpoint but for internal use.

    Args:
        db: Database session
        workspace_name: The workspace name
        session_name: The session name
        token_limit: Maximum tokens for the context
        cutoff: Optional message ID to stop at (exclusive)
        include_summary: Whether to include summary if available

    Returns:
        Tuple of (summary, messages) where summary is a Summary pydantic model (or None)
        and messages is the list of message objects
    """
    if token_limit <= 0:
        return None, []

    summary = None
    messages_tokens = token_limit
    messages_start_id = 0

    if include_summary:
        # Allocate 40% of tokens to summary, 60% to messages
        summary_tokens_limit = int(token_limit * 0.4)

        latest_short_summary, latest_long_summary = await get_both_summaries(
            db, workspace_name, session_name
        )

        long_len = latest_long_summary["token_count"] if latest_long_summary else 0
        short_len = latest_short_summary["token_count"] if latest_short_summary else 0

        # Return the longest summary that fits within the token limit
        if (
            latest_long_summary
            and long_len <= summary_tokens_limit
            and long_len > short_len
        ):
            summary = schemas.Summary(
                content=latest_long_summary["content"],
                message_id=latest_long_summary["message_id"],
                summary_type=latest_long_summary["summary_type"],
                created_at=latest_long_summary["created_at"],
                token_count=latest_long_summary["token_count"],
                message_public_id=latest_long_summary.get("message_public_id", ""),
            )
            messages_tokens = token_limit - latest_long_summary["token_count"]
            messages_start_id = latest_long_summary["message_id"]
        elif (
            latest_short_summary and short_len <= summary_tokens_limit and short_len > 0
        ):
            summary = schemas.Summary(
                content=latest_short_summary["content"],
                message_id=latest_short_summary["message_id"],
                summary_type=latest_short_summary["summary_type"],
                created_at=latest_short_summary["created_at"],
                token_count=latest_short_summary["token_count"],
                message_public_id=latest_short_summary.get("message_public_id", ""),
            )
            messages_tokens = token_limit - latest_short_summary["token_count"]
            messages_start_id = latest_short_summary["message_id"]
        elif latest_short_summary or latest_long_summary:
            # A summary exists but does not fit the 40% allocation. The caller
            # receives `summary: null`, which is indistinguishable from a session
            # that has none, so this is reported rather than left at debug.
            logger.info(
                "Summary dropped: budget %s too small (short=%s, long=%s, limit=%s)",
                summary_tokens_limit,
                short_len or None,
                long_len or None,
                token_limit,
            )
        else:
            logger.debug(
                "No summary for get_context with token limit %s. Normal for a new session.",
                token_limit,
            )

    # Get recent messages after summary
    messages = await crud.get_messages_id_range(
        db,
        workspace_name,
        session_name,
        start_id=messages_start_id,
        end_id=cutoff,
        token_limit=messages_tokens,
    )

    return summary, messages


async def get_session_context_formatted(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    token_limit: int,
    *,
    cutoff: int | None = None,
    include_summary: bool = True,
) -> str:
    """
    Get formatted session context as a string for internal use (e.g., deriver).

    This is a convenience wrapper around get_session_context that formats
    the output as a string.
    """
    if token_limit <= 0:
        return ""

    summary, messages = await get_session_context(
        db,
        workspace_name,
        session_name,
        token_limit,
        cutoff=cutoff,
        include_summary=include_summary,
    )

    # Format the messages
    messages_text = _format_messages(messages)
    summary_content = summary.content if summary else ""

    if summary_content and messages_text:
        return f"""<summary>
{summary_content}
</summary>

<recent_messages>
{messages_text}
</recent_messages>"""
    elif summary_content:
        return f"""<summary>
{summary_content}
</summary>"""
    elif messages_text:
        return messages_text
    else:
        return ""


def _format_messages(messages: list[models.Message]) -> str:
    """
    Format a list of messages into a string by concatenating their content and
    prefixing each with the peer name.
    """
    if len(messages) == 0:
        return ""
    return "\n".join([f"{msg.peer_name}: {msg.content}" for msg in messages])


# Covers prompt text the scaffold estimate leaves out: the "no previous summary"
# placeholder and the digits of the target length.
_PROMPT_SLACK_TOKENS = 64

_TRUNCATION_MARKER = " [truncated]"


def _count_prompt_tokens(text: str) -> int:
    """Count `text`'s tokens as they will appear in a summary prompt.

    The prompt builders run `inspect.cleandoc`, which expands tabs, so tabs are
    counted expanded. Text that looks like a tokenizer control token is counted
    as ordinary text rather than falling back to an estimate from its length.
    """
    return len(tokenizer.encode(text.expandtabs(), disallowed_special=()))


def _fit_messages_to_budget(
    messages: list[models.Message], budget_tokens: int
) -> tuple[list[models.Message], str, bool]:
    """Keep the newest messages whose formatted lines fit in `budget_tokens`.

    Messages are taken newest first and the older ones dropped. If even the
    newest does not fit, its content is clipped to the budget and marked as
    truncated. The messages themselves are never modified; clipping only
    changes the formatted text.

    Args:
        messages: The summary window, oldest first.
        budget_tokens: Tokens available for the formatted messages.

    Returns:
        The kept messages, oldest first; their text formatted as by
        `_format_messages`; and whether the one kept message was clipped. The
        messages and text are empty when nothing fits.
    """
    lines: list[str] = []
    used_tokens = 0
    for message in reversed(messages):
        line = f"{message.peer_name}: {message.content}"
        # One more token for the newline that joins this line to the next.
        cost = _count_prompt_tokens(line) + 1
        if used_tokens + cost > budget_tokens:
            break
        lines.append(line)
        used_tokens += cost

    clipped = False
    if not lines and messages:
        clipped_line = _clip_message_line(messages[-1], budget_tokens - 1)
        if clipped_line is not None:
            lines.append(clipped_line)
            clipped = True

    kept = messages[len(messages) - len(lines) :] if lines else []
    return kept, "\n".join(reversed(lines)), clipped


def _fit_oldest_messages_to_budget(
    messages: list[models.Message], target_tokens: int, budget_tokens: int
) -> tuple[list[models.Message], str, bool]:
    """Keep the oldest messages whose formatted lines fit in `target_tokens`.

    The catch-up counterpart of `_fit_messages_to_budget`: messages are taken
    oldest first and the newer ones left for the next chunk. The oldest message
    is always taken, even past the target, while it fits in `budget_tokens`;
    past that, its content is clipped to the budget and marked as truncated.

    Returns:
        The kept messages, oldest first; their text formatted as by
        `_format_messages`; and whether the one kept message was clipped.
    """
    lines: list[str] = []
    used_tokens = 0
    for message in messages:
        line = f"{message.peer_name}: {message.content}"
        cost = _count_prompt_tokens(line) + 1
        limit = budget_tokens if not lines else target_tokens
        if used_tokens + cost > limit:
            break
        lines.append(line)
        used_tokens += cost

    if not lines and messages:
        clipped_line = _clip_message_line(messages[0], budget_tokens - 1)
        if clipped_line is None:
            return [], "", False
        return messages[:1], clipped_line, True
    return messages[: len(lines)], "\n".join(lines), False


def _clip_message_line(message: models.Message, budget_tokens: int) -> str | None:
    """Format `message` with its content clipped to fit in `budget_tokens`.

    Keeps the longest prefix of the content that fits. The line is measured as
    the prompt will see it (tabs expanded), which can be far longer than the raw
    content's tokens, so the prefix is found by binary search over raw tokens
    rather than by subtracting an overflow.

    Returns None when no content fits beside the peer name and truncation
    marker.
    """
    prefix = f"{message.peer_name}: "
    content_tokens = tokenizer.encode(message.content, disallowed_special=())

    def line_of(kept_tokens: int) -> str:
        """The line keeping the first `kept_tokens` tokens of the content."""
        return (
            prefix + tokenizer.decode(content_tokens[:kept_tokens]) + _TRUNCATION_MARKER
        )

    best: str | None = None
    low, high = 1, len(content_tokens)
    while low <= high:
        middle = (low + high) // 2
        line = line_of(middle)
        if _count_prompt_tokens(line) <= budget_tokens:
            best = line
            low = middle + 1
        else:
            high = middle - 1
    return best
