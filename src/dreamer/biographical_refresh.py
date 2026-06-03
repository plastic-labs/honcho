"""
Biographical-refresh pass.

The stored peer card (the biographical *Profile*) is written by the deduction
specialist during dreams, but nothing systematically re-checks each card entry
against newer messages. So a fact can go stale and never get reconciled — e.g.
the card keeps saying "close to accepting a role" long after the messages
confirmed the role was accepted (signed contract, got an EIN).

This module adds a focused reconciliation pass that runs as part of the dream
cycle. It diffs the current peer card against recent messages authored by the
observed peer and supersedes facts that have gone stale. Like the minimal
deriver and the summarizer — and unlike the specialists — it is a single
structured-output LLM call, not an agentic tool loop: predictable cost, runs at
most once per dream.

Entry point: `run_biographical_refresh`, called from `orchestrator.run_dream`
after the specialists have run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from pydantic import BaseModel, Field
from sqlalchemy import select

from src import crud, models
from src.config import settings
from src.dependencies import tracked_db
from src.llm import honcho_llm_call
from src.llm.types import LLMTelemetryContext
from src.schemas.configuration import ResolvedConfiguration
from src.telemetry.events.llm import CallPurpose
from src.utils.agent_tools import MAX_PEER_CARD_FACTS, normalize_peer_card_entries
from src.utils.formatting import format_new_turn_with_timestamp
from src.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)


@dataclass
class BiographicalRefreshResult:
    """Outcome of a biographical-refresh pass, for logging/telemetry."""

    # Whether the LLM call completed without raising.
    success: bool
    # Whether the card was actually rewritten (a stale fact was superseded).
    card_updated: bool
    # Number of entries the model reported as superseded/stale.
    superseded_count: int
    input_tokens: int
    output_tokens: int
    duration_ms: float


class _Reconciliation(BaseModel):
    """Structured output of the reconciliation call."""

    no_changes: bool = Field(
        default=False,
        description=(
            "True if no entry in the current card is stale, outdated, or "
            "contradicted by the recent messages. When true, reconciled_card "
            "is ignored and the card is left untouched."
        ),
    )
    reconciled_card: list[str] = Field(
        default_factory=list,
        description=(
            "The COMPLETE reconciled peer card: re-emit every still-valid entry "
            "verbatim, rewrite superseded entries to their current value, and "
            "drop entries that recent messages have made false. Omitting an "
            "entry deletes it, so only omit facts that are now false. Each entry "
            "must keep the IDENTITY:/ATTRIBUTE:/RELATIONSHIP:/INSTRUCTION: "
            "prefix format."
        ),
    )
    superseded: list[str] = Field(
        default_factory=list,
        description=(
            "Short human-readable notes for each change, e.g. "
            "'close to accepting a role' -> 'works at Carrum (1099 contractor)'. "
            "For logging only."
        ),
    )


def _build_prompt(observed: str, peer_card: list[str], formatted_messages: str) -> str:
    card_block = "\n".join(f"- {entry}" for entry in peer_card)
    return f"""You are a biographical reconciliation agent. Your only job is to keep \
the stored Profile of {observed} consistent with what the recent messages show \
to be true RIGHT NOW.

You are given the CURRENT PROFILE (a list of stable identity facts about \
{observed}) and the most RECENT MESSAGES authored by {observed}. A fact on the \
Profile can drift out of date: a plan becomes a completed action, a job search \
becomes a job, a city someone was "moving to" becomes the city they live in.

## CURRENT PROFILE

{card_block}

## RECENT MESSAGES (most recent last)

{formatted_messages}

## YOUR TASK

Go through each Profile entry and decide whether the recent messages have made it \
stale, outdated, or false:

- **Still valid** — re-emit the entry verbatim in `reconciled_card`.
- **Superseded** — the messages show the fact has moved on. Rewrite the entry to \
its current value (keeping the same prefix), and add a `superseded` note. \
Example: `ATTRIBUTE: Close to accepting a role at Carrum` becomes \
`ATTRIBUTE: Works at Carrum as a 1099 contractor` once messages confirm the \
signed contract and EIN.
- **Now false** — the messages contradict the fact and there is no replacement. \
Omit the entry from `reconciled_card`.

## RULES

1. **Conservative.** Only change an entry when the recent messages CLEARLY \
contradict or supersede it. Ambiguity or silence is not evidence of change — \
when in doubt, keep the entry verbatim.
2. **Reconcile, don't author.** Do not add brand-new facts that are merely \
mentioned in the messages — that is the deriver's job. Only rewrite entries that \
already exist on the Profile (and only to track a value that genuinely moved).
3. **Preserve format.** Every entry in `reconciled_card` must start with one of \
`IDENTITY:`, `ATTRIBUTE:`, `RELATIONSHIP:`, or `INSTRUCTION:` followed by a \
space. Keep each entry to one concise fact.
4. **Subject is {observed}.** Never write facts about other participants.
5. **No churn.** If nothing is stale, set `no_changes` to true and stop. Do not \
reword entries cosmetically.

Return the complete reconciled card only when at least one entry actually changed."""


async def _get_recent_authored_messages(observed: str, workspace_name: str) -> str:
    """Fetch and format the observed peer's most recent messages.

    Pulls up to MAX_RECENT_MESSAGES authored by `observed` across the workspace,
    newest first, then keeps the newest ones that fit within MAX_INPUT_TOKENS and
    renders them oldest-first for the prompt. Returns "" when there are none.
    """
    cfg = settings.DREAM.BIOGRAPHICAL_REFRESH
    async with tracked_db("biographical_refresh.messages") as db:
        stmt = (
            select(models.Message)
            .where(
                models.Message.workspace_name == workspace_name,
                models.Message.peer_name == observed,
            )
            .order_by(models.Message.id.desc())
            .limit(cfg.MAX_RECENT_MESSAGES)
        )
        result = await db.execute(stmt)
        recent_newest_first = list(result.scalars().all())

    if not recent_newest_first:
        return ""

    # Keep the newest messages that fit the token budget, then flip to
    # chronological order for the prompt.
    budget = cfg.MAX_INPUT_TOKENS
    kept: list[models.Message] = []
    used = 0
    for msg in recent_newest_first:
        cost = estimate_tokens(msg.content) or 0
        if kept and used + cost > budget:
            break
        kept.append(msg)
        used += cost

    kept.reverse()
    return "\n".join(
        format_new_turn_with_timestamp(msg.content, msg.created_at, msg.peer_name)
        for msg in kept
    )


async def run_biographical_refresh(
    workspace_name: str,
    observer: str,
    observed: str,
    *,
    configuration: ResolvedConfiguration | None = None,
    parent_run_id: str | None = None,
) -> BiographicalRefreshResult | None:
    """Reconcile the (observer, observed) peer card against recent messages.

    Returns None when the pass is skipped (disabled, peer-card writes disabled,
    empty/too-small card, or no recent messages). Otherwise returns a result
    describing the outcome — `card_updated` is True only when a stale fact was
    actually superseded.

    Uses short-lived DB sessions and never holds one across the LLM call.
    """
    cfg = settings.DREAM.BIOGRAPHICAL_REFRESH
    if not cfg.ENABLED:
        return None

    # Respect the same gate the specialists use: if peer-card writes are
    # disabled for this workspace/session, there is nothing to reconcile.
    if configuration is not None and not configuration.peer_card.create:
        return None

    start_time = time.perf_counter()

    # 1. Current card (the biographical Profile).
    async with tracked_db("biographical_refresh.card") as db:
        peer_card = await crud.get_peer_card(
            db, workspace_name, observer=observer, observed=observed
        )

    if not peer_card or len(peer_card) < cfg.MIN_CARD_ENTRIES:
        return None

    # 2. Recent messages authored by the observed peer.
    formatted_messages = await _get_recent_authored_messages(observed, workspace_name)
    if not formatted_messages:
        return None

    # 3. Single structured-output reconciliation call.
    prompt = _build_prompt(observed, peer_card, formatted_messages)
    try:
        response = await honcho_llm_call(
            model_config=settings.DREAM.DEDUCTION_MODEL_CONFIG,
            prompt=prompt,
            max_tokens=cfg.MAX_OUTPUT_TOKENS,
            track_name="Biographical Refresh",
            response_model=_Reconciliation,
            json_mode=True,
            enable_retry=True,
            retry_attempts=3,
            trace_name="biographical_refresh",
            telemetry=LLMTelemetryContext(
                workspace_name=workspace_name,
                call_purpose=CallPurpose.DREAM_BIOGRAPHICAL_REFRESH.value,
                parent_category="biographical_refresh",
                run_id=parent_run_id,
                observer=observer,
                observed=observed,
            ),
        )
    except Exception:
        logger.error(
            "[%s] Biographical refresh LLM call failed for %s/%s/%s",
            parent_run_id,
            workspace_name,
            observer,
            observed,
            exc_info=True,
        )
        duration_ms = (time.perf_counter() - start_time) * 1000
        return BiographicalRefreshResult(
            success=False,
            card_updated=False,
            superseded_count=0,
            input_tokens=0,
            output_tokens=0,
            duration_ms=duration_ms,
        )

    parsed = response.content
    duration_ms = (time.perf_counter() - start_time) * 1000

    def _result(card_updated: bool, superseded_count: int) -> BiographicalRefreshResult:
        return BiographicalRefreshResult(
            success=True,
            card_updated=card_updated,
            superseded_count=superseded_count,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            duration_ms=duration_ms,
        )

    if parsed.no_changes:
        return _result(card_updated=False, superseded_count=0)

    # 4. Normalize/validate with the same rules as `update_peer_card`.
    normalized, _rejected = normalize_peer_card_entries(parsed.reconciled_card)
    if len(normalized) > MAX_PEER_CARD_FACTS:
        normalized = normalized[:MAX_PEER_CARD_FACTS]

    # Guard: never clear the card if the model returned nothing usable, and skip
    # the write when the reconciled card is identical to the current one.
    if not normalized or normalized == peer_card:
        return _result(card_updated=False, superseded_count=0)

    # 5. Persist the reconciled card.
    async with tracked_db("biographical_refresh.write") as db:
        await crud.set_peer_card(
            db,
            workspace_name,
            normalized,
            observer=observer,
            observed=observed,
        )

    logger.info(
        "[%s] Biographical refresh updated peer card for %s/%s/%s "
        + "(%d -> %d entries, %d superseded)",
        parent_run_id,
        workspace_name,
        observer,
        observed,
        len(peer_card),
        len(normalized),
        len(parsed.superseded),
    )
    return _result(card_updated=True, superseded_count=len(parsed.superseded))
