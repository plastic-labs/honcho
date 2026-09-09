"""Seed conclusions at every reasoning level, from inside the api container.

Run by sandbox.sh, never from the host:

    compose exec -T -e SANDBOX_FIXTURE_JSON="$(cat fixture.json)" api \
        /app/.venv/bin/python - < sandbox/inject_conclusions.py

Level and premise links are not reachable from the public API. `crud.create_observations`
hardcodes `level="explicit"`, `source_ids=NULL`, `internal_metadata={}`, and `ConclusionCreate`
has no field for any of them. The columns exist on Document; only an in-process caller can set
them. So this script imports Honcho's own write helpers, which is why it runs in the container
rather than beside seed.py: the container already *is* Honcho's venv, with the api's settings
and a correctly wired embedding client.

That comes at a price worth stating. These are internal APIs with no stability contract, and
they are imported out of the *pinned* image, not the working tree — so a digest bump can move
them underneath us. `check_signatures` fails loudly on that rather than letting a broken seed
look like a working one.

Two passes, not three: premise indices point into the same peer's `explicit` list, so both
derived levels reference pass-1 ids and nothing has to reference a derived id.

Everything here is asserted exactly, because every silent-failure mode in this path reduces a
count without raising: exact-content dedup is always on and cannot be switched off, semantic
dedup replaces rows, per-item embedding failures drop rows, and the session-purity invariant
skips explicit rows with no session. A seed that quietly wrote nothing is the failure this
whole sandbox exists to prevent.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

LEVELS = ("explicit", "deductive", "inductive")
DERIVED_LEVELS = ("deductive", "inductive")

# Levels whose premise text renders under a different metadata key. The working representation
# prints DocumentMetadata.premises for deductive and .sources for inductive, and each is read
# only for its own level, so putting the text under the wrong one renders nothing.
PREMISE_FIELD = {"deductive": "premises", "inductive": "sources"}


class InjectError(RuntimeError):
    """Seeding conclusions did not reach a usable state."""


@dataclass(frozen=True)
class SeededConclusion:
    """One conclusion from the fixture, with its premises resolved to explicit indices."""

    content: str
    premises: tuple[int, ...] = ()


@dataclass(frozen=True)
class PairPlan:
    """One (observer, observed) collection's worth of seeding, keyed by level."""

    observer: str
    observed: str
    by_level: dict[str, list[SeededConclusion]]

    def derived(self) -> dict[str, list[SeededConclusion]]:
        """The levels that carry premises, skipping any the fixture left empty."""
        return {
            level: self.by_level[level]
            for level in DERIVED_LEVELS
            if self.by_level[level]
        }

    def __str__(self) -> str:
        return f"{self.observer} -> {self.observed}"


@dataclass(frozen=True)
class Internals:
    """The Honcho internals this script writes through.

    Resolved once inside run() rather than imported at module scope, because importing
    src.* binds the engine and embedding client from the environment and that must happen
    with the api's settings in place.
    """

    crud: Any
    schemas: Any
    models: Any
    embedding_client: Any
    tracked_db: Any


def log(message: str) -> None:
    print(f"[conclusions] {message}", flush=True)


def check_signatures(crud: Any, schemas: Any) -> None:
    """Fail loudly if the image's internals moved.

    Cheap insurance against the real hazard of importing unversioned internals out of a pinned
    image: without this, a renamed keyword surfaces as a TypeError mid-seed, or worse, as a
    seed that silently wrote fewer rows.
    """
    expected = {
        "create_documents": (
            "documents",
            "workspace_name",
            "observer",
            "observed",
            "deduplicate",
        ),
        "create_observations": ("observations", "workspace_name"),
        "get_or_create_collection": ("workspace_name", "observer", "observed"),
        "get_documents_by_ids": ("workspace_name", "document_ids"),
        "get_child_observations": (
            "workspace_name",
            "parent_id",
            "observer",
            "observed",
        ),
    }
    for name, params in expected.items():
        signature = inspect.signature(getattr(crud, name))
        missing = [param for param in params if param not in signature.parameters]
        if missing:
            raise InjectError(
                f"crud.{name} is missing expected parameters {missing} in this image. "
                "The sandbox seeds conclusions through Honcho internals, which carry no "
                "stability contract; bump sandbox/image.env and update this script together."
            )

    for model, fields in (
        (
            schemas.DocumentCreate,
            ("content", "level", "metadata", "embedding", "source_ids"),
        ),
        (
            schemas.DocumentMetadata,
            ("message_ids", "message_created_at", "premises", "sources"),
        ),
    ):
        missing = [name for name in fields if name not in model.model_fields]
        if missing:
            raise InjectError(
                f"{model.__name__} is missing expected fields {missing} in this image. "
                "Bump sandbox/image.env and update this script together."
            )


def normalize(
    items: list[Any], level: str, peer_id: str, explicit_count: int
) -> list[SeededConclusion]:
    """Accept either a bare string or {content, premises}, and validate premise indices.

    An out-of-range index is rejected here rather than written as a dangling source id. Nothing
    downstream validates source_ids -- a bad one resolves to "referenced N premise IDs but none
    found in database" at read time, which is precisely the quiet wrongness to avoid.
    """
    normalized: list[SeededConclusion] = []
    for position, item in enumerate(items):
        if isinstance(item, str):
            content, premises = item, []
        elif isinstance(item, dict):
            content = item.get("content")
            premises = item.get("premises", [])
            unknown = set(item) - {"content", "premises"}
            if unknown:
                raise InjectError(
                    f"{peer_id}.{level}[{position}] has unknown keys {sorted(unknown)}"
                )
        else:
            raise InjectError(
                f"{peer_id}.{level}[{position}] must be a string or an object, "
                f"got {type(item).__name__}"
            )

        if not content or not content.strip():
            raise InjectError(f"{peer_id}.{level}[{position}] has empty content")
        if premises and level == "explicit":
            raise InjectError(
                f"{peer_id}.explicit[{position}] declares premises. Explicit conclusions come "
                "straight from messages and are the premises other levels point at."
            )
        for index in premises:
            if not isinstance(index, int) or not 0 <= index < explicit_count:
                raise InjectError(
                    f"{peer_id}.{level}[{position}] premise {index!r} is not a valid index into "
                    f"{peer_id}.explicit (which has {explicit_count} entries)"
                )

        normalized.append(SeededConclusion(content=content, premises=tuple(premises)))
    return normalized


def resolve_observers(spec: dict[str, Any], peers: list[dict[str, Any]]) -> list[str]:
    """Who holds conclusions about this peer.

    The peer carrying the keys is the observed. Absent an explicit override, the observers are
    every peer configured to observe others -- which for the standard harness shape is exactly
    the assistant. Resolving to nobody is an error: the conclusions would be written into no
    collection at all and the seed would report success having stored nothing.
    """
    observed = spec["id"]
    override = spec.get("observer")
    if override:
        return [override]

    observers = [
        peer["id"]
        for peer in peers
        if peer.get("observe_others") and peer["id"] != observed
    ]
    if not observers:
        raise InjectError(
            f"peer {observed!r} has seeded conclusions but no other peer observes it: "
            "no fixture peer besides itself sets observe_others, and it declares no "
            f'explicit "observer". The conclusions would be written nowhere. Either give '
            f'{observed!r} an "observer", or set observe_others on the peer that should '
            "hold them."
        )
    return observers


async def latest_message_timestamp(
    internals: Internals, db: Any, workspace: str, session: str
) -> str | None:
    """The conversation's own clock, for the representation to render.

    Derived conclusions are stamped with the last ingested message time rather than the seed's
    wall-clock, so the representation shows when the conversation happened. Honcho's own dream
    path back-dates the same way.
    """
    from sqlalchemy import func, select

    messages = internals.models.Message
    result = await db.execute(
        select(func.max(messages.created_at)).where(
            messages.workspace_name == workspace,
            messages.session_name == session,
        )
    )
    newest = result.scalar_one_or_none()
    return newest.strftime("%Y-%m-%dT%H:%M:%SZ") if newest else None


def assert_clean(result: Any, requested: int, label: str) -> None:
    """No row may be dropped, deduped, or replaced.

    create_documents reports these as counters rather than raising, so without this the seed
    reports success while having written fewer conclusions than the fixture declares. Exact
    content dedup cannot be disabled, so this fires on a reseed over existing content too --
    which is correct: seed.py always starts from an empty database.
    """
    created = len(result.created_documents)
    counters = {
        "exact duplicates within the batch": result.exact_dup_in_batch_count,
        "exact duplicates already stored": result.exact_dup_existing_count,
        "semantically rejected": result.semantic_dup_rejected_count,
        "semantically replaced": result.semantic_dup_replaced_count,
    }
    dropped = {reason: count for reason, count in counters.items() if count}
    if created != requested or dropped:
        detail = ", ".join(f"{reason}: {count}" for reason, count in dropped.items())
        raise InjectError(
            f"{label}: asked for {requested} conclusions, stored {created}"
            + (f" ({detail})" if dropped else "")
            + ". Conclusion contents must be unique; near-identical text is collapsed by "
            "Honcho's dedup before it reaches the database."
        )


async def seed_explicit(
    internals: Internals, workspace: str, session: str, plan: PairPlan
) -> list[str]:
    """Pass 1. Returns the stored ids, in fixture order, for premises to point at.

    The public write path is the only one that hands back rows, so it is how the premise ids
    are obtained -- and it does no dedup, so nothing is silently collapsed.
    """
    crud = internals.crud
    explicit = plan.by_level["explicit"]

    async with internals.tracked_db("sandbox.seed_explicit") as db:
        await crud.get_or_create_collection(
            db, workspace, observer=plan.observer, observed=plan.observed
        )
        if not explicit:
            return []

        documents = await crud.create_observations(
            db,
            [
                internals.schemas.ConclusionCreate(
                    content=conclusion.content,
                    observer_id=plan.observer,
                    observed_id=plan.observed,
                    # Explicit rows must carry a session: create_documents refuses
                    # session-less explicit rows on a session-purity invariant, and an
                    # explicit conclusion genuinely does come from a conversation.
                    session_id=session,
                )
                for conclusion in explicit
            ],
            workspace,
        )
        if len(documents) != len(explicit):
            raise InjectError(
                f"{plan}: asked for {len(explicit)} explicit conclusions, "
                f"stored {len(documents)}"
            )
        return [document.id for document in documents]


def build_derived_document(
    internals: Internals,
    session: str,
    level: str,
    conclusion: SeededConclusion,
    explicit: list[SeededConclusion],
    premise_ids: list[str],
    embedding: list[float],
    message_created_at: str,
) -> Any:
    """One DocumentCreate for a derived conclusion, with its premise links and display text."""
    source_ids = [premise_ids[index] for index in conclusion.premises]
    metadata: dict[str, Any] = {
        # Not derived from messages, but the representation reads this to render the
        # conclusion's timestamp.
        "message_ids": [],
        "message_created_at": message_created_at,
        "source_ids": source_ids,
        PREMISE_FIELD[level]: [
            explicit[index].content for index in conclusion.premises
        ],
    }
    if level == "inductive":
        metadata["pattern_type"] = "tendency"
        metadata["confidence"] = "high" if len(source_ids) > 1 else "low"

    return internals.schemas.DocumentCreate(
        content=conclusion.content,
        # The session, not None: the dreamer stamps its output with one session
        # (dream_scheduler picks whichever holds the most recent explicit conclusion) and
        # threads it through create_tool_executor, so a session-less derived row is not
        # what real mode would produce.
        #
        # This does not make them visible to a session-scoped read. ALLOWLIST_SAFE_LEVELS
        # in src/utils/representation.py serves only explicit under a session allowlist,
        # whatever the stamp says, because the dreamer reads across all sessions and
        # scoping its output would leak. So representation(session=...) is explicit-only
        # by construction; drop the session argument to see these.
        session_name=session,
        level=level,
        times_derived=1,
        metadata=internals.schemas.DocumentMetadata(**metadata),
        embedding=embedding,
        source_ids=source_ids,
    )


async def seed_derived(
    internals: Internals,
    workspace: str,
    session: str,
    plan: PairPlan,
    premise_ids: list[str],
) -> dict[str, int]:
    """Pass 2. Both derived levels cite pass-1 ids, so neither needs the other's ids back."""
    crud = internals.crud
    explicit = plan.by_level["explicit"]
    stored: dict[str, int] = {}

    async with internals.tracked_db("sandbox.seed_derived") as db:
        message_created_at = await latest_message_timestamp(
            internals, db, workspace, session
        )
        if message_created_at is None:
            raise InjectError(
                f"session {session!r} has no messages, so derived conclusions have no "
                "conversation timestamp to carry. Seed messages before conclusions."
            )

        for level, conclusions in plan.derived().items():
            contents = [conclusion.content for conclusion in conclusions]
            embeddings = await internals.embedding_client.simple_batch_embed(
                contents, on_oversize="truncate"
            )
            if len(embeddings) != len(contents):
                raise InjectError(
                    f"{plan}: embedded {len(embeddings)} of {len(contents)} "
                    f"{level} conclusions"
                )

            documents = [
                build_derived_document(
                    internals,
                    session,
                    level,
                    conclusion,
                    explicit,
                    premise_ids,
                    embedding,
                    message_created_at,
                )
                for conclusion, embedding in zip(conclusions, embeddings, strict=True)
            ]
            result = await crud.create_documents(
                db,
                documents,
                workspace,
                observer=plan.observer,
                observed=plan.observed,
                # Semantic dedup would replace or reject a seeded row against whatever the
                # deriver already wrote, making the fixture's counts depend on the provider.
                deduplicate=False,
            )
            assert_clean(result, len(conclusions), f"{plan} {level}")
            stored[level] = len(result.created_documents)

    return stored


async def inject_pair(
    internals: Internals, workspace: str, session: str, plan: PairPlan
) -> dict[str, int]:
    """Seed one (observer, observed) collection. Returns per-level counts actually stored."""
    premise_ids = await seed_explicit(internals, workspace, session, plan)
    stored = {"explicit": len(premise_ids)} if premise_ids else {}

    if plan.derived():
        stored |= await seed_derived(internals, workspace, session, plan, premise_ids)
        # Outside the write session on purpose: create_documents commits as it goes, and the
        # check should read the committed rows back on its own connection rather than through
        # the identity map of the session that wrote them.
        await verify_links(internals, workspace, plan, premise_ids)

    return stored


async def verify_links(
    internals: Internals, workspace: str, plan: PairPlan, premise_ids: list[str]
) -> None:
    """Confirm the premise links actually traverse, through Honcho's own read helpers.

    Nothing in the write path validates source_ids: a dangling id is stored happily and then
    degrades quietly at read time to "referenced N premise IDs but none found in database",
    while the conclusions endpoint does not expose source_ids at all. So a completely broken
    reasoning tree looks exactly like a working one from outside.

    Both directions are read back from the stored rows rather than from what this script
    intended, which is the only version of the check that can fail. Downward: every cited
    premise must be reachable from its children, which is what catches a child written into a
    different (observer, observed) collection. Upward: the source_ids those children actually
    carry must all resolve to live rows.
    """
    crud = internals.crud
    cited = sorted(
        {
            premise_ids[index]
            for conclusions in plan.derived().values()
            for conclusion in conclusions
            for index in conclusion.premises
        }
    )
    if not cited:
        return

    async with internals.tracked_db("sandbox.verify_links") as db:
        declared: set[str] = set()
        for premise_id in cited:
            children = await crud.get_child_observations(
                db,
                workspace,
                premise_id,
                observer=plan.observer,
                observed=plan.observed,
            )
            if not children:
                raise InjectError(
                    f"{plan}: premise {premise_id} has no reachable children, so the "
                    "reasoning tree does not traverse downward. Premise and conclusion "
                    "must share one (observer, observed) pair."
                )
            for child in children:
                declared.update(child.source_ids or [])

        resolved = await crud.get_documents_by_ids(db, workspace, sorted(declared))
        missing = declared - {document.id for document in resolved}
        if missing:
            raise InjectError(
                f"{plan}: {len(missing)} stored premise id(s) resolve to nothing, so the "
                f"reasoning chain would read as empty. First: {sorted(missing)[0]}"
            )

    log(f"{plan}: {len(cited)} premise link(s) traverse both ways")


def plan_pairs(fixture: dict[str, Any]) -> list[PairPlan]:
    """Turn the fixture's per-peer conclusion keys into one plan per collection."""
    peers = fixture["peers"]
    planned: list[PairPlan] = []
    for spec in peers:
        if not any(spec.get(level) for level in LEVELS):
            continue
        explicit_count = len(spec.get("explicit", []))
        by_level = {
            level: normalize(spec.get(level, []), level, spec["id"], explicit_count)
            for level in LEVELS
        }
        if any(by_level[level] for level in DERIVED_LEVELS) and not explicit_count:
            raise InjectError(
                f"peer {spec['id']!r} has derived conclusions but no explicit ones for their "
                "premises to point at."
            )
        for observer in resolve_observers(spec, peers):
            planned.append(PairPlan(observer, spec["id"], by_level))
    return planned


async def run(fixture: dict[str, Any]) -> int:
    from src import crud, models, schemas
    from src.cache.client import close_cache, init_cache
    from src.db import engine
    from src.dependencies import tracked_db
    from src.embedding_client import embedding_client

    check_signatures(crud, schemas)

    planned = plan_pairs(fixture)
    if not planned:
        log("fixture declares no conclusions - nothing to seed")
        return 0

    internals = Internals(
        crud=crud,
        schemas=schemas,
        models=models,
        embedding_client=embedding_client,
        tracked_db=tracked_db,
    )

    # cashews decorates the collection lookups, and an unconfigured backend raises rather than
    # degrading, so the cache has to be up before the first crud call.
    await init_cache()
    try:
        for plan in planned:
            stored = await inject_pair(
                internals, fixture["workspace"], fixture["session"], plan
            )
            summary = " ".join(f"{level}={stored.get(level, 0)}" for level in LEVELS)
            log(f"{plan}: {summary}")
    finally:
        await close_cache()
        # Without this the process can hang on exit holding pool connections, which inside
        # `compose exec` looks like the seed itself wedging.
        await engine.dispose()

    return 0


def main() -> int:
    raw = os.environ.get("SANDBOX_FIXTURE_JSON")
    if not raw:
        raise InjectError(
            "SANDBOX_FIXTURE_JSON is unset. This script is run by sandbox.sh, which passes the "
            "fixture in through the environment; it is not meant to be run by hand."
        )
    return asyncio.run(run(json.loads(raw)))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except InjectError as exc:
        print(f"[conclusions] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
