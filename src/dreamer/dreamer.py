import logging
from collections.abc import Sequence

import sentry_sdk
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.dependencies import tracked_db
from src.dreamer.prompts import consolidation_prompt
from src.embedding_client import embedding_client
from src.utils.clients import honcho_llm_call
from src.utils.formatting import format_datetime_utc
from src.utils.queue_payload import DreamPayload
from src.utils.representation import (
    ExplicitObservation,
    Representation,
)

logger = logging.getLogger(__name__)


@sentry_sdk.trace
async def process_dream(
    payload: DreamPayload,
) -> None:
    """
    Process a dream task by performing collection maintenance operations.

    Args:
        payload: The dream task payload containing workspace, peer, and dream type information
    """
    logger.info(
        f"Processing dream task: {payload.dream_type} for {payload.workspace_name}/{payload.observer}/{payload.observed}"
    )

    try:
        if payload.dream_type == "consolidate":
            await _process_consolidate_dream(payload)
        ## TODO other dream types

    except Exception as e:
        logger.error(
            f"Error processing dream task {payload.dream_type} for {payload.observer}/{payload.observed}: {str(e)}",
            exc_info=True,
        )
        if settings.SENTRY.ENABLED:
            sentry_sdk.capture_exception(e)
        # Don't re-raise - we want to mark the dream task as processed even if it fails


async def _process_consolidate_dream(payload: DreamPayload) -> None:
    """
    Process a consolidation dream task.

    Consolidation means taking all the documents in a collection and merging
    similar observations into a single, best-quality observation document.

    TODO: need to determine a way to do this on a subset of documents since
    collections will grow very large.
    """
    logger.info(
        f"""
(っ- ‸ - ς)ᶻ z 𐰁 ᶻ z 𐰁 ᶻ z 𐰁\n
DREAM: consolidating documents for {payload.workspace_name}/{payload.observer}/{payload.observed}\n
𐰁 z ᶻ 𐰁 z ᶻ 𐰁 z ᶻ(っ- ‸ - ς)"""
    )

    # get all documents in the collection
    async with tracked_db("dream_consolidate") as db:
        documents = await crud.get_all_documents(
            db,
            payload.workspace_name,
            observer=payload.observer,
            observed=payload.observed,
        )

        logger.info("found %d documents to consolidate", len(documents))

        # TODO: create clusters of documents based on cosine similarity
        # clusters = await create_document_clusters(documents)

        # logger.info("created %d clusters", len(clusters))
        clusters = [documents]

        # for each cluster, call llm to consolidate the representation if possible
        for cluster in clusters:
            await _consolidate_cluster(
                cluster,
                payload.workspace_name,
                db,
                observer=payload.observer,
                observed=payload.observed,
            )


async def _consolidate_cluster(
    cluster: Sequence[models.Document],
    workspace_name: str,
    db: AsyncSession,
    *,
    observer: str,
    observed: str,
) -> None:
    """
    Consolidate a cluster of documents, treated as a Representation, into a smaller one.
    Removes old documents and replaces them with consolidated versions while preserving metadata.
    """
    if len(cluster) <= 1:
        logger.info("Cluster has %d documents, skipping consolidation", len(cluster))
        return

    cluster_representation = crud.representation_from_documents(cluster)
    logger.info("unconsolidated representation:\n%s", cluster_representation)

    consolidated_representation = await consolidate_call(cluster_representation)
    logger.info("consolidated representation:\n%s", consolidated_representation)

    collection = await crud.get_collection(
        db, workspace_name, observer=observer, observed=observed
    )
    if not collection:
        logger.error(
            "Collection for %s/%s not found, cannot save consolidated documents",
            observer,
            observed,
        )
        return

    # TODO: less hacky preservation of times_derived
    total_times_derived = sum(
        doc.internal_metadata.get("times_derived", 1) for doc in cluster
    )

    new_documents = [
        *consolidated_representation.explicit,
        *consolidated_representation.deductive,
    ]

    documents_to_create: list[schemas.DocumentCreate] = []

    for obs in new_documents:
        if isinstance(obs, ExplicitObservation):
            content = obs.content
            level = "explicit"
            premises = None
        else:
            content = obs.conclusion
            level = "deductive"
            premises = obs.premises
        # NOTE: other kinds of observations here in the future

        metadata = schemas.DocumentMetadata(
            times_derived=total_times_derived,
            message_ids=obs.message_ids,
            message_created_at=format_datetime_utc(obs.created_at),
            session_name=obs.session_name or "",
            level=level,
            premises=premises,
        )

        embedding = await embedding_client.embed(content)

        documents_to_create.append(
            schemas.DocumentCreate(
                content=content,
                metadata=metadata,
                embedding=embedding,
            )
        )

    # bulk create documents
    await crud.create_documents(
        db, documents_to_create, workspace_name, observer=observer, observed=observed
    )

    # delete old documents
    for doc in cluster:
        await db.delete(doc)

    await db.commit()

    logger.info(
        "consolidated %d documents into %d new documents",
        len(cluster),
        len(new_documents),
    )


async def consolidate_call(
    representation: Representation,
) -> Representation:
    prompt = consolidation_prompt(representation)

    response = await honcho_llm_call(
        provider=settings.DREAM.PROVIDER,
        model=settings.DREAM.MODEL,
        prompt=prompt,
        max_tokens=settings.DREAM.MAX_OUTPUT_TOKENS,
        track_name="Dream Call",
        response_model=Representation,
        enable_retry=True,
        retry_attempts=3,
    )

    return response.content
