import logging
from inspect import cleandoc as c

from sqlalchemy import delete

from src import crud, models, schemas
from src.config import settings
from src.dependencies import tracked_db
from src.embedding_client import embedding_client
from src.exceptions import ResourceNotFoundException
from src.utils.clients import honcho_llm_call
from src.utils.formatting import format_datetime_utc
from src.utils.logging import conditional_observe
from src.utils.queue_payload import DreamPayload
from src.utils.representation import (
    ExplicitObservation,
    Representation,
)

logger = logging.getLogger(__name__)


def consolidation_prompt(
    representation: Representation,
) -> str:
    """
    Generate the prompt for user representation consolidation.

    Args:
        representation: The user representation to consolidate

    Returns:
        A prompt string for the LLM to consolidate the representation
    """
    representation_as_json = representation.model_dump_json(indent=2)

    return c(
        f"""
You are an agent that consolidates observations about an entity. You will be presented with a list of EXPLICIT and DEDUCTIVE observations. **Reduce** the number of observations, if possible, by combining similar observations. **ONLY** include information that is **GIVEN**. Create the highest-quality observations with the given information. Observations must always be maximally concise.

{representation_as_json}
"""
    )


@conditional_observe(name="[Dream] Consolidate Call")
async def _consolidate_call(
    representation: Representation,
) -> Representation:
    prompt = consolidation_prompt(representation)

    response = await honcho_llm_call(
        llm_settings=settings.DREAM,
        prompt=prompt,
        max_tokens=settings.DREAM.MAX_OUTPUT_TOKENS,
        track_name="Dream Call",
        response_model=Representation,
        enable_retry=True,
        retry_attempts=3,
    )

    return response.content


async def process_consolidate_dream(payload: DreamPayload, workspace_name: str) -> None:
    """
    Process a consolidation dream task.

    Consolidation means taking all the documents in a collection and merging
    similar observations into a single, best-quality observation document.
    """

    logger.info(
        "Starting consolidate dream for workspace=%s, observer=%s, observed=%s",
        workspace_name,
        payload.observer,
        payload.observed,
    )

    # grab 100 recent documents in the collection
    # in the future, we can perform clustering on documents by semantic similarity and do
    # multiple clusters at once. for now, can just sample documents and do what we can.
    async with tracked_db("dream_consolidate") as db:
        # First verify the collection exists
        try:
            collection = await crud.get_collection(
                db,
                workspace_name,
                observer=payload.observer,
                observed=payload.observed,
            )
            logger.debug(
                "Found collection id=%s for workspace=%s, observer=%s, observed=%s",
                collection.id,
                workspace_name,
                payload.observer,
                payload.observed,
            )
        except ResourceNotFoundException:
            logger.warning(
                "Collection does not exist for workspace=%s, observer=%s, observed=%s",
                workspace_name,
                payload.observer,
                payload.observed,
            )
            return

        documents_query = crud.get_all_documents(
            workspace_name,
            observer=payload.observer,
            observed=payload.observed,
            limit=100,
        )

        logger.debug(
            "Executing document query: %s",
            str(documents_query.compile(compile_kwargs={"literal_binds": True})),
        )

        result = await db.execute(documents_query)
        documents = result.scalars().all()

        if not documents:
            return

        logger.info("consolidating %d documents", len(documents))

        # Pre-calculate data structures needed for processing so we don't need attached objects
        cluster_representation = Representation.from_documents(documents)
        document_ids = [doc.id for doc in documents]
        total_times_derived = sum(doc.times_derived for doc in documents)

    # We treat all fetched documents as a single cluster for now
    clusters = [(cluster_representation, document_ids, total_times_derived)]

    # for each cluster, call llm to consolidate the representation if possible
    for representation, doc_ids, times_derived in clusters:
        await _consolidate_cluster(
            representation,
            doc_ids,
            times_derived,
            workspace_name,
            observer=payload.observer,
            observed=payload.observed,
        )


async def _consolidate_cluster(
    representation: Representation,
    document_ids: list[str],
    total_times_derived: int,
    workspace_name: str,
    *,
    observer: str,
    observed: str,
) -> None:
    """
    Consolidate a cluster of documents, treated as a Representation, into a smaller one.
    Removes old documents and replaces them with consolidated versions while preserving metadata.
    """
    if len(document_ids) <= 1:
        logger.info(
            "Cluster has %d documents, skipping consolidation", len(document_ids)
        )
        return

    logger.info("unconsolidated representation:\n%s", representation)

    consolidated_representation = await _consolidate_call(representation)
    logger.info("consolidated representation:\n%s", consolidated_representation)

    new_documents = [
        *consolidated_representation.explicit,
        *consolidated_representation.deductive,
    ]

    if not new_documents:
        return

    # Collect all contents for batch embedding
    contents: list[str] = []
    for obs in new_documents:
        if isinstance(obs, ExplicitObservation):
            contents.append(obs.content)
        else:
            contents.append(obs.conclusion)

    # Batch embed all contents at once for better performance
    embeddings = await embedding_client.simple_batch_embed(contents)

    documents_to_create: list[schemas.DocumentCreate] = []

    for i, obs in enumerate(new_documents):
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
            message_ids=obs.message_ids,
            message_created_at=format_datetime_utc(obs.created_at),
            premises=premises,
        )

        documents_to_create.append(
            schemas.DocumentCreate(
                content=content,
                session_name=obs.session_name,
                level=level,
                times_derived=total_times_derived,
                metadata=metadata,
                embedding=embeddings[i],
            )
        )

    async with tracked_db("dream_consolidate_write") as db:
        # bulk create documents
        await crud.create_documents(
            db,
            documents_to_create,
            workspace_name,
            observer=observer,
            observed=observed,
        )

        # delete old documents
        await db.execute(
            delete(models.Document).where(models.Document.id.in_(document_ids))
        )

        await db.commit()

    logger.info(
        "consolidated %d documents into %d new documents",
        len(document_ids),
        len(new_documents),
    )
