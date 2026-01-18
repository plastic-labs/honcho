"""Provenance storage for persisting agent execution traces.

Provides functions for storing provenance traces to the database,
with support for both single and batch operations.
"""

from logging import getLogger

from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.provenance.tracer import ProvenanceTrace

logger = getLogger(__name__)


async def store_trace(
    db: AsyncSession,
    trace: ProvenanceTrace,
) -> models.FalsificationTrace:
    """
    Store a single provenance trace to the database.

    Currently, provenance traces are stored as FalsificationTrace records.
    This may be extended to support different trace types in the future.

    Args:
        db: Database session
        trace: ProvenanceTrace to store

    Returns:
        Created FalsificationTrace database object

    Raises:
        ValueError: If required fields are missing
    """
    # For now, we store all agent traces as FalsificationTrace records
    # The reasoning_chain field contains the complete trace information

    # Build reasoning chain from trace data
    reasoning_chain = {
        "trace_id": trace.trace_id,
        "agent_type": trace.agent_type,
        "start_time": trace.start_time.isoformat(),
        "end_time": trace.end_time.isoformat() if trace.end_time else None,
        "duration_ms": trace.duration_ms,
        "success": trace.success,
        "error_message": trace.error_message,
        "input_data": trace.input_data,
        "output_data": trace.output_data,
        "steps": [
            {
                "step_number": step.step_number,
                "action": step.action,
                "input_data": step.input_data,
                "output_data": step.output_data,
                "timestamp": step.timestamp.isoformat(),
                "duration_ms": step.duration_ms,
                "metadata": step.metadata,
            }
            for step in trace.steps
        ],
        "metadata": trace.metadata,
    }

    # Extract search queries from steps (for GIN index)
    search_queries: list[str] = []
    for step in trace.steps:
        if "query" in step.output_data:
            search_queries.append(str(step.output_data["query"]))
        elif "queries" in step.output_data:
            queries = step.output_data["queries"]
            if isinstance(queries, list):
                search_queries.extend([str(q) for q in queries])

    # Determine final status
    if not trace.success:
        final_status = "untested"
    else:
        # Extract status from output data if available
        final_status = trace.output_data.get("status", "untested")

    # Extract prediction_id from input or metadata
    prediction_id = trace.input_data.get("prediction_id") or trace.metadata.get(
        "prediction_id", "unknown"
    )

    # Create FalsificationTrace schema
    trace_create = schemas.FalsificationTraceCreate(
        prediction_id=prediction_id,
        search_queries=search_queries if search_queries else None,
        contradicting_premise_ids=trace.output_data.get("contradicting_premise_ids"),
        reasoning_chain=reasoning_chain,
        final_status=final_status,
        search_count=len(search_queries),
        search_efficiency_score=trace.metadata.get("search_efficiency_score"),
        collection_id=trace.collection_id,
    )

    # Import crud function to avoid circular imports
    from src.crud.trace import create_trace

    trace_obj = await create_trace(db, trace_create, trace.workspace_name)

    logger.debug(
        "Stored provenance trace %s for agent %s in workspace %s",
        trace.trace_id,
        trace.agent_type,
        trace.workspace_name,
    )

    return trace_obj


async def batch_store_traces(
    db: AsyncSession,
    traces: list[ProvenanceTrace],
) -> list[models.FalsificationTrace]:
    """
    Store multiple provenance traces to the database in a batch.

    This function uses async batch insertion for better performance
    when storing many traces at once.

    Args:
        db: Database session
        traces: List of ProvenanceTrace objects to store

    Returns:
        List of created FalsificationTrace database objects

    Raises:
        ValueError: If any trace has missing required fields
    """
    if not traces:
        return []

    # Store each trace individually
    # TODO: Optimize with true batch insertion using db.add_all()
    trace_objects: list[models.FalsificationTrace] = []
    for trace in traces:
        try:
            trace_obj = await store_trace(db, trace)
            trace_objects.append(trace_obj)
        except Exception as e:
            logger.error(
                "Failed to store trace %s for agent %s: %s",
                trace.trace_id,
                trace.agent_type,
                e,
            )
            # Continue with other traces
            continue

    logger.info(
        "Batch stored %d/%d provenance traces",
        len(trace_objects),
        len(traces),
    )

    return trace_objects


async def store_document_provenance(
    db: AsyncSession,
    document: models.Document,
    agent_type: str,
    trace_id: str,
) -> models.Document:
    """
    Update a document with provenance information.

    Links a document to the agent that created it and its execution trace.

    Args:
        db: Database session
        document: Document to update
        agent_type: Agent type (abducer | predictor | falsifier | inductor | extractor)
        trace_id: Trace ID from the agent execution

    Returns:
        Updated document object
    """
    document.provenance_type = agent_type
    document.agent_trace_id = trace_id

    await db.commit()
    await db.refresh(document)

    logger.debug(
        "Updated document %s with provenance: agent=%s, trace=%s",
        document.id,
        agent_type,
        trace_id,
    )

    return document


async def batch_update_document_provenance(
    db: AsyncSession,
    documents: list[models.Document],
    agent_type: str,
    trace_id: str,
) -> list[models.Document]:
    """
    Update multiple documents with provenance information in a batch.

    Args:
        db: Database session
        documents: List of documents to update
        agent_type: Agent type
        trace_id: Trace ID from the agent execution

    Returns:
        List of updated document objects
    """
    if not documents:
        return []

    for document in documents:
        document.provenance_type = agent_type
        document.agent_trace_id = trace_id

    await db.commit()

    # Refresh all documents
    for document in documents:
        await db.refresh(document)

    logger.debug(
        "Batch updated %d documents with provenance: agent=%s, trace=%s",
        len(documents),
        agent_type,
        trace_id,
    )

    return documents
