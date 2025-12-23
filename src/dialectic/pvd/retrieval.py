"""
PVD Retrieval Orchestrator.

Coordinates all PVD components to provide enhanced memory retrieval.
"""

import asyncio
import logging
import time
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings
from src.dialectic.pvd.classifier import QueryClassification, classify_query
from src.dialectic.pvd.parameters import PVDParameters, generate_pvd_parameters
from src.dialectic.pvd.probability import compute_probabilities
from src.dialectic.pvd.scoring import compute_pvd_scores, rerank_by_pvd
from src.embedding_client import embedding_client

logger = logging.getLogger(__name__)


async def pvd_search(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    query: str,
    session_name: str | None,
    top_k: int = 25,
    semantic_oversample: int | None = None,
    levels: list[str] | None = None,
) -> tuple[list[models.Document], dict[str, Any]]:
    """
    PVD-enhanced memory search.

    Complete pipeline:
    1. Classify query → task type (parallel with embedding)
    2. Generate query embedding (parallel with classification)
    3. Generate parameters → α, β, γ
    4. Semantic oversample → fetch top_k * oversample candidates
    5. Build trees → compute p(v|e,a) and p(v|a)
    6. Compute PVD scores → apply formula
    7. Rerank and return top_k

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: Observer peer name
        observed: Observed peer name
        query: User's query
        session_name: Session identifier (None for global queries)
        top_k: Number of documents to return
        semantic_oversample: Oversample factor (defaults to settings)
        levels: Optional filter for document levels

    Returns:
        Tuple of (documents, metadata) where metadata includes:
        - query_type: Classified query type
        - parameters: PVD parameters used
        - score_breakdown: List of score details
        - timing_info: Performance metrics
        - pvd_fallback: Whether fallback was triggered
    """
    start_time = time.time()
    timing_info = {}
    pvd_fallback = False

    try:
        # Use configured oversample if not specified
        if semantic_oversample is None:
            semantic_oversample = settings.DIALECTIC.PVD.SEMANTIC_OVERSAMPLE

        logger.debug(
            f"PVD search: query='{query[:50]}...', session={session_name}, "
            f"top_k={top_k}, oversample={semantic_oversample}"
        )

        # Step 1 & 2: Classify query and generate embedding IN PARALLEL
        classify_start = time.time()
        classification_task = asyncio.create_task(classify_query(query, session_name))
        embedding_task = asyncio.create_task(embedding_client.embed(query))

        classification, query_embedding = await asyncio.gather(
            classification_task, embedding_task
        )
        query_embedding_np = np.array(query_embedding)
        timing_info["classification_ms"] = (time.time() - classify_start) * 1000

        logger.info(
            f"PVD: Classified as '{classification.query_type}' "
            f"(confidence: {classification.confidence:.2f})"
        )

        # Step 3: Generate PVD parameters
        params_start = time.time()
        parameters = await generate_pvd_parameters(
            query, classification, session_name
        )
        timing_info["parameter_generation_ms"] = (time.time() - params_start) * 1000

        logger.info(
            f"PVD parameters: α={parameters.alpha:.2f}, β={parameters.beta:.2f}, γ={parameters.gamma:.2f}"
        )

        # Step 4: Semantic oversample
        oversample_start = time.time()
        oversample_count = top_k * semantic_oversample
        candidate_docs = await crud.query_documents(
            db=db,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            query=query,
            top_k=oversample_count,
            embedding=query_embedding,  # Reuse embedding
            filters={"level": levels} if levels else None,
        )
        timing_info["semantic_oversample_ms"] = (time.time() - oversample_start) * 1000

        if not candidate_docs:
            logger.warning(f"No candidates found for query: {query[:50]}")
            return [], {
                "query_type": classification.query_type,
                "parameters": parameters,
                "score_breakdown": [],
                "timing_info": timing_info,
                "pvd_fallback": False,
            }

        logger.debug(
            f"Fetched {len(candidate_docs)} candidates (requested {oversample_count})"
        )

        # Step 5: Build trees and compute probabilities
        # Skip if weights are negligible (optimization)
        prob_start = time.time()
        if (
            parameters.beta >= settings.DIALECTIC.PVD.MIN_BETA_FOR_SESSION_TREE
            or parameters.gamma >= settings.DIALECTIC.PVD.MIN_GAMMA_FOR_GLOBAL_TREE
        ):
            probabilities = await compute_probabilities(
                db=db,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                session_name=session_name,
                candidate_docs=candidate_docs,
                tree_type=settings.DIALECTIC.PVD.TREE_TYPE,
                k=settings.DIALECTIC.PVD.TREE_K,
            )
        else:
            # Skip tree building, use uniform probabilities
            logger.debug("Skipping tree building (weights negligible)")
            from src.dialectic.pvd.probability import ProbabilityResult

            probabilities = {
                doc.id: ProbabilityResult(
                    document_id=doc.id,
                    p_entity=0.5,
                    p_anchor=0.5,
                    surprisal_entity=0.0,
                    surprisal_anchor=0.0,
                )
                for doc in candidate_docs
            }
        timing_info["probability_computation_ms"] = (time.time() - prob_start) * 1000

        # Step 6: Compute PVD scores
        scoring_start = time.time()
        scored_docs = compute_pvd_scores(
            candidate_docs=candidate_docs,
            query_embedding=query_embedding_np,
            probabilities=probabilities,
            parameters=parameters,
        )
        timing_info["scoring_ms"] = (time.time() - scoring_start) * 1000

        # Step 7: Rerank and return top-k
        rerank_start = time.time()
        top_docs = rerank_by_pvd(scored_docs, top_k)
        timing_info["rerank_ms"] = (time.time() - rerank_start) * 1000

        # Build score breakdown for top docs
        score_breakdown = [
            {
                "document_id": scored.document.id,
                "pvd_score": scored.pvd_score,
                "semantic_score": scored.semantic_score,
                "entity_log_prob": scored.entity_log_prob,
                "anchor_log_prob": scored.anchor_log_prob,
                "p_entity": scored.p_entity,
                "p_anchor": scored.p_anchor,
            }
            for scored in scored_docs[:top_k]
        ]

        timing_info["total_ms"] = (time.time() - start_time) * 1000

        logger.info(
            f"PVD search complete: returned {len(top_docs)}/{len(candidate_docs)} docs "
            f"in {timing_info['total_ms']:.0f}ms"
        )

        return top_docs, {
            "query_type": classification.query_type,
            "parameters": parameters,
            "score_breakdown": score_breakdown,
            "timing_info": timing_info,
            "pvd_fallback": pvd_fallback,
        }

    except Exception as e:
        logger.error(f"PVD search failed, falling back to semantic search: {e}", exc_info=True)
        pvd_fallback = True

        # Fallback: Pure semantic search
        try:
            fallback_start = time.time()
            fallback_docs = await crud.query_documents(
                db=db,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                query=query,
                top_k=top_k,
                filters={"level": levels} if levels else None,
            )
            timing_info["fallback_ms"] = (time.time() - fallback_start) * 1000
            timing_info["total_ms"] = (time.time() - start_time) * 1000

            logger.warning(
                f"PVD fallback: returned {len(fallback_docs)} docs via semantic search"
            )

            return fallback_docs, {
                "query_type": "info_extract",  # Default
                "parameters": None,
                "score_breakdown": [],
                "timing_info": timing_info,
                "pvd_fallback": pvd_fallback,
            }
        except Exception as fallback_error:
            logger.error(
                f"Fallback search also failed: {fallback_error}", exc_info=True
            )
            # Return empty results
            return [], {
                "query_type": "info_extract",
                "parameters": None,
                "score_breakdown": [],
                "timing_info": {},
                "pvd_fallback": pvd_fallback,
            }
