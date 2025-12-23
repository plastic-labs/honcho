"""
Probabilistic Vector Database (PVD) retrieval system for Honcho.

This module implements anchor-conditioned probability distributions
and surprisal-based ranking to optimize dialectic retrieval for
benchmark performance.
"""

from src.dialectic.pvd.classifier import QueryClassification, QueryType, classify_query
from src.dialectic.pvd.parameters import PVDParameters, generate_pvd_parameters
from src.dialectic.pvd.probability import ProbabilityResult, compute_probabilities
from src.dialectic.pvd.retrieval import pvd_search
from src.dialectic.pvd.scoring import ScoredDocument, compute_pvd_scores, rerank_by_pvd

__all__ = [
    "QueryClassification",
    "QueryType",
    "classify_query",
    "PVDParameters",
    "generate_pvd_parameters",
    "ProbabilityResult",
    "compute_probabilities",
    "ScoredDocument",
    "compute_pvd_scores",
    "rerank_by_pvd",
    "pvd_search",
]
