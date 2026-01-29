"""
Introspection module for meta-cognitive analysis of Honcho's performance.

This module provides functionality to analyze how Honcho is performing for a workspace
and generate configuration suggestions based on observed patterns.
"""

from __future__ import annotations

import datetime
import json
import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings
from src.schemas import (
    IntrospectionReport,
    IntrospectionSignals,
    IntrospectionSuggestion,
)
from src.utils.clients import honcho_llm_call

logger = logging.getLogger(__name__)


# Reserved peer names for storing introspection reports
SYSTEM_OBSERVER = "_system"
INTROSPECTION_OBSERVED = "_introspection"


async def gather_introspection_context(
    db: AsyncSession,
    workspace_name: str,
) -> IntrospectionSignals:
    """
    Gather performance signals from a workspace for introspection analysis.

    Args:
        db: Database session
        workspace_name: Name of the workspace to analyze

    Returns:
        IntrospectionSignals containing performance metrics and configuration
    """
    # Calculate time window (last 30 days)
    since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)

    # Get dialectic trace statistics
    try:
        trace_stats = await crud.get_dialectic_trace_stats(
            db, workspace_name, since=since
        )
    except Exception as e:
        logger.warning(f"Failed to get dialectic trace stats: {e}")
        trace_stats = {
            "total_queries": 0,
            "avg_duration_ms": 0.0,
            "abstention_count": 0,
            "abstention_rate": 0.0,
        }

    # Get recent dialectic queries for sample analysis
    recent_queries: list[str] = []
    try:
        traces = await crud.get_dialectic_traces(db, workspace_name, limit=20)
        recent_queries = [t.query for t in traces]
    except Exception as e:
        logger.warning(f"Failed to get recent dialectic traces: {e}")

    # Get observation counts by level
    observations_by_level: dict[str, int] = {}
    total_observations = 0
    contradiction_count = 0
    try:
        # Query documents grouped by level
        stmt = (
            select(
                models.Document.level,
                func.count(models.Document.id).label("count"),
            )
            .where(models.Document.workspace_name == workspace_name)
            .where(models.Document.deleted_at.is_(None))
            .group_by(models.Document.level)
        )
        result = await db.execute(stmt)
        for row in result.all():
            level, count = row
            observations_by_level[level] = count
            total_observations += count
            if level == "contradiction":
                contradiction_count = count
    except Exception as e:
        logger.warning(f"Failed to get observation counts: {e}")

    # Get peer count
    total_peers = 0
    try:
        stmt = (
            select(func.count(models.Peer.id))
            .where(models.Peer.workspace_name == workspace_name)
            .where(~models.Peer.name.startswith("_"))  # Exclude system peers
        )
        result = await db.execute(stmt)
        total_peers = result.scalar() or 0
    except Exception as e:
        logger.warning(f"Failed to get peer count: {e}")

    # Get session count
    total_sessions = 0
    try:
        stmt = select(func.count(models.Session.id)).where(
            models.Session.workspace_name == workspace_name
        )
        result = await db.execute(stmt)
        total_sessions = result.scalar() or 0
    except Exception as e:
        logger.warning(f"Failed to get session count: {e}")

    # Get current agent configuration
    current_deriver_rules = ""
    current_dialectic_rules = ""
    try:
        agent_config = await crud.get_workspace_agent_config(db, workspace_name)
        current_deriver_rules = agent_config.deriver_rules
        current_dialectic_rules = agent_config.dialectic_rules
    except Exception as e:
        logger.warning(f"Failed to get workspace agent config: {e}")

    return IntrospectionSignals(
        total_dialectic_queries=int(trace_stats["total_queries"]),
        avg_dialectic_duration_ms=float(trace_stats["avg_duration_ms"]),
        abstention_count=int(trace_stats["abstention_count"]),
        abstention_rate=float(trace_stats["abstention_rate"]),
        recent_queries=recent_queries,
        total_observations=total_observations,
        observations_by_level=observations_by_level,
        contradiction_count=contradiction_count,
        total_peers=total_peers,
        total_sessions=total_sessions,
        current_deriver_rules=current_deriver_rules,
        current_dialectic_rules=current_dialectic_rules,
    )


def build_introspection_prompt(signals: IntrospectionSignals) -> str:
    """
    Build a prompt for the LLM to analyze workspace signals and suggest improvements.

    Args:
        signals: Performance signals gathered from the workspace

    Returns:
        A formatted prompt string for the LLM
    """
    queries_sample = (
        "\n".join(f"  - {q}" for q in signals.recent_queries[:10])
        if signals.recent_queries
        else "  (No recent queries)"
    )

    observations_breakdown = (
        "\n".join(
            f"  - {level}: {count}"
            for level, count in signals.observations_by_level.items()
        )
        if signals.observations_by_level
        else "  (No observations)"
    )

    return f"""You are analyzing the performance of a Honcho workspace to identify issues and suggest configuration improvements.

## Workspace Performance Signals

### Dialectic API Usage (Last 30 Days)
- Total queries: {signals.total_dialectic_queries}
- Average response time: {signals.avg_dialectic_duration_ms:.1f}ms
- Abstention count: {signals.abstention_count}
- Abstention rate: {signals.abstention_rate:.1%}

### Recent Query Samples
{queries_sample}

### Memory Statistics
- Total observations: {signals.total_observations}
- Observations by level:
{observations_breakdown}
- Contradictions detected: {signals.contradiction_count}

### Workspace Scale
- Total peers: {signals.total_peers}
- Total sessions: {signals.total_sessions}

### Current Configuration
Deriver rules:
```
{signals.current_deriver_rules or "(empty - using defaults)"}
```

Dialectic rules:
```
{signals.current_dialectic_rules or "(empty - using defaults)"}
```

## Your Task

1. **Analyze the application type**: Based on the query samples and usage patterns, what kind of application is this workspace likely supporting? (e.g., customer support, personal assistant, educational tool, etc.)

2. **Identify performance issues**: Look for:
   - High abstention rate (>30% suggests memory gaps or query-memory mismatch)
   - High contradiction count (suggests conflicting information being stored)
   - Low observation count relative to query volume (suggests underutilization of memory)
   - Query patterns that suggest certain topics aren't being captured

3. **Suggest specific rule changes**: For each suggestion, specify:
   - Whether it's for `deriver_rules` or `dialectic_rules`
   - The exact suggested rule text
   - Why this change would help

Respond with a JSON object matching this schema:
{{
    "performance_summary": "A 2-3 sentence summary of the workspace's performance",
    "identified_issues": ["Issue 1", "Issue 2", ...],
    "suggestions": [
        {{
            "target": "deriver_rules" | "dialectic_rules",
            "current_value": "current rule text",
            "suggested_value": "new rule text to add or replace with",
            "rationale": "why this change would help",
            "confidence": "high" | "medium" | "low"
        }}
    ]
}}

If there's insufficient data to make recommendations, return an empty suggestions array with an appropriate performance_summary."""


class IntrospectionLLMResponse(IntrospectionReport):
    """Schema for LLM response, excluding fields we'll fill in ourselves."""

    # Override these to make them optional since LLM won't provide them
    workspace_name: str = ""
    generated_at: datetime.datetime = datetime.datetime.now(datetime.timezone.utc)
    signals: IntrospectionSignals = IntrospectionSignals()


async def run_introspection(
    db: AsyncSession,
    workspace_name: str,
) -> IntrospectionReport | None:
    """
    Run introspection analysis for a workspace and generate a report.

    Args:
        db: Database session
        workspace_name: Name of the workspace to analyze

    Returns:
        IntrospectionReport with analysis and suggestions, or None if insufficient data
    """
    logger.info(f"Starting introspection for workspace {workspace_name}")

    # Gather signals
    signals = await gather_introspection_context(db, workspace_name)

    # Check for minimum data
    if signals.total_dialectic_queries == 0 and signals.total_observations == 0:
        logger.info(
            f"Insufficient data for introspection in workspace {workspace_name}"
        )
        report = IntrospectionReport(
            workspace_name=workspace_name,
            generated_at=datetime.datetime.now(datetime.timezone.utc),
            performance_summary="Insufficient data for analysis. No dialectic queries or observations found.",
            identified_issues=[],
            suggestions=[],
            signals=signals,
        )
        await store_introspection_report(db, workspace_name, report)
        return report

    # Build prompt and call LLM
    prompt = build_introspection_prompt(signals)

    try:
        llm_response = await honcho_llm_call(
            llm_settings=settings.DREAM,
            prompt=prompt,
            max_tokens=4096,
            track_name="introspection",
            json_mode=True,
            temperature=0.3,
        )

        # Parse the response
        response_text = (
            llm_response.content
            if hasattr(llm_response, "content")
            else str(llm_response)
        )

        # Parse JSON and validate
        try:
            response_data: dict[str, object] = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            response_data = {
                "performance_summary": f"Error parsing LLM response: {str(e)[:100]}",
                "identified_issues": [],
                "suggestions": [],
            }

        # Build suggestions list
        suggestions: list[IntrospectionSuggestion] = []
        raw_suggestions = response_data.get("suggestions", [])
        if isinstance(raw_suggestions, list):
            for raw_suggestion in raw_suggestions:
                if isinstance(raw_suggestion, dict):
                    try:
                        # Validate target field
                        target_raw = raw_suggestion.get("target", "deriver_rules")
                        if target_raw not in ("deriver_rules", "dialectic_rules"):
                            target_raw = "deriver_rules"

                        # Validate confidence field
                        confidence_raw = raw_suggestion.get("confidence", "low")
                        if confidence_raw not in ("high", "medium", "low"):
                            confidence_raw = "low"

                        suggestions.append(
                            IntrospectionSuggestion(
                                target=target_raw,  # type: ignore[arg-type]
                                current_value=str(raw_suggestion.get("current_value", "")),
                                suggested_value=str(raw_suggestion.get("suggested_value", "")),
                                rationale=str(raw_suggestion.get("rationale", "")),
                                confidence=confidence_raw,  # type: ignore[arg-type]
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to parse suggestion: {e}")

        # Extract fields with type coercion
        performance_summary = response_data.get("performance_summary", "Analysis completed.")
        if not isinstance(performance_summary, str):
            performance_summary = "Analysis completed."

        identified_issues_raw = response_data.get("identified_issues", [])
        identified_issues: list[str] = []
        if isinstance(identified_issues_raw, list):
            for issue in identified_issues_raw:
                identified_issues.append(str(issue))

        report = IntrospectionReport(
            workspace_name=workspace_name,
            generated_at=datetime.datetime.now(datetime.timezone.utc),
            performance_summary=performance_summary,
            identified_issues=identified_issues,
            suggestions=suggestions,
            signals=signals,
        )

    except Exception as e:
        logger.error(f"LLM call failed during introspection: {e}")
        report = IntrospectionReport(
            workspace_name=workspace_name,
            generated_at=datetime.datetime.now(datetime.timezone.utc),
            performance_summary=f"Error during analysis: {str(e)[:200]}",
            identified_issues=[],
            suggestions=[],
            signals=signals,
        )

    # Store the report
    await store_introspection_report(db, workspace_name, report)

    logger.info(
        f"Introspection completed for {workspace_name}: {len(report.suggestions)} suggestions"
    )
    return report


async def store_introspection_report(
    db: AsyncSession,
    workspace_name: str,
    report: IntrospectionReport,
) -> None:
    """
    Store an introspection report as a document in a reserved collection.

    Reports are stored with:
    - observer: _system
    - observed: _introspection
    - content: JSON-serialized report
    - embedding: None (not for semantic search)

    Args:
        db: Database session
        workspace_name: Name of the workspace
        report: The introspection report to store
    """
    from src import schemas

    try:
        # Ensure system peers exist
        await crud.get_or_create_peers(
            db,
            workspace_name,
            [
                schemas.PeerCreate(name=SYSTEM_OBSERVER),
                schemas.PeerCreate(name=INTROSPECTION_OBSERVED),
            ],
        )

        # Ensure collection exists
        await crud.get_or_create_collection(
            db,
            workspace_name,
            observer=SYSTEM_OBSERVER,
            observed=INTROSPECTION_OBSERVED,
        )

        # Serialize report to JSON
        report_json = report.model_dump_json()

        # Create document (without embedding - not for semantic search)
        doc = models.Document(
            workspace_name=workspace_name,
            observer=SYSTEM_OBSERVER,
            observed=INTROSPECTION_OBSERVED,
            content=report_json,
            level="explicit",
            times_derived=1,
            internal_metadata={
                "report_type": "introspection",
                "generated_at": report.generated_at.isoformat(),
            },
            session_name=None,
            embedding=None,
            sync_state="synced",  # No need to sync - no embedding
        )

        db.add(doc)
        await db.commit()

        logger.debug(
            f"Stored introspection report for {workspace_name}, doc_id={doc.id}"
        )

    except Exception as e:
        logger.error(f"Failed to store introspection report: {e}")
        await db.rollback()
        # Don't re-raise - storing the report is secondary to generating it
