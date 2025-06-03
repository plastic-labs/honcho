# DIA Framework Architecture

## Overview

The DIA framework builds psychological models of users through surprise-based reasoning across three cognitive levels. When a user message violates expectations, the system extracts facts about their psychology and maintains these insights across sessions.

## System Flow

User messages trigger background processing that creates summaries, detects surprising information, and stores psychological facts for future conversations. The system operates asynchronously to avoid blocking chat responses.

**Key Components:**
- `src/deriver/queue.py` - Session-based message queuing
- `src/deriver/consumer.py` - Processing orchestration
- `src/deriver/surprise_reasoner.py` - Surprise detection and fact generation
- `src/deriver/fact_saver.py` - Asynchronous fact storage

## Surprise Detection

The system maintains expectations about users across three reasoning levels. When new information contradicts these expectations, it triggers fact extraction and model updates.

**Reasoning Levels:**
- **Abductive**: High-level psychological insights and hypotheses
- **Inductive**: Behavioral patterns and tendencies
- **Deductive**: Explicit facts and concrete statements

The reasoner uses conservative analysis to prevent constant revision. Early versions showed excessive fact changes, so the system now requires substantial evidence (20% threshold) to modify existing understanding.

## Summary System

Conversations are summarized at regular intervals to provide context for fact interpretation:
- Short summaries every 20 messages
- Long summaries every 60 messages

Summaries create hierarchical context, with long summaries informing short ones. Facts link to their originating summary for contextual retrieval.

## Fact Storage

Facts are stored asynchronously through a queue system that prevents database conflicts. Each fact includes metadata linking it to sessions, summaries, and reasoning levels.

**Metadata Structure:**
```python
{
    "level": "abductive|inductive|deductive",
    "session_id": "session_id", 
    "summary_id": "summary_public_id",
    "session_context": "truncated_summary"
}
```

Duplicate detection uses vector similarity (0.85 threshold) to prevent redundant facts.

## Dialectic API

The chat endpoint retrieves relevant facts using semantic search across multiple queries. Facts are grouped by reasoning level and session context:

```
=== REASONING-BASED USER UNDERSTANDING ===

## ABDUCTIVE (High-level insights):
From session: Career discussion...
  • Values autonomy in decisions
  • Shows risk-averse patterns

## INDUCTIVE (Behavioral patterns):
From session: Productivity focus...
  • Consistently optimizes workflows
  • Prefers visual planning tools

## DEDUCTIVE (Explicit facts):
From session: Personal details...
  • Works remotely from Portland
  • 5+ years data science experience
```

## Cross-Session Intelligence

User models persist across all sessions. Facts accumulate over time while maintaining session-specific context. Recent facts are prioritized, and the system preserves conversation history through linked summaries.

## Model Configuration

Different models handle specialized tasks:
- **Claude 3.7 Sonnet**: Main dialectic responses
- **Llama 3.1 8B**: Query generation
- **Gemini 2.0 Flash**: Surprise reasoning
- **Llama 3.3 70B**: Theory of mind processing

## Error Handling

The system degrades gracefully when components fail. Individual fact failures don't block processing, and the queue system prevents database bottlenecks through serialized writes.

## Implementation Status

The system is fully operational with asynchronous fact saving, conservative reasoning, and multi-session intelligence. Recent enhancements include improved transaction safety and comprehensive error tracking.
