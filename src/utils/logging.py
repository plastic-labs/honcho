"""
Custom utility logging functions for Langfuse integration.
This module provides specialized formatters for all @observe decorated functions
to create beautiful, human-readable markdown output in Langfuse traces.
"""

import json
from textwrap import shorten
from typing import Any, cast

from rich.console import Console

# Global console instance for consistent formatting
console = Console(markup=False)


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_dict_as_markdown(data: dict[str, Any], title: str = "Data") -> str:
    """Format a dictionary as readable markdown."""
    lines = [f"### {title}"]

    for key, value in data.items():
        if isinstance(value, dict | list):
            formatted_value = json.dumps(value, indent=2)
            lines.append(f"**{key}:**")
            lines.append(f"```json\n{formatted_value}\n```")
        else:
            lines.append(f"**{key}:** {value}")

    return "\n".join(lines)


def format_list_as_markdown(items: list[Any], title: str = "Items") -> str:
    """Format a list as readable markdown."""
    if not items:
        return f"### {title}\n*No items*"

    lines = [f"### {title} ({len(items)})"]

    for i, item in enumerate(items, 1):
        if isinstance(item, str):
            lines.append(f"{i}. {truncate_text(item, 300)}")
        else:
            lines.append(f"{i}. {str(item)}")

    return "\n".join(lines)


def format_metadata_section(
    workspace_name: str,
    peer_name: str,
    session_name: str | None = None,
    additional_context: dict[str, Any] | None = None,
) -> str:
    """Format standard metadata section for context."""
    lines = [
        "### 🎯 Context",
        f"**Workspace:** {workspace_name}",
        f"**Peer:** {peer_name}",
    ]

    if session_name:
        lines.append(f"**Session:** {session_name}")

    if additional_context:
        for key, value in additional_context.items():
            lines.append(f"**{key.title()}:** {value}")

    return "\n".join(lines)


def format_timing_section(
    start_time: float | None = None, end_time: float | None = None
) -> str:
    """Format timing information."""
    if start_time and end_time:
        duration = end_time - start_time
        return f"### ⏱️ Performance\n**Duration:** {duration:.2f}s"
    return ""


# =============================================================================
# AGENT FUNCTIONS
# =============================================================================


def format_chat_input(
    workspace_name: str,
    peer_name: str,
    session_name: str | None,
    query: str,
    stream: bool = False,
) -> str:
    """Format input for agent.chat() function."""
    lines = ["# 🤖 Agent Chat Input\n"]

    # Context section
    lines.append(
        format_metadata_section(
            workspace_name,
            peer_name,
            session_name,
            {"Stream Mode": "Yes" if stream else "No"},
        )
    )

    # Query section
    lines.append("\n### 💬 Query")
    lines.append(f"```\n{truncate_text(query, 500)}\n```")

    return "\n".join(lines)


def format_chat_output(
    response_content: str,
    elapsed_time: float | None = None,
    additional_metrics: dict[str, Any] | None = None,
) -> str:
    """Format output for agent.chat() function."""
    lines = ["# 🤖 Agent Chat Output\n"]

    # Response section
    lines.append("### 💡 Response")
    lines.append(f"```\n{truncate_text(response_content, 1000)}\n```")

    # Performance section
    if elapsed_time:
        lines.append("\n### ⏱️ Performance")
        lines.append(f"**Response Time:** {elapsed_time:.2f}s")

    # Additional metrics
    if additional_metrics:
        lines.append(f"\n{format_dict_as_markdown(additional_metrics, '📊 Metrics')}")

    return "\n".join(lines)


# =============================================================================
# MESSAGE PROCESSING FUNCTIONS
# =============================================================================


def format_process_message_input(
    content: str,
    workspace_name: str,
    peer_name: str,
    target_name: str,
    session_name: str | None,
    message_id: int,
    created_at_str: str | None = None,
) -> str:
    """Format input for process_message() function."""
    lines = ["# 📝 Process Message Input\n"]

    # Context section
    lines.append(
        format_metadata_section(
            workspace_name,
            peer_name,
            session_name,
            {
                "Target": target_name,
                "Message ID": message_id,
                "Created At": created_at_str or "Not specified",
            },
        )
    )

    # Message content
    lines.append("\n### 💬 Message Content")
    lines.append(f"```\n{truncate_text(content, 800)}\n```")

    return "\n".join(lines)


def format_process_message_output(
    extracted_facts: list[str],
    processing_time: float | None = None,
    storage_metrics: dict[str, Any] | None = None,
) -> str:
    """Format output for process_message() function."""
    lines = ["# 📝 Process Message Output\n"]

    # Facts section
    lines.append(format_list_as_markdown(extracted_facts, "🧠 Extracted Facts"))

    # Performance section
    if processing_time:
        lines.append("\n### ⏱️ Performance")
        lines.append(f"**Processing Time:** {processing_time:.2f}s")

    # Storage metrics
    if storage_metrics:
        lines.append(
            f"\n{format_dict_as_markdown(storage_metrics, '💾 Storage Metrics')}"
        )

    return "\n".join(lines)


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================


def format_judge_input(
    prompt: str, max_retries: int = 3, additional_context: dict[str, Any] | None = None
) -> str:
    """Format input for LLM judge functions."""
    lines = ["# ⚖️ LLM Judge Input\n"]

    lines.append("### 🎯 Configuration")
    lines.append(f"**Max Retries:** {max_retries}")

    if additional_context:
        lines.append(f"\n{format_dict_as_markdown(additional_context, '📋 Context')}")

    lines.append("\n### 📝 Judge Prompt")
    lines.append(f"```\n{truncate_text(prompt, 800)}\n```")

    return "\n".join(lines)


def format_judge_output(
    judgment: str, score: float | None = None, metadata: dict[str, Any] | None = None
) -> str:
    """Format output for LLM judge functions."""
    lines = ["# ⚖️ LLM Judge Output\n"]

    if score is not None:
        lines.append("### 📊 Score")
        lines.append(f"**Result:** {score}")

    lines.append("\n### 🧠 Judgment")
    lines.append(f"```\n{truncate_text(judgment, 800)}\n```")

    if metadata:
        lines.append(f"\n{format_dict_as_markdown(metadata, '📋 Metadata')}")

    return "\n".join(lines)


def format_question_eval_input(
    qa_data: dict[str, Any],
    eval_key: str,
    workspace: str,
    dialectic_name: str,
    additional_config: dict[str, Any] | None = None,
) -> str:
    """Format input for question evaluation functions."""
    lines = ["# 📊 Question Evaluation Input\n"]

    lines.append("### 🎯 Configuration")
    lines.append(f"**Workspace:** {workspace}")
    lines.append(f"**Dialectic:** {dialectic_name}")
    lines.append(f"**Eval Key:** {eval_key}")

    if additional_config:
        lines.append(
            f"\n{format_dict_as_markdown(additional_config, '⚙️ Additional Config')}"
        )

    # Evidence section (optional)
    if additional_config:
        ev_ids = additional_config.get("evidence_ids")
        ev_links = additional_config.get("evidence_ingestion_traces")
        ev_text_map = additional_config.get("evidence_text_map")
        if ev_ids:
            lines.append("\n### 📚 Evidence")
            lines.append(format_list_as_markdown(ev_ids, "Evidence IDs"))
            if ev_links:
                lines.append("\n#### 🔗 Ingestion Trace Links")
                for ev in ev_ids:
                    text_part = ""  # default
                    if isinstance(ev_text_map, dict):
                        # Safely get and cast the text value with proper typing
                        text_map: dict[str, Any] = cast(dict[str, Any], ev_text_map)
                        raw_txt: Any = text_map.get(ev, "")
                        try:
                            txt: str = str(raw_txt) if raw_txt else ""
                        except (TypeError, ValueError):
                            txt = ""

                        if txt:
                            text_part = (
                                f' - "{shorten(txt, width=60, placeholder="...")}"'
                            )
                    bullet = f"* {ev}{text_part}"
                    if isinstance(ev_links, dict) and ev in ev_links:
                        bullet = f"* [{ev}{text_part}]({ev_links[ev]})"
                    lines.append(bullet)

    lines.append("\n### ❓ Question Data")
    if "question" in qa_data:
        lines.append(f"**Question:** {truncate_text(str(qa_data['question']), 300)}")

    if "answer" in qa_data:
        lines.append(
            f"**Expected Answer:** {truncate_text(str(qa_data['answer']), 300)}"
        )

    # Show other relevant fields
    other_fields = {k: v for k, v in qa_data.items() if k not in ["question", "answer"]}
    if other_fields:
        lines.append(f"\n{format_dict_as_markdown(other_fields, '📋 Additional Data')}")

    return "\n".join(lines)


def format_question_eval_output(
    scored_qa: dict[str, Any],
    _scores: list[float] | None = None,
    evaluation_metadata: dict[str, Any] | None = None,
) -> str:
    """Format output for question evaluation functions."""
    lines = ["# 📊 Question Evaluation Output\n"]

    # Include individual scores but omit the average score to reduce clutter

    lines.append("\n### 📝 Scored QA")
    if "prediction" in scored_qa:
        lines.append(f"**Prediction:** {truncate_text(scored_qa['prediction'], 4000)}")

    # Show scoring details
    scoring_fields = {k: v for k, v in scored_qa.items() if "score" in k.lower()}
    if scoring_fields:
        lines.append(
            f"\n{format_dict_as_markdown(scoring_fields, '📊 Scoring Details')}"
        )

    if evaluation_metadata:
        lines.append(
            f"\n{format_dict_as_markdown(evaluation_metadata, '📋 Evaluation Metadata')}"
        )

    return "\n".join(lines)


# =============================================================================
# DIALECTIC FUNCTIONS
# =============================================================================


def format_semantic_queries_input(query: str) -> str:
    """Format input for semantic query generation."""
    lines = ["# 🔍 Semantic Query Generation Input\n"]

    lines.append("### 📝 Original Query")
    lines.append(f"```\n{truncate_text(query, 500)}\n```")

    return "\n".join(lines)


def format_semantic_queries_output(queries: list[str]) -> str:
    """Format output for semantic query generation."""
    lines = ["# 🔍 Semantic Query Generation Output\n"]

    lines.append(format_list_as_markdown(queries, "🎯 Generated Queries"))

    return "\n".join(lines)


def format_tom_inference_input(chat_history: str) -> str:
    """Format input for theory-of-mind inference."""
    lines = ["# 🧠 Theory of Mind Inference Input\n"]

    lines.append("### 💬 Chat History")
    lines.append(f"```\n{truncate_text(chat_history, 800)}\n```")

    return "\n".join(lines)


def format_tom_inference_output(inference: str) -> str:
    """Format output for theory-of-mind inference."""
    lines = ["# 🧠 Theory of Mind Inference Output\n"]

    lines.append("### 💡 Inference")
    lines.append(f"```\n{truncate_text(inference, 800)}\n```")

    return "\n".join(lines)


def format_long_term_facts_input(
    query: str, workspace_name: str, peer_name: str, collection_name: str
) -> str:
    """Format input for long-term facts retrieval."""
    lines = ["# 💾 Long-term Facts Retrieval Input\n"]

    lines.append(
        format_metadata_section(
            workspace_name,
            peer_name,
            additional_context={"Collection": collection_name},
        )
    )

    lines.append("\n### 🔍 Query")
    lines.append(f"```\n{truncate_text(query, 400)}\n```")

    return "\n".join(lines)


def format_long_term_facts_output(facts: list[str]) -> str:
    """Format output for long-term facts retrieval."""
    lines = ["# 💾 Long-term Facts Retrieval Output\n"]

    lines.append(format_list_as_markdown(facts, "📚 Retrieved Facts"))

    return "\n".join(lines)


def format_dialectic_chat_input(
    workspace_name: str,
    peer_name: str,
    session_name: str | None,
    query: str,
    stream: bool = False,
) -> str:
    """Format input for dialectic chat functions."""
    lines = ["# 🗣️ Dialectic Chat Input\n"]

    lines.append(
        format_metadata_section(
            workspace_name,
            peer_name,
            session_name,
            {"Stream Mode": "Yes" if stream else "No"},
        )
    )

    lines.append("\n### 💬 Queries")
    lines.append(f"```\n{truncate_text(query, 500)}\n```")

    return "\n".join(lines)


def format_dialectic_chat_output(response: str) -> str:
    """Format output for dialectic chat functions."""
    lines = ["# 🗣️ Dialectic Chat Output\n"]

    lines.append("### 💡 Response")
    lines.append(f"```\n{truncate_text(response, 1000)}\n```")

    return "\n".join(lines)


# =============================================================================
# THEORY-OF-MIND FUNCTIONS
# =============================================================================


def format_user_representation_input(
    chat_history: str,
    user_representation: str = "None",
    tom_inference: str = "None",
    facts: list[str] | None = None,
) -> str:
    """Format input for user representation generation."""
    lines = ["# 👤 User Representation Input\n"]

    lines.append("### 💬 Chat History")
    lines.append(f"```\n{truncate_text(chat_history, 600)}\n```")

    lines.append("\n### 🧠 Current Representation")
    lines.append(f"```\n{truncate_text(user_representation, 400)}\n```")

    lines.append("\n### 💡 ToM Inference")
    lines.append(f"```\n{truncate_text(tom_inference, 400)}\n```")

    if facts:
        lines.append(f"\n{format_list_as_markdown(facts, '📚 Long-term Facts')}")

    return "\n".join(lines)


def format_user_representation_output(representation: str) -> str:
    """Format output for user representation generation."""
    lines = ["# 👤 User Representation Output\n"]

    lines.append("### 🎭 Updated Representation")
    lines.append(f"```\n{truncate_text(representation, 1000)}\n```")

    return "\n".join(lines)


def format_extract_facts_input(chat_history: str) -> str:
    """Format input for fact extraction."""
    lines = ["# 📚 Fact Extraction Input\n"]

    lines.append("### 💬 Chat History")
    lines.append(f"```\n{truncate_text(chat_history, 800)}\n```")

    return "\n".join(lines)


def format_extract_facts_output(facts: list[str]) -> str:
    """Format output for fact extraction."""
    lines = ["# 📚 Fact Extraction Output\n"]

    lines.append(format_list_as_markdown(facts, "🧠 Extracted Facts"))

    return "\n".join(lines)


# =============================================================================
# LAB/EVAL FUNCTIONS
# =============================================================================


def format_ingest_turn_input(
    message_content: str,
    turn_metadata: dict[str, Any] | None = None,
    conversation_history: str | None = None,
) -> str:
    """Format input for dataset ingestion turn processing.
    New optional parameter *conversation_history* allows including the
    chat history leading up to the current turn so that Langfuse traces
    contain richer context for debugging and analysis.
    """
    lines = ["# 📥 Ingest Turn Input\n"]

    if turn_metadata:
        lines.append(format_dict_as_markdown(turn_metadata, "📋 Turn Metadata"))
        lines.append("")

    # Conversation history (optional)
    if conversation_history:
        lines.append("### 🔄 Conversation History")
        lines.append(f"```\n{truncate_text(conversation_history, 1000)}\n```")
        lines.append("")

    lines.append("### 💬 Message Content")
    lines.append(f"```\n{truncate_text(message_content, 600)}\n```")

    return "\n".join(lines)


def format_ingest_turn_output(
    processing_result: Any, extracted_data: dict[str, Any] | None = None
) -> str:
    """Format output for dataset ingestion turn processing."""
    lines = ["# 📥 Ingest Turn Output\n"]

    lines.append("### ✅ Processing Result")
    lines.append(f"```\n{str(processing_result)}\n```")

    if extracted_data:
        lines.append(
            f"\n{format_dict_as_markdown(extracted_data, '📊 Extracted Data')}"
        )

    return "\n".join(lines)
