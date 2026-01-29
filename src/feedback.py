"""
Developer Feedback Channel for configuring Honcho's agent behavior.

This module provides a natural language interface for developers to configure
workspace agent settings through conversation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, cast

from sqlalchemy.ext.asyncio import AsyncSession

from src import crud
from src.config import settings
from src.schemas import (
    ConfigChange,
    FeedbackRequest,
    FeedbackResponse,
    IntrospectionReport,
    WorkspaceAgentConfig,
)
from src.utils.clients import honcho_llm_call

logger = logging.getLogger(__name__)


INTERVIEW_QUESTIONS = """Before I can help configure Honcho for your workspace, I'd like to understand your application better. Please tell me:

1. **What type of application are you building?** (e.g., journaling app, customer support bot, educational tutor, personal assistant, etc.)

2. **What aspects of your users do you want Honcho to focus on?** (e.g., emotions, preferences, technical skills, learning progress, goals, habits)

3. **How should the Dialectic API respond to questions about users?** (e.g., detailed analysis, brief summaries, specific focus areas)

4. **Are there any topics or patterns you want Honcho to explicitly ignore or avoid?**

Feel free to answer any or all of these questions, and I'll help configure your workspace accordingly."""


def is_simple_greeting(message: str) -> bool:
    """Check if a message is a simple greeting or question that should trigger interview mode."""
    message_lower = message.lower().strip()

    # Check length - short messages are more likely greetings
    if len(message) > 100:
        return False

    # Common greetings and simple starts
    greeting_patterns = [
        r"^h(i|ello|ey)\b",
        r"^good (morning|afternoon|evening)",
        r"^what('s| is) up",
        r"^how('s| are) (it going|you|things)",
        r"^yo\b",
        r"^sup\b",
        r"^greetings",
        r"^howdy",
        r"^help$",
        r"^help me",
        r"^how do i",
        r"^what can you",
        r"^configure",
        r"^setup",
        r"^start",
        r"^begin",
        r"^get started",
    ]

    return any(re.match(pattern, message_lower) for pattern in greeting_patterns)


def config_is_empty(config: WorkspaceAgentConfig) -> bool:
    """Check if config has no custom rules set."""
    return not config.deriver_rules.strip() and not config.dialectic_rules.strip()


def build_feedback_prompt(
    message: str,
    current_config: WorkspaceAgentConfig,
    introspection_report: IntrospectionReport | None = None,
) -> str:
    """
    Build a prompt for the LLM to process developer feedback.

    Args:
        message: The developer's feedback message
        current_config: Current workspace agent configuration
        introspection_report: Optional introspection report for context

    Returns:
        A formatted prompt string for the LLM
    """
    introspection_section = ""
    if introspection_report:
        introspection_section = f"""
## Recent Introspection Report

**Performance Summary:** {introspection_report.performance_summary}

**Identified Issues:**
{chr(10).join(f"- {issue}" for issue in introspection_report.identified_issues) if introspection_report.identified_issues else "(none)"}

**Suggestions:**
{chr(10).join(f"- [{s.target}] {s.rationale} (confidence: {s.confidence})" for s in introspection_report.suggestions) if introspection_report.suggestions else "(none)"}

"""

    return f"""You are a configuration assistant for Honcho, an AI memory infrastructure system.

A developer is interacting with the feedback channel to configure their workspace's agent behavior.

## Current Configuration

**Deriver Rules** (guides memory extraction):
```
{current_config.deriver_rules or "(empty - using defaults)"}
```

**Dialectic Rules** (guides question answering):
```
{current_config.dialectic_rules or "(empty - using defaults)"}
```
{introspection_section}
## Developer Message

{message}

## Your Task

1. **Understand the intent**: Is the developer asking a question, providing configuration instructions, or just chatting?

2. **Determine configuration changes**: Based on the message, decide if any configuration changes should be made:
   - `deriver_rules`: Controls what the memory extraction agent focuses on
   - `dialectic_rules`: Controls how the question-answering agent responds

3. **Be incremental**: When adding rules, PRESERVE existing rules unless the developer explicitly asks to replace them. Append new rules to existing ones.

4. **Respond helpfully**: Provide a clear, friendly response explaining what you understood and what changes (if any) you made.

## Response Format

Respond with a JSON object:
```json
{{
    "message": "Your response to the developer",
    "understood_intent": "Brief description of what you understood the developer wants",
    "changes": [
        {{
            "field": "deriver_rules" | "dialectic_rules",
            "new_value": "The complete new value for this field (including preserved old rules if applicable)"
        }}
    ]
}}
```

If no changes are needed (e.g., the developer is asking a question), return an empty `changes` array.

Important:
- Keep rules concise and actionable
- Each rule should be on its own line for clarity
- When adding to existing rules, put a newline between old and new rules
- Be helpful and explain what the rules will do"""


async def process_feedback(
    db: AsyncSession,
    workspace_name: str,
    request: FeedbackRequest,
    introspection_report: IntrospectionReport | None = None,
) -> FeedbackResponse:
    """
    Process developer feedback and update workspace configuration.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        request: The feedback request
        introspection_report: Optional introspection report for context

    Returns:
        FeedbackResponse with the result
    """
    # Get current config
    current_config = await crud.get_workspace_agent_config(db, workspace_name)

    # Check for interview mode: empty config + simple greeting
    if config_is_empty(current_config) and is_simple_greeting(request.message):
        logger.info(
            f"Feedback channel: Interview mode triggered for workspace {workspace_name}"
        )
        return FeedbackResponse(
            message=INTERVIEW_QUESTIONS,
            understood_intent="First-time setup - gathering information about the application",
            changes_made=[],
            current_config=current_config,
        )

    # Build prompt and call LLM
    prompt = build_feedback_prompt(
        message=request.message,
        current_config=current_config,
        introspection_report=introspection_report,
    )

    try:
        llm_response = await honcho_llm_call(
            llm_settings=settings.DREAM,
            prompt=prompt,
            max_tokens=4096,
            track_name="feedback_channel",
            json_mode=True,
            temperature=0.3,
        )

        # Parse response
        response_text = llm_response.content
        try:
            response_data: dict[str, object] = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse feedback LLM response: {e}")
            return FeedbackResponse(
                message="I had trouble processing your request. Could you try rephrasing?",
                understood_intent="Error parsing response",
                changes_made=[],
                current_config=current_config,
            )

        # Process changes
        changes_made: list[ConfigChange] = []
        raw_changes = response_data.get("changes", [])

        if isinstance(raw_changes, list):
            for change_item in cast(list[dict[str, Any]], raw_changes):
                change_dict: dict[str, object] = change_item
                field = str(change_dict.get("field", ""))
                new_value = str(change_dict.get("new_value", ""))

                if field not in ("deriver_rules", "dialectic_rules"):
                    continue

                # Get previous value
                previous_value = (
                    current_config.deriver_rules
                    if field == "deriver_rules"
                    else current_config.dialectic_rules
                )

                # Skip if no actual change
                if previous_value == new_value:
                    continue

                changes_made.append(
                    ConfigChange(
                        field=field,  # type: ignore[arg-type]
                        previous_value=previous_value,
                        new_value=new_value,
                    )
                )

        # Apply changes if any
        if changes_made:
            new_config = WorkspaceAgentConfig(
                deriver_rules=current_config.deriver_rules,
                dialectic_rules=current_config.dialectic_rules,
            )

            for change in changes_made:
                if change.field == "deriver_rules":
                    new_config.deriver_rules = change.new_value
                elif change.field == "dialectic_rules":
                    new_config.dialectic_rules = change.new_value

            await crud.set_workspace_agent_config(db, workspace_name, new_config)
            current_config = new_config

            logger.info(
                f"Feedback channel: Applied {len(changes_made)} changes to workspace {workspace_name}"
            )

        message = response_data.get("message", "Configuration updated.")
        understood_intent = response_data.get("understood_intent", "Processed feedback")

        return FeedbackResponse(
            message=str(message),
            understood_intent=str(understood_intent),
            changes_made=changes_made,
            current_config=current_config,
        )

    except Exception as e:
        logger.error(f"Feedback channel LLM call failed: {e}")
        return FeedbackResponse(
            message="I encountered an error processing your feedback. Please try again.",
            understood_intent="Error during processing",
            changes_made=[],
            current_config=current_config,
        )
