import logging
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.agent import prompts, tools
from src.config import settings
from src.models import Message
from src.schemas import ResolvedConfiguration
from src.utils.clients import honcho_llm_call

logger = logging.getLogger(__name__)


class ToolCall(BaseModel):
    tool_name: Literal[
        "create_observations",
        "update_peer_card",
        "get_recent_history",
        "search_memory",
    ]
    arguments: dict[str, Any]


class NextStep(BaseModel):
    thought: str = Field(description="Reasoning for the next step")
    tool_calls: list[ToolCall] = Field(default_factory=list)
    is_done: bool = Field(description="Whether the agent has completed its task")


class Agent:
    def __init__(
        self,
        db: AsyncSession,
        workspace_name: str,
        session_name: str,
        configuration: schemas.ResolvedConfiguration,
        observer: str,
        observed: str,
        observed_peer_card: list[str] | None = None,
    ):
        self.db: AsyncSession = db
        self.workspace_name: str = workspace_name
        self.session_name: str = session_name
        self.configuration: ResolvedConfiguration = configuration
        self.observer: str = observer
        self.observed: str = observed
        self.observed_peer_card: list[str] | None = observed_peer_card
        self.messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": prompts.agent_system_prompt(
                    observer, observed, observed_peer_card
                ),
            }
        ]
        self._current_messages: list[Message] = []

    async def run_loop(self, messages: list[models.Message]) -> None:
        """
        Run the agent loop for a batch of messages.

        Args:
            messages: List of messages to process in this agent run
        """
        if not messages:
            logger.warning("run_loop called with empty message list")
            return

        logger.info(
            f"Starting agent loop for {len(messages)} messages (observer={self.observer}, observed={self.observed})"
        )

        # Add all new messages to context at once
        messages_summary: list[Any] = []
        for msg in messages:
            messages_summary.append(
                f"[{msg.created_at}] {msg.peer_name}: {msg.content}"
            )

        self.messages.append(
            {
                "role": "user",
                "content": "New messages to process:\n" + "\n".join(messages_summary),  # noqa: ISC003
            }
        )

        # Store messages for tool execution
        self._current_messages = messages

        max_steps = 10
        step = 0

        while step < max_steps:
            logger.debug(f"Agent step {step}")

            # Call LLM to get next step
            try:
                # Convert messages to prompt string
                prompt = "\n\n".join(
                    f"{msg['role'].upper()}: {msg['content']}" for msg in self.messages
                )

                response = await honcho_llm_call(
                    llm_settings=settings.DIALECTIC,
                    prompt=prompt,
                    max_tokens=32768,
                    response_model=NextStep,
                )
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                break

            # Extract the NextStep from the response
            next_step: NextStep = response.content

            logger.info(f"Agent thought: {next_step.thought}")

            # Execute tool calls first (before checking is_done)
            tool_results: list[Any] = []
            for tool_call in next_step.tool_calls:
                result = await self._execute_tool(tool_call)
                tool_results.append(f"{tool_call.tool_name}: {result}")

            # Add result to history
            if tool_results:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": "Tool results:\n" + "\n".join(tool_results),  # noqa: ISC003
                    }
                )

            # Check if done AFTER executing tools
            if next_step.is_done:
                logger.info("Agent completed processing batch.")
                break

            step += 1

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """
        Execute a single tool call. All context (workspace, session, observer, observed)
        is provided by the agent instance from the queue item.

        Returns:
            String result describing what was done
        """
        logger.info(f"Executing tool: {tool_call.tool_name}")
        args = tool_call.arguments

        try:
            if tool_call.tool_name == "create_observations":
                observations = args.get("observations", [])

                if not observations:
                    return "ERROR: observations list is empty"

                # Validate all observations have required fields and valid levels
                for i, obs in enumerate(observations):
                    if "content" not in obs:
                        return f"ERROR: observation {i} missing 'content' field"
                    if "level" not in obs:
                        return f"ERROR: observation {i} missing 'level' field"
                    if obs["level"] not in ["explicit", "deductive"]:
                        return f"ERROR: observation {i} has invalid level '{obs['level']}', must be 'explicit' or 'deductive'"

                # Extract all message IDs from current batch
                message_ids = [msg.id for msg in self._current_messages]
                message_created_at = str(self._current_messages[-1].created_at)

                await tools.create_observations(
                    self.db,
                    observations=observations,
                    observer=self.observer,
                    observed=self.observed,
                    session_name=self.session_name,
                    workspace_name=self.workspace_name,
                    message_ids=message_ids,
                    message_created_at=message_created_at,
                )

                explicit_count = sum(
                    1 for o in observations if o.get("level") == "explicit"
                )
                deductive_count = sum(
                    1 for o in observations if o.get("level") == "deductive"
                )
                return f"Created {len(observations)} observations for {self.observed} by {self.observer} ({explicit_count} explicit, {deductive_count} deductive)"

            elif tool_call.tool_name == "update_peer_card":
                await tools.update_peer_card(
                    self.db,
                    workspace_name=self.workspace_name,
                    observer=self.observer,
                    observed=self.observed,
                    content=args["content"],
                )
                return f"Updated peer card for {self.observed} by {self.observer}"

            elif tool_call.tool_name == "get_recent_history":
                history: list[models.Message] = await tools.get_recent_history(
                    self.db,
                    workspace_name=self.workspace_name,
                    session_name=self.session_name,
                    token_limit=args.get("token_limit", 8192),
                )
                history_text = "\n".join(
                    [f"{m.peer_name}: {m.content}" for m in history]
                )
                self.messages.append(
                    {
                        "role": "system",
                        "content": f"Recent history:\n{history_text}",
                    }
                )
                return f"Retrieved {len(history)} messages from history"

            elif tool_call.tool_name == "search_memory":
                results = await tools.search_memory(
                    self.db,
                    workspace_name=self.workspace_name,
                    observer=self.observer,
                    observed=self.observed,
                    query=args["query"],
                )
                results_text = "\n".join([f"- {d.content}" for d in results])
                self.messages.append(
                    {
                        "role": "system",
                        "content": f"Search results for '{args['query']}':\n{results_text}",
                    }
                )
                return f"Found {len(results)} results for query: {args['query']}"

            return "Unknown tool result"

        except Exception as e:
            error_msg = f"Tool {tool_call.tool_name} failed: {e}"
            logger.error(error_msg)
            return error_msg
