from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.utils.representation import PromptRepresentation

from ..backend import CompletionResult
from ..structured_output import repair_response_model_json
from .openai import OpenAIBackend, extract_openai_cache_tokens


class OpenAICompatibleBackend(OpenAIBackend):
    """Backend for explicit OpenAI-compatible transports like vLLM/OpenRouter."""

    def __init__(self, client: Any, provider_name: str = "openai_compatible") -> None:
        super().__init__(client)
        self._provider_name = provider_name

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        thinking_budget_tokens: int | None = None,
        thinking_effort: str | None = None,
        max_output_tokens: int | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> CompletionResult:
        del api_key, api_base
        if thinking_budget_tokens is not None:
            raise ValueError(
                "OpenAI-compatible backend does not support thinking_budget_tokens"
            )

        if (
            self._provider_name in {"vllm", "hosted_vllm"}
            and response_format is not None
        ):
            if response_format is not PromptRepresentation:
                raise NotImplementedError(
                    "vLLM structured output currently supports only PromptRepresentation"
                )
            params = self._build_params(
                model=model,
                messages=messages,
                max_tokens=max_output_tokens or max_tokens,
                temperature=temperature,
                stop=stop,
                tools=tools,
                tool_choice=tool_choice,
                thinking_effort=thinking_effort,
                extra_params=extra_params,
            )
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": response_format.model_json_schema(),
                },
            }
            response = await self._client.chat.completions.create(**params)
            raw_content = response.choices[0].message.content or ""
            repaired = repair_response_model_json(
                raw_content,
                response_format,
                self._strip_prefix(model),
            )
            cache_creation, cache_read = extract_openai_cache_tokens(response.usage)
            return CompletionResult(
                content=repaired,
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
                finish_reason=response.choices[0].finish_reason or "stop",
                raw_response=response,
            )

        processed_messages = messages
        if self._provider_name in {"custom", "openrouter"}:
            processed_messages = []
            for message in messages:
                if message.get("role") == "system" and isinstance(
                    message.get("content"),
                    str,
                ):
                    processed_messages.append(
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": message["content"],
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        }
                    )
                else:
                    processed_messages.append(message)

        return await super().complete(
            model=model,
            messages=processed_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking_budget_tokens=thinking_budget_tokens,
            thinking_effort=thinking_effort,
            max_output_tokens=max_output_tokens,
            extra_params=extra_params,
        )
