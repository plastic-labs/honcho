# Lesson: Langfuse Custom Model Token Tracking

**Date:** 2026-05-05
**Topic:** Langfuse UI token observability for non-OpenAI/Anthropic models
**Role:** Leader

## The Issue
When integrating Langfuse for LLM traces, models like `gpt-4o-mini` or `claude-3-haiku` automatically displayed token usage in the Langfuse dashboard, while custom models (e.g., `qwen/qwen3.5-9b` hosted via LMStudio) showed empty token fields despite successful completions.

## Root Cause
Langfuse features an automatic server-side tokenizer that calculates token usage if it recognizes the `model` identifier. If the model is not recognized (as is the case with custom or local models), Langfuse silently fails to compute tokens unless explicit `usage_details` are pushed by the application SDK.

## The Fix / Lesson
To ensure consistent token logging across **all** models (including custom and local deployments), the SDK must explicitly capture `input_tokens` and `output_tokens` from the underlying client response and report them via the Langfuse generation active observer. 

Example Implementation:
```python
if usage_data:
    langfuse.get_client().update_current_generation(
        usage_details={
            "input": input_tokens,
            "output": output_tokens
        }
    )
```
Always explicitly pass usage to Langfuse to guarantee observability parity regardless of backend.
