# Current Journal

**Date:** 2026-05-05
**Role:** Leader

## Current Status
- Successfully archived `honcho-langfuse-generation-traces`.
- Tracing infrastructure is fully operational. Both API and background workers (Deriver) are logging `GENERATION` traces correctly.
- Custom models (e.g. `lmstudio` `qwen3.5`) are actively reporting explicit `usage_details` (tokens) to Langfuse.
- Main specifications for `observability-langfuse` have been established and updated in the project repository.

## Active Focus
- Standing by for new OpenSpec initiatives or orchestration tasks.

## Critical Context
- Langfuse configuration relies on explicitly passing `as_type="generation"` to the `@conditional_observe` decorator in `honcho_llm_call`.
- Explicit `usage_details` must be provided to Langfuse active generation object to track tokens for custom models reliably.
