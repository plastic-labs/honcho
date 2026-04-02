# Live LLM Tests

These tests call real provider APIs and are disabled by default.

Run them with:

```bash
uv run pytest tests/live_llm -n 0 --live-llm --no-header -q
```

Required API key env vars:

- `LLM_ANTHROPIC_API_KEY`
- `LLM_OPENAI_API_KEY`
- `LLM_GEMINI_API_KEY`

Model-family env vars:

- `LIVE_LLM_ANTHROPIC_45_PLUS_MODELS`
- `LIVE_LLM_OPENAI_GPT4_MODELS`
- `LIVE_LLM_OPENAI_GPT5_MODELS`
- `LIVE_LLM_GEMINI_25_MODELS`
- `LIVE_LLM_GEMINI_30_MODELS`
- `LIVE_LLM_GEMINI_31_MODELS`

Each model env var accepts a comma-separated list of bare model ids or provider-qualified ids.

Examples:

```bash
export LIVE_LLM_ANTHROPIC_45_PLUS_MODELS="claude-sonnet-4-5,claude-sonnet-4-6"
export LIVE_LLM_OPENAI_GPT4_MODELS="gpt-4.1"
export LIVE_LLM_OPENAI_GPT5_MODELS="gpt-5"
export LIVE_LLM_GEMINI_25_MODELS="gemini-2.5-flash,gemini-2.5-pro"
export LIVE_LLM_GEMINI_30_MODELS="gemini-3-flash-preview"
export LIVE_LLM_GEMINI_31_MODELS="gemini-3.1-pro-preview"
```

Coverage by provider:

- Anthropic: structured output path, prompt caching metrics, thinking blocks, multi-turn tool replay
- OpenAI GPT-4 class: structured outputs, prompt caching
- OpenAI GPT-5 class: structured outputs, prompt caching, `reasoning_effort`
- Gemini 2.5/3.0 classes: structured outputs, cached-content reuse, thought signatures, multi-turn tool replay
- Gemini 3.1 class: thinking and tool replay coverage by default; structured-output/caching coverage should only be added once Google documents support for that path
