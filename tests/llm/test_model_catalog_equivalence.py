"""A config that names catalog entries loads exactly like one that spells them out.

Each pair boots both configs in fresh interpreters (see
`model_config_snapshot.py`) and compares everything the probe records: every
slot's configured settings, its runtime `ModelConfig`, the config its final
retry attempt uses, and the capability rows of its primary and fallback.

Capabilities of a Claude model reached over the `openai` transport are the
one place a catalog entry deliberately differs from an inline slot, so these
pairs keep Claude on the `anthropic` transport. That difference is covered in
`test_model_catalog.py`.
"""

from typing import Any

from tests.llm.model_config_snapshot import differences, run_all

_ANTHROPIC_HAIKU_ENV = {
    "MODELS__haiku__TRANSPORT": "anthropic",
    "MODELS__haiku__MODEL": "claude-haiku-4-5",
    "MODELS__haiku__THINKING_BUDGET_TOKENS": "2048",
    "MODELS__haiku__OVERRIDES__BASE_URL": "https://anthropic-proxy.example",
    "MODELS__haiku__OVERRIDES__API_KEY": "sk-anthropic",
}

_TIERED_CATALOG_TOML = """
[models.light]
transport = "openai"
model = "google/gemini-2.5-flash-lite"
structured_output_mode = "json_object"
overrides = { base_url = "https://openrouter.ai/api/v1", api_key = "sk-or" }

[models.medium]
transport = "openai"
model = "openai/gpt-5"
thinking_effort = "low"
overrides = { base_url = "https://openrouter.ai/api/v1", api_key = "sk-or" }

[models.heavy]
transport = "anthropic"
model = "claude-opus-4-6"
thinking_budget_tokens = 4096
max_output_tokens = 16000

[models.nano]
transport = "openai"
model = "gpt-5-nano"
thinking_effort = "minimal"
max_output_tokens = 8000
"""

_LIGHT = """transport = "openai"
model = "google/gemini-2.5-flash-lite"
structured_output_mode = "json_object"
overrides = { base_url = "https://openrouter.ai/api/v1", api_key = "sk-or" }"""
_MEDIUM = """transport = "openai"
model = "openai/gpt-5"
thinking_effort = "low"
overrides = { base_url = "https://openrouter.ai/api/v1", api_key = "sk-or" }"""
_HEAVY = """transport = "anthropic"
model = "claude-opus-4-6"
thinking_budget_tokens = 4096
max_output_tokens = 16000"""
_NANO = """transport = "openai"
model = "gpt-5-nano"
thinking_effort = "minimal"
max_output_tokens = 8000"""

# name -> (spelled out, through the catalog)
PAIRS: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "env_one_slot_with_slot_override": (
        {
            "env": {
                "DERIVER_MODEL_CONFIG__TRANSPORT": "anthropic",
                "DERIVER_MODEL_CONFIG__MODEL": "claude-haiku-4-5",
                "DERIVER_MODEL_CONFIG__THINKING_BUDGET_TOKENS": "2048",
                "DERIVER_MODEL_CONFIG__OVERRIDES__BASE_URL": (
                    "https://anthropic-proxy.example"
                ),
                "DERIVER_MODEL_CONFIG__OVERRIDES__API_KEY": "sk-anthropic",
                "DERIVER_MODEL_CONFIG__MAX_OUTPUT_TOKENS": "5000",
            }
        },
        {
            "env": {
                **_ANTHROPIC_HAIKU_ENV,
                "DERIVER_MODEL_CONFIG__MODEL_REF": "haiku",
                "DERIVER_MODEL_CONFIG__MAX_OUTPUT_TOKENS": "5000",
            }
        },
    ),
    "env_slot_field_beats_entry_field": (
        {
            "env": {
                "SUMMARY_MODEL_CONFIG__TRANSPORT": "anthropic",
                "SUMMARY_MODEL_CONFIG__MODEL": "claude-haiku-4-5",
                "SUMMARY_MODEL_CONFIG__THINKING_BUDGET_TOKENS": "1024",
                "SUMMARY_MODEL_CONFIG__OVERRIDES__BASE_URL": (
                    "https://anthropic-proxy.example"
                ),
                "SUMMARY_MODEL_CONFIG__OVERRIDES__API_KEY": "sk-slot",
            }
        },
        {
            "env": {
                **_ANTHROPIC_HAIKU_ENV,
                "SUMMARY_MODEL_CONFIG__MODEL_REF": "haiku",
                "SUMMARY_MODEL_CONFIG__THINKING_BUDGET_TOKENS": "1024",
                "SUMMARY_MODEL_CONFIG__OVERRIDES__API_KEY": "sk-slot",
            }
        },
    ),
    # The spelled-out side uses THINKING_EFFORT: an inline slot drops an env
    # REASONING_EFFORT, because default filling inserts thinking_effort=None
    # and that name is read before the alias (recorded as-is in the snapshot's
    # env_reasoning_effort_alias case).
    "env_reasoning_effort_alias_beats_entry": (
        {
            "env": {
                "DERIVER_MODEL_CONFIG__MODEL": "gpt-5",
                "DERIVER_MODEL_CONFIG__THINKING_EFFORT": "high",
            }
        },
        {
            "env": {
                "MODELS__gpt5__MODEL": "gpt-5",
                "MODELS__gpt5__TRANSPORT": "openai",
                "MODELS__gpt5__THINKING_EFFORT": "low",
                "DERIVER_MODEL_CONFIG__MODEL_REF": "gpt5",
                "DERIVER_MODEL_CONFIG__REASONING_EFFORT": "high",
            }
        },
    ),
    "env_cross_family_fallback": (
        {
            "env": {
                "SUMMARY_MODEL_CONFIG__TRANSPORT": "anthropic",
                "SUMMARY_MODEL_CONFIG__MODEL": "claude-haiku-5-5",
                "SUMMARY_MODEL_CONFIG__THINKING_EFFORT": "low",
                "SUMMARY_MODEL_CONFIG__MAX_OUTPUT_TOKENS": "6000",
                "SUMMARY_MODEL_CONFIG__FALLBACK__TRANSPORT": "openai",
                "SUMMARY_MODEL_CONFIG__FALLBACK__MODEL": "gpt-5-nano",
                "SUMMARY_MODEL_CONFIG__FALLBACK__THINKING_EFFORT": "minimal",
                "SUMMARY_MODEL_CONFIG__FALLBACK__STRUCTURED_OUTPUT_MODE": (
                    "json_object"
                ),
                "SUMMARY_MODEL_CONFIG__FALLBACK__MAX_OUTPUT_TOKENS": "8000",
                "SUMMARY_MODEL_CONFIG__FALLBACK__OVERRIDES__BASE_URL": (
                    "https://fallback.example/v1"
                ),
            }
        },
        {
            "env": {
                "MODELS__haiku55__TRANSPORT": "anthropic",
                "MODELS__haiku55__MODEL": "claude-haiku-5-5",
                "MODELS__haiku55__THINKING_EFFORT": "low",
                "MODELS__nano__TRANSPORT": "openai",
                "MODELS__nano__MODEL": "gpt-5-nano",
                "MODELS__nano__THINKING_EFFORT": "minimal",
                "MODELS__nano__STRUCTURED_OUTPUT_MODE": "json_object",
                "MODELS__nano__MAX_OUTPUT_TOKENS": "8000",
                "MODELS__nano__OVERRIDES__BASE_URL": "https://fallback.example/v1",
                "SUMMARY_MODEL_CONFIG__MODEL_REF": "haiku55",
                "SUMMARY_MODEL_CONFIG__MAX_OUTPUT_TOKENS": "6000",
                "SUMMARY_MODEL_CONFIG__FALLBACK__MODEL_REF": "nano",
            }
        },
    ),
    "env_dialectic_levels": (
        {
            "env": {
                "DIALECTIC_LEVELS__high__MODEL_CONFIG__TRANSPORT": "anthropic",
                "DIALECTIC_LEVELS__high__MODEL_CONFIG__MODEL": "claude-haiku-4-5",
                "DIALECTIC_LEVELS__high__MODEL_CONFIG__THINKING_BUDGET_TOKENS": (
                    "4096"
                ),
                "DIALECTIC_LEVELS__high__MODEL_CONFIG__OVERRIDES__BASE_URL": (
                    "https://anthropic-proxy.example"
                ),
                "DIALECTIC_LEVELS__high__MODEL_CONFIG__OVERRIDES__API_KEY": (
                    "sk-anthropic"
                ),
                "DIALECTIC_LEVELS__high__MAX_OUTPUT_TOKENS": "9000",
                "DIALECTIC_LEVELS__low__MODEL_CONFIG__TRANSPORT": "anthropic",
                "DIALECTIC_LEVELS__low__MODEL_CONFIG__MODEL": "claude-haiku-4-5",
                "DIALECTIC_LEVELS__low__MODEL_CONFIG__THINKING_BUDGET_TOKENS": ("2048"),
                "DIALECTIC_LEVELS__low__MODEL_CONFIG__OVERRIDES__BASE_URL": (
                    "https://anthropic-proxy.example"
                ),
                "DIALECTIC_LEVELS__low__MODEL_CONFIG__OVERRIDES__API_KEY": (
                    "sk-anthropic"
                ),
                "DIALECTIC_LEVELS__low__TOOL_CHOICE": "any",
            }
        },
        {
            "env": {
                **_ANTHROPIC_HAIKU_ENV,
                "DIALECTIC_LEVELS__high__MODEL_CONFIG__MODEL_REF": "haiku",
                "DIALECTIC_LEVELS__high__MODEL_CONFIG__THINKING_BUDGET_TOKENS": (
                    "4096"
                ),
                "DIALECTIC_LEVELS__high__MAX_OUTPUT_TOKENS": "9000",
                "DIALECTIC_LEVELS__low__MODEL_CONFIG__MODEL_REF": "haiku",
                "DIALECTIC_LEVELS__low__TOOL_CHOICE": "any",
            }
        },
    ),
    "env_provider_params_merge": (
        {
            "env": {
                "DREAM_DEDUCTION_MODEL_CONFIG__MODEL": "deduction-alias",
                "DREAM_DEDUCTION_MODEL_CONFIG__OVERRIDES__BASE_URL": (
                    "https://proxy.example/v1"
                ),
                "DREAM_DEDUCTION_MODEL_CONFIG__OVERRIDES__PROVIDER_PARAMS": (
                    '{"extra_body": {"slot": true}, "timeout": 90,'
                    + ' "extra_headers": {"x-team": "a"}}'
                ),
            }
        },
        {
            "env": {
                "MODELS__proxy__MODEL": "deduction-alias",
                "MODELS__proxy__TRANSPORT": "openai",
                "MODELS__proxy__OVERRIDES__BASE_URL": "https://proxy.example/v1",
                "MODELS__proxy__OVERRIDES__PROVIDER_PARAMS": (
                    '{"extra_body": {"entry": true}, "timeout": 30,'
                    + ' "extra_headers": {"x-team": "a"}}'
                ),
                "DREAM_DEDUCTION_MODEL_CONFIG__MODEL_REF": "proxy",
                # extra_body is replaced whole; other keys merge.
                "DREAM_DEDUCTION_MODEL_CONFIG__OVERRIDES__PROVIDER_PARAMS": (
                    '{"extra_body": {"slot": true}, "timeout": 90}'
                ),
            }
        },
    ),
    "env_entry_from_json_and_shorthand_model": (
        {
            "env": {
                "DREAM_INDUCTION_MODEL_CONFIG__TRANSPORT": "anthropic",
                "DREAM_INDUCTION_MODEL_CONFIG__MODEL": "claude-opus-4-6",
                "DREAM_INDUCTION_MODEL_CONFIG__THINKING_BUDGET_TOKENS": "4096",
            }
        },
        {
            "env": {
                "MODELS": (
                    '{"opus": {"model": "anthropic/claude-opus-4-6",'
                    + ' "thinking_budget_tokens": 4096}}'
                ),
                "DREAM_INDUCTION_MODEL_CONFIG__MODEL_REF": "Opus",
            }
        },
    ),
    "env_blank_model_ref_is_unset": (
        {"env": {"DERIVER_MODEL_CONFIG__MODEL": "gpt-4.1"}},
        {
            "env": {
                "DERIVER_MODEL_CONFIG__MODEL_REF": "",
                "DERIVER_MODEL_CONFIG__MODEL": "gpt-4.1",
            }
        },
    ),
    "toml_tiered_all_slots": (
        {
            "toml": f"""
[deriver.model_config]
{_LIGHT}
[deriver.model_config.fallback]
{_NANO}

[summary.model_config]
{_LIGHT}
max_output_tokens = 9000

[dream.deduction_model_config]
{_HEAVY}
[dream.deduction_model_config.fallback]
{_MEDIUM}

[dream.induction_model_config]
{_HEAVY}

[dialectic.levels.minimal.model_config]
{_LIGHT}
[dialectic.levels.low.model_config]
{_LIGHT}
[dialectic.levels.medium.model_config]
{_MEDIUM}
[dialectic.levels.high.model_config]
{_MEDIUM}
[dialectic.levels.high.model_config.fallback]
{_NANO}
[dialectic.levels.max.model_config]
{_HEAVY}
[dialectic.levels.max]
max_output_tokens = 20000
"""
        },
        {
            "toml": _TIERED_CATALOG_TOML
            + """
[deriver.model_config]
model_ref = "light"
[deriver.model_config.fallback]
model_ref = "nano"

[summary.model_config]
model_ref = "light"
max_output_tokens = 9000

[dream.deduction_model_config]
model_ref = "heavy"
[dream.deduction_model_config.fallback]
model_ref = "medium"

[dream.induction_model_config]
model_ref = "heavy"

[dialectic.levels.minimal.model_config]
model_ref = "light"
[dialectic.levels.low.model_config]
model_ref = "light"
[dialectic.levels.medium.model_config]
model_ref = "medium"
[dialectic.levels.high.model_config]
model_ref = "medium"
[dialectic.levels.high.model_config.fallback]
model_ref = "nano"
[dialectic.levels.max.model_config]
model_ref = "heavy"
[dialectic.levels.max]
max_output_tokens = 20000
"""
        },
    ),
    "catalog_in_toml_refs_in_env": (
        {
            "env": {
                "DERIVER_MODEL_CONFIG__TRANSPORT": "openai",
                "DERIVER_MODEL_CONFIG__MODEL": "gpt-5-nano",
                "DERIVER_MODEL_CONFIG__THINKING_EFFORT": "minimal",
                "DERIVER_MODEL_CONFIG__MAX_OUTPUT_TOKENS": "8000",
                "DIALECTIC_LEVELS__max__MODEL_CONFIG__TRANSPORT": "anthropic",
                "DIALECTIC_LEVELS__max__MODEL_CONFIG__MODEL": "claude-opus-4-6",
                "DIALECTIC_LEVELS__max__MODEL_CONFIG__THINKING_BUDGET_TOKENS": ("4096"),
                "DIALECTIC_LEVELS__max__MODEL_CONFIG__MAX_OUTPUT_TOKENS": "16000",
            }
        },
        {
            "toml": _TIERED_CATALOG_TOML,
            "env": {
                "DERIVER_MODEL_CONFIG__MODEL_REF": "nano",
                "DIALECTIC_LEVELS__max__MODEL_CONFIG__MODEL_REF": "heavy",
            },
        },
    ),
    "ref_in_toml_slot_field_from_env": (
        {
            "env": {
                "SUMMARY_MODEL_CONFIG__TRANSPORT": "openai",
                "SUMMARY_MODEL_CONFIG__MODEL": "gpt-5-nano",
                "SUMMARY_MODEL_CONFIG__THINKING_EFFORT": "minimal",
                "SUMMARY_MODEL_CONFIG__MAX_OUTPUT_TOKENS": "12000",
            }
        },
        {
            "toml": _TIERED_CATALOG_TOML
            + """
[summary.model_config]
model_ref = "nano"
""",
            "env": {"SUMMARY_MODEL_CONFIG__MAX_OUTPUT_TOKENS": "12000"},
        },
    ),
    # A profile belongs to the transport and model, so a slot that spells out
    # the alias gets it as surely as one that names the entry.
    "declared_profile_with_and_without_ref": (
        {
            "toml": """
[models.summarizer]
transport = "openai"
model = "summarizer"
capabilities = { infer_from = "claude-haiku-5-5", forced_tool_choice = "rejected" }

[summary.model_config]
transport = "openai"
model = "summarizer"
"""
        },
        {
            "toml": """
[models.summarizer]
transport = "openai"
model = "summarizer"
capabilities = { infer_from = "claude-haiku-5-5", forced_tool_choice = "rejected" }

[summary.model_config]
model_ref = "summarizer"
"""
        },
    ),
    # A context window and a declared token field come along with the entry.
    "context_window_and_token_field_with_and_without_ref": (
        {
            "toml": """
[models.astra]
transport = "openai"
model = "astra"
context_window = 400000
capabilities = { infer_from = "gpt-6-astra" }

[summary.model_config]
transport = "openai"
model = "astra"
context_window = 400000
"""
        },
        {
            "toml": """
[models.astra]
transport = "openai"
model = "astra"
context_window = 400000
capabilities = { infer_from = "gpt-6-astra" }

[summary.model_config]
model_ref = "astra"
"""
        },
    ),
}


def test_named_entries_load_like_spelled_out_config() -> None:
    cases = {
        f"{name}:{side}": case
        for name, pair in PAIRS.items()
        for side, case in zip(("inline", "catalog"), pair, strict=True)
    }
    results = run_all(cases)

    problems: list[str] = []
    for name in sorted(PAIRS):
        inline, catalog = results[f"{name}:inline"], results[f"{name}:catalog"]
        if "load_error" in inline:
            problems.append(f"{name}: the spelled-out side failed to load")
        if "load_error" in catalog:
            problems.append(f"{name}: catalog side failed: {catalog['load_error']}")
        problems.extend(differences(name, inline, catalog))
    assert not problems, "\n".join(problems)


def test_pairs_exercise_what_they_claim() -> None:
    """Guard against pairs that match only because both sides are defaults."""
    cases = {
        "tiered": PAIRS["toml_tiered_all_slots"][1],
        "fallback": PAIRS["env_cross_family_fallback"][1],
        "profile": PAIRS["declared_profile_with_and_without_ref"][1],
        "window": PAIRS["context_window_and_token_field_with_and_without_ref"][1],
    }
    results = run_all(cases)

    tiered = results["tiered"]["slots"]
    assert tiered["dialectic.max"]["resolved"]["model"] == "claude-opus-4-6"
    assert tiered["dialectic.max"]["resolved"]["thinking_budget_tokens"] == 4096
    assert tiered["deriver"]["resolved"]["fallback"]["model"] == "gpt-5-nano"
    assert tiered["summary"]["resolved"]["base_url"] == "https://openrouter.ai/api/v1"
    assert tiered["summary"]["resolved"]["max_output_tokens"] == 9000

    final = results["fallback"]["slots"]["summary"]["final_attempt"]
    assert final["model"] == "gpt-5-nano"
    assert final["structured_output_mode"] == "json_object"
    assert final["thinking_effort"] == "minimal"
    assert final["max_output_tokens"] == 8000

    caps = results["profile"]["slots"]["summary"]["capabilities"]
    assert caps["thinking"] == "adaptive"
    assert caps["preserved_thinking"] is True
    assert caps["forced_tool_choice"] == "rejected"

    window = results["window"]["slots"]["summary"]
    assert window["configured"]["context_window"] == 400000
    assert window["capabilities"]["max_tokens_param"] == "max_completion_tokens"
