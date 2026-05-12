import os
import re
from pathlib import Path
from typing import Any, cast

import pytest

from src.config import (
    AppSettings,
    ConfiguredEmbeddingModelSettings,
    ConfiguredModelSettings,
    DialecticLevelSettings,
    DreamSettings,
    EmbeddingSettings,
    ModelConfig,
    ModelOverrideSettings,
    SummarySettings,
    VectorStoreSettings,
    load_toml_config,
    resolve_embedding_model_config,
    resolve_model_config,
)


def test_fallback_config_is_independent() -> None:
    """Fallback config has its own transport and reasoning params."""
    from src.config import ResolvedFallbackConfig

    config = ModelConfig(
        model="claude-haiku-4-5",
        transport="anthropic",
        thinking_budget_tokens=1024,
        fallback=ResolvedFallbackConfig(
            model="gpt-4.1-mini",
            transport="openai",
            base_url="https://example.com/v1",
        ),
    )
    assert config.fallback is not None
    assert config.fallback.transport == "openai"
    assert config.fallback.thinking_budget_tokens is None
    assert config.fallback.base_url == "https://example.com/v1"


def test_base_url_is_allowed_for_any_transport() -> None:
    config = ModelConfig(
        model="claude-haiku-4-5",
        transport="anthropic",
        base_url="https://anthropic-proxy.example/v1",
    )

    assert config.base_url == "https://anthropic-proxy.example/v1"


def test_anthropic_thinking_budget_has_minimum() -> None:
    with pytest.raises(ValueError, match="thinking_budget_tokens must be >= 1024"):
        ModelConfig(
            model="claude-haiku-4-5",
            transport="anthropic",
            thinking_budget_tokens=512,
        )


def test_reasoning_effort_alias_populates_generic_thinking_effort() -> None:
    config = ModelConfig.model_validate(
        {
            "model": "gpt-5",
            "transport": "openai",
            "reasoning_effort": "minimal",
        }
    )

    assert config.thinking_effort == "minimal"
    assert config.reasoning_effort == "minimal"


def test_for_model_overrides_model_and_transport() -> None:
    config = ModelConfig(
        model="claude-haiku-4-5",
        transport="anthropic",
    )

    updated = config.for_model(
        "gpt-5-mini",
        transport_override="openai",
    )

    assert updated.model == "gpt-5-mini"
    assert updated.transport == "openai"
    assert config.transport == "anthropic"


def test_configured_model_settings_validate_like_runtime_model_config() -> None:
    with pytest.raises(ValueError, match="thinking_budget_tokens must be >= 1024"):
        ConfiguredModelSettings(
            model="claude-haiku-4-5",
            transport="anthropic",
            thinking_budget_tokens=512,
        )


def test_summary_settings_accept_nested_model_config() -> None:
    from src.config import FallbackModelSettings

    settings = SummarySettings(
        MODEL_CONFIG=ConfiguredModelSettings(
            model="claude-haiku-4-5",
            transport="anthropic",
            fallback=FallbackModelSettings(
                model="gemini-2.5-pro",
                transport="gemini",
            ),
            thinking_budget_tokens=1024,
        ),
    )

    assert settings.MODEL_CONFIG.model == "claude-haiku-4-5"
    assert settings.MODEL_CONFIG.transport == "anthropic"
    assert settings.MODEL_CONFIG.fallback is not None
    assert settings.MODEL_CONFIG.fallback.model == "gemini-2.5-pro"
    assert settings.MODEL_CONFIG.fallback.transport == "gemini"
    assert settings.MODEL_CONFIG.thinking_budget_tokens == 1024


def test_resolve_model_config_reads_override_env_and_provider_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUMMARY_LOCAL_API_KEY", "test-key")

    configured = ConfiguredModelSettings(
        model="my-local-model",
        transport="openai",
        overrides=ModelOverrideSettings(
            api_key_env="SUMMARY_LOCAL_API_KEY",
            base_url="http://localhost:8000/v1",
            provider_params={"verbosity": "low"},
        ),
    )

    resolved = resolve_model_config(configured)

    assert resolved.api_key == "test-key"
    assert resolved.base_url == "http://localhost:8000/v1"
    assert resolved.provider_params == {"verbosity": "low"}


def test_resolve_embedding_model_config_reads_override_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_LOCAL_API_KEY", "embed-key")

    configured = ConfiguredEmbeddingModelSettings(
        transport="openai",
        model="text-embedding-3-small",
        overrides=ModelOverrideSettings(
            api_key_env="EMBEDDING_LOCAL_API_KEY",
            base_url="http://localhost:8000/v1",
        ),
    )

    resolved = resolve_embedding_model_config(configured)

    assert resolved.api_key == "embed-key"
    assert resolved.base_url == "http://localhost:8000/v1"


def test_dialectic_level_settings_accepts_nested_model_config() -> None:
    from src.config import FallbackModelSettings

    settings = DialecticLevelSettings(
        MODEL_CONFIG=ConfiguredModelSettings(
            model="claude-haiku-4-5",
            transport="anthropic",
            fallback=FallbackModelSettings(
                model="gemini-2.5-pro",
                transport="gemini",
            ),
            thinking_budget_tokens=1024,
        ),
        MAX_TOOL_ITERATIONS=2,
    )

    resolved = resolve_model_config(settings.MODEL_CONFIG)
    assert resolved.model == "claude-haiku-4-5"
    assert resolved.transport == "anthropic"
    assert resolved.fallback is not None
    assert resolved.fallback.model == "gemini-2.5-pro"
    assert resolved.fallback.transport == "gemini"


def test_dialectic_level_settings_require_nested_model_config() -> None:
    with pytest.raises(ValueError, match="Field required"):
        DialecticLevelSettings.model_validate({"MAX_TOOL_ITERATIONS": 2})


def test_dialectic_level_settings_reject_legacy_flat_model_shape() -> None:
    with pytest.raises(ValueError, match="Field required"):
        DialecticLevelSettings.model_validate(
            {
                "MODEL": "claude-haiku-4-5",
                "THINKING_BUDGET_TOKENS": 1024,
                "MAX_TOOL_ITERATIONS": 2,
            }
        )


def test_legacy_prefixed_model_strings_are_normalized() -> None:
    config = ModelConfig.model_validate({"model": "gemini/gemini-2.5-flash"})
    configured = ConfiguredModelSettings.model_validate(
        {"model": "anthropic/claude-haiku-4-5"}
    )

    assert config.transport == "gemini"
    assert config.model == "gemini-2.5-flash"
    assert configured.transport == "anthropic"
    assert configured.model == "claude-haiku-4-5"


def test_dream_specialist_model_configs_are_independent() -> None:
    """Specialist configs carry their own defaults and don't inherit from a parent."""

    dream = DreamSettings(
        DEDUCTION_MODEL_CONFIG=ConfiguredModelSettings(
            model="claude-haiku-4-5",
            transport="anthropic",
            thinking_budget_tokens=2048,
        ),
        INDUCTION_MODEL_CONFIG=ConfiguredModelSettings(
            model="claude-opus-4-1",
            transport="anthropic",
            max_output_tokens=8000,
        ),
    )

    assert dream.DEDUCTION_MODEL_CONFIG.model == "claude-haiku-4-5"
    assert dream.DEDUCTION_MODEL_CONFIG.thinking_budget_tokens == 2048
    assert dream.DEDUCTION_MODEL_CONFIG.max_output_tokens is None

    assert dream.INDUCTION_MODEL_CONFIG.model == "claude-opus-4-1"
    assert dream.INDUCTION_MODEL_CONFIG.max_output_tokens == 8000
    assert dream.INDUCTION_MODEL_CONFIG.thinking_budget_tokens is None


def test_app_settings_propagate_embedding_dimensions_to_vector_store() -> None:
    settings = AppSettings(
        EMBEDDING=EmbeddingSettings(VECTOR_DIMENSIONS=2048),
        VECTOR_STORE=VectorStoreSettings(TYPE="lancedb", MIGRATED=True),
    )

    assert settings.EMBEDDING.VECTOR_DIMENSIONS == 2048
    assert settings.VECTOR_STORE.DIMENSIONS == 2048


def test_app_settings_require_matching_embedding_and_vector_store_dimensions() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "VECTOR_STORE.DIMENSIONS must match EMBEDDING.VECTOR_DIMENSIONS"
        ),
    ):
        AppSettings(
            EMBEDDING=EmbeddingSettings(VECTOR_DIMENSIONS=2048),
            VECTOR_STORE=VectorStoreSettings(
                TYPE="lancedb",
                MIGRATED=True,
                DIMENSIONS=1536,
            ),
        )


def test_app_settings_reject_non_1536_dimensions_while_pgvector_or_dual_write_active() -> (
    None
):
    with pytest.raises(
        ValueError,
        match=re.escape("EMBEDDING.VECTOR_DIMENSIONS must remain 1536"),
    ):
        AppSettings(
            EMBEDDING=EmbeddingSettings(VECTOR_DIMENSIONS=2048),
            VECTOR_STORE=VectorStoreSettings(TYPE="pgvector", MIGRATED=True),
        )

    with pytest.raises(
        ValueError,
        match=re.escape("EMBEDDING.VECTOR_DIMENSIONS must remain 1536"),
    ):
        AppSettings(
            EMBEDDING=EmbeddingSettings(VECTOR_DIMENSIONS=2048),
            VECTOR_STORE=VectorStoreSettings(TYPE="lancedb", MIGRATED=False),
        )


def test_config_toml_example_uses_nested_model_config_sections() -> None:
    config_path = Path(__file__).resolve().parents[2] / "config.toml.example"
    config_data = load_toml_config(str(config_path))

    deriver_config = ConfiguredModelSettings.model_validate(
        config_data["deriver"]["model_config"]
    )
    minimal_level = DialecticLevelSettings.model_validate(
        config_data["dialectic"]["levels"]["minimal"]
    )
    low_level = DialecticLevelSettings.model_validate(
        config_data["dialectic"]["levels"]["low"]
    )
    max_level = DialecticLevelSettings.model_validate(
        config_data["dialectic"]["levels"]["max"]
    )
    embedding_config = ConfiguredEmbeddingModelSettings.model_validate(
        config_data["embedding"]["model_config"]
    )
    summary_config = ConfiguredModelSettings.model_validate(
        config_data["summary"]["model_config"]
    )
    deduction_model_config = ConfiguredModelSettings.model_validate(
        config_data["dream"]["deduction_model_config"]
    )
    induction_model_config = ConfiguredModelSettings.model_validate(
        config_data["dream"]["induction_model_config"]
    )
    dream = DreamSettings.model_validate(
        {
            "DEDUCTION_MODEL_CONFIG": deduction_model_config,
            "INDUCTION_MODEL_CONFIG": induction_model_config,
        }
    )

    # config.toml.example ships the same minimal defaults the app uses:
    # transport=openai, model=gpt-5.4-mini across every text-generation
    # feature, with embeddings on openai/text-embedding-3-small. Asserting
    # these keeps the example file and the in-code defaults in lockstep.
    assert deriver_config.transport == "openai"
    assert deriver_config.model == "gpt-5.4-mini"
    assert deriver_config.thinking_budget_tokens is None
    assert minimal_level.MODEL_CONFIG.model == "gpt-5.4-mini"
    assert minimal_level.MODEL_CONFIG.transport == "openai"
    assert minimal_level.TOOL_CHOICE == "auto"
    assert low_level.TOOL_CHOICE == "auto"
    assert max_level.MODEL_CONFIG.model == "gpt-5.4-mini"
    assert max_level.MODEL_CONFIG.transport == "openai"
    assert max_level.MODEL_CONFIG.thinking_budget_tokens is None
    assert embedding_config.transport == "openai"
    assert embedding_config.model == "text-embedding-3-small"
    assert summary_config.model == "gpt-5.4-mini"
    assert summary_config.transport == "openai"
    assert dream.DEDUCTION_MODEL_CONFIG.model == "gpt-5.4-mini"
    assert dream.INDUCTION_MODEL_CONFIG.model == "gpt-5.4-mini"


def test_env_template_uses_nested_model_config_keys() -> None:
    env_template_path = Path(__file__).resolve().parents[2] / ".env.template"
    env_template = env_template_path.read_text()

    assert "EMBEDDING_MODEL_CONFIG__MODEL" in env_template
    assert "EMBEDDING_VECTOR_DIMENSIONS" in env_template
    assert "DERIVER_MODEL_CONFIG__MODEL" in env_template
    assert "DIALECTIC_LEVELS__minimal__MODEL_CONFIG__MODEL" in env_template
    assert "DIALECTIC_LEVELS__minimal__TOOL_CHOICE=auto" in env_template
    assert "DIALECTIC_LEVELS__low__TOOL_CHOICE=auto" in env_template
    assert "SUMMARY_MODEL_CONFIG__MODEL" in env_template
    assert "DREAM_DEDUCTION_MODEL_CONFIG__MODEL" in env_template

    assert "DERIVER_PROVIDER=" not in env_template
    assert "SUMMARY_PROVIDER=" not in env_template
    assert "DIALECTIC_LEVELS__minimal__PROVIDER=" not in env_template
    assert "DREAM_PROVIDER=" not in env_template
    assert "DREAM_DEDUCTION_MODEL=" not in env_template


def _clear_deriver_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any DERIVER_MODEL_CONFIG__* env that would interfere with
    direct-construction tests."""
    for name in list(os.environ):
        if name.startswith("DERIVER_MODEL_CONFIG__") or name == "DERIVER_MODEL_CONFIG":
            monkeypatch.delenv(name, raising=False)


def test_partial_env_override_of_transport_drops_default_thinking_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial env override of transport must not leak the default's thinking
    params into a transport that rejects them.

    Regression: setting DERIVER_MODEL_CONFIG__TRANSPORT=openai +
    DERIVER_MODEL_CONFIG__MODEL=gpt-4.1-mini (without clearing the default
    thinking_budget_tokens=1024 carried over from the gemini default) used to
    produce a merged ConfiguredModelSettings with thinking_budget_tokens=1024,
    which the OpenAI backend then rejected at call time.
    """
    from src.config import DeriverSettings

    _clear_deriver_env(monkeypatch)
    # Exercise the @model_validator(mode="before") merge path with a raw dict
    # — pyright can't see through the pre-validator that accepts dict input.
    settings = DeriverSettings(
        MODEL_CONFIG={"transport": "openai", "model": "gpt-4.1-mini"},  # pyright: ignore[reportArgumentType]
    )

    assert settings.MODEL_CONFIG.transport == "openai"
    assert settings.MODEL_CONFIG.model == "gpt-4.1-mini"
    assert settings.MODEL_CONFIG.thinking_budget_tokens is None
    assert settings.MODEL_CONFIG.thinking_effort is None


def test_partial_env_override_same_transport_keeps_default_thinking_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When env preserves the default transport, default thinking params still
    apply — we only strip on actual transport change.

    The app-level defaults are intentionally minimal (transport + model only)
    to avoid clobbering operator config, so this test patches in a deliberately
    rich default to exercise the merge-preservation behavior.
    """
    from src.config import ConfiguredModelSettings, DeriverSettings

    _clear_deriver_env(monkeypatch)

    def _rich_default() -> ConfiguredModelSettings:
        return ConfiguredModelSettings(
            transport="gemini",
            model="gemini-2.5-flash-lite",
            thinking_budget_tokens=1024,
            max_output_tokens=4096,
        )

    monkeypatch.setattr(DeriverSettings, "_MODEL_CONFIG_DEFAULT", _rich_default)

    settings = DeriverSettings(
        MODEL_CONFIG={"model": "gemini-2.5-pro"},  # pyright: ignore[reportArgumentType]
    )

    assert settings.MODEL_CONFIG.transport == "gemini"
    assert settings.MODEL_CONFIG.model == "gemini-2.5-pro"
    assert settings.MODEL_CONFIG.thinking_budget_tokens == 1024
    assert settings.MODEL_CONFIG.max_output_tokens == 4096


def test_explicit_thinking_effort_survives_transport_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User-set thinking params in the override are always preserved."""
    from src.config import DeriverSettings

    _clear_deriver_env(monkeypatch)
    settings = DeriverSettings(
        MODEL_CONFIG={  # pyright: ignore[reportArgumentType]
            "transport": "openai",
            "model": "gpt-5",
            "thinking_effort": "high",
        },
    )

    assert settings.MODEL_CONFIG.transport == "openai"
    assert settings.MODEL_CONFIG.thinking_effort == "high"
    assert settings.MODEL_CONFIG.thinking_budget_tokens is None


def test_dialectic_level_transport_override_drops_default_thinking_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same leak existed in DialecticSettings._merge_level_defaults.
    Regression: when a level default has thinking_budget_tokens=0 under a
    gemini transport and env flips the override to openai, the 0 used to leak
    through and trip the OpenAI backend's thinking-param rejection.

    The app-level defaults are intentionally minimal (transport + model only)
    to avoid clobbering operator config, so this test patches in a rich
    level default to exercise the strip-on-transport-change behavior.

    Exercises the before-validator directly to avoid DialecticSettings'
    "all 5 levels required" constraint.
    """
    from src.config import (
        ConfiguredModelSettings,
        DialecticLevelSettings,
        DialecticSettings,
    )

    def _rich_levels() -> dict[str, DialecticLevelSettings]:
        return {
            "minimal": DialecticLevelSettings(
                MODEL_CONFIG=ConfiguredModelSettings(
                    transport="gemini",
                    model="gemini-2.5-flash-lite",
                    thinking_budget_tokens=0,
                ),
                MAX_TOOL_ITERATIONS=1,
                MAX_OUTPUT_TOKENS=250,
                TOOL_CHOICE="any",
            ),
        }

    monkeypatch.setattr("src.config._default_dialectic_levels", _rich_levels)

    data: dict[str, object] = {
        "LEVELS": {
            "minimal": {
                "MODEL_CONFIG": {
                    "transport": "openai",
                    "model": "gpt-4.1-mini",
                }
            }
        }
    }
    # The @model_validator decorator wraps the classmethod in a descriptor proxy
    # that pyright can't see as callable; at runtime pydantic routes it correctly.
    merged = cast(
        dict[str, Any],
        DialecticSettings._merge_level_defaults(data),  # pyright: ignore[reportPrivateUsage, reportCallIssue]
    )
    levels = cast(dict[str, dict[str, Any]], merged["LEVELS"])
    minimal_mc = cast(dict[str, Any], levels["minimal"]["MODEL_CONFIG"])
    assert minimal_mc["transport"] == "openai"
    assert minimal_mc["model"] == "gpt-4.1-mini"
    assert "thinking_budget_tokens" not in minimal_mc
    assert "thinking_effort" not in minimal_mc


def test_dialectic_settings_backfills_missing_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operators only need to override the levels they care about.

    Env-var overrides replace the LEVELS dict wholesale (bypassing the
    default_factory), so without a backfill the unmentioned levels would be
    dropped and _validate_all_levels_present would fail.
    """
    from src.config import (
        DialecticSettings,
        _default_dialectic_levels,  # pyright: ignore[reportPrivateUsage]
    )

    for key in list(os.environ):
        if key.startswith("DIALECTIC_LEVELS"):
            monkeypatch.delenv(key)

    settings = DialecticSettings(
        LEVELS={  # pyright: ignore[reportArgumentType]
            "low": {
                "MODEL_CONFIG": {
                    "transport": "anthropic",
                    "model": "claude-haiku-4-5-20251001",
                    "thinking_budget_tokens": 1024,
                },
                "MAX_OUTPUT_TOKENS": 2500,
            }
        }
    )

    assert set(settings.LEVELS.keys()) == {"minimal", "low", "medium", "high", "max"}
    assert settings.LEVELS["low"].MODEL_CONFIG.transport == "anthropic"
    assert settings.LEVELS["low"].MODEL_CONFIG.model == "claude-haiku-4-5-20251001"
    assert settings.LEVELS["low"].MAX_OUTPUT_TOKENS == 2500
    # Backfilled levels come from _default_dialectic_levels()
    defaults = _default_dialectic_levels()
    assert (
        settings.LEVELS["minimal"].MAX_TOOL_ITERATIONS
        == defaults["minimal"].MAX_TOOL_ITERATIONS
    )
    assert (
        settings.LEVELS["max"].MAX_TOOL_ITERATIONS
        == defaults["max"].MAX_TOOL_ITERATIONS
    )
