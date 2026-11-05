"""The MODELS catalog: validation, `model_ref` expansion, and capability profiles.

Whole-config behavior (env and TOML, every slot, before vs after) is covered
by `test_model_config_snapshot.py` and `test_model_catalog_equivalence.py`.
"""

import os
from typing import Any

import pytest
from pydantic import ValidationError

from src import config
from src.config import (
    AppSettings,
    ConfiguredModelSettings,
    DeriverSettings,
    DialecticSettings,
    FallbackModelSettings,
    ModelCatalogEntry,
    ModelOverrideSettings,
    TomlConfigSettingsSource,
    using_model_catalog,
)
from src.llm.backend import CompletionResult
from src.llm.backends.anthropic import AnthropicBackend
from src.llm.capabilities import (
    anthropic_capabilities,
    capabilities_for,
    catalog_entry_capabilities,
    model_capabilities,
)
from src.llm.messages import ModelOrigin
from src.llm.profiles import profile_for
from tests.llm.test_backends.test_anthropic import (
    SEARCH_TOOL,
    _client_with_text_response,  # pyright: ignore[reportPrivateUsage]
)
from tests.llm.test_codecs import (
    CALLS,
    _render,  # pyright: ignore[reportPrivateUsage]
    _round,  # pyright: ignore[reportPrivateUsage]
)

_WORKLOAD_ENV_PREFIXES = (
    "DERIVER_",
    "SUMMARY_",
    "DREAM_",
    "DIALECTIC_",
    "EMBEDDING_",
    "MODELS",
)


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings built here see only what each test passes in."""
    monkeypatch.setattr("src.config.TOML_CONFIG", {})
    for name in list(os.environ):
        if name.upper().startswith(_WORKLOAD_ENV_PREFIXES):
            monkeypatch.delenv(name)


def _entry(**fields: Any) -> ModelCatalogEntry:
    return ModelCatalogEntry.model_validate(fields)


HAIKU = _entry(
    model="claude-haiku-4-5",
    transport="anthropic",
    thinking_budget_tokens=2048,
    temperature=0.3,
    overrides={
        "base_url": "https://anthropic-proxy.example",
        "api_key": "sk-entry",
        "provider_params": {"extra_body": {"entry": True}, "timeout": 30},
    },
)
NANO = _entry(
    model="gpt-5-nano",
    transport="openai",
    thinking_effort="minimal",
    structured_output_mode="json_object",
    max_output_tokens=8000,
)
CATALOG = {"haiku": HAIKU, "nano": NANO}


# --- catalog validation -----------------------------------------------------


@pytest.mark.parametrize("name", ["Haiku", "gpt-5-nano", "5mini", "", "has space"])
def test_catalog_names_must_be_lowercase_identifiers(name: str) -> None:
    with pytest.raises(ValidationError, match="must be lowercase letters"):
        AppSettings(MODELS={name: {"model": "m", "transport": "openai"}})  # pyright: ignore[reportArgumentType]


def test_catalog_accepts_lowercase_names() -> None:
    loaded = AppSettings(
        MODELS={"gpt5_nano": {"model": "gpt-5-nano", "transport": "openai"}}  # pyright: ignore[reportArgumentType]
    )
    assert loaded.MODELS["gpt5_nano"].model == "gpt-5-nano"


def test_catalog_entry_rejects_unknown_fields() -> None:
    """A typo in an entry fails at load instead of being dropped."""
    with pytest.raises(ValidationError, match="modle"):
        _entry(modle="gpt-5", model="gpt-5", transport="openai")
    with pytest.raises(ValidationError, match="infer_frm"):
        _entry(model="x", transport="openai", capabilities={"infer_frm": "y"})


def test_catalog_entry_cannot_reference_another_entry() -> None:
    with pytest.raises(ValidationError, match="can't set model_ref"):
        _entry(model_ref="haiku")


def test_catalog_entry_keeps_the_slot_validations() -> None:
    with pytest.raises(ValidationError, match="only supported on the 'openai'"):
        _entry(
            model="claude-haiku-4-5",
            transport="anthropic",
            structured_output_mode="json_object",
        )
    with pytest.raises(ValidationError, match=">= 1024"):
        _entry(
            model="claude-haiku-4-5", transport="anthropic", thinking_budget_tokens=10
        )


def test_catalog_entry_accepts_provider_shorthand() -> None:
    entry = _entry(model="anthropic/claude-opus-4-6")
    assert (entry.transport, entry.model) == ("anthropic", "claude-opus-4-6")


def test_entries_for_one_model_must_agree_on_capabilities() -> None:
    with pytest.raises(ValidationError, match="declare different capabilities"):
        AppSettings(
            MODELS={  # pyright: ignore[reportArgumentType]
                "a": {"model": "alias", "transport": "openai"},
                "b": {
                    "model": "alias",
                    "transport": "openai",
                    "capabilities": {"infer_from": "claude-haiku-5-5"},
                },
            }
        )


def test_entries_for_one_model_may_differ_in_everything_else() -> None:
    loaded = AppSettings(
        MODELS={  # pyright: ignore[reportArgumentType]
            "cheap": {
                "model": "gpt-5",
                "transport": "openai",
                "thinking_effort": "low",
            },
            "deep": {
                "model": "gpt-5",
                "transport": "openai",
                "thinking_effort": "high",
            },
            "direct": {"model": "gpt-5", "transport": "anthropic"},
        }
    )
    assert set(loaded.MODELS) == {"cheap", "deep", "direct"}


def test_invalid_catalog_reports_its_own_errors_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slots that name entries in a broken catalog don't hide why it's broken."""
    monkeypatch.setenv("DERIVER_MODEL_CONFIG__MODEL_REF", "haiku")
    with pytest.raises(ValidationError) as exc_info:
        AppSettings(MODELS={"haiku": {"transport": "anthropic"}})  # pyright: ignore[reportArgumentType]
    message = str(exc_info.value)
    assert "MODELS.haiku.model: Field required" in message
    assert "model_ref" not in message


def test_root_models_table_is_read_from_toml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.config.TOML_CONFIG",
        {
            "models": {"nano": {"model": "gpt-5-nano", "transport": "openai"}},
            "app": {"models": {"ignored": {"model": "x", "transport": "openai"}}},
        },
    )
    values = TomlConfigSettingsSource(AppSettings)()
    assert values["MODELS"] == {"nano": {"model": "gpt-5-nano", "transport": "openai"}}
    assert set(AppSettings().MODELS) == {"nano"}


# --- model_ref expansion ----------------------------------------------------


def test_ref_expands_to_the_entry() -> None:
    with using_model_catalog(CATALOG):
        slot = ConfiguredModelSettings.model_validate({"model_ref": "haiku"})
    assert slot == ConfiguredModelSettings.model_validate(
        HAIKU.model_dump(exclude={"capabilities"}, exclude_none=True)
    )
    assert slot.overrides.base_url == "https://anthropic-proxy.example"
    assert slot.temperature == 0.3


def test_ref_is_case_insensitive() -> None:
    with using_model_catalog(CATALOG):
        slot = ConfiguredModelSettings.model_validate({"MODEL_REF": "Haiku"})
    assert slot.model == "claude-haiku-4-5"


def test_slot_fields_win_over_the_entry() -> None:
    with using_model_catalog(CATALOG):
        slot = ConfiguredModelSettings.model_validate(
            {
                "model_ref": "haiku",
                "temperature": 0.9,
                "thinking_budget_tokens": 4096,
                "max_output_tokens": 10_000,
            }
        )
    assert slot.temperature == 0.9
    assert slot.thinking_budget_tokens == 4096
    assert slot.max_output_tokens == 10_000


def test_slot_overrides_merge_key_by_key_into_the_entry() -> None:
    with using_model_catalog(CATALOG):
        slot = ConfiguredModelSettings.model_validate(
            {
                "model_ref": "haiku",
                "overrides": {
                    "api_key": "sk-slot",
                    "provider_params": {
                        "timeout": 90,
                        "extra_headers": {"x-team": "a"},
                    },
                },
            }
        )
    assert slot.overrides.base_url == "https://anthropic-proxy.example"
    assert slot.overrides.api_key == "sk-slot"
    assert slot.overrides.provider_params == {
        "extra_body": {"entry": True},
        "timeout": 90.0,
        "extra_headers": {"x-team": "a"},
    }


def test_slot_extra_body_replaces_the_entry_extra_body_whole() -> None:
    with using_model_catalog(CATALOG):
        slot = ConfiguredModelSettings.model_validate(
            {
                "model_ref": "haiku",
                "overrides": {"provider_params": {"extra_body": {"slot": True}}},
            }
        )
    assert slot.overrides.provider_params == {
        "extra_body": {"slot": True},
        "timeout": 30.0,
    }


def test_slot_overrides_given_as_a_model_merge_like_a_dict() -> None:
    with using_model_catalog(CATALOG):
        slot = ConfiguredModelSettings(
            model_ref="haiku",  # pyright: ignore[reportCallIssue]
            overrides=ModelOverrideSettings(api_key="sk-slot"),
        )
    assert slot.overrides.api_key == "sk-slot"
    assert slot.overrides.base_url == "https://anthropic-proxy.example"
    assert slot.overrides.provider_params["extra_body"] == {"entry": True}


def test_slot_reasoning_effort_alias_wins_over_entry_thinking_effort() -> None:
    with using_model_catalog(CATALOG):
        slot = ConfiguredModelSettings.model_validate(
            {"model_ref": "nano", "reasoning_effort": "high"}
        )
    assert slot.thinking_effort == "high"


def test_entry_does_not_leak_into_the_slot_fallback() -> None:
    with using_model_catalog(CATALOG):
        slot = ConfiguredModelSettings.model_validate(
            {
                "model_ref": "haiku",
                "fallback": {"model": "gpt-5-mini", "transport": "openai"},
            }
        )
    assert slot.fallback is not None
    assert slot.fallback.temperature is None
    assert slot.fallback.overrides.base_url is None


def test_fallback_can_name_an_entry() -> None:
    with using_model_catalog(CATALOG):
        slot = ConfiguredModelSettings.model_validate(
            {"model_ref": "haiku", "fallback": {"model_ref": "nano"}}
        )
    assert slot.fallback == FallbackModelSettings.model_validate(
        NANO.model_dump(exclude={"capabilities"}, exclude_none=True)
    )

    resolved = config.resolve_model_config(slot)
    assert resolved.fallback is not None
    assert resolved.fallback.structured_output_mode == "json_object"
    assert resolved.fallback.max_output_tokens == 8000


@pytest.mark.parametrize("field", ["model", "transport", "MODEL"])
def test_ref_cannot_be_combined_with_model_or_transport(field: str) -> None:
    value = "openai" if field == "transport" else "gpt-5"
    with (
        using_model_catalog(CATALOG),
        pytest.raises(ValidationError, match="already sets the model"),
    ):
        ConfiguredModelSettings.model_validate({"model_ref": "haiku", field: value})


def test_unknown_ref_names_the_entries() -> None:
    with (
        using_model_catalog(CATALOG),
        pytest.raises(ValidationError, match=r"'opus' is not in .*haiku, nano"),
    ):
        ConfiguredModelSettings.model_validate({"model_ref": "opus"})


def test_unknown_ref_with_no_catalog() -> None:
    with (
        using_model_catalog({}),
        pytest.raises(ValidationError, match="none defined"),
    ):
        FallbackModelSettings.model_validate({"model_ref": "nano"})


@pytest.mark.parametrize("blank", [None, ""])
def test_blank_ref_is_ignored(blank: str | None) -> None:
    with using_model_catalog(CATALOG):
        slot = ConfiguredModelSettings.model_validate(
            {"model_ref": blank, "model": "gpt-5", "transport": "openai"}
        )
    assert (slot.transport, slot.model) == ("openai", "gpt-5")


def test_ref_must_be_a_string() -> None:
    with (
        using_model_catalog(CATALOG),
        pytest.raises(ValidationError, match="must be the name"),
    ):
        ConfiguredModelSettings.model_validate({"model_ref": 3})


def test_without_an_active_catalog_refs_use_the_loaded_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, "MODELS", {"nano": NANO})
    slot = ConfiguredModelSettings.model_validate({"model_ref": "nano"})
    assert slot.model == "gpt-5-nano"


def test_using_model_catalog_restores_the_previous_catalog() -> None:
    with using_model_catalog(CATALOG):
        with using_model_catalog({}), pytest.raises(ValidationError):
            ConfiguredModelSettings.model_validate({"model_ref": "haiku"})
        assert ConfiguredModelSettings.model_validate({"model_ref": "haiku"})


# --- slot defaults ------------------------------------------------------------


def test_slot_default_is_not_merged_into_a_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    """None of the slot's default fields leak into a slot that names an entry."""

    def _rich_default() -> ConfiguredModelSettings:
        return ConfiguredModelSettings(
            transport="openai",
            model="gpt-5.4-mini",
            thinking_effort="high",
            top_p=0.5,
            max_output_tokens=4096,
        )

    monkeypatch.setattr(DeriverSettings, "_MODEL_CONFIG_DEFAULT", _rich_default)
    with using_model_catalog(CATALOG):
        deriver = DeriverSettings(MODEL_CONFIG={"model_ref": "haiku"})  # pyright: ignore[reportArgumentType]

    assert deriver.MODEL_CONFIG.model == "claude-haiku-4-5"
    assert deriver.MODEL_CONFIG.thinking_effort is None
    assert deriver.MODEL_CONFIG.top_p is None
    assert deriver.MODEL_CONFIG.max_output_tokens is None


def test_slot_default_still_fills_a_blank_ref() -> None:
    with using_model_catalog(CATALOG):
        deriver = DeriverSettings(MODEL_CONFIG={"model_ref": "", "model": "gpt-4.1"})  # pyright: ignore[reportArgumentType]
    assert (deriver.MODEL_CONFIG.transport, deriver.MODEL_CONFIG.model) == (
        "openai",
        "gpt-4.1",
    )


@pytest.mark.parametrize("key", ["MODEL_CONFIG", "model_config"])
def test_dialectic_level_ref_skips_the_level_default_model(key: str) -> None:
    with using_model_catalog(CATALOG):
        dialectic = DialecticSettings(
            LEVELS={
                "high": {key: {"model_ref": "haiku", "thinking_budget_tokens": 4096}}
            }  # pyright: ignore[reportArgumentType]
        )
    high = dialectic.LEVELS["high"]
    assert high.MODEL_CONFIG.model == "claude-haiku-4-5"
    assert high.MODEL_CONFIG.thinking_budget_tokens == 4096
    assert high.MODEL_CONFIG.overrides.base_url == "https://anthropic-proxy.example"
    # The level's own (non-model) defaults still apply.
    assert high.MAX_TOOL_ITERATIONS == 4
    assert dialectic.LEVELS["low"].MODEL_CONFIG.model == "gpt-5.4-mini"


def test_dialectic_budget_check_sees_the_entry_budget() -> None:
    big = _entry(
        model="claude-opus-4-6", transport="anthropic", thinking_budget_tokens=9000
    )
    with (
        using_model_catalog({"big": big}),
        pytest.raises(ValidationError, match="MAX_OUTPUT_TOKENS must be greater"),
    ):
        DialecticSettings(LEVELS={"low": {"MODEL_CONFIG": {"model_ref": "big"}}})  # pyright: ignore[reportArgumentType]


def test_app_settings_children_resolve_against_its_own_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Child settings built while AppSettings validates see that catalog, not
    the one already loaded."""
    monkeypatch.setattr(config.settings, "MODELS", {})
    monkeypatch.setenv("DERIVER_MODEL_CONFIG__MODEL_REF", "nano")
    monkeypatch.setenv("SUMMARY_MODEL_CONFIG__FALLBACK__MODEL_REF", "nano")

    loaded = AppSettings(MODELS={"nano": NANO.model_dump()})  # pyright: ignore[reportArgumentType]

    assert loaded.DERIVER.MODEL_CONFIG.model == "gpt-5-nano"
    assert loaded.SUMMARY.MODEL_CONFIG.fallback is not None
    assert loaded.SUMMARY.MODEL_CONFIG.fallback.model == "gpt-5-nano"


def test_unused_provider_key_warning_counts_referenced_entries(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("DERIVER_MODEL_CONFIG__MODEL_REF", "haiku")
    with caplog.at_level("WARNING", logger="src.config"):
        AppSettings(
            MODELS={"haiku": HAIKU.model_dump()},  # pyright: ignore[reportArgumentType]
            LLM=config.LLMSettings(
                OPENAI_API_KEY="fake-openai", ANTHROPIC_API_KEY="fake-anthropic"
            ),
        )
    assert "LLM_ANTHROPIC_API_KEY" not in caplog.text


# --- capability profiles ------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "claude-haiku-5-5",
        "anthropic/claude-sonnet-4.6",
        "us.anthropic.claude-opus-5-5-v1:0",
        "claude-3-5-sonnet-20241022",
    ],
)
def test_model_capabilities_reads_claude_ids_on_any_transport(model: str) -> None:
    assert model_capabilities(model) == anthropic_capabilities(model)


def test_model_capabilities_for_other_models() -> None:
    assert model_capabilities("mistral-small-3.2").user_after_tool is False
    assert model_capabilities("gpt-5") == capabilities_for("openai", "gpt-5")


def test_entry_profile_is_inferred_from_its_model_whatever_the_transport() -> None:
    over_proxy = _entry(model="anthropic/claude-opus-5-5", transport="openai")
    assert catalog_entry_capabilities(over_proxy) == anthropic_capabilities(
        "claude-opus-5-5"
    )


def test_entry_profile_infer_from_names_the_model_behind_an_alias() -> None:
    alias = _entry(
        model="summarizer",
        transport="openai",
        capabilities={"infer_from": "claude-haiku-5-5"},
    )
    assert catalog_entry_capabilities(alias) == anthropic_capabilities(
        "claude-haiku-5-5"
    )


def test_entry_profile_fields_override_the_inferred_row() -> None:
    alias = _entry(
        model="private-model",
        transport="openai",
        capabilities={
            "infer_from": "claude-haiku-5-5",
            "forced_tool_choice": "rejected",
            "user_after_tool": False,
        },
    )
    caps = catalog_entry_capabilities(alias)
    assert caps.forced_tool_choice == "rejected"
    assert caps.user_after_tool is False
    assert caps.thinking == "adaptive"


def test_entry_without_a_profile_for_an_unknown_model_gets_the_default_row() -> None:
    assert catalog_entry_capabilities(
        _entry(model="summarizer", transport="openai")
    ) == capabilities_for("openai", "summarizer")


@pytest.fixture
def declared_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """`summarizer` on openai and `claude-proxy` on anthropic, both declared."""
    monkeypatch.setattr(
        config.settings,
        "MODELS",
        {
            "summarizer": _entry(
                model="summarizer",
                transport="openai",
                capabilities={
                    "infer_from": "claude-haiku-5-5",
                    "user_after_tool": False,
                },
            ),
            "claude_proxy": _entry(
                model="claude-proxy",
                transport="anthropic",
                capabilities={"infer_from": "claude-opus-5-5"},
            ),
        },
    )


@pytest.mark.usefixtures("declared_alias")
def test_capabilities_for_uses_the_declared_profile() -> None:
    caps = capabilities_for("openai", "summarizer")
    assert caps.thinking == "adaptive"
    assert caps.preserved_thinking is True
    assert caps.user_after_tool is False
    # The profile is bound to the transport too.
    assert capabilities_for("anthropic", "summarizer") == anthropic_capabilities(
        "summarizer"
    )
    assert profile_for("openai", "summarizer").capabilities == caps


@pytest.mark.usefixtures("declared_alias")
def test_models_outside_the_catalog_are_unaffected() -> None:
    assert capabilities_for("openai", "gpt-5") == model_capabilities("gpt-5")
    assert capabilities_for("anthropic", "claude-haiku-4-5") == (
        anthropic_capabilities("claude-haiku-4-5")
    )
    # Claude over the openai transport keeps the default row unless declared.
    assert capabilities_for("openai", "claude-opus-5-5") == capabilities_for(
        "openai", "gpt-5"
    )


def test_capability_lookup_follows_a_replaced_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, "MODELS", {})
    before = capabilities_for("openai", "summarizer")
    monkeypatch.setattr(
        config.settings,
        "MODELS",
        {
            "s": _entry(
                model="summarizer",
                transport="openai",
                capabilities={"infer_from": "claude-haiku-5-5"},
            )
        },
    )
    assert capabilities_for("openai", "summarizer") != before
    monkeypatch.setattr(config.settings, "MODELS", {})
    assert capabilities_for("openai", "summarizer") == before


@pytest.mark.usefixtures("declared_alias")
def test_anthropic_backend_reads_the_declared_profile() -> None:
    supports_prefill = AnthropicBackend._supports_assistant_prefill  # pyright: ignore[reportPrivateUsage]
    assert supports_prefill("claude-proxy") is False
    assert supports_prefill("claude-3-5-sonnet-20241022") is True


@pytest.mark.asyncio
@pytest.mark.usefixtures("declared_alias")
async def test_anthropic_backend_relaxes_forced_tool_choice_for_a_declared_alias() -> (
    None
):
    client = _client_with_text_response()
    await AnthropicBackend(client).complete(
        model="claude-proxy",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=4096,
        tools=[SEARCH_TOOL],
        tool_choice="required",
        thinking_budget_tokens=2048,
    )
    call = client.messages.create.await_args.kwargs
    assert call["tool_choice"] == {"type": "auto"}
    assert call["thinking"] == {"type": "adaptive"}


@pytest.mark.usefixtures("declared_alias")
def test_openai_codec_reads_the_declared_role_order() -> None:
    alias = ModelOrigin(transport="openai", model="summarizer")
    plain = ModelOrigin(transport="openai", model="gpt-5")
    result = CompletionResult(content=None, tool_calls=CALLS)
    assert _render(_round(result, alias), alias)[-1]["role"] == "tool"
    assert _render(_round(result, plain), plain)[-1]["role"] == "user"
