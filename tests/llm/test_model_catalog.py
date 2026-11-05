"""The MODELS catalog: validation, `model_ref` expansion, and capability profiles.

Whole-config behavior (env and TOML, every slot, before vs after) is covered
by `test_model_config_snapshot.py` and `test_model_catalog_equivalence.py`.
"""

import os
from typing import Any
from unittest.mock import AsyncMock

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
from src.llm.backends.openai import OpenAIBackend
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
from tests.llm.test_backends.test_openai import (
    _conflict_client,  # pyright: ignore[reportPrivateUsage]
    _empty_stream,  # pyright: ignore[reportPrivateUsage]
    _sent_kwargs,  # pyright: ignore[reportPrivateUsage]
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


@pytest.mark.parametrize(
    ("entry_cred", "slot_cred", "expected"),
    [
        ({"api_key": "entry-key"}, {"api_key_env": "SLOT_KEY_VAR"}, "slot-env-key"),
        ({"api_key_env": "ENTRY_KEY_VAR"}, {"api_key": "slot-key"}, "slot-key"),
        ({"api_key": "entry-key"}, {"api_key": "slot-key"}, "slot-key"),
        # A slot that sets no credential keeps the entry's.
        ({"api_key": "entry-key"}, {"base_url": "https://slot.example"}, "entry-key"),
    ],
)
@pytest.mark.parametrize("role", ["primary", "fallback"])
def test_slot_credential_replaces_the_entry_credential(
    monkeypatch: pytest.MonkeyPatch,
    entry_cred: dict[str, str],
    slot_cred: dict[str, str],
    expected: str,
    role: str,
) -> None:
    monkeypatch.setenv("SLOT_KEY_VAR", "slot-env-key")
    monkeypatch.setenv("ENTRY_KEY_VAR", "entry-env-key")
    entry = _entry(model="m", transport="openai", overrides=entry_cred)
    slot = {"model_ref": "e", "overrides": slot_cred}
    with using_model_catalog({"e": entry}):
        if role == "primary":
            resolved = config.resolve_model_config(
                ConfiguredModelSettings.model_validate(slot)
            )
            assert resolved.api_key == expected
        else:
            resolved = config.resolve_model_config(
                ConfiguredModelSettings.model_validate(
                    {"model": "p", "transport": "openai", "fallback": slot}
                )
            )
            assert resolved.fallback is not None
            assert resolved.fallback.api_key == expected


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
    """Declared aliases: `summarizer` (Haiku 5.5), `opus-proxy` (Opus 5.5) and
    `astra` (gpt-6-astra) on openai, `claude-proxy` on anthropic, and
    `gpt-5.4-pinned`, a GPT-5 entry declared to take `max_tokens`."""
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
            "opus_proxy": _entry(
                model="opus-proxy",
                transport="openai",
                capabilities={"infer_from": "claude-opus-5-5"},
            ),
            "astra": _entry(
                model="astra",
                transport="openai",
                capabilities={"infer_from": "gpt-6-astra"},
            ),
            "gpt5_pinned": _entry(
                model="gpt-5.4-pinned",
                transport="openai",
                capabilities={"max_tokens_param": "max_tokens"},
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
        "openai", "gpt-4.1"
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


async def _openai_request(
    model: str, *, tool_choice: str | dict[str, Any] | None = None, stream: bool = False
) -> dict[str, Any]:
    """The kwargs the OpenAI backend sends for one request with a tool."""
    client = _conflict_client()
    backend = OpenAIBackend(client)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1000,
        "tools": [SEARCH_TOOL],
        "tool_choice": tool_choice,
    }
    if stream:
        client.chat.completions.create = AsyncMock(return_value=_empty_stream())
        async for _ in backend.stream(**kwargs):
            pass
    else:
        await backend.complete(**kwargs)
    return _sent_kwargs(client)


@pytest.mark.asyncio
@pytest.mark.usefixtures("declared_alias")
@pytest.mark.parametrize("stream", [False, True], ids=["complete", "stream"])
@pytest.mark.parametrize("tool_choice", ["required", "any", {"name": "search"}])
async def test_openai_backend_relaxes_forced_tool_choice_for_a_declared_alias(
    tool_choice: str | dict[str, Any],
    stream: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="src.llm.backends.openai"):
        sent = await _openai_request(
            "opus-proxy", tool_choice=tool_choice, stream=stream
        )
    assert sent["tool_choice"] == "auto"
    assert "opus-proxy rejects forced tool use" in caplog.text


@pytest.mark.asyncio
@pytest.mark.usefixtures("declared_alias")
@pytest.mark.parametrize(
    "model",
    [
        # Declared, but Haiku 5.5 accepts a forced tool choice.
        "summarizer",
        # Not in the catalog: Claude over the openai transport keeps the
        # default row, as before the catalog existed.
        "claude-opus-5-5",
        "anthropic/claude-sonnet-5-5",
    ],
)
async def test_openai_backend_keeps_forced_tool_choice_otherwise(model: str) -> None:
    sent = await _openai_request(model, tool_choice="required")
    assert sent["tool_choice"] == "required"


@pytest.mark.asyncio
@pytest.mark.usefixtures("declared_alias")
async def test_openai_backend_leaves_an_unforced_tool_choice_alone() -> None:
    assert (await _openai_request("opus-proxy", tool_choice="auto"))[
        "tool_choice"
    ] == "auto"
    assert "tool_choice" not in await _openai_request("opus-proxy")


@pytest.mark.asyncio
@pytest.mark.usefixtures("declared_alias")
@pytest.mark.parametrize(
    ("model", "field"),
    [
        # An alias inferred from an OpenAI reasoning model.
        ("astra", "max_completion_tokens"),
        # A declared field beats the inference from the id.
        ("gpt-5.4-pinned", "max_tokens"),
        # Not in the catalog: decided by the id, as before.
        ("gpt-5-mini", "max_completion_tokens"),
        ("o3-mini", "max_completion_tokens"),
        ("gpt-4.1", "max_tokens"),
        ("summarizer", "max_tokens"),
    ],
)
async def test_openai_backend_puts_the_cap_in_the_declared_field(
    model: str, field: str
) -> None:
    sent = await _openai_request(model)
    assert sent[field] == 1000
    other = {"max_tokens", "max_completion_tokens"} - {field}
    assert not other & set(sent)


def test_entry_max_tokens_param_is_inferred_and_overridable() -> None:
    assert (
        catalog_entry_capabilities(
            _entry(model="o3", transport="openai")
        ).max_tokens_param
        == "max_completion_tokens"
    )
    assert (
        catalog_entry_capabilities(
            _entry(
                model="reasoner",
                transport="openai",
                capabilities={"max_tokens_param": "max_completion_tokens"},
            )
        ).max_tokens_param
        == "max_completion_tokens"
    )
    with pytest.raises(ValidationError, match="max_tokens_param"):
        _entry(
            model="x", transport="openai", capabilities={"max_tokens_param": "tokens"}
        )


# --- context windows ----------------------------------------------------------


def test_no_context_window_means_no_check() -> None:
    loaded = AppSettings()
    assert loaded.DERIVER.MODEL_CONFIG.context_window is None


@pytest.mark.parametrize(
    ("env", "fits"),
    [
        # Deriver: input budget 25000 + LLM_DEFAULT_MAX_TOKENS 2500.
        ({"DERIVER_MODEL_CONFIG__CONTEXT_WINDOW": "27500"}, True),
        ({"DERIVER_MODEL_CONFIG__CONTEXT_WINDOW": "27499"}, False),
        # The deriver sends its primary's max_output_tokens instead.
        (
            {
                "DERIVER_MODEL_CONFIG__CONTEXT_WINDOW": "26000",
                "DERIVER_MODEL_CONFIG__MAX_OUTPUT_TOKENS": "1000",
            },
            True,
        ),
        # Summary: input budget 100000 + the larger summary cap, 6000.
        ({"SUMMARY_MODEL_CONFIG__CONTEXT_WINDOW": "106000"}, True),
        ({"SUMMARY_MODEL_CONFIG__CONTEXT_WINDOW": "105999"}, False),
        # Dialectic: input budget 100000 + the level's cap (minimal: 250)...
        ({"DIALECTIC_LEVELS__minimal__MODEL_CONFIG__CONTEXT_WINDOW": "100250"}, True),
        ({"DIALECTIC_LEVELS__minimal__MODEL_CONFIG__CONTEXT_WINDOW": "100249"}, False),
        # ...or DIALECTIC_MAX_OUTPUT_TOKENS (8192) for a level without one.
        ({"DIALECTIC_LEVELS__high__MODEL_CONFIG__CONTEXT_WINDOW": "108192"}, True),
        ({"DIALECTIC_LEVELS__high__MODEL_CONFIG__CONTEXT_WINDOW": "108191"}, False),
        # The dream specialists have no input budget to check.
        ({"DREAM_DEDUCTION_MODEL_CONFIG__CONTEXT_WINDOW": "1"}, True),
    ],
)
def test_context_window_must_fit_the_input_budget_and_output_cap(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], fits: bool
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    if fits:
        AppSettings()
    else:
        with pytest.raises(ValidationError, match="has a context_window of"):
            AppSettings()


def test_context_window_checks_the_fallback_with_the_primary_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUMMARY_MODEL_CONFIG__FALLBACK__TRANSPORT", "openai")
    monkeypatch.setenv("SUMMARY_MODEL_CONFIG__FALLBACK__MODEL", "gpt-5-nano")
    monkeypatch.setenv("SUMMARY_MODEL_CONFIG__FALLBACK__CONTEXT_WINDOW", "100000")
    # The fallback's own max_output_tokens isn't sent, so it doesn't count.
    monkeypatch.setenv("SUMMARY_MODEL_CONFIG__FALLBACK__MAX_OUTPUT_TOKENS", "100")
    with pytest.raises(ValidationError) as excinfo:
        AppSettings()
    message = str(excinfo.value)
    assert (
        "SUMMARY.MODEL_CONFIG.fallback: openai/gpt-5-nano has a context_window of "
        + "100000, but the input budget (100000) plus the output cap (6000) is "
        + "106000"
    ) in message


def test_context_window_reports_every_slot_that_does_not_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DERIVER_MODEL_CONFIG__CONTEXT_WINDOW", "1000")
    monkeypatch.setenv("DIALECTIC_LEVELS__low__MODEL_CONFIG__CONTEXT_WINDOW", "1000")
    with pytest.raises(ValidationError) as excinfo:
        AppSettings()
    message = str(excinfo.value)
    assert "DERIVER.MODEL_CONFIG:" in message
    assert "DIALECTIC.LEVELS.low.MODEL_CONFIG:" in message


def test_context_window_of_an_entry_applies_where_it_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small = {**NANO.model_dump(exclude_none=True), "context_window": 50000}
    monkeypatch.setenv("DERIVER_MODEL_CONFIG__MODEL_REF", "nano")
    # Deriver: 25000 + the entry's max_output_tokens (8000) fits in 50000.
    loaded = AppSettings(MODELS={"nano": small})  # pyright: ignore[reportArgumentType]
    assert loaded.DERIVER.MODEL_CONFIG.context_window == 50000

    monkeypatch.setenv("SUMMARY_MODEL_CONFIG__MODEL_REF", "nano")
    with pytest.raises(
        ValidationError, match="SUMMARY.MODEL_CONFIG: openai/gpt-5-nano"
    ):
        AppSettings(MODELS={"nano": small})  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("value", ["0", "-5"])
def test_context_window_must_be_positive(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("DERIVER_MODEL_CONFIG__CONTEXT_WINDOW", value)
    with pytest.raises(ValidationError, match="greater than 0"):
        AppSettings()
