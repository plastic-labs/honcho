"""Wizard mapping: ``answers_to_env`` and image-toml defaults."""

from __future__ import annotations

from honcho_cli.local.setup import (
    DIALECTIC_LEVELS,
    SetupAnswers,
    answers_drop_keys,
    answers_to_env,
    chat_model_default,
    load_toml_setup_defaults,
)


def test_basic_openai_applies_chat_model_everywhere():
    env = answers_to_env(
        SetupAnswers(
            mode="basic",
            provider="openai",
            api_key="sk-test",
            chat_model="gpt-test",
        )
    )
    assert env["LLM_OPENAI_API_KEY"] == "sk-test"
    assert env["DERIVER_MODEL_CONFIG__MODEL"] == "gpt-test"
    assert env["SUMMARY_MODEL_CONFIG__MODEL"] == "gpt-test"
    for level in DIALECTIC_LEVELS:
        assert env[f"DIALECTIC_LEVELS__{level}__MODEL_CONFIG__MODEL"] == "gpt-test"
    assert "DREAM_ENABLED" not in env
    assert "EMBEDDING_MODEL_CONFIG__MODEL" not in env


def test_basic_anthropic_keeps_openai_embeddings_default():
    env = answers_to_env(
        SetupAnswers(
            mode="basic",
            provider="anthropic",
            api_key="sk-ant",
            chat_model="claude-haiku-4-5",
            embedding_api_key="sk-embed",
            embedding_key_transport="openai",
        )
    )
    assert env["LLM_ANTHROPIC_API_KEY"] == "sk-ant"
    assert env["LLM_OPENAI_API_KEY"] == "sk-embed"
    assert env["DERIVER_MODEL_CONFIG__TRANSPORT"] == "anthropic"
    assert "EMBEDDING_MODEL_CONFIG__TRANSPORT" not in env


def test_openai_compatible_copies_base_url_to_embeddings():
    env = answers_to_env(
        SetupAnswers(
            mode="basic",
            provider="openai-compatible",
            api_key="sk-or-test",
            chat_model="gpt-test",
            base_url="https://openrouter.ai/api/v1",
        )
    )
    assert env["LLM_OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert (
        env["EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL"]
        == "https://openrouter.ai/api/v1"
    )


def test_leaving_openai_compatible_drops_proxy_urls():
    dropped = answers_drop_keys(
        SetupAnswers(
            mode="basic", provider="openai", api_key="sk", chat_model="gpt-test"
        )
    )
    assert "LLM_OPENAI_BASE_URL" in dropped
    assert "EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL" in dropped


def test_chat_default_comes_from_image_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "[deriver.model_config]\n"
        'transport = "openai"\n'
        'model = "gpt-from-image"\n'
    )
    defaults = load_toml_setup_defaults(path)
    assert defaults.chat_model == "gpt-from-image"
    assert chat_model_default("openai", {}, defaults) == "gpt-from-image"
    assert chat_model_default("openai-compatible", {}, defaults) == "gpt-from-image"
    assert chat_model_default("anthropic", {}, defaults) == ""
    assert chat_model_default(
        "openai",
        {"DERIVER_MODEL_CONFIG__MODEL": "gpt-from-env"},
        defaults,
        inferred="openai",
    ) == "gpt-from-env"
