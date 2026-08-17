"""Wizard mapping: ``answers_to_env``."""

from __future__ import annotations

from honcho_cli.local.setup import DIALECTIC_LEVELS, SetupAnswers, answers_to_env


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
