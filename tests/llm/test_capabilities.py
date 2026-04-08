from src.config import ModelConfig
from src.llm.capabilities import get_model_capabilities


def test_anthropic_capabilities() -> None:
    capabilities = get_model_capabilities(
        ModelConfig(model="claude-haiku-4-5", transport="anthropic")
    )

    assert capabilities.transport == "anthropic"
    assert capabilities.history_format == "anthropic"
    assert capabilities.structured_output_mode == "repair_wrapper"
    assert capabilities.reasoning_mode == "budget"
    assert capabilities.cache_mode == "prefix"
    assert capabilities.cache_metrics_mode == "anthropic"
    assert capabilities.shared_reasoning_budget is False


def test_gemini_capabilities() -> None:
    capabilities = get_model_capabilities(
        ModelConfig(model="gemini-2.5-flash", transport="gemini")
    )

    assert capabilities.history_format == "gemini"
    assert capabilities.structured_output_mode == "native"
    assert capabilities.reasoning_mode == "budget"
    assert capabilities.cache_mode == "gemini_cached_content"
    assert capabilities.cache_metrics_mode == "gemini"
    assert capabilities.shared_reasoning_budget is True


def test_openai_capabilities_support_effort_reasoning_and_prefix_cache() -> None:
    capabilities = get_model_capabilities(
        ModelConfig(
            model="gpt-5",
            transport="openai",
        )
    )

    assert capabilities.transport == "openai"
    assert capabilities.history_format == "openai"
    assert capabilities.structured_output_mode == "native"
    assert capabilities.reasoning_mode == "effort"
    assert capabilities.cache_mode == "prefix"
