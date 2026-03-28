from src.config import ModelConfig
from src.llm.capabilities import get_model_capabilities


def test_anthropic_capabilities() -> None:
    capabilities = get_model_capabilities(
        ModelConfig(model="anthropic/claude-haiku-4-5")
    )

    assert capabilities.transport == "provider_native"
    assert capabilities.history_format == "anthropic"
    assert capabilities.structured_output_mode == "repair_wrapper"
    assert capabilities.reasoning_mode == "budget"
    assert capabilities.cache_mode == "prefix"
    assert capabilities.cache_metrics_mode == "anthropic"
    assert capabilities.shared_reasoning_budget is False


def test_gemini_capabilities() -> None:
    capabilities = get_model_capabilities(ModelConfig(model="gemini/gemini-2.5-flash"))

    assert capabilities.history_format == "gemini"
    assert capabilities.structured_output_mode == "native"
    assert capabilities.reasoning_mode == "budget"
    assert capabilities.cache_mode == "gemini_cached_content"
    assert capabilities.cache_metrics_mode == "gemini"
    assert capabilities.shared_reasoning_budget is True


def test_openai_compatible_capabilities_are_conservative() -> None:
    capabilities = get_model_capabilities(
        ModelConfig(
            model="openai/my-local-model",
            transport="openai_compatible",
            api_key="test-key",
            base_url="http://localhost:8000/v1",
        )
    )

    assert capabilities.transport == "openai_compatible"
    assert capabilities.history_format == "openai"
    assert capabilities.structured_output_mode == "repair_wrapper"
    assert capabilities.reasoning_mode == "none"
    assert capabilities.cache_mode == "none"
