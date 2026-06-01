from src.llm.registry import get_openai_override_client


def test_kimi_coding_override_client_uses_kimi_cli_user_agent() -> None:
    get_openai_override_client.cache_clear()

    client = get_openai_override_client("https://api.kimi.com/coding/v1", "test-key")

    assert client.default_headers["User-Agent"] == "KimiCLI/1.5"
