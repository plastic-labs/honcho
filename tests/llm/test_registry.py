from src.llm.registry import KIMI_CODING_USER_AGENT, get_openai_override_client


def test_kimi_coding_override_client_uses_kimi_cli_user_agent() -> None:
    """Verify Kimi coding endpoint clients include the KimiCLI User-Agent."""
    get_openai_override_client.cache_clear()

    client = get_openai_override_client("https://api.kimi.com/coding/v1", "test-key")

    assert client.default_headers.get("User-Agent") == KIMI_CODING_USER_AGENT


def test_non_kimi_override_client_has_no_custom_user_agent() -> None:
    """Verify ordinary OpenAI-compatible endpoints do not get Kimi headers."""
    get_openai_override_client.cache_clear()

    client = get_openai_override_client("https://api.openai.com/v1", "test-key")

    assert client.default_headers.get("User-Agent") != KIMI_CODING_USER_AGENT


def test_kimi_user_agent_rejects_lookalike_urls() -> None:
    """Verify only api.kimi.com coding paths get the Kimi User-Agent."""
    get_openai_override_client.cache_clear()

    lookalike_urls = [
        "https://malicious-api.kimi.com/coding/v1",
        "https://evil.example/api.kimi.com/coding/v1",
        "https://api.kimi.com/coding-proxy/v1",
    ]

    for url in lookalike_urls:
        client = get_openai_override_client(url, "test-key")
        assert client.default_headers.get("User-Agent") != KIMI_CODING_USER_AGENT
