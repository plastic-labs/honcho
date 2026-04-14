from src.deriver.prompts import minimal_deriver_prompt


def test_minimal_deriver_prompt_explicitly_requires_content_key() -> None:
    prompt = minimal_deriver_prompt(
        peer_id="Aubrey",
        messages="[2026-04-14 04:00:00] Aubrey: I live in Wisconsin.",
    )

    assert "Each object must use the key `content`" in prompt
    assert "Do NOT use keys like `fact`" in prompt
    assert '{"explicit":[{"content":"Aubrey lives in Wisconsin"}]}' in prompt
