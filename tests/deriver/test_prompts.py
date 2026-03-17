from src.deriver.prompts import (
    estimate_deriver_prompt_tokens,
    estimate_minimal_deriver_prompt_tokens,
    minimal_deriver_prompt,
)


class TestMinimalDeriverPrompt:
    def test_includes_custom_instructions_section_when_present(self) -> None:
        prompt = minimal_deriver_prompt(
            peer_id="alice",
            messages="alice: hello",
            custom_instructions="Focus on durable preferences only.",
        )

        assert "CUSTOM INSTRUCTIONS:" in prompt
        assert "Focus on durable preferences only." in prompt

    def test_omits_custom_instructions_section_when_absent(self) -> None:
        prompt = minimal_deriver_prompt(peer_id="alice", messages="alice: hello")

        assert "CUSTOM INSTRUCTIONS:" not in prompt

    def test_custom_instructions_increase_prompt_token_estimate(self) -> None:
        base_tokens = estimate_minimal_deriver_prompt_tokens()
        custom_tokens = estimate_deriver_prompt_tokens(
            "Focus on durable preferences only."
        )

        assert custom_tokens > base_tokens
