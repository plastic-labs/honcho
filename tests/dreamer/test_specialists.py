"""Tests for dreamer specialist prompts."""

from src.dreamer.specialists import DeductionSpecialist


class TestDeductionSpecialistPrompt:
    """Tests for DeductionSpecialist prompt content."""

    def test_deduction_specialist_prompt_mentions_supersession(self):
        """The deduction specialist prompt includes superseded_by instructions."""
        specialist = DeductionSpecialist()
        prompt: str = specialist.build_system_prompt(
            observed="test_peer",
            peer_card_enabled=False,
        )

        assert "superseded_by" in prompt
        assert "[id:" in prompt
        assert "FIRST: Create the updated deductive observation" in prompt
