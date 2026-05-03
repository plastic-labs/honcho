from src.deriver.prompts import minimal_deriver_prompt


def test_minimal_deriver_prompt_requires_json_only_output() -> None:
    prompt = minimal_deriver_prompt(
        peer_id="alice",
        messages="[2026-05-03] alice: I enjoy hiking on weekends.",
    )

    assert "Return ONLY valid JSON" in prompt
    assert "No markdown" in prompt
    assert '"explicit"' in prompt
    assert '{"explicit":[]}' in prompt
