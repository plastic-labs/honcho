import pytest
from dotenv import load_dotenv

from src.deriver.prompts import minimal_deriver_prompt
from src.dreamer.specialists import DeductionSpecialist, InductionSpecialist
from src.utils.summarizer import long_summary_prompt, short_summary_prompt

load_dotenv(override=True)

# Estonian parliament debate excerpt
ESTONIAN_MESSAGES = """\
Kristen-Michal: Austatud Riigikogu, täna arutame eelnõu 603, mis käsitleb välismaalaste \
viisaotsuse kohtus vaidlustamise õigust. Praegune kaheastmeline vaidemenetlus ei vasta \
Euroopa Kohtu 2017. aasta otsusele kohtuasjas C-403/16.
Lauri-Laats: Palun täpsustage, milline on riigilõivu suurus välismaalasele halduskohtusse \
pöördumisel võrreldes Eesti kodanikuga?
Kristen-Michal: Riigilõiv on 280 eurot, samas kui Eesti kodaniku jaoks on see tavaliselt \
20 eurot. Justiitsministeerium selgitas, et see tuleneb kolmest komponendist: kulupõhisus, \
mõjutusmeede ja pakutav hüve.\
"""

ESTONIAN_CHARS = set("õäöüÕÄÖÜ")


def _contains_estonian(text: str) -> bool:
    return bool(ESTONIAN_CHARS & set(text))


async def _call_llm(prompt: str, max_tokens: int = 500) -> str:
    """Call the configured LLM directly, bypassing Honcho middleware."""
    from openai import AsyncOpenAI  # type: ignore[import]
    from src.config import settings

    client = AsyncOpenAI(
        base_url=settings.LLM.OPENAI_COMPATIBLE_BASE_URL,
        api_key=settings.LLM.OPENAI_COMPATIBLE_API_KEY,
    )
    response = await client.chat.completions.create(
        model=settings.DERIVER.MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


# --- Prompt content tests (fast, no LLM) ---

def test_deriver_prompt_includes_language_preservation():
    prompt = minimal_deriver_prompt(peer_id="alice", messages="alice: Tere maailm")
    assert "LANGUAGE PRESERVATION" in prompt
    assert "Do NOT translate observations" in prompt


def test_summary_prompts_include_language_preservation():
    short_prompt = short_summary_prompt(
        formatted_messages="alice: Tere maailm",
        output_words=100,
        previous_summary_text="",
    )
    long_prompt = long_summary_prompt(
        formatted_messages="alice: Tere maailm",
        output_words=300,
        previous_summary_text="",
    )

    assert "LANGUAGE PRESERVATION" in short_prompt
    assert "Do NOT translate the conversation" in short_prompt
    assert "LANGUAGE PRESERVATION" in long_prompt
    assert "Do NOT translate the conversation" in long_prompt


def test_dream_specialist_prompts_include_language_preservation():
    deduction_prompt = DeductionSpecialist().build_system_prompt("alice")
    induction_prompt = InductionSpecialist().build_system_prompt("alice")

    assert "LANGUAGE PRESERVATION" in deduction_prompt
    assert "Do NOT translate created observations or peer card entries" in deduction_prompt
    assert "LANGUAGE PRESERVATION" in induction_prompt
    assert "Do NOT translate created observations or peer card entries" in induction_prompt


# --- LLM integration tests (make real API calls) ---

@pytest.mark.llm
async def test_deriver_preserves_estonian_language():
    prompt = minimal_deriver_prompt(
        peer_id="Kristen-Michal",
        messages=ESTONIAN_MESSAGES,
    )
    result = await _call_llm(prompt)
    assert result, "LLM returned empty response"
    assert _contains_estonian(result), (
        f"Expected Estonian characters in deriver output, got:\n{result}"
    )


@pytest.mark.llm
async def test_short_summary_preserves_estonian_language():
    from src.utils.tokens import estimate_tokens
    output_words = int(min(estimate_tokens(ESTONIAN_MESSAGES), 1000) * 0.75)
    prompt = short_summary_prompt(
        formatted_messages=ESTONIAN_MESSAGES,
        output_words=output_words,
        previous_summary_text="",
    )
    result = await _call_llm(prompt, max_tokens=1000)
    assert result, "LLM returned empty response"
    assert _contains_estonian(result), (
        f"Expected Estonian characters in short summary output, got:\n{result}"
    )


@pytest.mark.llm
async def test_long_summary_preserves_estonian_language():
    output_words = int(4000 * 0.75)
    prompt = long_summary_prompt(
        formatted_messages=ESTONIAN_MESSAGES,
        output_words=output_words,
        previous_summary_text="",
    )
    result = await _call_llm(prompt, max_tokens=2000)
    assert result, "LLM returned empty response"
    assert _contains_estonian(result), (
        f"Expected Estonian characters in long summary output, got:\n{result}"
    )
