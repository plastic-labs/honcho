from anthropic import AsyncAnthropic
from google import genai
from groq import Groq
from openai import AsyncOpenAI

from src.config import settings
from src.utils.types import Providers

clients: dict[Providers, AsyncAnthropic | AsyncOpenAI | genai.Client | Groq] = {}

if settings.LLM.ANTHROPIC_API_KEY:
    anthropic = AsyncAnthropic(api_key=settings.LLM.ANTHROPIC_API_KEY)
    clients["anthropic"] = anthropic

if settings.LLM.OPENAI_API_KEY:
    openai_client = AsyncOpenAI(
        api_key=settings.LLM.OPENAI_API_KEY,
    )
    clients["openai"] = openai_client

if settings.LLM.OPENAI_COMPATIBLE_BASE_URL:
    clients["custom"] = AsyncOpenAI(
        api_key=settings.LLM.OPENAI_COMPATIBLE_API_KEY,
        base_url=settings.LLM.OPENAI_COMPATIBLE_BASE_URL,
    )

if settings.LLM.GEMINI_API_KEY:
    google = genai.Client(api_key=settings.LLM.GEMINI_API_KEY)
    clients["google"] = google

if settings.LLM.GROQ_API_KEY:
    groq = Groq(api_key=settings.LLM.GROQ_API_KEY)
    clients["groq"] = groq
