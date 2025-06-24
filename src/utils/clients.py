from anthropic import AsyncAnthropic
from google import genai
from groq import Groq
from mirascope import Provider
from openai import AsyncOpenAI

from src.config import settings

anthropic = AsyncAnthropic(api_key=settings.LLM.ANTHROPIC_API_KEY)
openai_client = AsyncOpenAI(
    api_key=settings.LLM.OPENAI_API_KEY,
    base_url=settings.LLM.OPENAI_COMPATIBLE_BASE_URL,
)
google = genai.Client(api_key=settings.LLM.GEMINI_API_KEY)
groq = Groq(api_key=settings.LLM.GROQ_API_KEY)


clients: dict[Provider, AsyncAnthropic | AsyncOpenAI | genai.Client | Groq] = {
    "anthropic": anthropic,
    "openai": openai_client,
    "google": google,
    "groq": groq,
}
