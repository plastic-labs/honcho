from typing import Any

from anthropic import Anthropic
from google import genai
from groq import Groq
from mirascope import Provider
from openai import AsyncOpenAI

from src.config import settings

anthropic = Anthropic(api_key=settings.LLM.ANTHROPIC_API_KEY)
openai = AsyncOpenAI(api_key=settings.LLM.OPENAI_API_KEY)
google = genai.Client(api_key=settings.LLM.GEMINI_API_KEY)
groq = Groq(api_key=settings.LLM.GROQ_API_KEY)


clients: dict[Provider, Any] = {
    "anthropic": anthropic,
    "openai": openai,
    "google": google,
    "groq": groq,
}
