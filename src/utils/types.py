from typing import Literal

SupportedProviders = Literal["anthropic", "openai", "google", "groq", "custom", "vllm"]
TaskType = Literal["webhook", "summary", "representation", "dream"]
DocumentLevel = Literal["explicit", "implicit", "deductive"]
