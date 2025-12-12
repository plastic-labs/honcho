from typing import Literal

SupportedProviders = Literal["anthropic", "openai", "google", "groq", "custom", "vllm"]
TaskType = Literal["webhook", "summary", "representation", "dream", "deletion"]
DocumentLevel = Literal["explicit", "deductive", "inductive", "vignette"]
