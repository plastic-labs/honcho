"""LLM provider backends.

Intentionally empty: importing this package must not pull in every provider
SDK. ``registry.py`` imports the submodule for the configured provider only,
so a single-provider deployment never loads the others (google-genai alone
costs ~32MB RSS). Import the submodule directly:

    from src.llm.backends.anthropic import AnthropicBackend
"""
