"""Interactive ``honcho start --setup`` wizard.

Writes curated LLM/feature overrides for the local stack. Secrets and knobs
go to the profile ``.env`` (env wins over ``config.toml``). Prompts are TTY
only — the start command rejects ``--setup`` in JSON / non-TTY mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console

from honcho_cli.local.env import is_placeholder_key, read_env_file
from honcho_cli.output import print_error

SETUP_MODES = ("basic", "advanced")
DIALECTIC_LEVELS = ("minimal", "low", "medium", "high", "max")
PROVIDERS = ("openai", "anthropic", "gemini", "openai-compatible")
EMBEDDING_TRANSPORTS = ("openai", "gemini")

DEFAULT_CHAT_MODELS = {
    "openai": "gpt-5.4-mini",
    "anthropic": "claude-haiku-4-5",
    "gemini": "gemini-2.5-flash",
    "openai-compatible": "gpt-5.4-mini",
}
DEFAULT_EMBEDDING = {
    "openai": ("text-embedding-3-small", 1536),
    "gemini": ("gemini-embedding-001", 3072),
}

_CHAT_PREFIXES = (
    "DERIVER_MODEL_CONFIG",
    "SUMMARY_MODEL_CONFIG",
    "DREAM_DEDUCTION_MODEL_CONFIG",
    "DREAM_INDUCTION_MODEL_CONFIG",
    *(f"DIALECTIC_LEVELS__{level}__MODEL_CONFIG" for level in DIALECTIC_LEVELS),
)

_PROVIDER_KEY_ENV = {
    "openai": "LLM_OPENAI_API_KEY",
    "openai-compatible": "LLM_OPENAI_API_KEY",
    "anthropic": "LLM_ANTHROPIC_API_KEY",
    "gemini": "LLM_GEMINI_API_KEY",
}

_console = Console(stderr=True)


@dataclass(frozen=True)
class SetupAnswers:
    """Curated knobs collected by the wizard (or tests)."""

    mode: str
    provider: str
    api_key: str
    chat_model: str
    base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_key_transport: str | None = None
    embedding_transport: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    deriver_model: str | None = None
    dialectic_model: str | None = None
    dreams_enabled: bool | None = None
    flush_enabled: bool | None = None


def transport_of(provider: str) -> str:
    """Honcho ``MODEL_CONFIG.transport`` for a wizard provider id."""
    return "openai" if provider == "openai-compatible" else provider


def answers_to_env(answers: SetupAnswers) -> dict[str, str]:
    """Map wizard answers to Honcho env overrides."""
    transport = transport_of(answers.provider)
    env: dict[str, str] = {}

    env[_PROVIDER_KEY_ENV[answers.provider]] = answers.api_key
    if answers.base_url:
        env["LLM_OPENAI_BASE_URL"] = answers.base_url

    if answers.embedding_api_key and answers.embedding_key_transport:
        embed_key = (
            "LLM_OPENAI_API_KEY"
            if answers.embedding_key_transport == "openai"
            else "LLM_GEMINI_API_KEY"
        )
        env[embed_key] = answers.embedding_api_key

    for prefix in _CHAT_PREFIXES:
        env[f"{prefix}__TRANSPORT"] = transport
        env[f"{prefix}__MODEL"] = answers.chat_model

    if answers.deriver_model:
        env["DERIVER_MODEL_CONFIG__TRANSPORT"] = transport
        env["DERIVER_MODEL_CONFIG__MODEL"] = answers.deriver_model

    if answers.dialectic_model:
        for level in DIALECTIC_LEVELS:
            env[f"DIALECTIC_LEVELS__{level}__MODEL_CONFIG__TRANSPORT"] = transport
            env[f"DIALECTIC_LEVELS__{level}__MODEL_CONFIG__MODEL"] = (
                answers.dialectic_model
            )

    if answers.embedding_transport:
        model = (
            answers.embedding_model
            or DEFAULT_EMBEDDING[answers.embedding_transport][0]
        )
        env["EMBEDDING_MODEL_CONFIG__TRANSPORT"] = answers.embedding_transport
        env["EMBEDDING_MODEL_CONFIG__MODEL"] = model
        dims = answers.embedding_dimensions
        if dims is None:
            dims = DEFAULT_EMBEDDING[answers.embedding_transport][1]
        env["EMBEDDING_VECTOR_DIMENSIONS"] = str(dims)
    elif answers.embedding_key_transport == "gemini":
        # Basic + Anthropic chat: Gemini embed key is unused unless we switch.
        model, dims = DEFAULT_EMBEDDING["gemini"]
        env["EMBEDDING_MODEL_CONFIG__TRANSPORT"] = "gemini"
        env["EMBEDDING_MODEL_CONFIG__MODEL"] = model
        env["EMBEDDING_VECTOR_DIMENSIONS"] = str(dims)

    if answers.dreams_enabled is not None:
        env["DREAM_ENABLED"] = "true" if answers.dreams_enabled else "false"
    if answers.flush_enabled is not None:
        env["DERIVER_FLUSH_ENABLED"] = "true" if answers.flush_enabled else "false"
    return env


def answers_drop_keys(answers: SetupAnswers) -> tuple[str, ...]:
    """Keys to remove so a previous wizard run cannot leak into this one."""
    if answers.provider == "openai-compatible":
        return ()
    return ("LLM_OPENAI_BASE_URL",)


def openai_key_for_managed(answers: SetupAnswers) -> str:
    """Value for the managed ``LLM_OPENAI_API_KEY`` line."""
    if answers.provider in ("openai", "openai-compatible"):
        return answers.api_key
    if answers.embedding_key_transport == "openai" and answers.embedding_api_key:
        return answers.embedding_api_key
    if answers.embedding_transport == "openai" and answers.embedding_api_key:
        return answers.embedding_api_key
    return ""


def run_setup(
    mode: str,
    env_path: Path,
    *,
    llm_api_key_flag: str | None = None,
) -> SetupAnswers:
    """Prompt for ``basic`` or ``advanced`` knobs. Enter keeps the default."""
    env = read_env_file(env_path)
    _console.print()
    _console.print(
        "  [dim]Configure the local stack. Press Enter to keep the default.[/dim]"
    )
    _console.print(
        "  [dim]These values go in .env (they override config.toml).[/dim]"
    )
    _console.print()

    inferred = infer_provider(env)
    provider = _choose(
        "LLM provider",
        [
            ("openai", "OpenAI"),
            ("anthropic", "Anthropic"),
            ("gemini", "Gemini"),
            ("openai-compatible", "OpenAI-compatible (OpenRouter, vLLM, Ollama, …)"),
        ],
        inferred if inferred in PROVIDERS else "openai",
    )

    base_url: str | None = None
    if provider == "openai-compatible":
        base_url = _prompt_text(
            "OpenAI-compatible base URL",
            env.get("LLM_OPENAI_BASE_URL") or "https://openrouter.ai/api/v1",
        )

    key_env = _PROVIDER_KEY_ENV[provider]
    current_key = env.get(key_env)
    if (
        provider in ("openai", "openai-compatible")
        and llm_api_key_flag
        and not is_placeholder_key(llm_api_key_flag)
    ):
        current_key = llm_api_key_flag
    api_key = _prompt_secret("API key", current_key)

    same_provider = inferred == provider
    chat_default = (
        env.get("DERIVER_MODEL_CONFIG__MODEL")
        if same_provider and env.get("DERIVER_MODEL_CONFIG__MODEL")
        else DEFAULT_CHAT_MODELS[provider]
    )
    chat_model = _prompt_text("Chat model (deriver, dialectic, summary, dream)", chat_default)

    embedding_api_key: str | None = None
    embedding_key_transport: str | None = None
    embedding_transport: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    deriver_model: str | None = None
    dialectic_model: str | None = None
    dreams_enabled: bool | None = None
    flush_enabled: bool | None = None

    if mode == "advanced":
        embedding_transport = _choose(
            "Embedding provider",
            [("openai", "OpenAI"), ("gemini", "Gemini")],
            _default_embedding_transport(provider, env),
        )
        embed_defaults = DEFAULT_EMBEDDING[embedding_transport]
        current_embed_model = env.get("EMBEDDING_MODEL_CONFIG__MODEL")
        same_embed = env.get("EMBEDDING_MODEL_CONFIG__TRANSPORT") == embedding_transport
        embedding_model = _prompt_text(
            "Embedding model",
            current_embed_model if same_embed and current_embed_model else embed_defaults[0],
        )
        dim_default = env.get("EMBEDDING_VECTOR_DIMENSIONS") or str(embed_defaults[1])
        embedding_dimensions = _prompt_int("Embedding dimensions", int(dim_default))
        embedding_key_transport, embedding_api_key = _embedding_key_if_needed(
            provider, embedding_transport, env
        )
        deriver_model = _prompt_text("Deriver model", chat_model)
        dialectic_model = _prompt_text("Dialectic model (all reasoning levels)", chat_model)
        dreams_enabled = _choose_bool(
            "Dreams (periodic deeper reasoning)",
            _env_bool(env.get("DREAM_ENABLED"), default=True),
        )
        flush_enabled = _choose_bool(
            "Snappy local deriver (flush work immediately, skip batching)",
            _env_bool(env.get("DERIVER_FLUSH_ENABLED"), default=False),
        )
    elif provider == "anthropic":
        embedding_key_transport = _choose(
            "Embeddings (Anthropic has none — pick a provider)",
            [("openai", "OpenAI"), ("gemini", "Gemini")],
            "openai",
        )
        embed_key_env = _PROVIDER_KEY_ENV[
            "openai" if embedding_key_transport == "openai" else "gemini"
        ]
        embedding_api_key = _prompt_secret("Embedding API key", env.get(embed_key_env))

    _console.print()
    return SetupAnswers(
        mode=mode,
        provider=provider,
        api_key=api_key,
        chat_model=chat_model,
        base_url=base_url,
        embedding_api_key=embedding_api_key,
        embedding_key_transport=embedding_key_transport,
        embedding_transport=embedding_transport,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        deriver_model=deriver_model,
        dialectic_model=dialectic_model,
        dreams_enabled=dreams_enabled,
        flush_enabled=flush_enabled,
    )


def infer_provider(env: dict[str, str]) -> str:
    """Best-effort provider from an existing profile ``.env``."""
    if env.get("LLM_OPENAI_BASE_URL"):
        return "openai-compatible"
    transport = env.get("DERIVER_MODEL_CONFIG__TRANSPORT")
    if transport in ("anthropic", "gemini", "openai"):
        return transport
    if env.get("LLM_ANTHROPIC_API_KEY") and not env.get("LLM_OPENAI_API_KEY"):
        return "anthropic"
    if env.get("LLM_GEMINI_API_KEY") and not env.get("LLM_OPENAI_API_KEY"):
        return "gemini"
    return "openai"


def _default_embedding_transport(provider: str, env: dict[str, str]) -> str:
    current = env.get("EMBEDDING_MODEL_CONFIG__TRANSPORT")
    if current in EMBEDDING_TRANSPORTS:
        return current
    if provider == "gemini":
        return "gemini"
    return "openai"


def _embedding_key_if_needed(
    chat_provider: str,
    embed_transport: str,
    env: dict[str, str],
) -> tuple[str | None, str | None]:
    """Prompt for an embedding key when the chat provider cannot supply it."""
    chat_transport = transport_of(chat_provider)
    if embed_transport == chat_transport or (
        chat_provider == "openai-compatible" and embed_transport == "openai"
    ):
        return None, None
    key_env = _PROVIDER_KEY_ENV[embed_transport]
    key = _prompt_secret(f"{embed_transport} embedding API key", env.get(key_env))
    return embed_transport, key


def _choose(label: str, options: list[tuple[str, str]], default: str) -> str:
    ids = [item[0] for item in options]
    default_idx = ids.index(default) + 1 if default in ids else 1
    _console.print(f"  [dim]{label}[/dim]")
    for i, (_oid, desc) in enumerate(options, 1):
        _console.print(f"  [dim]({i})[/dim] {desc}")
    raw = typer.prompt(
        "  Choice",
        default=str(default_idx),
        show_default=True,
        prompt_suffix=": ",
    ).strip()
    try:
        idx = int(raw)
    except ValueError:
        if raw in ids:
            return raw
        return options[default_idx - 1][0]
    if 1 <= idx <= len(options):
        return options[idx - 1][0]
    return options[default_idx - 1][0]


def _choose_bool(label: str, default: bool) -> bool:
    return (
        _choose(label, [("true", "On"), ("false", "Off")], "true" if default else "false")
        == "true"
    )


def _prompt_text(label: str, default: str) -> str:
    raw = typer.prompt(
        f"  {label}",
        default=default,
        show_default=True,
        prompt_suffix=": ",
    ).strip()
    return raw or default


def _prompt_int(label: str, default: int) -> int:
    while True:
        raw = typer.prompt(
            f"  {label}",
            default=str(default),
            show_default=True,
            prompt_suffix=": ",
        ).strip()
        try:
            value = int(raw)
        except ValueError:
            _console.print("  [red]Enter an integer[/red]")
            continue
        if value > 0:
            return value
        _console.print("  [red]Must be a positive integer[/red]")


def _prompt_secret(label: str, current: str | None) -> str:
    if current and not is_placeholder_key(current):
        _console.print(f"  [dim]Current {label}: {_redact(current)}[/dim]")
        _console.print("  [dim](1)[/dim] Keep current key")
        _console.print("  [dim](2)[/dim] Enter a new key")
        choice = typer.prompt(
            "  Choice", default="1", show_default=True, prompt_suffix=": "
        ).strip()
        if choice != "2":
            return current
    _console.print(f"  [dim]{label}[/dim]")
    raw = typer.prompt(
        f"  {label}",
        default="",
        show_default=False,
        hide_input=True,
        prompt_suffix=": ",
    ).strip()
    if not raw or is_placeholder_key(raw):
        print_error(
            "MISSING_LLM_KEY",
            f"{label} is required.",
        )
        raise typer.Exit(1)
    return raw


def _redact(key: str) -> str:
    if len(key) <= 4:
        return "***"
    return "***" + key[-4:]


def _env_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")
