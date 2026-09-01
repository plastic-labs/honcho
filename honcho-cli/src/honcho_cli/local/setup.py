"""Interactive ``honcho start --setup`` wizard.

Writes curated LLM/feature overrides for the local stack. Secrets and knobs
go to the profile ``.env`` (env wins over ``config.toml``). Prompts are TTY
only — the start command rejects ``--setup`` in JSON / non-TTY mode.
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console

from honcho_cli.local.env import is_placeholder_key, read_env_file, settings_from_environ
from honcho_cli.output import print_error

SETUP_MODES = ("basic", "advanced")
DIALECTIC_LEVELS = ("minimal", "low", "medium", "high", "max")
PROVIDERS = ("openai", "anthropic", "gemini", "openai-compatible")
EMBEDDING_TRANSPORTS = ("openai", "gemini")

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
class TomlSetupDefaults:
    """Model/feature defaults copied from the image ``config.toml``.

    Honcho only ships OpenAI chat/embedding defaults. Other providers have
    no suggested model in that file — the wizard does not invent one.
    """

    chat_transport: str | None = None
    chat_model: str | None = None
    embed_transport: str | None = None
    embed_model: str | None = None
    embed_dims: int | None = None
    dreams_enabled: bool | None = None
    flush_enabled: bool | None = None


def load_toml_setup_defaults(path: Path | None) -> TomlSetupDefaults:
    """Read prompt defaults from the profile ``config.toml`` (image-aligned)."""
    if path is None or not path.is_file():
        return TomlSetupDefaults()
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        deriver = data.get("deriver") or {}
        chat = deriver.get("model_config") or {}
        embedding = data.get("embedding") or {}
        embed = embedding.get("model_config") or {}
        dream = data.get("dream") or {}
        dims = embedding.get("VECTOR_DIMENSIONS")
        return TomlSetupDefaults(
            chat_transport=chat.get("transport"),
            chat_model=chat.get("model"),
            embed_transport=embed.get("transport"),
            embed_model=embed.get("model"),
            embed_dims=dims if isinstance(dims, int) and dims > 0 else None,
            dreams_enabled=dream.get("ENABLED"),
            flush_enabled=deriver.get("FLUSH_ENABLED"),
        )
    except (OSError, tomllib.TOMLDecodeError, TypeError, AttributeError):
        return TomlSetupDefaults()


def chat_model_default(
    provider: str,
    env: dict[str, str],
    toml: TomlSetupDefaults,
    *,
    inferred: str | None = None,
) -> str:
    """Prefer a previous wizard choice, else the image toml when transports match."""
    if inferred is None:
        inferred = infer_provider(env)
    if inferred == provider:
        current = env.get("DERIVER_MODEL_CONFIG__MODEL")
        if current:
            return current
    if toml.chat_model and _provider_matches_transport(provider, toml.chat_transport):
        return toml.chat_model
    return ""


def _provider_matches_transport(provider: str, transport: str | None) -> bool:
    if not transport:
        return False
    return transport_of(provider) == transport


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
        # Embeddings do not inherit this URL; write it so OpenRouter/vLLM
        # keys are not sent to api.openai.com.
        if (answers.embedding_transport or "openai") == "openai":
            env["EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL"] = answers.base_url

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
        env["EMBEDDING_MODEL_CONFIG__TRANSPORT"] = answers.embedding_transport
        if answers.embedding_model:
            env["EMBEDDING_MODEL_CONFIG__MODEL"] = answers.embedding_model
        if answers.embedding_dimensions is not None:
            env["EMBEDDING_VECTOR_DIMENSIONS"] = str(answers.embedding_dimensions)
    elif answers.embedding_key_transport == "gemini":
        # Basic + Anthropic chat: a Gemini key is unused unless embeddings switch.
        env["EMBEDDING_MODEL_CONFIG__TRANSPORT"] = "gemini"

    if answers.dreams_enabled is not None:
        env["DREAM_ENABLED"] = "true" if answers.dreams_enabled else "false"
    if answers.flush_enabled is not None:
        env["DERIVER_FLUSH_ENABLED"] = "true" if answers.flush_enabled else "false"
    return env


def answers_drop_keys(answers: SetupAnswers) -> tuple[str, ...]:
    """Keys to remove so a previous wizard run cannot leak into this one."""
    drop: list[str] = []
    if answers.provider != "openai-compatible":
        drop.append("LLM_OPENAI_BASE_URL")
    embed_openai = (answers.embedding_transport or "openai") == "openai"
    if answers.provider != "openai-compatible" or not embed_openai:
        drop.append("EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL")
    return tuple(drop)


def run_setup(
    mode: str,
    env_path: Path,
    *,
    config_path: Path | None = None,
) -> SetupAnswers:
    """Prompt for ``basic`` or ``advanced`` knobs. Enter keeps the default."""
    env = read_env_file(env_path)
    env.update(settings_from_environ())
    defaults = load_toml_setup_defaults(config_path)
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
    api_key = _prompt_secret("API key", env.get(key_env))

    chat_default = chat_model_default(
        provider, env, defaults, inferred=inferred
    )
    chat_model = _prompt_text(
        "Chat model (deriver, dialectic, summary, dream)",
        chat_default,
        required=True,
    )

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
            _default_embedding_transport(provider, env, defaults),
        )
        same_embed = env.get("EMBEDDING_MODEL_CONFIG__TRANSPORT") == embedding_transport
        current_embed = env.get("EMBEDDING_MODEL_CONFIG__MODEL") if same_embed else None
        embed_from_toml = (
            defaults.embed_model
            if defaults.embed_transport == embedding_transport
            else None
        )
        embedding_model = (
            _prompt_text("Embedding model", current_embed or embed_from_toml or "")
            or None
        )
        dim_default = (
            int(env["EMBEDDING_VECTOR_DIMENSIONS"])
            if env.get("EMBEDDING_VECTOR_DIMENSIONS", "").isdigit()
            else (defaults.embed_dims or 1536)
        )
        embedding_dimensions = _prompt_int("Embedding dimensions", dim_default)
        embedding_key_transport, embedding_api_key = _embedding_key_if_needed(
            provider, embedding_transport, env
        )
        deriver_model = _prompt_text("Deriver model", chat_model)
        dialectic_model = _prompt_text("Dialectic model (all reasoning levels)", chat_model)
        dreams_enabled = _choose_bool(
            "Dreams (periodic deeper reasoning)",
            _env_bool(
                env.get("DREAM_ENABLED"),
                default=True if defaults.dreams_enabled is None else defaults.dreams_enabled,
            ),
        )
        flush_enabled = _choose_bool(
            "Snappy local deriver (flush work immediately, skip batching)",
            _env_bool(
                env.get("DERIVER_FLUSH_ENABLED"),
                default=False if defaults.flush_enabled is None else defaults.flush_enabled,
            ),
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


def _default_embedding_transport(
    provider: str, env: dict[str, str], defaults: TomlSetupDefaults
) -> str:
    current = env.get("EMBEDDING_MODEL_CONFIG__TRANSPORT")
    if current in EMBEDDING_TRANSPORTS:
        return current
    if defaults.embed_transport in EMBEDDING_TRANSPORTS:
        return defaults.embed_transport
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


def _prompt_text(label: str, default: str, *, required: bool = False) -> str:
    while True:
        raw = typer.prompt(
            f"  {label}",
            default=default,
            show_default=bool(default),
            prompt_suffix=": ",
        ).strip()
        value = raw or default
        if value or not required:
            return value
        _console.print("  [red]A model name is required[/red]")


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
    raw = _prompt_masked(f"  {label}: ").strip()
    if not raw or is_placeholder_key(raw):
        print_error(
            "MISSING_LLM_KEY",
            f"{label} is required.",
        )
        raise typer.Exit(1)
    return raw


def _prompt_masked(prompt: str) -> str:
    """Read a secret, echoing ``*`` per character so paste is visibly received."""
    stream = sys.stderr
    stream.write(prompt)
    stream.flush()
    chars: list[str] = []

    def _write(text: str) -> None:
        stream.write(text)
        stream.flush()

    def _feed(ch: str) -> bool:
        """Return True when input is complete."""
        if not ch or ch in ("\n", "\r", "\x04"):
            _write("\n")
            return True
        if ch in ("\x7f", "\x08"):
            if chars:
                chars.pop()
                _write("\b \b")
            return False
        if ch == "\x1b":
            return False
        if ch.isprintable():
            chars.append(ch)
            _write("*")
        return False

    if sys.platform == "win32":
        import msvcrt

        while True:
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                msvcrt.getwch()
                continue
            if _feed(ch):
                return "".join(chars)

    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                nxt = sys.stdin.read(1)
                if nxt == "[":
                    while True:
                        seq = sys.stdin.read(1)
                        if not seq or "@" <= seq <= "~":
                            break
                continue
            if _feed(ch):
                return "".join(chars)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return "".join(chars)


def _redact(key: str) -> str:
    if len(key) <= 4:
        return "***"
    return "***" + key[-4:]


def _env_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")
