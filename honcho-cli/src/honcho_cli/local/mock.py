"""Build a complete local-provider preset without importing the Honcho server."""

from __future__ import annotations

import json
from typing import Any

from honcho_cli.local.docker import DockerError

MODEL_SECTIONS = ("LLM", "EMBEDDING", "DERIVER", "DIALECTIC", "SUMMARY", "DREAM")
PRESET_SECTIONS = (*MODEL_SECTIONS, "TELEMETRY", "SENTRY", "VECTOR_STORE")
FAKE_KEY = "honcho-local-mock"


def _value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _key(data: dict[str, Any], name: str) -> str:
    return next((key for key in data if key.lower() == name.lower()), name)


def _set(data: dict[str, Any], path: list[str], value: Any) -> None:
    key = _key(data, path[0])
    if len(path) == 1:
        data[key] = value
        return
    if not isinstance(data.get(key), dict):
        data[key] = {}
    _set(data[key], path[1:], value)


def _section(env: dict[str, str], section: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in sorted(env, key=lambda key: (key.count("__"), len(key))):
        upper = key.upper()
        if upper == section:
            raw = _value(env[key])
            if not isinstance(raw, dict):
                raise DockerError(
                    "INVALID_CONFIG", f"{section} must contain a JSON object."
                )
            result.update(raw)
        elif upper.startswith(section + "_"):
            _set(result, key[len(section) :].lstrip("_").split("__"), _value(env[key]))
    return result


def environment(env: dict[str, str], base_url: str) -> dict[str, str]:
    """Replace every model/credential override, including nested fallbacks.

    Whole-section JSON has init precedence in the server's nested settings.
    Remove all competing spellings first so nested env vars cannot escape it.
    Non-model controls (workers, batch sizes, token limits) are preserved.
    """
    sections = {name: _section(env, name) for name in PRESET_SECTIONS}
    result = {
        key: value
        for key, value in env.items()
        if not any(
            key.upper() == section or key.upper().startswith(section + "_")
            for section in PRESET_SECTIONS
        )
        and not key.upper().startswith("LANGFUSE_")
    }
    overrides = {
        "api_key": FAKE_KEY,
        "api_key_env": None,
        "base_url": base_url,
        "provider_params": {},
    }

    def model() -> dict[str, Any]:
        return {
            "transport": "openai",
            "model": "mock-model",
            "overrides": overrides,
            "fallback": {
                "transport": "openai",
                "model": "mock-fallback",
                "overrides": overrides,
            },
        }

    sections["LLM"] = {
        "OPENAI_API_KEY": FAKE_KEY,
        "OPENAI_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": None,
        "GEMINI_API_KEY": None,
        "ANTHROPIC_BASE_URL": None,
        "GEMINI_BASE_URL": None,
    }
    for section, field in (
        ("DERIVER", "MODEL_CONFIG"),
        ("SUMMARY", "MODEL_CONFIG"),
        ("DREAM", "DEDUCTION_MODEL_CONFIG"),
        ("DREAM", "INDUCTION_MODEL_CONFIG"),
    ):
        _set(sections[section], [field], model())
    _set(
        sections["EMBEDDING"],
        ["MODEL_CONFIG"],
        {"transport": "openai", "model": "mock-embedding", "overrides": overrides},
    )
    for level in ("minimal", "low", "medium", "high", "max"):
        _set(sections["DIALECTIC"], ["LEVELS", level, "MODEL_CONFIG"], model())
    sections["TELEMETRY"] = {"ENABLED": False, "ENDPOINT": None}
    sections["SENTRY"] = {"ENABLED": False, "DSN": None}
    _set(sections["VECTOR_STORE"], ["TYPE"], "pgvector")
    for section, settings in sections.items():
        result[section] = json.dumps(settings)
    result.update(
        LANGFUSE_PUBLIC_KEY="",
        LANGFUSE_SECRET_KEY="",
        LANGFUSE_HOST="",
        PYTHON_DOTENV_DISABLED="1",
        HONCHO_CONFIG_TOML_DISABLED="1",
    )
    return result


# Run only in the selected server interpreter/image, after applying the preset.
# Fail closed if a new model path defaults to an external transport/credential.
VALIDATE = """
from src.config import settings
def check(value):
    if isinstance(value, dict):
        if 'transport' in value and 'model' in value:
            override = value.get('overrides', {})
            assert value['transport'] == 'openai', 'mock transport escaped'
            assert override.get('api_key') == 'honcho-local-mock', 'mock credential escaped'
            assert override.get('base_url') == settings.LLM.OPENAI_BASE_URL, 'mock endpoint escaped'
            assert not override.get('api_key_env'), 'mock credential indirection escaped'
        for child in value.values():
            check(child)
    elif isinstance(value, list):
        for child in value:
            check(child)
check(settings.model_dump())
assert not settings.TELEMETRY.ENABLED
assert not settings.SENTRY.ENABLED
assert not settings.LANGFUSE_PUBLIC_KEY
print('Mock provider configuration validated')
"""
