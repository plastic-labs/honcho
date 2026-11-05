"""Snapshot of how model configuration resolves, for before/after comparison.

Each case boots `src.config` in its own interpreter, the way a deployment does:
a clean environment (no `.env`), the case's env vars, and the case's
`config.toml` in the working directory. The probe records what every model
slot resolves to (the configured settings, the runtime `ModelConfig`, the
config chosen for the final retry attempt) and the capability row each
primary and fallback model gets, or the validation errors when config fails
to load. A capability sweep records `capabilities_for` over a grid of
transports and model names.

`model_config_snapshot.json` holds the output recorded before the model
catalog existed. `test_model_config_snapshot.py` replays every case and
requires identical output, so a config that doesn't use the catalog resolves
exactly as it did before. Regenerate the file only for an intended change:

    uv run python -m tests.llm.model_config_snapshot --write
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = Path(__file__).with_name("model_config_snapshot.json")

# Runs inside the case's interpreter. It prints one JSON document.
_PROBE = r"""
import json
import sys

from pydantic import ValidationError

def errors_of(exc):
    return [
        {"loc": [str(part) for part in err["loc"]], "msg": err["msg"]}
        for err in exc.errors()
    ]

try:
    from src import config
except ValidationError as exc:
    print(json.dumps({"load_error": errors_of(exc)}, sort_keys=True))
    sys.exit(0)

from dataclasses import asdict

from src.llm.capabilities import capabilities_for
from src.llm.runtime import select_model_config_for_attempt

s = config.settings
slots = {
    "deriver": s.DERIVER.MODEL_CONFIG,
    "summary": s.SUMMARY.MODEL_CONFIG,
    "dream.deduction": s.DREAM.DEDUCTION_MODEL_CONFIG,
    "dream.induction": s.DREAM.INDUCTION_MODEL_CONFIG,
    **{
        f"dialectic.{level}": level_settings.MODEL_CONFIG
        for level, level_settings in s.DIALECTIC.LEVELS.items()
    },
}

out = {"slots": {}, "dialectic_levels": {}}
for name, configured in slots.items():
    resolved = config.resolve_model_config(configured)
    final_attempt = select_model_config_for_attempt(
        resolved, attempt=3, retry_attempts=3
    )
    entry = {
        "configured": configured.model_dump(mode="json"),
        "resolved": resolved.model_dump(mode="json"),
        "final_attempt": final_attempt.model_dump(mode="json"),
        "capabilities": asdict(capabilities_for(resolved.transport, resolved.model)),
    }
    if resolved.fallback is not None:
        entry["fallback_capabilities"] = asdict(
            capabilities_for(resolved.fallback.transport, resolved.fallback.model)
        )
    out["slots"][name] = entry
for level, level_settings in s.DIALECTIC.LEVELS.items():
    out["dialectic_levels"][level] = level_settings.model_dump(
        mode="json", exclude={"MODEL_CONFIG"}
    )
out["dialectic_max_output_tokens"] = s.DIALECTIC.MAX_OUTPUT_TOKENS

if "--sweep" in sys.argv:
    models = [
        "claude-3-5-sonnet-20241022",
        "claude-opus-4-20250514",
        "claude-sonnet-4-5",
        "claude-opus-4-6",
        "claude-sonnet-4.6",
        "anthropic/claude-haiku-4-5",
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-haiku-5-5",
        "claude-fable-5-1",
        "us.anthropic.claude-sonnet-5-5-v1:0",
        "gpt-5.4-mini",
        "gpt-5-nano",
        "mistral-small-3.2-24b-instruct-2506",
        "mistralai/mistral-small-3.2-24b-instruct",
        "devstral-medium",
        "gemini-2.5-flash",
        "MiniMax-M2",
        "summarizer",
        "dialectic-high",
    ]
    out["capability_sweep"] = {
        f"{transport}:{model}": asdict(capabilities_for(transport, model))
        for transport in ("anthropic", "openai", "gemini")
        for model in models
    }

print(json.dumps(out, sort_keys=True))
"""

# name -> {"env": {...}, "toml": "..."}. The env is the case's whole
# environment apart from PATH/HOME and the switches that keep `.env` out.
CASES: dict[str, dict[str, Any]] = {
    "defaults": {},
    "env_partial_model": {"env": {"DERIVER_MODEL_CONFIG__MODEL": "gpt-4.1"}},
    "env_transport_switch_with_budget": {
        "env": {
            "DERIVER_MODEL_CONFIG__TRANSPORT": "anthropic",
            "DERIVER_MODEL_CONFIG__MODEL": "claude-sonnet-4-5",
            "DERIVER_MODEL_CONFIG__THINKING_BUDGET_TOKENS": "2048",
        }
    },
    "env_provider_shorthand": {
        "env": {"SUMMARY_MODEL_CONFIG__MODEL": "anthropic/claude-haiku-4-5"}
    },
    "env_reasoning_effort_alias": {
        "env": {
            "DERIVER_MODEL_CONFIG__MODEL": "gpt-5",
            "DERIVER_MODEL_CONFIG__REASONING_EFFORT": "high",
        }
    },
    "env_structured_output_mode": {
        "env": {"SUMMARY_MODEL_CONFIG__STRUCTURED_OUTPUT_MODE": "json_object"}
    },
    "env_fallback_full": {
        "env": {
            "DERIVER_MODEL_CONFIG__FALLBACK__MODEL": "gpt-5-nano",
            "DERIVER_MODEL_CONFIG__FALLBACK__TRANSPORT": "openai",
            "DERIVER_MODEL_CONFIG__FALLBACK__MAX_OUTPUT_TOKENS": "4000",
            "DERIVER_MODEL_CONFIG__FALLBACK__THINKING_EFFORT": "minimal",
            "DERIVER_MODEL_CONFIG__FALLBACK__STRUCTURED_OUTPUT_MODE": "json_object",
        }
    },
    "env_cross_family_fallback": {
        "env": {
            "SUMMARY_MODEL_CONFIG__TRANSPORT": "anthropic",
            "SUMMARY_MODEL_CONFIG__MODEL": "claude-haiku-5-5",
            "SUMMARY_MODEL_CONFIG__THINKING_EFFORT": "low",
            "SUMMARY_MODEL_CONFIG__FALLBACK__MODEL": "openai/gpt-5-nano",
            "SUMMARY_MODEL_CONFIG__FALLBACK__THINKING_EFFORT": "minimal",
            "SUMMARY_MODEL_CONFIG__FALLBACK__OVERRIDES__BASE_URL": (
                "https://fallback.example/v1"
            ),
            "SUMMARY_MODEL_CONFIG__FALLBACK__OVERRIDES__API_KEY": "sk-fallback",
        }
    },
    "env_overrides_and_provider_params": {
        "env": {
            "DREAM_DEDUCTION_MODEL_CONFIG__OVERRIDES__BASE_URL": (
                "https://proxy.example/v1"
            ),
            "DREAM_DEDUCTION_MODEL_CONFIG__OVERRIDES__API_KEY": "sk-proxy",
            "DREAM_DEDUCTION_MODEL_CONFIG__OVERRIDES__PROVIDER_PARAMS": json.dumps(
                {"extra_body": {"reasoning": {"enabled": True}}, "timeout": "45"}
            ),
            "DREAM_INDUCTION_MODEL_CONFIG__OVERRIDES__API_KEY_ENV": "CUSTOM_KEY",
            "CUSTOM_KEY": "sk-from-env",
        }
    },
    "env_whole_model_config_json": {
        "env": {
            "DREAM_INDUCTION_MODEL_CONFIG": json.dumps(
                {
                    "model": "claude-opus-4-6",
                    "transport": "anthropic",
                    "thinking_budget_tokens": 4096,
                    "max_output_tokens": 8000,
                }
            )
        }
    },
    "env_dialectic_levels": {
        "env": {
            "DIALECTIC_LEVELS__high__MODEL_CONFIG__TRANSPORT": "anthropic",
            "DIALECTIC_LEVELS__high__MODEL_CONFIG__MODEL": "claude-sonnet-4-5",
            "DIALECTIC_LEVELS__high__MODEL_CONFIG__THINKING_BUDGET_TOKENS": "4096",
            "DIALECTIC_LEVELS__minimal__MAX_OUTPUT_TOKENS": "300",
            "DIALECTIC_LEVELS__max__MODEL_CONFIG__MODEL": "gpt-5",
            "DIALECTIC_LEVELS__max__MODEL_CONFIG__THINKING_EFFORT": "high",
            "DIALECTIC_LEVELS__low__TOOL_CHOICE": "any",
        }
    },
    "env_dialectic_level_fallback": {
        "env": {
            "DIALECTIC_LEVELS__medium__MODEL_CONFIG__FALLBACK__MODEL": "gpt-5-nano",
            "DIALECTIC_LEVELS__medium__MODEL_CONFIG__FALLBACK__TRANSPORT": "openai",
        }
    },
    "env_all_slots_gemini": {
        "env": {
            "DERIVER_MODEL_CONFIG__TRANSPORT": "gemini",
            "DERIVER_MODEL_CONFIG__MODEL": "gemini-2.5-flash",
            "SUMMARY_MODEL_CONFIG__TRANSPORT": "gemini",
            "SUMMARY_MODEL_CONFIG__MODEL": "gemini-2.5-flash",
            "DREAM_DEDUCTION_MODEL_CONFIG__TRANSPORT": "gemini",
            "DREAM_DEDUCTION_MODEL_CONFIG__MODEL": "gemini-2.5-pro",
            "DREAM_INDUCTION_MODEL_CONFIG__TRANSPORT": "gemini",
            "DREAM_INDUCTION_MODEL_CONFIG__MODEL": "gemini-2.5-pro",
        }
    },
    "env_proxy_aliases": {
        "env": {
            "LLM_OPENAI_BASE_URL": "https://tentacruel.example/v1",
            "LLM_OPENAI_API_KEY": "sk-proxy",
            "SUMMARY_MODEL_CONFIG__MODEL": "summarizer",
            "DERIVER_MODEL_CONFIG__MODEL": "deriver",
            "DIALECTIC_LEVELS__high__MODEL_CONFIG__MODEL": "dialectic-high",
            "SUMMARY_MODEL_CONFIG__FALLBACK__MODEL": "gpt-5-nano",
            "SUMMARY_MODEL_CONFIG__FALLBACK__TRANSPORT": "openai",
        }
    },
    "env_mistral_and_minimax": {
        "env": {
            "DERIVER_MODEL_CONFIG__MODEL": "mistralai/mistral-small-3.2-24b-instruct",
            "SUMMARY_MODEL_CONFIG__MODEL": "MiniMax-M2",
        }
    },
    # These env names mean nothing before the catalog exists, so they must
    # leave every slot exactly as configured.
    "env_unreferenced_catalog_vars": {
        "env": {
            "MODELS__HAIKU__MODEL": "claude-haiku-5-5",
            "MODELS__HAIKU__TRANSPORT": "anthropic",
            "MODELS": json.dumps(
                {"nano": {"model": "gpt-5-nano", "transport": "openai"}}
            ),
        }
    },
    "toml_full": {
        "toml": """
[deriver.model_config]
transport = "anthropic"
model = "claude-haiku-4-5"
thinking_budget_tokens = 1024
max_output_tokens = 6000

[deriver.model_config.fallback]
transport = "openai"
model = "gpt-5-nano"
thinking_effort = "minimal"

[summary.model_config]
model = "openai/gpt-5.4-mini"
structured_output_mode = "json_schema"

[summary.model_config.overrides]
base_url = "https://proxy.example/v1"
api_key = "sk-toml"

[summary.model_config.overrides.provider_params.extra_body]
reasoning = { effort = "low" }

[dream.deduction_model_config]
transport = "gemini"
model = "gemini-2.5-pro"
thinking_budget_tokens = 2048

[dialectic.levels.max.model_config]
transport = "anthropic"
model = "claude-opus-5-5"
thinking_effort = "high"

[dialectic.levels.minimal]
max_tool_iterations = 2
max_output_tokens = 400
"""
    },
    "toml_unreferenced_models_table": {
        "toml": """
[models.haiku]
model = "claude-haiku-5-5"
transport = "anthropic"

[deriver.model_config]
model = "gpt-4.1-mini"
"""
    },
    "toml_and_env_precedence": {
        "toml": """
[deriver.model_config]
transport = "openai"
model = "from-toml"
thinking_effort = "low"
max_output_tokens = 1234
""",
        "env": {
            "DERIVER_MODEL_CONFIG__MODEL": "from-env",
            "DERIVER_MODEL_CONFIG__TRANSPORT": "openai",
        },
    },
    "toml_env_transport_switch": {
        "toml": """
[summary.model_config]
transport = "anthropic"
model = "claude-sonnet-4-5"
thinking_budget_tokens = 2048
""",
        "env": {
            "SUMMARY_MODEL_CONFIG__TRANSPORT": "openai",
            "SUMMARY_MODEL_CONFIG__MODEL": "gpt-5",
        },
    },
    "error_structured_output_on_anthropic": {
        "env": {
            "DERIVER_MODEL_CONFIG__TRANSPORT": "anthropic",
            "DERIVER_MODEL_CONFIG__MODEL": "claude-haiku-4-5",
            "DERIVER_MODEL_CONFIG__STRUCTURED_OUTPUT_MODE": "json_object",
        }
    },
    "error_anthropic_budget_too_small": {
        "env": {
            "SUMMARY_MODEL_CONFIG__TRANSPORT": "anthropic",
            "SUMMARY_MODEL_CONFIG__MODEL": "claude-haiku-4-5",
            "SUMMARY_MODEL_CONFIG__THINKING_BUDGET_TOKENS": "512",
        }
    },
    "error_dialectic_budget_exceeds_output": {
        "env": {
            "DIALECTIC_LEVELS__low__MODEL_CONFIG__TRANSPORT": "anthropic",
            "DIALECTIC_LEVELS__low__MODEL_CONFIG__MODEL": "claude-sonnet-4-5",
            "DIALECTIC_LEVELS__low__MODEL_CONFIG__THINKING_BUDGET_TOKENS": "9000",
        }
    },
    "error_fallback_missing_model": {
        "env": {"DERIVER_MODEL_CONFIG__FALLBACK__TRANSPORT": "openai"}
    },
    "error_unknown_transport": {"env": {"SUMMARY_MODEL_CONFIG__TRANSPORT": "bedrock"}},
}


def run_case(name: str, case: dict[str, Any]) -> dict[str, Any]:
    """Boot config for one case in a fresh interpreter and return the probe's output."""
    with tempfile.TemporaryDirectory() as cwd:
        toml = case.get("toml")
        if toml is not None:
            Path(cwd, "config.toml").write_text(toml)
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHON_DOTENV_DISABLED": "1",
            **case.get("env", {}),
        }
        args = [sys.executable, "-c", _PROBE]
        if name == "defaults":
            args.append("--sweep")
        result = subprocess.run(
            args, cwd=cwd, env=env, capture_output=True, text=True, check=False
        )
    if result.returncode != 0:
        raise RuntimeError(f"case {name!r} crashed:\n{result.stderr}")
    return json.loads(result.stdout)


def run_all(cases: dict[str, dict[str, Any]] = CASES) -> dict[str, Any]:
    """Run every case, each in its own interpreter."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {name: pool.submit(run_case, name, c) for name, c in cases.items()}
        return {name: future.result() for name, future in futures.items()}


def differences(path: str, old: Any, new: Any) -> list[str]:
    """Every leaf where `new` differs from `old`, as `path.to.leaf: old -> new`."""
    if isinstance(old, dict) and isinstance(new, dict):
        old_d: dict[str, Any] = old  # pyright: ignore[reportUnknownVariableType]
        new_d: dict[str, Any] = new  # pyright: ignore[reportUnknownVariableType]
        found: list[str] = []
        for key in sorted(set(old_d) | set(new_d)):
            if key not in new_d:
                found.append(f"{path}.{key}: removed (was {old_d[key]!r})")
            elif key not in old_d:
                found.append(f"{path}.{key}: added ({new_d[key]!r})")
            else:
                found.extend(differences(f"{path}.{key}", old_d[key], new_d[key]))
        return found
    if old != new:
        return [f"{path}: {old!r} -> {new!r}"]
    return []


def main() -> None:
    if "--write" not in sys.argv:
        print("pass --write to regenerate", SNAPSHOT_PATH)
        sys.exit(2)
    SNAPSHOT_PATH.write_text(json.dumps(run_all(), indent=1, sort_keys=True) + "\n")
    print("wrote", SNAPSHOT_PATH)


if __name__ == "__main__":
    main()
