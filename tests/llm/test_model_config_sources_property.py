"""Where the catalog is used, env wins over `config.toml` whatever the spelling.

Hypothesis builds a config that uses the `MODELS` catalog, then splits it
across env and `config.toml`. Each setting goes to a random source in a
random accepted spelling. A setting that env holds may also appear in
`config.toml` under another spelling with a different value, as a stale copy.
The config must load exactly like the same settings written once, in one
source.

The spellings covered are the ones that mean the same setting: `model_ref`
against `model` + `transport`, `thinking_effort` against `reasoning_effort`,
and `overrides.api_key` against `overrides.api_key_env`. They're exercised on
a catalog entry, a slot, its fallback and a dialectic level.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from unittest import mock

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src import config
from src.config import AppSettings

# Where each scope lives in env and in config.toml.
_SCOPES: dict[str, tuple[str, tuple[str, ...]]] = {
    "entry_a": ("MODELS__a__", ("models", "a")),
    "slot": ("DERIVER_MODEL_CONFIG__", ("deriver", "model_config")),
    "fallback": (
        "DERIVER_MODEL_CONFIG__FALLBACK__",
        ("deriver", "model_config", "fallback"),
    ),
    "level": (
        "DIALECTIC_LEVELS__minimal__MODEL_CONFIG__",
        ("dialectic", "levels", "minimal", "model_config"),
    ),
}
_ENTRIES = {
    "a": {"transport": "openai", "model": "gpt-5-nano"},
    "b": {"transport": "anthropic", "model": "claude-haiku-4-5"},
}
_KEY_VARS = {"KEY_A": "sk-a-from-env-var", "KEY_STALE": "sk-stale-from-env-var"}
_EFFORTS = ("minimal", "low", "medium", "high")
_WORKLOAD_PREFIXES = ("LLM_", "DERIVER_", "SUMMARY_", "DREAM_", "DIALECTIC_", "MODELS")


@dataclass(frozen=True)
class Setting:
    """One setting: where it lives, its key path, and its value."""

    scope: str
    path: tuple[str, ...]
    value: Any


@dataclass(frozen=True)
class Placement:
    """A setting as env or config.toml holds it."""

    source: str  # "env" or "toml"
    setting: Setting


def _effort(scope: str) -> st.SearchStrategy[tuple[Setting, list[Placement]]]:
    """A thinking effort in either spelling, maybe with a stale TOML copy."""

    @st.composite
    def build(draw: st.DrawFn) -> tuple[Setting, list[Placement]]:
        value = draw(st.sampled_from(_EFFORTS))
        canonical = Setting(scope, ("thinking_effort",), value)
        spelling = draw(st.sampled_from(("thinking_effort", "reasoning_effort")))
        source = draw(st.sampled_from(("env", "toml")))
        placements = [Placement(source, Setting(scope, (spelling,), value))]
        if source == "env" and draw(st.booleans()):
            stale = draw(st.sampled_from([e for e in _EFFORTS if e != value]))
            stale_spelling = draw(
                st.sampled_from(("thinking_effort", "reasoning_effort"))
            )
            if stale_spelling != spelling:
                placements.append(
                    Placement("toml", Setting(scope, (stale_spelling,), stale))
                )
        return canonical, placements

    return build()


def _credential(scope: str) -> st.SearchStrategy[tuple[Setting, list[Placement]]]:
    """A key or a key variable, maybe with a stale TOML copy of the other."""

    @st.composite
    def build(draw: st.DrawFn) -> tuple[Setting, list[Placement]]:
        real = draw(
            st.sampled_from(
                (("api_key", "sk-a-literal"), ("api_key_env", "KEY_A")),
            )
        )
        canonical = Setting(scope, ("overrides", real[0]), real[1])
        source = draw(st.sampled_from(("env", "toml")))
        placements = [Placement(source, canonical)]
        if source == "env" and draw(st.booleans()):
            stale = (
                ("api_key_env", "KEY_STALE")
                if real[0] == "api_key"
                else ("api_key", "sk-stale-literal")
            )
            placements.append(
                Placement("toml", Setting(scope, ("overrides", stale[0]), stale[1]))
            )
        return canonical, placements

    return build()


def _selector(scope: str) -> st.SearchStrategy[tuple[list[Setting], list[Placement]]]:
    """A `model_ref`, maybe with a stale inline model or other ref in TOML."""

    @st.composite
    def build(draw: st.DrawFn) -> tuple[list[Setting], list[Placement]]:
        ref = draw(st.sampled_from(sorted(_ENTRIES)))
        canonical = [Setting(scope, ("model_ref",), ref)]
        source = draw(st.sampled_from(("env", "toml")))
        placements = [Placement(source, canonical[0])]
        if source == "env":
            stale = draw(st.sampled_from(("none", "inline", "other_ref")))
            if stale == "inline":
                placements += [
                    Placement("toml", Setting(scope, ("transport",), "gemini")),
                    Placement("toml", Setting(scope, ("model",), "gemini-2.5-flash")),
                ]
            elif stale == "other_ref":
                other = next(name for name in _ENTRIES if name != ref)
                placements.append(
                    Placement("toml", Setting(scope, ("model_ref",), other))
                )
        return canonical, placements

    return build()


@st.composite
def split_configs(draw: st.DrawFn) -> tuple[list[Setting], list[Placement]]:
    """The settings of one config, and the same config split across sources."""
    canonical: list[Setting] = []
    placements: list[Placement] = []

    effort, placed = draw(_effort("entry_a"))
    canonical.append(effort)
    placements += placed
    credential, placed = draw(_credential("entry_a"))
    canonical.append(credential)
    placements += placed

    for scope in ("slot", "fallback", "level"):
        selectors, placed = draw(_selector(scope))
        canonical += selectors
        placements += placed
        # Thinking effort only on the OpenAI entry; Anthropic takes a budget.
        if selectors[0].value == "a" and draw(st.booleans()):
            effort, placed = draw(_effort(scope))
            canonical.append(effort)
            placements += placed
    return canonical, placements


def _put(tree: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    for part in path[:-1]:
        tree = tree.setdefault(part, {})
    tree[path[-1]] = value


def _sources(placements: list[Placement]) -> tuple[dict[str, str], dict[str, Any]]:
    env: dict[str, str] = dict(_KEY_VARS)
    toml: dict[str, Any] = {"models": {name: dict(e) for name, e in _ENTRIES.items()}}
    for placement in placements:
        prefix, toml_path = _SCOPES[placement.setting.scope]
        if placement.source == "env":
            name = prefix + "__".join(part.upper() for part in placement.setting.path)
            env[name] = str(placement.setting.value)
        else:
            _put(toml, toml_path + placement.setting.path, placement.setting.value)
    return env, toml


@contextmanager
def _loaded_from(env: dict[str, str], toml: dict[str, Any]) -> Generator[None]:
    kept = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().startswith(_WORKLOAD_PREFIXES)
    }
    with (
        mock.patch.dict(os.environ, {**kept, **env}, clear=True),
        mock.patch.object(config, "TOML_CONFIG", toml),
    ):
        yield


def _load(env: dict[str, str], toml: dict[str, Any]) -> dict[str, Any]:
    with _loaded_from(env, toml):
        loaded = AppSettings()
    return {
        "models": {
            name: entry.model_dump() for name, entry in sorted(loaded.MODELS.items())
        },
        "deriver": loaded.DERIVER.MODEL_CONFIG.model_dump(),
        "level": loaded.DIALECTIC.LEVELS["minimal"].MODEL_CONFIG.model_dump(),
    }


@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(split_configs())
def test_split_config_loads_like_one_source(
    config_and_split: tuple[list[Setting], list[Placement]],
) -> None:
    canonical, placements = config_and_split
    one_source = _load(*_sources([Placement("toml", s) for s in canonical]))
    split = _load(*_sources(placements))
    assert split == one_source
