import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import tiktoken

from tests import conftest as test_configuration


@pytest.fixture
def isolated_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MagicMock, MagicMock]:
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setattr(
        test_configuration, "settings", SimpleNamespace(LLM=SimpleNamespace())
    )
    preflight = MagicMock()
    monkeypatch.setattr(test_configuration, "_check_test_database_reachable", preflight)
    monkeypatch.setattr(test_configuration, "_sweep_stale_test_databases", MagicMock())
    config = MagicMock(spec=pytest.Config)
    config.rootpath = Path(__file__).resolve().parents[1]
    config.args = ["tests/test_dependencies.py"]
    config.getoption.return_value = False
    config.option = SimpleNamespace(
        collectonly=False,
        help=False,
        showfixtures=False,
        markers=False,
    )
    return config, preflight


@pytest.mark.parametrize("mode", ["collectonly", "help", "showfixtures", "markers"])
def test_informational_modes_skip_database_preflight(
    isolated_configuration: tuple[MagicMock, MagicMock],
    mode: str,
) -> None:
    config, preflight = isolated_configuration
    setattr(config.option, mode, True)

    test_configuration.pytest_configure(config)

    preflight.assert_not_called()


@pytest.mark.parametrize(
    "path",
    [
        "tests/llm",
        "tests/live_llm",
        "tests/mock_provider",
        "tests/test_config.py",
        "tests/test_pytest_configuration.py",
        "tests/utils/test_length_finish_reason.py",
        "tests/utils/test_clients.py",
        "tests/crud/test_session_scope_clauses.py",
        "tests/deriver/test_prompts.py",
        "tests/utils/test_custom_instructions_prompts.py",
        "tests/test_security.py",
        "tests/test_generate_jwt_script.py",
        "tests/llm/test_api.py::test_backend",
    ],
)
def test_database_free_paths_skip_database_preflight(
    isolated_configuration: tuple[MagicMock, MagicMock],
    path: str,
) -> None:
    config, preflight = isolated_configuration
    config.args = [path]

    test_configuration.pytest_configure(config)

    preflight.assert_not_called()


@pytest.mark.parametrize(
    "paths",
    [
        [],
        ["tests"],
        ["tests/alembic"],
        ["tests/bench"],
        ["tests/unified"],
        ["tests/test_dependencies.py"],
        ["tests/llm", "tests/test_dependencies.py"],
        ["tests/llm", "tests/alembic"],
    ],
)
def test_database_required_paths_run_database_preflight(
    isolated_configuration: tuple[MagicMock, MagicMock],
    paths: list[str],
) -> None:
    config, preflight = isolated_configuration
    config.args = paths

    test_configuration.pytest_configure(config)

    preflight.assert_called_once()


def test_conftest_preloads_every_tiktoken_encoding_src_uses() -> None:
    # A new encoding missing from TIKTOKEN_ENCODINGS is downloaded mid-test, which
    # pytest-socket blocks; it passes locally on a warm cache and fails in CI.
    src_dir = Path(__file__).resolve().parents[1] / "src"
    used = {
        name
        for path in src_dir.rglob("*.py")
        for name in re.findall(
            r"""get_encoding\(\s*["']([^"']+)["']""", path.read_text()
        )
    }
    used.add(tiktoken.encoding_name_for_model("text-embedding-3-small"))

    assert used
    assert used <= set(test_configuration.TIKTOKEN_ENCODINGS)


def test_xdist_workers_do_not_repeat_database_preflight(
    isolated_configuration: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, preflight = isolated_configuration
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")

    test_configuration.pytest_configure(config)

    preflight.assert_not_called()
