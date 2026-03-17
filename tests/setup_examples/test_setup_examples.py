import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_db_connection_uri(tmp_path: Path, env_overrides: dict[str, str]) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.pop("DB_CONNECTION_URI", None)
    env.update(env_overrides)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from src.config import settings; "
                "import json; "
                "print(json.dumps({'db_uri': settings.DB.CONNECTION_URI}))"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)["db_uri"]


def test_env_var_overrides_dotenv_and_toml(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "DB_CONNECTION_URI=postgresql+psycopg://dotenv:dotenv@localhost:5432/dotenv\n"
    )
    (tmp_path / "config.toml").write_text(
        '[db]\nCONNECTION_URI = "postgresql+psycopg://toml:toml@localhost:5432/toml"\n'
    )

    db_uri = _load_db_connection_uri(
        tmp_path,
        {
            "DB_CONNECTION_URI": "postgresql+psycopg://env:env@localhost:5432/env",
        },
    )

    assert db_uri == "postgresql+psycopg://env:env@localhost:5432/env"


def test_dotenv_overrides_toml(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "DB_CONNECTION_URI=postgresql+psycopg://dotenv:dotenv@localhost:5432/dotenv\n"
    )
    (tmp_path / "config.toml").write_text(
        '[db]\nCONNECTION_URI = "postgresql+psycopg://toml:toml@localhost:5432/toml"\n'
    )

    db_uri = _load_db_connection_uri(tmp_path, {})

    assert db_uri == "postgresql+psycopg://dotenv:dotenv@localhost:5432/dotenv"


def test_toml_overrides_builtin_defaults(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        '[db]\nCONNECTION_URI = "postgresql+psycopg://toml:toml@localhost:5432/toml"\n'
    )

    db_uri = _load_db_connection_uri(tmp_path, {})

    assert db_uri == "postgresql+psycopg://toml:toml@localhost:5432/toml"


def test_examples_share_the_canonical_local_database_contract() -> None:
    env_template = (REPO_ROOT / ".env.template").read_text()
    config_example = (REPO_ROOT / "config.toml.example").read_text()

    assert (
        "DB_CONNECTION_URI=postgresql+psycopg://testuser:testpwd@localhost:5432/honcho"
        in env_template
    )
    assert (
        'CONNECTION_URI = "postgresql+psycopg://testuser:testpwd@localhost:5432/honcho"'
        in config_example
    )


def test_compose_example_matches_the_canonical_contract() -> None:
    compose_example = (REPO_ROOT / "docker-compose.yml.example").read_text()

    assert (
        "DB_CONNECTION_URI=postgresql+psycopg://testuser:testpwd@database:5432/honcho"
        in compose_example
    )
    assert "CACHE_ENABLED=true" in compose_example
    assert compose_example.count("METRICS_ENABLED=true") == 2
    assert (
        "./database/init.sql:/docker-entrypoint-initdb.d/init.sql" in compose_example
    )
    assert (REPO_ROOT / "database" / "init.sql").exists()


def test_examples_surface_default_llm_requirements() -> None:
    env_template = (REPO_ROOT / ".env.template").read_text()
    config_example = (REPO_ROOT / "config.toml.example").read_text()
    compose_example = (REPO_ROOT / "docker-compose.yml.example").read_text()

    assert "LLM_OPENAI_API_KEY=your-openai-api-key-here" in env_template
    assert "LLM_ANTHROPIC_API_KEY=your-anthropic-api-key-here" in env_template
    assert "LLM_GEMINI_API_KEY=your-google-api-key-here" in env_template
    assert "Required to boot with the shipped defaults:" in config_example
    assert "two different custom OpenAI-compatible" in config_example
    assert "Provide required LLM_* keys" in compose_example
