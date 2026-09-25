"""Verify src/models.py honors EMBEDDING_VECTOR_DIMENSIONS at import time."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_in_fresh_interpreter(env_overrides: dict[str, str]) -> dict[str, int]:
    """Import src.models in a fresh interpreter and return the vector dims.

    A subprocess is required because src.models reads
    settings.EMBEDDING.VECTOR_DIMENSIONS at module import time to construct
    SQLAlchemy column types — reloading the module in-process would conflict
    with the existing Base.registry from earlier imports.

    PYTHON_DOTENV_DISABLED=1 prevents config.py:20 from reloading the
    developer's .env file (which calls load_dotenv with override=True)
    and clobbering our test overrides.
    """
    env: dict[str, str] = {
        **os.environ,
        "PYTHON_DOTENV_DISABLED": "1",
        **env_overrides,
    }
    snippet = (
        "import json\n"
        "from src.models import Document, MessageEmbedding\n"
        "print(json.dumps({\n"
        "    'message_embedding_dim': MessageEmbedding.__table__.c.embedding.type.dim,\n"
        "    'document_dim': Document.__table__.c.embedding.type.dim,\n"
        "}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        env=env,
        cwd=str(_PROJECT_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    parsed: dict[str, int] = json.loads(result.stdout.strip().splitlines()[-1])
    return parsed


def test_models_uses_default_1536_when_no_env_override() -> None:
    dims = _run_in_fresh_interpreter({})
    assert dims == {"message_embedding_dim": 1536, "document_dim": 1536}


def test_models_honors_explicit_embedding_vector_dimensions() -> None:
    dims = _run_in_fresh_interpreter({"EMBEDDING_VECTOR_DIMENSIONS": "768"})
    assert dims == {"message_embedding_dim": 768, "document_dim": 768}
