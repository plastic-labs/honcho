"""Cold search imports must work without pytest's eager CRUD imports."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run_fresh_process(code: str) -> None:
    """Run the installed project interpreter with placeholder provider keys."""
    env = os.environ | {
        "LLM_OPENAI_API_KEY": "test-key-not-a-real-credential",
        "LLM_ANTHROPIC_API_KEY": "test-key-not-a-real-credential",
        "LLM_GEMINI_API_KEY": "test-key-not-a-real-credential",
        "LANGFUSE_PUBLIC_KEY": "",
        "SENTRY_ENABLED": "false",
        "MULTI_TENANT": "false",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "modules",
    [
        ("src.utils.search",),
        ("src.utils.search", "src.crud.lifecycle", "src.crud.message"),
        ("src.crud.lifecycle", "src.utils.search"),
        ("src.crud.message", "src.utils.search"),
        ("src.crud", "src.utils.search"),
        ("src.utils.search", "src.main"),
    ],
)
def test_search_imports_in_fresh_process(modules: tuple[str, ...]) -> None:
    """Import real modules in a new interpreter with all network sockets blocked."""
    code = f"""
import importlib
import sys
import pytest_socket

pytest_socket.disable_socket(allow_unix_socket=True)
assert 'src.crud' not in sys.modules
assert 'src.utils.search' not in sys.modules
for name in {json.dumps(modules)}:
    importlib.import_module(name)
search = importlib.import_module('src.utils.search')
for name in ('matching_messages', 'fetch_messages_by_ids', '_semantic_search_pgvector', 'search'):
    assert callable(getattr(search, name)), name
assert search.reciprocal_rank_fusion(['a', 'b'], ['b'], limit=2) == ['b', 'a']
"""
    _run_fresh_process(code)


@pytest.mark.parametrize(
    "first_call", ["fetch_messages_by_ids", "_semantic_search_pgvector", "search"]
)
def test_lifecycle_import_on_first_search_call(first_call: str) -> None:
    """Exercise each local lifecycle import after a cold search-first load."""
    code = f"""
import asyncio
import datetime
import sys
import types
from contextlib import asynccontextmanager
import pytest_socket

pytest_socket.disable_socket(allow_unix_socket=True)
assert 'src.crud' not in sys.modules
import src.utils.search as search
assert 'src.crud' not in sys.modules

class EmptyResult:
    def __init__(self, rows=()):
        self.rows = list(rows)
    def scalars(self):
        return EmptyResult()
    def all(self):
        return self.rows
    def __iter__(self):
        return iter(self.rows)

PAGE_ROW = types.SimpleNamespace(id=1, created_at=datetime.datetime(2026, 1, 1), rank=0.0)

class StatementCollector:
    def __init__(self):
        self.statements = []
    async def execute(self, statement):
        self.statements.append(statement)
        columns = getattr(statement, 'selected_columns', None)
        if columns is not None and 'rank' in columns:
            return EmptyResult([PAGE_ROW])
        return EmptyResult()

async def run():
    db = StatementCollector()
    if {first_call!r} == 'fetch_messages_by_ids':
        rows = await search.fetch_messages_by_ids(db, ['message-id'])
    elif {first_call!r} == '_semantic_search_pgvector':
        rows = await search._semantic_search_pgvector(db, 'workspace', [1.0], 1)
    else:
        @asynccontextmanager
        async def tracked_db(*args, **kwargs):
            yield db
        search.tracked_db = tracked_db
        search.settings.EMBED_MESSAGES = False
        rows = await search.search('needle', filters={{'workspace_id': 'workspace'}})
    assert rows == []
    # search() also sets its statement timeout and reads a text-match page; the
    # row fetch through the lifecycle-filtered statement is the last one.
    sql = str(db.statements[-1])
    assert 'messages.deleted_at IS NULL' in sql, sql
    for table in ('workspaces', 'peers', 'sessions'):
        assert f'{{table}}_1.deleted_at IS NULL' in sql, sql
    assert 'src.crud.lifecycle' in sys.modules

asyncio.run(run())
"""
    _run_fresh_process(code)
