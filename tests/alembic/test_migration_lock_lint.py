"""Lint migrations for locks the request path needs.

Self-hosted and dedicated databases run every revision against live traffic,
so a revision must not hold a strong lock on a request-path table for longer
than its DDL takes. Three shapes are refused:

1. CREATE/DROP INDEX CONCURRENTLY by hand, instead of create_index_online /
   drop_index_online (migrations/utils.py), which handle partitioned tables,
   retries, invalid leftovers and the autocommit block.
2. op.create_index / op.drop_index on a tenant table that the shared pool
   partitions: a plain build takes SHARE (writes blocked) for its whole run.
3. A backfill (INSERT ... SELECT, or an UPDATE over a table) in the same
   transaction as DDL that locks a request-path table above ROW EXCLUSIVE:
   the lock is held until the backfill commits.

A revision that is safe for another reason sets a module-level
LOCK_LINT_EXEMPT = "<reason>". Revisions released before this lint are listed
in _GRANDFATHERED with the reason they may stay as they are.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from src.db import Base
from src.models import TenantPartitioned

_VERSIONS = Path(__file__).resolve().parents[2] / "migrations" / "versions"

REQUEST_PATH_TABLES = frozenset(
    {"queue", "messages", "documents", "message_embeddings", "sessions", "peers"}
)
PARTITIONED_TABLES = frozenset(
    mapper.local_table.name
    for mapper in Base.registry.mappers
    if issubclass(mapper.class_, TenantPartitioned)
)
_HEAVY_OPS = frozenset(
    {
        "create_index",
        "drop_index",
        "add_column",
        "alter_column",
        "drop_column",
        "create_foreign_key",
        "create_unique_constraint",
        "create_primary_key",
        "drop_constraint",
    }
)
_HEAVY_SQL = re.compile(
    r"\b(ALTER\s+TABLE|CREATE\s+TRIGGER\b[^;]*?\bON|LOCK\s+TABLE|"
    + r"CREATE\s+(UNIQUE\s+)?INDEX(?!\s+CONCURRENTLY)\b[^;]*?\bON)\s+"
    + r"(ONLY\s+)?(\{schema\}\.|\"?\w+\"?\.)?\"?(\w+)",
    re.IGNORECASE,
)
_BACKFILL_SQL = re.compile(
    r"\bINSERT\s+INTO\b[^;]*?\bSELECT\b|\bUPDATE\s+(\{schema\}\.)?\w+\s+(AS\s+\w+\s+)?SET\b",
    re.IGNORECASE | re.DOTALL,
)

# Released (v3.5.0 or earlier) before this lint: every database that will run
# them has, or they ran on databases far smaller than today's. Rewriting a
# released revision changes nothing for the databases that already applied it.
_RELEASED_BEFORE_LINT = "released in v3.5.0 or earlier, before this lint"
_GRANDFATHERED: dict[str, str] = {
    revision: _RELEASED_BEFORE_LINT
    for revision in (
        "066e87ca5b07",
        "08894082221a",
        "119a52b73c60",
        "3e7a1c9d5b20",
        "556a16564f50",
        "564ba40505c5",
        "88b0fb10906f",
        "a7c3e9f1b2d4",
        "b8d2f4a6c9e1",
        "baa22cad81e2",
        "c3828084f472",
        "f1a2b3c4d5e6",
    )
}


def _string_of(node: ast.AST) -> str:
    """The literal text of a str or f-string argument ({...} kept as written)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                parts.append("{" + ast.unparse(value.value) + "}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _string_of(node.left) + _string_of(node.right)
    if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name)):
        name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        # sa.text("...").bindparams(...): the SQL is the text() call it wraps.
        if name == "bindparams" and isinstance(node.func, ast.Attribute):
            return _string_of(node.func.value)
        if name == "text" and node.args:
            return _string_of(node.args[0])
    return ""


# op.<name>(...) calls whose table is the second positional argument.
_TABLE_SECOND = frozenset(
    {
        "create_index",
        "drop_index",
        "create_foreign_key",
        "create_unique_constraint",
        "create_primary_key",
        "drop_constraint",
    }
)


def _op_table(call: ast.Call, name: str) -> str | None:
    """The table an op.<name>(...) call names, from its usual positional or keyword."""
    for keyword in call.keywords:
        if keyword.arg in {"table_name", "source_table"}:
            return _string_of(keyword.value) or None
    position = 1 if name in _TABLE_SECOND else 0
    if len(call.args) > position:
        return _string_of(call.args[position]) or None
    return None


def _upgrade(tree: ast.Module) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "upgrade":
            return node
    return None


def _module_string(tree: ast.Module, name: str) -> str | None:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return _string_of(node.value) or None
    return None


def lint_revision(source: str) -> list[str]:
    """Problems in one revision's source, empty when it passes."""
    tree = ast.parse(source)
    upgrade = _upgrade(tree)
    if upgrade is None:
        return []
    problems: list[str] = []
    heavy: list[str] = []
    backfills: list[str] = []
    uses_autocommit = False
    created = {
        _string_of(node.args[0])
        for node in ast.walk(upgrade)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_table"
        and node.args
    }
    for node in ast.walk(upgrade):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in {"autocommit_block", "caller_owns_transaction"}:
            uses_autocommit = True
        owner = (
            func.value.id
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
            else ""
        )
        if owner == "op" and name in _HEAVY_OPS:
            table = _op_table(node, name)
            if table in created:
                continue
            if name in {"create_index", "drop_index"} and table in PARTITIONED_TABLES:
                problems.append(f"op.{name} on partitioned {table}: use {name}_online")
            if table in REQUEST_PATH_TABLES:
                heavy.append(f"op.{name}({table})")
        if owner == "op" and name == "execute" and node.args:
            sql = _string_of(node.args[0])
            if re.search(r"\bINDEX\s+CONCURRENTLY\b", sql, re.IGNORECASE):
                problems.append(
                    "hand-written INDEX CONCURRENTLY: use create_index_online / drop_index_online"
                )
            for match in _HEAVY_SQL.finditer(sql):
                if match.group(5).lower() in REQUEST_PATH_TABLES:
                    verb = " ".join(match.group(1).split()[:2]).upper()
                    heavy.append(f"{verb} {match.group(5)}")
            if _BACKFILL_SQL.search(sql):
                backfills.append(sql.strip().splitlines()[0][:60])
    if heavy and backfills and not uses_autocommit:
        problems.append(
            f"backfill in the same transaction as {sorted(set(heavy))[0]}"
            + " (holds the lock until the backfill commits)"
        )
    return problems


def _revisions() -> list[Path]:
    return sorted(p for p in _VERSIONS.glob("*.py") if not p.name.startswith("__"))


@pytest.mark.parametrize("path", _revisions(), ids=lambda p: p.name.split("_")[0])
def test_revision_takes_no_long_request_path_lock(path: Path) -> None:
    revision = path.name.split("_")[0]
    if revision in _GRANDFATHERED:
        return
    source = path.read_text()
    if _module_string(ast.parse(source), "LOCK_LINT_EXEMPT"):
        return
    problems = lint_revision(source)
    assert not problems, f"{path.name}: " + "; ".join(problems)


_TRIGGER_THEN_BACKFILL = """
def upgrade():
    op.execute("CREATE TRIGGER t AFTER INSERT ON {schema}.queue FOR EACH ROW EXECUTE FUNCTION f()")
    op.execute("INSERT INTO {schema}.queue_item_batches SELECT * FROM {schema}.queue")
"""


@pytest.mark.parametrize(
    ("source", "problem"),
    [
        (
            _TRIGGER_THEN_BACKFILL,
            "backfill in the same transaction as CREATE TRIGGER queue",
        ),
        (
            'def upgrade():\n    op.execute("CREATE INDEX CONCURRENTLY ix ON messages (id)")\n',
            "hand-written INDEX CONCURRENTLY",
        ),
        (
            'def upgrade():\n    op.create_index("ix", "documents", ["id"])\n',
            "op.create_index on partitioned documents",
        ),
        (
            'def upgrade():\n    op.execute(sa.text("CREATE INDEX CONCURRENTLY ix ON messages (id)"))\n',
            "hand-written INDEX CONCURRENTLY",
        ),
        (
            "def upgrade():\n    op.execute(\n"
            + '        sa.text("CREATE INDEX CONCURRENTLY ix ON messages (id) WHERE id > :n")'
            + ".bindparams(n=1)\n    )\n",
            "hand-written INDEX CONCURRENTLY",
        ),
        (
            "def upgrade():\n"
            + '    op.execute("CREATE TRIGGER t AFTER INSERT ON " + "queue FOR EACH ROW EXECUTE FUNCTION f()")\n'
            + '    op.execute("INSERT INTO queue_item_batches " + "SELECT * FROM queue")\n',
            "backfill in the same transaction as CREATE TRIGGER queue",
        ),
    ],
    ids=[
        "backfill-under-lock",
        "hand-concurrently",
        "plain-index-on-partitioned",
        "text-wrapped",
        "bindparams-wrapped",
        "concatenated",
    ],
)
def test_the_lint_flags_each_shape(source: str, problem: str) -> None:
    assert any(problem in found for found in lint_revision(source))


def test_the_lint_passes_the_safe_shapes() -> None:
    source = """
def upgrade():
    op.create_table("new_table", sa.Column("id", sa.Integer))
    op.create_index("ix_new", "new_table", ["id"])
    create_index_online("ix_docs", "documents", ["id"])
    op.execute("UPDATE {schema}.tenants SET tier = 'shared'")
"""
    assert lint_revision(source) == []
