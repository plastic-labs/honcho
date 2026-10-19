"""Docker Compose wrapper for the matrix stack, plus direct database access."""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, LiteralString

import psycopg
from psycopg import sql

logger = logging.getLogger(__name__)

COMPOSE_FILE = Path(__file__).with_name("compose.yml")

# Runs inside an image to drive its registered backfills. "partial" moves
# half of each backfill's pending rows, so the database holds both shapes;
# "drain" runs the reconciler's own cycle until nothing is pending. Prints
# the pending count per backfill as the last line.
BACKFILL_SCRIPT = """
import asyncio, json, sys
from src.dependencies import tracked_db
from src.reconciler.backfill import BACKFILLS, run_backfill_cycle

async def main(mode):
    pending = {}
    for backfill in BACKFILLS.values():
        if mode == "partial":
            async with tracked_db("cmx_backfill") as db:
                count = await backfill.count_pending(db)
            if count:
                async with tracked_db("cmx_backfill") as db:
                    await backfill.run_batch(db, max(1, count // 2))
                    await db.commit()
        else:
            while (await run_backfill_cycle(backfill)).still_pending:
                pass
        async with tracked_db("cmx_backfill") as db:
            pending[backfill.name] = await backfill.count_pending(db)
    print("CMX_BACKFILL " + json.dumps(pending))

asyncio.run(main(sys.argv[1]))
"""


class Stack:
    """The compose project: one database, both images, API and worker each."""

    def __init__(
        self, old_image: str, new_image: str, db_schema: str, pg_port: int
    ) -> None:
        self.db_schema: str = db_schema
        self.pg_port: int = pg_port
        self._env: dict[str, str] = {
            **os.environ,
            "HONCHO_OLD_IMAGE": old_image,
            "HONCHO_NEW_IMAGE": new_image,
            "CMX_DB_SCHEMA": db_schema,
            "CMX_PG_PORT": str(pg_port),
        }

    def compose(self, *args: str, capture: bool = False) -> str:
        result = subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), *args],
            env=self._env,
            check=False,
            text=True,
            capture_output=capture,
        )
        if result.returncode != 0:
            detail = (result.stderr or "")[-2000:] if capture else ""
            raise RuntimeError(
                f"docker compose {' '.join(args[:3])} failed ({result.returncode})"
                + (f":\n{detail}" if detail else "")
            )
        return result.stdout or ""

    def up(self, *services: str) -> None:
        self.compose("up", "-d", "--wait", "--quiet-pull", *services)

    def stop(self, *services: str) -> None:
        self.compose("stop", *services)

    def run(self, service: str, *command: str) -> str:
        """Run a one-off command in a service's image, after its dependencies."""
        output = self.compose(
            "run", "--rm", "--no-deps", "-T", service, *command, capture=True
        )
        return output

    def logs(self, *services: str) -> str:
        return self.compose("logs", "--no-color", *services, capture=True)

    def down(self) -> None:
        self.compose("down", "-v", "--remove-orphans")

    def backfill(self, service: str, mode: str) -> dict[str, int]:
        output = self.run(service, "python", "-c", BACKFILL_SCRIPT, mode)
        prefix = "CMX_BACKFILL "
        line = next(
            (row for row in reversed(output.splitlines()) if row.startswith(prefix)),
            None,
        )
        if line is None:
            raise RuntimeError(f"backfill {mode} printed no result:\n{output[-2000:]}")
        return json.loads(line.removeprefix(prefix))

    # Direct SQL, for seeding legacy rows and reading queue state.

    def connect(self) -> psycopg.Connection[Any]:
        conn = psycopg.connect(
            f"postgresql://postgres:postgres@localhost:{self.pg_port}/honcho",
            autocommit=True,
        )
        conn.execute(
            sql.SQL("SET search_path TO {}, public").format(
                sql.Identifier(self.db_schema)
            )
        )
        return conn

    def query(
        self, statement: LiteralString, params: Any = None
    ) -> list[tuple[Any, ...]]:
        with self.connect() as conn:
            return conn.execute(statement, params).fetchall()

    def schema_revision(self) -> str | None:
        rows = (
            self.query("SELECT version_num FROM alembic_version")
            if self._has_version_table()
            else []
        )
        return rows[0][0] if rows else None

    def _has_version_table(self) -> bool:
        return bool(
            self.query(
                "SELECT 1 FROM information_schema.tables"
                + " WHERE table_schema = %s AND table_name = 'alembic_version'",
                (self.db_schema,),
            )
        )

    def wait_for_queue(self, timeout: float = 180.0) -> None:
        """Block until no user-facing or deletion work is left unprocessed."""
        deadline = time.monotonic() + timeout
        while True:
            pending = self.query(
                "SELECT count(*) FROM queue"
                + " WHERE NOT processed AND task_type <> 'reconciler'"
            )[0][0]
            active = self.query("SELECT count(*) FROM active_queue_sessions")[0][0]
            if pending == 0 and active == 0:
                return
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"queue did not drain in {timeout}s"
                    + f" ({pending} pending, {active} active)"
                )
            time.sleep(0.5)

    def queue_errors(self) -> list[tuple[str, str, str]]:
        return [
            (str(row[0]), row[1], row[2])
            for row in self.query(
                "SELECT id, task_type, error FROM queue"
                + " WHERE error IS NOT NULL ORDER BY id"
            )
        ]
