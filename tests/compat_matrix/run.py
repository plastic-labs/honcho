"""Old/new image compatibility matrix (migration rule 7).

Proves the previous release image and a candidate image can both run against
the candidate's schema, before and after its backfills finish:

1. The old image builds the schema the way its lineage does.
2. Old API and worker seed data, legacy-shaped rows are added, and work is
   left queued.
3. The new image runs ``python -m src.migrate``. Nothing ever downgrades.
4. Round A, backfills half done: every old/new writer and worker pairing
   writes, processes and deletes, and the whole ledger is re-read through
   both APIs after each one.
5. The new image drains every backfill, and round B repeats the pairings.

Run with ``uv run python -m tests.compat_matrix.run --old-image ... --new-image ...``.
"""

import argparse
import functools
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from tests.compat_matrix.api import HonchoAPI
from tests.compat_matrix.ledger import Ledger
from tests.compat_matrix.scenarios import (
    Context,
    enqueue_unprocessed,
    run_combo,
    seed_legacy_rows,
)
from tests.compat_matrix.stack import Stack

logger = logging.getLogger("compat_matrix")

SERVICES = ("api-old", "api-new", "deriver-old", "deriver-new")


@dataclass(frozen=True)
class Lineage:
    """How a database reaches the schema the old image runs on (rule 8)."""

    name: str
    db_schema: str
    description: str
    # Why the lineage cannot run here yet; empty when it can.
    missing: tuple[str, ...] = ()


LINEAGES = {
    lineage.name: lineage
    for lineage in (
        Lineage(
            "dedicated",
            "cmx_tenant",
            "One database per tenant. The chart's provision-db init container"
            + " replays the whole chain with DB_SCHEMA=<app_name>.",
        ),
        Lineage(
            "self-hosted",
            "public",
            "The dedicated chain replayed into the public schema, as a"
            + " self-hosted install does.",
        ),
        Lineage(
            "shared",
            "public",
            "The shared pool: tenant_id hash-partitioned tables built by a"
            + " bootstrap, then stamped instead of replayed.",
            missing=(
                "The shared-pool bootstrap DDL and its row-level-security"
                + " policies are applied out of band and are not checked in"
                + " anywhere a CI fixture could build them from.",
                "The revision the shared pool was stamped at is not recorded"
                + " (at or after e5fe7f8bcf62).",
                "No released image carries the MULTI_TENANT code yet"
                + " (feat/shared-tenants is unmerged), so there is no"
                + " previous image for this lineage.",
            ),
        ),
    )
}


@dataclass
class Report:
    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def phase(self, name: str, fn: Callable[[], object]) -> None:
        start = time.monotonic()
        logger.info("== %s", name)
        fn()
        self.lines.append(f"| {name} | {time.monotonic() - start:.0f}s |")

    def check(self, where: str, problems: list[str]) -> None:
        if problems:
            logger.error("%s: %d problem(s)", where, len(problems))
            for problem in problems:
                logger.error("  %s", problem)
        self.problems += [f"{where}: {p}" for p in problems]


def bootstrap(stack: Stack) -> None:
    """Build the old image's schema with the same command its chart runs."""
    stack.up("database", "redis", "llm-stub")
    stack.run("api-old", "python", "scripts/provision_db.py")


def migrate(stack: Stack) -> str:
    stack.run("api-new", "python", "-m", "src.migrate")
    revision = stack.schema_revision()
    if revision is None:
        raise RuntimeError("no alembic_version after migrate")
    return revision


def run_round(ctx: Context, report: Report, label: str, carry_writer: str) -> None:
    """Every writer/worker pairing, re-reading the full ledger after each.

    The first pairing's worker is the image that did not write the work left
    queued by the previous phase, so queued work always crosses the swap.
    """
    first_worker = "deriver-new" if carry_writer == "old" else "deriver-old"
    pairings = [("old", [first_worker]), ("new", [first_worker])]
    pairings += [
        (writer, [worker])
        for writer in ("old", "new")
        for worker in ("deriver-old", "deriver-new")
        if worker != first_worker
    ]
    pairings.append(("new", ["deriver-old", "deriver-new"]))
    for n, (writer, workers) in enumerate(pairings):
        tag = f"{label}{n}{writer}"
        name = f"{label}: write {writer}, process {'+'.join(workers)}"
        # A pairing that cannot complete is a finding, not the end of the run:
        # the remaining pairings still show what else the candidate breaks.
        try:
            report.phase(name, functools.partial(run_combo, ctx, tag, writer, workers))
        except Exception as exc:
            logger.exception("%s failed", name)
            report.lines.append(f"| {name} | failed |")
            report.check(name, [f"{type(exc).__name__}: {exc}"])
            ctx.stack.stop(*workers)
        ctx.ledger.add(*ctx.carried)
        ctx.carried.clear()
        report.check(f"{label} after {tag}", ctx.ledger.verify(list(ctx.apis.values())))


def main() -> int:
    parser = argparse.ArgumentParser(description="Old/new image compatibility matrix.")
    parser.add_argument("--old-image", required=True)
    parser.add_argument("--new-image", required=True)
    parser.add_argument("--lineage", choices=sorted(LINEAGES), default="dedicated")
    parser.add_argument("--pg-port", type=int, default=15432)
    parser.add_argument("--keep", action="store_true", help="leave the stack up")
    parser.add_argument("--logs-dir", type=Path, default=Path("compat-matrix-logs"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    lineage = LINEAGES[args.lineage]
    if lineage.missing:
        logger.error(
            "Lineage %r is not runnable yet: %s", lineage.name, lineage.description
        )
        for gap in lineage.missing:
            logger.error("  - %s", gap)
        return 2

    stack = Stack(args.old_image, args.new_image, lineage.db_schema, args.pg_port)
    ctx = Context(
        stack=stack,
        ledger=Ledger(),
        apis={
            "old": HonchoAPI(
                "old", f"http://localhost:{os.environ.get('CMX_OLD_API_PORT', '18001')}"
            ),
            "new": HonchoAPI(
                "new", f"http://localhost:{os.environ.get('CMX_NEW_API_PORT', '18002')}"
            ),
        },
    )
    report = Report()
    status = 1
    try:
        stack.down()
        report.phase("bootstrap with old image", lambda: bootstrap(stack))
        old_revision = stack.schema_revision()

        def seed() -> None:
            stack.up("api-old")
            ws = run_combo(ctx, "seed", "old", ["deriver-old"])
            seed_legacy_rows(ctx, ws)
            enqueue_unprocessed(ctx, "seedcarry", "old")
            stack.stop(*SERVICES)

        report.phase("seed with old image", seed)
        report.phase("migrate with new image", lambda: migrate(stack))
        revision = stack.schema_revision()
        logger.info("schema %s -> %s", old_revision, revision)

        partial: dict[str, int] = {}
        report.phase(
            "backfills: one partial batch",
            lambda: partial.update(stack.backfill("deriver-new", "partial")),
        )
        logger.info("pending after partial backfill: %s", partial)
        if not any(partial.values()):
            report.check("backfill", ["partial backfill left nothing pending"])

        stack.up("api-old", "api-new")
        report.check("after migrate", ctx.ledger.verify(list(ctx.apis.values())))
        run_round(ctx, report, "ra", carry_writer="old")
        enqueue_unprocessed(ctx, "racarry", "new")
        stack.stop(*SERVICES)

        drained: dict[str, int] = {}
        report.phase(
            "backfills: drain",
            lambda: drained.update(stack.backfill("deriver-new", "drain")),
        )
        report.check(
            "backfill",
            [f"{name} still has {n} pending" for name, n in drained.items() if n],
        )

        stack.up("api-old", "api-new")
        report.check("after drain", ctx.ledger.verify(list(ctx.apis.values())))
        run_round(ctx, report, "rb", carry_writer="new")

        report.check(
            "queue",
            [
                f"{kind} item {qid} failed: {err[:300]}"
                for qid, kind, err in stack.queue_errors()
            ],
        )
        if stack.schema_revision() != revision:
            report.check("schema", [f"revision moved off {revision} during the run"])
        status = 1 if report.problems else 0
    except Exception:
        logger.exception("matrix aborted")
        report.problems.append("matrix aborted; see the log")
    finally:
        _write_summary(report, args, lineage)
        if status != 0:
            _save_logs(stack, args.logs_dir)
        if not args.keep:
            stack.down()
    return status


def _save_logs(stack: Stack, logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    for service in SERVICES:
        try:
            (logs_dir / f"{service}.log").write_text(stack.logs(service))
        except Exception:
            logger.warning("could not save logs for %s", service, exc_info=True)


def _write_summary(report: Report, args: argparse.Namespace, lineage: Lineage) -> None:
    verdict = "failed" if report.problems else "passed"
    lines = [
        f"### Compatibility matrix ({lineage.name}): {verdict}",
        "",
        f"Old `{args.old_image}`, new `{args.new_image}`, DB_SCHEMA `{lineage.db_schema}`.",
        "",
        "| Phase | Time |",
        "| --- | --- |",
        *report.lines,
    ]
    if report.problems:
        lines += ["", "#### Problems", "", *(f"- {p}" for p in report.problems[:100])]
    text = "\n".join(lines) + "\n"
    print(text)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as summary:
            summary.write(text)


if __name__ == "__main__":
    sys.exit(main())
