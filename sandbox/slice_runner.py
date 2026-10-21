"""Erasure-slice acceptance runner.

Drives the two-api / two-deriver sandbox through one message delete while a
multi-observer deriver batch is paused, Redis is down, and the stack is swapped
to a previous image, then checks every surface the deleted message could leak
through. Invoked by sandbox/slice.sh; see `slice.sh --help`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import psycopg

HERE = Path(__file__).resolve().parent
PROJECT = os.environ.get("SLICE_PROJECT", "slice_harness")
API_PORTS = {
    "api-1": int(os.environ.get("SLICE_API1_PORT", "18100")),
    "api-2": int(os.environ.get("SLICE_API2_PORT", "18101")),
}
DB_PORT = int(os.environ.get("SLICE_DB_PORT", "15532"))
REDIS_PORT = int(os.environ.get("SLICE_REDIS_PORT", "16479"))
DB_URL = f"postgresql://postgres:postgres@127.0.0.1:{DB_PORT}/honcho_sandbox"

WORKSPACE = "slice-ws"
SESSION = "slice-session"
OBSERVED = "alice"
OBSERVERS = ["alice", "bob", "carol"]
SUMMARY_KEYS = {
    "short": "honcho_chat_summary_short",
    "long": "honcho_chat_summary_long",
}
MIGRATION_FREE_START = (
    "exec /app/.venv/bin/fastapi run --host 0.0.0.0 --workers 1 src/main.py"
)


def now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


class Run:
    """State and output for one scenario run."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.state_dir = HERE / ".slice"
        self.pause_dir = self.state_dir / "pause"
        self.canary = f"CANARY{uuid.uuid4().hex[:12].upper()}"
        self.app_image = args.image_under_test
        self.api_start = "sh docker/entrypoint.sh"
        self.steps: list[dict[str, Any]] = []
        self.checks: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.message_public_id = ""
        self.message_id = 0
        self.paused_on = ""
        self.names_before: dict[str, str] = {}
        self.started = time.monotonic()
        self.slow_requests: list[str] = []

    # ------------------------------------------------------------------ plumbing

    def compose_env(self) -> dict[str, str]:
        return {
            **os.environ,
            "COMPOSE_PROJECT_NAME": PROJECT,
            "HONCHO_SANDBOX_IMAGE": self.args.image_under_test,
            "SLICE_APP_IMAGE": self.app_image,
            "SLICE_API_START": self.api_start,
            "SLICE_PAUSE_DIR": str(self.pause_dir),
            "SLICE_PAUSE_MATCH": self.canary,
            "SLICE_STALE_MINUTES": "1" if self.args.release_on == "other" else "5",
            "SANDBOX_API_PORT": str(API_PORTS["api-1"]),
            "SLICE_API2_PORT": str(API_PORTS["api-2"]),
            "SANDBOX_DB_PORT": str(DB_PORT),
            "SANDBOX_REDIS_PORT": str(REDIS_PORT),
        }

    def compose(
        self, *argv: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            "docker", "compose", "-p", PROJECT,
            "-f", str(HERE / "compose.yml"),
            "-f", str(HERE / "compose.mock.yml"),
            "-f", str(HERE / "compose.slice.yml"),
            *argv,
        ]  # fmt: skip
        result = subprocess.run(
            cmd, env=self.compose_env(), capture_output=True, text=True
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"compose {' '.join(argv)} failed:\n{result.stdout}\n{result.stderr}"
            )
        return result

    def step(self, name: str, ok: bool, detail: str = "") -> None:
        elapsed = time.monotonic() - self.started
        print(
            f"[{'ok' if ok else 'FAIL'}] +{elapsed:.0f}s {name}: {detail}", flush=True
        )
        self.steps.append({"name": name, "ok": ok, "detail": detail})

    def check(
        self,
        check_id: str,
        surface: str,
        replica: str,
        phase: str,
        passed: bool | None,
        evidence: str,
    ) -> None:
        label = {True: "pass", False: "FAIL", None: "n/a"}[passed]
        print(f"  [{label}] {phase} {replica} {check_id}: {evidence}", flush=True)
        self.checks.append(
            {
                "id": check_id,
                "surface": surface,
                "replica": replica,
                "phase": phase,
                "pass": passed,
                "evidence": evidence,
            }
        )

    def http(
        self, replica: str, method: str, path: str, **kwargs: Any
    ) -> httpx.Response:
        url = f"http://127.0.0.1:{API_PORTS[replica]}/v3{path}"
        start = time.monotonic()
        try:
            response = httpx.request(method, url, timeout=60, **kwargs)
        except httpx.HTTPError as exc:
            response = httpx.Response(599, text=repr(exc))
        elapsed = time.monotonic() - start
        if elapsed > 1:
            self.slow_requests.append(
                f"{replica} {method} {path.split('?')[0][-40:]} {elapsed:.1f}s"
            )
        return response

    def sql(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        with psycopg.connect(DB_URL, autocommit=True) as conn:
            cur = conn.execute(query, params)
            return cur.fetchall() if cur.description else []

    def wait_for(self, what: str, predicate: Any, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if predicate():
                    return True
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
        self.notes.append(f"timed out after {timeout}s waiting for {what}")
        return False

    def wait_healthy(self, timeout: float = 90) -> bool:
        def healthy() -> bool:
            for port in API_PORTS.values():
                httpx.get(
                    f"http://127.0.0.1:{port}/health", timeout=2
                ).raise_for_status()
            return True

        return self.wait_for("api replicas healthy", healthy, timeout)

    # ------------------------------------------------------------------ scenario

    def up(self) -> None:
        self.compose("down", "-v", "--remove-orphans", check=False)
        shutil.rmtree(self.pause_dir, ignore_errors=True)
        self.pause_dir.mkdir(parents=True)
        self.pause_dir.chmod(0o777)
        self.compose("up", "-d", "--wait")
        self.step("topology-up", self.wait_healthy(), f"image {self.app_image}")

    def seed(self) -> None:
        api = "api-1"
        self.http(api, "POST", "/workspaces", json={"id": WORKSPACE}).raise_for_status()
        for peer in OBSERVERS:
            self.http(
                api, "POST", f"/workspaces/{WORKSPACE}/peers", json={"id": peer}
            ).raise_for_status()
        # alice is observed (and observes herself); bob and carol observe others.
        # bob is not observed, so his context message makes no work unit of its own.
        peers = {
            "alice": {"observe_me": True, "observe_others": False},
            "bob": {"observe_me": False, "observe_others": True},
            "carol": {"observe_me": False, "observe_others": True},
        }
        self.http(
            api,
            "POST",
            f"/workspaces/{WORKSPACE}/sessions",
            json={"id": SESSION, "peers": peers},
        ).raise_for_status()
        response = self.http(
            api,
            "POST",
            f"/workspaces/{WORKSPACE}/sessions/{SESSION}/messages",
            json={
                "messages": [
                    {"peer_id": "bob", "content": "what did you name the cat?"},
                    {
                        "peer_id": OBSERVED,
                        "content": f"the cat is called {self.canary}, after the pylon",
                    },
                ]
            },
        )
        response.raise_for_status()
        self.message_public_id = response.json()[1]["id"]
        self.message_id = self.sql(
            "SELECT id FROM messages WHERE public_id = %s", (self.message_public_id,)
        )[0][0]
        self.step(
            "seed-messages",
            True,
            f"M={self.message_public_id} (id {self.message_id}) canary {self.canary}",
        )

        # Stale cards: the mock never echoes message text, so these are written
        # through the card route as if a dream had derived them from M.
        for observer in OBSERVERS:
            params = {} if observer == OBSERVED else {"target": OBSERVED}
            self.http(
                api,
                "PUT",
                f"/workspaces/{WORKSPACE}/peers/{observer}/card",
                params=params,
                json={"peer_card": [f"alice's cat is named {self.canary}"]},
            ).raise_for_status()
        # Stale summaries: no summary route accepts writes, so both tiers go in
        # through SQL in the summarizer's storage format.
        summaries = {
            key: {
                "content": f"alice said her cat is called {self.canary}",
                "message_id": self.message_id,
                "summary_type": tier,
                "created_at": now_iso(),
                "token_count": 12,
            }
            for tier, key in SUMMARY_KEYS.items()
        }
        self.sql(
            "UPDATE sessions SET internal_metadata = internal_metadata"
            " || jsonb_build_object('summaries', %s::jsonb)"
            " WHERE workspace_name = %s AND name = %s",
            (json.dumps(summaries), WORKSPACE, SESSION),
        )
        self.step(
            "seed-stale-card-and-summary", True, "cards via API, summaries via SQL"
        )
        self.names_before = self.name_ids()

    def wait_paused(self) -> None:
        paused = self.wait_for(
            "paused batch",
            lambda: any(self.pause_dir.glob("paused-*.json")),
            60,
        )
        if paused:
            marker_path = next(self.pause_dir.glob("paused-*.json"))
            marker = json.loads(marker_path.read_text())
            self.paused_on = marker["hostname"]
            in_input = self.message_id in marker["message_ids"]
            self.step(
                "batch-paused",
                in_input and len(marker["observers"]) > 1,
                f"on {self.paused_on}, observers {marker['observers']},"
                + f" M in input: {in_input}",
            )
        else:
            self.step("batch-paused", False, "no batch reached the pause hook")
        embedded = self.wait_for(
            "message embedding",
            lambda: self.sql(
                "SELECT 1 FROM message_embeddings WHERE message_id = %s"
                " AND embedding IS NOT NULL",
                (self.message_public_id,),
            ),
            30,
        )
        self.step("message-embedded", embedded, "embedding present before delete")

    def warm_reads(self) -> None:
        # Every read surface once per replica, so the post-delete reads can hit
        # entries cached while M was live.
        checks, self.checks = self.checks, []
        for replica in API_PORTS:
            self.api_checks(replica, "warm")
        cached = sum(1 for c in self.checks if c["pass"] is False)
        self.checks = checks
        self.step("warm-reads", True, f"{cached} surfaces showed the canary pre-delete")

    def stop_redis(self) -> None:
        self.compose("stop", "redis")
        self.step("redis-stopped", True, "")

    def delete(self) -> None:
        if self.args.delete_action == "api":
            response = self.http(
                "api-1",
                "DELETE",
                f"/workspaces/{WORKSPACE}/sessions/{SESSION}/messages/"
                + self.message_public_id,
            )
            ok = response.status_code < 300
            self.step(
                "delete-api",
                ok,
                f"DELETE via api-1 -> {response.status_code} {response.text[:200]}",
            )
            return
        try:
            self.sql("DELETE FROM messages WHERE id = %s", (self.message_id,))
            self.step("delete-sql", True, "bare DELETE FROM messages succeeded")
        except psycopg.Error as exc:
            self.step("delete-sql-bare", False, f"bare DELETE blocked: {exc}".strip())
            # The smallest change that lets the row go: detach the queue rows
            # that reference it, leaving everything else as it was.
            self.sql(
                "UPDATE queue SET message_id = NULL WHERE message_id = %s",
                (self.message_id,),
            )
            self.sql("DELETE FROM messages WHERE id = %s", (self.message_id,))
            self.step(
                "delete-sql",
                True,
                "queue.message_id detached, then DELETE FROM messages succeeded",
            )

    def release(self) -> None:
        if self.args.release_on == "other" and self.paused_on:
            # The held batch lives in process memory, so "another replica" means
            # killing the holder and letting the survivor re-claim the stale unit.
            holder = "deriver" if self.paused_on == "deriver-1" else "deriver-2"
            self.compose("kill", holder)
            (self.pause_dir / "release").touch()
            survivor = "deriver-2" if holder == "deriver" else "deriver-1"
            self.step("release", True, f"killed {holder}; {survivor} re-claims")
            done = self.wait_for("takeover", self.batch_settled, 100)
            self.step("batch-settled", done, self.batch_outcome())
            self.compose("up", "-d", "--no-deps", holder)
            return
        (self.pause_dir / "release").touch()
        released = self.wait_for(
            "released marker",
            lambda: any(self.pause_dir.glob("released-*.json")),
            20,
        )
        self.step("release", released, f"released in place on {self.paused_on}")
        done = self.wait_for("batch to settle", self.batch_settled, 60)
        self.step("batch-settled", done, self.batch_outcome())
        holders = sorted(p.name for p in self.pause_dir.glob("paused-*.json"))
        self.step(
            "single-holder",
            len(holders) == 1,
            f"replicas that reached the pause with M: {holders}",
        )

    def batch_settled(self) -> bool:
        pending = self.sql(
            "SELECT count(*) FROM queue WHERE task_type = 'representation'"
            " AND workspace_name = %s AND NOT processed",
            (WORKSPACE,),
        )[0][0]
        claims = self.sql(
            "SELECT count(*) FROM active_queue_sessions WHERE work_unit_key LIKE %s",
            (f"representation:{WORKSPACE}:%",),
        )[0][0]
        return pending == 0 and claims == 0

    def batch_outcome(self) -> str:
        rows = self.sql(
            "SELECT id, processed, error, message_id FROM queue"
            " WHERE task_type = 'representation' AND workspace_name = %s",
            (WORKSPACE,),
        )
        return f"representation queue rows (id, processed, error, message_id): {rows}"

    def swap(self) -> None:
        if not self.args.previous_image:
            self.step("swap", True, "skipped: no --previous-image")
            return
        self.app_image = self.args.previous_image
        services = ["api", "api-2", "deriver", "deriver-2"]
        self.compose("up", "-d", "--no-deps", "--force-recreate", "api")
        started = self.wait_for(
            "previous api",
            lambda: self.api_ok("api-1") or self.container_exited("api"),
            60,
        ) and self.api_ok("api-1")
        if not started:
            logs = self.compose("logs", "--tail", "15", "api", check=False).stdout
            self.step("swap-api-with-migrations", False, logs.strip()[-900:])
            self.api_start = MIGRATION_FREE_START
            self.compose("up", "-d", "--no-deps", "--force-recreate", "api")
            started = self.wait_for("previous api", lambda: self.api_ok("api-1"), 45)
        self.compose("up", "-d", "--no-deps", "--force-recreate", *services[1:])
        healthy = started and self.wait_healthy(60)
        self.warm_up()
        self.step(
            "swap",
            healthy,
            f"api, api-2, deriver, deriver-2 -> {self.app_image}"
            + (
                " (migration-free start)"
                if self.api_start == MIGRATION_FREE_START
                else ""
            ),
        )

    def api_ok(self, replica: str) -> bool:
        url = f"http://127.0.0.1:{API_PORTS[replica]}/health"
        try:
            return httpx.get(url, timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    def container_exited(self, service: str) -> bool:
        result = self.compose(
            "ps", "-a", "--format", "{{.State}}", service, check=False
        )
        return "exited" in result.stdout

    def warm_up(self) -> None:
        # The first search on a fresh process pays one-off client start-up cost.
        for replica in API_PORTS:
            self.http(
                replica,
                "POST",
                f"/workspaces/{WORKSPACE}/sessions/{SESSION}/search",
                json={"query": "warm up"},
            )

    def start_redis(self) -> None:
        self.compose("start", "redis")
        self.step("redis-started", True, "")
        # Let derivers poll and reconcilers tick against the previous image.
        time.sleep(self.args.settle_seconds)

    # ------------------------------------------------------------------ checks

    def name_ids(self) -> dict[str, str]:
        rows = self.sql(
            "SELECT 'peer:' || name, id FROM peers WHERE workspace_name = %s"
            " UNION ALL SELECT 'session:' || name, id FROM sessions"
            " WHERE workspace_name = %s"
            " UNION ALL SELECT 'workspace:' || name, id FROM workspaces WHERE name = %s",
            (WORKSPACE, WORKSPACE, WORKSPACE),
        )
        return {name: row_id for name, row_id in rows}

    def derived_documents(self) -> list[tuple[Any, ...]]:
        return self.sql(
            "SELECT id, observer, deleted_at IS NULL, embedding IS NOT NULL, sync_state"
            " FROM documents WHERE workspace_name = %s"
            " AND internal_metadata -> 'message_ids' @> %s::jsonb ORDER BY id",
            (WORKSPACE, json.dumps([self.message_id])),
        )

    def run_checks(self, phase: str) -> None:
        print(f"-- checks: {phase}", flush=True)
        for replica in API_PORTS:
            try:
                self.api_checks(replica, phase)
            except (httpx.HTTPError, ValueError) as exc:
                self.check("api-reachable", "HTTP", replica, phase, False, repr(exc))
        self.db_checks(phase)

    def contains_canary(self, response: httpx.Response) -> bool:
        return self.canary in response.text

    def api_checks(self, replica: str, phase: str) -> None:
        base = f"/workspaces/{WORKSPACE}/sessions/{SESSION}"
        response = self.http(
            replica, "GET", f"{base}/messages/{self.message_public_id}"
        )
        self.check(
            "message-hidden-get", "GET message", replica, phase,
            response.status_code == 404,
            f"{response.status_code} canary={self.contains_canary(response)}",
        )  # fmt: skip
        response = self.http(replica, "POST", f"{base}/messages/list", json={})
        self.check(
            "message-hidden-list", "POST messages/list", replica, phase,
            response.status_code == 200 and not self.contains_canary(response),
            f"{response.status_code} canary={self.contains_canary(response)}",
        )  # fmt: skip
        for surface, path in (
            ("session search", f"{base}/search"),
            ("workspace search", f"/workspaces/{WORKSPACE}/search"),
            ("peer search", f"/workspaces/{WORKSPACE}/peers/{OBSERVED}/search"),
        ):
            response = self.http(replica, "POST", path, json={"query": self.canary})
            self.check(
                "message-hidden-search", surface, replica, phase,
                response.status_code == 200 and not self.contains_canary(response),
                f"{response.status_code} canary={self.contains_canary(response)}",
            )  # fmt: skip
        response = self.http(replica, "GET", f"{base}/context")
        self.check(
            "context-clean", "GET session context", replica, phase,
            response.status_code == 200 and not self.contains_canary(response),
            f"{response.status_code} canary={self.contains_canary(response)}",
        )  # fmt: skip

        response = self.http(replica, "GET", f"{base}/summaries")
        body = response.json() if response.status_code == 200 else {}
        for tier in ("short", "long"):
            content = json.dumps(body.get(f"{tier}_summary"))
            self.check(
                f"summary-clean-{tier}", "GET session summaries", replica, phase,
                response.status_code == 200 and self.canary not in content,
                f"{response.status_code} canary={self.canary in content}",
            )  # fmt: skip

        for observer in OBSERVERS:
            params = {} if observer == OBSERVED else {"target": OBSERVED}
            response = self.http(
                replica,
                "GET",
                f"/workspaces/{WORKSPACE}/peers/{observer}/card",
                params=params,
            )
            self.check(
                f"card-clean-{observer}", "GET peer card", replica, phase,
                response.status_code == 200 and not self.contains_canary(response),
                f"{response.status_code} canary={self.contains_canary(response)}",
            )  # fmt: skip

        derived_ids = {row[0] for row in self.derived_documents()}
        response = self.http(
            replica,
            "POST",
            f"/workspaces/{WORKSPACE}/conclusions/list",
            json={},
            params={"size": 100},
        )
        listed = (
            {item["id"] for item in response.json().get("items", [])}
            if response.status_code == 200
            else set()
        )
        leaked = sorted(listed & derived_ids)
        self.check(
            "conclusion-hidden-api", "POST conclusions/list", replica, phase,
            response.status_code == 200 and not leaked,
            f"{response.status_code} listed {len(listed)}, derived-from-M visible: {leaked}",
        )  # fmt: skip

    def db_checks(self, phase: str) -> None:
        db = "db"
        rows = self.sql(
            "SELECT id FROM messages WHERE public_id = %s",
            (self.message_public_id,),
        )
        self.check("message-row-gone", "messages", db, phase, not rows, f"rows {rows}")
        rows = self.sql(
            "SELECT id, sync_state, embedding IS NOT NULL FROM message_embeddings"
            " WHERE message_id = %s",
            (self.message_public_id,),
        )
        self.check(
            "message-embedding-gone", "message_embeddings", db, phase,
            not rows, f"rows {rows}",
        )  # fmt: skip

        documents = self.derived_documents()
        for observer in OBSERVERS:
            live = [d for d in documents if d[1] == observer and d[2]]
            self.check(
                f"no-derived-conclusion-{observer}",
                f"documents ({observer}->{OBSERVED}, message_ids @> M)",
                db, phase, not live,
                f"live {len(live)} of {len([d for d in documents if d[1] == observer])}",
            )  # fmt: skip
        vectors = [d for d in documents if d[3]]
        self.check(
            "no-derived-vector", "documents.embedding (pgvector)", db, phase,
            not vectors,
            f"{len(vectors)} derived docs carry a vector"
            + f" (sync_state {sorted({d[4] for d in vectors})})",
        )  # fmt: skip

        cards = self.sql(
            "SELECT name, internal_metadata::text FROM peers WHERE workspace_name = %s",
            (WORKSPACE,),
        )
        for observer in OBSERVERS:
            stored = next((meta for name, meta in cards if name == observer), "")
            self.check(
                f"card-row-clean-{observer}", "peers.internal_metadata", db, phase,
                self.canary not in stored, f"canary={self.canary in stored}",
            )  # fmt: skip
        stored = self.sql(
            "SELECT internal_metadata -> 'summaries' FROM sessions"
            " WHERE workspace_name = %s AND name = %s",
            (WORKSPACE, SESSION),
        )
        summaries = stored[0][0] or {} if stored else {}
        for tier, key in SUMMARY_KEYS.items():
            content = json.dumps(summaries.get(key))
            self.check(
                f"summary-row-clean-{tier}", "sessions.internal_metadata", db, phase,
                self.canary not in content, f"canary={self.canary in content}",
            )  # fmt: skip

        # Rows for M, plus representation rows whose message is gone or detached:
        # the batch fetch joins on the message, so those can never complete.
        queued = self.sql(
            "SELECT id, task_type, message_id FROM queue WHERE workspace_name = %s"
            " AND NOT processed AND (message_id = %s OR (task_type = 'representation'"
            " AND (message_id IS NULL"
            " OR message_id NOT IN (SELECT id FROM messages))))",
            (WORKSPACE, self.message_id),
        )
        self.check(
            "no-queued-work", "queue (unprocessed for M or orphaned)", db, phase,
            not queued, f"rows (id, task_type, message_id) {queued}",
        )  # fmt: skip
        claims = self.sql(
            "SELECT work_unit_key FROM active_queue_sessions WHERE work_unit_key LIKE %s",
            (f"%{WORKSPACE}:{SESSION}%",),
        )
        self.check(
            "no-active-claim", "active_queue_sessions", db, phase,
            not claims, f"claims {claims}",
        )  # fmt: skip

        current = self.name_ids()
        recreated = sorted(
            name
            for name, row_id in current.items()
            if name in self.names_before and self.names_before[name] != row_id
        )
        appeared = sorted(set(current) - set(self.names_before))
        self.check(
            "no-recreated-name", "workspaces/peers/sessions", db, phase,
            not recreated and not appeared,
            f"recreated {recreated}, new {appeared}"
            + " (message delete removes no name, so this guards get_or_create paths)",
        )  # fmt: skip

        if phase != "after-delete":
            released = any(self.pause_dir.glob("released-*.json"))
            wrote = [d[0] for d in documents]
            outcome = (
                "wrote"
                if wrote
                else ("rejected" if released else "never-ran (holder killed)")
            )
            self.check(
                "paused-batch-outcome", "documents from the paused batch", db, phase,
                outcome != "wrote",
                f"{outcome}: {len(wrote)} docs carry M in message_ids"
                + f" (observers {sorted({d[1] for d in documents})})",
            )  # fmt: skip

    # ------------------------------------------------------------------ driver

    def execute(self) -> dict[str, Any]:
        started = time.monotonic()
        started_at = now_iso()
        try:
            self.up()
            self.seed()
            self.wait_paused()
            self.warm_reads()
            self.stop_redis()
            self.delete()
            self.step("delete-done", True, "starting after-delete reads")
            self.run_checks("after-delete")
            self.release()
            self.run_checks("after-release")
            self.step("after-release-reads-done", True, "")
            self.swap()
            self.start_redis()
            self.run_checks("after-swap")
        except Exception as exc:  # noqa: BLE001
            self.step("runner-error", False, repr(exc))
        failed = [c for c in self.checks if c["pass"] is False]
        blocked = any(s["name"] == "runner-error" for s in self.steps)
        return {
            "label": self.args.label,
            "image_under_test": self.args.image_under_test,
            "previous_image": self.args.previous_image,
            "started_at": started_at,
            "duration_s": round(time.monotonic() - started),
            "topology": {
                "api_replicas": 2,
                "derivers": 2,
                "redis_down_during": "delete+reads+release+swap",
                "delete_action": self.args.delete_action,
                "release_on": self.args.release_on,
                "paused_on": self.paused_on,
            },
            "steps": self.steps,
            "checks": self.checks,
            "verdict": "blocked" if blocked else ("fail" if failed else "pass"),
            "notes": "; ".join(
                [f"canary {self.canary}", f"{len(failed)} failing checks", *self.notes]
            ),
            "slow_requests": self.slow_requests,
        }


def main() -> None:
    parser = argparse.ArgumentParser(prog="slice.sh run")
    parser.add_argument("--label", required=True)
    parser.add_argument("--image-under-test", required=True)
    parser.add_argument("--previous-image", default="")
    parser.add_argument("--delete-action", choices=["api", "sql"], default="api")
    parser.add_argument("--release-on", choices=["same", "other"], default="same")
    parser.add_argument("--settle-seconds", type=float, default=8)
    parser.add_argument("--results-dir", default=str(HERE / ".slice" / "results"))
    args = parser.parse_args()

    result = Run(args).execute()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{args.label}.json"
    out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    failed = sum(1 for c in result["checks"] if c["pass"] is False)
    print(
        f"\n{result['verdict']}: {failed}/{len(result['checks'])} checks failed -> {out}"
    )
    sys.exit(0 if result["verdict"] == "pass" else 1)


if __name__ == "__main__":
    main()
