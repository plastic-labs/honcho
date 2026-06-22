#!/usr/bin/env python3
"""Fail Honcho embedding compute back to local Ollama when MBP2020 is offline.

This watchdog is intentionally one-way: it switches remote -> local and emails Andy.
It does not automatically switch back to the laptop, because changing embedding runtime
while Honcho is healthy should be a deliberate operator action.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

HONCHO_DIR = Path(os.environ.get("HONCHO_DIR", "/home/pi/honcho"))
CONFIG_PATH = HONCHO_DIR / "config.toml"
STATE_PATH = Path(
    os.environ.get(
        "HONCHO_EMBED_WATCHDOG_STATE",
        "/home/pi/.cache/honcho-embedding-endpoint-watchdog.json",
    )
)

REMOTE_BASE_URL = os.environ.get(
    "HONCHO_REMOTE_EMBEDDING_BASE_URL", "http://192.168.1.145:11434/v1"
)
LOCAL_BASE_URL = os.environ.get(
    "HONCHO_LOCAL_EMBEDDING_BASE_URL", "http://host.docker.internal:11434/v1"
)
LOCAL_PROBE_BASE_URL = os.environ.get(
    "HONCHO_LOCAL_OLLAMA_BASE_URL", "http://localhost:11434"
)
MODEL = os.environ.get("HONCHO_EMBEDDING_MODEL", "mxbai-embed-large")
EMAIL_TARGET = os.environ.get("HONCHO_EMBEDDING_ALERT_TARGET", "email:andylin@gmail.com")
EMAIL_SUBJECT = os.environ.get(
    "HONCHO_EMBEDDING_ALERT_SUBJECT", "[Hermes][Honcho] Embedding fallback"
)
HERMES_BIN = os.environ.get("HERMES_BIN", "/home/pi/.local/bin/hermes")

BASE_URL_RE = re.compile(
    r'(?m)^(\s*base_url\s*=\s*")(?P<url>[^"]+)("\s*)$', re.MULTILINE
)


def now() -> str:
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def load_state() -> dict[str, object]:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {}


def save_state(state: dict[str, object]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(STATE_PATH)


def http_json(url: str, timeout: float = 4.0) -> tuple[bool, object | str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "honcho-embed-watchdog/1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(200_000)
            if resp.status != 200:
                return False, f"HTTP {resp.status}"
            return True, json.loads(body.decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def probe_ollama(base: str) -> tuple[bool, str]:
    ok, payload = http_json(f"{base.rstrip('/')}/api/tags")
    if not ok:
        return False, str(payload)
    models = []
    if isinstance(payload, dict):
        models = [str(m.get("name") or m.get("model") or "") for m in payload.get("models", [])]
    model_ok = any(m == MODEL or m == f"{MODEL}:latest" or m.startswith(f"{MODEL}:") for m in models)
    if not model_ok:
        return False, f"{MODEL} not present; models={models!r}"
    return True, f"ok models={models!r}"


def read_current_base_url() -> str:
    text = CONFIG_PATH.read_text()
    in_embedding_override = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[embedding.MODEL_CONFIG.overrides]":
            in_embedding_override = True
            continue
        if in_embedding_override and stripped.startswith("["):
            break
        if in_embedding_override:
            match = re.match(r'base_url\s*=\s*"([^"]+)"', stripped)
            if match:
                return match.group(1)
    raise RuntimeError(f"Could not find embedding override base_url in {CONFIG_PATH}")


def replace_embedding_base_url(new_url: str) -> bool:
    text = CONFIG_PATH.read_text()
    lines = text.splitlines(keepends=True)
    in_embedding_override = False
    changed = False
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "[embedding.MODEL_CONFIG.overrides]":
            in_embedding_override = True
            out.append(line)
            continue
        if in_embedding_override and stripped.startswith("["):
            in_embedding_override = False
        if in_embedding_override and re.match(r"\s*base_url\s*=", line):
            indent = line[: len(line) - len(line.lstrip())]
            old = re.search(r'"([^"]+)"', line)
            if old and old.group(1) == new_url:
                out.append(line)
            else:
                out.append(f'{indent}base_url = "{new_url}"\n')
                changed = True
            continue
        out.append(line)
    if changed:
        CONFIG_PATH.write_text("".join(out))
    return changed


def run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 120) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout.strip()


def restart_honcho() -> str:
    code, output = run(["docker", "compose", "restart", "api", "deriver"], cwd=HONCHO_DIR, timeout=180)
    if code != 0:
        raise RuntimeError(f"docker compose restart failed ({code}):\n{output}")
    return output


def verify_container_embedding() -> str:
    snippet = r'''
import asyncio, math
from src.config import settings, resolve_embedding_model_config
from src.embedding_client import _EmbeddingClient
cfg = resolve_embedding_model_config(settings.EMBEDDING.MODEL_CONFIG)
print("resolved_base_url=", cfg.base_url)
client = _EmbeddingClient(
    cfg,
    vector_dimensions=settings.EMBEDDING.VECTOR_DIMENSIONS,
    max_input_tokens=settings.EMBEDDING.MAX_INPUT_TOKENS,
    max_tokens_per_request=settings.EMBEDDING.MAX_TOKENS_PER_REQUEST,
    send_dimensions=False,
)
async def main():
    emb = await client.embed("Honcho local fallback watchdog verification")
    print("embedding_dim=", len(emb))
    print("has_nan=", any(math.isnan(x) for x in emb))
asyncio.run(main())
'''
    code, output = run(["docker", "exec", "-i", "honcho-api-1", "python", "-c", snippet], timeout=90)
    if code != 0:
        raise RuntimeError(f"container embedding verification failed ({code}):\n{output}")
    return output


def send_email(body: str) -> str:
    if not Path(HERMES_BIN).exists():
        return f"hermes binary missing at {HERMES_BIN}; email not sent"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as fh:
        fh.write(body)
        path = fh.name
    try:
        code, output = run(
            [HERMES_BIN, "send", "--quiet", "--to", EMAIL_TARGET, "--subject", EMAIL_SUBJECT, "--file", path],
            timeout=60,
        )
        if code != 0:
            return f"email send failed ({code}): {output}"
        return f"email sent to {EMAIL_TARGET} subject={EMAIL_SUBJECT!r}"
    finally:
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass


def build_body(*, old_url: str, remote_reason: str, local_probe: str, restart_output: str, verify_output: str) -> str:
    return f"""## Honcho embedding fallback activated

Honcho was configured to use MBP2020 for embeddings, but the remote Ollama endpoint failed. I switched Honcho back to local Pi Ollama and restarted the Honcho API/deriver containers.

- **Host:** {socket.gethostname()}
- **Time:** {now()}
- **Old endpoint:** `{old_url}`
- **New endpoint:** `{LOCAL_BASE_URL}`
- **Remote probe failure:** `{remote_reason}`
- **Local probe:** `{local_probe}`
- **Model:** `{MODEL}`

### Restart output
```text
{restart_output}
```

### Verification output
```text
{verify_output}
```

### Reference
`REF: HERMES-NOTIFY:honcho:embedding-fallback:mbp2020-offline`
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="probe and report without changing config")
    parser.add_argument("--force", action="store_true", help="force fallback even if remote probe passes")
    args = parser.parse_args()

    state = load_state()
    current_url = read_current_base_url()
    remote_probe_base = REMOTE_BASE_URL.removesuffix("/v1")
    remote_ok, remote_reason = probe_ollama(remote_probe_base)
    local_ok, local_reason = probe_ollama(LOCAL_PROBE_BASE_URL)

    summary = {
        "time": now(),
        "current_url": current_url,
        "remote_ok": remote_ok,
        "remote_reason": remote_reason,
        "local_ok": local_ok,
        "local_reason": local_reason,
    }

    if current_url == LOCAL_BASE_URL and not args.force:
        state.update(summary | {"last_status": "already_local"})
        save_state(state)
        return 0

    if current_url != REMOTE_BASE_URL and not args.force:
        state.update(summary | {"last_status": "unmanaged_endpoint"})
        save_state(state)
        print(f"[SILENT] unmanaged endpoint {current_url}; not changing")
        return 0

    if remote_ok and not args.force:
        state.update(summary | {"last_status": "remote_ok"})
        save_state(state)
        return 0

    if not local_ok:
        state.update(summary | {"last_status": "blocked_local_unhealthy"})
        save_state(state)
        print(f"Honcho embedding fallback BLOCKED: remote failed ({remote_reason}); local failed ({local_reason})")
        return 2

    if args.dry_run:
        print(json.dumps(summary | {"would_switch_to": LOCAL_BASE_URL}, indent=2))
        return 0

    changed = replace_embedding_base_url(LOCAL_BASE_URL)
    restart_output = restart_honcho() if changed or args.force else "config already local; restart skipped"
    verify_output = verify_container_embedding()
    body = build_body(
        old_url=current_url,
        remote_reason=str(remote_reason),
        local_probe=str(local_reason),
        restart_output=restart_output,
        verify_output=verify_output,
    )
    email_result = send_email(body)
    state.update(
        summary
        | {
            "last_status": "switched_to_local",
            "switched_at": now(),
            "email_result": email_result,
            "verify_output": verify_output,
        }
    )
    save_state(state)
    print(f"Honcho embedding fallback activated: {current_url} -> {LOCAL_BASE_URL}; {email_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
