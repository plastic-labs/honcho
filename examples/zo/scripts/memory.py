#!/usr/bin/env python3
"""Small Zo Skill wrapper around the official Honcho CLI."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

def default_user_peer() -> str:
    for name in ("HONCHO_USER_ID", "HONCHO_PEER_ID", "ZO_USER", "ZO_USERNAME", "USER"):
        value = os.environ.get(name)
        if value and value not in {"root", "zo"}:
            return value
    return "user"


DEFAULT_USER = default_user_peer()
DEFAULT_ASSISTANT = os.environ.get("HONCHO_ASSISTANT_ID", "assistant")
DEFAULT_SESSION = os.environ.get("HONCHO_SESSION_ID", "default")
DEFAULT_WORKSPACE = os.environ.get("HONCHO_WORKSPACE_ID", "default")


def fail(message: str, code: int = 1):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def ensure_cli() -> str:
    binary = shutil.which("honcho") or str(Path.home() / ".local/bin/honcho")
    if Path(binary).exists():
        return binary
    if not shutil.which("uv"):
        fail("Honcho CLI is missing and `uv` is not installed. Install uv, then run: uv tool install honcho-cli")
    subprocess.run(["uv", "tool", "install", "honcho-cli"], check=True)
    binary = shutil.which("honcho") or str(Path.home() / ".local/bin/honcho")
    if not Path(binary).exists():
        fail("Honcho CLI installed but `honcho` was not found. Try opening a new shell or add ~/.local/bin to PATH.")
    return binary


def require_key():
    if not os.environ.get("HONCHO_API_KEY"):
        fail("HONCHO_API_KEY is missing. Add it in Zo Settings > Advanced > Secrets.", 3)


def env() -> dict[str, str]:
    out = os.environ.copy()
    out["PATH"] = f"{Path.home() / '.local/bin'}:{out.get('PATH', '')}"
    out.setdefault("HONCHO_JSON", "1")
    return out


def scope(args, include_session: bool = True) -> list[str]:
    flags = ["--workspace", args.workspace or DEFAULT_WORKSPACE]
    if getattr(args, "user", None):
        flags += ["--peer", args.user]
    if include_session and getattr(args, "session", None):
        flags += ["--session", args.session]
    return flags


def run_honcho(args_list: list[str], *, capture: bool = False, input_text: str | None = None, allow_doctor_partial: bool = False):
    require_key()
    binary = ensure_cli()
    cmd = [binary, *args_list]
    if "--json" not in cmd:
        cmd.append("--json")
    result = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=capture,
        env=env(),
    )
    if result.returncode != 0:
        if allow_doctor_partial and args_list and args_list[0] == "doctor":
            text = result.stdout or result.stderr or ""
            try:
                data = json.loads(text)
                checks = data.get("checks", [])
                critical_failures = [c for c in checks if not c.get("ok") and c.get("check") != "Config file"]
                if not critical_failures:
                    return text
            except Exception:
                pass
        message = (result.stderr or result.stdout or "Honcho command failed").strip()
        fail(message, result.returncode)
    return result.stdout if capture else ""


def parse_messages(args) -> list[dict[str, str]]:
    raw = args.text
    if not raw and not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
    if not raw:
        fail("Provide message text, JSON, or pipe JSON into stdin.")
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            messages = []
            for item in data:
                if not isinstance(item, dict) or "content" not in item:
                    fail("JSON messages must be objects with at least a content field.")
                messages.append({"role": str(item.get("role", args.role)), "content": str(item["content"])})
            return messages
    except json.JSONDecodeError:
        pass
    return [{"role": args.role, "content": raw}]


def peer_for_role(role: str, args) -> str:
    normalized = role.lower()
    if normalized in {"assistant", "ai", "bot"}:
        return args.assistant or DEFAULT_ASSISTANT
    return args.user or DEFAULT_USER


def cmd_save(args):
    session = args.session or DEFAULT_SESSION
    workspace = args.workspace or DEFAULT_WORKSPACE
    messages = parse_messages(args)
    run_honcho(["session", "create", session, "--peers", f"{args.user or DEFAULT_USER},{args.assistant or DEFAULT_ASSISTANT}", "--workspace", workspace], capture=True)
    for item in messages:
        peer = peer_for_role(item["role"], args)
        metadata = json.dumps({"role": item["role"], "source": "zo-skill:honcho-memory"})
        run_honcho(["message", "create", item["content"], "--peer", peer, "--session", session, "--workspace", workspace, "--metadata", metadata], capture=True)
    print(f"Saved {len(messages)} message(s) to Honcho session '{session}'.")


def cmd_ask(args):
    output = run_honcho(["peer", "chat", args.question, *scope(args), "--reasoning", args.reasoning], capture=True)
    print(output.strip())


def cmd_context(args):
    command = ["session", "context", args.session or DEFAULT_SESSION, "--workspace", args.workspace or DEFAULT_WORKSPACE]
    if args.tokens:
        command += ["--tokens", str(args.tokens)]
    if not args.summary:
        command.append("--no-summary")
    output = run_honcho(command, capture=True)
    print(output.strip())


def cmd_messages(args):
    command = ["message", "list", args.session or DEFAULT_SESSION, "--workspace", args.workspace or DEFAULT_WORKSPACE, "--last", str(args.limit)]
    if args.user:
        command += ["--peer", args.user]
    output = run_honcho(command, capture=True)
    print(output.strip())


def cmd_search(args):
    output = run_honcho(["conclusion", "search", args.query, "--observer", args.user or DEFAULT_USER, "--workspace", args.workspace or DEFAULT_WORKSPACE, "--top-k", str(args.limit)], capture=True)
    print(output.strip())


def cmd_doctor(args):
    output = run_honcho(["doctor"], capture=True, allow_doctor_partial=True)
    print(output.strip())


def cmd_test(args):
    test_session = args.session
    if not test_session or test_session == DEFAULT_SESSION:
        test_session = f"honcho-memory-test-{int(time.time())}"
    content = f"Honcho CLI skill test at {int(time.time())}"
    save_args = argparse.Namespace(
        text=content,
        role="user",
        user=args.user or DEFAULT_USER,
        assistant=args.assistant or DEFAULT_ASSISTANT,
        session=test_session,
        workspace=args.workspace or DEFAULT_WORKSPACE,
    )
    cmd_save(save_args)
    output = run_honcho(["message", "list", test_session, "--workspace", args.workspace or DEFAULT_WORKSPACE, "--last", "5"], capture=True)
    if content not in output:
        fail("Test message was saved but not found in message list.")
    print(f"OK: Honcho CLI saved and listed a test message in session '{test_session}'.")


def add_common(parser):
    parser.add_argument("--user", default=DEFAULT_USER, help="User peer ID")
    parser.add_argument("--assistant", default=DEFAULT_ASSISTANT, help="Assistant peer ID")
    parser.add_argument("--session", default=DEFAULT_SESSION, help="Session ID")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="Workspace ID")


def build_parser():
    parser = argparse.ArgumentParser(description="Simple Honcho memory CLI wrapper")
    sub = parser.add_subparsers(dest="command", required=True)

    save = sub.add_parser("save", help="Save one message or JSON messages to Honcho")
    save.add_argument("text", nargs="?", help="Message text or JSON messages")
    save.add_argument("--role", default="user", choices=["user", "assistant", "system"], help="Role for plain text input")
    add_common(save)
    save.set_defaults(func=cmd_save)

    ask = sub.add_parser("ask", help="Ask Honcho about remembered context")
    ask.add_argument("question")
    ask.add_argument("--reasoning", default="medium", choices=["minimal", "low", "medium", "high", "max"])
    add_common(ask)
    ask.set_defaults(func=cmd_ask)

    context = sub.add_parser("context", help="Print an LLM-ready context window as JSON")
    context.add_argument("--tokens", type=int, default=0)
    context.add_argument("--summary", action=argparse.BooleanOptionalAction, default=True)
    add_common(context)
    context.set_defaults(func=cmd_context)

    messages = sub.add_parser("messages", help="List recent session messages")
    messages.add_argument("--limit", type=int, default=20)
    add_common(messages)
    messages.set_defaults(func=cmd_messages)

    search = sub.add_parser("search", help="Search Honcho conclusions for the user peer")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)
    add_common(search)
    search.set_defaults(func=cmd_search)

    doctor = sub.add_parser("doctor", help="Check Honcho CLI connectivity")
    doctor.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    doctor.set_defaults(func=cmd_doctor)

    test = sub.add_parser("test", help="Save and list a tiny memory to verify setup")
    add_common(test)
    test.set_defaults(func=cmd_test)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
