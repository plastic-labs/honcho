"""Deterministic OpenAI-compatible stand-in for every LLM and embedding call.

Runs in its own container with only the standard library, so the old and new
Honcho images talk to the same fake provider over HTTP and neither needs an
API key. Responses are a pure function of the request:

- A structured-output call (the deriver) returns one explicit conclusion per
  ``cmx-<token>`` marker found in the prompt.
- A call that offers ``create_observations_deductive`` (the dreamer's
  deduction specialist) first asks for ``get_recent_observations``, then cites
  the first two ``[id:...]`` it was shown as ``source_ids``, then stops. This
  is the path that writes conclusion provenance, the one
  plastic-labs/honcho#952 changed.
- Any other call (summaries, dialectic, the induction specialist) gets a
  short plain-text answer and no tool calls.
- ``/v1/embeddings`` returns a unit vector derived from a hash of the input.
"""

import base64
import hashlib
import json
import math
import re
import struct
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

MARKER = re.compile(r"cmx-[a-z0-9]{6,}")
OBSERVATION_ID = re.compile(r"\[id:([A-Za-z0-9_-]{21})\]")
DIMENSIONS = 1536


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = cast(list[Any], content)
        return " ".join(
            str(cast(dict[str, Any], part).get("text", ""))
            for part in parts
            if isinstance(part, dict)
        )
    return ""


def _tool_names(body: dict[str, Any]) -> set[str]:
    tools = cast(list[dict[str, Any]], body.get("tools") or [])
    return {str(tool.get("function", {}).get("name", "")) for tool in tools}


def _completion(body: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
    finish = "tool_calls" if message.get("tool_calls") else "stop"
    return {
        "id": "chatcmpl-cmx",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "stub"),
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


def _tool_call(name: str, arguments: dict[str, Any], turn: int) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_cmx_{turn}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def chat(body: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = body.get("messages") or []
    prompt = "\n".join(_text(m.get("content")) for m in messages)

    if body.get("response_format", {}).get("type") in {"json_schema", "json_object"}:
        markers = sorted(set(MARKER.findall(prompt)))
        payload = {
            "explicit": [{"content": f"The peer mentioned {m}."} for m in markers]
        }
        return _completion(body, {"role": "assistant", "content": json.dumps(payload)})

    if "create_observations_deductive" in _tool_names(body):
        tool_results = [m for m in messages if m.get("role") == "tool"]
        called = [
            str(call["function"]["name"])
            for m in messages
            if m.get("role") == "assistant"
            for call in cast(list[dict[str, Any]], m.get("tool_calls") or [])
        ]
        if not tool_results:
            return _completion(
                body, _tool_call("get_recent_observations", {"limit": 20}, 0)
            )
        if "create_observations_deductive" not in called:
            ids: list[str] = []
            for result in tool_results:
                for found in OBSERVATION_ID.findall(_text(result.get("content"))):
                    if found not in ids:
                        ids.append(found)
            if len(ids) >= 2:
                arguments = {
                    "observations": [
                        {
                            "content": "cmx deduction drawn from "
                            + " and ".join(ids[:2]),
                            "source_ids": ids[:2],
                            "premises": ["first premise", "second premise"],
                        }
                    ]
                }
                return _completion(
                    body,
                    _tool_call("create_observations_deductive", arguments, 1),
                )

    return _completion(body, {"role": "assistant", "content": "cmx stub answer."})


def embed(body: dict[str, Any]) -> dict[str, Any]:
    inputs = body.get("input")
    if isinstance(inputs, str):
        inputs = [inputs]
    data: list[dict[str, Any]] = []
    for index, item in enumerate(inputs or []):
        seed = hashlib.sha256(str(item).encode()).digest()
        raw = [(seed[i % len(seed)] - 127.5) + (i % 7) for i in range(DIMENSIONS)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        vector = [x / norm for x in raw]
        if body.get("encoding_format") == "base64":
            encoded: Any = base64.b64encode(
                struct.pack(f"<{DIMENSIONS}f", *vector)
            ).decode()
        else:
            encoded = vector
        data.append({"object": "embedding", "index": index, "embedding": encoded})
    return {
        "object": "list",
        "data": data,
        "model": body.get("model", "stub"),
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._send(200, {"ok": True})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path.endswith("/chat/completions"):
            self._send(200, chat(body))
        elif self.path.endswith("/embeddings"):
            self._send(200, embed(body))
        else:
            self._send(404, {"error": {"message": f"no stub for {self.path}"}})

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"llm-stub {self.command} {self.path}\n")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
