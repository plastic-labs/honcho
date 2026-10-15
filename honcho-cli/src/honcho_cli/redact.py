"""Redaction for diagnostic output (``honcho config``): an allowlist.

A value is printed only when its key says it is safe to print — a workspace,
a model, a port, a session strategy — or when it is a boolean. Everything
else is reported as set and never shown. The decision is made by key name at
every nesting level, and a secret-sounding ancestor (``credentials``,
``[auth]``) hides everything beneath it, so a secret in a shape nobody
anticipated is hidden by default rather than leaked. Three refinements keep
the report useful:

- A recognisable machine-generated key (``hch-…``, ``sk-…``, a JWT, …) is
  shown as its public prefix and last four characters (``hch-v3-…l0bo``)
  plus a short SHA-256 fingerprint, so the same key can be spotted in two
  files without exposing it. The fingerprint is unsalted on purpose — that
  is what makes it comparable — and is only ever computed for those long
  random keys, never for a password someone typed.
- URLs keep their scheme, host, port and, for URL settings, path. The
  username, password, query string and fragment are dropped, however the
  password is spelled.
- ``${VAR}`` references are shown as written: they name a secret, they are
  not one.

Command-line arguments (an MCP server's ``args``) get the same treatment per
token: flags, URLs, paths and package names are shown, anything else hidden.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from honcho_cli.local.env import is_placeholder_key

HIDDEN: dict[str, Any] = {"set": True}

# A key-name part that makes the value (and everything beneath it) secret.
_SECRET_PARTS = frozenset(
    {
        "password", "passwd", "pass", "pwd", "passphrase", "secret", "secrets", "token",
        "apikey", "key", "keys", "credential", "credentials", "creds", "auth",
        "authorization", "bearer", "cookie", "cookies", "signature", "sig", "dsn",
        "webhook", "pin", "otp", "salt", "jwt", "private", "cert", "certificate", "ssh",
    }
)
# A key-name part that makes a string value safe to print.
_SAFE_PARTS = frozenset(
    {
        # where, and who
        "url", "uri", "endpoint", "host", "hostname", "environment", "region", "namespace",
        "workspace", "workspaces", "peer", "peers", "session", "sessions", "ai", "user", "name",
        "names", "profile", "scope", "client", "id", "ids", "project", "agent", "assistant",
        # how
        "model", "models", "transport", "provider", "backend", "mode", "strategy", "level",
        "type", "kind", "format", "source", "protocol", "frequency", "template", "prefix",
        "encoding", "tokenizer", "language", "locale", "effort", "reasoning", "recall",
        "observation", "injection", "statusline", "logging", "channel", "policy", "boundary",
        "choice", "types", "levels", "class", "include", "exclude",
        # what
        "image", "version", "command", "path", "dir", "directory", "file", "schema",
    }
)
# A key-name part that makes a number (or numeric string) safe to print.
_NUMERIC_PARTS = frozenset(
    {
        "port", "ports", "count", "max", "min", "limit", "size", "tokens", "top", "k",
        "threshold", "interval", "timeout", "ttl", "ms", "seconds", "minutes", "hours", "days",
        "dimensions", "cadence", "depth", "chars", "distance", "retries", "workers", "concurrency",
        "expires", "batch", "window", "rate", "temperature", "priority", "ratio", "percent",
        "messages", "conclusions", "results", "version", "recycle", "backoff", "multiplier",
    }
)
# Names whose numeric values are counts even though a part sounds secret
# (HISTORY_TOKEN_LIMIT, MAX_TOKEN_COUNT): only these weak parts are excused.
_WEAK_SECRET_PARTS = frozenset({"token"})
_COUNT_PARTS = frozenset({"limit", "max", "min", "count", "size", "threshold", "budget"})
_COMMAND_WORDS = frozenset(
    {"npx", "bunx", "bun", "node", "deno", "uvx", "uv", "python", "python3", "pipx", "docker",
     "run", "exec", "x", "mcp-remote", "stdio", "serve", "start", "--"}
)

# ``${HONCHO_API_KEY}``, optionally as ``Bearer ${VAR}`` or with a prefix
# (``${env:VAR}`` in Cursor, ``${input:id}`` in VS Code). ``${VAR:-default}``
# is not a reference: the default may be a real key.
_REFERENCE = re.compile(r"^(?:Bearer\s+)?\$\{(?:[A-Za-z]+:[\w.-]+|[A-Za-z_]\w*)\}$", re.IGNORECASE)
_BOOLISH = re.compile(r"^(?:true|false|yes|no|on|off|null|none)$", re.IGNORECASE)
_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
_URL_START = re.compile(r"^[A-Za-z][\w+.-]*://")
_URL_ANY = re.compile(r"[A-Za-z][\w+.-]*://[^\s'\"<>]+")

_KEY_PREFIX = (
    r"hch-[a-z0-9]+-|sk-ant-|sk-proj-|sk-|AIza|gsk_|hf_|ghp_|gho_|ghu_|ghs_"
    r"|github_pat_|xox[abprs]-"
)
_PREFIX_OF = re.compile(rf"^(?:{_KEY_PREFIX}|eyJ)")
_PREFIXED = re.compile(rf"(?<![A-Za-z0-9])(?:{_KEY_PREFIX})[A-Za-z0-9_\-]{{12,}}")
_JWT = re.compile(r"(?<![A-Za-z0-9_])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*")
# RFC 6750 b64token: stops at any quote, bracket, separator or escape.
_BEARER = re.compile(r"\b(Bearer\s+)([A-Za-z0-9\-._~+/]{8,}=*)", re.IGNORECASE)
# ``Authorization: <scheme> <credential>`` in free text, any scheme.
_AUTHZ = re.compile(r"\b(authorization\s*[:=]\s*)(?:([A-Za-z]+)\s+)?([^\s'\",;]+)", re.IGNORECASE)
# ``Some-Secret: value`` / ``api_key=value`` in free text.
_NAMED_SECRET = re.compile(
    r"\b([\w-]*(?:secret|token|password|passwd|api[-_]?key|apikey)[\w-]*\s*[:=]\s*)([^\s'\",;]+)",
    re.IGNORECASE,
)
_FLAG = re.compile(r"^--?[A-Za-z][\w-]*$")
_SCOPED_PACKAGE = re.compile(r"^@[\w.-]+/[\w.-]+(?:@[\w.-]+)?$")
_PATHLIKE = ("/", "./", "../", "~/")
_HEADER_FLAGS = frozenset({"header", "h", "headers"})


def _safe_key(name: str) -> str:
    """A mapping key as printed: a key that is itself a secret is masked."""
    return _masked(name) if _key_prefix(name) is not None else name


def fingerprint(value: str) -> str:
    """``sha256:<8 hex>`` of ``value`` — same input, same fingerprint, every run."""
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def _strip_bearer(value: str) -> str:
    match = re.fullmatch(r"(?i)bearer\s+(\S+)", value.strip())
    return match.group(1) if match else value.strip()


def _key_prefix(token: str) -> str | None:
    """Public prefix of a long machine-generated key, or None if ``token`` isn't one."""
    if _JWT.fullmatch(token):
        return "eyJ"
    match = _PREFIX_OF.match(token)
    if match and len(token) >= 24 and not re.search(r"\s", token):
        return match.group(0)
    return None


def _kind(prefix: str) -> str:
    if prefix == "eyJ":
        return "jwt"
    if prefix.startswith("hch-"):
        return "hch"
    if prefix.startswith("sk-"):
        return "sk"
    return "key"


def secret_marker(value: str) -> dict[str, Any]:
    """The stand-in written wherever a secret was.

    A recognisable key keeps its prefix, last four and fingerprint (a
    ``Bearer <token>`` header is described by its token). Anything else is
    just "set".
    """
    token = _strip_bearer(value)
    prefix = _key_prefix(token)
    if prefix is None:
        return dict(HIDDEN)
    return {
        "redacted": True,
        "kind": _kind(prefix),
        "masked": f"{prefix}…{token[-4:]}",
        "fingerprint": fingerprint(token),
    }


def _masked(token: str) -> str:
    prefix = _key_prefix(token)
    return f"{prefix}…{token[-4:]}" if prefix is not None else "***"


def is_marker(value: Any) -> bool:
    """True for the dicts :func:`redact_tree` substitutes for hidden values."""
    return isinstance(value, dict) and ("redacted" in value or "set" in value or "placeholder" in value)


def marker_text(value: dict[str, Any]) -> str:
    """One-line text for a marker: the masked key, or a placeholder note."""
    if "placeholder" in value:
        return f"placeholder {value['placeholder']!r}"
    return value.get("masked") or "<hidden>"


def safe_url(value: str, *, keep_path: bool = True) -> str:
    """``scheme://host:port[/path]`` — never the userinfo, query or fragment.

    The userinfo runs to the last ``@`` before the host, so a password
    containing ``@``, ``/``, ``?`` or ``#`` is dropped whole. A host that URL
    parsers might read differently (a backslash, whitespace) is not echoed.
    """
    match = re.match(r"^([A-Za-z][\w+.-]*)://(.*)$", value.strip(), re.DOTALL)
    if not match:
        return "<hidden>"
    scheme, rest = match.groups()
    had_userinfo = False
    at = rest.rfind("@")
    if at != -1 and (":" in rest[:at] or "/" not in rest[:at]):
        had_userinfo, rest = True, rest[at + 1 :]
    end = min([i for i in (rest.find(c) for c in "/?#") if i != -1] or [len(rest)])
    host, tail = rest[:end], rest[end:]
    if not host or re.search(r"[\s\\]", host):
        return f"{scheme}://(unreadable host)"
    out = f"{scheme}://{'***@' if had_userinfo else ''}{host}"
    path = re.split(r"[?#]", tail, maxsplit=1)[0]
    if keep_path and path:
        out += _mask_tokens(path)
    if keep_path and re.search(r"[?#]", tail):
        out += "?…"
    return out


def _mask_tokens(text: str) -> str:
    out = _BEARER.sub(
        lambda m: m.group(0) if _REFERENCE.match(m.group(2)) else m.group(1) + _masked(m.group(2)), text
    )
    out = _JWT.sub(lambda m: _masked(m.group(0)), out)
    return _PREFIXED.sub(lambda m: _masked(m.group(0)), out)


def mask_text(value: str) -> str:
    """Mask every secret embedded in free text: URL credentials, auth headers,
    ``name=secret`` pairs and recognisable keys. Used for text the report
    builds itself (warnings, probe details) and as a backstop on shown values.
    """
    out = _URL_ANY.sub(lambda m: safe_url(m.group(0)), value)
    out = _AUTHZ.sub(
        lambda m: m.group(0)
        if _REFERENCE.match(m.group(3))
        else m.group(1) + (f"{m.group(2)} " if m.group(2) else "") + _masked(m.group(3)),
        out,
    )
    out = _NAMED_SECRET.sub(
        lambda m: m.group(0) if _REFERENCE.match(m.group(2)) or _BOOLISH.match(m.group(2))
        else m.group(1) + "***",
        out,
    )
    return _mask_tokens(out)


def _parts(name: str | None) -> set[str]:
    if not name:
        return set()
    snake = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", str(name))
    return {p for p in re.split(r"[^a-z0-9]+", snake.lower()) if p}


def _secret_name(name: str | None) -> bool:
    return bool(_parts(name) & _SECRET_PARTS)


def _safe_name(name: str | None) -> bool:
    return bool(_parts(name) & _SAFE_PARTS)


def _is_count(name: str | None) -> bool:
    parts = _parts(name)
    return bool(parts & _COUNT_PARTS) and not (parts & (_SECRET_PARTS - _WEAK_SECRET_PARTS))


def _numeric_name(name: str | None) -> bool:
    return bool(_parts(name) & (_NUMERIC_PARTS | _SAFE_PARTS))


def is_secret_path(names: list[str] | tuple[str, ...]) -> bool:
    """True when any key on a path names a secret (``hosts.x.webhookUrl``)."""
    return any(_secret_name(n) for n in names)


def redact_value(key: str | None, value: Any, *, parents: tuple[str, ...] = ()) -> Any:
    """Decide one leaf. ``parents`` are the enclosing key names, outermost first."""
    if value is None or isinstance(value, bool):
        return value
    if key is not None and "__" in str(key):
        # A nested env setting (DERIVER_MODEL_CONFIG__OVERRIDES__X): judge it
        # like the same setting in a file — by its last segment and parent.
        *outer, key = [s for s in str(key).split("__") if s] or [str(key)]
        parents = (*parents, *outer)
    secret = _secret_name(key) or any(_secret_name(p) for p in parents)
    safe = _safe_name(key) or bool(parents and _safe_name(parents[-1]))
    if secret and _is_count(key) and not any(_secret_name(p) for p in parents):
        secret = False  # e.g. HISTORY_TOKEN_LIMIT: a number of tokens, not a token
        if not (isinstance(value, (int, float)) or _NUMBER.match(str(value).strip())):
            return dict(HIDDEN)
    if isinstance(value, (int, float)):
        return value if not secret and (safe or _numeric_name(key)) else dict(HIDDEN)
    if not isinstance(value, str):
        return dict(HIDDEN)
    text = value.strip()
    if not text or _REFERENCE.match(text) or _BOOLISH.match(text):
        return value
    if text[:1] in "{[":
        try:
            decoded = json.loads(text)
        except ValueError:
            decoded = None
        if isinstance(decoded, (dict, list)):
            # A dict given as JSON (pydantic reads nested settings this way):
            # redact its contents, not the string.
            return redact_tree(decoded, key=key, parents=parents)
    if _key_prefix(_strip_bearer(text)) is not None:
        return secret_marker(text)
    if secret:
        return {"placeholder": value} if is_placeholder_key(value) else dict(HIDDEN)
    if _NUMBER.match(text):
        return value if safe or _numeric_name(key) else dict(HIDDEN)
    if str(key or "").lower() == "command" and re.search(r"\s", text):
        return " ".join(_args(text.split()))
    if _URL_START.match(text):
        return safe_url(text, keep_path=safe)
    if safe:
        return mask_text(value)
    return dict(HIDDEN)


def _inline(name: str, value: str) -> str:
    """A value inside an argument: shown only if its name allows it and it
    doesn't look like a credential pair (``user:pass``, ``a@b``)."""
    if _REFERENCE.match(value.strip()):
        return value
    shown = redact_value(name, value)
    if isinstance(shown, dict):
        return marker_text(shown) if not shown.get("set") else "<hidden>"
    if not isinstance(shown, str):
        return "<hidden>"
    if shown == value and not _URL_START.match(value) and re.search(r"[:@\s]", value):
        return "<hidden>"
    return shown


def _standalone(text: str) -> bool:
    """An argument that is safe to show on its own: never a bare value."""
    return (
        _FLAG.match(text) is not None
        or text in _COMMAND_WORDS
        or (text.startswith(_PATHLIKE) and "://" not in text)
        or _SCOPED_PACKAGE.match(text) is not None
    )


def _args(tokens: list[Any]) -> list[Any]:
    """Command-line arguments. A token after a ``--flag`` is that flag's value
    and is judged by the flag's name; anything that isn't a flag, URL, path,
    package or known command is hidden."""
    out: list[Any] = []
    flag: str | None = None
    for token in tokens:
        if not isinstance(token, str):
            out.append(redact_tree(token, key="args"))
            flag = None
            continue
        out.append(_arg(token, after=flag))
        text = token.strip()
        flag = text.lstrip("-") if _FLAG.match(text) and text != "--" else None
    return out


def _arg(token: str, *, after: str | None = None) -> str:
    text = token.strip()
    if not text or _REFERENCE.match(text):
        return token
    if _URL_START.match(text):
        return safe_url(text)
    if _key_prefix(_strip_bearer(text)) is not None:
        return _masked(_strip_bearer(text))
    header = re.fullmatch(r"([A-Za-z][\w-]*)\s*:\s*(.+)", text, re.DOTALL)
    if after is not None and after.lower() in _HEADER_FLAGS and header:
        return f"{header.group(1)}: {_inline(header.group(1), header.group(2))}"
    if after is not None and not _standalone(text):
        return _inline(after, text)  # --flag value: the flag names the value
    if _standalone(text):
        return mask_text(token)
    assign = re.fullmatch(r"(--?[A-Za-z][\w-]*|[A-Za-z_][A-Za-z0-9_]*)=(.*)", text, re.DOTALL)
    if assign:
        return f"{assign.group(1)}={_inline(assign.group(1).lstrip('-'), assign.group(2))}"
    if header:
        return f"{header.group(1)}: {_inline(header.group(1), header.group(2))}"
    return "<hidden>"


def redact_tree(obj: Any, *, key: str | None = None, parents: tuple[str, ...] = ()) -> Any:
    """Recursively redact a parsed JSON/TOML/dotenv structure."""
    if isinstance(obj, dict):
        inner = (*parents, key) if key is not None else parents
        return {_safe_key(str(k)): redact_tree(v, key=str(k), parents=inner) for k, v in obj.items()}
    if isinstance(obj, list):
        if key is not None and str(key).lower() == "args":
            return _args(obj)
        return [redact_tree(v, key=key, parents=parents) for v in obj]
    return redact_value(key, obj, parents=parents)
