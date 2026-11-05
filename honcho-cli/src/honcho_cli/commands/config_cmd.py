"""Config inspection command: ``honcho config``.

The default view fits on one screen: the values the CLI resolved and where
each came from, the integrations wired up on this machine, local stacks, and
anything that needs attention. ``--all`` expands every section and gives the
fix for each problem; ``--json`` prints the complete report to attach to a bug
report. Secrets are redacted in every view (see :mod:`honcho_cli.redact`).
Read-only: writing to ``~/.honcho/config.json`` is done only via
``honcho init``.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Iterator
from typing import Any, Optional

import typer
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from honcho_cli._help import HonchoTyperGroup
from honcho_cli.branding import BRAND
from honcho_cli.common import handle_cmd_flags
from honcho_cli.diagnostics import build_report
from honcho_cli.output import print_json, stdout_console, use_json
from honcho_cli.redact import is_marker

app = typer.Typer(
    cls=HonchoTyperGroup,
    help=(
        "Show resolved config, integrations and local stacks, with secrets redacted.\n\n"
        "Flags override environment variables, which override ~/.honcho/config.json; "
        "built-in defaults apply last. One exception: a valid OAuth login is used "
        "before HONCHO_API_KEY."
    ),
    invoke_without_command=True,
)

_console = Console(stderr=True)


@app.callback(invoke_without_command=True)
def config(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
    show_all: bool = typer.Option(
        False, "--all", help="Expand every section, with the fix for each problem"
    ),
    deep: bool = typer.Option(
        False, "--deep", help="Also probe the API and read server versions over the network"
    ),
    no_network: bool = typer.Option(False, "--no-network", help="Skip Docker and all HTTP checks"),
    profile: Optional[str] = typer.Option(
        None, "--profile", envvar="HONCHO_PROFILE", help="Only include this local stack"
    ),
) -> None:
    """Show resolved config, integrations and local stacks, with secrets redacted.

    Attach `honcho config --json --deep` to bug reports.
    """
    if ctx.invoked_subcommand is not None:
        return
    handle_cmd_flags(json_output=json_output)
    report = build_report(network=not no_network, deep=deep and not no_network, profile=profile)
    if use_json():
        print_json(report)
        return
    _emit(_render_all if show_all else _render_summary, report)
    if not show_all:
        count = sum(1 for w in report["warnings"] if w["severity"] == "warning")
        _console.print(
            Text(
                f"\n{_plural(count, 'warning')} · --all for details and fixes"
                " · --json --deep for bug reports\n",
                style="dim",
            )
        )


def _emit(render: Callable[[dict[str, Any]], None], report: dict[str, Any]) -> None:
    """Render, then drop the padding rich adds after the last cell of each row.

    Without this every line ends in spaces, which follow the report into
    whatever issue or chat it is pasted into.
    """
    with stdout_console.capture() as captured:
        render(report)
    lines = captured.get().split("\n")
    sys.stdout.write("\n".join(_TRAILING.sub(r"\1", line) for line in lines))


_TRAILING = re.compile(r"[ \t]+((?:\x1b\[[0-9;]*m)*)$")


# --------------------------------------------------------------------------- #
# default view


def _render_summary(report: dict[str, Any]) -> None:
    out = stdout_console
    warnings = report["warnings"]
    shown: set[int] = set()

    out.print(_header(report))
    out.print()
    out.print(_grid(_setting_rows(report), first="bold"))
    out.print()
    out.print(_title("Integrations"))
    out.print(_grid(_integration_rows(report, shown)))
    out.print()
    out.print(_title("Local stacks"))
    out.print(_grid(_stack_rows(report, shown)))

    rest = [w for i, w in enumerate(warnings) if i not in shown]
    if rest:
        out.print()
        out.print(_title("Notes"))
        out.print(_grid([[_glyph(w), _note(w)] for w in rest]))


def _setting_rows(report: dict[str, Any], *, full: bool = False) -> list[list[Any]]:
    r = report["resolved"]
    cred = r["credential"]
    rows: list[list[Any]] = [
        ["base_url", r["base_url"]["value"], _dim(r["base_url"]["source"])],
        ["api key", _credential(cred, full=full), _dim(_credential_source(cred))],
    ]
    oauth = r["oauth"]
    if oauth.get("present"):
        rows.append(["login", _oauth(oauth), _dim(_oauth_source(oauth))])
    # The CLI never persists a workspace/peer/session; they are per command.
    # Only worth a row when this shell sets one — it then applies to every
    # `honcho` command (and some plugins read the same env vars).
    for name in ("workspace", "peer", "session"):
        scope = r[name]
        if scope["value"]:
            rows.append([name, scope["value"], _dim(scope["source"] or "")])
        elif full:
            rows.append([name, _dim("not set"), _dim(f"chosen per command: {scope['set_with']}")])
    rows.append(["api", *_api(report["probes"])])
    return rows


def _integration_rows(report: dict[str, Any], shown: set[int]) -> list[list[Any]]:
    integrations = report["integrations"]
    base = str(report["resolved"]["base_url"]["value"]).rstrip("/")
    rows: list[list[Any]] = []
    for name, entry in integrations["detected"].items():
        issues = _issues(report["warnings"], name, shown)
        if issues:
            rows.append([_mark("!"), _bold(name), entry.get("version") or "",
                         Text("; ".join(w["short"] for w in issues), style="yellow")])
            continue
        bits = []
        if entry.get("workspace"):
            bits.append(f"workspace {entry['workspace']}")
        own = [k for k in entry.get("keys", []) if not k.get("matches")]
        if own:
            bits.append("own key" + _key_outcome(own))
        if entry.get("base_url") and str(entry["base_url"]).rstrip("/") != base:
            bits.append(f"→ {entry['base_url']}")
        if not bits and entry.get("host_blocks"):
            bits.append(f"hosts.{entry['host_blocks'][0]}")
        rows.append([_mark("✓"), _bold(name), entry.get("version") or "", " · ".join(bits) or "installed"])
    for name, client in integrations["mcp_clients"].items():
        issues = _issues(report["warnings"], name, shown)
        if issues:
            rows.append([_mark("!"), _bold(name), "MCP", Text("; ".join(w["short"] for w in issues), style="yellow")])
            continue
        target = (client.get("url") or client.get("serverUrl") or client.get("uri")
                  or (f"via {client['command']}" if client.get("command") else "")
                  or ("via args" if client.get("args") else client.get("note", "")))
        own = [k for k in client.get("keys", []) if not k.get("matches")]
        rows.append([_mark("✓"), _bold(name), "MCP", " · ".join(
            p for p in (str(target), ("own key" + _key_outcome(own)) if own else "") if p)])
    if not rows:
        rows.append([_mark("·"), _dim("none detected")])
    return rows


def _stack_rows(report: dict[str, Any], shown: set[int]) -> list[list[Any]]:
    stack = report["local_stack"]
    docker = stack.get("docker") or {}
    rows: list[list[Any]] = []
    stopped: list[str] = []
    if docker.get("available") is False:
        rows.append([_mark("!"), _bold("docker"), "", Text(str(docker.get("error")), style="yellow")])
    for p in stack["profiles"]:
        issues = _issues(report["warnings"], p["name"], shown)
        if not p.get("exists", True):
            rows.append([_mark("!"), _bold(p["name"]), "",
                         Text(issues[0]["short"] if issues else "not found", style="yellow")])
            continue
        port = f":{p['ports']['api']}"
        if issues:
            rows.append([_mark("!"), _bold(p["name"]), port,
                         Text("; ".join(w["short"] for w in issues), style="yellow")])
        elif p["state"] == "running":
            rows.append([_mark("✓"), _bold(p["name"]), port, _image(p)])
        elif p["state"] == "stopped":
            stopped.append(p["name"])
        else:
            rows.append([_mark("·"), _bold(p["name"]), port, _dim(p["state"])])
    if stopped:
        rows.append([_mark("·"), _dim(", ".join(stopped)), "", _dim("stopped")])
    others = stack.get("other_containers")
    if others:
        listed = ", ".join(f"{c['name']} ({c.get('state')})" for c in others)
        rows.append([_mark("·"), _dim("other"), "", _dim(f"Honcho containers: {listed}")])
    if not rows:
        hint = "not checked (--no-network)" if docker.get("checked") is False else "`honcho start` runs Honcho locally"
        rows.append([_mark("·"), _dim("none"), "", _dim(hint)])
    return rows


# --------------------------------------------------------------------------- #
# --all


def _render_all(report: dict[str, Any]) -> None:
    out = stdout_console
    out.print(_header(report))
    out.print(_dim(report["meta"]["precedence"]))

    out.print()
    out.print(_title("Config"))
    out.print(_grid(_setting_rows(report, full=True), first="bold"))

    files = report["files"]["config_file"]
    config_file = report["config_file"]
    out.print()
    out.print(_title("Config file"))
    if config_file is None:
        out.print(_indent(_dim(f"{files['path']} not found")))
    elif "error" in config_file:
        out.print(_indent(Text(f"{files['path']}: {config_file['error']}", style="yellow")))
    else:
        out.print(_indent(_dim(
            f"{files['path']} · mode {files.get('mode')} · {_size(files.get('size'))}"
            f" · modified {files.get('mtime')}"
        )))
        out.print(_grid(list(_kv_rows({k: v for k, v in config_file.items() if k != "hosts"})), first="bold"))
        hosts = config_file.get("hosts")
        if isinstance(hosts, dict):
            for name, block in hosts.items():
                out.print(_indent(_bold(f"hosts.{name}")))
                out.print(_grid(list(_kv_rows(block if isinstance(block, dict) else {"": block})), indent=4))

    env = report["env"]
    out.print()
    out.print(_title("Environment"))
    env_rows = [[k, _value(v)] for k, v in {**env["honcho"], **env["stack_forwarded"]}.items()]
    out.print(_grid(env_rows or [[_dim("no HONCHO_* or stack variables set")]], first="bold"))

    out.print()
    out.print(_title("Integrations"))
    integrations = report["integrations"]
    for name, entry in integrations["detected"].items():
        out.print(_indent(_heading(name, entry.get("version"))))
        out.print(_grid(_integration_detail(entry), first="dim", indent=4))
        for section in ("mcp", "project_mcp", "plugin", "config", "own_file"):
            block = entry.get(section)
            for item in block if isinstance(block, list) else [block]:
                if not isinstance(item, dict):
                    continue
                label = {"own_file": "own config", "mcp": "MCP entry", "project_mcp": "MCP entry"}.get(
                    section, section
                )
                where = " ".join(str(item[k]) for k in ("path", "key") if item.get(k))
                out.print(_indent(Text.assemble(_bold(label), "  ", _dim(where)), 4))
                body = {k: v for k, v in item.items() if k not in ("path", "key", "project")}
                out.print(_grid(list(_kv_rows(body)), first="dim", indent=6))
    for name, client in integrations["mcp_clients"].items():
        out.print(_indent(_heading(name, "MCP client")))
        body = {k: v for k, v in client.items() if k != "keys"}
        rows = list(_kv_rows(body))
        rows += [["keys" if i == 0 else "", _key_line(key)] for i, key in enumerate(client.get("keys", []))]
        out.print(_grid(rows, first="dim", indent=4))
    if not integrations["detected"] and not integrations["mcp_clients"]:
        out.print(_indent(_dim("none detected")))
    out.print(_indent(_dim("checked: " + ", ".join(integrations["checked"]))))

    out.print()
    out.print(_title("Local stacks"))
    _render_stacks(report["local_stack"])

    out.print()
    out.print(_title("API"))
    if "skipped" in report["probes"]:
        out.print(_indent(_dim(report["probes"]["skipped"])))
    else:
        out.print(_grid(_api_rows(report["probes"]), first="dim"))

    out.print()
    out.print(_title("Warnings"))
    if report["warnings"]:
        for w in sorted(report["warnings"], key=lambda w: w["severity"] != "warning"):
            _render_warning(w)
    else:
        out.print(_indent(Text.assemble(_mark("✓"), "  none")))

    meta = report["meta"]
    out.print()
    out.print(_dim(
        f"honcho-cli {meta['cli_version']} · honcho-ai {meta['sdk_version']} · python {meta['python']}"
        f" · {meta['platform']} · {meta['mode']} mode · {meta['generated_at']}"
    ))


def _render_stacks(stack: dict[str, Any]) -> None:
    out = stdout_console
    docker = stack.get("docker") or {}
    if docker.get("checked") is False:
        out.print(_indent(_dim("Docker not checked (--no-network)")))
    elif docker.get("available") is False:
        out.print(_indent(Text(f"Docker unavailable: {docker.get('error')}", style="yellow")))
    for p in stack["profiles"]:
        if not p.get("exists", True):
            out.print(_indent(Text.assemble(_bold(p["name"]), "  ", Text("not found", style="yellow"))))
            continue
        style = {"running": "green", "stopped": "dim", "unhealthy": "yellow"}.get(p["state"], "dim")
        out.print(_indent(Text.assemble(_bold(p["name"]), "  ", Text(p["state"], style=style), "  ",
                                        _dim(p["api_url"]))))
        rows: list[list[Any]] = [["dir", p["dir"]], ["image", _short_image(p["image"])]]
        version = p.get("image_version") or {}
        build = []
        if version.get("version") not in (None, "latest"):
            build.append(f"v{version['version']}")
        if version.get("revision"):
            build.append(f"revision {version['revision'][:12]}")
        if version.get("created"):
            build.append(f"built {version['created'][:10]}")
        if build:
            rows.append(["build", " · ".join(build)])
        if p.get("server_version"):
            rows.append(["server", p["server_version"]])
        services = p.get("services")
        if isinstance(services, dict) and services:
            rows.append(["services", " · ".join(f"{k} {v}" for k, v in services.items())])
        ports = p["ports"]
        rows.append(["ports", f"api {ports['api']} · db {ports['db']} · redis {ports['redis']}"])
        env = p.get("env")
        if isinstance(env, dict):
            for i, line in enumerate(_env_lines(env)):
                rows.append([".env" if i == 0 else "", line])
        toml = p.get("config_toml")
        if isinstance(toml, dict):
            if "error" in toml:
                rows.append(["config.toml", Text(str(toml["error"]), style="yellow")])
            else:
                rows.append(["config.toml", f"{len(toml)} sections: {', '.join(toml)}"])
        out.print(_grid(rows, first="dim", indent=4))
    others = stack.get("other_containers")
    if others:
        out.print(_indent(_bold("other Honcho containers")))
        rows = []
        for c in others:
            build = c.get("image_version") or {}
            rev = f" · {build['revision'][:7]}" if build.get("revision") else ""
            rows.append([c["name"], f"{c['image']}{rev}", _dim(str(c.get("status") or c.get("state")))])
        out.print(_grid(rows, indent=4))
    if not stack["profiles"] and not others and docker.get("available"):
        out.print(_indent(_dim("none — `honcho start` runs Honcho locally")))


def _integration_detail(entry: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for key in ("workspace", "base_url"):
        if entry.get(key):
            rows.append([key, entry[key]])
    if entry.get("host_blocks"):
        rows.append(["config", ", ".join(f"hosts.{b}" for b in entry["host_blocks"])])
    for i, plugin in enumerate(entry.get("plugins") or []):
        rows.append(["plugins" if i == 0 else "", _plugin_line(plugin)])
    if entry.get("statusline"):
        rows.append(["statusline", entry["statusline"]])
    if entry.get("hooks_enabled") is not None:
        rows.append(["hooks", "enabled" if entry["hooks_enabled"] else "disabled"])
    for i, key in enumerate(entry.get("keys", [])):
        rows.append(["keys" if i == 0 else "", _key_line(key)])
    if entry.get("evidence"):
        rows.append(["seen in", _dim(", ".join(entry["evidence"]))])
    return rows


def _plugin_line(plugin: dict[str, Any]) -> Text:
    text = Text(f"{plugin.get('name')} {plugin.get('version') or ''}".strip())
    if plugin.get("scope") == "project" and plugin.get("project"):
        text.append(f"  only in {plugin['project']}", style="dim")
    elif plugin.get("scope") == "user":
        text.append("  all projects" + ("" if plugin.get("enabled_globally", True) else ", not enabled"), style="dim")
    elif plugin.get("scope"):
        text.append(f"  {plugin['scope']} scope", style="dim")
    return text


def _key_line(key: dict[str, Any]) -> Text:
    text = Text(key["masked"], style="magenta") if key.get("masked") else _dim("set (hidden)")
    if key.get("fingerprint"):
        text.append(f"  {key['fingerprint']}", style="dim")
    origin = f"same as {key['matches']}" if key.get("matches") else "own key"
    status = key.get("status")
    against = key.get("checked_against")
    if status == "accepted":
        text.append(f"  {origin} · accepted by {against}", style="green")
    elif status == "scoped":
        text.append(f"  {origin} · valid scoped key ({against})", style="green")
    elif status == "rejected":
        text.append(f"  {origin} · rejected by {against}: {key.get('detail')}", style="yellow")
    elif status == "malformed":
        text.append(f"  {origin} · {key.get('detail')}", style="yellow")
    elif status in ("expired", "unreachable", "error", "untested"):
        text.append(f"  {origin} · {status}: {key.get('detail')}", style="dim")
    else:
        text.append(f"  {origin} · not tested (add --deep)", style="dim")
    text.append(f"\n{key['where']}", style="dim")
    return text


def _key_outcome(keys: list[dict[str, Any]]) -> str:
    """Summary suffix for an integration's own key(s) after --deep."""
    statuses = {k.get("status") for k in keys}
    if statuses <= {"accepted", "scoped"}:
        return ", accepted"
    if "untested" in statuses:
        return ", not testable"
    return ""


def _render_warning(w: dict[str, Any]) -> None:
    head = Text.assemble(_glyph(w), "  ", Text(w["check"], style="bold"))
    if w.get("subject"):
        head.append(f"  {w['subject']}", style="dim")
    body: list[list[Any]] = [["", Text(w["short"])]]
    if w.get("detail"):
        body.append(["", _dim(w["detail"])])
    if w.get("path"):
        body.append(["", _dim(f"at {w['path']}")])
    if w.get("fix"):
        body.append(["", Text.assemble(Text("fix: ", style="cyan"), w["fix"])])
    stdout_console.print(_indent(head))
    stdout_console.print(_grid(body, indent=3))


# --------------------------------------------------------------------------- #
# pieces


def _header(report: dict[str, Any]) -> Text:
    files = report["files"]["config_file"]
    text = Text(files["path"], style="bold")
    if not files["exists"]:
        text.append("  (not found)", style="yellow")
    elif files.get("parse_ok") is False:
        text.append("  (invalid JSON)", style="yellow")
    return text


def _credential(cred: dict[str, Any], *, full: bool) -> Text:
    if cred["mode"] == "none":
        return Text("not set", style="yellow")
    text = Text(cred["masked"], style="magenta") if cred.get("masked") else _dim("set (hidden)")
    if cred["mode"].startswith("oauth"):
        text.append("  OAuth token", style="dim")
    if full and cred.get("fingerprint"):
        text.append(f"  {cred['fingerprint']}", style="dim")
    return text


def _credential_source(cred: dict[str, Any]) -> str:
    source = cred.get("source") or "—"
    shadows = cred.get("shadows", [])
    if cred["mode"].startswith("oauth"):
        # An OAuth token never matches an API key; just say the key sits unused.
        return f"{source} · API key not used" if shadows else source
    parts = [source]
    for s in shadows:
        where = "config.json" if s["source"].startswith("config.json") else s["source"].removeprefix("env ")
        parts.append(f"also in {where}" if s["same_key"] else f"{where} differs, unused")
    return " · ".join(parts)


def _oauth(oauth: dict[str, Any]) -> Text:
    if not oauth.get("host_matches_base_url", True):
        return Text(f"for {oauth.get('host')} (not used)", style="dim")
    left = oauth.get("expires_in_seconds")
    if left is None:
        return Text("expiry unreadable", style="yellow")
    if left > 0:
        return Text(f"valid for {_duration(left)}")
    style = "dim" if oauth.get("refresh_token_present") else "yellow"
    return Text(f"expired {_duration(-left)} ago", style=style)


def _oauth_source(oauth: dict[str, Any]) -> str:
    refresh = "refresh token present" if oauth.get("refresh_token_present") else "no refresh token"
    return f"config.json oauth · {refresh}"


def _api(probes: dict[str, Any]) -> list[Any]:
    if "skipped" in probes:
        reason = probes["skipped"]
        return [_dim("not checked"), _dim("add --deep" if "deep" in reason else reason)]
    api = probes["api"]
    if not api.get("reachable"):
        return [Text("✗ unreachable", style="red"), _dim(api.get("error", ""))]
    version = api.get("server_version")
    timing = f"HTTP {api.get('health_status')} · {api.get('latency_ms')} ms"
    if api.get("authenticated"):
        head = f"✓ server {version}" if version else "✓ reachable"
        accepted = "scoped key accepted" if api.get("key_status") == "scoped" else "key accepted"
        return [Text(head, style="green"), _dim(f"{timing} · {accepted}")]
    if api.get("authenticated") is None:
        return [Text("· key not verified", style="dim"), _dim(f"{timing} · {api.get('detail')}")]
    return [Text("! key rejected", style="yellow"), _dim(f"{timing} · {api.get('detail')}")]


def _api_rows(probes: dict[str, Any]) -> list[list[Any]]:
    if "skipped" in probes:
        return [["", _dim(probes["skipped"])]]
    return [[k, _value(v)] for k, v in probes["api"].items()]


def _image(p: dict[str, Any]) -> Text:
    version = p.get("image_version") or {}
    parts = []
    tag = p.get("server_version") or (version.get("version") if version.get("version") not in (None, "latest") else None)
    if tag:
        parts.append(f"v{tag}")
    if version.get("revision"):
        parts.append(version["revision"][:7])
    if version.get("created"):
        parts.append(f"built {version['created'][:10]}")
    if not parts:
        parts.append(str(p.get("image", "")).rsplit("/", 1)[-1][:40])
    return Text("honcho " + " · ".join(parts))


_MODEL_KEY = re.compile(
    r"^(?:DIALECTIC_LEVELS__(?P<level>\w+?)__|(?P<workload>[A-Z_]+?)_)MODEL_CONFIG__(?P<field>TRANSPORT|MODEL)$"
)
# Shown on their own rows above the .env block.
_ENV_SHOWN_ELSEWHERE = frozenset({"API_PORT", "DB_PORT", "REDIS_PORT", "HONCHO_IMAGE"})


def _env_lines(env: dict[str, Any]) -> list[Text]:
    """``KEY=value`` lines, with the per-workload model pairs folded together."""
    models: dict[str, dict[str, str]] = {}
    lines: list[Text] = []
    for key, value in env.items():
        if key in _ENV_SHOWN_ELSEWHERE:
            continue
        match = _MODEL_KEY.match(key)
        if match and isinstance(value, str):
            name = f"dialectic {match['level']}" if match["level"] else match["workload"].lower().replace("_", " ")
            models.setdefault(name, {})[match["field"].lower()] = value
        else:
            lines.append(Text.assemble(f"{key}=", _value(value)))
    by_model: dict[str, list[str]] = {}
    for name, fields in models.items():
        model = "/".join(fields[f] for f in ("transport", "model") if fields.get(f))
        by_model.setdefault(model, []).append(name)
    for model, names in by_model.items():
        levels = [n.removeprefix("dialectic ") for n in names if n.startswith("dialectic ")]
        names = [n for n in names if not n.startswith("dialectic ")]
        if levels:
            names.append("dialectic " + "/".join(levels))
        lines.append(Text.assemble(Text(model), _dim(f"  {', '.join(names)}")))
    return lines


def _short_image(image: str) -> str:
    repo, sep, digest = image.partition("@sha256:")
    return f"{repo}@sha256:{digest[:12]}…" if sep and len(digest) > 12 else image


def _issues(warnings: list[dict[str, Any]], subject: str, shown: set[int]) -> list[dict[str, Any]]:
    """Warning-severity findings about ``subject``; marks them shown."""
    found = []
    for i, w in enumerate(warnings):
        if w.get("subject") == subject and w["severity"] == "warning":
            shown.add(i)
            found.append(w)
    return found


def _note(w: dict[str, Any]) -> Text:
    text = Text()
    if w.get("subject"):
        text.append(f"{w['subject']}: ", style="bold")
    text.append(w["short"], style="yellow" if w["severity"] == "warning" else "")
    return text


def _kv_rows(obj: Any, prefix: str = "") -> Iterator[list[Any]]:
    """Dotted key/value rows; lists of objects are indexed."""
    if is_marker(obj) or not isinstance(obj, dict):
        yield [prefix, _value(obj)]
        return
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and value and not is_marker(value):
            yield from _kv_rows(value, path)
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            for i, item in enumerate(value):
                yield from _kv_rows(item, f"{path}[{i}]")
        else:
            yield [path, _value(value)]


def _value(value: Any) -> Text:
    if is_marker(value):
        if "placeholder" in value:
            return Text(f"placeholder {value['placeholder']!r}", style="yellow")
        if not value.get("masked"):
            return _dim("set (hidden)")
        text = Text(value["masked"], style="magenta")
        if value.get("fingerprint"):
            text.append(f"  {value['fingerprint']}", style="dim")
        return text
    if value is None:
        return _dim("—")
    if isinstance(value, bool):
        return Text("true" if value else "false")
    if isinstance(value, list) and all(not isinstance(v, (dict, list)) for v in value):
        return Text(", ".join(str(v) for v in value) or "[]")
    if isinstance(value, (dict, list)):
        return Text(json.dumps(value, default=str))
    return Text(str(value))


def _grid(rows: list[list[Any]], *, first: str = "", indent: int = 2) -> Padding:
    width = max((len(r) for r in rows), default=1)
    table = Table.grid(padding=(0, 2))
    for i in range(width):
        last = i == width - 1
        table.add_column(style=first if i == 0 else "", no_wrap=not last, overflow="fold")
    for row in rows:
        table.add_row(*[c if isinstance(c, Text) else Text("" if c is None else str(c)) for c in row])
    return Padding(table, (0, 0, 0, indent), expand=False)


def _indent(renderable: Any, n: int = 2) -> Padding:
    return Padding(renderable, (0, 0, 0, n), expand=False)


def _title(text: str) -> Text:
    return Text(text, style=f"bold {BRAND}")


def _heading(name: str, extra: str | None) -> Text:
    text = Text(name, style="bold")
    if extra:
        text.append(f"  {extra}", style="dim")
    return text


def _bold(text: str) -> Text:
    return Text(text, style="bold")


def _dim(text: str) -> Text:
    return Text(text, style="dim")


def _mark(glyph: str) -> Text:
    return Text(glyph, style={"✓": "green", "!": "yellow", "✗": "red"}.get(glyph, "dim"))


def _glyph(w: dict[str, Any]) -> Text:
    return _mark("!" if w["severity"] == "warning" else "·")


def _plural(n: int, word: str) -> str:
    return f"no {word}s" if n == 0 else f"{n} {word}{'' if n == 1 else 's'}"


def _duration(seconds: float) -> str:
    seconds = int(abs(seconds))
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


def _size(n: Any) -> str:
    if not isinstance(n, int):
        return "?"
    return f"{n / 1024:.1f} KB" if n >= 1024 else f"{n} B"
