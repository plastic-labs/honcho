"""Local stack lifecycle: ``honcho start``, ``honcho stop``, ``honcho status``.

Does not mutate ``~/.honcho/config.json``. The CLI stays pointed at whatever
``honcho init`` configured (typically api.honcho.dev). Print the local URL
and a one-shot ``HONCHO_BASE_URL=...`` hint instead.
"""

from __future__ import annotations

import os

import typer
from rich.console import Console

from honcho_cli.branding import BRAND, ICON_FAIL, ICON_OK, ICON_RUN
from honcho_cli.local import (
    DEFAULT_HEALTH_TIMEOUT,
    DEFAULT_IMAGE,
    DEFAULT_PROFILE,
    INFERENCE_MODES,
    STACK_SERVICES,
    SUPPORTED_INFERENCE,
)
from honcho_cli.local.docker import (
    DockerError,
    allocate_host_ports,
    compose_down,
    compose_ps,
    compose_up,
    ensure_docker,
    services_running,
)
from honcho_cli.local.env import is_placeholder_key, read_env_value, render_stack
from honcho_cli.local.health import stack_healthy, wait_for_health
from honcho_cli.local.profile import (
    LocalProfile,
    load_profile,
    resolve_profile_name,
    save_profile,
)
from honcho_cli.output import (
    print_error,
    print_json,
    print_result,
    set_json_mode,
    use_json,
)

_console = Console(stderr=True)


def _die(code: str, message: str, details: dict | None = None) -> None:
    print_error(code, message, details)
    raise typer.Exit(1)


def _step(msg: str) -> None:
    if not use_json():
        _console.print(f"  {ICON_RUN}  {msg}")


def _ok(msg: str) -> None:
    if not use_json():
        _console.print(f"  {ICON_OK}  {msg}")


def _fail(msg: str) -> None:
    if not use_json():
        _console.print(f"  {ICON_FAIL}  {msg}")


def _validate_inference(inference: str) -> str:
    if inference not in INFERENCE_MODES:
        _die(
            "INVALID_INFERENCE",
            f"Unknown inference mode {inference!r}. Use cloud, local, or hybrid.",
            {"inference": inference},
        )
    if inference not in SUPPORTED_INFERENCE:
        _die(
            "INFERENCE_UNSUPPORTED",
            "Local and hybrid inference are not available yet. Use --inference cloud.",
            {"inference": inference},
        )
    return inference


def _resolve_llm_key(flag: str | None, profile: LocalProfile) -> str:
    for candidate in (
        flag,
        os.environ.get("HONCHO_LLM_API_KEY"),
        os.environ.get("LLM_OPENAI_API_KEY"),
        read_env_value(profile.env_file(), "LLM_OPENAI_API_KEY"),
    ):
        if candidate and not is_placeholder_key(candidate):
            return candidate.strip()

    if use_json():
        _die(
            "MISSING_LLM_KEY",
            "Cloud inference needs an OpenAI-compatible API key. "
            "Pass --llm-api-key or set LLM_OPENAI_API_KEY / HONCHO_LLM_API_KEY.",
        )

    _console.print(
        "  [dim]OpenAI-compatible API key for the local deriver (not a Honcho API key)[/dim]"
    )
    raw = typer.prompt(
        "  LLM API key",
        default="",
        show_default=False,
        hide_input=True,
        prompt_suffix=": ",
    ).strip()
    if not raw or is_placeholder_key(raw):
        _die(
            "MISSING_LLM_KEY",
            "Cloud inference needs an OpenAI-compatible API key. "
            "Pass --llm-api-key or set LLM_OPENAI_API_KEY / HONCHO_LLM_API_KEY.",
        )
    return raw


def _payload(
    profile: LocalProfile, status: str, services: dict[str, str] | None = None
) -> dict:
    endpoints = profile.endpoints()
    inference = endpoints.pop("inference")
    return {
        "profile": profile.name,
        "status": status,
        "inference": inference,
        "endpoints": endpoints,
        "services": services or {},
        "hint": f"HONCHO_BASE_URL={profile.base_url} honcho workspace list",
    }


def _print_stack(payload: dict) -> None:
    if use_json():
        print_json(payload)
        return
    endpoints = payload["endpoints"]
    _console.print()
    table_data = {
        "API": endpoints["api"],
        "Docs": endpoints["docs"],
        "Postgres": endpoints["postgres"],
        "Redis": endpoints["redis"],
        "Inference": payload["inference"],
    }
    print_result(table_data)
    _console.print()
    _console.print(
        "  [dim]CLI still points at your configured server (typically api.honcho.dev).[/dim]"
    )
    _console.print(f"  [dim]To talk to this stack:[/dim] {payload['hint']}")
    _console.print()


def _ensure_docker() -> None:
    _step("Checking Docker")
    try:
        ensure_docker()
    except DockerError as e:
        e.exit()
    _ok("Docker")


def start(
    inference: str = typer.Option(
        "cloud",
        "--inference",
        help="Where the deriver runs LLMs: cloud now; local and hybrid later",
    ),
    profile_name: str = typer.Option(
        DEFAULT_PROFILE,
        "--profile",
        envvar="HONCHO_PROFILE",
        help="Local stack profile name",
    ),
    api_port: int | None = typer.Option(
        None, "--api-port", min=1, max=65535, help="Host port for the API"
    ),
    db_port: int | None = typer.Option(
        None, "--db-port", min=1, max=65535, help="Host port for Postgres"
    ),
    redis_port: int | None = typer.Option(
        None, "--redis-port", min=1, max=65535, help="Host port for Redis"
    ),
    llm_api_key: str | None = typer.Option(
        None,
        "--llm-api-key",
        envvar="HONCHO_LLM_API_KEY",
        help="OpenAI-compatible key for cloud inference",
    ),
    image: str | None = typer.Option(
        None, "--image", help=f"Honcho image (default: {DEFAULT_IMAGE})"
    ),
    timeout: int = typer.Option(
        DEFAULT_HEALTH_TIMEOUT,
        "--timeout",
        min=1,
        help="Seconds to wait for /health after compose up",
    ),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Start a local Honcho stack (API, deriver, Postgres, Redis).

    Requires Docker. Uses cloud LLM inference. Does not change the CLI's
    configured server URL — pass HONCHO_BASE_URL to talk to this stack.
    """
    if json_output:
        set_json_mode(True)

    inference = _validate_inference(inference)
    name = resolve_profile_name(profile_name)
    profile = load_profile(name).overlay(
        inference=inference,
        api_port=api_port,
        db_port=db_port,
        redis_port=redis_port,
        image=image,
    )
    pinned = frozenset(
        name
        for name, value in (
            ("api", api_port),
            ("database", db_port),
            ("redis", redis_port),
        )
        if value is not None
    )

    if not use_json():
        _console.print(f"\n[bold {BRAND}]Honcho Start[/bold {BRAND}]\n")

    _ensure_docker()

    if stack_healthy(profile):
        _ok(f"Already running ({profile.base_url})")
        _print_stack(
            _payload(profile, "running", services_running(compose_ps(profile)))
        )
        return

    try:
        profile, remapped = allocate_host_ports(profile, pinned=pinned)
    except DockerError as e:
        e.exit()
    for service, (old, new) in remapped.items():
        _step(f"Port {old} in use; {service} on {new}")

    key = _resolve_llm_key(llm_api_key, profile)
    _step(f"Writing stack config to {profile.dir()}")
    save_profile(profile)
    render_stack(profile, key)
    _ok(f"Profile '{profile.name}'")

    _step("Starting containers (first run pulls images)")
    try:
        compose_up(profile)
    except DockerError as e:
        e.exit()

    _step(f"Waiting for API at {profile.base_url}/health")
    if not wait_for_health(profile, timeout=float(timeout)):
        _fail("Timed out waiting for /health")
        _die(
            "HEALTH_TIMEOUT",
            f"Stack started but {profile.base_url}/health did not become ready within {timeout}s. "
            f"Check `docker compose -p {profile.project_name} logs`.",
            {
                "base_url": profile.base_url,
                "timeout": timeout,
                "project": profile.project_name,
            },
        )

    _ok("Honcho is running")
    _print_stack(_payload(profile, "running", services_running(compose_ps(profile))))


def stop(
    profile_name: str = typer.Option(
        DEFAULT_PROFILE,
        "--profile",
        envvar="HONCHO_PROFILE",
        help="Local stack profile name",
    ),
    wipe: bool = typer.Option(
        False, "--wipe", help="Also delete volumes (Postgres data)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Stop the local stack started by `honcho start`. Keeps data unless --wipe."""
    if json_output:
        set_json_mode(True)

    name = resolve_profile_name(profile_name)
    profile = load_profile(name)

    if not profile.compose_file().exists():
        payload = _payload(profile, "stopped")
        if use_json():
            print_json(payload)
        else:
            _console.print(f"  [dim]No local stack for profile '{profile.name}'.[/dim]")
        return

    _ensure_docker()
    try:
        compose_down(profile, wipe=wipe)
    except DockerError as e:
        e.exit()

    state = "wiped" if wipe else "stopped"
    _ok(f"Stopped profile '{profile.name}'" + (" (volumes removed)" if wipe else ""))
    _print_stack(_payload(profile, state))


def status(
    profile_name: str = typer.Option(
        DEFAULT_PROFILE,
        "--profile",
        envvar="HONCHO_PROFILE",
        help="Local stack profile name",
    ),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Show local stack endpoints and container health."""
    if json_output:
        set_json_mode(True)

    name = resolve_profile_name(profile_name)
    profile = load_profile(name)

    if not profile.compose_file().exists():
        _die(
            "STACK_NOT_FOUND",
            f"No local stack for profile '{profile.name}'. Run `honcho start` first.",
            {"profile": profile.name},
        )

    _ensure_docker()
    services = services_running(compose_ps(profile))
    running = stack_healthy(profile)
    state = "running" if running else "stopped"
    if not use_json():
        icon = ICON_OK if running else ICON_FAIL
        _console.print(f"\n  {icon}  profile '{profile.name}' is {state}\n")
        if services:
            for svc in STACK_SERVICES:
                detail = services.get(svc, "missing")
                _console.print(f"  {svc:<10}  [dim]{detail}[/dim]")
    _print_stack(_payload(profile, state, services))
    if not running:
        raise typer.Exit(1)
