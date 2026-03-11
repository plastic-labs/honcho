"""Schema introspection: describe resource schemas from live OpenAPI spec."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer

from honcho_cli.output import print_error, print_result, set_json_mode, set_quiet_mode, status

app = typer.Typer(help="Schema introspection from live server.")

# Local cache for OpenAPI spec
_CACHE_DIR = Path.home() / ".honcho" / "cache"
_CACHE_FILE = _CACHE_DIR / "openapi.json"
_CACHE_TTL = 3600  # 1 hour

# Map CLI resource names to OpenAPI schema names
RESOURCE_SCHEMA_MAP: dict[str, list[str]] = {
    "workspace": ["WorkspaceCreate", "WorkspaceResponse", "WorkspaceConfiguration"],
    "peer": ["PeerCreate", "PeerResponse", "PeerConfig"],
    "session": ["SessionCreate", "SessionResponse", "SessionConfiguration", "SessionPeerConfig"],
    "message": ["MessageCreate", "MessageResponse", "MessageConfiguration"],
    "conclusion": ["ConclusionCreate", "ConclusionResponse"],
    "key": ["JWTParams"],
}


def _fetch_openapi(base_url: str, api_key: str | None = None) -> dict:
    """Fetch OpenAPI spec from server, with caching."""
    # Check cache
    if _CACHE_FILE.exists():
        mtime = _CACHE_FILE.stat().st_mtime
        if time.time() - mtime < _CACHE_TTL:
            return json.loads(_CACHE_FILE.read_text())

    import httpx

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/openapi.json", headers=headers, timeout=15)
        resp.raise_for_status()
        spec = resp.json()
    except Exception as e:
        # Fall back to cache if available
        if _CACHE_FILE.exists():
            status("Using cached OpenAPI spec (server unreachable)")
            return json.loads(_CACHE_FILE.read_text())
        raise

    # Cache it
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(spec))

    return spec


def _resolve_ref(spec: dict, ref: str) -> dict:
    """Resolve a $ref in an OpenAPI spec."""
    parts = ref.lstrip("#/").split("/")
    node = spec
    for part in parts:
        node = node[part]
    return node


def _extract_schema(spec: dict, schema_name: str) -> dict | None:
    """Extract a schema from the OpenAPI spec components."""
    schemas = spec.get("components", {}).get("schemas", {})

    # Try exact match first
    if schema_name in schemas:
        return _flatten_schema(spec, schemas[schema_name])

    # Try case-insensitive match
    for name, schema in schemas.items():
        if name.lower() == schema_name.lower():
            return _flatten_schema(spec, schema)

    return None


def _flatten_schema(spec: dict, schema: dict) -> dict:
    """Flatten a schema, resolving $ref and allOf."""
    if "$ref" in schema:
        return _flatten_schema(spec, _resolve_ref(spec, schema["$ref"]))

    if "allOf" in schema:
        merged: dict = {"type": "object", "properties": {}}
        for sub in schema["allOf"]:
            resolved = _flatten_schema(spec, sub)
            if "properties" in resolved:
                merged["properties"].update(resolved["properties"])
            if "required" in resolved:
                merged.setdefault("required", []).extend(resolved["required"])
        return merged

    return schema


def _format_schema(schema: dict) -> dict:
    """Format schema for display."""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields = {}
    for name, prop in props.items():
        field_type = prop.get("type", prop.get("$ref", "unknown"))
        if "anyOf" in prop:
            types = [t.get("type", "?") for t in prop["anyOf"] if t.get("type") != "null"]
            field_type = " | ".join(types) if types else "unknown"
            if any(t.get("type") == "null" for t in prop["anyOf"]):
                field_type += " (optional)"

        info: dict = {"type": field_type}
        if name in required:
            info["required"] = True
        if "default" in prop:
            info["default"] = prop["default"]
        if "description" in prop:
            info["description"] = prop["description"]

        fields[name] = info

    return fields


@app.command("resource")
def describe_resource(
    resource: str = typer.Argument(help="Resource type: workspace, peer, session, message, conclusion, key"),
) -> None:
    """Describe a resource schema from the live server."""
    from honcho_cli.main import get_resolved_config

    resource = resource.lower()
    if resource not in RESOURCE_SCHEMA_MAP:
        print_error(
            "UNKNOWN_RESOURCE",
            f"Unknown resource: '{resource}'",
            {"valid_resources": list(RESOURCE_SCHEMA_MAP.keys())},
        )
        raise typer.Exit(1)

    config = get_resolved_config()

    try:
        spec = _fetch_openapi(config.base_url, config.api_key)
    except Exception as e:
        print_error("OPENAPI_ERROR", f"Failed to fetch OpenAPI spec: {e}", {"base_url": config.base_url})
        raise typer.Exit(1)

    schema_names = RESOURCE_SCHEMA_MAP[resource]
    result: dict = {}

    for schema_name in schema_names:
        schema = _extract_schema(spec, schema_name)
        if schema:
            result[schema_name] = _format_schema(schema)

    if not result:
        print_error("SCHEMA_NOT_FOUND", f"No schemas found for '{resource}'", {"resource": resource})
        raise typer.Exit(1)

    print_result(result)


# Make `honcho describe <resource>` work as the default command
# by aliasing the resource command as the callback
@app.callback(invoke_without_command=True)
def describe_callback(
    ctx: typer.Context,
    resource: Optional[str] = typer.Argument(None, help="Resource type to describe"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress status messages"),
) -> None:
    """Describe resource schemas from the live server's OpenAPI spec."""
    if json_output:
        set_json_mode(True)
    if quiet:
        set_quiet_mode(True)
    if resource and not ctx.invoked_subcommand:
        ctx.invoke(describe_resource, resource=resource)
