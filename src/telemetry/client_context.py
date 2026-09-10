"""Per-request client identity for telemetry enrichment.

Clients may send ``X-Honcho-Host``, ``X-Honcho-Plugin`` and
``X-Honcho-Agent-Model`` to identify themselves. The API middleware parks the
values in ContextVars for the duration of the request; the CloudEvents emitter
reads them when it serializes an event body, as a nested ``client`` object::

    "client": {
        "host": "claude-code/2.1.3 (darwin)",
        "plugin": "claude-honcho/0.2.11",
        "agent_model": "claude-sonnet-4-5"
    }

Outside a request (deriver worker, tests, startup) the vars are unset and
every member is ``null``; the ``client`` object itself is always present so
``data.client.host`` is a safe path for consumers.

These are emitter-injected body fields, like ``honcho_version``, and are
exempt from per-event schema versioning.
"""

from contextvars import ContextVar, Token

HEADER_HOST = "X-Honcho-Host"
HEADER_PLUGIN = "X-Honcho-Plugin"
HEADER_AGENT_MODEL = "X-Honcho-Agent-Model"

# Header values are client-controlled; cap them so a misbehaving client can't
# bloat every event body.
_MAX_VALUE_LEN = 256

client_host: ContextVar[str | None] = ContextVar("client_host", default=None)
client_plugin: ContextVar[str | None] = ContextVar("client_plugin", default=None)
client_agent_model: ContextVar[str | None] = ContextVar(
    "client_agent_model", default=None
)

ClientContextTokens = tuple[Token[str | None], Token[str | None], Token[str | None]]


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return value[:_MAX_VALUE_LEN]


def set_client_context(
    *, host: str | None, plugin: str | None, agent_model: str | None
) -> ClientContextTokens:
    """Set the client ContextVars; returns tokens for ``reset_client_context``."""
    return (
        client_host.set(_clean(host)),
        client_plugin.set(_clean(plugin)),
        client_agent_model.set(_clean(agent_model)),
    )


def reset_client_context(tokens: ClientContextTokens) -> None:
    """Restore the client ContextVars to their pre-request values."""
    host_token, plugin_token, model_token = tokens
    client_host.reset(host_token)
    client_plugin.reset(plugin_token)
    client_agent_model.reset(model_token)


def client_context_body() -> dict[str, str | None]:
    """The ``client`` object the emitter injects; members are ``None`` when unset."""
    return {
        "host": client_host.get(),
        "plugin": client_plugin.get(),
        "agent_model": client_agent_model.get(),
    }
