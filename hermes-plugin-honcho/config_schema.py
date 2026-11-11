"""Honcho's declared config surface — rendered by the generic desktop panel."""

import plugins.memory.config_schema as _core_schema
from plugins.memory.config_schema import (
    KIND_BOOL, KIND_JSON, KIND_SECRET, KIND_TEXT, ProviderConfigSchema, ProviderField,
)

# Older Hermes has no STORAGE_PROVIDER; newer Hermes has no STORAGE_HONCHO_HOST_BLOCK.
_STORAGE = getattr(_core_schema, "STORAGE_PROVIDER", None) or _core_schema.STORAGE_HONCHO_HOST_BLOCK


def _field(key, label, kind, description, *, group, **kw) -> ProviderField:
    return ProviderField(key=key, label=label, kind=kind, description=description, group=group, **kw)


# Inline fields form the curated compact panel; the rest surface only in the full-config modal.
CONFIG_SCHEMA = ProviderConfigSchema(
    name="honcho",
    label="Honcho",
    storage=_STORAGE,
    # Literal, not attribution.attribution_url(): Hermes loads this file on its own (no package, so no
    # relative imports). tests/test_attribution.py keeps it equal to the helper's output.
    docs_url="https://docs.honcho.dev/v3/guides/integrations/hermes?utm_source=hermes&utm_medium=plugin&utm_content=config-panel",
    fields=(
        # — Connection (inline) —
        _field("apiKey", "API key", KIND_SECRET, "Authenticate with Honcho Cloud. Not needed for a self-hosted base URL.",
               env_key="HONCHO_API_KEY", placeholder="Enter Honcho API key", inline=True, group="Connection"),
        _field("baseUrl", "Base URL", KIND_TEXT, "Self-hosted Honcho URL. Blank uses Honcho Cloud.",
               env_fallbacks=("HONCHO_BASE_URL",), placeholder="https://… (self-hosted)",
               inline=True, group="Connection"),
        _field("workspace", "Workspace", KIND_TEXT, "Honcho workspace ID. Defaults to the profile host.",
               inline=True, group="Connection"),
        # — Identity (inline) —
        _field("peerName", "Peer name", KIND_TEXT, "Your stable user peer. Unifies memory across platforms for single-user setups.",
               placeholder="e.g. eri", inline=True, group="Identity"),
        _field("aiPeer", "AI peer", KIND_TEXT, "The AI-side peer name. Defaults to the profile host.",
               inline=True, group="Identity"),
        # — Identity —
        _field("pinUserPeer", "Pin user peer", KIND_BOOL, "Pin the user peer to the peer name, ignoring gateway runtime identity. Unifies memory for single-user setups.",
               default="false", aliases=("pinPeerName",), group="Identity"),
        _field("runtimePeerPrefix", "Runtime peer prefix", KIND_TEXT, "Prefix applied to unknown gateway runtime user IDs.",
               placeholder="e.g. telegram_", group="Identity"),
        _field("userPeerAliases", "User peer aliases", KIND_JSON, "Map gateway runtime user IDs to stable Honcho peers.",
               placeholder='{"telegram_123": "eri"}', group="Identity"),
        _field("a2aSessions", "Bot DM sessions", KIND_BOOL,
               "Write DMs from other bots into their own Honcho session per sender. Off skips bot-authored turns.",
               default="true", group="Identity"),
        # — Recall —
        _field("logging", "Injection audit log", KIND_BOOL,
               "Append what each turn injected, and why, to ~/.honcho/injection.log. The record holds what Honcho "
               "said about the user verbatim.", default="false", group="Recall"),
    ),
)
