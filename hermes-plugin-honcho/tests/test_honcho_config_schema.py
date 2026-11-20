"""Tests for Honcho's declared config surface.

Asserts on this plugin's CONFIG_SCHEMA directly: Hermes' `get_provider_config_schema("honcho")`
registry lookup resolves its bundled copy, not this one.
"""

from plugins.memory.config_schema import (
    KIND_BOOL,
    KIND_JSON,
    KIND_SECRET,
    STORAGE_PROVIDER,
)

from hermes_plugin_honcho.config_schema import CONFIG_SCHEMA as provider

# The curated set shown in the compact panel; everything else lives in the modal.
INLINE_KEYS = {
    "apiKey",
    "baseUrl",
    "workspace",
    "peerName",
    "aiPeer",
}


def test_honcho_is_declared():
    assert provider.name == "honcho"
    assert provider.label == "Honcho"
    assert provider.storage == STORAGE_PROVIDER
    # Field keys are unique, and the curated inline set is present.
    keys = [field.key for field in provider.fields]
    assert len(keys) == len(set(keys))
    assert INLINE_KEYS <= set(keys)


def test_inline_fields_are_the_curated_subset():
    assert {field.key for field in provider.inline_fields()} == INLINE_KEYS
    # The modal-only fields are a non-empty remainder.
    non_inline = {f.key for f in provider.fields} - INLINE_KEYS
    assert {"pinUserPeer", "userPeerAliases", "a2aSessions", "logging"} <= non_inline


def test_declares_the_new_field_kinds():
    by_key = {f.key: f for f in provider.fields}
    assert by_key["a2aSessions"].kind == KIND_BOOL
    assert by_key["a2aSessions"].default == "true"
    assert by_key["userPeerAliases"].kind == KIND_JSON


def test_api_key_is_a_secret_bound_to_env():
    api_key = next(f for f in provider.fields if f.key == "apiKey")
    assert api_key.kind == KIND_SECRET
    assert api_key.is_secret is True
    assert api_key.env_key == "HONCHO_API_KEY"


def test_every_field_lives_in_the_host_block():
    # Schema 2: the plugin reads only hosts.<host>, so older Hermes' host-block storage must write there too.
    assert {f.key for f in provider.fields if f.scope != "host"} == set()
