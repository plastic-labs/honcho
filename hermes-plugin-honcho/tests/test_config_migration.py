"""Schema 2: a profile's settings live in ``hosts.<host>`` of its own honcho.json, moved there once from
what 1.1.0 read. The table mirrors @honcho-ai/harness-plugin-core's ``migration`` test."""

import copy
import json
from pathlib import Path

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

from hermes_plugin_honcho.client import HonchoClientConfig
from hermes_plugin_honcho.host_config import CLOUD, DEAD_KEYS, legacy_settings, migrate_host, normalize_url, save_host
from hermes_plugin_honcho.oauth import install_grant

LOCAL, SELF = "http://localhost:8000", "https://self.example"
HOST = "hermes_work"


def test_reader_matches_what_1_1_0_resolved():
    """Fixtures recorded by running 1.1.0's own client.py functions over each file, with no env."""
    failures = []
    for case in json.loads((Path(__file__).parent / "fixtures" / "legacy-hermes-honcho.json").read_text()):
        got, host = legacy_settings(case["file"], case["host"]), case["host"]
        resolved = {"apiKey": got["apiKey"], "baseUrl": got["baseUrl"] or CLOUD, "workspace": got["workspace"] or host,
                    "peerName": got["peerName"], "aiPeer": got["aiPeer"] or host}
        if resolved != case["want"]:
            failures.append(f"{case['name']}: got {resolved}, want {case['want']}")
    assert failures == []


def _grant(token):
    return {"apiKey": token, "oauth": {"refreshToken": "hch-rt-r", "expiresAt": 9999999999, "clientId": "c",
                                       "tokenEndpoint": f"{CLOUD}/oauth/token"}}


TOPS = {
    "none": {},
    "cliLocal": {"apiKey": "cli", "environmentUrl": LOCAL},
    "self": {"apiKey": "top", "baseUrl": SELF},
    "endpointSelf": {"apiKey": "top", "endpoint": {"baseUrl": SELF + "/v3"}},
    "localEnv": {"environment": "local", "pinUserPeer": True, "recallMode": "tools"},
    "names": {"peerName": "top-peer", "workspace": "top-ws"},
}
BLOCKS = {
    "absent": None,
    "empty": {},
    "key": {"apiKey": "own"},
    "url": {"baseUrl": SELF + "/"},
    "grant": _grant("hch-at-grant"),
    "names": {"workspace": "own-ws", "peerName": "own-peer"},
    "marked": {"apiKey": "own", "baseUrl": LOCAL, "schemaVersion": 2},
}


def _resolve(own):
    cfg = HonchoClientConfig.from_global_config(host=HOST, config_path=own)
    return {"server": normalize_url(cfg.base_url), "token": cfg.api_key, "workspace": cfg.workspace_id,
            "peerName": cfg.peer_name, "pin": cfg.pin_peer_name}


def _sign_in(own):
    install_grant(own, HOST, {"access_token": "hch-at-signin", "refresh_token": "hch-rt-s", "expires_in": 3600},
                  client_id="c", token_endpoint=f"{CLOUD}/oauth/token")


def _read_then_save(own):
    """A setup screen that saves what it read, before anything migrated."""
    cfg = HonchoClientConfig.from_global_config(host=HOST, config_path=own)
    save_host(own, HOST, {"peerName": cfg.peer_name, "workspace": cfg.workspace_id, "baseUrl": cfg.base_url})
    migrate_host(own, HOST)


ORDERS = {
    "start": lambda own: migrate_host(own, HOST),
    "setup: name": lambda own: save_host(own, HOST, {"peerName": "me"}),
    "setup: cloud": lambda own: save_host(own, HOST, {"baseUrl": None}),
    "setup: local": lambda own: save_host(own, HOST, {"baseUrl": LOCAL}),
    "setup: key": lambda own: save_host(own, HOST, {"apiKey": "new"}),
    "setup screen saves what it read, before migrating": _read_then_save,
    "sign in": _sign_in,
}


def _expected(file, block, place, order):
    """What the profile should use afterwards: what 1.1.0 read, then the rules."""
    marked = place == "own" and (block or {}).get("schemaVersion") == 2
    grant = (block or {}).get("apiKey") if block and "oauth" in block else None
    old = block if marked else legacy_settings(file, HOST)
    server = CLOUD if grant and not marked else normalize_url(old.get("baseUrl"))
    key = grant or old.get("apiKey")
    pin = bool(not marked and file.get("pinUserPeer"))  # a kept setting 1.1.0 read from the top level
    names = {"workspace": old.get("workspace") or HOST, "peerName": old.get("peerName"), "pin": pin}

    def at(s, k):
        return {"server": s, "token": k, **names}
    if order == "setup: name":
        return {**at(server, key), "peerName": "me"}
    if order == "setup: cloud":
        return at(CLOUD, key if server == CLOUD else None)
    if order == "setup: local":
        return at(LOCAL, key if server == LOCAL else None)
    if order == "setup: key":
        return at(server, "new")
    if order == "sign in":
        return at(CLOUD, "hch-at-signin")
    return at(server, key)


def test_each_profile_keeps_what_1_1_0_used_once_and_only_its_block_counts_after(tmp_path):
    """For every file shape, wherever 1.1.0 found it (the profile's own file, the default profile's, or
    ~/.honcho's) and every order a profile might run things in: reading before the migration, a second
    start, and later writes to the top level or to the other files change nothing."""
    failures = []
    default_file, global_file = get_hermes_home() / "honcho.json", Path.home() / ".honcho" / "config.json"
    global_file.parent.mkdir(parents=True, exist_ok=True)
    for n, ((tn, top), (bn, block), place, (on, run)) in enumerate(
        (t, b, p, o) for t in TOPS.items() for b in BLOCKS.items() for p in ("own", "default", "global")
        for o in ORDERS.items()
    ):
        home = tmp_path / f"case{n}" / "profiles" / "work"
        home.mkdir(parents=True)
        own = home / "honcho.json"
        token = set_hermes_home_override(str(home))
        try:
            for f in (default_file, global_file):
                f.unlink(missing_ok=True)
            file = {**copy.deepcopy(top), **({"hosts": {HOST: copy.deepcopy(block)}} if block is not None else {})}
            {"own": own, "default": default_file, "global": global_file}[place].write_text(json.dumps(file))
            case = f"{tn} · {bn} · {place} · {on}"

            before = _resolve(own)
            run(own)
            got = _resolve(own)
            if got != (want := _expected(file, block, place, on)):
                failures.append(f"{case}: got {got}, want {want}")
            if on == "start" and before != got:
                failures.append(f"{case}: read before migrating gave {before}, after {got}")
            if migrate_host(own, HOST):
                failures.append(f"{case}: a second start changed the block")
            if dead := set(DEAD_KEYS) & set(json.loads(own.read_text())["hosts"][HOST]):
                failures.append(f"{case}: settings this release fixes stayed in the block: {sorted(dead)}")
            later = {"apiKey": "later", "baseUrl": "http://127.0.0.1:9999", "peerName": "later", "workspace": "later",
                     "pinUserPeer": True}
            for f in (own, default_file, global_file):
                f.write_text(json.dumps({**(json.loads(f.read_text()) if f.exists() else {}), **later}))
            migrate_host(own, HOST)
            if _resolve(own) != got:
                failures.append(f"{case}: a later top-level write reached the block: {_resolve(own)}")
        finally:
            reset_hermes_home_override(token)
    assert failures == []
