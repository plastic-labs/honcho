"""The panel must show the same opt-out phrases as runtime resolves."""

import pytest

from hermes_plugin_honcho import config_panel
from hermes_plugin_honcho.client import _parse_string_list


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ({"observationOptOutPhrases": None}, []),
        ({"observationOptOutPhrases": []}, []),
        ({"observationOptOutPhrases": "invalid"}, []),
        ({}, ["root phrase"]),
        ({"observationOptOutPhrases": [" host phrase ", "", 7]}, ["host phrase"]),
    ],
)
def test_panel_shows_effective_opt_out_phrases(monkeypatch, host, expected):
    raw = {
        "observationOptOutPhrases": ["root phrase"],
        "peerName": "root peer",
        "hosts": {"hermes": {**host, "peerName": None}},
    }
    monkeypatch.setattr(config_panel, "resolve_active_host", lambda: "hermes")
    monkeypatch.setattr(config_panel, "resolve_config_path", lambda: None)
    monkeypatch.setattr(config_panel, "_read_config_strict", lambda path: raw)

    values = config_panel.read_fields()["values"]
    assert values["observationOptOutPhrases"] == _parse_string_list(host, raw, "observationOptOutPhrases") == expected
    assert values["peerName"] == "root peer"
