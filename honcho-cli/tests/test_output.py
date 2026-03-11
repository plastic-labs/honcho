"""Tests for output formatting."""

import json

from honcho_cli.output import is_tty, set_json_mode, set_quiet_mode, use_json


class TestOutputModes:
    def test_force_json(self):
        set_json_mode(True)
        assert use_json() is True
        set_json_mode(False)

    def test_non_tty_defaults_to_json(self):
        # In test context, stdout is not a TTY
        set_json_mode(False)
        assert use_json() is True  # pytest redirects stdout


class TestPrintNdjson:
    def test_ndjson_format(self, capsys):
        from honcho_cli.output import print_ndjson

        items = [{"id": "1", "name": "a"}, {"id": "2", "name": "b"}]
        print_ndjson(items)
        output = capsys.readouterr().out
        lines = output.strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"id": "1", "name": "a"}
        assert json.loads(lines[1]) == {"id": "2", "name": "b"}


class TestPrintJson:
    def test_json_format(self, capsys):
        from honcho_cli.output import print_json

        data = {"workspace_id": "test", "peer_count": 5}
        print_json(data)
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert parsed["workspace_id"] == "test"
        assert parsed["peer_count"] == 5


class TestPrintError:
    def test_error_json_format(self, capsys):
        set_json_mode(True)
        from honcho_cli.output import print_error

        print_error("PEER_NOT_FOUND", "Peer 'abc' not found", {"peer_id": "abc"})
        output = capsys.readouterr().err
        parsed = json.loads(output)
        assert parsed["error"]["code"] == "PEER_NOT_FOUND"
        assert parsed["error"]["details"]["peer_id"] == "abc"
        set_json_mode(False)
