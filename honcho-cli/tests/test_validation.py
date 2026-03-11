"""Tests for input hardening / validation."""

import pytest

from honcho_cli.validation import validate_resource_id, validate_workspace_name


class TestValidateResourceId:
    def test_valid_id(self):
        assert validate_resource_id("abc123") == "abc123"

    def test_valid_nanoid(self):
        assert validate_resource_id("V1StGXR8_Z5jdHi6B-myT") == "V1StGXR8_Z5jdHi6B-myT"

    def test_empty_id(self):
        with pytest.raises(SystemExit):
            validate_resource_id("")

    def test_question_mark(self):
        with pytest.raises(SystemExit):
            validate_resource_id("abc?def")

    def test_hash(self):
        with pytest.raises(SystemExit):
            validate_resource_id("abc#def")

    def test_percent(self):
        with pytest.raises(SystemExit):
            validate_resource_id("abc%def")

    def test_control_chars(self):
        with pytest.raises(SystemExit):
            validate_resource_id("abc\x00def")

    def test_null_byte(self):
        with pytest.raises(SystemExit):
            validate_resource_id("abc\x01def")

    def test_path_traversal(self):
        with pytest.raises(SystemExit):
            validate_resource_id("../etc/passwd")

    def test_forward_slash(self):
        with pytest.raises(SystemExit):
            validate_resource_id("abc/def")

    def test_backslash(self):
        with pytest.raises(SystemExit):
            validate_resource_id("abc\\def")

    def test_tab(self):
        with pytest.raises(SystemExit):
            validate_resource_id("abc\tdef")


class TestValidateWorkspaceName:
    def test_valid_name(self):
        assert validate_workspace_name("my-workspace") == "my-workspace"

    def test_valid_underscore(self):
        assert validate_workspace_name("my_workspace_123") == "my_workspace_123"

    def test_empty_name(self):
        with pytest.raises(SystemExit):
            validate_workspace_name("")

    def test_spaces(self):
        with pytest.raises(SystemExit):
            validate_workspace_name("my workspace")

    def test_special_chars(self):
        with pytest.raises(SystemExit):
            validate_workspace_name("my@workspace")

    def test_dots(self):
        with pytest.raises(SystemExit):
            validate_workspace_name("my.workspace")
