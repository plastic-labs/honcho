"""Recall-boundary flag parsing (DEV-2690)."""

from __future__ import annotations

import pytest
import typer

from honcho_cli.recall import parse_csv_repeatable, reject_incompatible_recall, scope_for_sdk


class TestParseCsvRepeatable:
    def test_none_and_empty(self):
        assert parse_csv_repeatable(None, kind="scope", flag="--scope") is None
        assert parse_csv_repeatable([], kind="scope", flag="--scope") is None

    def test_repeatable_and_csv(self):
        assert parse_csv_repeatable(["therapy", "work,home"], kind="scope", flag="--scope") == [
            "therapy",
            "work",
            "home",
        ]

    def test_dedupes_and_strips(self):
        assert parse_csv_repeatable([" therapy ", "therapy,work"], kind="scope", flag="--scope") == [
            "therapy",
            "work",
        ]

    def test_given_but_empty_is_an_error(self):
        for values in ([""], [","], [" , "]):
            with pytest.raises(typer.Exit):
                parse_csv_repeatable(values, kind="session", flag="--sessions")

    def test_rejects_unsafe_id(self):
        with pytest.raises(SystemExit):
            parse_csv_repeatable(["bad/slash"], kind="scope", flag="--scope")


class TestScopeForSdk:
    def test_single_stays_string(self):
        assert scope_for_sdk(["therapy"]) == "therapy"

    def test_several_become_list(self):
        assert scope_for_sdk(["therapy", "work"]) == ["therapy", "work"]

    def test_none(self):
        assert scope_for_sdk(None) is None


class TestRejectIncompatible:
    def test_one_bound_ok(self):
        reject_incompatible_recall(session_id="s1", scope=None, sessions=None)
        reject_incompatible_recall(session_id=None, scope=["a"], sessions=None)
        reject_incompatible_recall(session_id=None, scope=None, sessions=["s1"])

    def test_scope_and_session(self):
        with pytest.raises(typer.Exit):
            reject_incompatible_recall(session_id="s1", scope=["a"], sessions=None)

    def test_scope_and_sessions(self):
        with pytest.raises(typer.Exit):
            reject_incompatible_recall(session_id=None, scope=["a"], sessions=["s1"])
