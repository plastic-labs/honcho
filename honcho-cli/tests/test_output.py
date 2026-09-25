"""Transcript rendering: timestamp normalization and content fidelity.

`session view` is a debugging surface, so the human-mode table must show what
was actually stored — no Markdown reflow, no truncated identifiers.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from honcho_cli import output
from honcho_cli.output import _format_timestamp, print_transcript


@pytest.fixture
def render(monkeypatch, capsys):
    """Render a transcript in human mode at a fixed width and return stdout."""
    monkeypatch.setattr(output, "is_tty", lambda: True)
    monkeypatch.setattr(output, "_force_json", False)

    def _render(messages, width: int = 120, **kwargs):
        monkeypatch.setattr(output, "stdout_console", output.Console(width=width, no_color=True))
        print_transcript(messages, **kwargs)
        return capsys.readouterr().out

    return _render


def _msg(content: str = "hi", **overrides) -> dict:
    return {
        "id": "V1StGXR8_Z5jdHi6B-myT",
        "peer_id": "alice",
        "content": content,
        "created_at": "2026-01-01T00:00:00Z",
        **overrides,
    }


class TestFormatTimestamp:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("2026-01-01T00:00:00Z", "2026-01-01T00:00:00.000Z"),
            ("2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00.000Z"),
            ("2026-01-01 00:00:00+00:00", "2026-01-01T00:00:00.000Z"),
            ("2026-01-01 00:00:00", "2026-01-01T00:00:00.000Z"),
            ("2026-01-01T00:00:00.080000Z", "2026-01-01T00:00:00.080Z"),
        ],
    )
    def test_normalizes_utc_shapes(self, value, expected):
        assert _format_timestamp(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("2026-01-01T14:30:00-05:00", "2026-01-01T19:30:00.000Z"),
            ("2026-01-01T14:30:00+02:00", "2026-01-01T12:30:00.000Z"),
        ],
    )
    def test_converts_offsets_instead_of_relabelling_them(self, value, expected):
        """An offset must be converted to UTC, not dropped and stamped `Z`."""
        assert _format_timestamp(value) == expected

    def test_keeps_sub_second_precision(self):
        """Messages inside the same second must stay distinguishable."""
        a = _format_timestamp("2026-01-01T00:00:03.000Z")
        b = _format_timestamp("2026-01-01T00:00:03.080Z")
        assert a != b
        assert (a, b) == ("2026-01-01T00:00:03.000Z", "2026-01-01T00:00:03.080Z")

    def test_accepts_datetime_objects(self):
        value = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert _format_timestamp(value) == "2026-01-01T12:00:00.000Z"

    @pytest.mark.parametrize("value", [None, ""])
    def test_empty_values_render_blank(self, value):
        assert _format_timestamp(value) == ""

    def test_unparseable_values_pass_through(self):
        assert _format_timestamp("not-a-date") == "not-a-date"

    def test_output_matches_the_declared_column_width(self):
        assert len(_format_timestamp("2026-01-01T00:00:00Z")) == output.TIMESTAMP_WIDTH


class TestTranscriptFidelity:
    def test_newlines_are_not_reflowed_into_a_paragraph(self, render):
        out = render([_msg("line one\nline two\nline three")], session_id="s1")
        assert "line one line two line three" not in out
        for line in ("line one", "line two", "line three"):
            assert line in out

    def test_tagged_content_is_not_stripped(self, render):
        """Agent transcripts are full of `<thinking>`-style tags; they must survive."""
        out = render([_msg("<thinking>reasoning</thinking> answer")], session_id="s1")
        assert "<thinking>" in out
        assert "</thinking>" in out

    def test_console_markup_is_not_interpreted(self, render):
        out = render([_msg("literal [bold]not markup[/bold] text")], session_id="s1")
        assert "[bold]" in out

    def test_ids_are_shown_in_full(self, render):
        """A displayed ID must be usable with `honcho message get`."""
        out = render([_msg()], session_id="s1", show_ids=True)
        assert "V1StGXR8_Z5jdHi6B-myT" in out
        assert "…" not in out

    def test_long_peer_ids_stay_distinguishable(self, render):
        out = render(
            [
                _msg(peer_id="user_1234567890abcdef"),
                _msg(peer_id="user_1234567890abcXYZ"),
            ],
            session_id="s1",
        )
        assert "abcdef" in out
        assert "abcXYZ" in out

    def test_empty_transcript_reports_the_session(self, render):
        out = render([], session_id="s1")
        assert "s1" in out
        assert "(empty)" in out


class TestNextPageHint:
    def test_given_hint_is_printed(self, render, monkeypatch):
        printed: list[str] = []
        monkeypatch.setattr(output, "status", printed.append)
        render([_msg()], session_id="s1", page=1, pages=3, next_page_hint="honcho ... --page 2")
        assert printed == ["more: honcho ... --page 2"]

    def test_no_hint_when_none_given(self, render, monkeypatch):
        printed: list[str] = []
        monkeypatch.setattr(output, "status", printed.append)
        render([_msg()], session_id="s1", page=3, pages=3)
        assert printed == []
