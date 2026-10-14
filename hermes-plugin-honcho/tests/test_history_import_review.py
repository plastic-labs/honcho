"""Review regressions exercise the real source database, CLI, and SDK boundary."""

import builtins
from datetime import UTC, datetime

import pytest
from agent.redact import redact_sensitive_text
from hermes_plugin_honcho import HonchoMemoryProvider
from hermes_plugin_honcho import history_import as importer
from hermes_plugin_honcho.import_state import ImportStateError
from test_history_import import history_source as history_source
from test_history_import import run_cli


def uploaded(source):
    return [message for batch in source.api.writes for message in batch]


def test_redaction_precedes_chunks_and_hashes(history_source, monkeypatch, capsys):
    source = history_source
    source.session()
    token = "ghp_" + "Q" * 36
    text = "An old credential: " + token + "; DATABASE_PASSWORD=opaque-password-value"
    source.message(content=text)
    source.configure(messageMaxChars=24)
    monkeypatch.setattr("agent.redact._redact_enabled", lambda: False)
    expected = redact_sensitive_text(text, force=True, redact_url_credentials=True)
    assert run_cli(["import", "--yes"]) == 0
    messages = uploaded(source)
    assert [
        message["content"] for message in messages
    ] == HonchoMemoryProvider._chunk_message(expected, 24)
    assert token not in str(source.api.requests)
    assert "opaque-password-value" not in str(source.api.requests)
    for message in messages:
        timestamp = datetime.fromisoformat(message["created_at"])
        assert message["metadata"]["payload_sha256"] == importer.payload_hash(
            message["content"], message["peer_id"], timestamp, "history-test"
        )
    assert token not in capsys.readouterr().out


def test_redaction_failure_stops_before_client_or_ledger(
    history_source, monkeypatch, capsys
):
    source = history_source
    source.session()
    source.message(content="Private source text")

    def fail(*args, **kwargs):
        raise RuntimeError("secret failure details")

    monkeypatch.setattr("agent.redact.redact_sensitive_text", fail)
    assert run_cli(["import", "--yes"]) == 2
    source.factory.assert_not_called()
    assert not source.ledger.exists()
    assert "secret failure details" not in capsys.readouterr().out


def test_in_place_compaction_preserves_originals_and_excludes_rewind(history_source):
    source = history_source
    source.session()
    source.message(content="Original user", timestamp=source.now - 100)
    source.message(
        role="assistant", content="Original assistant", timestamp=source.now - 99
    )
    rewind = source.message(content="Undone text", timestamp=source.now - 98)
    source.db._write_sql(
        "UPDATE messages SET active = 0, compacted = 0 WHERE id = ?", (rewind,)
    )
    source.db.archive_and_compact(
        "source-one", [{"role": "system", "content": "Compression summary"}]
    )
    source.message(content="New turn", timestamp=source.now - 90)
    before = (source.home / "state.db").read_bytes()
    assert run_cli(["import", "--yes"]) == 0
    assert [message["content"] for message in uploaded(source)] == [
        "Original user",
        "Original assistant",
        "New turn",
    ]
    assert (source.home / "state.db").read_bytes() == before
    assert run_cli(["import", "--yes"]) == 0
    assert len(uploaded(source)) == 3


def test_compacted_rows_cross_source_pages_without_loss(history_source):
    source = history_source
    source.session()
    for index in range(205):
        source.message(
            content=f"Archived row {index}", timestamp=source.now - 1000 + index
        )
    source.db.archive_and_compact(
        "source-one", [{"role": "system", "content": "Summary"}]
    )
    assert run_cli(["import", "--yes"]) == 0
    assert [message["content"] for message in uploaded(source)] == [
        f"Archived row {index}" for index in range(205)
    ]
    assert [len(batch) for batch in source.api.writes] == [100, 100, 5]


def test_target_order_merges_interleaved_sources(history_source):
    source = history_source
    source.session("a")
    source.session("b")
    source.message("a", content="First", timestamp=source.now - 100)
    source.message("a", content="Third", timestamp=source.now - 50)
    source.message("b", content="Second", timestamp=source.now - 75)
    source.message("b", content="Fourth", timestamp=source.now - 25)
    assert run_cli(["import", "--yes"]) == 0
    assert [message["content"] for message in uploaded(source)] == [
        "First",
        "Second",
        "Third",
        "Fourth",
    ]


def test_timestamp_order_handles_regressing_source_clock(history_source):
    source = history_source
    source.session()
    source.message(content="Inserted first, later time", timestamp=source.now - 10)
    source.message(content="Inserted second, earlier time", timestamp=source.now - 20)
    assert run_cli(["import", "--yes"]) == 0
    assert [message["content"] for message in uploaded(source)] == [
        "Inserted second, earlier time",
        "Inserted first, later time",
    ]


def test_expanded_window_cannot_append_older_history(history_source, capsys):
    source = history_source
    source.session()
    source.message(content="Old", timestamp=source.now - 40 * 86400)
    source.message(content="Recent", timestamp=source.now - 3 * 86400)
    assert run_cli(["import", "--yes"]) == 0
    assert run_cli(["import", "--days", "60", "--yes"]) == 3
    assert [message["content"] for message in uploaded(source)] == ["Recent"]
    assert "older" in capsys.readouterr().out.lower()
    assert (
        run_cli(["import", "--days", "60", "--workspace", "fresh-history", "--yes"])
        == 0
    )
    assert [message["content"] for message in source.api.writes[-1]] == [
        "Old",
        "Recent",
    ]


@pytest.mark.parametrize("failure", [None, "timeout_after"])
def test_nul_is_normalized_before_response_and_recovery(history_source, failure):
    source = history_source
    source.session()
    source.message(content="Before\x00After")
    source.api.failure = failure
    assert run_cli(["import", "--yes"]) == (3 if failure else 0)
    assert source.api.writes[0][0]["content"] == "BeforeAfter"
    assert run_cli(["import", "--yes"]) == 0
    assert len(source.api.writes) == 1


def test_sorted_batch_recovery_keeps_cross_session_order(history_source):
    source = history_source
    source.session("a")
    source.session("b")
    for index in range(210):
        source.message(
            "a" if index % 2 else "b",
            content=f"Turn {index}",
            timestamp=source.now - 1000 + index,
        )
    source.api.failure = "timeout_after"
    assert run_cli(["import", "--yes"]) == 3
    assert [message["content"] for message in source.api.writes[0]] == [
        f"Turn {index}" for index in range(100)
    ]
    assert run_cli(["import", "--yes"]) == 0
    assert [message["content"] for message in uploaded(source)] == [
        f"Turn {index}" for index in range(210)
    ]


def test_local_import_does_not_require_gateway_package(history_source, monkeypatch):
    source = history_source
    source.session()
    source.message()
    original = builtins.__import__

    def without_gateway(name, *args, **kwargs):
        if name == "gateway.config":
            raise ImportError("gateway package unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_gateway)
    assert run_cli(["import", "--yes"]) == 0
    assert run_cli(["import", "--source", "telegram", "--dry-run"]) == 2


@pytest.mark.parametrize("error", [ImportStateError, RuntimeError])
def test_exception_after_upload_reports_partial_summary(
    history_source, monkeypatch, capsys, error
):
    source = history_source
    source.configure(sessionStrategy="per-session")
    source.session("a")
    source.session("b")
    source.message("a")
    source.message("b")
    original = importer._prepare_session

    def fail_second(client, target_id, *args):
        if target_id == "b":
            raise error("private failure body")
        return original(client, target_id, *args)

    monkeypatch.setattr(importer, "_prepare_session", fail_second)
    assert run_cli(["import", "--yes"]) == 3
    output = capsys.readouterr().out
    assert "Import partial or unresolved: 1 chunks uploaded" in output
    assert "private failure body" not in output


@pytest.mark.parametrize(
    "text",
    [
        "API_KEY=opaque-private-key-value",
        "https://someone:opaque-password@example.invalid/path?access_token=opaque-query-secret",
        "An old credential ghp_" + "T" * 16 + "\x00" + "T" * 20,
    ],
)
def test_normalized_wire_content_matches_hash_and_does_not_modify_source(
    history_source, text
):
    source = history_source
    source.session()
    row_id = source.message(content=text)
    expected = redact_sensitive_text(
        text.replace("\x00", ""), force=True, redact_url_credentials=True
    ).strip()
    assert run_cli(["import", "--yes"]) == 0
    message = uploaded(source)[0]
    assert message["content"] == expected
    assert message["metadata"]["payload_sha256"] == importer.payload_hash(
        expected, "owner", datetime.fromisoformat(message["created_at"]), "history-test"
    )
    assert source.db.get_messages("source-one")[0]["id"] == row_id
    assert source.db.get_messages("source-one")[0]["content"] == text
    assert run_cli(["import", "--yes"]) == 0
    assert len(source.api.writes) == 1


@pytest.mark.parametrize("text", ["\x00", " \x00\n\x00 "])
def test_nul_only_text_does_not_create_remote_objects(history_source, text):
    source = history_source
    source.session()
    source.message(content=text)
    assert run_cli(["import", "--yes"]) == 0
    source.factory.assert_not_called()
    assert not source.ledger.exists()


def test_user_role_compression_summary_is_not_uploaded(history_source):
    source = history_source
    source.session()
    source.message(content="Original retained", timestamp=source.now - 100)
    source.db.archive_and_compact(
        "source-one",
        [
            {
                "role": "user",
                "content": "Generated summary",
                "_compressed_summary": True,
            },
            {
                "role": "assistant",
                "content": "Summary acknowledgement",
                "_compressed_summary": True,
            },
        ],
    )
    source.message(content="Next original", timestamp=source.now - 90)
    assert run_cli(["import", "--yes"]) == 0
    assert [message["content"] for message in uploaded(source)] == [
        "Original retained",
        "Next original",
    ]


def test_compaction_cloned_tail_does_not_duplicate_original(history_source):
    source = history_source
    source.session()
    watermark = source.message(
        content="Summarized original", timestamp=source.now - 100
    )
    source.message(content="Concurrent tail", timestamp=source.now - 90)
    source.db.archive_and_compact(
        "source-one",
        [
            {"role": "user", "content": "Summary", "_compressed_summary": True},
        ],
        watermark=watermark,
    )
    assert run_cli(["import", "--yes"]) == 0
    assert [message["content"] for message in uploaded(source)] == [
        "Summarized original",
        "Concurrent tail",
    ]


def test_older_and_newer_candidates_block_only_affected_target(history_source):
    source = history_source
    source.configure(sessionStrategy="per-session")
    source.session("a")
    source.message("a", content="Already recent", timestamp=source.now - 50)
    assert run_cli(["import", "--yes"]) == 0
    source.message("a", content="Too old", timestamp=source.now - 100)
    source.message("a", content="Later new", timestamp=source.now - 20)
    source.session("b")
    source.message("b", content="Independent target", timestamp=source.now - 10)
    assert run_cli(["import", "--yes"]) == 3
    assert [message["content"] for message in uploaded(source)] == [
        "Already recent",
        "Independent target",
    ]


def test_known_boundary_can_append_after_older_unmarked_remote_history(history_source):
    source = history_source
    source.session()
    source.message(content="Earlier existing", timestamp=source.now - 10 * 86400)
    assert run_cli(["import", "--yes"]) == 0
    source.api.sessions[("history-test", "history-test")]["messages"][0][
        "metadata"
    ] = {}
    source.message(content="Later selected", timestamp=source.now - 2 * 86400)
    assert run_cli(["import", "--yes"]) == 3
    boundary = datetime.fromtimestamp(source.now - 86400, UTC).isoformat()
    assert run_cli(["import", "--before", boundary, "--yes"]) == 0
    assert [message["content"] for message in uploaded(source)] == [
        "Earlier existing",
        "Later selected",
    ]


def test_legacy_unredacted_progress_is_not_reset_or_resent(
    history_source, monkeypatch, capsys
):
    source = history_source
    source.session()
    source.message(content="ghp_" + "L" * 36)
    normalize = importer._normalize_import_text
    monkeypatch.setattr(importer, "_normalize_import_text", lambda text: text)
    assert run_cli(["import", "--yes"]) == 0
    before = source.ledger.read_bytes()
    monkeypatch.setattr(importer, "_normalize_import_text", normalize)
    source.factory.reset_mock()
    assert run_cli(["import", "--yes"]) == 2
    source.factory.assert_not_called()
    assert source.ledger.read_bytes() == before
    assert len(source.api.writes) == 1
    assert "normalized payload changed" in capsys.readouterr().out


def test_partial_remote_confirmation_save_failure_is_reported(
    history_source, monkeypatch, capsys
):
    source = history_source
    source.session()
    source.message()
    assert run_cli(["import", "--yes"]) == 0
    source.ledger.unlink()
    source.factory.reset_mock()

    def fail_save(*args, **kwargs):
        raise OSError("private filesystem path")

    monkeypatch.setattr(importer, "write_state", fail_save)
    assert run_cli(["import", "--yes"]) == 3
    output = capsys.readouterr().out
    assert "1 already remote" in output
    assert "private filesystem path" not in output
    assert len(source.api.writes) == 1
