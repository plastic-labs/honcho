"""Conclusion errors cross the real provider/manager boundary without extra writes."""

import json
from unittest.mock import MagicMock

import pytest
from hermes_plugin_honcho import HonchoMemoryProvider
from hermes_plugin_honcho.session import HonchoSession, HonchoSessionManager
from honcho.http.exceptions import (
    APIError,
    AuthenticationError,
    ConnectionError,
    NotFoundError,
    PermissionDeniedError,
    TimeoutError,
)


@pytest.fixture
def conclusion_writer(monkeypatch):
    """Provide an isolated conclusion tool with mocked SDK peers and OAuth refresh."""
    client = MagicMock()
    monkeypatch.setattr(
        "hermes_plugin_honcho.session.get_honcho_client", lambda config: client
    )
    manager = HonchoSessionManager()
    session = HonchoSession(
        key="telegram:123",
        user_peer_id="robert",
        assistant_peer_id="hermes",
        honcho_session_id="telegram-123",
    )
    manager._cache[session.key] = session
    observer = MagicMock()
    scope = observer.conclusions_of.return_value
    manager._peers_cache[session.assistant_peer_id] = observer
    provider = HonchoMemoryProvider()
    provider._session_initialized = True
    provider._session_key = session.key
    provider._manager = manager
    refresh = MagicMock(return_value=False)
    monkeypatch.setattr(manager, "_force_reauth", refresh)
    yield provider, manager, session, observer, scope, refresh
    manager.shutdown()


@pytest.mark.parametrize(
    "detail",
    [
        "Peer 'custom-peer' is not a participant in session 'telegram-123'.",
        "Conclusion resource was not found.",
    ],
)
def test_api_error_reaches_tool_response(conclusion_writer, detail):
    """Preserve useful API failure details in the tool response without retrying."""
    provider, _, _, observer, scope, refresh = conclusion_writer
    scope.create.side_effect = NotFoundError(detail)

    response = json.loads(
        provider.handle_tool_call(
            "honcho_conclude",
            {"conclusion": "Prefers dark mode", "peer": "custom-peer"},
        )
    )

    assert detail in response["error"]
    assert "404" in response["error"]
    assert "result" not in response
    observer.conclusions_of.assert_called_once_with("custom-peer")
    scope.create.assert_called_once()
    refresh.assert_not_called()


@pytest.mark.parametrize(
    "exc, expected",
    [
        (
            PermissionDeniedError("Workspace write permission is required."),
            "Workspace write permission is required.",
        ),
        (
            APIError("Invalid conclusion payload.", status=422),
            "Invalid conclusion payload.",
        ),
        (TimeoutError("private transport details"), "Honcho request timed out"),
        (
            ConnectionError("https://private-host.invalid"),
            "Could not connect to Honcho",
        ),
    ],
)
def test_failure_categories_do_not_retry(conclusion_writer, exc, expected):
    """Distinguish transport and API failures while attempting each write once."""
    provider, _, _, _, scope, refresh = conclusion_writer
    scope.create.side_effect = exc
    response = json.loads(
        provider.handle_tool_call("honcho_conclude", {"conclusion": "Fact"})
    )
    assert expected in response["error"]
    assert "private" not in response["error"]
    if isinstance(exc, (TimeoutError, ConnectionError)):
        assert "before retrying" in response["error"]
    scope.create.assert_called_once()
    refresh.assert_not_called()


@pytest.mark.parametrize(
    "peer, target, observes_others",
    [
        ("user", "robert", True),
        ("ai", "hermes", True),
        ("custom-peer", "custom-peer", True),
        ("user", "robert", False),
    ],
)
def test_success_preserves_alias_observer_and_payload(
    conclusion_writer, peer, target, observes_others
):
    """Keep existing observer, target, and conclusion payload behavior on success."""
    provider, manager, session, observer, scope, refresh = conclusion_writer
    manager._ai_observe_others = observes_others
    if not observes_others:
        manager._peers_cache[session.user_peer_id] = observer
        manager._peers_cache.pop(session.assistant_peer_id)
    response = json.loads(
        provider.handle_tool_call(
            "honcho_conclude",
            {"conclusion": "  Prefers dark mode  ", "peer": peer},
        )
    )
    assert response == {"result": f"Conclusion saved for {peer}: Prefers dark mode"}
    observer.conclusions_of.assert_called_once_with(target)
    scope.create.assert_called_once_with(
        [{"content": "Prefers dark mode", "session_id": session.honcho_session_id}]
    )
    refresh.assert_not_called()


def test_default_failure_and_strict_exception_identity(conclusion_writer):
    """Retain default suppression while strict writes raise the original exception."""
    _, manager, session, _, scope, _ = conclusion_writer
    exc = NotFoundError("Peer is not a participant.")
    scope.create.side_effect = exc
    assert manager.create_conclusion(session.key, "Fact") is False
    with pytest.raises(NotFoundError) as caught:
        manager.create_conclusion(session.key, "Fact", raise_on_error=True)
    assert caught.value is exc
    assert scope.create.call_count == 2


def test_background_memory_write_keeps_default_mode(conclusion_writer, monkeypatch):
    """Keep background memory writes on the existing non-strict failure path."""
    provider, manager, _, _, scope, _ = conclusion_writer
    scope.create.side_effect = NotFoundError("Background save failed.")
    monkeypatch.setattr(provider, "_writes_enabled", lambda: True)
    monkeypatch.setattr(provider, "_ready_or_kick_init", lambda: True)
    create = MagicMock(wraps=manager.create_conclusion)
    monkeypatch.setattr(manager, "create_conclusion", create)
    provider.on_memory_write("add", "user", "Fact")
    provider._memwrite_thread.join(timeout=2)
    assert not provider._memwrite_thread.is_alive()
    create.assert_called_once_with(provider._session_key, "Fact")
    scope.create.assert_called_once()


@pytest.mark.parametrize(
    "condition, reason",
    [
        ("empty", "content is empty"),
        ("missing", "No Honcho session is cached"),
        ("peer", "Could not resolve the conclusion peer"),
    ],
)
def test_local_preconditions(conclusion_writer, monkeypatch, condition, reason):
    """Report local session and peer preconditions through typed write errors."""
    from hermes_plugin_honcho.session_context import ConclusionWriteError

    provider, manager, session, _, scope, _ = conclusion_writer
    content = " " if condition == "empty" else "Fact"
    if condition == "missing":
        manager._cache.clear()
    if condition == "peer":
        monkeypatch.setattr(manager, "_resolve_peer_id", lambda session, peer: None)
    assert manager.create_conclusion(session.key, content) is False
    with pytest.raises(ConclusionWriteError, match=reason):
        manager.create_conclusion(session.key, content, raise_on_error=True)
    if condition != "empty":
        response = json.loads(
            provider.handle_tool_call("honcho_conclude", {"conclusion": content})
        )
        assert reason in response["error"]
    scope.create.assert_not_called()


@pytest.mark.parametrize("outcome", ["recovered", "refresh_failed", "retry_rejected"])
def test_auth_recovery_keeps_existing_contract(conclusion_writer, outcome):
    """Preserve refresh and retry behavior for authentication failures."""
    provider, _, _, _, scope, refresh = conclusion_writer
    refresh.return_value = outcome != "refresh_failed"
    scope.create.side_effect = (
        [AuthenticationError(), None]
        if outcome == "recovered"
        else AuthenticationError()
    )
    response = json.loads(
        provider.handle_tool_call("honcho_conclude", {"conclusion": "Fact"})
    )
    if outcome == "recovered":
        assert response == {"result": "Conclusion saved for user: Fact"}
    else:
        assert "Honcho memory authentication failed" in response["error"]
        assert "hermes honcho setup" in response["error"]
        assert "Failed to save conclusion" not in response["error"]
    assert scope.create.call_count == (1 if outcome == "refresh_failed" else 2)
    refresh.assert_called_once()


def test_api_detail_and_new_log_are_redacted(conclusion_writer, caplog):
    """Mask credential-bearing API details in both the tool output and logs."""
    provider, _, _, _, scope, _ = conclusion_writer
    secrets = [
        "sk-" + "f" * 48,
        "fake-url-user",
        "fake-url-password",
        "fake-query-token",
    ]
    detail = (
        f"Peer is not a participant. api_key={secrets[0]} "
        f"https://{secrets[1]}:{secrets[2]}@example.invalid/path?token={secrets[3]}"
    )
    scope.create.side_effect = APIError(
        detail, status=404, body={"secret": "body-must-not-appear"}
    )
    response = json.loads(
        provider.handle_tool_call("honcho_conclude", {"conclusion": "Fact"})
    )
    assert "Peer is not a participant" in response["error"]
    for secret in secrets + ["body-must-not-appear"]:
        assert secret not in response["error"]
        assert secret not in caplog.text


def test_unknown_error_does_not_expose_raw_text(conclusion_writer, caplog):
    """Hide an unexpected exception's private message from the tool response."""
    provider, _, _, _, scope, _ = conclusion_writer
    scope.create.side_effect = ValueError("unknown-private-response-body")
    response = json.loads(
        provider.handle_tool_call("honcho_conclude", {"conclusion": "Fact"})
    )
    assert "ValueError" in response["error"]
    assert "check the Honcho logs" in response["error"]
    assert "unknown-private-response-body" not in response["error"] + caplog.text


def test_redaction_precedes_control_cleanup_and_truncation(
    conclusion_writer, monkeypatch, caplog
):
    """Redact the original diagnostic before removing controls or limiting length."""
    from agent import redact

    provider, _, _, _, scope, _ = conclusion_writer
    secret = "test-secret-spans-the-truncation-boundary"
    detail = "a" * 940 + secret + "\x00\n\t" + "b" * 1400
    seen = []

    def redactor(text, **kwargs):
        """Capture the unmodified diagnostic and replace the synthetic secret."""
        seen.append((text, kwargs))
        return text.replace(secret, "[REDACTED]")

    monkeypatch.setattr(redact, "redact_sensitive_text", redactor)
    scope.create.side_effect = APIError(detail, status=404)
    response = json.loads(
        provider.handle_tool_call("honcho_conclude", {"conclusion": "Fact"})
    )
    assert seen == [
        (
            f"Failed to save conclusion (404): {detail}",
            {"force": True, "redact_url_credentials": True},
        )
    ]
    message = response["error"]
    assert len(message) == 1000
    assert message.endswith("...")
    assert "[REDACTED]" in message
    assert secret[:10] not in message + caplog.text
    assert not any(ord(char) < 32 for char in message)


def test_redactor_failure_uses_safe_fallback(conclusion_writer, monkeypatch, caplog):
    """Return the fixed safe diagnostic when the redactor itself fails."""
    provider, _, _, _, scope, _ = conclusion_writer
    scope.create.side_effect = NotFoundError("private-error-detail")
    monkeypatch.setattr(
        "agent.redact.redact_sensitive_text",
        MagicMock(side_effect=RuntimeError("redactor failed")),
    )
    response = json.loads(
        provider.handle_tool_call("honcho_conclude", {"conclusion": "Fact"})
    )
    assert "Error details unavailable" in response["error"]
    assert "private-error-detail" not in response["error"] + caplog.text


@pytest.mark.parametrize("status", [0, True, "404", 999])
def test_invalid_status_is_not_rendered(status):
    """Exclude values outside valid integer HTTP status codes from diagnostics."""
    from hermes_plugin_honcho import _conclusion_error_message

    assert (
        _conclusion_error_message(APIError("Reason.", status=status))
        == "Failed to save conclusion: Reason."
    )
