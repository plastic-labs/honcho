"""Unit tests for the SDK's scope / session-allowlist option handling.

Pure logic — no server, no database. These pin the wire translation the server
expects, so a rename or a shape change fails here rather than as a 422 at runtime.
"""

import sys
from pathlib import Path

import pytest

# Add the SDK src to the path to allow imports
sdk_src_path = Path(__file__).parent.parent.parent / "sdks" / "python" / "src"
sys.path.insert(0, str(sdk_src_path))

from sdks.python.src.honcho.utils.scopes import (  # noqa: E402
    MAX_SCOPES_PER_OPTION,
    MAX_SESSION_ALLOWLIST_ENTRIES,
    MAX_SESSIONS_PER_ADD,
    resolve_scope_membership,
    resolve_scope_option,
    resolve_scope_session,
    scope_context_fields,
    scope_recall_fields,
    validate_scope_id,
)


def context_fields(**overrides: object) -> dict[str, object]:
    """Call scope_context_fields with the neutral defaults filled in."""
    kwargs: dict[str, object] = {
        "scope": None,
        "sessions": None,
        "peer_target": "user",
        "peer_perspective": None,
        "limit_to_session": False,
    }
    kwargs.update(overrides)
    return scope_context_fields(**kwargs)  # pyright: ignore[reportArgumentType]


class TestValidateScopeId:
    def test_accepts_a_plain_name(self):
        assert validate_scope_id("therapy") == "therapy"

    def test_rejects_the_reserved_prefix_by_name(self):
        # 'scope.therapy' violates both the prefix rule and the charset. The
        # prefix message is the actionable one, so it must be the one raised.
        with pytest.raises(ValueError, match="reserved prefix"):
            validate_scope_id("scope.therapy")

    def test_rejects_characters_outside_the_charset(self):
        with pytest.raises(ValueError, match="must match pattern"):
            validate_scope_id("my scope")

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="between 1 and"):
            validate_scope_id("")

    def test_rejects_a_name_that_leaves_no_room_for_the_prefix(self):
        # 512 - len("scope.") is the ceiling: the server stores the name prefixed
        # into a 512-character peer name.
        with pytest.raises(ValueError, match="between 1 and"):
            validate_scope_id("a" * 507)


class TestResolveScopeOption:
    def test_a_single_scope_stays_a_string(self):
        # The shapes are not interchangeable to the server: one scope reads that
        # scope's own view, a list restricts to the union of member sessions.
        assert resolve_scope_option("therapy") == "therapy"

    def test_a_sequence_becomes_a_list(self):
        assert resolve_scope_option(["therapy", "work"]) == ["therapy", "work"]

    def test_rejects_an_empty_sequence(self):
        with pytest.raises(ValueError, match="at least one scope"):
            resolve_scope_option([])

    def test_rejects_an_over_cap_sequence(self):
        with pytest.raises(ValueError, match="at most"):
            resolve_scope_option([f"s{i}" for i in range(MAX_SCOPES_PER_OPTION + 1)])


class TestScopeRecallFields:
    def test_neither_option_contributes_nothing(self):
        assert scope_recall_fields(scope=None, sessions=None) == {}

    def test_sessions_becomes_a_session_id_filter(self):
        # `sessions` is sugar. It must never reach the wire as its own key —
        # the server rejects unknown keys with a 422.
        fields = scope_recall_fields(scope=None, sessions=["a", "b"])
        assert fields == {"filters": {"session_id": ["a", "b"]}}
        assert "sessions" not in fields

    def test_scope_passes_through_under_its_own_key(self):
        assert scope_recall_fields(scope="therapy", sessions=None) == {
            "scope": "therapy"
        }

    def test_scope_and_sessions_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            scope_recall_fields(scope="therapy", sessions=["a"])

    def test_scope_and_a_single_session_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            scope_recall_fields(scope="therapy", sessions=None, session_id="a")

    def test_sessions_composes_with_a_single_session(self):
        # Unlike `scope`, an allowlist may accompany a session_id — the server
        # only requires that the session be inside the allowlist.
        assert scope_recall_fields(scope=None, sessions=["a", "b"], session_id="a") == {
            "filters": {"session_id": ["a", "b"]}
        }

    def test_rejects_an_empty_allowlist(self):
        # An empty allowlist is fail-closed server-side (recalls nothing), which
        # is never what `sessions=[]` intends.
        with pytest.raises(ValueError, match="at least one session"):
            scope_recall_fields(scope=None, sessions=[])

    def test_rejects_an_over_cap_allowlist(self):
        with pytest.raises(ValueError, match="at most"):
            scope_recall_fields(
                scope=None,
                sessions=[f"s{i}" for i in range(MAX_SESSION_ALLOWLIST_ENTRIES + 1)],
            )

    def test_resolves_objects_with_an_id(self):
        class FakeSession:
            id: str = "session-a"

        fields = scope_recall_fields(scope=None, sessions=[FakeSession()])  # pyright: ignore[reportArgumentType]
        assert fields == {"filters": {"session_id": ["session-a"]}}


class TestScopeContextFields:
    """The context route takes these as query params, not as a `filters` body."""

    def test_neither_option_contributes_nothing(self):
        assert context_fields() == {}

    def test_scope_passes_through(self):
        assert context_fields(scope="therapy") == {"scope": "therapy"}

    def test_sessions_stays_a_plain_list(self):
        # Not wrapped in `filters` — this route reads a repeated query parameter.
        assert context_fields(sessions=["a", "b"]) == {"sessions": ["a", "b"]}

    def test_scope_and_peer_perspective_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            context_fields(scope="therapy", peer_perspective="assistant")

    def test_scope_and_sessions_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            context_fields(scope="therapy", sessions=["a"])

    def test_sessions_and_limit_to_session_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            context_fields(sessions=["a"], limit_to_session=True)

    @pytest.mark.parametrize("option", [{"scope": "therapy"}, {"sessions": ["a"]}])
    def test_either_option_requires_a_peer_target(self, option: dict[str, object]):
        # Both only reach the representation, and there is none without a target.
        # Refused rather than accepted and silently ignored.
        with pytest.raises(ValueError, match="peer_target"):
            context_fields(peer_target=None, **option)

    def test_limit_to_session_alone_is_untouched(self):
        # The neutral case must not start emitting a scope/sessions key.
        assert context_fields(limit_to_session=True) == {}


class TestResolveScopeMembership:
    def test_resolves_ids_and_objects_in_order(self):
        class FakeSession:
            id: str = "session-b"

        assert resolve_scope_membership(["session-a", FakeSession()]) == [  # pyright: ignore[reportArgumentType]
            "session-a",
            "session-b",
        ]

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="At least one session"):
            resolve_scope_membership([])

    def test_rejects_over_the_per_call_cap_rather_than_chunking(self):
        with pytest.raises(ValueError, match="At most"):
            resolve_scope_membership([f"s{i}" for i in range(MAX_SESSIONS_PER_ADD + 1)])

    def test_rejects_a_malformed_id(self):
        with pytest.raises(ValueError, match="must match pattern"):
            resolve_scope_membership(["ok-session", "valid-session?typo"])


class TestResolveScopeSession:
    """Guards the ID that gets interpolated into a scope membership URL path."""

    def test_resolves_a_plain_id(self):
        assert resolve_scope_session("session-a") == "session-a"

    def test_resolves_an_object(self):
        class FakeSession:
            id: str = "session-a"

        assert resolve_scope_session(FakeSession()) == "session-a"  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize(
        "malformed",
        [
            "valid-session?typo",  # would address `valid-session` + a query string
            "valid-session/../other",  # would climb the path
            "valid session",
            "",
        ],
    )
    def test_rejects_ids_that_would_alter_the_request_path(self, malformed: str):
        # This value lands in a DELETE path. An unvalidated id silently changes
        # which session is removed, and removal triggers reconciliation against
        # whatever it hits.
        with pytest.raises(ValueError, match="Session ID"):
            resolve_scope_session(malformed)


def test_deprecated_conclusion_scope_aliases_still_resolve():
    """The rename keeps working for callers on the old name."""
    from sdks.python.src.honcho import (
        ConclusionScope,
        ConclusionScopeAio,
        ConclusionsView,
        ConclusionsViewAio,
    )

    assert ConclusionScope is ConclusionsView
    assert ConclusionScopeAio is ConclusionsViewAio
