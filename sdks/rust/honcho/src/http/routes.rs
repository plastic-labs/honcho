#![allow(dead_code)] // Phase 2 route helpers — consumed by Phase 3 high-level API

pub(crate) const API_BASE_PATH: &str = "v3";

/// Builds path for listing all workspaces.
pub(crate) fn workspaces() -> String {
    format!("/{API_BASE_PATH}/workspaces")
}

/// Builds path for the workspace list endpoint.
pub(crate) fn workspaces_list() -> String {
    format!("/{API_BASE_PATH}/workspaces/list")
}

/// Builds path for a specific workspace.
pub(crate) fn workspace(workspace_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}")
}

/// Builds path for searching within a workspace.
pub(crate) fn workspace_search(workspace_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/search")
}

/// Builds path for workspace queue status.
pub(crate) fn workspace_queue_status(workspace_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/queue/status")
}

/// Builds path for scheduling a dream in a workspace.
pub(crate) fn workspace_schedule_dream(workspace_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/schedule_dream")
}

/// Builds path for listing peers in a workspace.
pub(crate) fn peers(workspace_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/peers")
}

/// Builds path for the peer list endpoint in a workspace.
pub(crate) fn peers_list(workspace_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/peers/list")
}

/// Builds path for a specific peer.
pub(crate) fn peer(workspace_id: &str, peer_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!peer_id.is_empty(), "peer_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/peers/{peer_id}")
}

/// Builds path for the peer chat (dialectic) endpoint.
pub(crate) fn peer_chat(workspace_id: &str, peer_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!peer_id.is_empty(), "peer_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/peers/{peer_id}/chat")
}

/// Builds path for a peer's representation.
pub(crate) fn peer_representation(workspace_id: &str, peer_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!peer_id.is_empty(), "peer_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/peers/{peer_id}/representation")
}

/// Builds path for a peer's card.
pub(crate) fn peer_card(workspace_id: &str, peer_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!peer_id.is_empty(), "peer_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/peers/{peer_id}/card")
}

/// Builds path for a peer's context.
pub(crate) fn peer_context(workspace_id: &str, peer_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!peer_id.is_empty(), "peer_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/peers/{peer_id}/context")
}

/// Builds path for searching within a peer.
pub(crate) fn peer_search(workspace_id: &str, peer_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!peer_id.is_empty(), "peer_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/peers/{peer_id}/search")
}

/// Builds path for listing a peer's sessions.
pub(crate) fn peer_sessions_list(workspace_id: &str, peer_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!peer_id.is_empty(), "peer_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/peers/{peer_id}/sessions")
}

/// Builds path for creating/listing sessions in a workspace.
pub(crate) fn sessions(workspace_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/sessions")
}

/// Builds path for the session list endpoint.
pub(crate) fn sessions_list(workspace_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/sessions/list")
}

/// Builds path for a specific session.
pub(crate) fn session(workspace_id: &str, session_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!session_id.is_empty(), "session_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/sessions/{session_id}")
}

/// Builds path for cloning a session.
pub(crate) fn session_clone(workspace_id: &str, session_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!session_id.is_empty(), "session_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/sessions/{session_id}/clone")
}

/// Builds path for a session's context.
pub(crate) fn session_context(workspace_id: &str, session_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!session_id.is_empty(), "session_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/sessions/{session_id}/context")
}

/// Builds path for a session's summaries.
pub(crate) fn session_summaries(workspace_id: &str, session_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!session_id.is_empty(), "session_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/sessions/{session_id}/summaries")
}

/// Builds path for searching within a session.
pub(crate) fn session_search(workspace_id: &str, session_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!session_id.is_empty(), "session_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/sessions/{session_id}/search")
}

/// Builds path for listing peers in a session.
pub(crate) fn session_peers(workspace_id: &str, session_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!session_id.is_empty(), "session_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/sessions/{session_id}/peers")
}

/// Builds path for a peer's configuration within a session.
pub(crate) fn session_peer_config(workspace_id: &str, session_id: &str, peer_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!session_id.is_empty(), "session_id must not be empty");
    debug_assert!(!peer_id.is_empty(), "peer_id must not be empty");
    format!(
        "/{API_BASE_PATH}/workspaces/{workspace_id}/sessions/{session_id}/peers/{peer_id}/config"
    )
}

/// Builds path for creating/listing messages in a session.
pub(crate) fn messages(workspace_id: &str, session_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!session_id.is_empty(), "session_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/sessions/{session_id}/messages")
}

/// Builds path for the message list endpoint.
pub(crate) fn messages_list(workspace_id: &str, session_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!session_id.is_empty(), "session_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/sessions/{session_id}/messages/list")
}

/// Builds path for a specific message.
pub(crate) fn message(workspace_id: &str, session_id: &str, message_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!session_id.is_empty(), "session_id must not be empty");
    debug_assert!(!message_id.is_empty(), "message_id must not be empty");
    format!(
        "/{API_BASE_PATH}/workspaces/{workspace_id}/sessions/{session_id}/messages/{message_id}"
    )
}

/// Builds path for uploading messages to a session.
pub(crate) fn messages_upload(workspace_id: &str, session_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!session_id.is_empty(), "session_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/sessions/{session_id}/messages/upload")
}

/// Builds path for listing conclusions in a workspace.
pub(crate) fn conclusions(workspace_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/conclusions")
}

/// Builds path for the conclusions list endpoint.
pub(crate) fn conclusions_list(workspace_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/conclusions/list")
}

/// Builds path for querying conclusions.
pub(crate) fn conclusions_query(workspace_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/conclusions/query")
}

/// Builds path for a specific conclusion.
pub(crate) fn conclusion(workspace_id: &str, conclusion_id: &str) -> String {
    debug_assert!(!workspace_id.is_empty(), "workspace_id must not be empty");
    debug_assert!(!conclusion_id.is_empty(), "conclusion_id must not be empty");
    format!("/{API_BASE_PATH}/workspaces/{workspace_id}/conclusions/{conclusion_id}")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn test_workspaces() {
        assert_eq!(workspaces(), "/v3/workspaces");
    }

    #[test]
    fn test_workspaces_list() {
        assert_eq!(workspaces_list(), "/v3/workspaces/list");
    }

    #[test]
    fn test_workspace() {
        assert_eq!(workspace("ws1"), "/v3/workspaces/ws1");
    }

    #[test]
    fn test_workspace_search() {
        assert_eq!(workspace_search("ws1"), "/v3/workspaces/ws1/search");
    }

    #[test]
    fn test_workspace_queue_status() {
        assert_eq!(
            workspace_queue_status("ws1"),
            "/v3/workspaces/ws1/queue/status"
        );
    }

    #[test]
    fn test_workspace_schedule_dream() {
        assert_eq!(
            workspace_schedule_dream("ws1"),
            "/v3/workspaces/ws1/schedule_dream"
        );
    }

    #[test]
    fn test_peers() {
        assert_eq!(peers("ws1"), "/v3/workspaces/ws1/peers");
    }

    #[test]
    fn test_peers_list() {
        assert_eq!(peers_list("ws1"), "/v3/workspaces/ws1/peers/list");
    }

    #[test]
    fn test_peer() {
        assert_eq!(peer("ws1", "alice"), "/v3/workspaces/ws1/peers/alice");
    }

    #[test]
    fn test_peer_chat() {
        assert_eq!(
            peer_chat("ws1", "alice"),
            "/v3/workspaces/ws1/peers/alice/chat"
        );
    }

    #[test]
    fn test_peer_representation() {
        assert_eq!(
            peer_representation("ws1", "alice"),
            "/v3/workspaces/ws1/peers/alice/representation"
        );
    }

    #[test]
    fn test_peer_card() {
        assert_eq!(
            peer_card("ws1", "alice"),
            "/v3/workspaces/ws1/peers/alice/card"
        );
    }

    #[test]
    fn test_peer_context() {
        assert_eq!(
            peer_context("ws1", "alice"),
            "/v3/workspaces/ws1/peers/alice/context"
        );
    }

    #[test]
    fn test_peer_search() {
        assert_eq!(
            peer_search("ws1", "alice"),
            "/v3/workspaces/ws1/peers/alice/search"
        );
    }

    #[test]
    fn test_peer_sessions_list() {
        assert_eq!(
            peer_sessions_list("ws1", "alice"),
            "/v3/workspaces/ws1/peers/alice/sessions"
        );
    }

    #[test]
    fn test_sessions() {
        assert_eq!(sessions("ws1"), "/v3/workspaces/ws1/sessions");
    }

    #[test]
    fn test_sessions_list() {
        assert_eq!(sessions_list("ws1"), "/v3/workspaces/ws1/sessions/list");
    }

    #[test]
    fn test_session() {
        assert_eq!(session("ws1", "sess1"), "/v3/workspaces/ws1/sessions/sess1");
    }

    #[test]
    fn test_session_clone() {
        assert_eq!(
            session_clone("ws1", "sess1"),
            "/v3/workspaces/ws1/sessions/sess1/clone"
        );
    }

    #[test]
    fn test_session_context() {
        assert_eq!(
            session_context("ws1", "sess1"),
            "/v3/workspaces/ws1/sessions/sess1/context"
        );
    }

    #[test]
    fn test_session_summaries() {
        assert_eq!(
            session_summaries("ws1", "sess1"),
            "/v3/workspaces/ws1/sessions/sess1/summaries"
        );
    }

    #[test]
    fn test_session_search() {
        assert_eq!(
            session_search("ws1", "sess1"),
            "/v3/workspaces/ws1/sessions/sess1/search"
        );
    }

    #[test]
    fn test_session_peers() {
        assert_eq!(
            session_peers("ws1", "sess1"),
            "/v3/workspaces/ws1/sessions/sess1/peers"
        );
    }

    #[test]
    fn test_session_peer_config() {
        assert_eq!(
            session_peer_config("ws1", "sess1", "alice"),
            "/v3/workspaces/ws1/sessions/sess1/peers/alice/config"
        );
    }

    #[test]
    fn test_messages() {
        assert_eq!(
            messages("ws1", "sess1"),
            "/v3/workspaces/ws1/sessions/sess1/messages"
        );
    }

    #[test]
    fn test_messages_list() {
        assert_eq!(
            messages_list("ws1", "sess1"),
            "/v3/workspaces/ws1/sessions/sess1/messages/list"
        );
    }

    #[test]
    fn test_message() {
        assert_eq!(
            message("ws1", "sess1", "msg1"),
            "/v3/workspaces/ws1/sessions/sess1/messages/msg1"
        );
    }

    #[test]
    fn test_messages_upload() {
        assert_eq!(
            messages_upload("ws1", "sess1"),
            "/v3/workspaces/ws1/sessions/sess1/messages/upload"
        );
    }

    #[test]
    fn test_conclusions() {
        assert_eq!(conclusions("ws1"), "/v3/workspaces/ws1/conclusions");
    }

    #[test]
    fn test_conclusions_list() {
        assert_eq!(
            conclusions_list("ws1"),
            "/v3/workspaces/ws1/conclusions/list"
        );
    }

    #[test]
    fn test_conclusions_query() {
        assert_eq!(
            conclusions_query("ws1"),
            "/v3/workspaces/ws1/conclusions/query"
        );
    }

    #[test]
    fn test_conclusion() {
        assert_eq!(
            conclusion("ws1", "conc1"),
            "/v3/workspaces/ws1/conclusions/conc1"
        );
    }

    #[test]
    #[should_panic(expected = "workspace_id must not be empty")]
    fn test_workspace_empty_id_panics() {
        workspace("");
    }

    #[test]
    #[should_panic(expected = "peer_id must not be empty")]
    fn test_peer_empty_peer_id_panics() {
        peer("ws1", "");
    }

    #[test]
    #[should_panic(expected = "session_id must not be empty")]
    fn test_session_empty_session_id_panics() {
        session("ws1", "");
    }

    #[test]
    #[should_panic(expected = "message_id must not be empty")]
    fn test_message_empty_message_id_panics() {
        message("ws1", "sess1", "");
    }

    #[test]
    #[should_panic(expected = "conclusion_id must not be empty")]
    fn test_conclusion_empty_id_panics() {
        conclusion("ws1", "");
    }
}
