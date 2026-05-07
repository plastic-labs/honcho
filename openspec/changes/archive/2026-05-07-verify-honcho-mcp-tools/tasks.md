## 1. Setup & Lifecycle Validation (Isolated Data)

- [x] 1.0 Ensure isolation: All peer/session IDs MUST use `test-qa-` prefix to avoid touching existing data.

- [x] 1.1 Verify `inspect_workspace` and `list_workspaces`
- [x] 1.2 Verify `create_peer` and `list_peers`
- [x] 1.3 Verify `create_session` and `add_peers_to_session`
- [x] 1.4 Verify `get_session_peers` and `inspect_session`

## 2. Interaction & Data Insertion

- [x] 2.1 Verify `add_messages_to_session`
- [x] 2.2 Verify `get_session_messages`
- [x] 2.3 Verify `set_metadata` and `get_metadata`

## 3. Context & Search Verification

- [x] 3.1 Verify `search`
- [x] 3.2 Verify `get_session_context`

## 4. Knowledge Graph & Reasoning

- [x] 4.1 Verify `create_conclusions`, `list_conclusions`, and `query_conclusions`
- [x] 4.2 Verify `set_peer_card` and `get_peer_card`
- [x] 4.3 Verify `get_peer_context` and `get_representation`
- [x] 4.4 Verify `schedule_dream` and `get_queue_status`
- [x] 4.5 Verify `chat`

## 5. Teardown Validation (Strict Cleanup)

- [x] 5.1 Verify `remove_peers_from_session`
- [x] 5.2 Verify `delete_conclusion` (Remove all test conclusions)
- [x] 5.3 Verify `delete_session` (Delete the test session and verify via `list_sessions`)
