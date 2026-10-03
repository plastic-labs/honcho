"""Provider initialization contracts for memory-file migration."""

from hermes_honcho import HonchoMemoryProvider


class TestPerSessionMigrateGuard:
    """Verify migrate_memory_files is skipped under per-session strategy.

    per-session creates a fresh Honcho session every Hermes run. Uploading
    MEMORY.md/USER.md/SOUL.md to each short-lived session floods the backend
    with duplicate content. The guard was added to prevent orphan sessions
    containing only <prior_memory_file> wrappers.
    """

    def _make_provider_with_strategy(
        self,
        strategy,
        init_on_session_start=True,
        existing_messages=None,
    ):
        """Create a HonchoMemoryProvider and track migrate_memory_files calls."""
        from unittest.mock import MagicMock, patch

        from hermes_honcho.client import HonchoClientConfig

        cfg = HonchoClientConfig(
            api_key="test-key",
            enabled=True,
            recall_mode="tools",
            init_on_session_start=init_on_session_start,
            session_strategy=strategy,
        )

        provider = HonchoMemoryProvider()

        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.messages = existing_messages or []
        mock_manager.get_or_create.return_value = mock_session

        with (
            patch(
                "hermes_honcho.client.HonchoClientConfig.from_global_config",
                return_value=cfg,
            ),
            patch("hermes_honcho.client.get_honcho_client", return_value=MagicMock()),
            patch(
                "hermes_honcho.session.HonchoSessionManager", return_value=mock_manager
            ),
            patch("hermes_constants.get_hermes_home", return_value=MagicMock()),
        ):
            provider.initialize(session_id="test-session-001")

        return provider, mock_manager

    def test_migrate_skipped_for_per_session(self):
        """per-session strategy must NOT call migrate_memory_files."""
        _, mock_manager = self._make_provider_with_strategy("per-session")
        mock_manager.migrate_memory_files.assert_not_called()

    def test_migrate_runs_for_per_directory(self):
        """The ledger is checked even when the resolved session has messages."""
        _, mock_manager = self._make_provider_with_strategy(
            "per-directory",
            existing_messages=[{"role": "user", "content": "existing"}],
        )
        mock_manager.migrate_memory_files.assert_called_once()
