"""Tests for vector reconciliation configuration logic.

These are unit tests that test configuration logic without requiring database
or vector store fixtures.
"""

from src.config import VectorStoreSettings


def test_reconciliation_enabled_pgvector_primary():
    """Reconciliation enabled when pgvector is primary with secondary"""
    settings = VectorStoreSettings(
        PRIMARY_TYPE="pgvector",
        SECONDARY_TYPE="turbopuffer",
        TURBOPUFFER_API_KEY="test-key",
    )
    assert settings.should_run_reconciliation is True


def test_reconciliation_enabled_pgvector_secondary():
    """Reconciliation enabled when pgvector is secondary"""
    settings = VectorStoreSettings(
        PRIMARY_TYPE="turbopuffer",
        SECONDARY_TYPE="pgvector",
        TURBOPUFFER_API_KEY="test-key",
    )
    assert settings.should_run_reconciliation is True


def test_reconciliation_disabled_single_store():
    """Reconciliation disabled when no secondary configured"""
    settings = VectorStoreSettings(
        PRIMARY_TYPE="turbopuffer", TURBOPUFFER_API_KEY="test-key", SECONDARY_TYPE=None
    )
    assert settings.should_run_reconciliation is False


def test_reconciliation_disabled_no_pgvector():
    """Reconciliation disabled when both stores are non-pgvector"""
    settings = VectorStoreSettings(
        PRIMARY_TYPE="turbopuffer",
        SECONDARY_TYPE="lancedb",
        TURBOPUFFER_API_KEY="test-key",
    )
    assert settings.should_run_reconciliation is False


def test_reconciliation_disabled_pgvector_only():
    """Reconciliation disabled when pgvector is primary but no secondary"""
    settings = VectorStoreSettings(PRIMARY_TYPE="pgvector", SECONDARY_TYPE=None)
    assert settings.should_run_reconciliation is False
