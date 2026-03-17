import pytest


@pytest.fixture(autouse=True)
def mock_tracked_db():
    """Disable the DB-heavy tracked_db autouse fixture for setup example tests."""
    yield
