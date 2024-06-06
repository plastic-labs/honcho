import uuid

import pytest

from src.main import app


@pytest.fixture()
def user_name() -> str:
    return str(uuid.uuid4())


@pytest.fixture()
def client():
    with app.test_client() as client:
        yield client
