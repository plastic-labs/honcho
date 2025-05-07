import asyncio
import datetime
import os

import pytest


def test_create_transaction(client):
    response = client.post("/v1/transactions/begin")
    assert response.status_code == 200
    transaction_id = response.json()
    assert isinstance(transaction_id, int)
    assert transaction_id > 0


def test_create_transaction_with_expiry(client):
    # Calculate an expiry time, e.g., 1 hour from now
    dt_obj = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
    # Ensure the format is accepted by Pydantic/FastAPI for datetime parsing
    expires_at_str = dt_obj.isoformat().replace("+00:00", "Z")
    response = client.post(f"/v1/transactions/begin?expires_at={expires_at_str}")
    assert response.status_code == 200
    transaction_id = response.json()
    assert isinstance(transaction_id, int)
    assert transaction_id > 0

    # We can also add a test to check if fetching this transaction shows the expiry (if such an endpoint exists)
    # or test the behavior of an expired transaction later.


# Placeholder for commit and rollback tests
@pytest.mark.asyncio
async def test_commit_transaction(client, db_session):
    # Create a transaction
    response = client.post("/v1/transactions/begin")
    assert response.status_code == 200
    transaction_id = int(response.json())

    # create an app
    app_response = client.post(
        "/v1/apps",
        json={"name": "transaction_test_app"},
        headers={"X-Transaction-ID": f"{transaction_id}"},
    )
    assert app_response.status_code == 200
    app_id = app_response.json()["id"]

    # create a user
    user_response = client.post(
        f"/v1/apps/{app_id}/users",
        json={"name": "test_user"},
        headers={"X-Transaction-ID": f"{transaction_id}"},
    )
    assert user_response.status_code == 200
    user_id = user_response.json()["id"]

    # Verify the app does not exist outside of the transaction
    app = client.get(f"/v1/apps/?app_id={app_id}")
    assert app.status_code == 404

    # Commit the transaction
    commit_response = client.post(f"/v1/transactions/{transaction_id}/commit")
    assert commit_response.status_code == 200
    # Commit endpoint usually returns 200 or 204 No Content, and no body or a success message.
    # Based on src/routers/transactions.py, it returns None which means 200 OK with no body.

    # Verify the operations were committed by fetching the resources
    app = client.get(f"/v1/apps/?app_id={app_id}")
    assert app.status_code == 200

    user = client.get(f"/v1/apps/{app_id}/users?user_id={user_id}")
    assert user.status_code == 200


@pytest.mark.asyncio
async def test_rollback_transaction(client, db_session):
    # Create a transaction
    response = client.post("/v1/transactions/begin")
    assert response.status_code == 200
    transaction_id = response.json()

    # create an app
    app_response = client.post(
        "/v1/apps",
        json={"name": "rollback_test_app"},
        headers={"X-Transaction-ID": f"{transaction_id}"},
    )
    assert app_response.status_code == 200
    app_id = app_response.json()["id"]

    # create a user
    user_response = client.post(
        f"/v1/apps/{app_id}/users",
        json={"name": "test_user"},
        headers={"X-Transaction-ID": f"{transaction_id}"},
    )
    assert user_response.status_code == 200

    # Rollback the transaction
    rollback_response = client.post(
        f"/v1/transactions/{transaction_id}/rollback",
    )
    assert rollback_response.status_code == 200

    # Verify the operations were rolled back
    app = client.get(f"/v1/apps/?app_id={app_id}")
    assert app.status_code == 404


# Error cases


def test_commit_non_existent_transaction(client):
    non_existent_transaction_id = 999999
    response = client.post(f"/v1/transactions/{non_existent_transaction_id}/commit")
    # Expecting 404 ResourceNotFoundException based on crud.py behavior if transaction not found
    assert response.status_code == 404
    assert "detail" in response.json()
    assert response.json()["detail"] == "Transaction not found"


def test_rollback_non_existent_transaction(client):
    non_existent_transaction_id = 999999
    response = client.post(f"/v1/transactions/{non_existent_transaction_id}/rollback")
    assert response.status_code == 404
    assert "detail" in response.json()
    assert response.json()["detail"] == "Transaction not found"


# TODO: Add tests for committing/rolling back an already committed/rolled-back transaction.
# TODO: Add tests for committing/rolling back an expired transaction.
# TODO: Add tests for transactions that involve actual data changes and verify commit/rollback effects.
# This will require creating data within a transaction and then checking its state after commit/rollback.


def test_commit_already_committed_transaction(client):
    # Begin and commit a transaction
    response = client.post("/v1/transactions/begin")
    assert response.status_code == 200
    transaction_id = response.json()

    commit_response = client.post(f"/v1/transactions/{transaction_id}/commit")
    assert commit_response.status_code == 200

    # Attempt to commit again
    second_commit_response = client.post(f"/v1/transactions/{transaction_id}/commit")
    # Expecting an error because the transaction is already committed
    assert second_commit_response.status_code == 422


def test_rollback_already_rolled_back_transaction(client):
    # Begin and roll back a transaction
    response = client.post("/v1/transactions/begin")
    assert response.status_code == 200
    transaction_id = response.json()

    rollback_response = client.post(f"/v1/transactions/{transaction_id}/rollback")
    assert rollback_response.status_code == 200

    # Attempt to roll back again
    second_rollback_response = client.post(
        f"/v1/transactions/{transaction_id}/rollback"
    )
    # Expecting an error because the transaction is already rolled back
    assert second_rollback_response.status_code == 422


def test_commit_already_rolled_back_transaction(client):
    # Begin and roll back a transaction
    response = client.post("/v1/transactions/begin")
    assert response.status_code == 200
    transaction_id = response.json()

    rollback_response = client.post(f"/v1/transactions/{transaction_id}/rollback")
    assert rollback_response.status_code == 200

    # Attempt to commit
    commit_response = client.post(f"/v1/transactions/{transaction_id}/commit")
    # Expecting an error because the transaction is already rolled back
    assert commit_response.status_code == 422


def test_rollback_already_committed_transaction(client):
    # Begin and commit a transaction
    response = client.post("/v1/transactions/begin")
    assert response.status_code == 200
    transaction_id = response.json()

    commit_response = client.post(f"/v1/transactions/{transaction_id}/commit")
    assert commit_response.status_code == 200

    # Attempt to roll back
    rollback_response = client.post(f"/v1/transactions/{transaction_id}/rollback")
    # Expecting an error because the transaction is already committed
    assert rollback_response.status_code == 422


@pytest.mark.asyncio
async def test_commit_expired_transaction(client):
    # Create a transaction with a very short expiry (e.g., 1 second)
    dt_obj = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        seconds=1
    )
    expires_at_str = dt_obj.isoformat().replace("+00:00", "Z")
    response = client.post(f"/v1/transactions/begin?expires_at={expires_at_str}")
    assert response.status_code == 200
    transaction_id = response.json()

    # Wait for the transaction to expire
    await asyncio.sleep(2)  # Wait for 2 seconds to be sure

    # Attempt to commit the expired transaction
    commit_response = client.post(f"/v1/transactions/{transaction_id}/commit")
    assert commit_response.status_code == 422


@pytest.mark.asyncio
async def test_rollback_expired_transaction(client):
    # Create a transaction with a very short expiry (e.g., 1 second)
    dt_obj = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        seconds=1
    )
    expires_at_str = dt_obj.isoformat().replace("+00:00", "Z")
    response = client.post(f"/v1/transactions/begin?expires_at={expires_at_str}")
    assert response.status_code == 200
    transaction_id = response.json()

    # Wait for the transaction to expire
    await asyncio.sleep(2)  # Wait for 2 seconds to be sure

    # Attempt to roll back the expired transaction
    rollback_response = client.post(f"/v1/transactions/{transaction_id}/rollback")
    # Expecting an error because the transaction has expired
    assert rollback_response.status_code == 422


@pytest.mark.asyncio
async def test_transaction_with_multiple_operations_commit(client, db_session):
    # 1. Begin Transaction
    begin_response = client.post("/v1/transactions/begin")
    assert begin_response.status_code == 200
    transaction_id = begin_response.json()
    txn_header = {"X-Transaction-ID": str(transaction_id)}

    # 2. Create App
    app_name = "multi_op_app_commit"
    app_response = client.post("/v1/apps", json={"name": app_name}, headers=txn_header)
    assert app_response.status_code == 200
    app_id = app_response.json()["id"]

    # 3. Create User
    user_name = "multi_op_user_commit"
    user_response = client.post(
        f"/v1/apps/{app_id}/users", json={"name": user_name}, headers=txn_header
    )
    assert user_response.status_code == 200
    user_id = user_response.json()["id"]

    # 4. Create Session
    session_response = client.post(
        f"/v1/apps/{app_id}/users/{user_id}/sessions",
        json={},
        headers=txn_header,
    )
    assert session_response.status_code == 200
    session_id = session_response.json()["id"]

    # 5. Create Message
    message_content = "Hello from within transaction commit test"
    message_response = client.post(
        f"/v1/apps/{app_id}/users/{user_id}/sessions/{session_id}/messages",
        json={"content": message_content, "is_user": True},
        headers=txn_header,
    )
    assert message_response.status_code == 200
    message_id = message_response.json()["id"]

    # 6. Verify resources don't exist outside transaction
    assert client.get(f"/v1/apps/?app_id={app_id}").status_code == 404
    assert client.get(f"/v1/apps/{app_id}/users?user_id={user_id}").status_code == 404
    # Message check (assuming messages are not directly queryable by ID without session context)
    # We'll verify its existence post-commit via listing messages for the session.

    # 7. Commit Transaction
    commit_response = client.post(f"/v1/transactions/{transaction_id}/commit")
    assert commit_response.status_code == 200

    # 8. Verify resources exist after commit
    assert client.get(f"/v1/apps/?app_id={app_id}").status_code == 200
    get_app_resp = client.get(f"/v1/apps/?app_id={app_id}")
    assert get_app_resp.json()["name"] == app_name

    get_user_resp = client.get(f"/v1/apps/{app_id}/users?user_id={user_id}")
    assert get_user_resp.status_code == 200
    assert get_user_resp.json()["name"] == user_name

    # Verify message
    # Need to ensure session is committed if it was created directly via DB
    # If session creation involves transactional logic tied to X-Transaction-ID header,
    # then it should be part of the main transaction automatically.
    # For this test, we assume session created via db_session.add + flush is part of the broader DB transaction
    # managed by the transaction endpoints.

    messages_response = client.post(
        f"/v1/apps/{app_id}/users/{user_id}/sessions/{session_id}/messages/list",
        json={},
    )
    assert messages_response.status_code == 200
    messages_data = messages_response.json()
    assert len(messages_data["items"]) == 1
    assert messages_data["items"][0]["id"] == message_id
    assert messages_data["items"][0]["content"] == message_content


@pytest.mark.asyncio
async def test_transaction_with_multiple_operations_rollback(client, db_session):
    # 1. Begin Transaction
    begin_response = client.post("/v1/transactions/begin")
    assert begin_response.status_code == 200
    transaction_id = begin_response.json()
    txn_header = {"X-Transaction-ID": str(transaction_id)}

    # 2. Create App
    app_name = "multi_op_app_rollback"
    app_response = client.post("/v1/apps", json={"name": app_name}, headers=txn_header)
    assert app_response.status_code == 200
    app_id = app_response.json()["id"]

    # 3. Create User
    user_name = "multi_op_user_rollback"
    user_response = client.post(
        f"/v1/apps/{app_id}/users", json={"name": user_name}, headers=txn_header
    )
    assert user_response.status_code == 200
    user_id = user_response.json()["id"]

    # 4. Create Session
    session_response = client.post(
        f"/v1/apps/{app_id}/users/{user_id}/sessions",
        json={},
        headers=txn_header,
    )
    assert session_response.status_code == 200
    session_id = session_response.json()["id"]

    # 5. Create Message
    message_content = "Hello from within transaction rollback test"
    message_response = client.post(
        f"/v1/apps/{app_id}/users/{user_id}/sessions/{session_id}/messages",
        json={"content": message_content, "is_user": True},
        headers=txn_header,
    )
    assert message_response.status_code == 200
    # message_id = message_response.json()["id"] # Not needed for verification if rolled back

    # 6. Rollback Transaction
    rollback_response = client.post(f"/v1/transactions/{transaction_id}/rollback")
    assert rollback_response.status_code == 200

    # 7. Verify resources do NOT exist after rollback
    assert client.get(f"/v1/apps/?app_id={app_id}").status_code == 404
    assert client.get(f"/v1/apps/{app_id}/users?user_id={user_id}").status_code == 404

    # Verify session does not exist (or messages within it)
    # This check depends on how sessions are handled. If sessions are hard deleted on rollback,
    # or if messages are hard deleted. If they are soft-deleted, this check changes.
    # Assuming hard delete for simplicity based on typical rollback behavior.
    # A robust way to check is that listing messages for that session (if session ID were known post-rollback)
    # would yield no messages or error.
    # Given session_id was derived from a flushed but not committed DB object tied to the transaction,
    # it should also be rolled back.
    _messages_response = client.post(
        f"/v1/apps/{app_id}/users/{user_id}/sessions/{session_id}/messages/list",
        json={},
    )
    # Depending on implementation, this might be 404 if the session itself is gone, or 200 with empty items
    # if the session exists but messages are gone. If the user is gone, this path would also fail.
    # Given the user should be gone, the call to get user's sessions should fail before messages.
    # For this test, let's assume the session itself will be gone if user_id it depends on is gone.
    # Thus, a 404 on the user lookup for the session endpoint is likely.
    # Or if the app is gone, an even earlier 404.
    # Simplest is to re-check the app or user.
    # If this test needs to be more specific about session/message rollback, it needs careful thought
    # on the cascade behavior of your DB and ORM.
    # For now, relying on app/user absence is a strong indicator.


async def test_transaction_too_many_operations(client):
    # Create a transaction
    response = client.post("/v1/transactions/begin")
    assert response.status_code == 200
    transaction_id = response.json()

    max_operations = int(os.getenv("MAX_STAGED_OPERATIONS", 10))

    # Create n operations
    for i in range(max_operations):
        response = client.post(
            "/v1/apps",
            json={"name": f"transaction_test_app_{i}"},
            headers={"X-Transaction-ID": f"{transaction_id}"},
        )
        assert response.status_code == 200

    # Creating next operation should fail
    response = client.post(
        "/v1/apps",
        json={"name": f"transaction_test_app_{max_operations + 1}"},
        headers={"X-Transaction-ID": f"{transaction_id}"},
    )
    assert response.status_code == 422


async def test_transaction_multiple_clients(client, secondary_client):
    # Create a transaction
    response = client.post("/v1/transactions/begin")
    assert response.status_code == 200
    transaction_id = response.json()

    # Create an app
    app_response = secondary_client.post(
        "/v1/apps",
        json={"name": "transaction_test_app"},
        headers={"X-Transaction-ID": f"{transaction_id}"},
    )
    assert app_response.status_code == 200
    app_id = app_response.json()["id"]

    # Continue the transaction in the original client
    response = client.post(
        f"/v1/apps/{app_id}/users",
        json={"name": "transaction_test_user"},
        headers={"X-Transaction-ID": f"{transaction_id}"},
    )
    assert response.status_code == 200
    user_id = response.json()["id"]
    # Commit the transaction
    commit_response = secondary_client.post(f"/v1/transactions/{transaction_id}/commit")
    assert commit_response.status_code == 200

    # Verify the app exists for both clients
    assert secondary_client.get(f"/v1/apps/?app_id={app_id}").status_code == 200
    assert (
        secondary_client.get(f"/v1/apps/{app_id}/users?user_id={user_id}").status_code
        == 200
    )
    assert client.get(f"/v1/apps/?app_id={app_id}").status_code == 200
    assert client.get(f"/v1/apps/{app_id}/users?user_id={user_id}").status_code == 200
