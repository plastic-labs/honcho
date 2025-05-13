from nanoid import generate as generate_nanoid


def test_app_validations_api(client):
    # Test name too short
    response = client.post("/v1/apps", json={"name": "", "metadata": {}})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "name"]
    assert error["msg"] == "String should have at least 1 character"
    assert error["type"] == "string_too_short"

    # Test name too long
    response = client.post("/v1/apps", json={"name": "a" * 101, "metadata": {}})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "name"]
    assert error["msg"] == "String should have at most 100 characters"
    assert error["type"] == "string_too_long"

    # Test invalid metadata type
    response = client.post("/v1/apps", json={"name": "test", "metadata": "not a dict"})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "metadata"]
    assert error["type"] == "dict_type"


def test_user_validations_api(client, sample_data):
    test_app, _ = sample_data

    # Test name too short
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users", json={"name": "", "metadata": {}}
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "name"]
    assert error["msg"] == "String should have at least 1 character"
    assert error["type"] == "string_too_short"

    # Test name too long
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users", json={"name": "a" * 101, "metadata": {}}
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "name"]
    assert error["msg"] == "String should have at most 100 characters"
    assert error["type"] == "string_too_long"


def test_message_validations_api(client, sample_data):
    test_app, test_user = sample_data
    # Create a test session first
    session_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={"metadata": {}},
    )
    session_id = session_response.json()["id"]

    # Test content too long
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{session_id}/messages",
        json={"content": "a" * 50001, "is_user": True, "metadata": {}},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "content"]
    assert error["msg"] == "String should have at most 50000 characters"
    assert error["type"] == "string_too_long"

    # Test invalid is_user type
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{session_id}/messages",
        json={"content": "test", "is_user": "not a bool", "metadata": {}},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "is_user"]
    assert error["type"] == "bool_parsing"


def test_collection_validations_api(client, sample_data):
    test_app, test_user = sample_data

    # Test name too short
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": "", "metadata": {}},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "name"]
    assert error["msg"] == "String should have at least 1 character"
    assert error["type"] == "string_too_short"

    # Test name too long
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": "a" * 101, "metadata": {}},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "name"]
    assert error["msg"] == "String should have at most 100 characters"
    assert error["type"] == "string_too_long"

    # Test 'honcho' name restriction
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": "honcho", "metadata": {}},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "name"]
    assert error["msg"] == "Value error, Collection name cannot be 'honcho'"
    assert error["type"] == "value_error"


def test_document_validations_api(client, sample_data):
    test_app, test_user = sample_data
    # Create a collection first
    collection_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": str(generate_nanoid()), "metadata": {}},
    )
    collection_id = collection_response.json()["id"]

    # Test content too short
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection_id}/documents",
        json={"content": "", "metadata": {}},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "content"]
    assert error["msg"] == "String should have at least 1 character"
    assert error["type"] == "string_too_short"

    # Test content too long
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection_id}/documents",
        json={"content": "a" * 100001, "metadata": {}},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "content"]
    assert error["msg"] == "String should have at most 100000 characters"
    assert error["type"] == "string_too_long"


def test_document_query_validations_api(client, sample_data):
    test_app, test_user = sample_data
    # Create a collection first
    collection_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": str(generate_nanoid()), "metadata": {}},
    )
    collection_id = collection_response.json()["id"]

    # Test query too short
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection_id}/documents/query",
        json={"query": "", "top_k": 5},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "query"]
    assert error["msg"] == "String should have at least 1 character"
    assert error["type"] == "string_too_short"

    # Test query too long
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection_id}/documents/query",
        json={"query": "a" * 1001, "top_k": 5},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "query"]
    assert error["msg"] == "String should have at most 1000 characters"
    assert error["type"] == "string_too_long"

    # Test top_k too small
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection_id}/documents/query",
        json={"query": "test", "top_k": 0},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "top_k"]
    assert error["msg"] == "Input should be greater than or equal to 1"
    assert error["type"] == "greater_than_equal"

    # Test top_k too large
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection_id}/documents/query",
        json={"query": "test", "top_k": 51},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "top_k"]
    assert error["msg"] == "Input should be less than or equal to 50"
    assert error["type"] == "less_than_equal"


def test_message_batch_validations_api(client, sample_data):
    test_app, test_user = sample_data
    # Create a test session first
    session_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={"metadata": {}},
    )
    session_id = session_response.json()["id"]

    # Test batch too large
    messages = [
        {"content": f"test message {i}", "is_user": True, "metadata": {}}
        for i in range(101)  # Create 101 messages
    ]

    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{session_id}/messages/batch",
        json={"messages": messages},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "messages"]
    assert "List should have at most 100 items after validation" in error["msg"]
    assert error["type"] == "too_long"


def test_metamessage_validations_api(client, sample_data):
    test_app, test_user = sample_data
    # Create session and message first
    session_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={"metadata": {}},
    )
    session_id = session_response.json()["id"]

    message_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{session_id}/messages",
        json={"content": "test message", "is_user": True, "metadata": {}},
    )
    message_id = message_response.json()["id"]

    # Test label too short
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/metamessages",
        json={
            "label": "",
            "content": "test content",
            "session_id": session_id,
            "message_id": message_id,
            "metadata": {},
        },
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "label"]
    assert error["msg"] == "String should have at least 1 character"
    assert error["type"] == "string_too_short"

    # Test label too long
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/metamessages",
        json={
            "label": "a" * 51,
            "content": "test content",
            "session_id": session_id,
            "message_id": message_id,
            "metadata": {},
        },
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "label"]
    assert error["msg"] == "String should have at most 50 characters"
    assert error["type"] == "string_too_long"

    # Test content too long
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/metamessages",
        json={
            "label": "test_type",
            "content": "a" * 50001,
            "message_id": message_id,
            "session_id": session_id,
            "metadata": {},
        },
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "content"]
    assert error["msg"] == "String should have at most 50000 characters"
    assert error["type"] == "string_too_long"


def test_collection_update_validations_api(client, sample_data):
    test_app, test_user = sample_data
    # Create a collection first
    collection_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": str(generate_nanoid()), "metadata": {}},
    )
    collection_id = collection_response.json()["id"]

    # Test honcho name in update
    response = client.put(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection_id}",
        json={"name": "honcho", "metadata": {}},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "name"]
    assert error["msg"] == "Value error, Collection name cannot be 'honcho'"
    assert error["type"] == "value_error"


def test_document_update_validations_api(client, sample_data):
    test_app, test_user = sample_data
    # Create collection and document first
    collection_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": str(generate_nanoid()), "metadata": {}},
    )
    collection_id = collection_response.json()["id"]

    document_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection_id}/documents",
        json={"content": "test content", "metadata": {}},
    )
    document_id = document_response.json()["id"]

    # Test content too long in update
    response = client.put(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection_id}/documents/{document_id}",
        json={"content": "a" * 100001, "metadata": {}},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "content"]
    assert error["msg"] == "String should have at most 100000 characters"
    assert error["type"] == "string_too_long"


def test_session_validations_api(client, sample_data):
    test_app, test_user = sample_data
    # Create a test session first
    session_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={"metadata": {}},
    )
    session_id = session_response.json()["id"]

    # Test invalid metadata type
    response = client.put(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{session_id}",
        json={"metadata": "not a dict"},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "metadata"]
    assert error["type"] == "dict_type"

    # Test empty update
    response = client.put(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{session_id}",
        json={},
    )
    assert response.status_code == 422


def test_agent_query_validations_api(client, sample_data, monkeypatch):
    # Mock the functions in agent.py that are causing the database issues

    # Create a mock collection with a public_id
    class MockCollection:
        def __init__(self):
            self.public_id = "mock_collection_id"

    # Mock collection retrieval/creation function
    async def mock_get_or_create_collection(*args, **kwargs):
        return MockCollection()

    async def mock_chat_history(*args, **kwargs):
        return "Mock chat history", [], []

    async def mock_get_long_term_facts(*args, **kwargs):
        return ["Mock fact 1", "Mock fact 2"]

    async def mock_run_tom_inference(*args, **kwargs):
        return "Mock TOM inference"

    async def mock_generate_user_representation(*args, **kwargs):
        return "Mock user representation"

    # Mock the Dialectic.call method
    async def mock_dialectic_call(self):
        # Create a mock response that will work with line 300 in agent.py:
        # return schemas.AgentChat(content=response[0]["text"])
        return [{"text": "Mock response"}]

    # Mock the Dialectic.stream method
    def mock_dialectic_stream(self):
        class MockStream:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            @property
            def text_stream(self):
                yield "Mock streamed response"

        return MockStream()

    # Apply the monkeypatches
    monkeypatch.setattr(
        "src.crud.get_or_create_user_protected_collection",
        mock_get_or_create_collection,
    )
    monkeypatch.setattr("src.utils.history.get_summarized_history", mock_chat_history)
    monkeypatch.setattr("src.agent.get_long_term_facts", mock_get_long_term_facts)
    monkeypatch.setattr("src.agent.run_tom_inference", mock_run_tom_inference)
    monkeypatch.setattr(
        "src.agent.generate_user_representation", mock_generate_user_representation
    )
    monkeypatch.setattr("src.agent.Dialectic.call", mock_dialectic_call)
    monkeypatch.setattr("src.agent.Dialectic.stream", mock_dialectic_stream)

    test_app, test_user = sample_data
    # Create a session first since agent queries are likely session-based
    session_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={"metadata": {}},
    )
    session_id = session_response.json()["id"]

    # Test valid string query (under 10000 chars)
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{session_id}/chat",
        json={"queries": "a" * 9999},
    )
    assert response.status_code == 200

    # Test string query too long (over 10000 chars)
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{session_id}/chat",
        json={"queries": "a" * 10001},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "queries"]
    assert error["msg"] == "Value error, Query too long"
    assert error["type"] == "value_error"

    # Test valid list query (under 25 items, each under 10000 chars)
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{session_id}/chat",
        json={"queries": ["a" * 9999 for _ in range(25)]},
    )
    assert response.status_code == 200

    # Test list too long (over 25 items)
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{session_id}/chat",
        json={"queries": ["test" for _ in range(26)]},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "queries"]
    assert error["type"] == "value_error"

    # Test list item too long (item over 10000 chars)
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{session_id}/chat",
        json={"queries": ["a" * 10001]},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "queries"]
    assert error["msg"] == "Value error, One or more queries too long"
    assert error["type"] == "value_error"

    # Test that strings over 20 chars are allowed
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{session_id}/chat",
        json={"queries": "a" * 100},  # 100 chars should be fine
    )
    assert response.status_code == 200


def test_required_field_validations_api(client, sample_data):
    test_app, test_user = sample_data
    session_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={"metadata": {}},
    )
    session_id = session_response.json()["id"]

    # Test missing required content in message
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{session_id}/messages",
        json={"is_user": True, "metadata": {}},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "content"]
    assert error["type"] == "missing"

    # Test missing required is_user in message
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{session_id}/messages",
        json={"content": "test", "metadata": {}},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "is_user"]
    assert error["type"] == "missing"

    # Test missing required name in collection
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"metadata": {}},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "name"]
    assert error["type"] == "missing"


def test_filter_validations_api(client, sample_data):
    test_app, test_user = sample_data
    # Create a session first
    session_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={"metadata": {}},
    )
    session_id = session_response.json()["id"]

    # Test invalid filter type in message list (at session level)
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{session_id}/messages/list",
        json={"filter": "not a dict"},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "filter"]
    assert error["type"] == "dict_type"

    # Test invalid filter type in collection list (at user level)
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/list",
        json={"filter": "not a dict"},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "filter"]
    assert error["type"] == "dict_type"
