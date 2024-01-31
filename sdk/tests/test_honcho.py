from honcho import Client
from uuid import uuid1


def test_session_creation_retrieval():
    client = Client("http://localhost:8000")
    created_session = client.create_session("ayush")
    retrieved_session = client.get_session("ayush", created_session.id)
    assert retrieved_session.id == created_session.id
    assert retrieved_session.is_active == True
    assert retrieved_session.location_id == "default"
    assert retrieved_session.session_data == {}


def test_session_multiple_retrieval():
    client = Client("http://localhost:8000")
    user = str(uuid1())
    created_session_1 = client.create_session(user)
    created_session_2 = client.create_session(user)
    retrieved_sessions = client.get_sessions(user)
    assert len(retrieved_sessions) == 2
    assert retrieved_sessions[0].id == created_session_1.id
    assert retrieved_sessions[1].id == created_session_2.id


def test_session_update():
    client = Client("http://localhost:8000")
    created_session = client.create_session("ayush")
    assert created_session.update({"foo": "bar"})
    retrieved_session = client.get_session("ayush", created_session.id)
    assert retrieved_session.session_data == {"foo": "bar"}


def test_session_deletion():
    client = Client("http://localhost:8000")
    created_session = client.create_session("ayush")
    created_session.delete()
    retrieved_session = client.get_session("ayush", created_session.id)
    assert retrieved_session.is_active == False
    assert retrieved_session.id == created_session.id


def test_messages():
    client = Client("http://localhost:8000")
    created_session = client.create_session("ayush")
    created_session.create_message(is_user=True, content="Hello")
    created_session.create_message(is_user=False, content="Hi")
    retrieved_session = client.get_session("ayush", created_session.id)
    messages = retrieved_session.get_messages()
    assert len(messages) == 2
    user_message, ai_message = messages
    assert user_message.content == "Hello"
    assert user_message.is_user == True
    assert ai_message.content == "Hi"
    assert ai_message.is_user == False
