from collections.abc import AsyncIterator, Iterator
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.peer import Peer
from sdks.python.src.honcho.session import Session
from sdks.python.src.honcho.types import (
    AsyncDialecticStreamResponse,
    DialecticStreamResponse,
)


@pytest.mark.asyncio
async def test_peer_metadata(client_fixture: tuple[Honcho, str]):
    """
    Tests creation and metadata operations for peers.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-peer-meta")
        assert isinstance(peer, Peer)

        metadata = await peer.aio.get_metadata()
        assert metadata == {}

        await peer.aio.set_metadata({"foo": "bar"})
        metadata = await peer.aio.get_metadata()
        assert metadata == {"foo": "bar"}
    else:
        peer = honcho_client.peer(id="test-peer-meta")
        assert isinstance(peer, Peer)

        metadata = peer.get_metadata()
        assert metadata == {}

        peer.set_metadata({"foo": "bar"})
        metadata = peer.get_metadata()
        assert metadata == {"foo": "bar"}


@pytest.mark.asyncio
async def test_peer_sessions(client_fixture: tuple[Honcho, str]):
    """
    Tests retrieving the sessions a peer is a member of.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-peer-sessions")
        session1 = await honcho_client.aio.session(id="s1")
        session2 = await honcho_client.aio.session(id="s2")

        await session1.aio.add_peers(peer)
        await session2.aio.add_peers(peer)

        sessions_page = await peer.aio.sessions()
        sessions = sessions_page.items
        assert len(sessions) == 2
        session_ids = {s.id for s in sessions}
        assert "s1" in session_ids
        assert "s2" in session_ids
    else:
        peer = honcho_client.peer(id="test-peer-sessions")
        session1 = honcho_client.session(id="s1")
        session2 = honcho_client.session(id="s2")

        session1.add_peers(peer)
        session2.add_peers(peer)

        sessions_page = peer.sessions()
        sessions = list(sessions_page)
        assert len(sessions) == 2
        session_ids = {s.id for s in sessions}
        assert "s1" in session_ids
        assert "s2" in session_ids


@pytest.mark.asyncio
async def test_peer_card_global(client_fixture: tuple[Honcho, str]):
    """
    Tests getting a global peer card (no target).
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-card-global-peer")
        session = await honcho_client.aio.session(id="test-card-global-session")

        # Add some messages to create context
        await session.aio.add_messages([peer.message("I like pizza")])

        # Get global peer card
        card_response = await peer.aio.get_card()
        assert card_response is None or isinstance(card_response, list)
    else:
        peer = honcho_client.peer(id="test-card-global-peer")
        session = honcho_client.session(id="test-card-global-session")

        # Add some messages to create context
        session.add_messages([peer.message("I like pizza")])

        # Get global peer card
        card_response = peer.get_card()
        assert card_response is None or isinstance(card_response, list)


@pytest.mark.asyncio
async def test_peer_card_local(client_fixture: tuple[Honcho, str]):
    """
    Tests getting a local peer card (with target).
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        observer = await honcho_client.aio.peer(id="test-card-local-observer")
        target = await honcho_client.aio.peer(id="test-card-local-target")
        session = await honcho_client.aio.session(id="test-card-local-session")

        # Add messages from both peers
        await session.aio.add_messages(
            [observer.message("Hello"), target.message("Hi there")]
        )

        # Get local peer card with target as Peer object
        card_response = await observer.aio.get_card(target=target)
        assert card_response is None or isinstance(card_response, list)

        # Get local peer card with target as string
        card_response = await observer.aio.get_card(target=target.id)
        assert card_response is None or isinstance(card_response, list)
    else:
        observer = honcho_client.peer(id="test-card-local-observer")
        target = honcho_client.peer(id="test-card-local-target")
        session = honcho_client.session(id="test-card-local-session")

        # Add messages from both peers
        session.add_messages([observer.message("Hello"), target.message("Hi there")])

        # Get local peer card with target as Peer object
        card_response = observer.get_card(target=target)
        assert card_response is None or isinstance(card_response, list)

        # Get local peer card with target as string
        card_response = observer.get_card(target=target.id)
        assert card_response is None or isinstance(card_response, list)


@pytest.mark.asyncio
async def test_peer_card_with_empty_target(client_fixture: tuple[Honcho, str]):
    """
    Tests peer card with empty target string.

    Empty strings are passed through to the API - validation is handled server-side.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-card-validation-peer")
        session = await honcho_client.aio.session(id="test-card-validation-session")

        # Create the peer on the server by adding a message
        await session.aio.add_messages([peer.message("hello")])

        # Empty target is treated as no target (same as None)
        result = await peer.aio.get_card(target="")
        assert result is None or isinstance(result, list)
    else:
        peer = honcho_client.peer(id="test-card-validation-peer")
        session = honcho_client.session(id="test-card-validation-session")

        # Create the peer on the server by adding a message
        session.add_messages([peer.message("hello")])

        # Empty target is treated as no target (same as None)
        result = peer.get_card(target="")
        assert result is None or isinstance(result, list)


@pytest.mark.asyncio
async def test_peer_chat_streaming(client_fixture: tuple[Honcho, str]):
    """
    Tests streaming chat with mocked response generator.
    """

    honcho_client, client_type = client_fixture
    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-stream-async-peer")
        session = await honcho_client.aio.session(id="test-stream-async-session")

        # Add some messages to create context
        await session.aio.add_messages([peer.message("I like pizza")])

        # Mock the async streaming response - mock _http.stream to return chunks
        async def mock_astream(*args: object, **kwargs: object) -> AsyncIterator[bytes]:  # pyright: ignore[reportUnusedParameter]
            yield b'data: {"delta": {"content": "Hello"}}\n'
            yield b'data: {"delta": {"content": " async"}}\n'
            yield b'data: {"done": true}\n'

        # Mock the _http.stream method on the honcho's async http client
        with patch.object(
            peer._honcho._async_http_client,  # pyright: ignore[reportPrivateUsage]
            "stream",
            side_effect=mock_astream,
        ):
            result = await peer.aio.chat_stream("Tell me something")
            assert isinstance(result, AsyncDialecticStreamResponse)

            # Collect chunks
            chunks: list[str] = []
            async for chunk in result:
                assert isinstance(chunk, str)
                chunks.append(chunk)
            assert chunks == ["Hello", " async"]
            assert result.get_final_response()["content"] == "Hello async"
    else:
        peer = honcho_client.peer(id="test-stream-peer")
        session = honcho_client.session(id="test-stream-session")

        # Add some messages to create context
        session.add_messages([peer.message("I like pizza")])

        # Mock the streaming response - mock _http.stream to return chunks
        def mock_stream(*args: object, **kwargs: object) -> Iterator[bytes]:  # pyright: ignore[reportUnusedParameter]
            yield b'data: {"delta": {"content": "Hello"}}\n'
            yield b'data: {"delta": {"content": " world"}}\n'
            yield b'data: {"done": true}\n'

        # Mock the _http.stream method on the peer's internal http client
        with patch.object(
            peer._honcho._http,  # pyright: ignore[reportPrivateUsage]
            "stream",
            side_effect=mock_stream,
        ):
            result = peer.chat_stream("Tell me something")
            assert isinstance(result, DialecticStreamResponse)

            # Collect chunks
            chunks = list(result)
            assert chunks == ["Hello", " world"]
            assert result.get_final_response()["content"] == "Hello world"


@pytest.mark.asyncio
async def test_peer_chat_non_streaming(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests non-streaming chat (already using core SDK).
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-non-stream-peer")
        session = await honcho_client.aio.session(id="test-non-stream-session")

        # Add some messages
        await session.aio.add_messages([peer.message("I like pizza")])

        # Non-streaming chat
        response = await peer.aio.chat("What do I like?")
        # Response can be None or a string
        assert response is None or isinstance(response, str)
    else:
        peer = honcho_client.peer(id="test-non-stream-peer")
        session = honcho_client.session(id="test-non-stream-session")

        # Add some messages
        session.add_messages([peer.message("I like pizza")])

        # Non-streaming chat
        response = peer.chat("What do I like?")
        # Response can be None or a string
        assert response is None or isinstance(response, str)


@pytest.mark.asyncio
async def test_peer_representation_no_params(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests peer.representation() with no parameters (default behavior).
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-working-rep-no-params")
        session = await honcho_client.aio.session(
            id="test-working-rep-session-no-params"
        )

        # Add some messages to create context
        await session.aio.add_messages([peer.message("I enjoy hiking and nature")])

        # Get working representation with no parameters
        result = await peer.aio.representation()
        assert isinstance(result, str)
    else:
        peer = honcho_client.peer(id="test-working-rep-no-params")
        session = honcho_client.session(id="test-working-rep-session-no-params")

        # Add some messages to create context
        session.add_messages([peer.message("I enjoy hiking and nature")])

        # Get working representation with no parameters
        result = peer.representation()
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_peer_representation_with_session_string(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests peer.representation() with session parameter as string.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-working-rep-session-str")
        session = await honcho_client.aio.session(
            id="test-working-rep-session-str-sess"
        )

        # Add some messages to the session
        await session.aio.add_messages([peer.message("I like reading books")])

        # Get working representation scoped to session (as string)
        result = await peer.aio.representation(session=session.id)
        assert isinstance(result, str)
    else:
        peer = honcho_client.peer(id="test-working-rep-session-str")
        session = honcho_client.session(id="test-working-rep-session-str-sess")

        # Add some messages to the session
        session.add_messages([peer.message("I like reading books")])

        # Get working representation scoped to session (as string)
        result = peer.representation(session=session.id)
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_peer_representation_with_session_object(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests peer.representation() with session parameter as Session object.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-working-rep-session-obj")
        session = await honcho_client.aio.session(
            id="test-working-rep-session-obj-sess"
        )
        assert isinstance(session, Session)

        # Add some messages to the session
        await session.aio.add_messages([peer.message("I prefer tea over coffee")])

        # Get working representation scoped to session (as Session object)
        result = await peer.aio.representation(session=session)
        assert isinstance(result, str)
    else:
        peer = honcho_client.peer(id="test-working-rep-session-obj")
        session = honcho_client.session(id="test-working-rep-session-obj-sess")
        assert isinstance(session, Session)

        # Add some messages to the session
        session.add_messages([peer.message("I prefer tea over coffee")])

        # Get working representation scoped to session (as Session object)
        result = peer.representation(session=session)
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_peer_representation_with_target_string(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests peer.representation() with target parameter as string.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        observer = await honcho_client.aio.peer(
            id="test-working-rep-target-str-observer"
        )
        target = await honcho_client.aio.peer(id="test-working-rep-target-str-target")
        session = await honcho_client.aio.session(id="test-working-rep-target-str-sess")

        # Add messages from both peers
        await session.aio.add_messages(
            [
                observer.message("Hello there"),
                target.message("Hi, how are you?"),
            ]
        )

        # Get working representation of target from observer's perspective (as string)
        result = await observer.aio.representation(target=target.id)
        assert isinstance(result, str)
    else:
        observer = honcho_client.peer(id="test-working-rep-target-str-observer")
        target = honcho_client.peer(id="test-working-rep-target-str-target")
        session = honcho_client.session(id="test-working-rep-target-str-sess")

        # Add messages from both peers
        session.add_messages(
            [
                observer.message("Hello there"),
                target.message("Hi, how are you?"),
            ]
        )

        # Get working representation of target from observer's perspective (as string)
        result = observer.representation(target=target.id)
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_peer_representation_with_target_object(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests peer.representation() with target parameter as Peer object.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        observer = await honcho_client.aio.peer(
            id="test-working-rep-target-obj-observer"
        )
        target = await honcho_client.aio.peer(id="test-working-rep-target-obj-target")
        session = await honcho_client.aio.session(id="test-working-rep-target-obj-sess")
        assert isinstance(target, Peer)

        # Add messages from both peers
        await session.aio.add_messages(
            [
                observer.message("What do you think?"),
                target.message("I think it's great!"),
            ]
        )

        # Get working representation of target from observer's perspective (as Peer object)
        result = await observer.aio.representation(target=target)
        assert isinstance(result, str)
    else:
        observer = honcho_client.peer(id="test-working-rep-target-obj-observer")
        target = honcho_client.peer(id="test-working-rep-target-obj-target")
        session = honcho_client.session(id="test-working-rep-target-obj-sess")
        assert isinstance(target, Peer)

        # Add messages from both peers
        session.add_messages(
            [
                observer.message("What do you think?"),
                target.message("I think it's great!"),
            ]
        )

        # Get working representation of target from observer's perspective (as Peer object)
        result = observer.representation(target=target)
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_peer_representation_with_search_query(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests peer.representation() with search_query parameter.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-working-rep-search-query")
        session = await honcho_client.aio.session(
            id="test-working-rep-search-query-sess"
        )

        # Add some messages with different topics
        await session.aio.add_messages(
            [
                peer.message("I love programming in Python"),
                peer.message("I also enjoy playing basketball"),
            ]
        )

        # Get working representation with search query
        result = await peer.aio.representation(search_query="programming")
        assert isinstance(result, str)
    else:
        peer = honcho_client.peer(id="test-working-rep-search-query")
        session = honcho_client.session(id="test-working-rep-search-query-sess")

        # Add some messages with different topics
        session.add_messages(
            [
                peer.message("I love programming in Python"),
                peer.message("I also enjoy playing basketball"),
            ]
        )

        # Get working representation with search query
        result = peer.representation(search_query="programming")
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_peer_representation_with_size(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests peer.representation() with size parameter.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-working-rep-size")
        session = await honcho_client.aio.session(id="test-working-rep-size-sess")

        # Add multiple messages
        await session.aio.add_messages(
            [peer.message(f"Message number {i}") for i in range(10)]
        )

        # Get working representation with custom max_conclusions
        result = await peer.aio.representation(max_conclusions=5)
        assert isinstance(result, str)

        # Test with different max_conclusions values
        result = await peer.aio.representation(max_conclusions=1)
        assert isinstance(result, str)

        result = await peer.aio.representation(max_conclusions=100)
        assert isinstance(result, str)
    else:
        peer = honcho_client.peer(id="test-working-rep-size")
        session = honcho_client.session(id="test-working-rep-size-sess")

        # Add multiple messages
        session.add_messages([peer.message(f"Message number {i}") for i in range(10)])

        # Get working representation with custom size
        result = peer.representation(max_conclusions=5)
        assert isinstance(result, str)

        # Test with different max_conclusions values
        result = peer.representation(max_conclusions=1)
        assert isinstance(result, str)

        result = peer.representation(max_conclusions=100)
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_peer_representation_with_all_params(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests peer.representation() with all parameters combined.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        observer = await honcho_client.aio.peer(id="test-working-rep-all-observer")
        target = await honcho_client.aio.peer(id="test-working-rep-all-target")
        session = await honcho_client.aio.session(id="test-working-rep-all-sess")

        # Add messages from both peers
        await session.aio.add_messages(
            [
                observer.message("I think Python is great for data science"),
                target.message("I agree, especially with libraries like pandas"),
                observer.message("What about machine learning?"),
                target.message("TensorFlow and PyTorch are excellent choices"),
            ]
        )

        # Get working representation with all parameters
        result = await observer.aio.representation(
            session=session, target=target, search_query="Python", max_conclusions=10
        )
        assert isinstance(result, str)

        # Test with session as string and target as string
        result = await observer.aio.representation(
            session=session.id,
            target=target.id,
            search_query="machine learning",
            max_conclusions=5,
        )
        assert isinstance(result, str)
    else:
        observer = honcho_client.peer(id="test-working-rep-all-observer")
        target = honcho_client.peer(id="test-working-rep-all-target")
        session = honcho_client.session(id="test-working-rep-all-sess")

        # Add messages from both peers
        session.add_messages(
            [
                observer.message(content="I think Python is great for data science"),
                target.message("I agree, especially with libraries like pandas"),
                observer.message("What about machine learning?"),
                target.message("TensorFlow and PyTorch are excellent choices"),
            ]
        )

        # Get working representation with all parameters
        result = observer.representation(
            session=session, target=target, search_query="Python", max_conclusions=10
        )
        assert isinstance(result, str)

        # Test with session as string and target as string
        result = observer.representation(
            session=session.id,
            target=target.id,
            search_query="machine learning",
            max_conclusions=5,
        )
        assert isinstance(result, str)


class ChatFoodPreferences(BaseModel):
    favorite: str
    confidence: float


CHAT_SCHEMA_DICT = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {"type": "string"}}},
    "required": ["items"],
}


@pytest.mark.asyncio
async def test_peer_chat_response_format_pydantic(client_fixture: tuple[Honcho, str]):
    """A Pydantic model class is sent as JSON Schema and the response content
    is parsed back into a model instance."""
    honcho_client, client_type = client_fixture
    content = '{"favorite": "sushi", "confidence": 0.9}'

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-rf-async-peer")

        async def mock_post(*_args: object, **_kwargs: object) -> dict[str, object]:
            return {"content": content}

        with patch.object(
            peer._honcho._async_http_client,  # pyright: ignore[reportPrivateUsage]
            "post",
            side_effect=mock_post,
        ) as mock:
            result = await peer.aio.chat(
                "What do I like?", response_format=ChatFoodPreferences
            )
    else:
        peer = honcho_client.peer(id="test-rf-peer")
        with patch.object(
            peer._honcho._http,  # pyright: ignore[reportPrivateUsage]
            "post",
            return_value={"content": content},
        ) as mock:
            result = peer.chat("What do I like?", response_format=ChatFoodPreferences)

    body = mock.call_args.kwargs["body"]
    assert body["response_format"] == ChatFoodPreferences.model_json_schema()
    assert isinstance(result, ChatFoodPreferences)
    assert result.favorite == "sushi"


@pytest.mark.asyncio
async def test_peer_chat_response_format_dict(client_fixture: tuple[Honcho, str]):
    """A raw JSON Schema dict is sent as-is and the response stays a string."""
    honcho_client, client_type = client_fixture
    content = '{"items": ["sushi"]}'

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-rf-dict-async-peer")

        async def mock_post(*_args: object, **_kwargs: object) -> dict[str, object]:
            return {"content": content}

        with patch.object(
            peer._honcho._async_http_client,  # pyright: ignore[reportPrivateUsage]
            "post",
            side_effect=mock_post,
        ) as mock:
            result = await peer.aio.chat(
                "What do I like?", response_format=CHAT_SCHEMA_DICT
            )
    else:
        peer = honcho_client.peer(id="test-rf-dict-peer")
        with patch.object(
            peer._honcho._http,  # pyright: ignore[reportPrivateUsage]
            "post",
            return_value={"content": content},
        ) as mock:
            result = peer.chat("What do I like?", response_format=CHAT_SCHEMA_DICT)

    body = mock.call_args.kwargs["body"]
    assert body["response_format"] == CHAT_SCHEMA_DICT
    assert result == content


@pytest.mark.asyncio
async def test_peer_chat_response_format_empty_content(
    client_fixture: tuple[Honcho, str],
):
    """Empty/None content returns None even when a Pydantic class was given."""
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-rf-empty-async-peer")

        async def mock_post(*_args: object, **_kwargs: object) -> dict[str, object]:
            return {"content": None}

        with patch.object(
            peer._honcho._async_http_client,  # pyright: ignore[reportPrivateUsage]
            "post",
            side_effect=mock_post,
        ):
            result = await peer.aio.chat(
                "What do I like?", response_format=ChatFoodPreferences
            )
    else:
        peer = honcho_client.peer(id="test-rf-empty-peer")
        with patch.object(
            peer._honcho._http,  # pyright: ignore[reportPrivateUsage]
            "post",
            return_value={"content": None},
        ):
            result = peer.chat("What do I like?", response_format=ChatFoodPreferences)

    assert result is None


@pytest.mark.asyncio
async def test_peer_chat_stream_response_format(client_fixture: tuple[Honcho, str]):
    """chat_stream sends the schema in the body; chunks stay raw text."""
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-rf-stream-async-peer")

        async def mock_astream(
            *_args: object, **_kwargs: object
        ) -> AsyncIterator[bytes]:
            yield b'data: {"delta": {"content": "{\\"favorite\\":"}}\n'
            yield b'data: {"delta": {"content": "\\"sushi\\",\\"confidence\\":0.9}"}}\n'
            yield b'data: {"done": true}\n'

        with patch.object(
            peer._honcho._async_http_client,  # pyright: ignore[reportPrivateUsage]
            "stream",
            side_effect=mock_astream,
        ) as mock:
            result = await peer.aio.chat_stream(
                "What do I like?", response_format=ChatFoodPreferences
            )
            chunks = [chunk async for chunk in result]
    else:
        peer = honcho_client.peer(id="test-rf-stream-peer")

        def mock_stream(*_args: object, **_kwargs: object) -> Iterator[bytes]:
            yield b'data: {"delta": {"content": "{\\"favorite\\":"}}\n'
            yield b'data: {"delta": {"content": "\\"sushi\\",\\"confidence\\":0.9}"}}\n'
            yield b'data: {"done": true}\n'

        with patch.object(
            peer._honcho._http,  # pyright: ignore[reportPrivateUsage]
            "stream",
            side_effect=mock_stream,
        ) as mock:
            result = peer.chat_stream(
                "What do I like?", response_format=ChatFoodPreferences
            )
            chunks = list(result)

    body = mock.call_args.kwargs["body"]
    assert body["response_format"] == ChatFoodPreferences.model_json_schema()
    accumulated = "".join(chunks)
    assert ChatFoodPreferences.model_validate_json(accumulated).favorite == "sushi"
