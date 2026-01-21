"""
Honcho Python SDK

A Python client library for the Honcho conversational memory platform.
Provides tools for managing peers, sessions, and conversation context
across multi-party interactions.

Usage:
    from honcho import Honcho

    # Initialize client
    client = Honcho(api_key="your-api-key")

    # Create peers
    alice = client.peer("alice")
    bob = client.peer("bob")

    # Create a session
    session = client.session("conversation-1")

    # Add peers to session
    session.add_peers([alice, bob])

    # Add messages
    session.add_messages([
        alice.message("Hello, Bob!"),
        bob.message("Hi Alice, how are you?")
    ])

    # Query conversation context
    response = alice.chat("What did Bob say to me?", session=session)

    # Async operations via .aio accessor
    peer = await client.aio.peer("user-123")
    await peer.aio.chat("query", session=session)
    async for p in client.aio.peers():
        print(p.id)
"""

from .aio import ConclusionScopeAio, HonchoAio, PeerAio, SessionAio
from .api_types import MessageCreateParams
from .base import PeerBase, SessionBase
from .client import Honcho
from .conclusions import Conclusion, ConclusionScope
from .http.exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    ConnectionError,
    HonchoError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServerError,
    TimeoutError,
    UnprocessableEntityError,
)
from .message import Message
from .pagination import AsyncPage, SyncPage
from .peer import Peer
from .session import Session
from .session_context import SessionContext, SessionSummaries, Summary
from .types import (
    AsyncDialecticStreamResponse,
    DialecticStreamResponse,
)

__version__ = "2.0.0"
__author__ = "Plastic Labs"
__email__ = "hello@plasticlabs.ai"

__all__ = [
    # Client
    "Honcho",
    # Domain classes
    "Conclusion",
    "ConclusionScope",
    "Message",
    "MessageCreateParams",
    "Peer",
    "Session",
    # Aio views (for type hints)
    "ConclusionScopeAio",
    "HonchoAio",
    "PeerAio",
    "SessionAio",
    # Base classes
    "PeerBase",
    "SessionBase",
    # Response types
    "SessionContext",
    "SessionSummaries",
    "Summary",
    # Pagination
    "AsyncPage",
    "SyncPage",
    # Streaming
    "AsyncDialecticStreamResponse",
    "DialecticStreamResponse",
    # Exceptions
    "APIError",
    "AuthenticationError",
    "BadRequestError",
    "ConflictError",
    "ConnectionError",
    "HonchoError",
    "NotFoundError",
    "PermissionDeniedError",
    "RateLimitError",
    "ServerError",
    "TimeoutError",
    "UnprocessableEntityError",
]
