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

    # Wait for deriver to process all messages (only necessary if very recent messages are critical to query)
    client.poll_deriver_status()

    # Query conversation context
    response = alice.chat("What did Bob say to me?")
"""

from .async_client import (
    AsyncHoncho,
    AsyncPage,
    AsyncPeer,
    AsyncSession,
)
from .base import PeerBase, SessionBase
from .client import Honcho
from .observations import AsyncObservationScope, Observation, ObservationScope
from .pagination import SyncPage
from .peer import Peer
from .session import Session
from .session_context import SessionContext, SessionSummaries, Summary
from .types import (
    DeductiveObservation,
    DialecticStreamResponse,
    ExplicitObservation,
    PeerContext,
    Representation,
)

__version__ = "1.6.0"
__author__ = "Plastic Labs"
__email__ = "hello@plasticlabs.ai"

__all__ = [
    "AsyncHoncho",
    "AsyncObservationScope",
    "AsyncPeer",
    "AsyncSession",
    "AsyncPage",
    "Honcho",
    "Observation",
    "ObservationScope",
    "Peer",
    "PeerBase",
    "PeerContext",
    "Session",
    "SessionBase",
    "SessionContext",
    "SessionSummaries",
    "Summary",
    "SyncPage",
    "DialecticStreamResponse",
    "Representation",
    "ExplicitObservation",
    "DeductiveObservation",
]
