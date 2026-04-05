"""Peer-related utility functions."""

from __future__ import annotations

from typing import Any

from ..api_types import SessionPeerConfig
from ..base import PeerBase


def normalize_peers_to_dict(
    peers: str
    | PeerBase
    | tuple[str, SessionPeerConfig]
    | tuple[PeerBase, SessionPeerConfig]
    | list[PeerBase | str]
    | list[tuple[PeerBase | str, SessionPeerConfig]]
    | list[PeerBase | str | tuple[PeerBase | str, SessionPeerConfig]],
) -> dict[str, Any]:
    """
    Normalize various peer input formats into a dict mapping peer IDs to configs.

    Accepts:
        - str: Single peer ID
        - PeerBase: Single Peer object
        - tuple[str, SessionPeerConfig]: Single peer ID with config
        - tuple[PeerBase, SessionPeerConfig]: Single Peer object with config
        - list[PeerBase | str]: List of peers/IDs
        - list[tuple[PeerBase | str, SessionPeerConfig]]: List of peers with configs
        - Mixed lists combining all of the above

    Returns:
        Dict mapping peer IDs to their config dicts (empty dict if no config)
    """

    if not isinstance(peers, list):
        peers = [peers]

    peer_dict: dict[str, Any] = {}
    for peer in peers:
        if isinstance(peer, tuple):
            peer_id = peer[0] if isinstance(peer[0], str) else peer[0].id
            peer_config = peer[1]
            peer_dict[peer_id] = peer_config.model_dump(exclude_none=True)
        else:
            peer_id = peer if isinstance(peer, str) else peer.id
            peer_dict[peer_id] = {}

    return peer_dict
