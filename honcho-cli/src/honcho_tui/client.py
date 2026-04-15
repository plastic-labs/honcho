"""Thin Honcho client builder for the TUI, reusing CLI config."""

from __future__ import annotations

from honcho import Honcho
from honcho_cli.config import CLIConfig, get_client_kwargs


def build_client(workspace_id: str | None = None, peer_id: str | None = None) -> tuple[Honcho, CLIConfig]:
    """Build a Honcho client from the shared CLI config.

    Applies optional runtime overrides without modifying the config file.
    """
    config = CLIConfig.load()
    if workspace_id:
        config.workspace_id = workspace_id
    if peer_id:
        config.peer_id = peer_id
    return Honcho(**get_client_kwargs(config)), config


def collect_page(page) -> list:
    """Collect all items from a SyncPage, walking pagination."""
    try:
        items: list = list(page._raw_items)
        while page.has_next_page():
            page = page.get_next_page()
            if page is None:
                break
            items.extend(page._raw_items)
    except AttributeError:
        try:
            items = list(page.items)
        except AttributeError:
            items = list(page)
    return items
