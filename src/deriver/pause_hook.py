"""Test-only barrier that holds a representation batch before its output write.

Inert unless ``HONCHO_TEST_DERIVER_PAUSE_DIR`` is set. When it is, a batch whose
input contains ``HONCHO_TEST_DERIVER_PAUSE_MATCH`` writes ``paused-<hostname>.json``
into that directory and waits until a ``release`` file appears there.
"""

import asyncio
import json
import logging
import os
import socket
from pathlib import Path

from src.models import Message

logger = logging.getLogger(__name__)

PAUSE_DIR_ENV = "HONCHO_TEST_DERIVER_PAUSE_DIR"
PAUSE_MATCH_ENV = "HONCHO_TEST_DERIVER_PAUSE_MATCH"


async def maybe_pause_before_write(
    messages: list[Message], *, observers: list[str], observed: str
) -> None:
    """Hold the batch at the input-fetched / output-not-written seam, if enabled."""
    pause_dir = os.environ.get(PAUSE_DIR_ENV)
    match = os.environ.get(PAUSE_MATCH_ENV)
    if not pause_dir or not match:
        return
    if not any(match in message.content for message in messages):
        return

    directory = Path(pause_dir)
    hostname = socket.gethostname()
    marker = {
        "hostname": hostname,
        "observers": observers,
        "observed": observed,
        "message_ids": [message.id for message in messages],
        "message_public_ids": [message.public_id for message in messages],
    }
    (directory / f"paused-{hostname}.json").write_text(json.dumps(marker))
    logger.warning("Test pause hook holding batch on %s: %s", hostname, marker)

    while not (directory / "release").exists():
        await asyncio.sleep(0.2)

    (directory / f"released-{hostname}.json").write_text(json.dumps(marker))
    logger.warning("Test pause hook released batch on %s", hostname)
