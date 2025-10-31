"""
Prompts for the deriver module.

This module contains all prompt templates used by the deriver for critical analysis
and reasoning tasks.
"""

import datetime
from functools import cache
from inspect import cleandoc as c

from src.utils.representation import Representation
from src.utils.tokens import estimate_tokens

@cache
def estimate_base_prompt_tokens() -> int:
    """Estimate base prompt tokens by calling critical_analysis_prompt with empty values.

    This value is cached since it only changes on redeploys when the prompt template changes.
    """

    try:
        base_prompt = critical_analysis_prompt(
            peer_id="",
            peer_card=None,
            message_created_at=datetime.datetime.now(datetime.timezone.utc),
            working_representation=Representation(),
            history="",
            new_turns=[],
        )
        return estimate_tokens(base_prompt)
    except Exception:
        # Return a conservative estimate if estimation fails
        return 500
