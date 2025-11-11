"""Explicit reasoner implementation."""

import logging

from src import exceptions, prometheus
from src.config import settings
from src.crud.representation import RepresentationManager
from src.deriver.prompts import explicit_reasoning_prompt
from src.deriver.reasoners.base import BaseReasoner
from src.models import Message
from src.utils.clients import honcho_llm_call
from src.utils.formatting import format_new_turn_with_timestamp
from src.utils.representation import ExplicitResponse, Representation

logger = logging.getLogger(__name__)


class ExplicitReasoner(BaseReasoner):
    """Explicit reasoning implementation.

    This reasoner processes inputs through explicit, step-by-step reasoning,
    extracting facts that are LITERALLY stated in messages.
    """

    representation_manager: RepresentationManager
    ctx: list[Message]
    observer: str
    observed: str
    estimated_input_tokens: int

    def __init__(
        self,
        representation_manager: RepresentationManager,
        ctx: list[Message],
        *,
        observed: str,
        observer: str,
        estimated_input_tokens: int,
    ) -> None:
        """Initialize the explicit reasoner.

        Args:
            representation_manager: Manager for saving representations
            ctx: List of messages to analyze
            observed: The peer being observed
            observer: The peer doing the observing
            estimated_input_tokens: Estimated token count for input
        """
        self.representation_manager = representation_manager
        self.ctx = ctx
        self.observed = observed
        self.observer = observer
        self.estimated_input_tokens = estimated_input_tokens

    async def reason(
        self,
        working_representation: Representation,
        history: str,
        speaker_peer_card: list[str] | None,
    ) -> tuple[ExplicitResponse, str]:
        """Process input through explicit reasoning.

        Args:
            working_representation: Current representation context
            history: Recent conversation history
            speaker_peer_card: Peer card for the observed peer

        Returns:
            Tuple of (ExplicitResponse, prompt string)
        """
        latest_message = self.ctx[-1]
        new_turns = [
            format_new_turn_with_timestamp(m.content, m.created_at, m.peer_name)
            for m in self.ctx
        ]

        prompt = explicit_reasoning_prompt(
            peer_id=self.observed,
            peer_card=speaker_peer_card,
            message_created_at=latest_message.created_at,
            working_representation=working_representation,
            history=history,
            new_turns=new_turns,
        )

        try:
            response = await honcho_llm_call(
                provider=settings.DERIVER.PROVIDER,
                model=settings.DERIVER.MODEL,
                prompt=prompt,
                max_tokens=settings.DERIVER.MAX_OUTPUT_TOKENS
                or settings.LLM.DEFAULT_MAX_TOKENS,
                track_name="Explicit Reasoning Call",
                response_model=ExplicitResponse,
                json_mode=True,
                stop_seqs=["   \\n", "\\n\\n\\n\\n"],
                thinking_budget_tokens=settings.DERIVER.THINKING_BUDGET_TOKENS,
                reasoning_effort="minimal",
                verbosity="medium",
                enable_retry=True,
                retry_attempts=3,
            )

            prometheus.DERIVER_TOKENS_PROCESSED.labels(
                task_type="explicit_reasoning",
            ).inc(response.output_tokens + self.estimated_input_tokens)

            return response.content, prompt
        except Exception as e:
            raise exceptions.LLMError(
                speaker_peer_card=speaker_peer_card,
                working_representation=working_representation,
                history=history,
                new_turns=new_turns,
            ) from e
