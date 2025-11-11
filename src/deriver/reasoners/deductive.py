"""Deductive reasoner implementation."""

import logging

from src import exceptions, prometheus
from src.config import settings
from src.crud.representation import RepresentationManager
from src.deriver.prompts import deductive_reasoning_prompt
from src.deriver.reasoners.base import BaseReasoner
from src.models import Message
from src.utils.clients import honcho_llm_call
from src.utils.formatting import format_new_turn_with_timestamp
from src.utils.representation import DeductiveResponse, Representation

logger = logging.getLogger(__name__)


class DeductiveReasoner(BaseReasoner):
    """Deductive reasoning implementation.

    This reasoner applies deductive logic to derive conclusions from
    established facts and rules, building on explicit observations.
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
        """Initialize the deductive reasoner.

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
        atomic_propositions: list[str],
        history: str,
        speaker_peer_card: list[str] | None,
    ) -> tuple[DeductiveResponse, str]:
        """Process input through deductive reasoning.

        Args:
            working_representation: Current representation context
            atomic_propositions: New atomic propositions from explicit reasoning
                (both explicit and implicit observations as content strings)
            history: Recent conversation history
            speaker_peer_card: Peer card for the observed peer

        Returns:
            Tuple of (DeductiveResponse, prompt string)
        """
        latest_message = self.ctx[-1]
        new_turns = [
            format_new_turn_with_timestamp(m.content, m.created_at, m.peer_name)
            for m in self.ctx
        ]

        prompt = deductive_reasoning_prompt(
            peer_id=self.observed,
            peer_card=speaker_peer_card,
            message_created_at=latest_message.created_at,
            working_representation=working_representation,
            atomic_propositions=atomic_propositions,
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
                track_name="Deductive Reasoning Call",
                response_model=DeductiveResponse,
                json_mode=True,
                stop_seqs=["   \\n", "\\n\\n\\n\\n"],
                thinking_budget_tokens=settings.DERIVER.THINKING_BUDGET_TOKENS,
                reasoning_effort="minimal",
                verbosity="medium",
                enable_retry=True,
                retry_attempts=3,
            )

            prometheus.DERIVER_TOKENS_PROCESSED.labels(
                task_type="deductive_reasoning",
            ).inc(response.output_tokens + self.estimated_input_tokens)

            return response.content, prompt
        except Exception as e:
            raise exceptions.LLMError(
                speaker_peer_card=speaker_peer_card,
                working_representation=working_representation,
                history=history,
                new_turns=new_turns,
            ) from e
