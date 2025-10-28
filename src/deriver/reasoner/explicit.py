import time
import logging  
import datetime

import sentry_sdk

from src import prometheus
from src.schemas import Message
from src.config import settings
from src import crud, exceptions
from src.dependencies import tracked_db
from src.utils.clients import honcho_llm_call
from src.utils.peer_card import PeerCardQuery
from src.utils.files import load_prompt_template
from src.deriver.reasoner.base import BaseReasoner
from src.utils.langfuse_client import get_langfuse_client
from src.crud.representation import RepresentationManager
from src.utils.formatting import format_new_turn_with_timestamp
from src.utils.representation import PromptRepresentation, Representation
from src.utils.logging import (
    accumulate_metric, 
    log_representation,
    conditional_observe,
)

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

lf = get_langfuse_client() if settings.LANGFUSE_PUBLIC_KEY else None

class ExplicitReasoner(BaseReasoner):
    REASONING_TYPE: str = "explicit"
    representation_manager: RepresentationManager
    ctx: list[Message]
    observed: str
    observer: str

    def __init__(
        self,
        representation_manager: RepresentationManager,
        ctx: list[Message],
        *,
        observed: str,
        observer: str,
        estimated_input_tokens: int,
    ) -> None:
        """Explicit Reasoner for Deriver."""
        super().__init__(
            representation_manager, 
            ctx, 
            observed=observed, 
            observer=observer, 
            estimated_input_tokens=estimated_input_tokens
        )

        # Load Explicit Reasoning Prompt
        self.prompt = {}
        self.prompt['system'] = load_prompt_template(
            settings.DERIVER.PROMPTS_BASE_PATH / self.REASONING_TYPE, "system.jinja")
        self.prompt['user'] = load_prompt_template(
            settings.DERIVER.PROMPTS_BASE_PATH / self.REASONING_TYPE, "user.jinja")

        self.peer_prompt = {}
        self.peer_prompt['system'] = load_prompt_template(
            settings.PEER_CARD.PROMPTS_PATH, "system.jinja")
        self.peer_prompt['user'] = load_prompt_template(
            settings.PEER_CARD.PROMPTS_PATH, "user.jinja")

    @conditional_observe
    @sentry_sdk.trace
    async def reason(
        self,
        working_representation: Representation,
        history: str,
        speaker_peer_card: list[str] | None,
    ) -> Representation:
        """
        Single-pass reasoning function that performs explicit and 
        implicit derivations from the history and the latest message.
        
        Returns:
            Representation: Final observations
        """
        analysis_start = time.perf_counter()

        earliest_message = self.ctx[0]
        latest_message = self.ctx[-1]

        new_turns = [
            format_new_turn_with_timestamp(m.content, m.created_at, m.peer_name)
            for m in self.ctx
        ]

        logger.debug(
            "EXPLICIT REASONING: message_created_at='%s', new_turns_count=%s",
            latest_message.created_at,
            len(new_turns),
        )

        try:
            reasoning_response = await self._reasoning_call(
                peer_id=self.observed,
                peer_card=speaker_peer_card,
                message_created_at=latest_message.created_at,
                working_representation=working_representation,
                history=history,
                new_turns=new_turns,
                estimated_input_tokens=self.estimated_input_tokens,
            )
        except Exception as e:
            raise exceptions.LLMError(
                speaker_peer_card=speaker_peer_card,
                working_representation=working_representation,
                history=history,
                new_turns=new_turns,
            ) from e

        reasoning_response = Representation.from_prompt_representation(
            reasoning_response,
            (earliest_message.id, latest_message.id),
            latest_message.session_name,
            latest_message.created_at,
        )

        if lf:
            lf.update_current_generation(
                output=reasoning_response.format_as_markdown(),
            )

        analysis_duration_ms = (time.perf_counter() - analysis_start) * 1000
        accumulate_metric(
            f"deriver_{latest_message.id}_{self.observer}",
            "explicit_reasoning_duration",
            analysis_duration_ms,
            "ms",
        )

        # Save only the new observations that weren't in the original context
        new_observations = working_representation.diff_representation(
            reasoning_response
        )
        if not new_observations.is_empty():
            await self.representation_manager.save_representation(
                new_observations,
                (earliest_message.id, latest_message.id),
                latest_message.session_name,
                latest_message.created_at,
            )
        
        if settings.PEER_CARD.ENABLED:
            update_peer_card_start = time.perf_counter()
            if not new_observations.is_empty():
                await self.update_peer_card(speaker_peer_card, new_observations)
            update_peer_card_duration = (
                time.perf_counter() - update_peer_card_start
            ) * 1000
            accumulate_metric(
                f"deriver_{latest_message.id}_{self.observer}",
                "update_peer_card",
                update_peer_card_duration,
                "ms",
            )

        return reasoning_response

    async def _reasoning_call(
        self,
        peer_id: str,
        peer_card: list[str] | None,
        message_created_at: datetime.datetime,
        working_representation: Representation,
        history: str,
        new_turns: list[str],
        estimated_input_tokens: int,
    ) -> PromptRepresentation:
        """Explicit Reasoning Call"""
        # Transform Input Data for Prompt
        peer_card_section = (
            f"""{peer_id}'s known biographical information:
<peer_card>
{chr(10).join(peer_card)}
</peer_card>
"""
            if peer_card is not None
            else ""
        )

        working_representation_section = (
            f"""Current understanding of {peer_id}:
<current_context>
{str(working_representation)}
</current_context>
"""
            if not working_representation.is_empty()
            else ""
        )

        new_turns_section = "\n".join(new_turns)

        # Initialize and Render Prompts
        sys_prompt = self.prompt['system'].render(
            peer_id=peer_id,
            reasoning_type=self.REASONING_TYPE,
        ) if self.prompt['system'] else None
        usr_prompt = self.prompt['user'].render(
            peer_id=peer_id,
            peer_card_section=peer_card_section,
            message_created_at=message_created_at,
            working_representation_section=working_representation_section,
            history=history,
            new_turns_section=new_turns_section,
        ) if self.prompt['user'] else ''

        # Perform LLM Call
        response = await honcho_llm_call(
            provider=settings.DERIVER.PROVIDER,
            model=settings.DERIVER.MODEL,
            prompt=usr_prompt,
            max_tokens=settings.DERIVER.MAX_OUTPUT_TOKENS
            or settings.LLM.DEFAULT_MAX_TOKENS,
            track_name="Explicit Reasoning Call",
            response_model=PromptRepresentation,
            json_mode=True,
            stop_seqs=["   \n", "\n\n\n\n"],
            thinking_budget_tokens=settings.DERIVER.THINKING_BUDGET_TOKENS,
            reasoning_effort="minimal",
            verbosity="medium",
            enable_retry=True,
            retry_attempts=3,
            sys_prompt=sys_prompt,
        )

        prometheus.DERIVER_TOKENS_PROCESSED.labels(
            task_type="explicit_reasoning",
        ).inc(response.output_tokens + estimated_input_tokens)

        return response.content

    async def update_peer_card(
        self,
        old_peer_card: list[str] | None,
        new_observations: Representation,
    ) -> None:
        """
        Update the peer card by calling LLM with the old peer card and new observations.
        The new peer card is returned by the LLM and saved to peer internal metadata.
        """
        try:
            response = await self._peer_card_call(old_peer_card, new_observations)
            new_peer_card = response.card
            if not new_peer_card:
                return
            
            new_peer_card = [
                observation
                for observation in new_peer_card
                if not observation.lower().startswith(("note", "notes"))
            ]
            accumulate_metric(
                f"deriver_{self.ctx[-1].id}_{self.observer}",
                "new_peer_card",
                "\n".join(new_peer_card),
                "blob",
            )
            async with tracked_db("deriver.update_peer_card") as db:
                await crud.set_peer_card(
                    db,
                    self.ctx[0].workspace_name,
                    new_peer_card,
                    observer=self.observer,
                    observed=self.observed,
                )
        except Exception as e:
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_exception(e)
            logger.error("Error updating peer card! Skipping... %s", e)
    
    async def _peer_card_call(
        self,
        old_peer_card: list[str] | None,
        new_observations: Representation,
    ) -> list[str]:
        """
        Generate peer card prompt, call LLM with response model.
        """
        # Transform Input Data for Prompt
        old_peer_card_section = (
            f"""
Current user biographical card:
{chr(10).join(old_peer_card)}
    """
            if old_peer_card is not None
            else """
User does not have a card. Create one with any key observations.
    """
        )
        new_observations_section = (
            f"""
New observations:
{new_observations.str_no_timestamps()}
    """
        )

        # Initialize and Render Prompts
        sys_prompt = self.peer_prompt['system'].render(
            old_peer_card_section=old_peer_card_section,
            new_observations=new_observations_section,
        ) if self.peer_prompt['system'] else None
        usr_prompt = self.peer_prompt['user'].render(
            old_peer_card_section=old_peer_card_section,
            new_observations=new_observations_section,
        ) if self.peer_prompt['user'] else ''

        # Perform LLM Call
        response = await honcho_llm_call(
            provider=settings.PEER_CARD.PROVIDER,
            model=settings.PEER_CARD.MODEL,
            prompt=usr_prompt,
            max_tokens=settings.PEER_CARD.MAX_OUTPUT_TOKENS
            or settings.LLM.DEFAULT_MAX_TOKENS,
            track_name="Peer Card Call",
            response_model=PeerCardQuery,
            json_mode=True,
            stop_seqs=["   \n", "\n\n\n\n"],
            sys_prompt=sys_prompt,
        )

        return response.content
