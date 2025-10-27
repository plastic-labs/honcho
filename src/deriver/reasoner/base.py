from abc import ABC, abstractmethod

from src.schemas import Message
from src.utils.representation import Representation
from src.crud.representation import RepresentationManager

class BaseReasoner(ABC):
    def __init__(self,
        representation_manager: RepresentationManager,
        ctx: list[Message],
        *,
        observed: str,
        observer: str,
        estimated_input_tokens: int,
    ) -> None:
        self.representation_manager = representation_manager
        self.ctx = ctx
        self.observed = observed
        self.observer = observer
        self.estimated_input_tokens = estimated_input_tokens

    @abstractmethod
    async def reason(self,
        working_representation: Representation,
        history: str,
        speaker_peer_card: list[str] | None,
    ) -> Representation:
        raise NotImplementedError
    
    async def update_peer_card(self,
        old_peer_card: list[str] | None,
        new_observations: Representation,
    ) -> None:
        raise NotImplementedError
