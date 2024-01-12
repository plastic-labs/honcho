from abc import ABC, abstractmethod
from typing import List, Dict


class Mediator(ABC):
    @abstractmethod
    def get_sessions(self, user_id: str, location_id: str) -> List[Dict] | None:
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> Dict | None:
        pass

    @abstractmethod
    def add_session(self, user_id: str, location_id: str, metadata: Dict) -> Dict:
        pass

    @abstractmethod
    def update_session(self, session_id: str, metadata: Dict) -> None:
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        pass
