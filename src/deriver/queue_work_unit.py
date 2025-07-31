from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.models import QueueItem


@dataclass(frozen=True)
class WorkUnit(ABC):
    """Abstract base class for different types of work units in the queue system."""

    @abstractmethod
    def get_unique_key(self) -> str:
        """Generate a deterministic unique identifier for this work unit."""
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize work unit to JSON-compatible dict for metadata storage."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkUnit":
        """Deserialize work unit from dict."""
        pass

    @abstractmethod
    def build_queue_item_conditions(self, queue_item_model: Any) -> list[Any]:
        """Build SQLAlchemy where conditions for QueueItem queries."""
        pass


@dataclass(frozen=True)
class DeriverWorkUnit(WorkUnit):
    """Work unit for deriver tasks (summary, representation) that are session-scoped."""

    task_type: str
    session_id: str
    sender_name: str | None
    target_name: str | None

    def get_unique_key(self) -> str:
        """Create deterministic key including all identity fields."""
        sender = self.sender_name or "null"
        target = self.target_name or "null"
        return f"deriver:{self.task_type}:{self.session_id}:{sender}:{target}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "deriver",
            "task_type": self.task_type,
            "session_id": self.session_id,
            "sender_name": self.sender_name,
            "target_name": self.target_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeriverWorkUnit":
        return cls(
            task_type=data["task_type"],
            session_id=data["session_id"],
            sender_name=data.get("sender_name"),
            target_name=data.get("target_name"),
        )

    def build_queue_item_conditions(self, queue_item_model: QueueItem) -> list[Any]:
        conditions = [
            queue_item_model.task_type == self.task_type,
            queue_item_model.session_id == self.session_id,
        ]

        # For summary tasks, sender_name and target_name don't exist in payload
        if self.task_type != "summary":
            conditions.extend(
                [
                    queue_item_model.payload["sender_name"].astext == self.sender_name,
                    queue_item_model.payload["target_name"].astext == self.target_name,
                ]
            )

        return conditions

    def __str__(self) -> str:
        return f"({self.session_id}, {self.sender_name}, {self.target_name}, {self.task_type})"


@dataclass(frozen=True)
class WebhookWorkUnit(WorkUnit):
    """Work unit for webhook tasks that are workspace-scoped, not session-scoped."""

    task_type: str

    def get_unique_key(self) -> str:
        """Webhooks are grouped only by task type."""
        return f"webhook:{self.task_type}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "webhook",
            "task_type": self.task_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WebhookWorkUnit":
        return cls(task_type=data["task_type"])

    def build_queue_item_conditions(self, queue_item_model: QueueItem) -> list[Any]:
        return [
            queue_item_model.task_type == self.task_type,
            queue_item_model.session_id is None,  # pyright: ignore
        ]

    def __str__(self) -> str:
        return f"(task_type={self.task_type})"
