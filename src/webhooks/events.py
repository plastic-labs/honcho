from enum import Enum

from pydantic import BaseModel


class WebhookEventType(str, Enum):
    QUEUE_EMPTY = "queue.empty"


class QueueEmptyPayload(BaseModel):
    workspace_name: str
    queue_type: str


WebhookPayload = QueueEmptyPayload


class WebhookEvent(BaseModel):
    type: WebhookEventType
    data: WebhookPayload
    workspace_name: str
