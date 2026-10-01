import logging
from enum import Enum
from typing import Literal

from pydantic import BaseModel

from src.config import settings
from src.db import tenant_context
from src.dependencies import tracked_db
from src.models import QueueItem
from src.utils.queue_payload import create_webhook_payload
from src.utils.work_unit import construct_work_unit_key

logger = logging.getLogger(__name__)


class WebhookEventType(str, Enum):
    QUEUE_EMPTY = "queue.empty"
    TEST = "test.event"


class BaseWebhookEvent(BaseModel):
    """Base class for all webhook events."""

    workspace_id: str


class QueueEmptyEvent(BaseWebhookEvent):
    """Webhook event for when a queue becomes empty."""

    type: Literal[WebhookEventType.QUEUE_EMPTY] = WebhookEventType.QUEUE_EMPTY
    queue_type: str
    session_id: str | None = None
    observer: str | None = None
    observed: str | None = None


class TestEvent(BaseWebhookEvent):
    """Webhook event for testing."""

    type: Literal[WebhookEventType.TEST] = WebhookEventType.TEST


# Union type for all webhook events
WebhookEvent = QueueEmptyEvent | TestEvent


async def publish_webhook_event(
    event: WebhookEvent, *, tenant_id: str | None = None
) -> None:
    """
    Add a webhook event to our DB queue.

    Args:
        event: The webhook event to publish.
        tenant_id: The tenant this event belongs to. Required under MULTI_TENANT.
    """
    # region ai
    # webhook is a tenant-scoped task type. The deriver's queue-drain caller runs
    # AFTER process_work_unit has reset tenant_context — no ambient tenant there —
    # so it must (and does) pass the work unit's tenant explicitly. A request-path
    # caller (the /test route) instead has an ambient tenant_context set by auth
    # and may omit tenant_id, so resolve it here rather than at each use below:
    # tracked_db and construct_work_unit_key each fall back to tenant_context
    # internally when their tenant_id argument is None, and doing that
    # independently at every call below would namespace the work_unit_key to the
    # ambient tenant while this function's own QueueItem(tenant_id=...) kept the
    # original None — column and key would disagree, corrupting fair-scheduling
    # attribution. Resolving once, up front, keeps every use below in agreement.
    # Gated on MULTI_TENANT even though tenant_context is never set to a non-None
    # value flag-off today (every tenant_context.set() call site gates itself the
    # same way): flag-off QueueItem.tenant_id must stay NULL exactly as today,
    # not become hostage to some future setter skipping that gate.
    # endregion
    if tenant_id is None and settings.MULTI_TENANT:
        tenant_id = tenant_context.get()
    try:
        # Note: workspace_name is no longer included in the payload
        # It's stored directly on the queue item
        payload = create_webhook_payload(
            event_type=event.type.value,
            data=event.model_dump(mode="json", exclude={"type"}),
        )

        async with tracked_db("publish_webhook_event", tenant_id=tenant_id) as db:
            queue_item = QueueItem(
                work_unit_key=construct_work_unit_key(
                    event.workspace_id,
                    {
                        "task_type": "webhook",
                    },
                    tenant_id=tenant_id,
                ),
                tenant_id=tenant_id,
                payload=payload,
                session_id=None,
                task_type="webhook",
                workspace_name=event.workspace_id,
                message_id=None,  # Webhooks don't have a message_id
            )
            db.add(queue_item)
            await db.commit()
            logger.debug(
                "Published webhook event '%s' for workspace '%s'",
                event.type,
                event.workspace_id,
            )

    except Exception:
        logger.exception(
            "Failed to publish webhook event %s",
            event.type,
        )
