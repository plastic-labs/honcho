# Webhooks

Webhooks are used to deliver event notifications to user-configured URLs.

## System Architecture

The webhooks system consists of several key components:

* **API Endpoints (`routers/webhooks.py`):** Provides endpoints for users to create, list, delete, and test their webhook subscriptions.
* **Event Publishing (`webhooks/events.py`):** Defines the event types and allows us to publish new events to the processing queue.
* **Webhook Delivery (`webhooks/webhook_delivery.py`):** Contains the logic for sending the webhook to the subscriber's URL.

## Event Flow

1. An event is triggered within the application by calling `publish_webhook_event` with a defined event payload.
2. This function creates a `QueueItem` and stores it in the database.
3. The `QueueManager` background process polls the database for new items.
4. When a new event is found, it is passed to the `deliver_webhook` function.
5. `deliver_webhook` fetches all subscriber URLs for the event's workspace, signs the payload with a secret key, and sends an HTTP POST request to each URL.

Note that the webhooks require the *deriver* process to be running to faciliate the delivery of the webhook.
