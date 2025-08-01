# Deriver Testing

This directory contains tests for the deriver system, which handles background processing of messages to extract insights and update working representations.

## Structure

- `conftest.py` - Shared fixtures for deriver testing
- `test_queue_operations.py` - Tests for basic queue operations
- `test_deriver_processing.py` - Tests for deriver processing logic
- `test_queue_processing.py` - Tests for queue manager and work unit processing

## Key Fixtures

### Database Fixtures

- `sample_session_with_peers` - Creates a session with multiple peers having different observation configurations
- `sample_messages` - Creates sample messages for testing
- `sample_queue_items` - Creates queue items with various payload types (representation, summary)

### Queue Fixtures

- `create_queue_payload` - Helper to create queue payloads for testing
- `add_queue_items` - Helper to add queue items to the database
- `create_active_queue_session` - Helper to create active queue sessions for work unit tracking

### Mocking Fixtures

- `mock_deriver_process` - Mocks the deriver process_message method
- `mock_critical_analysis_call` - Mocks the critical analysis LLM call
- `mock_queue_manager` - Mocks the queue manager for testing
- `mock_embedding_store` - Mocks the embedding store operations

## Testing Patterns

### Creating Queue Items

```python
# Create representation payloads
payload = create_queue_payload(
    message=message,
    task_type="representation",
    sender_name=message.peer_name,
    target_name=observer_peer.name,
)

# Add to queue
queue_items = await add_queue_items([payload], session.id)
```

### Testing Work Units

```python
# Create a work unit
work_unit = WorkUnit(
    session_id=session.id,
    sender_name=sender.name,
    target_name=target.name,
    task_type="representation",
)

# Test string representation
assert str(work_unit) == f"({session.id}, {sender.name}, {target.name}, representation)"
```

### Mocking Deriver Processing

```python
# Use the mock_deriver_process fixture to avoid actual LLM calls
async def test_with_mocked_deriver(mock_deriver_process):
    # Deriver processing will use the mock
    await process_item(queue_item.payload)
```
