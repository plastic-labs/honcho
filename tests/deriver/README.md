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
    observer=observer_peer.name,
    observed=message.peer_name
)

# Add to queue
queue_items = await add_queue_items([payload], session.id)
```

### Testing Work Units

```python
# Create a work unit
work_unit = WorkUnit(
    session_id=session.id,
    task_type="representation",
    observer=observer
    observed=observed
)

# Test string representation
assert str(work_unit) == f"({session.id}, {observed.name}, {observer.name}, representation)"
```
