#!/usr/bin/env python3
"""Test KlimaShift filter with Honcho v3 API"""

import sys
sys.path.insert(0, '/app/backend/functions')

import klimashift_filter

print("=" * 60)
print("Testing KlimaShift Filter with Honcho v3 API")
print("=" * 60)

# Initialize filter
f = klimashift_filter.Filter()
print(f"\n✓ Filter initialized")
print(f"  Honcho API: {f.honcho_api}")
print(f"  Workspace: {f.workspace_id}")

# Test peer creation
print(f"\n1. Testing peer creation...")
peer_id = f.get_or_create_peer("test-user-123")
if peer_id:
    print(f"  ✓ Peer created/retrieved: {peer_id}")
else:
    print(f"  ✗ Failed to create peer")
    sys.exit(1)

# Test memory retrieval
print(f"\n2. Testing memory retrieval...")
memory = f.get_memory_context(peer_id, "test query")
print(f"  Memory context length: {len(memory)} chars")
if memory:
    print(f"  ✓ Memory retrieved")
else:
    print(f"  ⚠ No memory found (expected for new peer)")

# Test inlet (personality injection)
print(f"\n3. Testing inlet (personality injection)...")
body = {
    "messages": [
        {"role": "user", "content": "Hello"}
    ]
}
user = {"id": "test-user-123"}

result = f.inlet(body, user)
system_msg = None
for msg in result["messages"]:
    if msg.get("role") == "system":
        system_msg = msg.get("content", "")
        break

if system_msg and "KlimaShift Assistant" in system_msg:
    print(f"  ✓ Personality injected successfully")
    print(f"  System message length: {len(system_msg)} chars")
else:
    print(f"  ✗ Personality NOT injected")
    sys.exit(1)

print(f"\n4. Testing outlet (message saving)...")
body = {
    "messages": [
        {"role": "user", "content": "Test question"},
        {"role": "assistant", "content": "Test answer"}
    ],
    "chat_id": "test-chat-123",
    "__klimashift_peer_id": peer_id
}

result = f.outlet(body, user)
print(f"  ✓ Outlet executed (check Honcho database for saved messages)")

print(f"\n{'=' * 60}")
print(f"✓ All tests passed!")
print(f"{'=' * 60}")
