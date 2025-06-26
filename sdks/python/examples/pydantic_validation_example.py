"""
Example demonstrating Pydantic validation in the Honcho SDK.

This example shows how the SDK now uses Pydantic to validate inputs at runtime,
providing better error messages and type safety.
"""

import logging

from honcho import Honcho
from pydantic import ValidationError

logging.basicConfig(level=logging.INFO)


def demonstrate_validation():
    """Demonstrate various validation scenarios with the Pydantic-enhanced SDK."""

    print("=== Pydantic Validation Examples ===\n")

    # Example 1: Valid initialization
    print("1. Valid initialization:")
    try:
        honcho = Honcho(
            environment="local",
            workspace_id="test_workspace",
            timeout=30.0,
            max_retries=3,
        )
        print(f"✅ Successfully created client: {honcho}")
    except ValidationError as e:
        print(f"❌ Validation error: {e}")

    print("\n" + "=" * 50 + "\n")

    # Example 2: Invalid timeout (negative)
    print("2. Invalid timeout (negative value):")
    try:
        honcho = Honcho(environment="local", timeout=-5.0)
        print("✅ This shouldn't happen!")
    except ValidationError as e:
        print(f"❌ Validation caught invalid timeout: {e.errors()[0]['msg']}")

    print("\n" + "=" * 50 + "\n")

    # Example 3: Invalid max_retries (negative)
    print("3. Invalid max_retries (negative value):")
    try:
        honcho = Honcho(environment="local", max_retries=-1)
        print("✅ This shouldn't happen!")
    except ValidationError as e:
        print(f"❌ Validation caught invalid max_retries: {e.errors()[0]['msg']}")

    print("\n" + "=" * 50 + "\n")

    # Example 4: Invalid peer ID (empty)
    print("4. Invalid peer ID (empty string):")
    try:
        honcho = Honcho(environment="local", workspace_id="test")
        peer = honcho.peer("")
        print("✅ This shouldn't happen!")
    except ValidationError as e:
        print(f"❌ Validation caught empty peer ID: {e.errors()[0]['msg']}")

    print("\n" + "=" * 50 + "\n")

    # Example 5: Invalid session ID (empty)
    print("5. Invalid session ID (empty string):")
    try:
        honcho = Honcho(environment="local", workspace_id="test")
        session = honcho.session("")
        print("✅ This shouldn't happen!")
    except ValidationError as e:
        print(f"❌ Validation caught empty session ID: {e.errors()[0]['msg']}")

    print("\n" + "=" * 50 + "\n")

    # Example 6: Valid peer and session creation
    print("6. Valid peer and session creation:")
    try:
        honcho = Honcho(environment="local", workspace_id="test")
        peer = honcho.peer("alice")
        session = honcho.session("conversation_1")
        print(f"✅ Created peer: {peer}")
        print(f"✅ Created session: {session}")
    except ValidationError as e:
        print(f"❌ Validation error: {e}")

    print("\n" + "=" * 50 + "\n")

    # Example 7: Invalid message content (empty)
    print("7. Invalid message content (empty string):")
    try:
        honcho = Honcho(environment="local", workspace_id="test")
        peer = honcho.peer("alice")
        message = peer.message("")
        print("✅ This shouldn't happen!")
    except ValidationError as e:
        print(f"❌ Validation caught empty message content: {e.errors()[0]['msg']}")

    print("\n" + "=" * 50 + "\n")

    # Example 9: Valid operations
    print("9. Valid operations (no API calls made):")
    try:
        honcho = Honcho(environment="local", workspace_id="test")
        peer = honcho.peer("alice")
        session = honcho.session("conversation_1")

        # Create a valid message
        message = peer.message("Hello, world!", metadata={"type": "greeting"})
        print(
            f"✅ Created message: peer_id={message['peer_id']}, content='{message['content']}'"
        )

        # Valid peer operations (validation passes, but no API calls made)
        print("✅ All validations passed for standard operations")

    except ValidationError as e:
        print(f"❌ Validation error: {e}")

    print("\n" + "=" * 50 + "\n")
    print("🎉 Pydantic validation examples completed!")


if __name__ == "__main__":
    demonstrate_validation()
