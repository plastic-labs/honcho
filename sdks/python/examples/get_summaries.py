"""
Example of getting session summaries using the Honcho Python SDK.

This example demonstrates how to retrieve both short and long summaries
for a session, including their metadata like message ID, creation timestamp,
and token count.
"""

from honcho import Honcho, SessionSummaries
import asyncio
from honcho import AsyncHoncho
import os

# Initialize the Honcho client
api_key = os.getenv("HONCHO_API_KEY")
if not api_key:
    raise ValueError("Please set HONCHO_API_KEY environment variable")

client = Honcho(api_key=api_key)

# Get a session (replace with your actual session ID)
session = client.session("my-conversation-session")

# Get summaries for the session
summaries: SessionSummaries = session.get_summaries()

print(f"Session ID: {summaries.id}")
print("-" * 50)

# Check and display short summary
if summaries.short_summary:
    print("SHORT SUMMARY:")
    print(f"  Content: {summaries.short_summary.content[:200]}...")  # First 200 chars
    print(f"  Covers up to message ID: {summaries.short_summary.message_id}")
    print(f"  Created at: {summaries.short_summary.created_at}")
    print(f"  Token count: {summaries.short_summary.token_count}")
    print(f"  Type: {summaries.short_summary.summary_type}")
else:
    print("No short summary available yet")

print("-" * 50)

# Check and display long summary
if summaries.long_summary:
    print("LONG SUMMARY:")
    print(f"  Content: {summaries.long_summary.content[:200]}...")  # First 200 chars
    print(f"  Covers up to message ID: {summaries.long_summary.message_id}")
    print(f"  Created at: {summaries.long_summary.created_at}")
    print(f"  Token count: {summaries.long_summary.token_count}")
    print(f"  Type: {summaries.long_summary.summary_type}")
else:
    print("No long summary available yet")

# Example with async client
print("\n" + "=" * 50)
print("ASYNC EXAMPLE:")
print("=" * 50)


async def get_summaries_async():
    async_client = AsyncHoncho(api_key=api_key)

    # Get a session
    async_session = async_client.session("my-conversation-session")

    # Get summaries asynchronously
    summaries = await async_session.get_summaries()

    print(f"Session ID (async): {summaries.id}")

    if summaries.short_summary:
        print(
            f"Short summary available with {summaries.short_summary.token_count} tokens"
        )

    if summaries.long_summary:
        print(
            f"Long summary available with {summaries.long_summary.token_count} tokens"
        )


# Run the async example
asyncio.run(get_summaries_async())
