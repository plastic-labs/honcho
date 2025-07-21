"""
Script to generate embeddings for existing messages that don't already have embeddings.

# Note: When generating embeddings for messages, we need to consider two limits defined in the settings:
# 1. MAX_EMBEDDING_TOKENS: This is the maximum number of tokens that can be included in a single message for which an embedding is generated.
#    If a message exceeds this limit, it will be chunked into multiple embeddings.
# 2. MAX_EMBEDDING_TOKENS_PER_REQUEST: This is the maximum total number of tokens that can be included in a single request to the embedding provider.
#    If the total number of tokens across all messages in a batch exceeds this limit, the batch will need to be split into multiple batches.

Usage:
    python scripts/generate_message_embeddings.py [--workspace-name WORKSPACE] [--session-name SESSION] [--peer-name PEER]
"""

import argparse
import asyncio
import os
import sys

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import tiktoken  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from src import models  # noqa: E402
from src.config import settings  # noqa: E402
from src.dependencies import tracked_db  # noqa: E402
from src.embedding_client import EmbeddingClient  # noqa: E402


async def get_messages_without_embeddings(
    db: AsyncSession,
    workspace_name: str | None = None,
    session_name: str | None = None,
    peer_name: str | None = None,
) -> list[models.Message]:
    """
    Get all messages that don't have embeddings yet.

    Args:
        db: Database session
        workspace_name: Optional workspace name filter
        session_name: Optional session name filter
        peer_name: Optional peer name filter

    Returns:
        List of messages without embeddings
    """
    # Query messages that don't have embeddings
    stmt = (
        select(models.Message)
        .outerjoin(
            models.MessageEmbedding,
            models.Message.public_id == models.MessageEmbedding.message_id,
        )
        .where(models.MessageEmbedding.message_id.is_(None))  # No embedding exists
        .order_by(models.Message.id)
    )

    # Apply filters if provided
    if workspace_name:
        stmt = stmt.where(models.Message.workspace_name == workspace_name)

    if session_name:
        stmt = stmt.where(models.Message.session_name == session_name)

    if peer_name:
        stmt = stmt.where(models.Message.peer_name == peer_name)

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def create_embeddings_for_messages(
    db: AsyncSession,
    messages: list[models.Message],
    embedding_client: EmbeddingClient,
) -> int:
    """
    Create embeddings for a batch of messages.

    Args:
        db: Database session
        messages: List of messages to create embeddings for
        embedding_client: Embedding client instance

    Returns:
        Number of embeddings created
    """
    if not messages:
        return 0

    # Initialize tiktoken encoding (same as used in MessageCreate schema)
    encoding = tiktoken.get_encoding("cl100k_base")

    # Prepare data for batch embedding with proper token encoding
    id_resource_dict = {
        message.public_id: (
            message.content,
            encoding.encode(message.content),  # Properly encode the content
        )
        for message in messages
    }

    # Generate embeddings
    embedding_dict = await embedding_client.batch_embed(id_resource_dict)

    # Create MessageEmbedding objects
    embedding_objects: list[models.MessageEmbedding] = []
    embeddings_created = 0

    for message in messages:
        embeddings = embedding_dict.get(message.public_id, [])
        for embedding in embeddings:
            embedding_obj = models.MessageEmbedding(
                content=message.content,
                embedding=embedding,
                message_id=message.public_id,
                workspace_name=message.workspace_name,
                session_name=message.session_name,
                peer_name=message.peer_name,
            )
            embedding_objects.append(embedding_obj)
            embeddings_created += 1

    # Add to database
    if embedding_objects:
        db.add_all(embedding_objects)
        await db.commit()

    return embeddings_created


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate embeddings for messages that don't already have them",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of messages to process in each batch (default: 50)",
    )
    parser.add_argument(
        "--workspace-name",
        help="Only process messages from this workspace",
    )
    parser.add_argument(
        "--session-name",
        help="Only process messages from this session",
    )
    parser.add_argument(
        "--peer-name",
        help="Only process messages from this peer",
    )

    args = parser.parse_args()

    # Initialize embedding client
    embedding_client = EmbeddingClient(settings.LLM.OPENAI_API_KEY)

    print("Generating embeddings for messages...")
    if args.workspace_name:
        print(f"  Filtering by workspace: {args.workspace_name}")
    else:
        print("  Processing all workspaces")
    if args.session_name:
        print(f"  Filtering by session: {args.session_name}")
    if args.peer_name:
        print(f"  Filtering by peer: {args.peer_name}")

    # Use tracked_db context manager for proper database session handling
    async with tracked_db("generate_embeddings") as db:
        try:
            # Get messages without embeddings
            print("Finding messages without embeddings...")
            messages = await get_messages_without_embeddings(
                db, args.workspace_name, args.session_name, args.peer_name
            )

            if not messages:
                print("No messages found that need embeddings.")
                return

            print(f"Found {len(messages)} messages without embeddings.")

            # Process in batches
            batch_size = args.batch_size
            total_embeddings = 0

            for i in range(0, len(messages), batch_size):
                batch = messages[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(messages) + batch_size - 1) // batch_size

                print(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch)} messages)..."
                )

                embeddings_created = await create_embeddings_for_messages(
                    db, batch, embedding_client
                )
                total_embeddings += embeddings_created

                print(
                    f"  Created {embeddings_created} embeddings for batch {batch_num}"
                )

            print(
                f"\nCompleted! Created {total_embeddings} embeddings for {len(messages)} messages."
            )

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
