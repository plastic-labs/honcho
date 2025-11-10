"""Generate 500k test messages, message embeddings, and documents using raw psycopg3.

This script creates:
- 1 workspace
- 3 peers
- 3 sessions (with peers added to sessions via session_peers table)
- 500k messages distributed randomly across peers and sessions
- 500k message embeddings (one per message with constant zero vector)
- 1 collection per peer (3 total)
- 500k documents distributed across the collections

Usage:
    python manual_load.py [schema_name]

    If schema_name is not provided, uses DB_SCHEMA environment variable.

Environment variables required:
    DB_CONNECTION_URI: PostgreSQL connection URI
    DB_SCHEMA: Database schema to use (optional if passed as argument)
"""

import os
import random
import sys
import time

import psycopg
from dotenv import load_dotenv
from faker import Faker
from nanoid import generate as generate_nanoid

load_dotenv()

# Configuration
NUM_MESSAGES = 500_000
NUM_DOCUMENTS = 500_000
BATCH_SIZE = 10_000  # Batch size for inserts (messages and documents)
EMBEDDING_BATCH_SIZE = 1_000  # Smaller batch size for embeddings (they're large)
MAX_RETRIES = 3  # Number of retries for failed operations
RETRY_DELAY = 5  # Seconds to wait before retrying
NUM_PEERS = 3
NUM_SESSIONS = 3

# Load environment variables
DB_CONNECTION_URI = os.getenv("DB_CONNECTION_URI")

# Get schema from command line argument or environment variable
if len(sys.argv) > 1:
    DB_SCHEMA = sys.argv[1]
else:
    DB_SCHEMA = os.getenv("DB_SCHEMA")

if not DB_CONNECTION_URI:
    raise ValueError("DB_CONNECTION_URI environment variable is required")
if not DB_SCHEMA:
    raise ValueError(
        "DB_SCHEMA must be provided as command line argument or environment variable"
    )

# Convert SQLAlchemy-style URI to plain psycopg URI
# SQLAlchemy uses postgresql+psycopg://, psycopg uses postgresql://
if DB_CONNECTION_URI.startswith("postgresql+psycopg://"):
    DB_CONNECTION_URI = DB_CONNECTION_URI.replace("postgresql+psycopg://", "postgresql://")
elif DB_CONNECTION_URI.startswith("postgres+psycopg://"):
    DB_CONNECTION_URI = DB_CONNECTION_URI.replace("postgres+psycopg://", "postgresql://")

# Initialize Faker
fake = Faker()

# Constant embedding - all zeros for speed (1536 dimensions for OpenAI embeddings)
CONSTANT_EMBEDDING = str([0.0] * 1536)

# Generate IDs
WORKSPACE_ID = generate_nanoid()
WORKSPACE_NAME = "load_test_workspace"

PEER_IDS = [generate_nanoid() for _ in range(NUM_PEERS)]
PEER_NAMES = [f"load_peer_{i + 1}" for i in range(NUM_PEERS)]

SESSION_IDS = [generate_nanoid() for _ in range(NUM_SESSIONS)]
SESSION_NAMES = [f"load_session_{i + 1}" for i in range(NUM_SESSIONS)]

COLLECTION_IDS = [generate_nanoid() for _ in range(NUM_PEERS)]
# Collections use observer/observed paradigm - each peer observes itself
COLLECTION_OBSERVERS = PEER_NAMES
COLLECTION_OBSERVED = PEER_NAMES


def check_existing_data(conn, schema):
    """Check what data already exists in the database."""
    print("\nChecking existing data...")

    with conn.cursor() as cur:
        # Check workspace
        cur.execute(
            f'SELECT COUNT(*) FROM "{schema}"."workspaces" WHERE name = %s',
            (WORKSPACE_NAME,),
        )
        workspace_exists = cur.fetchone()[0] > 0

        # Check peers
        cur.execute(
            f'SELECT COUNT(*) FROM "{schema}"."peers" WHERE workspace_name = %s',
            (WORKSPACE_NAME,),
        )
        existing_peers = cur.fetchone()[0]

        # Check sessions
        cur.execute(
            f'SELECT COUNT(*) FROM "{schema}"."sessions" WHERE workspace_name = %s',
            (WORKSPACE_NAME,),
        )
        existing_sessions = cur.fetchone()[0]

        # Check collections (observer/observed pairs)
        cur.execute(
            f'SELECT COUNT(*) FROM "{schema}"."collections" WHERE workspace_name = %s',
            (WORKSPACE_NAME,),
        )
        existing_collections = cur.fetchone()[0]

        # Check messages
        cur.execute(
            f'SELECT COUNT(*) FROM "{schema}"."messages" WHERE workspace_name = %s',
            (WORKSPACE_NAME,),
        )
        existing_messages = cur.fetchone()[0]

        # Check message embeddings
        cur.execute(
            f'SELECT COUNT(*) FROM "{schema}"."message_embeddings" WHERE workspace_name = %s',
            (WORKSPACE_NAME,),
        )
        existing_embeddings = cur.fetchone()[0]

        # Check documents
        cur.execute(
            f'SELECT COUNT(*) FROM "{schema}"."documents" WHERE workspace_name = %s',
            (WORKSPACE_NAME,),
        )
        existing_documents = cur.fetchone()[0]

    print(f"  Workspace exists: {workspace_exists}")
    print(f"  Existing peers: {existing_peers}/{NUM_PEERS}")
    print(f"  Existing sessions: {existing_sessions}/{NUM_SESSIONS}")
    print(f"  Existing collections: {existing_collections}/{NUM_PEERS}")
    print(f"  Existing messages: {existing_messages:,}/{NUM_MESSAGES:,}")
    print(f"  Existing message embeddings: {existing_embeddings:,}/{NUM_MESSAGES:,}")
    print(f"  Existing documents: {existing_documents:,}/{NUM_DOCUMENTS:,}")

    return {
        "workspace_exists": workspace_exists,
        "existing_peers": existing_peers,
        "existing_sessions": existing_sessions,
        "existing_collections": existing_collections,
        "existing_messages": existing_messages,
        "existing_embeddings": existing_embeddings,
        "existing_documents": existing_documents,
    }


def create_entities(conn, schema, existing_data):
    """Create workspace, peers, sessions, and collections."""
    print("\nCreating workspace, peers, sessions, and collections...")

    with conn.cursor() as cur:
        # Create workspace if not exists
        if not existing_data["workspace_exists"]:
            cur.execute(
                f'INSERT INTO "{schema}"."workspaces" (id, name) VALUES (%s, %s)',
                (WORKSPACE_ID, WORKSPACE_NAME),
            )
            print(f"  ✓ Created workspace: {WORKSPACE_NAME}")
        else:
            print(f"  → Workspace already exists: {WORKSPACE_NAME}")

        # Create peers if needed
        if existing_data["existing_peers"] < NUM_PEERS:
            # Get existing peer names
            cur.execute(
                f'SELECT name FROM "{schema}"."peers" WHERE workspace_name = %s',
                (WORKSPACE_NAME,),
            )
            existing_peer_names = {row[0] for row in cur.fetchall()}

            for peer_id, peer_name in zip(PEER_IDS, PEER_NAMES, strict=False):
                if peer_name not in existing_peer_names:
                    cur.execute(
                        f'INSERT INTO "{schema}"."peers" (id, name, workspace_name) VALUES (%s, %s, %s)',
                        (peer_id, peer_name, WORKSPACE_NAME),
                    )
            new_peers = NUM_PEERS - existing_data["existing_peers"]
            print(
                f"  ✓ Created {new_peers} new peers ({existing_data['existing_peers']} already existed)"
            )
        else:
            print(f"  → All {NUM_PEERS} peers already exist")

        # Create sessions if needed
        if existing_data["existing_sessions"] < NUM_SESSIONS:
            # Get existing session names
            cur.execute(
                f'SELECT name FROM "{schema}"."sessions" WHERE workspace_name = %s',
                (WORKSPACE_NAME,),
            )
            existing_session_names = {row[0] for row in cur.fetchall()}

            for session_id, session_name in zip(
                SESSION_IDS, SESSION_NAMES, strict=False
            ):
                if session_name not in existing_session_names:
                    cur.execute(
                        f'INSERT INTO "{schema}"."sessions" (id, name, workspace_name) VALUES (%s, %s, %s)',
                        (session_id, session_name, WORKSPACE_NAME),
                    )
            new_sessions = NUM_SESSIONS - existing_data["existing_sessions"]
            print(
                f"  ✓ Created {new_sessions} new sessions ({existing_data['existing_sessions']} already existed)"
            )
        else:
            print(f"  → All {NUM_SESSIONS} sessions already exist")

        # Add peers to sessions via session_peers table
        # Check existing relationships
        cur.execute(
            f'SELECT COUNT(*) FROM "{schema}"."session_peers" WHERE workspace_name = %s',
            (WORKSPACE_NAME,),
        )
        existing_relationships = cur.fetchone()[0]
        expected_relationships = NUM_SESSIONS * NUM_PEERS

        if existing_relationships < expected_relationships:
            for session_name in SESSION_NAMES:
                for peer_name in PEER_NAMES:
                    # Check if relationship exists
                    cur.execute(
                        f'''SELECT COUNT(*) FROM "{schema}"."session_peers"
                        WHERE workspace_name = %s AND session_name = %s AND peer_name = %s''',
                        (WORKSPACE_NAME, session_name, peer_name),
                    )
                    if cur.fetchone()[0] == 0:
                        cur.execute(
                            f'''INSERT INTO "{schema}"."session_peers"
                            (workspace_name, session_name, peer_name)
                            VALUES (%s, %s, %s)''',
                            (WORKSPACE_NAME, session_name, peer_name),
                        )
            new_relationships = expected_relationships - existing_relationships
            print(
                f"  ✓ Added {new_relationships} new peer-session relationships ({existing_relationships} already existed)"
            )
        else:
            print("  → All peer-session relationships already exist")

        # Create collections (one per peer, self-observation) if needed
        if existing_data["existing_collections"] < NUM_PEERS:
            # Get existing collection observer/observed pairs
            cur.execute(
                f'SELECT observer, observed FROM "{schema}"."collections" WHERE workspace_name = %s',
                (WORKSPACE_NAME,),
            )
            existing_collections = {(row[0], row[1]) for row in cur.fetchall()}

            for collection_id, observer, observed in zip(
                COLLECTION_IDS, COLLECTION_OBSERVERS, COLLECTION_OBSERVED, strict=False
            ):
                if (observer, observed) not in existing_collections:
                    cur.execute(
                        f'INSERT INTO "{schema}"."collections" (id, observer, observed, workspace_name) VALUES (%s, %s, %s, %s)',
                        (collection_id, observer, observed, WORKSPACE_NAME),
                    )
            new_collections = NUM_PEERS - existing_data["existing_collections"]
            print(
                f"  ✓ Created {new_collections} new collections ({existing_data['existing_collections']} already existed)"
            )
        else:
            print(f"  → All {NUM_PEERS} collections already exist")

        conn.commit()


def generate_messages(conn, schema, existing_count):
    """Generate and insert messages up to NUM_MESSAGES total."""
    messages_to_create = NUM_MESSAGES - existing_count

    if messages_to_create <= 0:
        print(f"\n→ All {NUM_MESSAGES:,} messages already exist, skipping...")
        return []

    print(
        f"\nGenerating {messages_to_create:,} messages ({existing_count:,} already exist)..."
    )

    # Pre-generate all message data
    print("  Generating message data...")
    message_data = []
    for i in range(messages_to_create):
        public_id = generate_nanoid()
        peer_name = random.choice(PEER_NAMES)
        session_name = random.choice(SESSION_NAMES)
        content = fake.text(max_nb_chars=200)

        message_data.append(
            {
                "public_id": public_id,
                "peer_name": peer_name,
                "session_name": session_name,
                "content": content,
                "workspace_name": WORKSPACE_NAME,
            }
        )

    # Insert in batches
    print(f"  Inserting messages in batches of {BATCH_SIZE:,}...")
    total_batches = (messages_to_create + BATCH_SIZE - 1) // BATCH_SIZE

    with conn.cursor() as cur:
        for batch_num in range(total_batches):
            batch_start = batch_num * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, messages_to_create)
            batch = message_data[batch_start:batch_end]

            # Use execute_batch for better performance
            args = [
                (
                    msg["public_id"],
                    msg["session_name"],
                    msg["content"],
                    msg["peer_name"],
                    msg["workspace_name"],
                )
                for msg in batch
            ]

            cur.executemany(
                f'''INSERT INTO "{schema}"."messages"
                (public_id, session_name, content, peer_name, workspace_name)
                VALUES (%s, %s, %s, %s, %s)''',
                args,
            )
            conn.commit()

            messages_created = existing_count + batch_end
            progress = (messages_created / NUM_MESSAGES) * 100
            print(
                f"    Batch {batch_num + 1}/{total_batches}: {messages_created:,}/{NUM_MESSAGES:,} ({progress:.1f}%)"
            )

    print(f"  ✓ Successfully created {messages_to_create:,} new messages!")
    return message_data


def generate_message_embeddings(conn, schema, message_data, existing_count):
    """Generate and insert message embeddings for messages that don't have them."""
    embeddings_to_create = NUM_MESSAGES - existing_count

    if embeddings_to_create <= 0:
        print(f"\n→ All {NUM_MESSAGES:,} message embeddings already exist, skipping...")
        return conn

    print(
        f"\nGenerating {embeddings_to_create:,} message embeddings ({existing_count:,} already exist)..."
    )

    # If we just created messages, use that data
    if message_data:
        print("  Using newly created message data...")
    else:
        # Fetch messages that don't have embeddings yet in chunks
        # Using NOT EXISTS is much faster than LEFT JOIN for large tables
        print("  Fetching messages without embeddings in chunks...")
        message_data = []
        chunk_size = 50_000  # Fetch in chunks to avoid timeout
        total_fetched = 0

        with conn.cursor() as cur:
            while total_fetched < embeddings_to_create:
                remaining = embeddings_to_create - total_fetched
                fetch_limit = min(chunk_size, remaining)

                cur.execute(
                    f'''SELECT m.public_id, m.content, m.peer_name, m.session_name, m.workspace_name
                    FROM "{schema}"."messages" m
                    WHERE m.workspace_name = %s
                    AND NOT EXISTS (
                        SELECT 1 FROM "{schema}"."message_embeddings" me
                        WHERE me.message_id = m.public_id
                    )
                    LIMIT %s''',
                    (WORKSPACE_NAME, fetch_limit),
                )
                rows = cur.fetchall()

                if not rows:
                    break  # No more messages to fetch

                message_data.extend(
                    [
                        {
                            "public_id": row[0],
                            "content": row[1],
                            "peer_name": row[2],
                            "session_name": row[3],
                            "workspace_name": row[4],
                        }
                        for row in rows
                    ]
                )

                total_fetched += len(rows)
                print(
                    f"    Fetched {total_fetched:,}/{embeddings_to_create:,} messages..."
                )

    if not message_data:
        print("  ✓ No messages found that need embeddings")
        return conn

    # Insert in batches (smaller batch size for embeddings)
    print(f"  Inserting embeddings in batches of {EMBEDDING_BATCH_SIZE:,}...")
    total_batches = (
        len(message_data) + EMBEDDING_BATCH_SIZE - 1
    ) // EMBEDDING_BATCH_SIZE

    for batch_num in range(total_batches):
        batch_start = batch_num * EMBEDDING_BATCH_SIZE
        batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, len(message_data))
        batch = message_data[batch_start:batch_end]

        args = [
            (
                msg["content"],
                CONSTANT_EMBEDDING,
                msg["public_id"],
                msg["workspace_name"],
                msg["session_name"],
                msg["peer_name"],
            )
            for msg in batch
        ]

        # Retry logic for connection failures
        for attempt in range(MAX_RETRIES):
            try:
                with conn.cursor() as cur:
                    cur.executemany(
                        f'''INSERT INTO "{schema}"."message_embeddings"
                        (content, embedding, message_id, workspace_name, session_name, peer_name)
                        VALUES (%s, CAST(%s AS vector), %s, %s, %s, %s)''',
                        args,
                    )
                    conn.commit()
                break  # Success, exit retry loop
            except (psycopg.OperationalError, psycopg.InterfaceError):
                if attempt < MAX_RETRIES - 1:
                    print(
                        f"    ⚠ Connection error on batch {batch_num + 1}, retrying in {RETRY_DELAY}s... (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(RETRY_DELAY)
                    # Reconnect with keepalive settings
                    try:
                        conn.close()
                    except:
                        pass
                    conn = psycopg.connect(
                        DB_CONNECTION_URI,
                        keepalives=1,
                        keepalives_idle=30,
                        keepalives_interval=10,
                        keepalives_count=5,
                    )
                else:
                    print(f"    ✗ Failed after {MAX_RETRIES} attempts")
                    raise

        embeddings_created = existing_count + batch_end
        progress = (embeddings_created / NUM_MESSAGES) * 100
        print(
            f"    Batch {batch_num + 1}/{total_batches}: {embeddings_created:,}/{NUM_MESSAGES:,} ({progress:.1f}%)"
        )

    print(f"  ✓ Successfully created {len(message_data):,} new message embeddings!")
    return conn


def generate_documents(conn, schema, existing_count):
    """Generate and insert documents up to NUM_DOCUMENTS total."""
    documents_to_create = NUM_DOCUMENTS - existing_count

    if documents_to_create <= 0:
        print(f"\n→ All {NUM_DOCUMENTS:,} documents already exist, skipping...")
        return

    print(
        f"\nGenerating {documents_to_create:,} documents ({existing_count:,} already exist)..."
    )

    # Pre-generate all document data
    print("  Generating document data...")
    document_data = []
    for i in range(documents_to_create):
        doc_id = generate_nanoid()
        # Randomly assign to a collection/peer (observer/observed pairs)
        idx = random.randint(0, NUM_PEERS - 1)
        observer = COLLECTION_OBSERVERS[idx]
        observed = COLLECTION_OBSERVED[idx]
        # Randomly assign to a session
        session_name = random.choice(SESSION_NAMES)
        content = fake.text(max_nb_chars=300)

        document_data.append(
            {
                "id": doc_id,
                "content": content,
                "observer": observer,
                "observed": observed,
                "session_name": session_name,
                "workspace_name": WORKSPACE_NAME,
            }
        )

    # Insert in batches
    print(f"  Inserting documents in batches of {BATCH_SIZE:,}...")
    total_batches = (documents_to_create + BATCH_SIZE - 1) // BATCH_SIZE

    with conn.cursor() as cur:
        for batch_num in range(total_batches):
            batch_start = batch_num * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, documents_to_create)
            batch = document_data[batch_start:batch_end]

            args = [
                (
                    doc["id"],
                    doc["content"],
                    CONSTANT_EMBEDDING,
                    doc["observer"],
                    doc["observed"],
                    doc["workspace_name"],
                    doc["session_name"],
                )
                for doc in batch
            ]

            cur.executemany(
                f'''INSERT INTO "{schema}"."documents"
                (id, content, embedding, observer, observed, workspace_name, session_name)
                VALUES (%s, %s, CAST(%s AS vector), %s, %s, %s, %s)''',
                args,
            )
            conn.commit()

            documents_created = existing_count + batch_end
            progress = (documents_created / NUM_DOCUMENTS) * 100
            print(
                f"    Batch {batch_num + 1}/{total_batches}: {documents_created:,}/{NUM_DOCUMENTS:,} ({progress:.1f}%)"
            )

    print(f"  ✓ Successfully created {documents_to_create:,} new documents!")


def main():
    """Main execution function."""
    print("=" * 60)
    print("HONCHO LOAD TEST - Raw psycopg3 Implementation (Resumable)")
    print("=" * 60)
    print("Configuration:")
    print(f"  Database URI: {DB_CONNECTION_URI}")
    print(f"  Schema: {DB_SCHEMA}")
    print(f"  Target messages: {NUM_MESSAGES:,}")
    print(f"  Target message embeddings: {NUM_MESSAGES:,}")
    print(f"  Target documents: {NUM_DOCUMENTS:,}")
    print(f"  Batch size (messages/docs): {BATCH_SIZE:,}")
    print(f"  Batch size (embeddings): {EMBEDDING_BATCH_SIZE:,}")
    print("=" * 60)

    # Connect to database with keepalive settings
    print("\nConnecting to database...")
    conn = psycopg.connect(
        DB_CONNECTION_URI,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )

    try:
        # Check what already exists
        existing_data = check_existing_data(conn, DB_SCHEMA)

        # Create all entities
        create_entities(conn, DB_SCHEMA, existing_data)

        # Generate messages
        message_data = generate_messages(
            conn, DB_SCHEMA, existing_data["existing_messages"]
        )

        # Generate message embeddings (returns potentially reconnected connection)
        # conn = generate_message_embeddings(
        #     conn, DB_SCHEMA, message_data, existing_data["existing_embeddings"]
        # )

        # Generate documents
        generate_documents(conn, DB_SCHEMA, existing_data["existing_documents"])

        print("\n" + "=" * 60)
        print("✓ LOAD TEST COMPLETE!")
        print("=" * 60)
        print("Summary:")
        print(f"  Workspace: {WORKSPACE_NAME}")
        print(f"  Peers: {NUM_PEERS}")
        print(f"  Sessions: {NUM_SESSIONS}")
        print(f"  Collections: {NUM_PEERS}")
        print(f"  Messages: {NUM_MESSAGES:,}")
        print(f"  Message embeddings: {NUM_MESSAGES:,}")
        print(f"  Documents: {NUM_DOCUMENTS:,}")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        print(
            "\nThe script is resumable - simply run it again to continue from where it left off."
        )
        raise

    finally:
        conn.close()
        print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()
