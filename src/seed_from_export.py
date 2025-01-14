import asyncio
import csv
import datetime
import json
import uuid
from pathlib import Path
from sqlalchemy import text

from .db import SessionLocal, engine
from .old_models import App, User, Session as ChatSession, Message, OldBase

SOURCE_SCHEMA = 'honcho_old'

async def parse_csv_file(file_path: Path) -> list[dict]:
    """Parse a CSV file and return a list of dictionaries"""
    with open(str(file_path), 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
        # Sort by created_at in ascending order (oldest first)
        return sorted(rows, key=lambda x: x['created_at'])

async def parse_metadata(metadata_str: str) -> dict:
    """Parse metadata string into dictionary, handling empty cases"""
    if not metadata_str or metadata_str == '{}':
        return {}
    try:
        return json.loads(metadata_str)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse metadata: {metadata_str}")
        return {}

async def seed_from_export(dump_dir: str = "src/yousim_dump"):
    """Seed the database with data from exported CSV files"""
    dump_path = Path(dump_dir)
    
    # Create schema if it doesn't exist
    print("Ensuring schema exists...")
    async with engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS {SOURCE_SCHEMA}'))
    
    # Drop existing tables and create new ones
    print("Dropping existing tables...")
    async with engine.begin() as conn:
        await conn.run_sync(OldBase.metadata.drop_all)
    print("Creating new tables...")
    async with engine.begin() as conn:
        await conn.run_sync(OldBase.metadata.create_all)
    
    # Track stats for reporting
    stats = {
        'apps': {'imported': 0, 'skipped': 0},
        'users': {'imported': 0, 'skipped': 0},
        'sessions': {'imported': 0, 'skipped': 0},
        'messages': {'imported': 0, 'skipped': 0}
    }
    
    # Store mappings for foreign key relationships
    app_id_mapping = {}
    user_id_mapping = {}
    session_id_mapping = {}
    
    # Import Apps
    async with SessionLocal() as session:
        async with session.begin():
            apps_data = await parse_csv_file(dump_path / "apps_rows (1).csv")
            for app_row in apps_data:
                try:
                    metadata = await parse_metadata(app_row['metadata'])
                    app = App(
                        id=uuid.UUID(app_row['id']),
                        name=app_row['name'],
                        created_at=datetime.datetime.fromisoformat(app_row['created_at']),
                        h_metadata=metadata
                    )
                    session.add(app)
                    app_id_mapping[app_row['id']] = app.id
                    stats['apps']['imported'] += 1
                except Exception as e:
                    print(f"Error importing app {app_row['id']}: {str(e)}")
                    stats['apps']['skipped'] += 1

    # Import Users
    async with SessionLocal() as session:
        async with session.begin():
            users_data = await parse_csv_file(dump_path / "users_rows (1).csv")
            for user_row in users_data:
                try:
                    metadata = await parse_metadata(user_row['metadata'])
                    user = User(
                        id=uuid.UUID(user_row['id']),
                        name=user_row['name'],
                        app_id=app_id_mapping[user_row['app_id']],
                        created_at=datetime.datetime.fromisoformat(user_row['created_at']),
                        h_metadata=metadata
                    )
                    session.add(user)
                    user_id_mapping[user_row['id']] = user.id
                    stats['users']['imported'] += 1
                except Exception as e:
                    print(f"Error importing user {user_row['id']}: {str(e)}")
                    stats['users']['skipped'] += 1

    # Import Sessions
    async with SessionLocal() as session:
        async with session.begin():
            sessions_data = await parse_csv_file(dump_path / "sessions_rows.csv")
            for session_row in sessions_data:
                try:
                    metadata = await parse_metadata(session_row['metadata'])
                    # Removed legacy ID updates from here
                    chat_session = ChatSession(
                        id=uuid.UUID(session_row['id']),
                        is_active=session_row['is_active'].lower() == 'true',
                        user_id=user_id_mapping[session_row['user_id']],
                        created_at=datetime.datetime.fromisoformat(session_row['created_at']),
                        h_metadata=metadata  # Using original metadata without modifications
                    )
                    session.add(chat_session)
                    session_id_mapping[session_row['id']] = chat_session.id
                    stats['sessions']['imported'] += 1
                except Exception as e:
                    print(f"Error importing session {session_row['id']}: {str(e)}")
                    stats['sessions']['skipped'] += 1

    # Import Messages
    async with SessionLocal() as session:
        async with session.begin():
            messages_data = await parse_csv_file(dump_path / "messages_rows.csv")
            for message_row in messages_data:
                try:
                    metadata = await parse_metadata(message_row['metadata'])
                    message = Message(
                        id=uuid.UUID(message_row['id']),
                        session_id=session_id_mapping[message_row['session_id']],
                        content=message_row['content'],
                        is_user=message_row['is_user'].lower() == 'true',
                        created_at=datetime.datetime.fromisoformat(message_row['created_at']),
                        h_metadata=metadata
                    )
                    session.add(message)
                    stats['messages']['imported'] += 1
                except Exception as e:
                    print(f"Error importing message {message_row['id']}: {str(e)}")
                    stats['messages']['skipped'] += 1

    # Print import statistics
    print("\nImport Statistics:")
    for entity, counts in stats.items():
        print(f"{entity.title()}:")
        print(f"  Imported: {counts['imported']}")
        print(f"  Skipped: {counts['skipped']}")

if __name__ == "__main__":
    asyncio.run(seed_from_export()) 