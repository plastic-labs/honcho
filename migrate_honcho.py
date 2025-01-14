import asyncio
import json
import os
from typing import Dict, List, Set
from uuid import UUID
import pickle

from nanoid import generate as generate_nanoid
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from dotenv import load_dotenv

load_dotenv()

SOURCE_SCHEMA = 'honcho_old'
DEST_SCHEMA = 'honcho_new'
BATCH_SIZE = 100
MIGRATION_STATE_DIR = 'migration_state'

# Create state directory if it doesn't exist
os.makedirs(MIGRATION_STATE_DIR, exist_ok=True)

def load_processed_ids(table_name: str) -> Set[str]:
    """Load previously processed IDs from file"""
    try:
        with open(f'{MIGRATION_STATE_DIR}/{table_name}_processed.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return set()

def save_processed_ids(table_name: str, processed_ids: Set[str]):
    """Save processed IDs to file"""
    with open(f'{MIGRATION_STATE_DIR}/{table_name}_processed.pkl', 'wb') as f:
        pickle.dump(processed_ids, f)

def load_id_mapping(table_name: str) -> Dict[UUID, str]:
    """Load ID mapping from file"""
    try:
        with open(f'{MIGRATION_STATE_DIR}/{table_name}_mapping.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def save_id_mapping(table_name: str, mapping: Dict[UUID, str]):
    """Save ID mapping to file"""
    with open(f'{MIGRATION_STATE_DIR}/{table_name}_mapping.pkl', 'wb') as f:
        pickle.dump(mapping, f)

def create_db_engine(url: str) -> AsyncEngine:
    """Create an async database engine from a connection URL"""
    if url.startswith('postgresql://'):
        url = url.replace('postgresql://', 'postgresql+asyncpg://', 1)
    return create_async_engine(url, echo=False, pool_pre_ping=True)

async def migrate_apps(session: AsyncSession) -> None:
    """Migrate apps table"""
    print("Migrating apps...")
    processed_ids = load_processed_ids('apps')
    id_mapping = load_id_mapping('apps')
    
    while True:
        result = await session.execute(text(f'''
            SELECT id::text, name, created_at, metadata 
            FROM {SOURCE_SCHEMA}.apps 
            WHERE id::text NOT IN :processed_ids
            ORDER BY created_at ASC
            LIMIT :batch_size
        '''), {
            'processed_ids': tuple(processed_ids) if processed_ids else ('',),
            'batch_size': BATCH_SIZE
        })
        
        rows = result.mappings().all()
        if not rows:
            break
            
        for row in rows:
            public_id = generate_nanoid()
            uuid_id = UUID(row['id'])
            id_mapping[uuid_id] = public_id
            
            await session.execute(text(f'''
                INSERT INTO {DEST_SCHEMA}.apps (
                    public_id, name, created_at, metadata
                ) VALUES (
                    :public_id, :name, :created_at, cast(:metadata as jsonb)
                )
            '''), {
                'public_id': public_id,
                'name': row['name'],
                'created_at': row['created_at'],
                'metadata': json.dumps(row['metadata'] or {})
            })
            
            processed_ids.add(row['id'])
        
        await session.commit()
        save_processed_ids('apps', processed_ids)
        save_id_mapping('apps', id_mapping)
        print(f"Processed {len(processed_ids)} apps")

async def migrate_users(session: AsyncSession) -> None:
    """Migrate users table"""
    print("Migrating users...")
    processed_ids = load_processed_ids('users')
    id_mapping = load_id_mapping('users')
    apps_mapping = load_id_mapping('apps')
    
    while True:
        result = await session.execute(text(f'''
            SELECT id::text, name, app_id::text, created_at, metadata 
            FROM {SOURCE_SCHEMA}.users 
            WHERE id::text NOT IN :processed_ids
            ORDER BY created_at ASC
            LIMIT :batch_size
        '''), {
            'processed_ids': tuple(processed_ids) if processed_ids else ('',),
            'batch_size': BATCH_SIZE
        })
        
        rows = result.mappings().all()
        if not rows:
            break
            
        for row in rows:
            public_id = generate_nanoid()
            uuid_id = UUID(row['id'])
            id_mapping[uuid_id] = public_id
            
            metadata = row['metadata'] or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            metadata['legacy_id'] = str(row['id'])
            
            await session.execute(text(f'''
                INSERT INTO {DEST_SCHEMA}.users (
                    public_id, name, app_id, created_at, metadata
                ) VALUES (
                    :public_id, :name, :app_id, :created_at, cast(:metadata as jsonb)
                )
            '''), {
                'public_id': public_id,
                'name': row['name'],
                'app_id': apps_mapping[UUID(row['app_id'])],
                'created_at': row['created_at'],
                'metadata': json.dumps(metadata)
            })
            
            processed_ids.add(row['id'])
        
        await session.commit()
        save_processed_ids('users', processed_ids)
        save_id_mapping('users', id_mapping)
        print(f"Processed {len(processed_ids)} users")

async def migrate_sessions(session: AsyncSession) -> None:
    """Migrate sessions table"""
    print("Migrating sessions...")
    processed_ids = load_processed_ids('sessions')
    id_mapping = load_id_mapping('sessions')
    users_mapping = load_id_mapping('users')
    
    while True:
        result = await session.execute(text(f'''
            SELECT id::text, user_id::text, is_active, created_at, metadata 
            FROM {SOURCE_SCHEMA}.sessions 
            WHERE id::text NOT IN :processed_ids
            ORDER BY created_at ASC
            LIMIT :batch_size
        '''), {
            'processed_ids': tuple(processed_ids) if processed_ids else ('',),
            'batch_size': BATCH_SIZE
        })
        
        rows = result.mappings().all()
        if not rows:
            break
            
        for row in rows:
            public_id = generate_nanoid()
            uuid_id = UUID(row['id'])
            id_mapping[uuid_id] = public_id
            
            metadata = row['metadata'] or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            metadata['legacy_id'] = str(row['id'])
            
            await session.execute(text(f'''
                INSERT INTO {DEST_SCHEMA}.sessions (
                    public_id, user_id, is_active, created_at, metadata
                ) VALUES (
                    :public_id, :user_id, :is_active, :created_at, cast(:metadata as jsonb)
                )
            '''), {
                'public_id': public_id,
                'user_id': users_mapping[UUID(row['user_id'])],
                'is_active': row['is_active'],
                'created_at': row['created_at'],
                'metadata': json.dumps(metadata)
            })
            
            processed_ids.add(row['id'])
        
        await session.commit()
        save_processed_ids('sessions', processed_ids)
        save_id_mapping('sessions', id_mapping)
        print(f"Processed {len(processed_ids)} sessions")

async def migrate_messages(session: AsyncSession) -> None:
    """Migrate messages table"""
    print("Migrating messages...")
    processed_ids = load_processed_ids('messages')
    id_mapping = load_id_mapping('messages')
    sessions_mapping = load_id_mapping('sessions')
    
    while True:
        result = await session.execute(text(f'''
            SELECT id::text, session_id::text, is_user, content, created_at, metadata 
            FROM {SOURCE_SCHEMA}.messages 
            WHERE id::text NOT IN :processed_ids
            ORDER BY created_at ASC
            LIMIT :batch_size
        '''), {
            'processed_ids': tuple(processed_ids) if processed_ids else ('',),
            'batch_size': BATCH_SIZE
        })
        
        rows = result.mappings().all()
        if not rows:
            break
            
        for row in rows:
            public_id = generate_nanoid()
            uuid_id = UUID(row['id'])
            id_mapping[uuid_id] = public_id
            
            await session.execute(text(f'''
                INSERT INTO {DEST_SCHEMA}.messages (
                    public_id, session_id, is_user, content, created_at, metadata
                ) VALUES (
                    :public_id, :session_id, :is_user, :content, :created_at, cast(:metadata as jsonb)
                )
            '''), {
                'public_id': public_id,
                'session_id': sessions_mapping[UUID(row['session_id'])],
                'is_user': row['is_user'],
                'content': row['content'],
                'created_at': row['created_at'],
                'metadata': json.dumps(row['metadata'] or {})
            })
            
            processed_ids.add(row['id'])
        
        await session.commit()
        save_processed_ids('messages', processed_ids)
        save_id_mapping('messages', id_mapping)
        print(f"Processed {len(processed_ids)} messages")

async def migrate_all():
    """Run all migrations"""
    connection_uri = os.getenv('CONNECTION_URI')
    if not connection_uri:
        raise ValueError("CONNECTION_URI environment variable is not set")
    
    engine = create_db_engine(connection_uri)
    
    async with AsyncSession(engine) as session:
        # Run migrations separately
        await migrate_apps(session)
        await migrate_users(session)
        await migrate_sessions(session)
        await migrate_messages(session)
    
    print("Migration complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', choices=['apps', 'users', 'sessions', 'messages', 'all'])
    args = parser.parse_args()
    
    if args.table == 'all':
        asyncio.run(migrate_all())
    else:
        async def run_single_migration():
            connection_uri = os.getenv('CONNECTION_URI')
            if not connection_uri:
                raise ValueError("CONNECTION_URI environment variable is not set")
            
            engine = create_db_engine(connection_uri)
            async with AsyncSession(engine) as session:
                migration_func = globals()[f'migrate_{args.table}']
                await migration_func(session)
        
        asyncio.run(run_single_migration()) 