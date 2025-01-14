import asyncio
import json
import os
from typing import Dict
from uuid import UUID

from nanoid import generate as generate_nanoid
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from dotenv import load_dotenv

load_dotenv()

SOURCE_SCHEMA = 'honcho_old'
DEST_SCHEMA = 'honcho_new'

def create_db_engine(url: str) -> AsyncEngine:
    """Create an async database engine from a connection URL"""
    if url.startswith('postgresql://'):
        url = url.replace('postgresql://', 'postgresql+asyncpg://', 1)
    return create_async_engine(url, echo=False, pool_pre_ping=True)

async def migrate_data():
    """Migrate data between schemas in the same database"""
    print("Starting schema migration...")
    print(f"From: {SOURCE_SCHEMA} schema")
    print(f"To: {DEST_SCHEMA} schema")
    
    connection_uri = os.getenv('CONNECTION_URI')
    if not connection_uri:
        raise ValueError("CONNECTION_URI environment variable is not set")
    
    engine = create_db_engine(connection_uri)
    
    async with AsyncSession(engine) as session:
        async with session.begin():
            await migrate_schemas(session)
    
    print("Migration complete!")

async def migrate_schemas(session: AsyncSession):
    """Migrate data between schemas"""
    id_mappings: Dict[str, Dict[UUID, str]] = {
        'apps': {},
        'users': {},
        'sessions': {},
        'messages': {}
    }
    
    # Migrate apps
    print("Migrating apps...")
    result = await session.execute(text(f'''
        SELECT id::text, name, created_at, metadata 
        FROM {SOURCE_SCHEMA}.apps 
        ORDER BY created_at ASC
    '''))
    for row in result.mappings():
        public_id = generate_nanoid()
        id_mappings['apps'][UUID(row['id'])] = public_id
        
        await session.execute(text(f'''
            INSERT INTO {DEST_SCHEMA}.apps (
                public_id, 
                name, 
                created_at, 
                metadata
            ) VALUES (
                :public_id, 
                :name, 
                :created_at, 
                cast(:metadata as jsonb)
            )
        '''), {
            'public_id': public_id,
            'name': row['name'],
            'created_at': row['created_at'],
            'metadata': json.dumps(row['metadata'] or {})
        })

    # Migrate users
    print("Migrating users...")
    result = await session.execute(text(f'''
        SELECT id::text, name, app_id::text, created_at, metadata 
        FROM {SOURCE_SCHEMA}.users 
        ORDER BY created_at ASC
    '''))
    for row in result.mappings():
        public_id = generate_nanoid()
        id_mappings['users'][UUID(row['id'])] = public_id
        
        metadata = row['metadata'] or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        metadata.update({
            'legacy_id': str(row['id'])
        })
        
        await session.execute(text(f'''
            INSERT INTO {DEST_SCHEMA}.users (public_id, name, app_id, created_at, metadata)
            VALUES (:public_id, :name, :app_id, :created_at, cast(:metadata as jsonb))
        '''), {
            'public_id': public_id,
            'name': row['name'],
            'app_id': id_mappings['apps'][UUID(row['app_id'])],
            'created_at': row['created_at'],
            'metadata': json.dumps(metadata)
        })

    # Migrate sessions
    print("Migrating sessions...")
    result = await session.execute(text(f'''
        SELECT id::text, user_id::text, is_active, created_at, metadata 
        FROM {SOURCE_SCHEMA}.sessions 
        ORDER BY created_at ASC
    '''))
    for row in result.mappings():
        public_id = generate_nanoid()
        id_mappings['sessions'][UUID(row['id'])] = public_id
        
        metadata = row['metadata'] or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        metadata.update({
            'legacy_id': str(row['id'])
        })
        
        await session.execute(text(f'''
            INSERT INTO {DEST_SCHEMA}.sessions (
                public_id, 
                user_id, 
                is_active, 
                created_at, 
                metadata
            ) VALUES (
                :public_id, 
                :user_id, 
                :is_active, 
                :created_at, 
                cast(:metadata as jsonb)
            )
        '''), {
            'public_id': public_id,
            'user_id': id_mappings['users'][UUID(row['user_id'])],
            'is_active': row['is_active'],
            'created_at': row['created_at'],
            'metadata': json.dumps(metadata)
        })

    # Migrate messages
    print("Migrating messages...")
    result = await session.execute(text(f'''
        SELECT id::text, session_id::text, is_user, content, created_at, metadata 
        FROM {SOURCE_SCHEMA}.messages 
        ORDER BY created_at ASC
    '''))
    for row in result.mappings():
        public_id = generate_nanoid()
        id_mappings['messages'][UUID(row['id'])] = public_id
        
        await session.execute(text(f'''
            INSERT INTO {DEST_SCHEMA}.messages (public_id, session_id, is_user, content, created_at, metadata)
            VALUES (:public_id, :session_id, :is_user, :content, :created_at, cast(:metadata as jsonb))
        '''), {
            'public_id': public_id,
            'session_id': id_mappings['sessions'][UUID(row['session_id'])],
            'is_user': row['is_user'],
            'content': row['content'],
            'created_at': row['created_at'],
            'metadata': json.dumps(row['metadata'] or {})
        })

if __name__ == "__main__":
    asyncio.run(migrate_data()) 