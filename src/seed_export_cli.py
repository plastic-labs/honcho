import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text

from src.db import scaffold_db, engine
from src.seed_from_export import seed_from_export

async def drop_schema():
    """Drop the schema if it exists"""
    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA IF EXISTS honcho_old CASCADE"))

async def main():
    """Main function to scaffold database and seed from export"""
    load_dotenv()
    
    # Ensure we're using the right schema
    if 'DATABASE_SCHEMA' not in os.environ:
        os.environ['DATABASE_SCHEMA'] = 'honcho_old'
    
    print("Dropping existing schema...")
    await drop_schema()
    
    print("Scaffolding database...")
    scaffold_db()
    
    print("Seeding database from export...")
    await seed_from_export()
    
    print("Database seeding complete!")

if __name__ == "__main__":
    asyncio.run(main()) 