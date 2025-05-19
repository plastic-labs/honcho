from sqlalchemy import text, select
import ollama
import sys
import asyncio
from pathlib import Path
import os
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

from src.dependencies import tracked_db
from src import models

load_dotenv()

schema = os.getenv("DATABASE_SCHEMA", "public")
client = ollama.Client()

# Set up parameters
model = "nomic-embed-text:latest"
dimensions = 768

async def run_explain():
    prompt_text = "This is a dummy prompt for testing embeddings."

    response = client.embeddings(
        model=model,
        prompt=prompt_text,
        options={"dimensions": dimensions}
    )
    embedding = response['embedding']
    
    # Build query using SQLAlchemy
    stmt = (
        select(
            models.Document.id,
            models.Document.content,
            1 - models.Document.embedding.cosine_distance(embedding).label('cosine_similarity')
        )
        .where(models.Document.collection_id == "SZzP8vxCB9Y8WrKhXAKWp")
        .where(models.Document.embedding.cosine_distance(embedding) < 0.3)
        .order_by(models.Document.embedding.cosine_distance(embedding))
        .limit(5)
    )
    
    # Wrap in EXPLAIN ANALYZE
    explain_stmt = text(f"EXPLAIN ANALYZE {stmt.compile(compile_kwargs={'literal_binds': True})}")

    async with tracked_db("explain_query") as db:
        result = await db.execute(explain_stmt)
        explain_output = result.fetchall()
        for row in explain_output:
            print(row[0])

if __name__ == "__main__":
    asyncio.run(run_explain())
