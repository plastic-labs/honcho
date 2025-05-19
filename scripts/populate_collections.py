# 45k documents
# 3k collections
# generate 15 documents per collection
import os
import sys
import asyncio
from pathlib import Path
import concurrent.futures
from sqlalchemy import text

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import ollama
from src import models, schemas
from src.crud import create_collection, create_document
from src.dependencies import tracked_db

import random
import string
import time
from nanoid import generate as generate_nanoid

# Vocabulary for sentence generation
subjects = ["The researcher", "A scientist", "The algorithm", "That model", "My colleague", "The system", 
            "The student", "Our team", "The device", "This network", "The computer", "The project",
            "The engineer", "The database", "The application", "The framework", "The professor"]

verbs = ["analyzes", "generates", "processes", "examines", "transforms", "calculates", "identifies",
         "explores", "visualizes", "optimizes", "implements", "discovers", "predicts", "classifies",
         "extracts", "validates", "simulates", "monitors", "integrates", "constructs"]

objects = ["the dataset", "complex patterns", "various inputs", "semantic relationships", "numerical features",
           "statistical anomalies", "embedding vectors", "linguistic structures", "dimensional representations",
           "hidden correlations", "informational content", "categorical variables", "textual similarities",
           "network topology", "gradient descent", "neural activations"]

adverbs = ["quickly", "efficiently", "accurately", "thoroughly", "carefully", "precisely", "automatically",
           "reliably", "systematically", "consistently", "effectively", "intelligently", ""]

contexts = ["in high-dimensional spaces", "using advanced algorithms", "with mathematical precision",
            "in the research environment", "for analytical purposes", "in real-time systems", 
            "within specified parameters", "across multiple domains", "through iterative processing",
            "during the experiment", "in production environments", "using vector representations", ""]

# Add uniqueness generators
tech_terms = ["tensor", "matrix", "vector", "graph", "neural", "semantic", "synthetic", "quantum", 
              "binary", "digital", "analog", "parallel", "distributed", "cloud", "edge", "mobile"]

domains = ["healthcare", "finance", "logistics", "education", "manufacturing", "telecommunications",
           "agriculture", "retail", "energy", "transportation", "security", "entertainment"]

def generate_random_id():
    """Generate a random alphanumeric ID."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

def generate_random_sentence():
    """Generate a random grammatically correct sentence with high variance."""
    subject = random.choice(subjects)
    verb = random.choice(verbs)
    obj = random.choice(objects)
    adverb = random.choice(adverbs)
    context = random.choice(contexts)
    
    # Add unique elements to some sentences
    if random.random() > 0.7:
        tech_term = random.choice(tech_terms)
        domain = random.choice(domains)
        unique_id = generate_random_id()
        obj = f"{tech_term}-based {obj} for {domain} ({unique_id})"
    
    # Vary sentence structure for more uniqueness
    templates = [
        f"{subject} {adverb} {verb} {obj} {context}.",
        f"{context}, {subject} {verb} {obj}.",
        f"{subject} {verb} {obj} {adverb}.",
        f"Using {obj}, {subject} {verb} {context}.",
        f"{adverb.capitalize()}, {subject} {verb} {obj}."
    ]
    
    sentence = random.choice(templates)
    
    # Clean up extra spaces
    return " ".join(sentence.split())

def generate_document(doc_id):
    """Generate a single document with 1-3 sentences and guaranteed uniqueness."""
    num_sentences = random.randint(1, 3)
    sentences = [generate_random_sentence() for _ in range(num_sentences)]
    
    # Add unique identifier to ensure distinctness
    unique_marker = f"Document {doc_id} with unique signature {generate_random_id()}:"
    document = unique_marker + " " + " ".join(sentences)
    
    return document

load_dotenv()

schema = os.getenv("DATABASE_SCHEMA", "public")

app_id = '0xqLC4GGU5rrJgF_RllgW'
user_id = 'ntHjchkXfSs2ZDf4ZYrG2'

client = ollama.Client()

# Set up parameters
model = "nomic-embed-text:latest"
text = "This is a dummy prompt for testing embeddings."
dimensions = 768

# Create a thread pool for embedding generation
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def generate_embedding(text):
    """Generate embedding in a separate thread"""
    response = client.embeddings(
        model=model,
        prompt=text,
        options={"dimensions": dimensions}
    )
    return response['embedding']

async def generate_documents_batch(collection_id, batch_size=150):
    """Generate a batch of documents with embeddings in parallel"""
    # Generate all texts first
    texts = [generate_document(generate_nanoid()) for _ in range(batch_size)]
    
    # Generate embeddings in parallel using the thread pool
    loop = asyncio.get_event_loop()
    embedding_tasks = [
        loop.run_in_executor(executor, generate_embedding, text)
        for text in texts
    ]
    embeddings = await asyncio.gather(*embedding_tasks)
    
    # Create documents
    documents = [
        models.Document(
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
            public_id=generate_nanoid(),  # Generate unique public_id for each document
            content=text,
            h_metadata={},
            embedding=embedding,
        )
        for text, embedding in zip(texts, embeddings)
    ]
    
    return documents

async def main():
    print("Starting bulk collection and document creation")
    
    for i in range(3000):
        collection_name = generate_nanoid()
        # Create collection and its documents in a single transaction
        async with tracked_db("create_collection") as db:
            # Create collection
            collection = await create_collection(db, app_id=app_id, user_id=user_id, collection=schemas.CollectionCreate(name=collection_name))
            
            # Generate documents with parallel embedding generation
            documents = await generate_documents_batch(collection.public_id)
            
            # Bulk insert documents
            db.add_all(documents)
            await db.commit()
            
        if i % 10 == 0:  # Log progress every 10 collections
            print(f"Completed {i} collections")

    executor.shutdown()  # Clean up the thread pool

if __name__ == "__main__":
    asyncio.run(main())

