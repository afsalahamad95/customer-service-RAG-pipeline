"""Script to initialize the knowledge base with sample data."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import VectorStore, EmbeddingService
from src.utils import get_logger
from data.sample_kb import get_sample_data

logger = get_logger(__name__)


async def initialize_kb():
    """Initialize knowledge base with sample data."""
    logger.info("Initializing knowledge base...")
    
    # Initialize services
    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    
    # Get sample data
    documents = get_sample_data()
    
    logger.info(f"Ingesting {len(documents)} documents...")
    
    for i, doc in enumerate(documents):
        # Generate embedding
        embedding = embedding_service.embed(doc["content"])
        
        # Insert into vector store
        doc_id = vector_store.insert(
            content=doc["content"],
            embedding=embedding.flatten(),
            title=doc["title"],
            metadata=doc["metadata"]
        )
        
        logger.info(f"Inserted document {i+1}/{len(documents)}: {doc['title']} (ID: {doc_id})")
    
    logger.info("Knowledge base initialization complete!")
    print(f"\n✅ Successfully initialized KB with {len(documents)} documents")


if __name__ == "__main__":
    asyncio.run(initialize_kb())
