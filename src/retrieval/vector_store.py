"""Vector store using PostgreSQL with pgvector."""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
from sqlalchemy import text

from src.database.connection import db_manager
from src.utils import get_logger, load_config
from src.utils.exceptions import VectorStoreException

logger = get_logger(__name__)


@dataclass
class VectorSearchResult:
    """Vector search result."""
    id: int
    title: str
    content: str
    score: float
    rank: int
    metadata: Dict[str, Any] = None


class VectorStore:
    """PostgreSQL + pgvector interface."""
    
    def __init__(self):
        self.config = load_config()
        self.vector_config = self.config.get("retrieval", {}).get("vector_store", {})
        self.index_type = self.vector_config.get("index_type", "hnsw")
        logger.info("Initialized vector store")
    
    def insert(
        self,
        content: str,
        embedding: np.ndarray,
        title: str = None,
        metadata: Dict[str, Any] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ) -> int:
        """
        Insert a document with its embedding.
        
        Args:
            content: Document content
            embedding: Vector embedding
            title: Optional document title
            metadata: Optional metadata dict
            embedding_model: Model used to generate embedding
            
        Returns:
            Inserted document ID
        """
        try:
            # Convert numpy array to list for PostgreSQL
            emb_list = embedding.tolist()
            
            with db_manager.get_session() as session:
                query = text("""
                    INSERT INTO knowledge_base 
                    (title, content, embedding, embedding_model, metadata)
                    VALUES (:title, :content, :embedding::vector, :model, :metadata::jsonb)
                    RETURNING id
                """)
                
                result = session.execute(
                    query,
                    {
                        "title": title or "",
                        "content": content,
                        "embedding": str(emb_list),
                        "model": embedding_model,
                        "metadata": metadata or {}
                    }
                )
                
                doc_id = result.scalar()
                logger.debug(f"Inserted document with ID {doc_id}")
                return doc_id
                
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
            raise VectorStoreException(f"Insert operation failed: {e}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        metadata_filter: Dict[str, Any] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            metadata_filter: Optional metadata filters
            
        Returns:
            List of VectorSearchResults
        """
        try:
            emb_list = query_embedding.tolist()
            
            with db_manager.get_session() as session:
                # Build query with optional metadata filtering
                base_query = """
                    SELECT 
                        id, 
                        title, 
                        content,
                        metadata,
                        1 - (embedding <=> :embedding::vector) as similarity
                    FROM knowledge_base
                    WHERE is_active = true
                """
                
                # Add metadata filter if provided
                if metadata_filter:
                    base_query += " AND metadata @> :metadata::jsonb"
                
                base_query += """
                    ORDER BY embedding <=> :embedding::vector
                    LIMIT :limit
                """
                
                params = {
                    "embedding": str(emb_list),
                    "limit": top_k
                }
                
                if metadata_filter:
                    params["metadata"] = metadata_filter
                
                result = session.execute(text(base_query), params)
                rows = result.fetchall()
                
                # Build results
                results = [
                    VectorSearchResult(
                        id=row[0],
                        title=row[1],
                        content=row[2],
                        metadata=row[3],
                        score=float(row[4]),
                        rank=idx + 1
                    )
                    for idx, row in enumerate(rows)
                ]
                
                logger.debug(f"Vector search returned {len(results)} results")
                return results
                
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise VectorStoreException(f"Search operation failed: {e}")
    
    def update(self, doc_id: int, content: str = None, embedding: np.ndarray = None, metadata: Dict[str, Any] = None):
        """Update a document."""
        try:
            updates = []
            params = {"doc_id": doc_id}
            
            if content is not None:
                updates.append("content = :content")
                params["content"] = content
            
            if embedding is not None:
                updates.append("embedding = :embedding::vector")
                params["embedding"] = str(embedding.tolist())
            
            if metadata is not None:
                updates.append("metadata = :metadata::jsonb")
                params["metadata"] = metadata
            
            if not updates:
                logger.warning("No updates provided")
                return
            
            query = f"""
                UPDATE knowledge_base
                SET {', '.join(updates)}
                WHERE id = :doc_id
            """
            
            with db_manager.get_session() as session:
                session.execute(text(query), params)
                logger.debug(f"Updated document {doc_id}")
                
        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            raise VectorStoreException(f"Update operation failed: {e}")
    
    def delete(self, doc_id: int, soft_delete: bool = True):
        """Delete a document (soft or hard delete)."""
        try:
            with db_manager.get_session() as session:
                if soft_delete:
                    query = text("""
                        UPDATE knowledge_base
                        SET is_active = false
                        WHERE id = :doc_id
                    """)
                else:
                    query = text("""
                        DELETE FROM knowledge_base
                        WHERE id = :doc_id
                    """)
                
                session.execute(query, {"doc_id": doc_id})
                logger.debug(f"Deleted document {doc_id} (soft={soft_delete})")
                
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise VectorStoreException(f"Delete operation failed: {e}")
    
    def get_all_documents(self) -> List[Tuple[int, str]]:
        """Get all active documents for BM25 indexing."""
        try:
            with db_manager.get_session() as session:
                query = text("""
                    SELECT id, content
                    FROM knowledge_base
                    WHERE is_active = true
                    ORDER BY id
                """)
                
                result = session.execute(query)
                return [(row[0], row[1]) for row in result.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to fetch documents: {e}")
            raise VectorStoreException(f"Fetch operation failed: {e}")
    
    def get_document_count(self) -> int:
        """Get the count of active documents."""
        try:
            with db_manager.get_session() as session:
                query = text("""
                    SELECT COUNT(*)
                    FROM knowledge_base
                    WHERE is_active = true
                """)
                
                result = session.execute(query)
                return result.scalar()
                
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0
