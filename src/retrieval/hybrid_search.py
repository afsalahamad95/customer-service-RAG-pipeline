"""Hybrid search combining BM25 and semantic search."""

from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np

from src.retrieval.bm25_searcher import BM25Searcher, BM25Result
from src.retrieval.embedding_service import EmbeddingService
from src.retrieval.vector_store import VectorStore, VectorSearchResult
from src.utils import get_logger, load_config
from src.utils.exceptions import SearchException

logger = get_logger(__name__)


@dataclass
class HybridSearchResult:
    """Hybrid search result."""
    doc_id: int
    title: str
    content: str
    combined_score: float
    bm25_score: float
    semantic_score: float
    rank: int
    metadata: Dict[str, Any] = None


class HybridSearch:
    """Combines BM25 keyword search and semantic vector search."""
    
    def __init__(self):
        self.config = load_config()
        self.hybrid_config = self.config.get("retrieval", {}).get("hybrid_search", {})
        self.enabled = self.hybrid_config.get("enabled", True)
        self.alpha = self.hybrid_config.get("alpha", 0.5)  # 0=BM25 only, 1=semantic only
        self.top_k = self.hybrid_config.get("top_k", 5)
        self.fusion_method = self.hybrid_config.get("fusion_method", "rrf")
        
        # Initialize components
        self.bm25_searcher = BM25Searcher()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        
        logger.info(f"Initialized hybrid search (alpha={self.alpha}, fusion={self.fusion_method})")
    
    def index_documents(self):
        """
        Index all documents from vector store into BM25.
        Call this after adding documents to the vector store.
        """
        try:
            documents = self.vector_store.get_all_documents()
            self.bm25_searcher.index_documents(documents)
            logger.info(f"Indexed {len(documents)} documents for hybrid search")
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise SearchException(f"Indexing failed: {e}")
    
    def search(
        self,
        query: str,
        top_k: int = None,
        alpha: float = None,
        metadata_filter: Dict[str, Any] = None
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining BM25 and semantic search.
        
        Args:
            query: Search query
            top_k: Number of results to return (defaults to config value)
            alpha: Weight for semantic vs BM25 (0-1, defaults to config value)
            metadata_filter: Optional metadata filter for vector search
            
        Returns:
            List of HybridSearchResults
        """
        if not self.enabled:
            logger.warning("Hybrid search is disabled")
            return []
        
        if not query:
            return []
        
        k = top_k or self.top_k
        weight = alpha if alpha is not None else self.alpha
        
        try:
            # Get BM25 results
            bm25_results = self.bm25_searcher.search(query, top_k=k * 2)
            
            # Get semantic search results
            query_embedding = self.embedding_service.embed(query)
            vector_results = self.vector_store.search(
                query_embedding.flatten(),
                top_k=k * 2,
                metadata_filter=metadata_filter
            )
            
            # Combine results using fusion method
            if self.fusion_method == "rrf":
                combined = self._reciprocal_rank_fusion(bm25_results, vector_results, weight)
            else:
                combined = self._weighted_score_fusion(bm25_results, vector_results, weight)
            
            # Sort by combined score and take top-k
            combined.sort(key=lambda x: x.combined_score, reverse=True)
            results = combined[:k]
            
            # Update ranks
            for i, result in enumerate(results):
                result.rank = i + 1
            
            logger.info(f"Hybrid search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise SearchException(f"Search operation failed: {e}")
    
    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[BM25Result],
        vector_results: List[VectorSearchResult],
        alpha: float,
        k: int = 60
    ) -> List[HybridSearchResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF score = alpha * (1/(k + semantic_rank)) + (1-alpha) * (1/(k + bm25_rank))
        """
        # Build score mappings
        bm25_scores = {r.doc_id: 1.0 / (k + r.rank) for r in bm25_results}
        vector_scores = {r.id: 1.0 / (k + r.rank) for r in vector_results}
        
        # Get content mapping from vector results
        content_map = {r.id: (r.title, r.content, r.metadata) for r in vector_results}
        
        # Combine all doc IDs
        all_doc_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
        
        combined = []
        for doc_id in all_doc_ids:
            bm25_score = bm25_scores.get(doc_id, 0)
            semantic_score = vector_scores.get(doc_id, 0)
            
            # RRF combination
            combined_score = alpha * semantic_score + (1 - alpha) * bm25_score
            
            # Get content (prefer vector results as they have more info)
            if doc_id in content_map:
                title, content, metadata = content_map[doc_id]
            else:
                # Fallback: fetch from DB if needed
                title, content, metadata = "", "", {}
            
            combined.append(HybridSearchResult(
                doc_id=doc_id,
                title=title,
                content=content,
                combined_score=combined_score,
                bm25_score=bm25_score,
                semantic_score=semantic_score,
                rank=0,  # Will be set later
                metadata=metadata
            ))
        
        return combined
    
    def _weighted_score_fusion(
        self,
        bm25_results: List[BM25Result],
        vector_results: List[VectorSearchResult],
        alpha: float
    ) -> List[HybridSearchResult]:
        """
        Combine results using weighted score fusion.
        
        Combined score = alpha * semantic_score + (1-alpha) * normalized_bm25_score
        """
        # Normalize BM25 scores
        bm25_scores = {r.doc_id: r.score for r in bm25_results}
        if bm25_scores:
            max_bm25 = max(bm25_scores.values())
            if max_bm25 > 0:
                bm25_scores = {k: v / max_bm25 for k, v in bm25_scores.items()}
        
        # Vector scores are already normalized (cosine similarity)
        vector_scores = {r.id: r.score for r in vector_results}
        
        # Get content mapping
        content_map = {r.id: (r.title, r.content, r.metadata) for r in vector_results}
        
        # Combine all doc IDs
        all_doc_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
        
        combined = []
        for doc_id in all_doc_ids:
            bm25_score = bm25_scores.get(doc_id, 0)
            semantic_score = vector_scores.get(doc_id, 0)
            
            # Weighted combination
            combined_score = alpha * semantic_score + (1 - alpha) * bm25_score
            
            if doc_id in content_map:
                title, content, metadata = content_map[doc_id]
            else:
                title, content, metadata = "", "", {}
            
            combined.append(HybridSearchResult(
                doc_id=doc_id,
                title=title,
                content=content,
                combined_score=combined_score,
                bm25_score=bm25_score,
                semantic_score=semantic_score,
                rank=0,
                metadata=metadata
            ))
        
        return combined
