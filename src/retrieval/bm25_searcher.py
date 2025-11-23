"""BM25 keyword-based search."""

from typing import List, Tuple, Optional
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
import numpy as np

from src.utils import get_logger, load_config
from src.utils.exceptions import SearchException

logger = get_logger(__name__)


@dataclass
class BM25Result:
    """BM25 search result."""
    doc_id: int
    score: float
    rank: int


class BM25Searcher:
    """BM25 keyword searcher."""
    
    def __init__(self):
        self.config = load_config()
        self.bm25_config = self.config.get("retrieval", {}).get("bm25", {})
        self.enabled = self.bm25_config.get("enabled", True)
        
        self.k1 = self.bm25_config.get("k1", 1.5)
        self.b = self.bm25_config.get("b", 0.75)
        
        self.corpus = []
        self.doc_ids = []
        self.bm25 = None
        
        logger.info("Initialized BM25 searcher")
    
    def index_documents(self, documents: List[Tuple[int, str]]):
        """
        Index documents for BM25 search.
        
        Args:
            documents: List of (doc_id, text) tuples
        """
        if not documents:
            logger.warning("No documents to index")
            return
        
        try:
            self.doc_ids, texts = zip(*documents)
            self.doc_ids = list(self.doc_ids)
            
            # Tokenize documents (simple whitespace tokenization)
            self.corpus = [text.lower().split() for text in texts]
            
            # Create BM25 index
            self.bm25 = BM25Okapi(self.corpus, k1=self.k1, b=self.b)
            
            logger.info(f"Indexed {len(documents)} documents for BM25 search")
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise SearchException(f"BM25 indexing failed: {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[BM25Result]:
        """
        Search for relevant documents using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of BM25Results
        """
        if not self.enabled or not self.bm25:
            return []
        
        if not query:
            return []
        
        try:
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            # Build results
            results = [
                BM25Result(
                    doc_id=self.doc_ids[idx],
                    score=float(scores[idx]),
                    rank=rank + 1
                )
                for rank, idx in enumerate(top_indices)
                if scores[idx] > 0  # Only include positive scores
            ]
            
            logger.debug(f"BM25 search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            raise SearchException(f"BM25 search error: {e}")
    
    def update_document(self, doc_id: int, new_text: str):
        """Update a single document in the index."""
        if doc_id not in self.doc_ids:
            logger.warning(f"Document {doc_id} not found in index")
            return
        
        idx = self.doc_ids.index(doc_id)
        self.corpus[idx] = new_text.lower().split()
        
        # Rebuild BM25 index
        self.bm25 = BM25Okapi(self.corpus, k1=self.k1, b=self.b)
        logger.debug(f"Updated document {doc_id}")
    
    def remove_document(self, doc_id: int):
        """Remove a document from the index."""
        if doc_id not in self.doc_ids:
            logger.warning(f"Document {doc_id} not found in index")
            return
        
        idx = self.doc_ids.index(doc_id)
        del self.corpus[idx]
        del self.doc_ids[idx]
        
        # Rebuild BM25 index
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus, k1=self.k1, b=self.b)
        else:
            self.bm25 = None
        
        logger.debug(f"Removed document {doc_id}")
    
    def get_document_count(self) -> int:
        """Get the number of indexed documents."""
        return len(self.doc_ids)
