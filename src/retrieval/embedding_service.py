"""Embedding generation service."""

from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.utils import get_logger, load_config
from src.utils.exceptions import EmbeddingException

logger = get_logger(__name__)


class EmbeddingService:
    """Generates embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = None):
        self.config = load_config()
        self.embedding_config = self.config.get("retrieval", {}).get("embedding", {})
        
        # Use provided model or default from config
        if model_name:
            self.model_name = model_name
        else:
            default_model = self.embedding_config.get("default", "fast")
            models = self.embedding_config.get("models", {})
            self.model_name = models.get(
                default_model,
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        
        self.batch_size = self.embedding_config.get("batch_size", 32)
        self.device = self._get_device()
        
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"Loaded embedding model with dimension {self.embedding_dim} "
                f"on device {self.device}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise EmbeddingException(f"Model loading failed: {e}")
    
    def _get_device(self) -> str:
        """Determine the best device to use."""
        device_config = self.embedding_config.get("device", "mps")
        
        # Check MPS (Metal Performance Shaders) for M4 Pro
        if device_config == "mps" and torch.backends.mps.is_available():
            return "mps"
        
        # Check CUDA
        if torch.cuda.is_available():
            return "cuda"
        
        # Fallback to CPU
        return "cpu"
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            text: Single text or list of texts
            
        Returns:
            Numpy array of embeddings (shape: [1, dim] or [n, dim])
        """
        if not text:
            raise EmbeddingException("Empty text provided")
        
        try:
            # Ensure text is a list
            is_single = isinstance(text, str)
            texts = [text] if is_single else text
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            logger.debug(f"Generated embeddings for {len(texts)} text(s)")
            
            # Return same shape as input (single or batch)
            if is_single:
                return embeddings[0:1]  # Keep 2D shape [1, dim]
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingException(f"Failed to generate embeddings: {e}")
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Numpy array of shape [n, embedding_dim]
        """
        return self.embed(texts)
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        # Normalize if not already
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-9)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-9)
        
        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        return float(similarity)
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim
