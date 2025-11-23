"""Tokenization for various models."""

from typing import List, Optional

from transformers import AutoTokenizer

from src.utils import get_logger, load_config
from src.utils.exceptions import PreprocessingException

logger = get_logger(__name__)


class Tokenizer:
    """Tokenizes text for downstream models."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.config = load_config()
        self.tokenization_config = self.config.get("preprocessing", {}).get("tokenization", {})
        
        # Use provided model or default from config
        self.model_name = model_name or self.tokenization_config.get(
            "model",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.max_length = self.tokenization_config.get("max_length", 512)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"Loaded tokenizer: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise PreprocessingException(f"Failed to initialize tokenizer: {e}")
    
    def tokenize(
        self,
        text: str,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = False,
        return_tensors: Optional[str] = None
    ) -> dict:
        """
        Tokenize text.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            truncation: Whether to truncate long sequences
            padding: Whether to pad sequences
            return_tensors: Return format ('pt' for PyTorch, 'tf' for TensorFlow)
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        if not text:
            return {}
        
        max_len = max_length or self.max_length
        
        try:
            tokens = self.tokenizer(
                text,
                max_length=max_len,
                truncation=truncation,
                padding=padding,
                return_tensors=return_tensors
            )
            return tokens
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise PreprocessingException(f"Failed to tokenize text: {e}")
    
    def batch_tokenize(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = True,
        return_tensors: Optional[str] = None
    ) -> dict:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            truncation: Whether to truncate long sequences
            padding: Whether to pad sequences
            return_tensors: Return format
            
        Returns:
            Dictionary with batched tokens
        """
        if not texts:
            return {}
        
        max_len = max_length or self.max_length
        
        try:
            tokens = self.tokenizer(
                texts,
                max_length=max_len,
                truncation=truncation,
                padding=padding,
                return_tensors=return_tensors
            )
            return tokens
        except Exception as e:
            logger.error(f"Batch tokenization failed: {e}")
            raise PreprocessingException(f"Failed to tokenize batch: {e}")
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            raise PreprocessingException(f"Failed to decode tokens: {e}")
    
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in text."""
        tokens = self.tokenize(text)
        return len(tokens.get("input_ids", []))
