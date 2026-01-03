"""Text cleaning using spaCy."""

import re

import spacy

from ..utils import get_logger, load_config
from ..utils.exceptions import TextCleaningException

logger = get_logger(__name__)


class TextCleaner:
    """Cleans and normalizes text using spaCy."""

    def __init__(self):
        self.config = load_config()
        self.cleaning_config = self.config.get("preprocessing", {}).get("text_cleaning", {})
        self.enabled = self.cleaning_config.get("enabled", True)

        try:
            # Load spaCy model for text processing
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            logger.info("Loaded spaCy model for text cleaning")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise TextCleaningException(f"Failed to initialize text cleaner: {e}")

    def clean(self, text: str) -> str:
        """
        Clean and normalize input text.

        Args:
            text: Raw input text

        Returns:
            Cleaned text
        """
        if not self.enabled:
            return text

        if not text or not isinstance(text, str):
            return ""

        try:
            cleaned = text

            # Remove HTML tags
            if self.cleaning_config.get("remove_html", True):
                cleaned = self._remove_html(cleaned)

            # Normalize whitespace
            if self.cleaning_config.get("normalize_whitespace", True):
                cleaned = self._normalize_whitespace(cleaned)

            # Optional lowercase (usually not recommended for NER/sentiment)
            if self.cleaning_config.get("lowercase", False):
                cleaned = cleaned.lower()

            # Remove extra punctuation
            cleaned = self._clean_punctuation(cleaned)

            # Fix common typos (basic)
            cleaned = self._fix_common_typos(cleaned)

            return cleaned.strip()

        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            raise TextCleaningException(f"Failed to clean text: {e}")

    def _remove_html(self, text: str) -> str:
        """Remove HTML/markdown tags."""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove markdown links
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        # Remove markdown formatting
        text = re.sub(r"[*_`~]", "", text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text

    def _clean_punctuation(self, text: str) -> str:
        """Clean excessive punctuation."""
        # Replace multiple punctuation with single
        text = re.sub(r"([!?.]){2,}", r"\1", text)
        return text

    def _fix_common_typos(self, text: str) -> str:
        """Fix common typos and abbreviations."""
        replacements = {
            r"\bu\b": "you",
            r"\bur\b": "your",
            r"\br\b": "are",
            r"\bpls\b": "please",
            r"\bthx\b": "thanks",
            r"\btho\b": "though",
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        doc = self.nlp(text)
        return [token.text for token in doc]
