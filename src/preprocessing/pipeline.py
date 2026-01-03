"""Preprocessing pipeline orchestrator."""

from dataclasses import dataclass, field
from typing import Any

from ..preprocessing.language_detector import LanguageDetector
from ..preprocessing.pii_remover import PIIRemover
from ..preprocessing.text_cleaner import TextCleaner
from ..preprocessing.tokenizer import Tokenizer
from ..utils import get_logger
from ..utils.exceptions import PreprocessingException

logger = get_logger(__name__)


@dataclass
class PreprocessingResult:
    """Result of preprocessing pipeline."""

    original_text: str
    cleaned_text: str
    anonymized_text: str
    language: str
    detected_pii: list = field(default_factory=list)
    tokens: dict | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PreprocessingPipeline:
    """Orchestrates all preprocessing steps."""

    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.pii_remover = PIIRemover()
        self.language_detector = LanguageDetector()
        self.tokenizer = Tokenizer()
        logger.info("Initialized preprocessing pipeline")

    def process(
        self, text: str, include_tokenization: bool = False, detect_pii: bool = True
    ) -> PreprocessingResult:
        """
        Run full preprocessing pipeline.

        Args:
            text: Input text
            include_tokenization: Whether to tokenize the result
            detect_pii: Whether to detect and remove PII

        Returns:
            PreprocessingResult with all processed versions
        """
        if not text or not isinstance(text, str):
            raise PreprocessingException("Invalid input text")

        try:
            logger.debug(f"Processing text: {text[:100]}...")

            # Step 1: Text cleaning
            cleaned = self.text_cleaner.clean(text)
            logger.debug("Text cleaning completed")

            # Step 2: Language detection
            language = self.language_detector.detect(cleaned)
            logger.debug(f"Detected language: {language}")

            # Step 3: PII detection and removal
            anonymized = cleaned
            detected_pii = []
            if detect_pii:
                anonymized, detected_pii = self.pii_remover.remove_pii(cleaned, language)
                logger.debug(f"PII detection completed, found {len(detected_pii)} entities")

            # Step 4: Tokenization (optional)
            tokens = None
            if include_tokenization:
                tokens = self.tokenizer.tokenize(anonymized)
                logger.debug("Tokenization completed")

            # Build result
            result = PreprocessingResult(
                original_text=text,
                cleaned_text=cleaned,
                anonymized_text=anonymized,
                language=language,
                detected_pii=detected_pii,
                tokens=tokens,
                metadata={
                    "text_length": len(text),
                    "cleaned_length": len(cleaned),
                    "anonymized_length": len(anonymized),
                    "pii_count": len(detected_pii),
                },
            )

            logger.info("Preprocessing pipeline completed successfully")
            return result

        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {e}")
            raise PreprocessingException(f"Pipeline processing failed: {e}")

    def batch_process(
        self, texts: list[str], include_tokenization: bool = False, detect_pii: bool = True
    ) -> list[PreprocessingResult]:
        """
        Process multiple texts.

        Args:
            texts: List of input texts
            include_tokenization: Whether to tokenize results
            detect_pii: Whether to detect and remove PII

        Returns:
            List of PreprocessingResults
        """
        results = []
        for text in texts:
            try:
                result = self.process(text, include_tokenization, detect_pii)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process text in batch: {e}")
                # Continue with next text
                continue

        logger.info(f"Batch processed {len(results)}/{len(texts)} texts successfully")
        return results
