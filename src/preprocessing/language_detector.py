"""Language detection for multilingual support."""


from langdetect import LangDetectException, detect, detect_langs

from ..utils import get_logger, load_config

logger = get_logger(__name__)


class LanguageDetector:
    """Detects text language using langdetect."""

    def __init__(self):
        self.config = load_config()
        self.lang_config = self.config.get("preprocessing", {}).get("language_detection", {})
        self.enabled = self.lang_config.get("enabled", True)
        self.supported_languages = set(self.lang_config.get("supported_languages", ["en"]))
        self.default_language = "en"

    def detect(self, text: str) -> str:
        """
        Detect the language of input text.

        Args:
            text: Input text

        Returns:
            ISO 639-1 language code (e.g., 'en', 'es', 'fr')
        """
        if not self.enabled or not text:
            return self.default_language

        # Need sufficient text for accurate detection
        if len(text.strip()) < 10:
            return self.default_language

        try:
            detected_lang = detect(text)

            # Check if detected language is supported
            if detected_lang not in self.supported_languages:
                logger.warning(
                    f"Detected unsupported language '{detected_lang}', "
                    f"defaulting to '{self.default_language}'"
                )
                return self.default_language

            logger.debug(f"Detected language: {detected_lang}")
            return detected_lang

        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}, using default")
            return self.default_language
        except Exception as e:
            logger.error(f"Unexpected error in language detection: {e}")
            return self.default_language

    def detect_with_confidence(self, text: str) -> list[dict]:
        """
        Detect language with confidence scores.

        Args:
            text: Input text

        Returns:
            List of dicts with 'lang' and 'prob' keys, sorted by probability
        """
        if not self.enabled or not text or len(text.strip()) < 10:
            return [{"lang": self.default_language, "prob": 1.0}]

        try:
            lang_probs = detect_langs(text)
            results = [{"lang": lang.lang, "prob": lang.prob} for lang in lang_probs]
            return results
        except Exception as e:
            logger.warning(f"Language detection with confidence failed: {e}")
            return [{"lang": self.default_language, "prob": 1.0}]

    def is_supported(self, language_code: str) -> bool:
        """Check if a language is supported."""
        return language_code in self.supported_languages
