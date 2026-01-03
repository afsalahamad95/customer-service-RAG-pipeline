"""Sensitive topic detection."""


from ..utils import get_logger, load_config

logger = get_logger(__name__)


class SensitiveTopicsDetector:
    """Detects sensitive topics requiring human agent handling."""

    def __init__(self):
        self.config = load_config()
        self.sensitive_config = self.config.get("decision", {}).get("sensitive_topics", {})
        self.threshold = self.sensitive_config.get("threshold", 0.8)

        # Load keyword lists from config
        keywords_config = self.sensitive_config.get("keywords", {})
        self.payment_keywords = set(keywords_config.get("payment", []))
        self.legal_keywords = set(keywords_config.get("legal", []))
        self.account_keywords = set(keywords_config.get("account", []))

        logger.info("Initialized sensitive topics detector")

    def detect(self, text: str) -> dict:
        """
        Detect sensitive topics in text.

        Args:
            text: Input text

        Returns:
            Dict with detected topics and confidence
        """
        if not text:
            return {"has_sensitive": False, "topics": [], "confidence": 0.0}

        text_lower = text.lower()
        detected_topics = []

        # Check payment-related
        if self._check_keywords(text_lower, self.payment_keywords):
            detected_topics.append("payment")

        # Check legal-related
        if self._check_keywords(text_lower, self.legal_keywords):
            detected_topics.append("legal")

        # Check account/credential-related
        if self._check_keywords(text_lower, self.account_keywords):
            detected_topics.append("account_security")

        # Compute confidence based on keyword density
        total_keywords = (
            len(self.payment_keywords) + len(self.legal_keywords) + len(self.account_keywords)
        )
        if total_keywords > 0:
            matches = sum(
                [
                    self._count_matches(text_lower, self.payment_keywords),
                    self._count_matches(text_lower, self.legal_keywords),
                    self._count_matches(text_lower, self.account_keywords),
                ]
            )
            confidence = min(1.0, matches / 5.0)  # Normalize
        else:
            confidence = 0.0

        result = {
            "has_sensitive": len(detected_topics) > 0,
            "topics": detected_topics,
            "confidence": confidence,
        }

        if result["has_sensitive"]:
            logger.info(f"Detected sensitive topics: {detected_topics}")

        return result

    def _check_keywords(self, text: str, keywords: set[str]) -> bool:
        """Check if any keyword is present in text."""
        return any(keyword in text for keyword in keywords)

    def _count_matches(self, text: str, keywords: set[str]) -> int:
        """Count keyword matches in text."""
        return sum(1 for keyword in keywords if keyword in text)

    def requires_human(self, text: str) -> bool:
        """Determine if text requires human agent handling."""
        result = self.detect(text)
        return result["has_sensitive"] and result["confidence"] >= self.threshold
