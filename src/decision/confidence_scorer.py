"""Confidence scoring for retrieval results."""


import numpy as np

from ..retrieval import HybridSearchResult
from ..utils import get_logger, load_config

logger = get_logger(__name__)


class ConfidenceScorer:
    """Computes confidence scores for query responses."""

    def __init__(self):
        self.config = load_config()
        self.conf_config = self.config.get("decision", {}).get("confidence", {})
        self.threshold = self.conf_config.get("threshold", 0.7)
        self.min_retrieval_score = self.conf_config.get("min_retrieval_score", 0.3)
        self.consistency_weight = self.conf_config.get("consistency_weight", 0.3)
        logger.info(f"Initialized confidence scorer (threshold={self.threshold})")

    def compute_confidence(self, search_results: list[HybridSearchResult]) -> float:
        """
        Compute confidence score based on retrieval results.

        Args:
            search_results: List of hybrid search results

        Returns:
            Confidence score (0-1)
        """
        if not search_results:
            return 0.0

        try:
            # Factor 1: Top result score (40% weight)
            top_score = search_results[0].combined_score

            # Factor 2: Score consistency across top-k (30% weight)
            scores = [r.combined_score for r in search_results[:3]]
            if len(scores) > 1:
                consistency = 1.0 - np.std(scores) / (np.mean(scores) + 1e-9)
                consistency = max(0, min(1, consistency))
            else:
                consistency = 1.0

            # Factor 3: Minimum threshold check (30% weight)
            above_threshold = sum(
                1 for r in search_results if r.combined_score >= self.min_retrieval_score
            )
            threshold_score = min(1.0, above_threshold / 3.0)

            # Combine factors
            confidence = 0.4 * top_score + 0.3 * consistency + 0.3 * threshold_score

            logger.debug(
                f"Confidence: {confidence:.3f} "
                f"(top={top_score:.3f}, consistency={consistency:.3f}, threshold={threshold_score:.3f})"
            )

            return min(1.0, max(0.0, confidence))

        except Exception as e:
            logger.error(f"Confidence scoring failed: {e}")
            return 0.0

    def is_high_confidence(self, confidence: float) -> bool:
        """Check if confidence exceeds threshold."""
        return confidence >= self.threshold

    def get_confidence_level(self, confidence: float) -> str:
        """Get confidence level as a string."""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "low"
