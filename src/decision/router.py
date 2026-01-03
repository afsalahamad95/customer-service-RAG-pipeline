"""Routing logic for queries."""

from dataclasses import dataclass

from ..decision.confidence_scorer import ConfidenceScorer
from ..decision.sensitive_topics_detector import SensitiveTopicsDetector
from ..decision.sentiment_analyzer import SentimentAnalyzer
from ..retrieval import HybridSearchResult
from ..utils import get_logger, load_config

logger = get_logger(__name__)


@dataclass
class RoutingDecision:
    """Routing decision result."""

    route: str  # "auto_response" or "human_handoff"
    confidence: float
    sentiment: dict
    sensitive_topics: dict
    reason: str


class Router:
    """Routes queries to auto-response or human agent."""

    def __init__(self):
        self.config = load_config()
        self.routing_config = self.config.get("decision", {}).get("routing", {})
        self.auto_enabled = self.routing_config.get("auto_response_enabled", True)
        self.handoff_reasons = set(self.routing_config.get("human_handoff_reasons", []))

        # Initialize decision components
        self.confidence_scorer = ConfidenceScorer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.sensitive_detector = SensitiveTopicsDetector()

        logger.info("Initialized router")

    def route(self, query_text: str, search_results: list[HybridSearchResult]) -> RoutingDecision:
        """
        Determine routing decision for a query.

        Args:
            query_text: Original query text
            search_results: Retrieval results

        Returns:
            RoutingDecision
        """
        # Compute confidence
        confidence = self.confidence_scorer.compute_confidence(search_results)

        # Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze(query_text)
        is_urgent = self.sentiment_analyzer.is_urgent(query_text)

        # Detect sensitive topics
        sensitive_topics = self.sensitive_detector.detect(query_text)

        # Determine routing
        route = "auto_response"
        reasons = []

        # Check if auto-response is disabled
        if not self.auto_enabled:
            route = "human_handoff"
            reasons.append("auto_response_disabled")

        # Check confidence
        elif (
            "low_confidence" in self.handoff_reasons
            and not self.confidence_scorer.is_high_confidence(confidence)
        ):
            route = "human_handoff"
            reasons.append("low_confidence")

        # Check sentiment
        elif "negative_sentiment" in self.handoff_reasons and sentiment["label"] == "negative":
            route = "human_handoff"
            reasons.append("negative_sentiment")

        # Check urgency
        elif is_urgent and "urgent" in self.handoff_reasons:
            route = "human_handoff"
            reasons.append("urgency_detected")

        # Check sensitive topics
        elif "sensitive_topic" in self.handoff_reasons and sensitive_topics["has_sensitive"]:
            route = "human_handoff"
            reasons.append("sensitive_topic")

        reason = ", ".join(reasons) if reasons else "default_auto_response"

        decision = RoutingDecision(
            route=route,
            confidence=confidence,
            sentiment=sentiment,
            sensitive_topics=sensitive_topics,
            reason=reason,
        )

        logger.info(f"Routing decision: {route} (reason: {reason}, confidence: {confidence:.3f})")
        return decision

    def should_auto_respond(self, decision: RoutingDecision) -> bool:
        """Check if query should be auto-responded."""
        return decision.route == "auto_response"
