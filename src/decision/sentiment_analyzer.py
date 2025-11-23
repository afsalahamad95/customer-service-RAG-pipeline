"""Sentiment analysis using transformers."""

from typing import Dict
from transformers import pipeline

from src.utils import get_logger, load_config
from src.utils.exceptions import SentimentAnalysisException

logger = get_logger(__name__)


class SentimentAnalyzer:
    """Analyzes sentiment using BERT-based model."""
    
    def __init__(self):
        self.config = load_config()
        self.sentiment_config = self.config.get("decision", {}).get("sentiment", {})
        self.model_name = self.sentiment_config.get(
            "model",
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.escalate_on = set(self.sentiment_config.get("escalate_on", ["negative", "urgent"]))
        
        try:
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=-1  # CPU, change to 0 for GPU
            )
            logger.info(f"Loaded sentiment model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise SentimentAnalysisException(f"Model loading failed: {e}")
    
    def analyze(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            Dict with 'label' and 'score' keys
        """
        if not text:
            return {"label": "neutral", "score": 0.5}
        
        try:
            result = self.classifier(text[:512])[0]  # Truncate to model limit
            
            # Normalize label
            label = result["label"].lower()
            if "pos" in label:
                label = "positive"
            elif "neg" in label:
                label = "negative"
            else:
                label = "neutral"
            
            sentiment = {
                "label": label,
                "score": float(result["score"])
            }
            
            logger.debug(f"Sentiment: {sentiment['label']} ({sentiment['score']:.3f})")
            return sentiment
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            # Return neutral as fallback
            return {"label": "neutral", "score": 0.5}
    
    def should_escalate(self, text: str) -> bool:
        """
        Determine if text sentiment requires escalation to human agent.
        
        Args:
            text: Input text
            
        Returns:
            True if escalation needed
        """
        sentiment = self.analyze(text)
        return sentiment["label"] in self.escalate_on
    
    def is_urgent(self, text: str) -> bool:
        """Check if text indicates urgency (heuristic-based)."""
        urgent_keywords = [
            "urgent", "asap", "immediately", "emergency", "critical",
            "help", "stuck", "broken", "not working", "issue"
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in urgent_keywords)
