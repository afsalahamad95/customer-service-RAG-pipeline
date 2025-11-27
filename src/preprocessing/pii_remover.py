"""PII detection and removal using Presidio."""

from typing import List, Dict, Optional

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from ..utils import get_logger, load_config
from ..utils.exceptions import PIIDetectionException

logger = get_logger(__name__)


class PIIRemover:
    """Detects and removes PII using Microsoft Presidio."""
    
    def __init__(self):
        self.config = load_config()
        self.pii_config = self.config.get("preprocessing", {}).get("pii_removal", {})
        self.enabled = self.pii_config.get("enabled", True)
        
        if self.enabled:
            try:
                # Initialize Presidio engines
                self.analyzer = AnalyzerEngine()
                self.anonymizer = AnonymizerEngine()
                logger.info("Initialized Presidio for PII detection")
            except Exception as e:
                logger.error(f"Failed to initialize Presidio: {e}")
                raise PIIDetectionException(f"Failed to initialize PII remover: {e}")
        
        self.entities = self.pii_config.get("entities", [
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "SSN",
            "PERSON"
        ])
        
        self.redaction_strategy = self.pii_config.get("redaction_strategy", "replace")
    
    def remove_pii(self, text: str, language: str = "en") -> tuple[str, List[Dict]]:
        """
        Detect and remove PII from text.
        
        Args:
            text: Input text
            language: Language code (default: en)
            
        Returns:
            Tuple of (anonymized_text, detected_entities)
        """
        if not self.enabled or not text:
            return text, []
        
        try:
            # Analyze text for PII
            results = self.analyzer.analyze(
                text=text,
                language=language,
                entities=self.entities
            )
            
            if not results:
                return text, []
            
            # Anonymize detected PII
            operators = self._get_operator_config()
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=operators
            )
            
            # Format detected entities
            detected = [
                {
                    "entity_type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "score": result.score
                }
                for result in results
            ]
            
            logger.info(f"Detected and anonymized {len(results)} PII entities")
            return anonymized_result.text, detected
            
        except Exception as e:
            logger.error(f"PII removal failed: {e}")
            # Don't fail the entire pipeline - return original text
            return text, []
    
    def _get_operator_config(self) -> Dict:
        """Get anonymization operator configuration."""
        if self.redaction_strategy == "replace":
            return {
                "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
                "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[CREDIT_CARD]"}),
                "SSN": OperatorConfig("replace", {"new_value": "[SSN]"}),
                "PERSON": OperatorConfig("replace", {"new_value": "[NAME]"}),
            }
        elif self.redaction_strategy == "mask":
            return {
                "DEFAULT": OperatorConfig("mask", {
                    "masking_char": "*",
                    "chars_to_mask": 100,
                    "from_end": False
                })
            }
        elif self.redaction_strategy == "hash":
            return {
                "DEFAULT": OperatorConfig("hash", {"hash_type": "sha256"})
            }
        else:
            # Default to replace
            return {"DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})}
    
    def has_pii(self, text: str, language: str = "en") -> bool:
        """Check if text contains PII."""
        if not self.enabled or not text:
            return False
        
        try:
            results = self.analyzer.analyze(
                text=text,
                language=language,
                entities=self.entities
            )
            return len(results) > 0
        except Exception:
            return False
