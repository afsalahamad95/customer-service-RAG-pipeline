"""Custom exception classes for the RAG pipeline."""


class RAGPipelineException(Exception):
    """Base exception for all RAG pipeline errors."""

    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        super().__init__(self.message)


# Preprocessing exceptions
class PreprocessingException(RAGPipelineException):
    """Exception raised during preprocessing."""

    pass


class TextCleaningException(PreprocessingException):
    """Exception raised during text cleaning."""

    pass


class PIIDetectionException(PreprocessingException):
    """Exception raised during PII detection/removal."""

    pass


class LanguageDetectionException(PreprocessingException):
    """Exception raised during language detection."""

    pass


# Retrieval exceptions
class RetrievalException(RAGPipelineException):
    """Exception raised during retrieval."""

    pass


class EmbeddingException(RetrievalException):
    """Exception raised during embedding generation."""

    pass


class VectorStoreException(RetrievalException):
    """Exception raised during vector store operations."""

    pass


class SearchException(RetrievalException):
    """Exception raised during search operations."""

    pass


# Decision exceptions
class DecisionException(RAGPipelineException):
    """Exception raised during decision making."""

    pass


class ConfidenceScoringException(DecisionException):
    """Exception raised during confidence scoring."""

    pass


class SentimentAnalysisException(DecisionException):
    """Exception raised during sentiment analysis."""

    pass


class RoutingException(DecisionException):
    """Exception raised during routing decisions."""

    pass


# Response generation exceptions
class ResponseException(RAGPipelineException):
    """Exception raised during response generation."""

    pass


class LLMException(ResponseException):
    """Exception raised during LLM operations."""

    pass


class PromptException(ResponseException):
    """Exception raised during prompt construction."""

    pass


# Database exceptions
class DatabaseException(RAGPipelineException):
    """Exception raised during database operations."""

    pass


class ConnectionException(DatabaseException):
    """Exception raised during database connection."""

    pass


class QueryException(DatabaseException):
    """Exception raised during database queries."""

    pass


# Configuration exceptions
class ConfigurationException(RAGPipelineException):
    """Exception raised for configuration errors."""

    pass


# Validation exceptions
class ValidationException(RAGPipelineException):
    """Exception raised for validation errors."""

    pass


# Timeout exceptions
class TimeoutException(RAGPipelineException):
    """Exception raised when operations timeout."""

    pass


# Model exceptions
class ModelLoadException(RAGPipelineException):
    """Exception raised when model loading fails."""

    pass
