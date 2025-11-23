"""Request and response schemas."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# Query schemas
class QueryRequest(BaseModel):
    """Query request."""
    query: str = Field(..., min_length=1, description="User query text")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of search results")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")


class QueryResponse(BaseModel):
    """Query response."""
    query_id: Optional[int] = None
    response: Optional[str] = None
    routing_decision: str  # auto_response or human_handoff
    confidence: float
    sentiment: Dict[str, Any]
    latency_ms: int
    metadata: Dict[str, Any] = {}


# Knowledge base schemas
class KBDocument(BaseModel):
    """Knowledge base document."""
    title: Optional[str] = None
    content: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = {}


class KBDocumentResponse(BaseModel):
    """KB document response."""
    id: int
    title: Optional[str]
    content: str
    created_at: str
    metadata: Dict[str, Any]


class KBIngestRequest(BaseModel):
    """Request to ingest documents into KB."""
    documents: List[KBDocument]


class KBIngestResponse(BaseModel):
    """Response for KB ingestion."""
    success: bool
    document_ids: List[int]
    count: int


# Feedback schemas
class FeedbackRequest(BaseModel):
    """Feedback submission."""
    query_id: int
    response_id: Optional[int] = None
    feedback_type: str = Field(..., description="thumbs_up, thumbs_down, agent_approval, etc.")
    feedback_value: int = Field(..., ge=-1, le=1)
    corrected_response: Optional[str] = None
    agent_id: Optional[str] = None
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Feedback response."""
    success: bool
    feedback_id: int


# Health check
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    components: Dict[str, bool]


# Metrics
class MetricsResponse(BaseModel):
    """Metrics response."""
    total_queries: int
    auto_responses: int
    human_handoffs: int
    deflection_ratio: float
    avg_latency_ms: float
    avg_confidence: float
