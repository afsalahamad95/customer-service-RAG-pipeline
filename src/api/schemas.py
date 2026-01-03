"""Request and response schemas."""

from typing import Any

from pydantic import BaseModel, Field


# Query schemas
class QueryRequest(BaseModel):
    """Query request."""

    query: str = Field(..., min_length=1, description="User query text")
    top_k: int | None = Field(5, ge=1, le=20, description="Number of search results")
    session_id: str | None = Field(None, description="Session identifier")
    user_id: str | None = Field(None, description="User identifier")


class QueryResponse(BaseModel):
    """Query response."""

    query_id: int | None = None
    response: str | None = None
    routing_decision: str  # auto_response or human_handoff
    confidence: float
    sentiment: dict[str, Any]
    latency_ms: int
    metadata: dict[str, Any] = {}


# Knowledge base schemas
class KBDocument(BaseModel):
    """Knowledge base document."""

    title: str | None = None
    content: str = Field(..., min_length=1)
    metadata: dict[str, Any] = {}


class KBDocumentResponse(BaseModel):
    """KB document response."""

    id: int
    title: str | None
    content: str
    created_at: str
    metadata: dict[str, Any]


class KBIngestRequest(BaseModel):
    """Request to ingest documents into KB."""

    documents: list[KBDocument]


class KBIngestResponse(BaseModel):
    """Response for KB ingestion."""

    success: bool
    document_ids: list[int]
    count: int


# Feedback schemas
class FeedbackRequest(BaseModel):
    """Feedback submission."""

    query_id: int
    response_id: int | None = None
    feedback_type: str = Field(..., description="thumbs_up, thumbs_down, agent_approval, etc.")
    feedback_value: int = Field(..., ge=-1, le=1)
    corrected_response: str | None = None
    agent_id: str | None = None
    notes: str | None = None


class FeedbackResponse(BaseModel):
    """Feedback response."""

    success: bool
    feedback_id: int


# Health check
class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    components: dict[str, bool]


# Metrics
class MetricsResponse(BaseModel):
    """Metrics response."""

    total_queries: int
    auto_responses: int
    human_handoffs: int
    deflection_ratio: float
    avg_latency_ms: float
    avg_confidence: float
