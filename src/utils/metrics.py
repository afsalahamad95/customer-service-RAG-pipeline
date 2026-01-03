"""Prometheus metrics for RAG pipeline monitoring."""

from typing import Literal

from prometheus_client import Counter, Gauge, Histogram, Info

# Application info
app_info = Info("rag_application", "RAG Pipeline Application Information")

# Query metrics
queries_total = Counter(
    "rag_queries_total", "Total number of queries received", ["status"]  # success, error
)

query_duration = Histogram(
    "rag_query_duration_seconds",
    "Query processing duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# Routing decision metrics
routing_decisions_total = Counter(
    "rag_routing_decisions_total",
    "Total routing decisions made",
    ["decision"],  # auto_response, human_handoff
)

# Confidence metrics
confidence_score = Histogram(
    "rag_confidence_score",
    "Confidence score distribution",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Sentiment metrics
sentiment_total = Counter(
    "rag_sentiment_total",
    "Query sentiment distribution",
    ["sentiment"],  # positive, negative, neutral
)

# Pipeline stage metrics
pipeline_stage_duration = Histogram(
    "rag_pipeline_stage_duration_seconds",
    "Pipeline stage duration in seconds",
    ["stage"],  # preprocessing, retrieval, decision, response
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Knowledge base metrics
kb_documents_ingested = Counter(
    "rag_kb_documents_ingested_total", "Total documents ingested into knowledge base"
)

kb_searches_total = Counter(
    "rag_kb_searches_total",
    "Total knowledge base searches",
    ["search_type"],  # vector, bm25, hybrid
)

kb_documents_total = Gauge("rag_kb_documents_total", "Total documents in knowledge base")

# Retrieval metrics
retrieval_results = Histogram(
    "rag_retrieval_results_count",
    "Number of retrieved documents per query",
    buckets=[0, 1, 3, 5, 10, 20, 50],
)

# HTTP request metrics
http_requests_total = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

http_request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# LLM metrics
llm_requests_total = Counter(
    "rag_llm_requests_total",
    "Total LLM requests",
    ["provider", "status"],  # local, openai | success, error
)

llm_token_usage = Counter(
    "rag_llm_tokens_total", "Total LLM tokens used", ["provider", "type"]  # prompt, completion
)

llm_request_duration = Histogram(
    "rag_llm_request_duration_seconds",
    "LLM request duration in seconds",
    ["provider"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# Error metrics
errors_total = Counter("rag_errors_total", "Total errors by type", ["error_type", "component"])


# Helper functions for recording metrics
def record_query(status: Literal["success", "error"], duration: float):
    """Record query metrics."""
    queries_total.labels(status=status).inc()
    if status == "success":
        query_duration.observe(duration)


def record_routing_decision(decision: Literal["auto_response", "human_handoff"], confidence: float):
    """Record routing decision metrics."""
    routing_decisions_total.labels(decision=decision).inc()
    confidence_score.observe(confidence)


def record_sentiment(sentiment: Literal["positive", "negative", "neutral"]):
    """Record sentiment analysis result."""
    sentiment_total.labels(sentiment=sentiment).inc()


def record_pipeline_stage(
    stage: Literal["preprocessing", "retrieval", "decision", "response"], duration: float
):
    """Record pipeline stage duration."""
    pipeline_stage_duration.labels(stage=stage).observe(duration)


def record_kb_search(search_type: Literal["vector", "bm25", "hybrid"], result_count: int):
    """Record knowledge base search metrics."""
    kb_searches_total.labels(search_type=search_type).inc()
    retrieval_results.observe(result_count)


def record_http_request(method: str, endpoint: str, status: int, duration: float):
    """Record HTTP request metrics."""
    http_requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    http_request_duration.labels(method=method, endpoint=endpoint).observe(duration)


def record_llm_request(
    provider: str,
    status: Literal["success", "error"],
    duration: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
):
    """Record LLM request metrics."""
    llm_requests_total.labels(provider=provider, status=status).inc()
    llm_request_duration.labels(provider=provider).observe(duration)
    if prompt_tokens > 0:
        llm_token_usage.labels(provider=provider, type="prompt").inc(prompt_tokens)
    if completion_tokens > 0:
        llm_token_usage.labels(provider=provider, type="completion").inc(completion_tokens)


def record_error(error_type: str, component: str):
    """Record error occurrence."""
    errors_total.labels(error_type=error_type, component=component).inc()


# Initialize app info
app_info.info({"version": "1.0.0", "service": "rag-pipeline", "environment": "development"})
