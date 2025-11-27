"""FastAPI application."""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
import asyncio

from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    KBDocument,
    KBIngestRequest,
    KBIngestResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetricsResponse,
)
from src.orchestration.pipeline_executor import PipelineExecutor
from src.retrieval import VectorStore, EmbeddingService
from src.database.connection import get_db, db_manager
from src.utils import load_config, get_logger

logger = get_logger(__name__)
config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title=config.get("app", {}).get("name", "RAG Pipeline"),
    version=config.get("app", {}).get("version", "1.0.0"),
    description="Customer Service RAG Pipeline API",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("api", {}).get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline (lazy loading)
pipeline: PipelineExecutor = None
vector_store: VectorStore = None
embedding_service: EmbeddingService = None


def get_pipeline() -> PipelineExecutor:
    """Get or initialize pipeline."""
    global pipeline
    if pipeline is None:
        pipeline = PipelineExecutor()
    return pipeline


def get_vector_store() -> VectorStore:
    """Get or initialize vector store."""
    global vector_store
    if vector_store is None:
        vector_store = VectorStore()
    return vector_store


def get_embedding_service() -> EmbeddingService:
    """Get or initialize embedding service."""
    global embedding_service
    if embedding_service is None:
        embedding_service = EmbeddingService()
    return embedding_service


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting RAG Pipeline API...")

    # Check database connection
    if not db_manager.health_check():
        logger.error("Database connection failed!")
        raise Exception("Database not available")

    logger.info("API started successfully")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {"name": "RAG Customer Service Pipeline", "version": "1.0.0", "status": "running"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    db_healthy = db_manager.health_check()

    return HealthResponse(
        status="healthy" if db_healthy else "unhealthy",
        version=config.get("app", {}).get("version", "1.0.0"),
        components={"database": db_healthy, "pipeline": pipeline is not None},
    )


@app.post("/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest):
    """
    Submit a query to the RAG pipeline.

    Processes the query through preprocessing, retrieval, decision,
    and potentially auto-response generation.
    """
    try:
        pipe = get_pipeline()

        # Execute pipeline
        result = await pipe.execute(
            query=request.query, top_k=request.top_k or 5, generate_response=True
        )

        # TODO: Save to database

        return QueryResponse(
            query_id=result.query_id,
            response=result.response,
            routing_decision=result.routing_decision.route,
            confidence=result.routing_decision.confidence,
            sentiment=result.routing_decision.sentiment,
            latency_ms=result.latency_ms,
            metadata=result.metadata or {},
        )

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kb/ingest", response_model=KBIngestResponse)
async def ingest_documents(request: KBIngestRequest):
    """
    Ingest documents into the knowledge base.

    Generates embeddings and stores documents with vector representations.
    """
    try:
        vs = get_vector_store()
        emb = get_embedding_service()

        document_ids = []

        for doc in request.documents:
            # Generate embedding
            embedding = emb.embed(doc.content)

            # Insert into vector store
            doc_id = vs.insert(
                content=doc.content,
                embedding=embedding.flatten(),
                title=doc.title,
                metadata=doc.metadata,
            )
            document_ids.append(doc_id)

        # Rebuild search indices
        pipe = get_pipeline()
        pipe.initialize_search_index()

        logger.info(f"Ingested {len(document_ids)} documents")

        return KBIngestResponse(success=True, document_ids=document_ids, count=len(document_ids))

    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    """Submit feedback for a query response."""
    try:
        query = text(
            """
            INSERT INTO feedback 
            (query_id, response_id, feedback_type, feedback_value, corrected_response, agent_id, notes)
            VALUES (:query_id, :response_id, :feedback_type, :feedback_value, :corrected, :agent_id, :notes)
            RETURNING id
        """
        )

        result = db.execute(
            query,
            {
                "query_id": request.query_id,
                "response_id": request.response_id,
                "feedback_type": request.feedback_type,
                "feedback_value": request.feedback_value,
                "corrected": request.corrected_response,
                "agent_id": request.agent_id,
                "notes": request.notes,
            },
        )

        feedback_id = result.scalar()
        logger.info(f"Feedback submitted: {feedback_id}")

        return FeedbackResponse(success=True, feedback_id=feedback_id)

    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(db: Session = Depends(get_db)):
    """Get pipeline metrics."""
    try:
        # Get query counts
        query = text(
            """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN routing_decision = 'auto_response' THEN 1 ELSE 0 END) as auto,
                SUM(CASE WHEN routing_decision = 'human_handoff' THEN 1 ELSE 0 END) as human,
                AVG(confidence_score) as avg_confidence
            FROM queries
            WHERE created_at >= NOW() - INTERVAL '24 hours'
        """
        )

        result = db.execute(query).fetchone()

        total = result[0] or 0
        auto = result[1] or 0
        human = result[2] or 0
        avg_confidence = float(result[3] or 0)

        deflection_ratio = auto / total if total > 0 else 0

        # Get latency
        latency_query = text(
            """
            SELECT AVG(latency_ms)
            FROM responses
            WHERE created_at >= NOW() - INTERVAL '24 hours'
        """
        )

        avg_latency = db.execute(latency_query).scalar() or 0

        return MetricsResponse(
            total_queries=total,
            auto_responses=auto,
            human_handoffs=human,
            deflection_ratio=deflection_ratio,
            avg_latency_ms=float(avg_latency),
            avg_confidence=avg_confidence,
        )

    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=config.get("api", {}).get("host", "0.0.0.0"),
        port=config.get("api", {}).get("port", 8000),
        reload=config.get("api", {}).get("reload", True),
    )
