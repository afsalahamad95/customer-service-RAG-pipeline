"""Main pipeline orchestrator."""

from dataclasses import dataclass
from typing import Optional, List

from ..preprocessing import PreprocessingPipeline, PreprocessingResult
from ..retrieval import HybridSearch, HybridSearchResult
from ..decision import Router, RoutingDecision
from ..response.llm_service import LLMService
from ..utils import get_logger
from ..utils.exceptions import RAGPipelineException

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Complete pipeline result."""
    query_id: Optional[int]
    original_query: str
    preprocessed: PreprocessingResult
    search_results: List[HybridSearchResult]
    routing_decision: RoutingDecision
    response: Optional[str] = None
    latency_ms: Optional[int] = None
    metadata: dict = None


class PipelineExecutor:
    """Orchestrates the complete RAG pipeline."""
    
    def __init__(self):
        # Initialize all components
        self.preprocessor = PreprocessingPipeline()
        self.hybrid_search = HybridSearch()
        self.router = Router()
        self.llm_service = LLMService()
        
        logger.info("Pipeline executor initialized")
    
    async def execute(
        self,
        query: str,
        top_k: int = 5,
        generate_response: bool = True
    ) -> PipelineResult:
        """
        Execute the complete pipeline.
        
        Args:
            query: User query
            top_k: Number of search results
            generate_response: Whether to generate LLM response
            
        Returns:
            PipelineResult
        """
        import time
        start_time = time.time()
        
        try:
            # Step 1: Preprocessing
            logger.info(f"Processing query: {query[:100]}...")
            preprocessed = self.preprocessor.process(query, detect_pii=True)
            
            # Step 2: Retrieval
            search_results = self.hybrid_search.search(
                preprocessed.anonymized_text,
                top_k=top_k
            )
            
            # Step 3: Routing decision
            routing_decision = self.router.route(
                preprocessed.anonymized_text,
                search_results
            )
            
            # Step 4: Response generation (if auto-response)
            response = None
            if generate_response and routing_decision.route == "auto_response":
                # Build context from search results
                context = self._build_context(search_results)
                
                # Generate response
                response = self.llm_service.generate_response(
                    query=preprocessed.anonymized_text,
                    context=context,
                    stream=False
                )
                logger.info("Generated auto-response")
            elif routing_decision.route == "human_handoff":
                response = None
                logger.info(f"Routing to human agent: {routing_decision.reason}")
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            result = PipelineResult(
                query_id=None,  # Will be set when saved to DB
                original_query=query,
                preprocessed=preprocessed,
                search_results=search_results,
                routing_decision=routing_decision,
                response=response,
                latency_ms=latency_ms,
                metadata={
                    "top_k": top_k,
                    "retrieval_count": len(search_results)
                }
            )
            
            logger.info(f"Pipeline completed in {latency_ms}ms")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise RAGPipelineException(f"Pipeline failed: {e}")
    
    def _build_context(self, search_results: List[HybridSearchResult], max_chunks: int = 3) -> str:
        """Build context string from search results."""
        if not search_results:
            return ""
        
        context_parts = []
        for i, result in enumerate(search_results[:max_chunks]):
            context_parts.append(f"[{i+1}] {result.content}")
        
        return "\n\n".join(context_parts)
    
    def initialize_search_index(self):
        """Initialize search indices. Call after populating vector store."""
        try:
            self.hybrid_search.index_documents()
            logger.info("Search indices initialized")
        except Exception as e:
            logger.error(f"Failed to initialize indices: {e}")
            raise RAGPipelineException(f"Index initialization failed: {e}")
