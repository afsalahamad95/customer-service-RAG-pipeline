"""LLM service supporting multiple providers."""

from typing import Optional, Dict, Any, Generator
from abc import ABC, abstractmethod

# Local LLM support
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# API providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from src.utils import get_logger, load_config
from src.utils.exceptions import LLMException

logger = get_logger(__name__)


class BaseLLM(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt."""
        pass
    
    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream response from prompt."""
        pass


class LocalLLM(BaseLLM):
    """Local LLM using llama.cpp with Metal acceleration."""
    
    def __init__(self, config: Dict[str, Any]):
        if not LLAMA_CPP_AVAILABLE:
            raise LLMException("llama-cpp-python not installed")
        
        self.model_path = config.get("model_path")
        self.n_ctx = config.get("n_ctx", 4096)
        self.n_gpu_layers = config.get("n_gpu_layers", 1)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.max_tokens = config.get("max_tokens", 512)
        
        try:
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            logger.info(f"Loaded local LLM: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            raise LLMException(f"Local LLM initialization failed: {e}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response."""
        try:
            response = self.model(
                prompt,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                stop=kwargs.get("stop", []),
                echo=False
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Local LLM generation failed: {e}")
            raise LLMException(f"Generation failed: {e}")
    
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream response."""
        try:
            stream = self.model(
                prompt,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                stream=True
            )
            for output in stream:
                yield output["choices"][0]["text"]
        except Exception as e:
            logger.error(f"Local LLM streaming failed: {e}")
            raise LLMException(f"Streaming failed: {e}")


class OpenAILLM(BaseLLM):
    """OpenAI API LLM."""
    
    def __init__(self, config: Dict[str, Any]):
        if not OPENAI_AVAILABLE:
            raise LLMException("openai package not installed")
        
        self.api_key = config.get("api_key")
        self.model = config.get("model", "gpt-3.5-turbo")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 512)
        
        if not self.api_key:
            raise LLMException("OpenAI API key not provided")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        logger.info(f"Initialized OpenAI LLM: {self.model}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise LLMException(f"Generation failed: {e}")
    
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream response."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise LLMException(f"Streaming failed: {e}")


class LLMService:
    """LLM service with pluggable providers."""
    
    def __init__(self):
        self.config = load_config()
        self.llm_config = self.config.get("llm", {})
        self.provider = self.llm_config.get("provider", "local")
        self.prompts = self.llm_config.get("prompts", {})
        
        # Initialize provider
        self.llm = self._initialize_provider()
        logger.info(f"LLM service initialized with provider: {self.provider}")
    
    def _initialize_provider(self) -> BaseLLM:
        """Initialize the configured LLM provider."""
        if self.provider == "local":
            return LocalLLM(self.llm_config.get("local", {}))
        elif self.provider == "openai":
            return OpenAILLM(self.llm_config.get("openai", {}))
        elif self.provider == "anthropic":
            # Similar implementation for Anthropic
            raise LLMException("Anthropic provider not yet implemented")
        else:
            raise LLMException(f"Unknown provider: {self.provider}")
    
    def generate_response(
        self,
        query: str,
        context: str,
        stream: bool = False,
        **kwargs
    ) -> str | Generator[str, None, None]:
        """
        Generate response using RAG context.
        
        Args:
            query: User query
            context: Retrieved context
            stream: Whether to stream response
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response (string or stream)
        """
        # Build prompt using template
        prompt = self._build_rag_prompt(query, context)
        
        if stream:
            return self.llm.stream(prompt, **kwargs)
        else:
            return self.llm.generate(prompt, **kwargs)
    
    def _build_rag_prompt(self, query: str, context: str) -> str:
        """Build RAG prompt from template."""
        system_prompt = self.prompts.get("system", "")
        rag_template = self.prompts.get("rag_template", "")
        
        # Format template
        prompt = f"{system_prompt}\n\n{rag_template}"
        prompt = prompt.format(context=context, question=query)
        
        return prompt
