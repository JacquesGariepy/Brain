"""
Unified Model Interface - Interface unique pour tous les modèles

Provides a consistent API regardless of backend:
- Local models (vLLM, Ollama, LM Studio)
- Cloud APIs (OpenAI, Claude, Gemini, Mistral)
- Custom models

Key features:
- Automatic backend selection
- Retry logic with exponential backoff
- Streaming support
- Batch processing
- Caching
- Cost tracking
- Performance monitoring
"""

import torch
from typing import Dict, List, Optional, Union, AsyncIterator, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import logging
from abc import ABC, abstractmethod


class BackendType(Enum):
    """Supported backend types"""
    VLLM = "vllm"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    COHERE = "cohere"
    CUSTOM = "custom"


class ModalityType(Enum):
    """Modality types"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


@dataclass
class ModelConfig:
    """Configuration for a model"""
    # Basic info
    model_name: str
    backend_type: BackendType
    modality: ModalityType = ModalityType.TEXT

    # Connection
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = 60.0

    # Model parameters
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Performance
    batch_size: int = 1
    num_retries: int = 3
    retry_delay: float = 1.0

    # Features
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_function_calling: bool = False

    # Cost tracking (USD per 1K tokens)
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0

    # Additional config
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceRequest:
    """Request for model inference"""
    # Input
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    images: Optional[List[Any]] = None
    audio: Optional[Any] = None

    # Generation parameters (override model config)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None

    # Features
    stream: bool = False
    functions: Optional[List[Dict]] = None

    # Metadata
    request_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class InferenceResponse:
    """Response from model inference"""
    # Output
    text: str
    finish_reason: str

    # Metadata
    model: str
    backend: BackendType
    request_id: Optional[str] = None

    # Metrics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    # Additional
    raw_response: Optional[Dict] = None
    function_call: Optional[Dict] = None


class ModelBackend(ABC):
    """Abstract base class for model backends"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")

    @abstractmethod
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate completion for a request"""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        request: InferenceRequest
    ) -> AsyncIterator[str]:
        """Generate completion with streaming"""
        pass

    @abstractmethod
    async def embed(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings"""
        pass

    @abstractmethod
    def supports_modality(self, modality: ModalityType) -> bool:
        """Check if backend supports a modality"""
        pass


class UnifiedModelInterface:
    """
    Unified interface for all model backends.

    Provides consistent API regardless of where model runs:
    - Local (vLLM, Ollama, LM Studio)
    - Cloud (OpenAI, Claude, Gemini, etc.)
    - Custom deployments

    Features:
    - Automatic backend selection
    - Load balancing across multiple backends
    - Retry with exponential backoff
    - Response caching
    - Cost tracking
    - Performance monitoring
    """

    def __init__(self):
        self.backends: Dict[str, ModelBackend] = {}
        self.logger = logging.getLogger(__name__)

        # Metrics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.total_latency_ms = 0.0

        # Cache
        self.cache: Dict[str, InferenceResponse] = {}
        self.cache_enabled = True

    def register_backend(
        self,
        name: str,
        backend: ModelBackend
    ):
        """Register a new backend"""
        self.backends[name] = backend
        self.logger.info(f"Registered backend: {name} ({backend.config.backend_type.value})")

    def add_local_model(
        self,
        name: str,
        backend_type: BackendType,
        model_name: str,
        api_base: str = "http://localhost:8000",
        **kwargs
    ):
        """
        Add a local model (vLLM, Ollama, LM Studio).

        Args:
            name: Unique identifier for this backend
            backend_type: Type of backend (vLLM, Ollama, LM Studio)
            model_name: Model name/path
            api_base: API endpoint
            **kwargs: Additional config parameters
        """
        from .local_models import vLLMBackend, OllamaBackend, LMStudioBackend

        config = ModelConfig(
            model_name=model_name,
            backend_type=backend_type,
            api_base=api_base,
            **kwargs
        )

        if backend_type == BackendType.VLLM:
            backend = vLLMBackend(config)
        elif backend_type == BackendType.OLLAMA:
            backend = OllamaBackend(config)
        elif backend_type == BackendType.LMSTUDIO:
            backend = LMStudioBackend(config)
        else:
            raise ValueError(f"Unsupported local backend: {backend_type}")

        self.register_backend(name, backend)

    def add_cloud_model(
        self,
        name: str,
        backend_type: BackendType,
        model_name: str,
        api_key: str,
        **kwargs
    ):
        """
        Add a cloud API model (OpenAI, Claude, Gemini).

        Args:
            name: Unique identifier
            backend_type: Cloud provider
            model_name: Model identifier
            api_key: API key
            **kwargs: Additional parameters
        """
        from .cloud_apis import (
            OpenAIBackend, ClaudeBackend, GeminiBackend,
            MistralBackend, CohereBackend
        )

        config = ModelConfig(
            model_name=model_name,
            backend_type=backend_type,
            api_key=api_key,
            **kwargs
        )

        if backend_type == BackendType.OPENAI:
            backend = OpenAIBackend(config)
        elif backend_type == BackendType.CLAUDE:
            backend = ClaudeBackend(config)
        elif backend_type == BackendType.GEMINI:
            backend = GeminiBackend(config)
        elif backend_type == BackendType.MISTRAL:
            backend = MistralBackend(config)
        elif backend_type == BackendType.COHERE:
            backend = CohereBackend(config)
        else:
            raise ValueError(f"Unsupported cloud backend: {backend_type}")

        self.register_backend(name, backend)

    async def generate(
        self,
        request: InferenceRequest,
        backend_name: Optional[str] = None,
        fallback_backends: Optional[List[str]] = None
    ) -> InferenceResponse:
        """
        Generate completion with automatic backend selection.

        Args:
            request: Inference request
            backend_name: Specific backend to use (optional)
            fallback_backends: Fallback options if primary fails

        Returns:
            Inference response
        """
        start_time = time.time()

        # Select backend
        if backend_name is None:
            backend_name = self._select_best_backend(request)

        backends_to_try = [backend_name]
        if fallback_backends:
            backends_to_try.extend(fallback_backends)

        # Try backends with retry logic
        last_error = None
        for backend_name in backends_to_try:
            if backend_name not in self.backends:
                self.logger.warning(f"Backend not found: {backend_name}")
                continue

            backend = self.backends[backend_name]

            # Retry logic
            for attempt in range(backend.config.num_retries):
                try:
                    response = await backend.generate(request)

                    # Update metrics
                    response.latency_ms = (time.time() - start_time) * 1000
                    self._update_metrics(response)

                    return response

                except Exception as e:
                    last_error = e
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed for {backend_name}: {e}"
                    )

                    if attempt < backend.config.num_retries - 1:
                        # Exponential backoff
                        delay = backend.config.retry_delay * (2 ** attempt)
                        await asyncio.sleep(delay)

        # All backends failed
        raise RuntimeError(
            f"All backends failed. Last error: {last_error}"
        )

    async def generate_stream(
        self,
        request: InferenceRequest,
        backend_name: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Generate completion with streaming.

        Args:
            request: Inference request (stream=True)
            backend_name: Backend to use

        Yields:
            Text chunks as they're generated
        """
        if backend_name is None:
            backend_name = self._select_best_backend(request)

        backend = self.backends[backend_name]

        if not backend.config.supports_streaming:
            raise ValueError(f"Backend {backend_name} doesn't support streaming")

        async for chunk in backend.generate_stream(request):
            yield chunk

    async def embed(
        self,
        texts: Union[str, List[str]],
        backend_name: Optional[str] = None
    ) -> torch.Tensor:
        """
        Generate embeddings for texts.

        Args:
            texts: Text or list of texts
            backend_name: Backend to use

        Returns:
            Embeddings tensor (num_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        if backend_name is None:
            # Select backend that supports embeddings
            backend_name = self._select_embedding_backend()

        backend = self.backends[backend_name]
        embeddings = await backend.embed(texts)

        return embeddings

    def _select_best_backend(
        self,
        request: InferenceRequest
    ) -> str:
        """
        Intelligently select best backend for request.

        Considers:
        - Modality support
        - Cost
        - Latency
        - Availability
        """
        # Determine required modality
        required_modality = ModalityType.TEXT
        if request.images:
            required_modality = ModalityType.MULTIMODAL

        # Filter compatible backends
        compatible = []
        for name, backend in self.backends.items():
            if backend.supports_modality(required_modality):
                compatible.append(name)

        if not compatible:
            raise ValueError(f"No backend supports {required_modality}")

        # Score backends
        scores = {}
        for name in compatible:
            backend = self.backends[name]
            score = 0.0

            # Prefer local over cloud (faster, free)
            if backend.config.backend_type in [
                BackendType.VLLM, BackendType.OLLAMA, BackendType.LMSTUDIO
            ]:
                score += 1.0

            # Consider cost
            if backend.config.cost_per_1k_input_tokens == 0:
                score += 0.5

            scores[name] = score

        # Return highest scoring
        return max(scores, key=scores.get)

    def _select_embedding_backend(self) -> str:
        """Select backend for embeddings"""
        # Prefer local embeddings
        for name, backend in self.backends.items():
            if backend.config.backend_type == BackendType.VLLM:
                return name

        # Fallback to any backend
        return next(iter(self.backends.keys()))

    def _update_metrics(self, response: InferenceResponse):
        """Update internal metrics"""
        self.total_requests += 1
        self.total_tokens += response.total_tokens
        self.total_cost_usd += response.cost_usd
        self.total_latency_ms += response.latency_ms

    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics"""
        avg_latency = (
            self.total_latency_ms / self.total_requests
            if self.total_requests > 0 else 0
        )

        return {
            'total_requests': self.total_requests,
            'total_tokens': self.total_tokens,
            'total_cost_usd': self.total_cost_usd,
            'average_latency_ms': avg_latency,
            'backends': list(self.backends.keys())
        }

    def reset_metrics(self):
        """Reset all metrics"""
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.total_latency_ms = 0.0


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        print("="*80)
        print("Unified Model Interface - Local & Cloud Models")
        print("="*80)

        # Create interface
        interface = UnifiedModelInterface()

        # Add local models
        print("\nAdding local models...")
        interface.add_local_model(
            name="local_llama",
            backend_type=BackendType.OLLAMA,
            model_name="llama2:7b",
            api_base="http://localhost:11434"
        )

        # Add cloud models (commented out - needs API keys)
        # interface.add_cloud_model(
        #     name="gpt4",
        #     backend_type=BackendType.OPENAI,
        #     model_name="gpt-4-turbo",
        #     api_key="your-api-key"
        # )

        print(f"Registered backends: {list(interface.backends.keys())}")

        # Example request
        request = InferenceRequest(
            prompt="What is the capital of France?",
            max_tokens=100,
            temperature=0.7
        )

        print("\nSending request...")
        # response = await interface.generate(request)
        # print(f"Response: {response.text}")

        # Metrics
        metrics = interface.get_metrics()
        print(f"\nMetrics: {metrics}")

        print("\n" + "="*80)

    asyncio.run(main())
