"""
Local Model Backends - vLLM, Ollama, LM Studio

Support for running models locally:
- vLLM: High-performance inference server (production)
- Ollama: Easy model management (development)
- LM Studio: GUI-based local inference
"""

import aiohttp
import torch
from typing import Dict, List, Optional, AsyncIterator
import json
from .unified_interface import ModelBackend, ModelConfig, InferenceRequest, InferenceResponse, BackendType, ModalityType


class vLLMBackend(ModelBackend):
    """
    vLLM backend for high-performance local inference.

    vLLM features:
    - PagedAttention for efficient memory management
    - Continuous batching
    - Fast inference (up to 24x faster than HuggingFace)
    - OpenAI-compatible API
    - Supports: Llama, Mistral, Yi, Qwen, etc.

    Usage:
    ```bash
    # Start vLLM server
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-2-7b-chat-hf \
        --port 8000
    ```
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_base = config.api_base or "http://localhost:8000"
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate completion using vLLM"""
        session = await self._get_session()

        # Prepare request (OpenAI-compatible format)
        payload = {
            "model": self.config.model_name,
            "prompt": request.prompt if request.prompt else self._format_messages(request.messages),
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "temperature": request.temperature or self.config.temperature,
            "top_p": request.top_p or self.config.top_p,
            "stream": False
        }

        if request.stop:
            payload["stop"] = request.stop

        # Send request
        async with session.post(
            f"{self.api_base}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"vLLM error: {error_text}")

            result = await resp.json()

        # Parse response
        choice = result["choices"][0]
        usage = result.get("usage", {})

        return InferenceResponse(
            text=choice["text"],
            finish_reason=choice["finish_reason"],
            model=self.config.model_name,
            backend=BackendType.VLLM,
            request_id=request.request_id,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            raw_response=result
        )

    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """Stream generation with vLLM"""
        session = await self._get_session()

        payload = {
            "model": self.config.model_name,
            "prompt": request.prompt if request.prompt else self._format_messages(request.messages),
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "temperature": request.temperature or self.config.temperature,
            "stream": True
        }

        async with session.post(
            f"{self.api_base}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        ) as resp:
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    chunk = json.loads(data)
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        text = chunk['choices'][0].get('text', '')
                        if text:
                            yield text

    async def embed(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings (if model supports it)"""
        # vLLM doesn't directly support embeddings in completion mode
        # Would need separate embedding model
        raise NotImplementedError("Use dedicated embedding model for vLLM")

    def supports_modality(self, modality: ModalityType) -> bool:
        """vLLM supports text and some models support vision"""
        return modality in [ModalityType.TEXT, ModalityType.MULTIMODAL]

    def _format_messages(self, messages: Optional[List[Dict]]) -> str:
        """Format messages into prompt"""
        if not messages:
            return ""

        # Simple concatenation (actual format depends on model)
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"{role}: {content}\n"
        return prompt


class OllamaBackend(ModelBackend):
    """
    Ollama backend for easy local model management.

    Ollama features:
    - Simple model download and management
    - One-line installation
    - Supports: Llama, Mistral, Phi, Gemma, etc.
    - Built-in model library

    Usage:
    ```bash
    # Start Ollama
    ollama serve

    # Pull a model
    ollama pull llama2:7b

    # Run
    ollama run llama2:7b
    ```
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_base = config.api_base or "http://localhost:11434"
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate completion using Ollama"""
        session = await self._get_session()

        # Ollama API format
        payload = {
            "model": self.config.model_name,
            "prompt": request.prompt if request.prompt else self._format_messages(request.messages),
            "stream": False,
            "options": {
                "temperature": request.temperature or self.config.temperature,
                "top_p": request.top_p or self.config.top_p,
                "top_k": self.config.top_k,
                "num_predict": request.max_tokens or self.config.max_tokens
            }
        }

        if request.stop:
            payload["options"]["stop"] = request.stop

        # Handle images for multimodal models (llava)
        if request.images:
            payload["images"] = request.images

        async with session.post(
            f"{self.api_base}/api/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"Ollama error: {error_text}")

            result = await resp.json()

        # Parse Ollama response
        return InferenceResponse(
            text=result.get("response", ""),
            finish_reason="stop",
            model=self.config.model_name,
            backend=BackendType.OLLAMA,
            request_id=request.request_id,
            prompt_tokens=result.get("prompt_eval_count", 0),
            completion_tokens=result.get("eval_count", 0),
            total_tokens=result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
            raw_response=result
        )

    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """Stream generation with Ollama"""
        session = await self._get_session()

        payload = {
            "model": self.config.model_name,
            "prompt": request.prompt if request.prompt else self._format_messages(request.messages),
            "stream": True,
            "options": {
                "temperature": request.temperature or self.config.temperature,
                "num_predict": request.max_tokens or self.config.max_tokens
            }
        }

        async with session.post(
            f"{self.api_base}/api/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        ) as resp:
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if line:
                    chunk = json.loads(line)
                    text = chunk.get("response", "")
                    if text:
                        yield text
                    if chunk.get("done", False):
                        break

    async def embed(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings using Ollama"""
        session = await self._get_session()

        embeddings = []
        for text in texts:
            payload = {
                "model": self.config.model_name,
                "prompt": text
            }

            async with session.post(
                f"{self.api_base}/api/embeddings",
                json=payload
            ) as resp:
                result = await resp.json()
                embeddings.append(result["embedding"])

        return torch.tensor(embeddings)

    def supports_modality(self, modality: ModalityType) -> bool:
        """Ollama supports text and vision (llava models)"""
        if "llava" in self.config.model_name.lower():
            return modality in [ModalityType.TEXT, ModalityType.IMAGE, ModalityType.MULTIMODAL]
        return modality == ModalityType.TEXT

    def _format_messages(self, messages: Optional[List[Dict]]) -> str:
        """Format messages into prompt"""
        if not messages:
            return ""

        prompt = ""
        for msg in messages:
            content = msg.get("content", "")
            prompt += content + "\n"
        return prompt


class LMStudioBackend(ModelBackend):
    """
    LM Studio backend for GUI-based local inference.

    LM Studio features:
    - User-friendly GUI
    - Model discovery and download
    - OpenAI-compatible API
    - Cross-platform (Mac, Windows, Linux)
    - Supports: GGUF models

    Usage:
    1. Download LM Studio
    2. Load a model through GUI
    3. Start local server (OpenAI compatible)
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_base = config.api_base or "http://localhost:1234"
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate completion using LM Studio"""
        session = await self._get_session()

        # LM Studio uses OpenAI-compatible API
        payload = {
            "model": self.config.model_name,
            "messages": request.messages if request.messages else [
                {"role": "user", "content": request.prompt}
            ],
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "temperature": request.temperature or self.config.temperature,
            "top_p": request.top_p or self.config.top_p,
            "stream": False
        }

        if request.stop:
            payload["stop"] = request.stop

        async with session.post(
            f"{self.api_base}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"LM Studio error: {error_text}")

            result = await resp.json()

        # Parse OpenAI-format response
        choice = result["choices"][0]
        usage = result.get("usage", {})

        return InferenceResponse(
            text=choice["message"]["content"],
            finish_reason=choice["finish_reason"],
            model=self.config.model_name,
            backend=BackendType.LMSTUDIO,
            request_id=request.request_id,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            raw_response=result
        )

    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """Stream generation with LM Studio"""
        session = await self._get_session()

        payload = {
            "model": self.config.model_name,
            "messages": request.messages if request.messages else [
                {"role": "user", "content": request.prompt}
            ],
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "temperature": request.temperature or self.config.temperature,
            "stream": True
        }

        async with session.post(
            f"{self.api_base}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        ) as resp:
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    chunk = json.loads(data)
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        text = delta.get('content', '')
                        if text:
                            yield text

    async def embed(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings"""
        # LM Studio may support embeddings through /v1/embeddings
        session = await self._get_session()

        payload = {
            "model": self.config.model_name,
            "input": texts
        }

        async with session.post(
            f"{self.api_base}/v1/embeddings",
            json=payload
        ) as resp:
            result = await resp.json()
            embeddings = [item["embedding"] for item in result["data"]]
            return torch.tensor(embeddings)

    def supports_modality(self, modality: ModalityType) -> bool:
        """LM Studio primarily supports text"""
        return modality == ModalityType.TEXT


class LocalModelManager:
    """
    Manager for local model backends.

    Provides utilities for:
    - Backend discovery
    - Model listing
    - Health checks
    - Performance benchmarking
    """

    def __init__(self):
        self.backends: Dict[str, ModelBackend] = {}

    async def discover_backends(self) -> Dict[str, bool]:
        """
        Discover available local backends.

        Returns:
            Dictionary of backend availability
        """
        results = {}

        # Check vLLM
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8000/health", timeout=1) as resp:
                    results["vllm"] = resp.status == 200
        except:
            results["vllm"] = False

        # Check Ollama
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags", timeout=1) as resp:
                    results["ollama"] = resp.status == 200
        except:
            results["ollama"] = False

        # Check LM Studio
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:1234/v1/models", timeout=1) as resp:
                    results["lmstudio"] = resp.status == 200
        except:
            results["lmstudio"] = False

        return results

    async def list_ollama_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags") as resp:
                    result = await resp.json()
                    return [model["name"] for model in result.get("models", [])]
        except:
            return []


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        print("="*80)
        print("Local Model Backends - vLLM, Ollama, LM Studio")
        print("="*80)

        # Discover backends
        manager = LocalModelManager()
        available = await manager.discover_backends()

        print("\nAvailable backends:")
        for backend, status in available.items():
            status_str = "✓ Available" if status else "✗ Not available"
            print(f"  {backend}: {status_str}")

        # List Ollama models
        if available.get("ollama"):
            models = await manager.list_ollama_models()
            print(f"\nOllama models: {models}")

        print("\n" + "="*80)

    asyncio.run(main())
