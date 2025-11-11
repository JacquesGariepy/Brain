"""
Model Integrations - Connect to local and cloud model providers

Permet d'utiliser des modèles pré-entraînés de façon transparente:
- Local: vLLM, Ollama, LM Studio
- Cloud: OpenAI, Anthropic Claude, Google Gemini, Mistral, Cohere
"""

from .local_models import LocalModelManager, vLLMBackend, OllamaBackend, LMStudioBackend
from .cloud_apis import CloudAPIManager, OpenAIBackend, ClaudeBackend, GeminiBackend
from .unified_interface import UnifiedModelInterface, ModelConfig, InferenceRequest

__all__ = [
    'LocalModelManager', 'vLLMBackend', 'OllamaBackend', 'LMStudioBackend',
    'CloudAPIManager', 'OpenAIBackend', 'ClaudeBackend', 'GeminiBackend',
    'UnifiedModelInterface', 'ModelConfig', 'InferenceRequest'
]
