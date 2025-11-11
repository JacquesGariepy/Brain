"""
Brain REST API

FastAPI-based REST API for model serving and inference.
"""

from .app import create_app, app
from .models import (
    PredictionRequest,
    PredictionResponse,
    TrainingRequest,
    TrainingResponse,
    ModelInfo,
    HealthResponse,
)

__all__ = [
    'create_app',
    'app',
    'PredictionRequest',
    'PredictionResponse',
    'TrainingRequest',
    'TrainingResponse',
    'ModelInfo',
    'HealthResponse',
]
