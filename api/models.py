"""
API Data Models

Pydantic models for request/response validation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for predictions"""

    text: Optional[str] = Field(None, description="Input text for NLP tasks")
    image: Optional[str] = Field(None, description="Base64 encoded image or image URL")
    audio: Optional[str] = Field(None, description="Base64 encoded audio or audio URL")
    inputs: Optional[Dict[str, Any]] = Field(None, description="Generic input dictionary")
    model_name: str = Field(..., description="Model name to use for prediction")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional parameters")

    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, how are you?",
                "model_name": "gpt-2",
                "parameters": {"max_length": 50, "temperature": 0.7}
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    prediction: Any = Field(..., description="Model prediction")
    confidence: Optional[float] = Field(None, description="Confidence score")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    model_name: str = Field(..., description="Model used for prediction")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "prediction": "I'm doing well, thank you!",
                "confidence": 0.95,
                "latency_ms": 120.5,
                "model_name": "gpt-2",
                "metadata": {"tokens_generated": 10}
            }
        }


class TrainingRequest(BaseModel):
    """Request model for training"""

    model_name: str = Field(..., description="Model architecture name")
    dataset: str = Field(..., description="Dataset name or path")
    config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters")
    output_dir: str = Field(default="./checkpoints", description="Output directory for checkpoints")

    class Config:
        schema_extra = {
            "example": {
                "model_name": "bert-base-uncased",
                "dataset": "glue/sst2",
                "config": {
                    "num_epochs": 3,
                    "batch_size": 32,
                    "learning_rate": 2e-5
                },
                "hyperparameters": {
                    "warmup_steps": 500,
                    "weight_decay": 0.01
                },
                "output_dir": "./models/sst2"
            }
        }


class TrainingResponse(BaseModel):
    """Response model for training"""

    job_id: str = Field(..., description="Training job ID")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")
    config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration")

    class Config:
        schema_extra = {
            "example": {
                "job_id": "train-abc123",
                "status": "started",
                "message": "Training job started successfully",
                "config": {"num_epochs": 3, "batch_size": 32}
            }
        }


class ModelInfo(BaseModel):
    """Model information"""

    name: str = Field(..., description="Model name")
    architecture: str = Field(..., description="Model architecture")
    version: str = Field(..., description="Model version")
    description: Optional[str] = Field(None, description="Model description")
    parameters: Optional[int] = Field(None, description="Number of parameters")
    supported_tasks: List[str] = Field(default_factory=list, description="Supported tasks")
    input_format: Optional[str] = Field(None, description="Expected input format")
    output_format: Optional[str] = Field(None, description="Output format")

    class Config:
        schema_extra = {
            "example": {
                "name": "bert-base-uncased",
                "architecture": "BERT",
                "version": "1.0",
                "description": "BERT base model, uncased",
                "parameters": 110000000,
                "supported_tasks": ["text-classification", "token-classification"],
                "input_format": "text",
                "output_format": "logits"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    models_loaded: int = Field(..., description="Number of models loaded")
    gpu_available: bool = Field(..., description="GPU availability")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "uptime_seconds": 3600.5,
                "models_loaded": 3,
                "gpu_available": True
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""

    inputs: List[Dict[str, Any]] = Field(..., description="List of inputs")
    model_name: str = Field(..., description="Model name")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Shared parameters")

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {"text": "First input"},
                    {"text": "Second input"},
                ],
                "model_name": "bert-base-uncased",
                "parameters": {"max_length": 128}
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""

    predictions: List[Any] = Field(..., description="List of predictions")
    total_items: int = Field(..., description="Total number of items")
    successful: int = Field(..., description="Number of successful predictions")
    failed: int = Field(..., description="Number of failed predictions")
    total_latency_ms: float = Field(..., description="Total latency in milliseconds")

    class Config:
        schema_extra = {
            "example": {
                "predictions": ["prediction 1", "prediction 2"],
                "total_items": 2,
                "successful": 2,
                "failed": 0,
                "total_latency_ms": 250.5
            }
        }


class EvaluationRequest(BaseModel):
    """Request for model evaluation"""

    model_name: str = Field(..., description="Model name")
    dataset: str = Field(..., description="Evaluation dataset")
    metrics: List[str] = Field(default_factory=lambda: ["accuracy"], description="Metrics to compute")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Evaluation config")

    class Config:
        schema_extra = {
            "example": {
                "model_name": "bert-base-uncased",
                "dataset": "glue/sst2",
                "metrics": ["accuracy", "f1", "precision", "recall"],
                "config": {"batch_size": 32}
            }
        }


class EvaluationResponse(BaseModel):
    """Response for model evaluation"""

    job_id: str = Field(..., description="Evaluation job ID")
    status: str = Field(..., description="Job status")
    metrics: Optional[Dict[str, float]] = Field(None, description="Computed metrics")
    message: str = Field(..., description="Status message")

    class Config:
        schema_extra = {
            "example": {
                "job_id": "eval-xyz789",
                "status": "completed",
                "metrics": {
                    "accuracy": 0.95,
                    "f1": 0.94,
                    "precision": 0.93,
                    "recall": 0.95
                },
                "message": "Evaluation completed successfully"
            }
        }
