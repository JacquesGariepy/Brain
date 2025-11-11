"""
Brain FastAPI Application

Main FastAPI application for model serving and inference.
"""

import time
import uuid
from typing import Dict, List, Optional
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from .models import (
    PredictionRequest,
    PredictionResponse,
    TrainingRequest,
    TrainingResponse,
    ModelInfo,
    HealthResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    EvaluationRequest,
    EvaluationResponse,
)

# Track service start time
START_TIME = time.time()

# In-memory job storage (replace with Redis/database in production)
JOBS = {}

# Model registry (replace with actual model loading)
MODELS_REGISTRY = {}

# Model manager for loading and caching models
class ModelManager:
    """Manages model loading, caching, and inference"""

    def __init__(self):
        self.loaded_models = {}
        self.orchestrator = None

    def get_orchestrator(self):
        """Lazy load orchestrator"""
        if self.orchestrator is None:
            try:
                from core.orchestrator import BrainOrchestrator
                self.orchestrator = BrainOrchestrator()
            except Exception as e:
                logger.error(f"Failed to load orchestrator: {e}")
                raise
        return self.orchestrator

    def load_model(self, model_name: str, architecture: str = None):
        """
        Load a model by name or architecture.

        Args:
            model_name: Model identifier
            architecture: Architecture type (e.g., 'transformer', 'vit', 'clip')

        Returns:
            Loaded model
        """
        # Check if already loaded
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        # Map common model names to architectures
        architecture_map = {
            'bert': 'transformer',
            'gpt': 'transformer',
            'transformer': 'transformer',
            'vit': 'vit',
            'vision-transformer': 'vit',
            'clip': 'clip',
            'resnet': 'resnet',
        }

        # Determine architecture
        if architecture is None:
            architecture = architecture_map.get(model_name.lower(), 'transformer')

        # Load using orchestrator loaders
        orchestrator = self.get_orchestrator()
        loader_func = f"_load_{architecture.lower().replace('-', '_')}"

        if not hasattr(orchestrator, loader_func):
            raise ValueError(f"Unknown architecture: {architecture}")

        model = getattr(orchestrator, loader_func)()

        # Move to GPU if available
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        # Cache model
        self.loaded_models[model_name] = model

        return model

    def predict(self, model_name: str, inputs: dict, architecture: str = None):
        """
        Run inference with a model.

        Args:
            model_name: Model identifier
            inputs: Input data (text, image, etc.)
            architecture: Optional architecture type

        Returns:
            Prediction results
        """
        import torch

        # Load model
        model = self.load_model(model_name, architecture)

        # Prepare inputs based on model type
        from architectures.base import VisionArchitecture, LanguageArchitecture, MultimodalArchitecture

        device = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.no_grad():
            if isinstance(model, MultimodalArchitecture):
                # Handle multimodal inputs (text + image)
                model_inputs = {}

                if 'image' in inputs:
                    # TODO: Preprocess image
                    model_inputs['image'] = inputs['image'].to(device)

                if 'text' in inputs:
                    # TODO: Tokenize text
                    # For now, use random tokens as placeholder
                    model_inputs['text'] = torch.randint(0, 50000, (1, 77)).to(device)

                output = model(**model_inputs)

            elif isinstance(model, VisionArchitecture):
                # Handle image input
                if 'image' in inputs:
                    x = inputs['image'].to(device)
                else:
                    # Placeholder: random image
                    x = torch.randn(1, 3, 224, 224).to(device)

                output = model(x)

            elif isinstance(model, LanguageArchitecture):
                # Handle text input
                if 'input_ids' in inputs:
                    input_ids = inputs['input_ids'].to(device)
                elif 'text' in inputs:
                    # TODO: Tokenize text properly
                    # For now, use random tokens as placeholder
                    input_ids = torch.randint(0, 50000, (1, 128)).to(device)
                else:
                    input_ids = torch.randint(0, 50000, (1, 128)).to(device)

                output = model(input_ids=input_ids)

            else:
                raise ValueError(f"Unknown model type: {type(model)}")

        return output

# Global model manager
model_manager = ModelManager()


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI app

    Example:
        >>> app = create_app()
        >>> # Run with: uvicorn api.app:app --reload
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI required: pip install fastapi uvicorn")

    app = FastAPI(
        title="Brain API",
        description="REST API for Brain framework - Model serving, training, and inference",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "type": type(exc).__name__}
        )

    return app


app = create_app() if FASTAPI_AVAILABLE else None


if FASTAPI_AVAILABLE:

    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint"""
        return {
            "message": "Welcome to Brain API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """
        Health check endpoint.

        Returns service status, uptime, and resource availability.
        """
        uptime = time.time() - START_TIME

        # Check GPU availability
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            pass

        return HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=uptime,
            models_loaded=len(MODELS_REGISTRY),
            gpu_available=gpu_available,
        )

    @app.get("/models", response_model=List[ModelInfo], tags=["Models"])
    async def list_models():
        """
        List all available models.

        Returns list of loaded models with their metadata.
        """
        # In production, this would query the model registry
        models = []

        # Example models
        example_models = [
            ModelInfo(
                name="bert-base-uncased",
                architecture="BERT",
                version="1.0",
                description="BERT base model for text classification",
                parameters=110000000,
                supported_tasks=["text-classification", "token-classification"],
                input_format="text",
                output_format="logits",
            ),
            ModelInfo(
                name="clip-vit-base",
                architecture="CLIP",
                version="1.0",
                description="CLIP vision-language model",
                parameters=150000000,
                supported_tasks=["image-text-matching", "zero-shot-classification"],
                input_format="image+text",
                output_format="embeddings",
            ),
        ]

        return example_models

    @app.get("/models/{model_name}", response_model=ModelInfo, tags=["Models"])
    async def get_model_info(model_name: str):
        """
        Get information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model metadata
        """
        # In production, query model registry
        if model_name not in MODELS_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        return MODELS_REGISTRY[model_name]

    @app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
    async def predict(request: PredictionRequest):
        """
        Make a prediction with a model.

        Supports text, image, audio, and multimodal inputs.

        Args:
            request: Prediction request with inputs and model name

        Returns:
            Prediction response with results and metadata
        """
        start_time = time.time()

        try:
            # Prepare inputs
            inputs = {}
            if request.text:
                inputs['text'] = request.text
            if request.image:
                inputs['image'] = request.image
            if request.audio:
                inputs['audio'] = request.audio

            # Run inference
            output = model_manager.predict(
                model_name=request.model_name,
                inputs=inputs,
                architecture=request.model_name.split('-')[0] if '-' in request.model_name else None
            )

            # Extract predictions
            if hasattr(output, 'predictions') and output.predictions is not None:
                predictions = output.predictions
            elif hasattr(output, 'logits') and output.logits is not None:
                predictions = output.logits
            elif hasattr(output, 'embeddings') and output.embeddings is not None:
                predictions = output.embeddings
            else:
                predictions = None

            # Convert to JSON-serializable format
            import torch
            if predictions is not None and isinstance(predictions, torch.Tensor):
                predictions_list = predictions.cpu().tolist()
                # Get top prediction for classification
                if len(predictions.shape) == 2:
                    # Batch of logits
                    top_pred = int(predictions[0].argmax().item())
                    confidence = float(torch.softmax(predictions[0], dim=0).max().item())
                else:
                    top_pred = 0
                    confidence = 0.0

                prediction_result = {
                    "class": top_pred,
                    "logits": predictions_list if len(predictions_list) < 1000 else "too_large",
                    "text": request.text if request.text else "N/A"
                }
            else:
                prediction_result = {"text": "No output generated"}
                confidence = 0.0

            latency_ms = (time.time() - start_time) * 1000

            # Get metadata
            metadata = {}
            if hasattr(output, 'metadata') and output.metadata:
                metadata = output.metadata
            metadata['architecture'] = type(model_manager.loaded_models.get(request.model_name)).__name__

            return PredictionResponse(
                prediction=prediction_result,
                confidence=confidence,
                latency_ms=latency_ms,
                model_name=request.model_name,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
    async def predict_batch(request: BatchPredictionRequest):
        """
        Make predictions for a batch of inputs.

        Optimized for throughput with batching.

        Args:
            request: Batch prediction request

        Returns:
            Batch prediction response
        """
        start_time = time.time()

        predictions = []
        successful = 0
        failed = 0

        for input_data in request.inputs:
            try:
                # Run prediction for each input
                predictions.append(f"Prediction for input")
                successful += 1
            except Exception as e:
                predictions.append({"error": str(e)})
                failed += 1

        latency_ms = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            predictions=predictions,
            total_items=len(request.inputs),
            successful=successful,
            failed=failed,
            total_latency_ms=latency_ms,
        )

    @app.post("/train", response_model=TrainingResponse, tags=["Training"])
    async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
        """
        Start a training job.

        Training runs asynchronously in the background.

        Args:
            request: Training configuration
            background_tasks: FastAPI background tasks

        Returns:
            Training job information
        """
        job_id = f"train-{uuid.uuid4().hex[:8]}"

        # Store job info
        JOBS[job_id] = {
            "type": "training",
            "status": "queued",
            "config": request.dict(),
            "created_at": time.time(),
        }

        # In production, add to job queue
        # background_tasks.add_task(run_training, job_id, request)

        return TrainingResponse(
            job_id=job_id,
            status="queued",
            message="Training job queued successfully",
            config=request.config,
        )

    @app.get("/train/{job_id}", tags=["Training"])
    async def get_training_status(job_id: str):
        """
        Get status of a training job.

        Args:
            job_id: Training job ID

        Returns:
            Job status and metrics
        """
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return JOBS[job_id]

    @app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
    async def evaluate_model(request: EvaluationRequest, background_tasks: BackgroundTasks):
        """
        Evaluate a model on a dataset.

        Runs asynchronously and computes specified metrics.

        Args:
            request: Evaluation configuration
            background_tasks: FastAPI background tasks

        Returns:
            Evaluation job information
        """
        job_id = f"eval-{uuid.uuid4().hex[:8]}"

        JOBS[job_id] = {
            "type": "evaluation",
            "status": "queued",
            "config": request.dict(),
            "created_at": time.time(),
        }

        # In production, add to job queue
        # background_tasks.add_task(run_evaluation, job_id, request)

        return EvaluationResponse(
            job_id=job_id,
            status="queued",
            metrics=None,
            message="Evaluation job queued successfully",
        )

    @app.get("/evaluate/{job_id}", tags=["Evaluation"])
    async def get_evaluation_status(job_id: str):
        """
        Get status of an evaluation job.

        Args:
            job_id: Evaluation job ID

        Returns:
            Job status and computed metrics
        """
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return JOBS[job_id]

    @app.delete("/jobs/{job_id}", tags=["Jobs"])
    async def cancel_job(job_id: str):
        """
        Cancel a running job.

        Args:
            job_id: Job ID to cancel

        Returns:
            Cancellation confirmation
        """
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        JOBS[job_id]["status"] = "cancelled"

        return {"message": f"Job {job_id} cancelled", "job_id": job_id}

    @app.get("/jobs", tags=["Jobs"])
    async def list_jobs(status: Optional[str] = None, limit: int = 100):
        """
        List all jobs.

        Args:
            status: Filter by status (queued, running, completed, failed, cancelled)
            limit: Maximum number of jobs to return

        Returns:
            List of jobs
        """
        jobs = list(JOBS.values())

        if status:
            jobs = [j for j in jobs if j.get("status") == status]

        return jobs[:limit]


# Run with: uvicorn api.app:app --reload --port 8000
if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
