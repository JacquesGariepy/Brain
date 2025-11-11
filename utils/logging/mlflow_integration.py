"""
MLflow Integration

Provides comprehensive experiment tracking and model registry with MLflow.
"""

from typing import Optional, Dict, Any, List
import warnings
from pathlib import Path

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLflowLogger:
    """
    MLflow logger for experiment tracking and model registry.

    Features:
    - Experiment tracking
    - Parameter and metric logging
    - Model logging and registry
    - Artifact storage
    - Run comparison
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ):
        """
        Args:
            experiment_name: MLflow experiment name
            run_name: Run name (auto-generated if None)
            tracking_uri: MLflow tracking server URI
            artifact_location: Default artifact location
            tags: Run tags
            nested: Whether this is a nested run
        """
        if not MLFLOW_AVAILABLE:
            warnings.warn("MLflow not available. Install with: pip install mlflow")
            self.enabled = False
            return

        self.enabled = True

        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
            else:
                experiment_id = experiment.experiment_id

            mlflow.set_experiment(experiment_name)
        except Exception as e:
            warnings.warn(f"Failed to set experiment: {e}")
            experiment_id = None

        # Start run
        self.run = mlflow.start_run(run_name=run_name, nested=nested, tags=tags)
        self.run_id = self.run.info.run_id if self.run else None

    def log_param(self, key: str, value: Any):
        """
        Log a single parameter.

        Args:
            key: Parameter name
            value: Parameter value

        Example:
            >>> logger.log_param("learning_rate", 0.001)
        """
        if not self.enabled:
            return

        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]):
        """
        Log multiple parameters.

        Args:
            params: Dictionary of parameters

        Example:
            >>> logger.log_params({
            ...     "learning_rate": 0.001,
            ...     "batch_size": 32,
            ...     "epochs": 100,
            ... })
        """
        if not self.enabled:
            return

        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Step number

        Example:
            >>> logger.log_metric("loss", 0.5, step=100)
        """
        if not self.enabled:
            return

        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics.

        Args:
            metrics: Dictionary of metrics
            step: Step number

        Example:
            >>> logger.log_metrics({
            ...     "train_loss": 0.5,
            ...     "val_loss": 0.6,
            ...     "accuracy": 0.95,
            ... }, step=100)
        """
        if not self.enabled:
            return

        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a local file as an artifact.

        Args:
            local_path: Path to local file
            artifact_path: Artifact directory path

        Example:
            >>> logger.log_artifact("model.pt", artifact_path="models")
        """
        if not self.enabled:
            return

        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Log all files in a directory as artifacts.

        Args:
            local_dir: Path to local directory
            artifact_path: Artifact directory path

        Example:
            >>> logger.log_artifacts("./checkpoints", artifact_path="models")
        """
        if not self.enabled:
            return

        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)

    def log_model(
        self,
        model,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Log a PyTorch model.

        Args:
            model: PyTorch model
            artifact_path: Artifact path within run
            registered_model_name: Name for model registry
            **kwargs: Additional arguments for mlflow.pytorch.log_model

        Example:
            >>> logger.log_model(
            ...     model,
            ...     artifact_path="model",
            ...     registered_model_name="brain-model-v1"
            ... )
        """
        if not self.enabled:
            return

        try:
            mlflow.pytorch.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                **kwargs
            )
        except Exception as e:
            warnings.warn(f"Failed to log model: {e}")

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        """
        Log a dictionary as a JSON artifact.

        Args:
            dictionary: Dictionary to log
            artifact_file: Artifact filename

        Example:
            >>> logger.log_dict({"config": "values"}, "config.json")
        """
        if not self.enabled:
            return

        mlflow.log_dict(dictionary, artifact_file)

    def log_figure(self, figure, artifact_file: str):
        """
        Log a matplotlib figure.

        Args:
            figure: Matplotlib figure
            artifact_file: Artifact filename

        Example:
            >>> import matplotlib.pyplot as plt
            >>> fig = plt.figure()
            >>> plt.plot([1, 2, 3])
            >>> logger.log_figure(fig, "plot.png")
        """
        if not self.enabled:
            return

        mlflow.log_figure(figure, artifact_file)

    def log_text(self, text: str, artifact_file: str):
        """
        Log text as an artifact.

        Args:
            text: Text content
            artifact_file: Artifact filename

        Example:
            >>> logger.log_text("Model description", "description.txt")
        """
        if not self.enabled:
            return

        mlflow.log_text(text, artifact_file)

    def set_tag(self, key: str, value: Any):
        """
        Set a tag for the run.

        Args:
            key: Tag name
            value: Tag value

        Example:
            >>> logger.set_tag("model_type", "transformer")
        """
        if not self.enabled:
            return

        mlflow.set_tag(key, value)

    def set_tags(self, tags: Dict[str, Any]):
        """
        Set multiple tags.

        Args:
            tags: Dictionary of tags

        Example:
            >>> logger.set_tags({
            ...     "model_type": "transformer",
            ...     "dataset": "imagenet",
            ... })
        """
        if not self.enabled:
            return

        mlflow.set_tags(tags)

    def end_run(self, status: str = "FINISHED"):
        """
        End the current run.

        Args:
            status: Run status ("FINISHED", "FAILED", "KILLED")

        Example:
            >>> logger.end_run()
        """
        if not self.enabled:
            return

        mlflow.end_run(status=status)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type is not None:
            self.end_run(status="FAILED")
        else:
            self.end_run(status="FINISHED")


def init_mlflow(
    experiment_name: str,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    **kwargs
) -> MLflowLogger:
    """
    Initialize MLflow logger.

    Args:
        experiment_name: Experiment name
        run_name: Run name
        tracking_uri: Tracking server URI
        **kwargs: Additional MLflow arguments

    Returns:
        MLflowLogger instance

    Example:
        >>> logger = init_mlflow(
        ...     experiment_name="brain-training",
        ...     run_name="experiment-1",
        ...     tracking_uri="http://localhost:5000"
        ... )
        >>> logger.log_params({"lr": 0.001, "batch_size": 32})
        >>> logger.log_metrics({"loss": 0.5}, step=0)
        >>> logger.end_run()
    """
    return MLflowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        **kwargs
    )


def get_mlflow_run_info(run_id: str) -> Dict[str, Any]:
    """
    Get information about an MLflow run.

    Args:
        run_id: MLflow run ID

    Returns:
        Dictionary with run information

    Example:
        >>> info = get_mlflow_run_info("abc123")
        >>> print(info["metrics"])
    """
    if not MLFLOW_AVAILABLE:
        raise ImportError("MLflow required: pip install mlflow")

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    return {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "params": run.data.params,
        "metrics": run.data.metrics,
        "tags": run.data.tags,
    }


# Test function
def test_mlflow():
    """Test MLflow integration"""
    print("Testing MLflow Integration...")

    if not MLFLOW_AVAILABLE:
        print("MLflow not available, skipping test")
        return

    # Use local tracking
    logger = MLflowLogger(
        experiment_name="brain-test",
        run_name="test-run",
    )

    if logger.enabled:
        # Log parameters
        logger.log_params({
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
        })

        # Log metrics
        logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=0)
        logger.log_metrics({"loss": 0.3, "accuracy": 0.95}, step=1)

        # Set tags
        logger.set_tags({
            "model_type": "transformer",
            "dataset": "test",
        })

        print(f"   ✓ MLflow logging successful (run_id: {logger.run_id})")

        logger.end_run()
    else:
        print("   - MLflow not available")

    print("\n✓ MLflow tests complete!")


if __name__ == "__main__":
    test_mlflow()
