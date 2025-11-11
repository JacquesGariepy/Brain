"""
Weights & Biases (WandB) Integration

Provides comprehensive experiment tracking with WandB.
"""

from typing import Optional, Dict, Any, List
import warnings

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBLogger:
    """
    WandB logger for experiment tracking.

    Features:
    - Automatic metric logging
    - Model checkpointing
    - Hyperparameter tracking
    - Artifact management
    - System metrics
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        entity: Optional[str] = None,
        dir: Optional[str] = None,
        resume: Optional[str] = None,
        reinit: bool = False,
        mode: str = "online",
    ):
        """
        Args:
            project: WandB project name
            name: Run name (auto-generated if None)
            config: Configuration dictionary
            tags: List of tags
            notes: Run notes/description
            entity: WandB entity (username or team)
            dir: Directory for WandB files
            resume: Resume mode ("allow", "must", "never", "auto")
            reinit: Whether to allow re-initialization
            mode: "online", "offline", or "disabled"
        """
        if not WANDB_AVAILABLE:
            warnings.warn("WandB not available. Install with: pip install wandb")
            self.enabled = False
            return

        self.enabled = True
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            entity=entity,
            dir=dir,
            resume=resume,
            reinit=reinit,
            mode=mode,
        )

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """
        Log metrics to WandB.

        Args:
            metrics: Dictionary of metric name -> value
            step: Global step number
            commit: Whether to commit immediately

        Example:
            >>> logger.log({"loss": 0.5, "accuracy": 0.95}, step=100)
        """
        if not self.enabled:
            return

        wandb.log(metrics, step=step, commit=commit)

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log training metrics.

        Args:
            metrics: Dictionary of metrics
            step: Training step

        Example:
            >>> logger.log_metrics({
            ...     "train/loss": 0.5,
            ...     "train/accuracy": 0.95,
            ...     "lr": 0.001,
            ... }, step=100)
        """
        self.log(metrics, step=step)

    def log_hyperparameters(self, params: Dict[str, Any]):
        """
        Log hyperparameters.

        Args:
            params: Hyperparameter dictionary

        Example:
            >>> logger.log_hyperparameters({
            ...     "learning_rate": 0.001,
            ...     "batch_size": 32,
            ...     "epochs": 100,
            ... })
        """
        if not self.enabled:
            return

        wandb.config.update(params)

    def log_model(self, model_path: str, name: str = "model", aliases: Optional[List[str]] = None):
        """
        Log a model as a WandB artifact.

        Args:
            model_path: Path to model file
            name: Artifact name
            aliases: Artifact aliases (e.g., ["latest", "best"])

        Example:
            >>> logger.log_model("checkpoints/model.pt", name="model", aliases=["best"])
        """
        if not self.enabled:
            return

        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact, aliases=aliases)

    def log_image(self, image, caption: Optional[str] = None, key: str = "image"):
        """
        Log an image to WandB.

        Args:
            image: PIL Image, numpy array, or torch tensor
            caption: Image caption
            key: Logging key

        Example:
            >>> logger.log_image(image_tensor, caption="Prediction", key="predictions")
        """
        if not self.enabled:
            return

        wandb.log({key: wandb.Image(image, caption=caption)})

    def log_images(self, images: List, captions: Optional[List[str]] = None, key: str = "images"):
        """
        Log multiple images.

        Args:
            images: List of images
            captions: List of captions
            key: Logging key
        """
        if not self.enabled:
            return

        if captions is None:
            captions = [None] * len(images)

        wandb_images = [wandb.Image(img, caption=cap) for img, cap in zip(images, captions)]
        wandb.log({key: wandb_images})

    def log_table(self, data: List[List[Any]], columns: List[str], key: str = "table"):
        """
        Log a table to WandB.

        Args:
            data: Table data (list of rows)
            columns: Column names
            key: Logging key

        Example:
            >>> logger.log_table(
            ...     data=[[1, 0.5, 0.9], [2, 0.3, 0.95]],
            ...     columns=["epoch", "loss", "accuracy"],
            ...     key="results"
            ... )
        """
        if not self.enabled:
            return

        table = wandb.Table(data=data, columns=columns)
        wandb.log({key: table})

    def log_histogram(self, values, key: str = "histogram"):
        """
        Log a histogram of values.

        Args:
            values: Array-like values
            key: Logging key

        Example:
            >>> logger.log_histogram(gradients, key="gradients/layer1")
        """
        if not self.enabled:
            return

        wandb.log({key: wandb.Histogram(values)})

    def watch(self, model, log: str = "gradients", log_freq: int = 100):
        """
        Watch a PyTorch model (log gradients and parameters).

        Args:
            model: PyTorch model
            log: What to log ("gradients", "parameters", "all", None)
            log_freq: Logging frequency

        Example:
            >>> logger.watch(model, log="all", log_freq=100)
        """
        if not self.enabled:
            return

        wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self):
        """Finish the WandB run"""
        if not self.enabled:
            return

        wandb.finish()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.finish()


def init_wandb(
    project: str,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> WandBLogger:
    """
    Initialize WandB logger.

    Args:
        project: Project name
        name: Run name
        config: Configuration dictionary
        **kwargs: Additional WandB arguments

    Returns:
        WandBLogger instance

    Example:
        >>> logger = init_wandb(
        ...     project="brain-training",
        ...     name="experiment-1",
        ...     config={"lr": 0.001, "batch_size": 32}
        ... )
        >>> logger.log_metrics({"loss": 0.5}, step=0)
        >>> logger.finish()
    """
    return WandBLogger(
        project=project,
        name=name,
        config=config,
        **kwargs
    )


# Test function
def test_wandb():
    """Test WandB integration"""
    print("Testing WandB Integration...")

    if not WANDB_AVAILABLE:
        print("WandB not available, skipping test")
        return

    # Test in offline mode to avoid authentication
    logger = WandBLogger(
        project="brain-test",
        name="test-run",
        config={"lr": 0.001},
        mode="offline"
    )

    if logger.enabled:
        # Log metrics
        logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=0)
        logger.log_metrics({"loss": 0.3, "accuracy": 0.95}, step=1)

        # Log hyperparameters
        logger.log_hyperparameters({"batch_size": 32, "epochs": 10})

        print("   ✓ WandB logging successful")

        logger.finish()
    else:
        print("   - WandB not available")

    print("\n✓ WandB tests complete!")


if __name__ == "__main__":
    test_wandb()
