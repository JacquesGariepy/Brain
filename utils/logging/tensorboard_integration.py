"""
TensorBoard Integration

Provides visualization and logging with TensorBoard.
"""

from typing import Optional, Dict, Any, Union
import warnings
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    import torch
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class TensorBoardLogger:
    """
    TensorBoard logger for visualization and monitoring.

    Features:
    - Scalar logging (loss, accuracy, etc.)
    - Image logging
    - Histogram logging (weights, gradients)
    - Graph logging
    - Embedding visualization
    - PR curves
    """

    def __init__(
        self,
        log_dir: str = "./runs",
        comment: str = "",
        purge_step: Optional[int] = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = "",
    ):
        """
        Args:
            log_dir: Directory for TensorBoard logs
            comment: Comment string appended to default run name
            purge_step: Remove data before this step
            max_queue: Size of queue for pending events
            flush_secs: Flush interval in seconds
            filename_suffix: Suffix for event file
        """
        if not TENSORBOARD_AVAILABLE:
            warnings.warn("TensorBoard not available. Install with: pip install tensorboard torch")
            self.enabled = False
            return

        self.enabled = True
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            comment=comment,
            purge_step=purge_step,
            max_queue=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix,
        )

    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log a scalar value.

        Args:
            tag: Data identifier (e.g., "loss/train")
            value: Scalar value
            step: Global step

        Example:
            >>> logger.log_scalar("loss/train", 0.5, step=100)
        """
        if not self.enabled:
            return

        self.writer.add_scalar(tag, value, global_step=step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """
        Log multiple scalars in one plot.

        Args:
            main_tag: Main tag (e.g., "loss")
            tag_scalar_dict: Dictionary of subtags and values
            step: Global step

        Example:
            >>> logger.log_scalars("loss", {
            ...     "train": 0.5,
            ...     "val": 0.6,
            ... }, step=100)
        """
        if not self.enabled:
            return

        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step=step)

    def log_image(self, tag: str, image, step: int, dataformats: str = "CHW"):
        """
        Log an image.

        Args:
            tag: Data identifier
            image: Image tensor [C, H, W] or [H, W, C]
            step: Global step
            dataformats: Image data format ("CHW" or "HWC")

        Example:
            >>> logger.log_image("predictions", image_tensor, step=100)
        """
        if not self.enabled:
            return

        self.writer.add_image(tag, image, global_step=step, dataformats=dataformats)

    def log_images(self, tag: str, images, step: int, dataformats: str = "NCHW"):
        """
        Log multiple images as a grid.

        Args:
            tag: Data identifier
            images: Batch of images [N, C, H, W]
            step: Global step
            dataformats: Image data format

        Example:
            >>> logger.log_images("batch", image_batch, step=100)
        """
        if not self.enabled:
            return

        self.writer.add_images(tag, images, global_step=step, dataformats=dataformats)

    def log_histogram(self, tag: str, values, step: int, bins: str = "tensorflow"):
        """
        Log a histogram of values.

        Args:
            tag: Data identifier (e.g., "weights/layer1")
            values: Array or tensor of values
            step: Global step
            bins: Binning method

        Example:
            >>> logger.log_histogram("gradients/layer1", gradients, step=100)
        """
        if not self.enabled:
            return

        self.writer.add_histogram(tag, values, global_step=step, bins=bins)

    def log_figure(self, tag: str, figure, step: int, close: bool = True):
        """
        Log a matplotlib figure.

        Args:
            tag: Data identifier
            figure: Matplotlib figure
            step: Global step
            close: Whether to close figure after logging

        Example:
            >>> import matplotlib.pyplot as plt
            >>> fig = plt.figure()
            >>> plt.plot([1, 2, 3])
            >>> logger.log_figure("plot", fig, step=100)
        """
        if not self.enabled:
            return

        self.writer.add_figure(tag, figure, global_step=step, close=close)

    def log_text(self, tag: str, text: str, step: int):
        """
        Log text.

        Args:
            tag: Data identifier
            text: Text string
            step: Global step

        Example:
            >>> logger.log_text("description", "Model trained successfully", step=100)
        """
        if not self.enabled:
            return

        self.writer.add_text(tag, text, global_step=step)

    def log_graph(self, model, input_to_model):
        """
        Log model graph.

        Args:
            model: PyTorch model
            input_to_model: Example input tensor

        Example:
            >>> logger.log_graph(model, torch.randn(1, 3, 224, 224))
        """
        if not self.enabled:
            return

        try:
            self.writer.add_graph(model, input_to_model)
        except Exception as e:
            warnings.warn(f"Failed to log graph: {e}")

    def log_embedding(
        self,
        mat,
        metadata: Optional[list] = None,
        label_img = None,
        global_step: Optional[int] = None,
        tag: str = "default",
    ):
        """
        Log embeddings for visualization.

        Args:
            mat: Matrix [N, D] where N is number of data points
            metadata: List of labels for each data point
            label_img: Images corresponding to data points
            global_step: Global step
            tag: Data identifier

        Example:
            >>> embeddings = model.get_embeddings(data)
            >>> labels = ["cat", "dog", "bird"]
            >>> logger.log_embedding(embeddings, metadata=labels)
        """
        if not self.enabled:
            return

        self.writer.add_embedding(
            mat,
            metadata=metadata,
            label_img=label_img,
            global_step=global_step,
            tag=tag,
        )

    def log_pr_curve(
        self,
        tag: str,
        labels,
        predictions,
        step: int,
        num_thresholds: int = 127,
    ):
        """
        Log precision-recall curve.

        Args:
            tag: Data identifier
            labels: Ground truth labels
            predictions: Predicted probabilities
            step: Global step
            num_thresholds: Number of thresholds

        Example:
            >>> logger.log_pr_curve("pr_curve", labels, predictions, step=100)
        """
        if not self.enabled:
            return

        self.writer.add_pr_curve(
            tag,
            labels,
            predictions,
            global_step=step,
            num_thresholds=num_thresholds,
        )

    def log_hyperparameters(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Dict[str, float],
    ):
        """
        Log hyperparameters and metrics.

        Args:
            hparam_dict: Hyperparameter dictionary
            metric_dict: Metric dictionary

        Example:
            >>> logger.log_hyperparameters(
            ...     {"lr": 0.001, "batch_size": 32},
            ...     {"accuracy": 0.95, "loss": 0.5}
            ... )
        """
        if not self.enabled:
            return

        self.writer.add_hparams(hparam_dict, metric_dict)

    def flush(self):
        """Flush pending events to disk"""
        if not self.enabled:
            return

        self.writer.flush()

    def close(self):
        """Close the writer"""
        if not self.enabled:
            return

        self.writer.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def get_tensorboard_writer(log_dir: str = "./runs", **kwargs) -> TensorBoardLogger:
    """
    Get a TensorBoard writer/logger.

    Args:
        log_dir: Directory for logs
        **kwargs: Additional arguments for TensorBoardLogger

    Returns:
        TensorBoardLogger instance

    Example:
        >>> logger = get_tensorboard_writer(log_dir="./runs/experiment1")
        >>> logger.log_scalar("loss", 0.5, step=0)
        >>> logger.close()
    """
    return TensorBoardLogger(log_dir=log_dir, **kwargs)


# Test function
def test_tensorboard():
    """Test TensorBoard integration"""
    print("Testing TensorBoard Integration...")

    if not TENSORBOARD_AVAILABLE:
        print("TensorBoard not available, skipping test")
        return

    logger = TensorBoardLogger(log_dir="./test_runs", comment="test")

    if logger.enabled:
        # Log scalars
        for step in range(10):
            logger.log_scalar("loss/train", 1.0 / (step + 1), step=step)
            logger.log_scalar("accuracy/train", step / 10.0, step=step)

        # Log multiple scalars
        logger.log_scalars("metrics", {
            "loss": 0.5,
            "accuracy": 0.9,
        }, step=5)

        # Log histogram
        import torch
        weights = torch.randn(100)
        logger.log_histogram("weights", weights, step=0)

        # Log text
        logger.log_text("status", "Training completed successfully", step=10)

        print(f"   ✓ TensorBoard logging successful (log_dir: {logger.log_dir})")
        print(f"   → Run: tensorboard --logdir={logger.log_dir}")

        logger.close()
    else:
        print("   - TensorBoard not available")

    print("\n✓ TensorBoard tests complete!")


if __name__ == "__main__":
    test_tensorboard()
