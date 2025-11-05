"""
Brain Metrics Tracking

Comprehensive metrics computation and tracking for ML tasks.
"""

from typing import Dict, List, Optional, Union, Any
from collections import defaultdict
import warnings

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class MetricsTracker:
    """
    Track and compute metrics over training/evaluation.

    Features:
    - Accumulate metrics over batches
    - Compute running averages
    - Reset for new epochs
    - Support for multiple metrics
    """

    def __init__(self):
        """Initialize metrics tracker"""
        self.metrics = defaultdict(list)
        self.running_sums = defaultdict(float)
        self.running_counts = defaultdict(int)

    def update(self, metrics: Dict[str, float], count: int = 1):
        """
        Update metrics with new values.

        Args:
            metrics: Dictionary of metric name -> value
            count: Number of samples (for weighted averaging)

        Example:
            >>> tracker = MetricsTracker()
            >>> tracker.update({"loss": 0.5, "accuracy": 0.9}, count=32)
        """
        for key, value in metrics.items():
            self.metrics[key].append(value)
            self.running_sums[key] += value * count
            self.running_counts[key] += count

    def get_average(self, key: str) -> float:
        """
        Get average value for a metric.

        Args:
            key: Metric name

        Returns:
            Average value

        Example:
            >>> avg_loss = tracker.get_average("loss")
        """
        if key not in self.running_counts or self.running_counts[key] == 0:
            return 0.0

        return self.running_sums[key] / self.running_counts[key]

    def get_all_averages(self) -> Dict[str, float]:
        """
        Get averages for all tracked metrics.

        Returns:
            Dictionary of metric name -> average value

        Example:
            >>> averages = tracker.get_all_averages()
            >>> print(f"Loss: {averages['loss']:.4f}")
        """
        return {key: self.get_average(key) for key in self.running_counts.keys()}

    def get_latest(self, key: str) -> float:
        """
        Get latest value for a metric.

        Args:
            key: Metric name

        Returns:
            Latest value
        """
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0

        return self.metrics[key][-1]

    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.running_sums.clear()
        self.running_counts.clear()

    def summary(self) -> str:
        """
        Get a summary string of all metrics.

        Returns:
            Formatted summary string

        Example:
            >>> print(tracker.summary())
            loss: 0.5000, accuracy: 0.9500
        """
        averages = self.get_all_averages()
        return ", ".join(f"{k}: {v:.4f}" for k, v in averages.items())


def accuracy(predictions, targets) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Predicted labels or logits
        targets: Ground truth labels

    Returns:
        Accuracy as float

    Example:
        >>> preds = torch.tensor([0, 1, 2, 1])
        >>> targets = torch.tensor([0, 1, 1, 1])
        >>> acc = accuracy(preds, targets)  # 0.75
    """
    if TORCH_AVAILABLE and isinstance(predictions, torch.Tensor):
        if predictions.ndim > 1:  # Logits
            predictions = predictions.argmax(dim=-1)
        if targets.ndim > 1:
            targets = targets.argmax(dim=-1)

        correct = (predictions == targets).sum().item()
        total = targets.size(0)
        return correct / total

    elif NUMPY_AVAILABLE and isinstance(predictions, np.ndarray):
        if predictions.ndim > 1:
            predictions = predictions.argmax(axis=-1)
        if targets.ndim > 1:
            targets = targets.argmax(axis=-1)

        correct = (predictions == targets).sum()
        total = len(targets)
        return correct / total

    else:
        # Python lists
        if isinstance(predictions[0], (list, tuple)):
            predictions = [max(enumerate(p), key=lambda x: x[1])[0] for p in predictions]
        if isinstance(targets[0], (list, tuple)):
            targets = [max(enumerate(t), key=lambda x: x[1])[0] for t in targets]

        correct = sum(p == t for p, t in zip(predictions, targets))
        return correct / len(targets)


def precision(predictions, targets, num_classes: Optional[int] = None, average: str = "macro") -> float:
    """
    Compute precision score.

    Args:
        predictions: Predicted labels
        targets: Ground truth labels
        num_classes: Number of classes
        average: Averaging method ("macro", "micro", "weighted")

    Returns:
        Precision score

    Example:
        >>> prec = precision(preds, targets, num_classes=3, average="macro")
    """
    if not NUMPY_AVAILABLE:
        warnings.warn("NumPy required for precision calculation")
        return 0.0

    # Convert to numpy
    if TORCH_AVAILABLE and isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()

    if predictions.ndim > 1:
        predictions = predictions.argmax(axis=-1)
    if targets.ndim > 1:
        targets = targets.argmax(axis=-1)

    # Compute confusion matrix
    if num_classes is None:
        num_classes = max(predictions.max(), targets.max()) + 1

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(predictions.flatten(), targets.flatten()):
        cm[t, p] += 1

    # Compute precision per class
    precisions = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp

        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
        else:
            precisions.append(0.0)

    if average == "macro":
        return sum(precisions) / len(precisions)
    elif average == "micro":
        tp_total = np.diag(cm).sum()
        fp_total = cm.sum() - tp_total
        return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    elif average == "weighted":
        weights = cm.sum(axis=1)
        return sum(p * w for p, w in zip(precisions, weights)) / weights.sum()
    else:
        return precisions


def recall(predictions, targets, num_classes: Optional[int] = None, average: str = "macro") -> float:
    """
    Compute recall score.

    Args:
        predictions: Predicted labels
        targets: Ground truth labels
        num_classes: Number of classes
        average: Averaging method ("macro", "micro", "weighted")

    Returns:
        Recall score

    Example:
        >>> rec = recall(preds, targets, num_classes=3, average="macro")
    """
    if not NUMPY_AVAILABLE:
        warnings.warn("NumPy required for recall calculation")
        return 0.0

    # Convert to numpy
    if TORCH_AVAILABLE and isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()

    if predictions.ndim > 1:
        predictions = predictions.argmax(axis=-1)
    if targets.ndim > 1:
        targets = targets.argmax(axis=-1)

    # Compute confusion matrix
    if num_classes is None:
        num_classes = max(predictions.max(), targets.max()) + 1

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(predictions.flatten(), targets.flatten()):
        cm[t, p] += 1

    # Compute recall per class
    recalls = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp

        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
        else:
            recalls.append(0.0)

    if average == "macro":
        return sum(recalls) / len(recalls)
    elif average == "micro":
        tp_total = np.diag(cm).sum()
        fn_total = cm.sum() - tp_total
        return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    elif average == "weighted":
        weights = cm.sum(axis=1)
        return sum(r * w for r, w in zip(recalls, weights)) / weights.sum()
    else:
        return recalls


def f1_score(predictions, targets, num_classes: Optional[int] = None, average: str = "macro") -> float:
    """
    Compute F1 score.

    Args:
        predictions: Predicted labels
        targets: Ground truth labels
        num_classes: Number of classes
        average: Averaging method ("macro", "micro", "weighted")

    Returns:
        F1 score

    Example:
        >>> f1 = f1_score(preds, targets, num_classes=3, average="macro")
    """
    prec = precision(predictions, targets, num_classes, average)
    rec = recall(predictions, targets, num_classes, average)

    if prec + rec == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec)


def confusion_matrix(predictions, targets, num_classes: Optional[int] = None):
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted labels
        targets: Ground truth labels
        num_classes: Number of classes

    Returns:
        Confusion matrix [num_classes, num_classes]

    Example:
        >>> cm = confusion_matrix(preds, targets, num_classes=3)
        >>> print(cm)
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("NumPy required for confusion matrix")

    # Convert to numpy
    if TORCH_AVAILABLE and isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()

    if predictions.ndim > 1:
        predictions = predictions.argmax(axis=-1)
    if targets.ndim > 1:
        targets = targets.argmax(axis=-1)

    if num_classes is None:
        num_classes = max(predictions.max(), targets.max()) + 1

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(predictions.flatten(), targets.flatten()):
        cm[t, p] += 1

    return cm


def classification_report(
    predictions,
    targets,
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Generate a classification report.

    Args:
        predictions: Predicted labels
        targets: Ground truth labels
        num_classes: Number of classes
        class_names: Names for each class

    Returns:
        Formatted classification report

    Example:
        >>> report = classification_report(preds, targets, class_names=["cat", "dog"])
        >>> print(report)
    """
    if num_classes is None:
        if TORCH_AVAILABLE and isinstance(predictions, torch.Tensor):
            num_classes = max(predictions.max().item(), targets.max().item()) + 1
        elif NUMPY_AVAILABLE and isinstance(predictions, np.ndarray):
            num_classes = max(predictions.max(), targets.max()) + 1
        else:
            num_classes = max(max(predictions), max(targets)) + 1

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    # Compute metrics per class
    prec_per_class = precision(predictions, targets, num_classes, average=None)
    rec_per_class = recall(predictions, targets, num_classes, average=None)
    f1_per_class = [
        2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        for p, r in zip(prec_per_class, rec_per_class)
    ]

    # Build report
    report = "Classification Report:\n"
    report += "=" * 60 + "\n"
    report += f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n"
    report += "-" * 60 + "\n"

    for i, name in enumerate(class_names):
        report += f"{name:<20} {prec_per_class[i]:<12.4f} {rec_per_class[i]:<12.4f} {f1_per_class[i]:<12.4f}\n"

    report += "-" * 60 + "\n"

    # Overall metrics
    macro_prec = sum(prec_per_class) / len(prec_per_class)
    macro_rec = sum(rec_per_class) / len(rec_per_class)
    macro_f1 = sum(f1_per_class) / len(f1_per_class)

    report += f"{'Macro Avg':<20} {macro_prec:<12.4f} {macro_rec:<12.4f} {macro_f1:<12.4f}\n"
    report += f"{'Accuracy':<20} {accuracy(predictions, targets):<12.4f}\n"
    report += "=" * 60

    return report


def compute_metrics(predictions, targets, task: str = "classification") -> Dict[str, float]:
    """
    Compute standard metrics for a given task.

    Args:
        predictions: Model predictions
        targets: Ground truth
        task: Task type ("classification", "regression", "multilabel")

    Returns:
        Dictionary of metrics

    Example:
        >>> metrics = compute_metrics(preds, targets, task="classification")
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    metrics = {}

    if task == "classification":
        metrics["accuracy"] = accuracy(predictions, targets)
        metrics["precision"] = precision(predictions, targets)
        metrics["recall"] = recall(predictions, targets)
        metrics["f1"] = f1_score(predictions, targets)

    elif task == "regression":
        if TORCH_AVAILABLE and isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
            targets = targets.cpu().numpy()

        if NUMPY_AVAILABLE:
            # MSE
            metrics["mse"] = np.mean((predictions - targets) ** 2)
            # MAE
            metrics["mae"] = np.mean(np.abs(predictions - targets))
            # R2 score
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            metrics["r2"] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    elif task == "multilabel":
        # For multilabel classification
        metrics["accuracy"] = accuracy(predictions, targets)
        # Could add more multilabel-specific metrics

    return metrics


# Test function
def test_metrics():
    """Test metrics computation"""
    print("Testing Brain Metrics...")

    if TORCH_AVAILABLE:
        print("\n1. Testing Classification Metrics...")
        predictions = torch.tensor([0, 1, 2, 1, 0, 2])
        targets = torch.tensor([0, 1, 1, 1, 0, 2])

        acc = accuracy(predictions, targets)
        prec = precision(predictions, targets, num_classes=3)
        rec = recall(predictions, targets, num_classes=3)
        f1 = f1_score(predictions, targets, num_classes=3)

        print(f"   Accuracy: {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall: {rec:.4f}")
        print(f"   F1 Score: {f1:.4f}")

        print("\n2. Testing MetricsTracker...")
        tracker = MetricsTracker()
        tracker.update({"loss": 0.5, "accuracy": 0.9}, count=32)
        tracker.update({"loss": 0.3, "accuracy": 0.95}, count=32)

        print(f"   {tracker.summary()}")

        print("\n3. Testing Classification Report...")
        report = classification_report(
            predictions,
            targets,
            class_names=["cat", "dog", "bird"]
        )
        print(report)

    print("\n✓ Metrics tests complete!")


if __name__ == "__main__":
    test_metrics()
