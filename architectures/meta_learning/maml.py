"""
MAML (Model-Agnostic Meta-Learning)

Foundational meta-learning algorithm for few-shot learning (2017-2025).

Key features:
- Model-agnostic (works with any model trained with gradient descent)
- Fast adaptation to new tasks with few examples
- Second-order optimization (MAML) or first-order (FOMAML)
- Learn good initialization for rapid fine-tuning

Algorithm:
1. Sample batch of tasks
2. For each task: adapt with inner loop (few gradient steps)
3. Meta-update: improve initialization based on task performance

References:
- "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (Finn et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import OrderedDict
import copy


@dataclass
class MAMLConfig:
    """Configuration for MAML"""
    # Inner loop (task adaptation)
    inner_lr: float = 0.01  # Learning rate for task adaptation
    inner_steps: int = 5  # Number of gradient steps per task

    # Outer loop (meta-learning)
    outer_lr: float = 0.001  # Meta learning rate

    # Algorithm variant
    first_order: bool = False  # Use FOMAML (first-order approximation)

    # Training
    num_tasks_per_batch: int = 4  # Number of tasks per meta-batch

    # Task setup
    n_way: int = 5  # N-way classification
    k_shot: int = 1  # K-shot (examples per class in support set)
    q_query: int = 15  # Query examples per class


class MAML:
    """
    Complete MAML implementation.

    Learns good model initialization for fast adaptation.
    Works with any differentiable model.
    """

    def __init__(
        self,
        model: nn.Module,
        config: MAMLConfig
    ):
        self.model = model
        self.config = config

        # Meta-optimizer (for outer loop)
        self.meta_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.outer_lr
        )

    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        fast_weights: Optional[OrderedDict] = None
    ) -> Tuple[OrderedDict, float]:
        """
        Inner loop: Adapt model to task using support set.

        Args:
            support_x: Support set inputs (n_way * k_shot, ...)
            support_y: Support set labels (n_way * k_shot,)
            fast_weights: Initial weights (None = use model weights)

        Returns:
            fast_weights: Adapted weights
            support_loss: Loss on support set
        """
        if fast_weights is None:
            fast_weights = OrderedDict(self.model.named_parameters())

        # Perform gradient steps on support set
        for step in range(self.config.inner_steps):
            # Forward pass with fast weights
            logits = self._forward_with_weights(support_x, fast_weights)
            loss = F.cross_entropy(logits, support_y)

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                fast_weights.values(),
                create_graph=not self.config.first_order,  # For second-order MAML
                allow_unused=True
            )

            # Update fast weights
            fast_weights = OrderedDict(
                (name, param - self.config.inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), grads)
                if grad is not None
            )

        # Compute final support loss
        logits = self._forward_with_weights(support_x, fast_weights)
        support_loss = F.cross_entropy(logits, support_y)

        return fast_weights, support_loss

    def outer_loop(
        self,
        tasks: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Outer loop: Meta-update based on query set performance.

        Args:
            tasks: List of task dictionaries, each with:
                - support_x, support_y: Support set
                - query_x, query_y: Query set

        Returns:
            metrics: Dict with losses and accuracy
        """
        self.meta_optimizer.zero_grad()

        meta_loss = 0.0
        meta_acc = 0.0

        for task in tasks:
            support_x = task['support_x']
            support_y = task['support_y']
            query_x = task['query_x']
            query_y = task['query_y']

            # Inner loop: adapt to task
            fast_weights, support_loss = self.inner_loop(
                support_x, support_y
            )

            # Evaluate on query set
            query_logits = self._forward_with_weights(query_x, fast_weights)
            query_loss = F.cross_entropy(query_logits, query_y)

            # Accumulate meta loss
            meta_loss += query_loss

            # Compute accuracy
            query_pred = query_logits.argmax(dim=1)
            accuracy = (query_pred == query_y).float().mean()
            meta_acc += accuracy

        # Average over tasks
        meta_loss = meta_loss / len(tasks)
        meta_acc = meta_acc / len(tasks)

        # Meta-update
        meta_loss.backward()
        self.meta_optimizer.step()

        return {
            'meta_loss': meta_loss.item(),
            'meta_accuracy': meta_acc.item()
        }

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        steps: Optional[int] = None
    ) -> nn.Module:
        """
        Adapt model to new task.

        Args:
            support_x: Support set inputs
            support_y: Support set labels
            steps: Number of adaptation steps (default: config.inner_steps)

        Returns:
            Adapted model (copy of self.model with updated weights)
        """
        if steps is None:
            steps = self.config.inner_steps

        # Create model copy
        adapted_model = copy.deepcopy(self.model)

        # Optimizer for adaptation
        optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.config.inner_lr
        )

        # Adapt
        adapted_model.train()
        for _ in range(steps):
            optimizer.zero_grad()
            logits = adapted_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            loss.backward()
            optimizer.step()

        return adapted_model

    def _forward_with_weights(
        self,
        x: torch.Tensor,
        weights: OrderedDict
    ) -> torch.Tensor:
        """
        Forward pass using specific weights.

        This is needed for MAML to compute gradients through inner loop.
        """
        # Functional forward pass
        # Implementation depends on model architecture
        # For simplicity, we use a generic approach

        # Save original parameters
        original_params = OrderedDict(self.model.named_parameters())

        # Temporarily set new parameters
        for name, param in weights.items():
            if name in dict(self.model.named_parameters()):
                dict(self.model.named_parameters())[name].data = param.data

        # Forward pass
        output = self.model(x)

        # Restore original parameters
        for name, param in original_params.items():
            dict(self.model.named_parameters())[name].data = param.data

        return output

    def evaluate(
        self,
        tasks: List[Dict[str, torch.Tensor]],
        adaptation_steps: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate on test tasks.

        Args:
            tasks: List of test tasks
            adaptation_steps: Steps for adaptation

        Returns:
            metrics: Average accuracy and loss
        """
        self.model.eval()

        total_accuracy = 0.0
        total_loss = 0.0

        with torch.no_grad():
            for task in tasks:
                # Adapt to task
                adapted_model = self.adapt(
                    task['support_x'],
                    task['support_y'],
                    steps=adaptation_steps
                )

                # Evaluate on query set
                adapted_model.eval()
                query_logits = adapted_model(task['query_x'])
                query_loss = F.cross_entropy(query_logits, task['query_y'])

                query_pred = query_logits.argmax(dim=1)
                accuracy = (query_pred == task['query_y']).float().mean()

                total_accuracy += accuracy.item()
                total_loss += query_loss.item()

        return {
            'test_accuracy': total_accuracy / len(tasks),
            'test_loss': total_loss / len(tasks)
        }


def create_n_way_k_shot_task(
    data: torch.Tensor,
    labels: torch.Tensor,
    n_way: int,
    k_shot: int,
    q_query: int
) -> Dict[str, torch.Tensor]:
    """
    Create N-way K-shot task from dataset.

    Args:
        data: All data
        labels: All labels
        n_way: Number of classes
        k_shot: Examples per class in support set
        q_query: Examples per class in query set

    Returns:
        Task dictionary with support and query sets
    """
    # Sample N classes
    classes = torch.unique(labels)
    selected_classes = classes[torch.randperm(len(classes))[:n_way]]

    support_x = []
    support_y = []
    query_x = []
    query_y = []

    for i, cls in enumerate(selected_classes):
        # Get all examples of this class
        cls_indices = (labels == cls).nonzero(as_tuple=True)[0]
        cls_indices = cls_indices[torch.randperm(len(cls_indices))]

        # Split into support and query
        support_indices = cls_indices[:k_shot]
        query_indices = cls_indices[k_shot:k_shot + q_query]

        support_x.append(data[support_indices])
        support_y.append(torch.full((k_shot,), i, dtype=torch.long))

        query_x.append(data[query_indices])
        query_y.append(torch.full((q_query,), i, dtype=torch.long))

    return {
        'support_x': torch.cat(support_x),
        'support_y': torch.cat(support_y),
        'query_x': torch.cat(query_x),
        'query_y': torch.cat(query_y)
    }


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("MAML - Model-Agnostic Meta-Learning")
    print("="*80)

    # Create simple model for few-shot classification
    class SimpleConvNet(nn.Module):
        def __init__(self, num_classes=5):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, 3),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Linear(64 * 5 * 5, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    # Create MAML
    config = MAMLConfig(
        inner_lr=0.01,
        inner_steps=5,
        outer_lr=0.001,
        n_way=5,
        k_shot=1,
        q_query=15
    )

    model = SimpleConvNet(num_classes=config.n_way)
    maml = MAML(model, config)

    print(f"\nConfiguration:")
    print(f"  N-way: {config.n_way}")
    print(f"  K-shot: {config.k_shot}")
    print(f"  Inner LR: {config.inner_lr}")
    print(f"  Inner steps: {config.inner_steps}")
    print(f"  Outer LR: {config.outer_lr}")

    # Example task (dummy data)
    task = {
        'support_x': torch.randn(config.n_way * config.k_shot, 1, 28, 28),
        'support_y': torch.arange(config.n_way).repeat_interleave(config.k_shot),
        'query_x': torch.randn(config.n_way * config.q_query, 1, 28, 28),
        'query_y': torch.arange(config.n_way).repeat_interleave(config.q_query)
    }

    # Meta-training step
    tasks = [task] * config.num_tasks_per_batch
    metrics = maml.outer_loop(tasks)

    print(f"\nMeta-training metrics:")
    print(f"  Loss: {metrics['meta_loss']:.4f}")
    print(f"  Accuracy: {metrics['meta_accuracy']:.4f}")

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "="*80)
