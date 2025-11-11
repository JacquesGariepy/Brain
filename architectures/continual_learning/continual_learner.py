"""
Continual Learner - Universal wrapper for continual learning

Enables any PyTorch model to learn continuously without catastrophic forgetting.

Supported methods:
- EWC: Protects important weights using Fisher information
- iCaRL: Maintains exemplars and uses knowledge distillation
- LwF: Knowledge distillation from old model
- GEM: Gradient episodic memory
- A-GEM: Averaged GEM (more efficient)

Usage:
    model = YourModel()
    learner = ContinualLearner(model, method='ewc')

    # Train on task 1
    learner.train_task(task1_data, task_id=0)

    # Train on task 2 (without forgetting task 1)
    learner.train_task(task2_data, task_id=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import copy


class ContinualMethod(Enum):
    """Continual learning methods"""
    EWC = "ewc"
    ICARL = "icarl"
    LWF = "lwf"
    GEM = "gem"
    AGEM = "agem"
    NAIVE = "naive"  # No continual learning (baseline)


@dataclass
class ContinualConfig:
    """Configuration for continual learning"""
    # Method
    method: ContinualMethod = ContinualMethod.EWC

    # EWC parameters
    ewc_lambda: float = 5000.0  # Importance of old tasks
    ewc_gamma: float = 1.0  # Decay factor for multi-task

    # iCaRL parameters
    memory_size: int = 2000  # Number of exemplars to store
    num_classes: int = 10  # Total number of classes

    # LwF parameters
    lwf_temperature: float = 2.0  # Distillation temperature
    lwf_alpha: float = 0.5  # Weight of distillation loss

    # GEM/A-GEM parameters
    gem_memory_strength: float = 0.5  # Gradient projection strength
    gem_n_memories: int = 256  # Number of memories per task


class ContinualLearner:
    """
    Universal continual learning wrapper.

    Wraps any PyTorch model and adds continual learning capabilities.
    """

    def __init__(
        self,
        model: nn.Module,
        method: ContinualMethod = ContinualMethod.EWC,
        config: Optional[ContinualConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: Any PyTorch model
            method: Continual learning method
            config: Configuration
            device: Device to use
        """
        self.model = model
        self.method = method
        self.config = config or ContinualConfig(method=method)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)

        # Task tracking
        self.current_task = 0
        self.num_tasks = 0

        # Method-specific storage
        self.fisher_dict = {}  # EWC: Fisher information matrices
        self.optpar_dict = {}  # EWC: Optimal parameters
        self.exemplars = {}  # iCaRL: Stored exemplars
        self.old_model = None  # LwF: Previous model
        self.memory_data = []  # GEM: Memory buffer
        self.memory_labels = []

    def compute_fisher_information(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix for EWC.

        Args:
            dataloader: Data loader for current task
            num_samples: Number of samples to use (None = all)

        Returns:
            Dictionary mapping parameter names to Fisher diagonal
        """
        self.model.eval()

        # Initialize Fisher dict
        fisher_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param.data)

        # Accumulate Fisher information
        num_samples_seen = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            if num_samples is not None and num_samples_seen >= num_samples:
                break

            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.shape[0]

            # Forward pass
            output = self.model(data)

            # Log probabilities
            log_probs = F.log_softmax(output, dim=1)

            # Sample labels from model's predictions
            sampled_labels = torch.multinomial(log_probs.exp(), 1).squeeze()

            # Get log probability of sampled labels
            log_prob_sampled = log_probs.gather(1, sampled_labels.unsqueeze(1)).squeeze()

            # Compute gradients
            for i in range(batch_size):
                self.model.zero_grad()
                log_prob_sampled[i].backward(retain_graph=True)

                # Accumulate squared gradients (Fisher diagonal)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher_dict[name] += param.grad.data ** 2

            num_samples_seen += batch_size

        # Normalize
        for name in fisher_dict:
            fisher_dict[name] /= num_samples_seen

        return fisher_dict

    def ewc_loss(self, current_task: int) -> torch.Tensor:
        """
        Compute EWC regularization loss.

        Penalizes changes to important parameters from previous tasks.
        """
        loss = torch.tensor(0.0).to(self.device)

        for task in range(current_task):
            task_key = f"task_{task}"

            if task_key not in self.fisher_dict:
                continue

            for name, param in self.model.named_parameters():
                if name in self.fisher_dict[task_key] and param.requires_grad:
                    fisher = self.fisher_dict[task_key][name]
                    optpar = self.optpar_dict[task_key][name]

                    # EWC penalty: F * (θ - θ*)^2
                    loss += (fisher * (param - optpar) ** 2).sum()

        # Apply lambda and gamma
        loss *= self.config.ewc_lambda / 2.0
        loss *= (self.config.ewc_gamma ** (current_task - 1))

        return loss

    def save_task_parameters(self, task_id: int, fisher_dict: Dict[str, torch.Tensor]):
        """
        Save parameters and Fisher information for a task.

        Args:
            task_id: Task identifier
            fisher_dict: Fisher information matrix
        """
        task_key = f"task_{task_id}"

        # Save Fisher information
        self.fisher_dict[task_key] = fisher_dict

        # Save optimal parameters
        self.optpar_dict[task_key] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optpar_dict[task_key][name] = param.data.clone()

    def select_exemplars(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_exemplars: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Select exemplars for iCaRL using herding.

        Args:
            dataloader: Data loader for current task
            num_exemplars: Number of exemplars to select

        Returns:
            (exemplar_data, exemplar_labels)
        """
        self.model.eval()

        # Extract features for all samples
        all_features = []
        all_data = []
        all_labels = []

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self.device)

                # Get features (before final layer)
                # This assumes model has a method to get features
                if hasattr(self.model, 'get_features'):
                    features = self.model.get_features(data)
                else:
                    # Fallback: use model output
                    features = self.model(data)

                all_features.append(features.cpu())
                all_data.append(data.cpu())
                all_labels.append(labels)

        all_features = torch.cat(all_features, dim=0)
        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Compute class mean
        class_mean = all_features.mean(dim=0)

        # Herding: select exemplars closest to class mean
        exemplar_indices = []
        exemplar_sum = torch.zeros_like(class_mean)

        for k in range(num_exemplars):
            # Find sample that minimizes distance to class mean
            distances = torch.norm(
                (exemplar_sum + all_features) / (k + 1) - class_mean,
                dim=1,
                p=2
            )

            # Exclude already selected
            for idx in exemplar_indices:
                distances[idx] = float('inf')

            # Select best
            best_idx = distances.argmin().item()
            exemplar_indices.append(best_idx)
            exemplar_sum += all_features[best_idx]

        # Return selected exemplars
        exemplar_data = [all_data[i] for i in exemplar_indices]
        exemplar_labels = [all_labels[i] for i in exemplar_indices]

        return exemplar_data, exemplar_labels

    def distillation_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        old_outputs: torch.Tensor,
        temperature: float = 2.0,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Knowledge distillation loss (LwF).

        Combines classification loss with distillation from old model.
        """
        # Classification loss
        ce_loss = F.cross_entropy(outputs, targets)

        # Distillation loss
        soft_targets = F.softmax(old_outputs / temperature, dim=1)
        soft_outputs = F.log_softmax(outputs / temperature, dim=1)

        distill_loss = F.kl_div(
            soft_outputs,
            soft_targets,
            reduction='batchmean'
        ) * (temperature ** 2)

        # Combined loss
        total_loss = alpha * distill_loss + (1 - alpha) * ce_loss

        return total_loss

    def project_gradient(
        self,
        current_grad: torch.Tensor,
        memory_grad: torch.Tensor,
        margin: float = 0.5
    ) -> torch.Tensor:
        """
        Project gradient to not interfere with memory (GEM).

        Args:
            current_grad: Gradient on current task
            memory_grad: Gradient on memory
            margin: Margin for projection

        Returns:
            Projected gradient
        """
        # Flatten gradients
        current_flat = torch.cat([g.flatten() for g in current_grad if g is not None])
        memory_flat = torch.cat([g.flatten() for g in memory_grad if g is not None])

        # Check if projection is needed
        dot_product = torch.dot(current_flat, memory_flat)

        if dot_product < 0:
            # Project current gradient onto memory gradient
            memory_norm = torch.dot(memory_flat, memory_flat)

            if memory_norm > 0:
                projection = (dot_product / memory_norm) * memory_flat
                current_flat = current_flat - projection

        # Reshape back
        projected_grad = []
        start = 0
        for g in current_grad:
            if g is not None:
                size = g.numel()
                projected_grad.append(current_flat[start:start+size].view_as(g))
                start += size
            else:
                projected_grad.append(None)

        return projected_grad

    def train_task(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 10,
        task_id: Optional[int] = None,
        criterion: Optional[Callable] = None
    ):
        """
        Train on a new task with continual learning.

        Args:
            dataloader: Data loader for current task
            optimizer: Optimizer
            num_epochs: Number of training epochs
            task_id: Task identifier (auto-incremented if None)
            criterion: Loss criterion (default: CrossEntropy)
        """
        if task_id is None:
            task_id = self.num_tasks

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                output = self.model(data)

                # Task loss
                if self.method == ContinualMethod.LWF and self.old_model is not None:
                    # Distillation loss
                    with torch.no_grad():
                        old_output = self.old_model(data)
                    loss = self.distillation_loss(
                        output, target, old_output,
                        temperature=self.config.lwf_temperature,
                        alpha=self.config.lwf_alpha
                    )
                else:
                    # Standard loss
                    loss = criterion(output, target)

                # Add EWC penalty
                if self.method == ContinualMethod.EWC and task_id > 0:
                    loss += self.ewc_loss(task_id)

                # Backward
                loss.backward()

                # GEM gradient projection
                if self.method in [ContinualMethod.GEM, ContinualMethod.AGEM]:
                    if len(self.memory_data) > 0:
                        # Compute gradient on memory
                        memory_grads = []
                        for mem_data, mem_target in zip(self.memory_data, self.memory_labels):
                            mem_data = mem_data.to(self.device)
                            mem_target = mem_target.to(self.device)

                            optimizer.zero_grad()
                            mem_output = self.model(mem_data)
                            mem_loss = criterion(mem_output, mem_target)
                            mem_loss.backward()

                            # Store gradients
                            mem_grad = [p.grad.clone() if p.grad is not None else None
                                       for p in self.model.parameters()]
                            memory_grads.append(mem_grad)

                        # Project current gradient
                        current_grad = [p.grad.clone() if p.grad is not None else None
                                       for p in self.model.parameters()]

                        # Average memory gradients (A-GEM)
                        if self.method == ContinualMethod.AGEM:
                            avg_memory_grad = []
                            for i in range(len(current_grad)):
                                if current_grad[i] is not None:
                                    grads = [mg[i] for mg in memory_grads if mg[i] is not None]
                                    avg_memory_grad.append(torch.stack(grads).mean(dim=0))
                                else:
                                    avg_memory_grad.append(None)

                            projected = self.project_gradient(current_grad, avg_memory_grad)
                        else:
                            # GEM: project for each memory
                            projected = current_grad
                            for mem_grad in memory_grads:
                                projected = self.project_gradient(projected, mem_grad)

                        # Set projected gradients
                        for p, g in zip(self.model.parameters(), projected):
                            if g is not None:
                                p.grad = g

                # Optimizer step
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Task {task_id}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # After training, save task-specific information
        if self.method == ContinualMethod.EWC:
            # Compute and save Fisher information
            print("Computing Fisher information...")
            fisher_dict = self.compute_fisher_information(dataloader, num_samples=1000)
            self.save_task_parameters(task_id, fisher_dict)

        elif self.method == ContinualMethod.ICARL:
            # Select and store exemplars
            exemplars_per_class = self.config.memory_size // self.config.num_classes
            print(f"Selecting {exemplars_per_class} exemplars...")
            exemplar_data, exemplar_labels = self.select_exemplars(
                dataloader,
                num_exemplars=exemplars_per_class
            )
            self.exemplars[task_id] = (exemplar_data, exemplar_labels)

        elif self.method == ContinualMethod.LWF:
            # Save current model as old model
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()

        elif self.method in [ContinualMethod.GEM, ContinualMethod.AGEM]:
            # Store memory samples
            samples_per_task = self.config.gem_n_memories
            count = 0
            for data, target in dataloader:
                if count >= samples_per_task:
                    break
                self.memory_data.append(data)
                self.memory_labels.append(target)
                count += data.shape[0]

        # Increment task counter
        self.num_tasks = max(self.num_tasks, task_id + 1)
        self.current_task = task_id


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Continual Learner - Learn Without Forgetting")
    print("="*80)

    # Example model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    # Create continual learner with EWC
    config = ContinualConfig(
        method=ContinualMethod.EWC,
        ewc_lambda=5000.0
    )

    learner = ContinualLearner(model, method=ContinualMethod.EWC, config=config)

    print(f"\nMethod: {learner.method.value}")
    print(f"EWC lambda: {config.ewc_lambda}")

    # Simulated data loaders
    # In practice, these would be your actual data loaders for different tasks

    print("\nTraining on Task 0...")
    # learner.train_task(task0_dataloader, optimizer, task_id=0)

    print("\nTraining on Task 1 (with continual learning)...")
    # learner.train_task(task1_dataloader, optimizer, task_id=1)

    print("\n" + "="*80)
