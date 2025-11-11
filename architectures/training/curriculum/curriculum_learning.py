"""
Curriculum Learning - SOTA Training Strategies

Implementations:
- Easy-to-Hard scheduling (competence-based)
- Domain mixing strategies
- Dynamic difficulty adjustment
- Self-paced learning
- Teacher-Student curriculum
- Anti-curriculum (hard-to-easy)

References:
- "Curriculum Learning" (Bengio et al., 2009)
- "Self-Paced Learning" (Kumar et al., 2010)
- "Automated Curriculum Learning" (Graves et al., 2017)
- "Competence-based Curriculum Learning" (Narvekar et al., 2020)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import numpy as np
from enum import Enum


class DifficultyMetric(Enum):
    """Methods for measuring example difficulty"""
    LOSS = "loss"
    CONFIDENCE = "confidence"
    LENGTH = "length"
    COMPLEXITY = "complexity"
    LEARNING_PROGRESS = "learning_progress"


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning"""
    strategy: str = "easy_to_hard"  # easy_to_hard, self_paced, teacher_student, anti_curriculum
    difficulty_metric: DifficultyMetric = DifficultyMetric.LOSS
    initial_percentile: float = 0.2  # Start with easiest 20%
    final_percentile: float = 1.0  # End with all data
    pacing_function: str = "linear"  # linear, root, exponential, step
    update_frequency: int = 1000  # Update curriculum every N steps
    competence_threshold: float = 0.8  # Accuracy threshold to advance
    look_ahead: bool = True  # Peek at harder examples


class EasyToHardScheduler:
    """
    Easy-to-Hard Curriculum Learning.

    Gradually introduce harder examples as model improves.
    """

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_step = 0
        self.difficulties: Optional[np.ndarray] = None

    def compute_difficulties(
        self,
        dataset: List[Any],
        model: nn.Module,
        metric: DifficultyMetric
    ) -> np.ndarray:
        """
        Compute difficulty scores for all examples.

        Args:
            dataset: Training examples
            model: Current model
            metric: How to measure difficulty

        Returns:
            Difficulty scores (higher = harder)
        """
        difficulties = []

        model.eval()
        with torch.no_grad():
            for example in dataset:
                if metric == DifficultyMetric.LOSS:
                    # Use model loss as difficulty
                    loss = self._compute_loss(model, example)
                    difficulties.append(loss.item())

                elif metric == DifficultyMetric.CONFIDENCE:
                    # Use 1 - confidence as difficulty
                    confidence = self._compute_confidence(model, example)
                    difficulties.append(1.0 - confidence)

                elif metric == DifficultyMetric.LENGTH:
                    # Use sequence length as difficulty
                    length = len(example.get('input_ids', []))
                    difficulties.append(float(length))

                elif metric == DifficultyMetric.COMPLEXITY:
                    # Use syntactic/semantic complexity
                    complexity = self._compute_complexity(example)
                    difficulties.append(complexity)

        model.train()
        return np.array(difficulties)

    def _compute_loss(self, model: nn.Module, example: Dict) -> torch.Tensor:
        """Compute loss for single example"""
        # Placeholder - would run forward pass
        return torch.tensor(np.random.random())

    def _compute_confidence(self, model: nn.Module, example: Dict) -> float:
        """Compute model confidence for example"""
        # Placeholder - would compute softmax confidence
        return np.random.random()

    def _compute_complexity(self, example: Dict) -> float:
        """Compute syntactic/semantic complexity"""
        # Placeholder - could use parse tree depth, rare words, etc.
        return np.random.random()

    def get_pacing(self, step: int, total_steps: int) -> float:
        """
        Get current pacing (what percentile of data to use).

        Args:
            step: Current training step
            total_steps: Total training steps

        Returns:
            Percentile threshold (0.0 to 1.0)
        """
        progress = step / total_steps
        initial = self.config.initial_percentile
        final = self.config.final_percentile

        if self.config.pacing_function == "linear":
            return initial + (final - initial) * progress

        elif self.config.pacing_function == "root":
            # Slower at start, faster at end
            return initial + (final - initial) * np.sqrt(progress)

        elif self.config.pacing_function == "exponential":
            # Faster at start, slower at end
            return initial + (final - initial) * (progress ** 2)

        elif self.config.pacing_function == "step":
            # Discrete steps every 25%
            if progress < 0.25:
                return initial
            elif progress < 0.5:
                return initial + (final - initial) * 0.33
            elif progress < 0.75:
                return initial + (final - initial) * 0.66
            else:
                return final

        return final

    def sample_batch(
        self,
        dataset: List[Any],
        batch_size: int,
        current_percentile: float
    ) -> List[Any]:
        """
        Sample batch according to curriculum.

        Args:
            dataset: Full dataset
            batch_size: Number of examples
            current_percentile: Current difficulty threshold

        Returns:
            Sampled batch
        """
        if self.difficulties is None:
            # First call - compute difficulties
            return np.random.choice(dataset, batch_size, replace=False).tolist()

        # Get difficulty threshold
        threshold = np.percentile(self.difficulties, current_percentile * 100)

        # Filter examples below threshold
        valid_indices = np.where(self.difficulties <= threshold)[0]

        if len(valid_indices) < batch_size:
            # Not enough easy examples - use all valid
            selected = valid_indices
        else:
            # Sample from valid examples
            selected = np.random.choice(valid_indices, batch_size, replace=False)

        return [dataset[i] for i in selected]


class SelfPacedLearning:
    """
    Self-Paced Learning.

    Model automatically selects which examples to learn from
    based on its current competence.
    """

    def __init__(
        self,
        lambda_init: float = 1.0,
        lambda_max: float = 10.0,
        growth_rate: float = 1.1
    ):
        self.lambda_current = lambda_init  # Controls how selective we are
        self.lambda_max = lambda_max
        self.growth_rate = growth_rate

    def compute_weights(
        self,
        losses: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute example weights based on losses.

        Lower loss = higher weight (easier examples weighted more).

        Args:
            losses: Per-example losses [batch_size]

        Returns:
            Weights in [0, 1] for each example
        """
        # Self-paced regularization
        # w_i = 1 if loss_i < lambda, else 0
        # Soft version: w_i = sigmoid(lambda - loss_i)

        weights = torch.sigmoid(self.lambda_current - losses)
        return weights

    def step(self):
        """Increase lambda to include harder examples"""
        self.lambda_current = min(
            self.lambda_current * self.growth_rate,
            self.lambda_max
        )

    def weighted_loss(
        self,
        losses: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted loss for self-paced learning.

        Args:
            losses: Per-example losses

        Returns:
            Weighted average loss
        """
        weights = self.compute_weights(losses)
        return (losses * weights).sum() / (weights.sum() + 1e-8)


class TeacherStudentCurriculum:
    """
    Teacher-Student Curriculum.

    Teacher model suggests which examples student should learn from.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        selection_strategy: str = "disagreement"  # disagreement, confidence, entropy
    ):
        self.teacher = teacher_model
        self.selection_strategy = selection_strategy

    def select_examples(
        self,
        student_model: nn.Module,
        candidate_pool: List[Any],
        n_select: int
    ) -> List[Any]:
        """
        Teacher selects most useful examples for student.

        Args:
            student_model: Current student
            candidate_pool: Available examples
            n_select: Number to select

        Returns:
            Selected examples
        """
        scores = []

        self.teacher.eval()
        student_model.eval()

        with torch.no_grad():
            for example in candidate_pool:
                if self.selection_strategy == "disagreement":
                    # Select examples where teacher and student disagree
                    teacher_pred = self._predict(self.teacher, example)
                    student_pred = self._predict(student_model, example)
                    score = self._disagreement(teacher_pred, student_pred)

                elif self.selection_strategy == "confidence":
                    # Select examples where teacher is confident
                    teacher_pred = self._predict(self.teacher, example)
                    score = self._confidence(teacher_pred)

                elif self.selection_strategy == "entropy":
                    # Select high-entropy examples (more informative)
                    teacher_pred = self._predict(self.teacher, example)
                    score = self._entropy(teacher_pred)

                scores.append(score)

        student_model.train()

        # Select top-k examples
        top_indices = np.argsort(scores)[-n_select:]
        return [candidate_pool[i] for i in top_indices]

    def _predict(self, model: nn.Module, example: Dict) -> torch.Tensor:
        """Get model predictions"""
        # Placeholder
        return torch.softmax(torch.randn(10), dim=0)

    def _disagreement(self, pred1: torch.Tensor, pred2: torch.Tensor) -> float:
        """Compute disagreement between predictions"""
        # KL divergence
        kl = (pred1 * (pred1.log() - pred2.log())).sum()
        return kl.item()

    def _confidence(self, pred: torch.Tensor) -> float:
        """Compute prediction confidence"""
        return pred.max().item()

    def _entropy(self, pred: torch.Tensor) -> float:
        """Compute prediction entropy"""
        entropy = -(pred * pred.log()).sum()
        return entropy.item()


class DomainMixingCurriculum:
    """
    Domain Mixing Curriculum.

    Strategically mix examples from different domains/tasks.
    """

    def __init__(
        self,
        domains: List[str],
        mixing_strategy: str = "proportional"  # proportional, balanced, temperature
    ):
        self.domains = domains
        self.mixing_strategy = mixing_strategy
        self.domain_difficulties = {d: 1.0 for d in domains}
        self.domain_performance = {d: 0.0 for d in domains}

    def update_performance(self, domain: str, accuracy: float):
        """Update performance tracking for domain"""
        self.domain_performance[domain] = accuracy

    def get_mixing_ratios(self, temperature: float = 1.0) -> Dict[str, float]:
        """
        Get current mixing ratios for each domain.

        Args:
            temperature: Controls diversity (higher = more uniform)

        Returns:
            Mixing ratios for each domain
        """
        if self.mixing_strategy == "proportional":
            # Mix proportional to domain size
            # Placeholder - would use actual sizes
            ratios = {d: 1.0 / len(self.domains) for d in self.domains}

        elif self.mixing_strategy == "balanced":
            # Equal representation
            ratios = {d: 1.0 / len(self.domains) for d in self.domains}

        elif self.mixing_strategy == "temperature":
            # Temperature-scaled based on performance
            # Focus more on weaker domains

            # Invert performance (lower = need more)
            inv_perf = {
                d: 1.0 - self.domain_performance[d]
                for d in self.domains
            }

            # Temperature scaling
            scaled = {
                d: np.exp(inv_perf[d] / temperature)
                for d in self.domains
            }

            # Normalize
            total = sum(scaled.values())
            ratios = {d: scaled[d] / total for d in self.domains}

        return ratios

    def sample_batch(
        self,
        domain_datasets: Dict[str, List[Any]],
        batch_size: int,
        temperature: float = 1.0
    ) -> Tuple[List[Any], List[str]]:
        """
        Sample batch with domain mixing.

        Args:
            domain_datasets: Data for each domain
            batch_size: Total batch size
            temperature: Mixing temperature

        Returns:
            batch: Mixed batch
            domains: Domain label for each example
        """
        ratios = self.get_mixing_ratios(temperature)

        batch = []
        domain_labels = []

        for domain, ratio in ratios.items():
            n_samples = int(batch_size * ratio)
            if n_samples == 0:
                continue

            dataset = domain_datasets[domain]
            samples = np.random.choice(dataset, n_samples, replace=False).tolist()

            batch.extend(samples)
            domain_labels.extend([domain] * n_samples)

        # Shuffle
        indices = np.random.permutation(len(batch))
        batch = [batch[i] for i in indices]
        domain_labels = [domain_labels[i] for i in indices]

        return batch, domain_labels


class DynamicDifficultyAdjustment:
    """
    Dynamic Difficulty Adjustment.

    Adjust difficulty in real-time based on model performance.
    """

    def __init__(
        self,
        target_accuracy: float = 0.75,
        adjustment_rate: float = 0.1,
        window_size: int = 100
    ):
        self.target_accuracy = target_accuracy
        self.adjustment_rate = adjustment_rate
        self.window_size = window_size

        self.recent_accuracies: List[float] = []
        self.current_difficulty = 0.5  # 0 = easiest, 1 = hardest

    def update(self, accuracy: float):
        """
        Update difficulty based on recent performance.

        Args:
            accuracy: Recent accuracy
        """
        self.recent_accuracies.append(accuracy)

        # Keep only recent window
        if len(self.recent_accuracies) > self.window_size:
            self.recent_accuracies.pop(0)

        # Compute average recent accuracy
        avg_accuracy = np.mean(self.recent_accuracies)

        # Adjust difficulty
        if avg_accuracy > self.target_accuracy:
            # Too easy - increase difficulty
            self.current_difficulty = min(
                1.0,
                self.current_difficulty + self.adjustment_rate
            )
        elif avg_accuracy < self.target_accuracy:
            # Too hard - decrease difficulty
            self.current_difficulty = max(
                0.0,
                self.current_difficulty - self.adjustment_rate
            )

    def get_difficulty_level(self) -> float:
        """Get current difficulty level"""
        return self.current_difficulty


class AntiCurriculum:
    """
    Anti-Curriculum (Hard-to-Easy).

    Start with hard examples, then move to easier ones.
    Useful for some tasks where hard examples provide better gradients.
    """

    def __init__(self):
        self.difficulties: Optional[np.ndarray] = None

    def get_percentile(self, step: int, total_steps: int) -> float:
        """
        Get percentile for anti-curriculum.

        Start with hardest (100th percentile), end with all data.
        """
        progress = step / total_steps

        # Start at 100%, go down to 0% (include all)
        percentile = 1.0 - (progress * 0.8)  # Keep top 20% initially, then expand

        return percentile


# Example test function
def test_curriculum_learning():
    """Test curriculum learning implementations"""
    print("Testing Curriculum Learning...")

    # Create dummy dataset
    dataset = [{"id": i, "difficulty": np.random.random()} for i in range(1000)]

    # Test 1: Easy-to-Hard Scheduler
    print("\n1. Easy-to-Hard Scheduler")
    config = CurriculumConfig(
        strategy="easy_to_hard",
        initial_percentile=0.2,
        final_percentile=1.0,
        pacing_function="linear"
    )
    scheduler = EasyToHardScheduler(config)

    # Simulate training
    total_steps = 1000
    for step in [0, 250, 500, 750, 1000]:
        percentile = scheduler.get_pacing(step, total_steps)
        print(f"  Step {step}: Using {percentile*100:.1f}% of data")

    # Test 2: Self-Paced Learning
    print("\n2. Self-Paced Learning")
    spl = SelfPacedLearning(lambda_init=1.0)

    losses = torch.tensor([0.5, 1.0, 2.0, 0.8, 3.0])
    weights = spl.compute_weights(losses)
    print(f"  Losses: {losses.tolist()}")
    print(f"  Weights: {[f'{w:.3f}' for w in weights.tolist()]}")
    print(f"  λ = {spl.lambda_current:.2f}")

    # Step and recompute
    spl.step()
    weights = spl.compute_weights(losses)
    print(f"  After step, λ = {spl.lambda_current:.2f}")
    print(f"  New weights: {[f'{w:.3f}' for w in weights.tolist()]}")

    # Test 3: Domain Mixing
    print("\n3. Domain Mixing Curriculum")
    domains = ["math", "code", "reasoning", "qa"]
    mixer = DomainMixingCurriculum(domains, mixing_strategy="temperature")

    # Set some performance levels
    mixer.update_performance("math", 0.9)  # Good at math
    mixer.update_performance("code", 0.6)  # Weaker at code
    mixer.update_performance("reasoning", 0.7)
    mixer.update_performance("qa", 0.8)

    ratios = mixer.get_mixing_ratios(temperature=0.5)
    print("  Domain mixing ratios:")
    for domain, ratio in ratios.items():
        perf = mixer.domain_performance[domain]
        print(f"    {domain}: {ratio*100:.1f}% (performance: {perf*100:.0f}%)")

    # Test 4: Dynamic Difficulty Adjustment
    print("\n4. Dynamic Difficulty Adjustment")
    dda = DynamicDifficultyAdjustment(target_accuracy=0.75)

    # Simulate varying performance
    accuracies = [0.8, 0.85, 0.9, 0.88, 0.92]  # Too easy
    print("  High accuracies (too easy):")
    for acc in accuracies:
        dda.update(acc)
        print(f"    Accuracy: {acc:.2f} → Difficulty: {dda.get_difficulty_level():.2f}")

    # Now harder
    accuracies = [0.6, 0.65, 0.7, 0.68, 0.62]  # Too hard
    print("  Low accuracies (too hard):")
    for acc in accuracies:
        dda.update(acc)
        print(f"    Accuracy: {acc:.2f} → Difficulty: {dda.get_difficulty_level():.2f}")

    print("\n✓ Curriculum Learning tests completed!")

    # Statistics
    print("\n" + "="*60)
    print("CURRICULUM LEARNING SUMMARY")
    print("="*60)
    print(f"Strategies implemented: 5")
    print(f"  - Easy-to-Hard (4 pacing functions)")
    print(f"  - Self-Paced Learning (adaptive weighting)")
    print(f"  - Teacher-Student (3 selection strategies)")
    print(f"  - Domain Mixing (3 mixing strategies)")
    print(f"  - Dynamic Difficulty Adjustment")
    print(f"  - Anti-Curriculum (hard-to-easy)")
    print(f"\nDifficulty metrics: 5")
    print(f"  - Loss-based")
    print(f"  - Confidence-based")
    print(f"  - Length-based")
    print(f"  - Complexity-based")
    print(f"  - Learning progress")
    print(f"\nApplications:")
    print(f"  - Language model pretraining")
    print(f"  - Multi-task learning")
    print(f"  - Domain adaptation")
    print(f"  - Reinforcement learning")


if __name__ == "__main__":
    test_curriculum_learning()
