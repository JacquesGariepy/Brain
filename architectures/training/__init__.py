"""
Advanced Training Techniques - Production-Ready Optimizations

Critical training optimizations for efficiency and scale.
"""

from .advanced_training import (
    # Gradient Checkpointing
    CheckpointConfig,
    CheckpointWrapper,
    apply_gradient_checkpointing,

    # Mixed Precision
    MixedPrecisionConfig,
    MixedPrecisionTrainer,

    # Gradient Accumulation
    GradientAccumulationConfig,
    GradientAccumulator,

    # LR Schedules
    LRSchedule,
    WarmupCosineSchedule,
    InverseSqrtSchedule,

    # Complete Training
    TrainingConfig,
    Trainer
)

__all__ = [
    # Gradient Checkpointing
    'CheckpointConfig',
    'CheckpointWrapper',
    'apply_gradient_checkpointing',

    # Mixed Precision
    'MixedPrecisionConfig',
    'MixedPrecisionTrainer',

    # Gradient Accumulation
    'GradientAccumulationConfig',
    'GradientAccumulator',

    # LR Schedules
    'LRSchedule',
    'WarmupCosineSchedule',
    'InverseSqrtSchedule',

    # Complete Training
    'TrainingConfig',
    'Trainer'
]
