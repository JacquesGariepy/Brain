"""
Configuration System - YAML-based configuration for all architectures

Allows configuring any model, training setup, or orchestration from YAML files.

Features:
- Load any architecture configuration from YAML
- Support for all SOTA models
- Training and evaluation configs
- Orchestration configs
- Presets and templates
- Environment variable interpolation
"""

from .config_loader import ConfigLoader, load_config
from .model_configs import ModelConfigFactory
from .training_configs import TrainingConfig
from .orchestration_configs import OrchestrationConfig

__all__ = [
    'ConfigLoader',
    'load_config',
    'ModelConfigFactory',
    'TrainingConfig',
    'OrchestrationConfig'
]
