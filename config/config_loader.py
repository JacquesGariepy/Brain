"""
Config Loader - Load and parse YAML configurations

Supports:
- Model configurations
- Training configurations
- Orchestration configurations
- Environment variable interpolation
- Config inheritance and merging
- Validation
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import re


class ConfigLoader:
    """
    Universal configuration loader for all Brain components.

    Loads YAML configs with support for:
    - Environment variable substitution
    - Config inheritance
    - Validation
    - Default values
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Args:
            config_dir: Directory containing config files
        """
        if config_dir is None:
            # Default to Brain/config directory
            config_dir = Path(__file__).parent
        self.config_dir = Path(config_dir)

    def load(self, config_path: str, **overrides) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file
            **overrides: Override specific config values

        Returns:
            Configuration dictionary
        """
        # Resolve path
        if not Path(config_path).is_absolute():
            config_path = self.config_dir / config_path

        # Load YAML
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Handle inheritance
        if 'inherits' in config:
            parent_config = self.load(config['inherits'])
            config = self._merge_configs(parent_config, config)
            del config['inherits']

        # Interpolate environment variables
        config = self._interpolate_env_vars(config)

        # Apply overrides
        config = self._apply_overrides(config, overrides)

        return config

    def _merge_configs(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge two configurations (override takes precedence).

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                result[key] = self._merge_configs(result[key], value)
            else:
                # Override value
                result[key] = value

        return result

    def _interpolate_env_vars(self, config: Any) -> Any:
        """
        Interpolate environment variables in config.

        Supports syntax: ${ENV_VAR} or ${ENV_VAR:default}

        Args:
            config: Configuration (dict, list, or str)

        Returns:
            Configuration with env vars interpolated
        """
        if isinstance(config, dict):
            return {k: self._interpolate_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._interpolate_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Find ${...} patterns
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

            def replace_env(match):
                env_var = match.group(1)
                default = match.group(2)

                value = os.environ.get(env_var)

                if value is None:
                    if default is not None:
                        return default
                    else:
                        raise ValueError(f"Environment variable {env_var} not set and no default provided")

                return value

            return re.sub(pattern, replace_env, config)
        else:
            return config

    def _apply_overrides(
        self,
        config: Dict[str, Any],
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply command-line or programmatic overrides.

        Supports dot notation: model.hidden_size=512

        Args:
            config: Base configuration
            overrides: Override values

        Returns:
            Configuration with overrides applied
        """
        result = config.copy()

        for key, value in overrides.items():
            # Support dot notation
            keys = key.split('.')
            current = result

            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            current[keys[-1]] = value

        return result

    def save(self, config: Dict[str, Any], output_path: str):
        """
        Save configuration to YAML file.

        Args:
            config: Configuration to save
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_config(config_path: str, **overrides) -> Dict[str, Any]:
    """
    Convenience function to load a configuration.

    Args:
        config_path: Path to YAML config file
        **overrides: Override specific values

    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader()
    return loader.load(config_path, **overrides)


# Example configurations
EXAMPLE_CONFIGS = {
    'clip': """
# CLIP Configuration
model:
  name: CLIP
  architecture: multimodal

  vision:
    model_name: ViT-B/16
    image_size: 224
    patch_size: 16
    width: 768
    layers: 12
    heads: 12

  text:
    context_length: 77
    vocab_size: 49408
    width: 512
    layers: 12
    heads: 8

  embed_dim: 512

training:
  batch_size: 256
  learning_rate: 5.0e-4
  weight_decay: 0.2
  warmup_steps: 10000
  max_steps: 1000000

  loss:
    temperature: 0.07

device: cuda
""",

    'yolo': """
# YOLOv8 Configuration
model:
  name: YOLOv8
  architecture: computer_vision

  model_size: m  # n, s, m, l, x
  image_size: 640
  num_classes: 80

  detection:
    conf_threshold: 0.25
    iou_threshold: 0.7
    max_det: 300

training:
  batch_size: 16
  epochs: 300
  optimizer: SGD
  learning_rate: 0.01
  momentum: 0.937
  weight_decay: 0.0005

  augmentation:
    mosaic: 1.0
    mixup: 0.1
    hsv_h: 0.015
    hsv_s: 0.7
    hsv_v: 0.4

device: cuda
""",

    'tft': """
# Temporal Fusion Transformer Configuration
model:
  name: TFT
  architecture: time_series

  static_input_size: 4
  temporal_observed_size: 3
  temporal_known_size: 2
  target_size: 1

  encoder_length: 24
  decoder_length: 12

  hidden_size: 160
  num_heads: 4
  num_lstm_layers: 1
  dropout: 0.1

  quantiles: [0.1, 0.5, 0.9]

training:
  batch_size: 64
  epochs: 100
  learning_rate: 0.001
  optimizer: Adam

  loss:
    type: quantile

device: cuda
""",

    'orchestrator': """
# Intelligent Orchestrator Configuration
orchestrator:
  enabled: true
  mode: auto  # auto, manual, hybrid

  selection:
    strategy: performance  # performance, cost, latency
    use_history: true
    history_weight: 0.3

  fusion:
    default_strategy: cross_attention  # cross_attention, parallel, sequential, ensemble
    enable_dynamic: true

  constraints:
    max_latency_ms: 1000
    max_cost_usd: 0.1
    min_accuracy: 0.9

architectures:
  # Vision
  - name: CLIP
    enabled: true
    config: configs/clip.yaml

  - name: SAM
    enabled: true
    config: configs/sam.yaml

  - name: YOLOv8
    enabled: true
    config: configs/yolo.yaml

  # Audio
  - name: Whisper
    enabled: true
    config: configs/whisper.yaml

  # Time Series
  - name: TFT
    enabled: true
    config: configs/tft.yaml

  - name: PatchTST
    enabled: true
    config: configs/patchtst.yaml

device: cuda
""",

    'training_full': """
# Complete Training Configuration
experiment:
  name: multimodal_training
  project: Brain
  tags: [multimodal, sota, research]

model:
  name: LLaVA
  config: configs/llava.yaml

  checkpoint:
    load_from: null
    save_dir: checkpoints/
    save_every: 1000
    keep_last: 5

data:
  train:
    path: data/train
    batch_size: 32
    shuffle: true
    num_workers: 4

  val:
    path: data/val
    batch_size: 64
    shuffle: false
    num_workers: 4

training:
  epochs: 100
  max_steps: null  # null = use epochs

  optimizer:
    name: AdamW
    learning_rate: 5.0e-5
    weight_decay: 0.01
    betas: [0.9, 0.999]

  scheduler:
    name: cosine
    warmup_steps: 1000
    min_lr: 1.0e-6

  gradient:
    clip_norm: 1.0
    accumulation_steps: 4

  mixed_precision:
    enabled: true
    dtype: fp16

validation:
  every_n_steps: 1000
  metrics:
    - accuracy
    - loss
    - bleu

logging:
  every_n_steps: 100
  tensorboard: true
  wandb: false

callbacks:
  - early_stopping:
      patience: 10
      metric: val_loss
      mode: min

  - model_checkpoint:
      monitor: val_accuracy
      mode: max
      save_top_k: 3

device: cuda
distributed:
  enabled: false
  backend: nccl
  world_size: 1
"""
}


if __name__ == "__main__":
    print("="*80)
    print("Configuration System - YAML-based Config for All Architectures")
    print("="*80)

    loader = ConfigLoader()

    # Test loading with example
    print("\nExample: CLIP Configuration")
    print("-"*80)
    clip_yaml = EXAMPLE_CONFIGS['clip']
    print(clip_yaml)

    print("\n" + "="*80)
    print("Example: YOLOv8 Configuration")
    print("-"*80)
    yolo_yaml = EXAMPLE_CONFIGS['yolo']
    print(yolo_yaml)

    print("\n" + "="*80)
    print("Example: Orchestrator Configuration")
    print("-"*80)
    orch_yaml = EXAMPLE_CONFIGS['orchestrator']
    print(orch_yaml)

    print("\n" + "="*80)
