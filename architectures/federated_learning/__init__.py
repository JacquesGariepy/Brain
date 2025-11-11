"""
Federated Learning - Privacy-preserving distributed learning

Enables training across multiple clients without sharing raw data.

Supported algorithms:
- FedAvg: Federated Averaging (McMahan et al., 2017)
- FedProx: Federated Optimization with proximal term
- FedNova: Normalized Averaging
- FedAdam: Federated Adam optimizer
- FedYogi: Federated Yogi optimizer

Works with ANY PyTorch model in plug-and-play mode.
"""

from .federated_client import FederatedClient
from .federated_server import FederatedServer
from .federated_trainer import FederatedTrainer, FedAlgorithm

__all__ = [
    'FederatedClient',
    'FederatedServer',
    'FederatedTrainer',
    'FedAlgorithm'
]
