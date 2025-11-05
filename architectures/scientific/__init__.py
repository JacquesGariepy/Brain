"""
Scientific AI - Physics-Informed Neural Networks and Neural ODEs

Applies deep learning to scientific problems with physical constraints.
"""

from .scientific_ai import (
    # PINNs
    PINNConfig,
    PINN,

    # Neural ODEs
    NeuralODEConfig,
    ODEFunc,
    NeuralODE,
    AdjointNeuralODE
)

__all__ = [
    # PINNs
    'PINNConfig',
    'PINN',

    # Neural ODEs
    'NeuralODEConfig',
    'ODEFunc',
    'NeuralODE',
    'AdjointNeuralODE'
]
