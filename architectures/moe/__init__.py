"""
Mixture of Experts (MoE) - Sparse Scaling for LLMs

Enables 10x+ parameters with constant compute.
"""

from .mixture_of_experts import (
    MoEConfig,
    SwitchMoE,
    ExpertChoiceMoE,
    SoftMoE,
    create_moe
)

__all__ = [
    'MoEConfig',
    'SwitchMoE',
    'ExpertChoiceMoE',
    'SoftMoE',
    'create_moe'
]
