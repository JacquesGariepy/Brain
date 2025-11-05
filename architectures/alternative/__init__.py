"""
Alternative Architectures - Efficient Alternatives to Transformers

Sub-quadratic or linear-time sequence modeling.
"""

from .mamba import (
    MambaConfig,
    Mamba,
    MambaBlock
)

from .rwkv import (
    RWKVConfig,
    RWKV,
    RWKVBlock
)

from .retnet import (
    RetNetConfig,
    RetNet,
    RetentionBlock
)

from .hyena import (
    HyenaConfig,
    Hyena,
    HyenaBlock
)

__all__ = [
    # Mamba
    'MambaConfig',
    'Mamba',
    'MambaBlock',

    # RWKV
    'RWKVConfig',
    'RWKV',
    'RWKVBlock',

    # RetNet
    'RetNetConfig',
    'RetNet',
    'RetentionBlock',

    # Hyena
    'HyenaConfig',
    'Hyena',
    'HyenaBlock'
]
