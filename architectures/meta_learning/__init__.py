"""Meta-learning architectures for few-shot learning"""

from .maml import MAML, MAMLConfig
from .reptile import Reptile, ReptileConfig
from .protonet import PrototypicalNetworks, ProtoNetConfig

__all__ = [
    'MAML', 'MAMLConfig',
    'Reptile', 'ReptileConfig',
    'PrototypicalNetworks', 'ProtoNetConfig'
]
