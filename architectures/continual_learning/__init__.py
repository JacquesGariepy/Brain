"""
Continual Learning - Learn continuously without forgetting

Provides methods to prevent catastrophic forgetting:
- EWC (Elastic Weight Consolidation)
- iCaRL (Incremental Classifier and Representation Learning)
- LwF (Learning without Forgetting)
- GEM (Gradient Episodic Memory)
- A-GEM (Averaged GEM)

Works with ANY PyTorch model in plug-and-play mode.
"""

from .ewc import EWC, EWCLoss
from .icarl import iCaRL
from .lwf import LwF
from .gem import GEM
from .continual_learner import ContinualLearner

__all__ = [
    'EWC', 'EWCLoss',
    'iCaRL',
    'LwF',
    'GEM',
    'ContinualLearner'
]
