from dataclasses import dataclass
from typing import Tuple

import torch

from .tch_geometric import *


@dataclass
class UniformSampler:
    with_replacement: bool = False


@dataclass
class WeightedSampler:
    weights: torch.Tensor


TEMPORAL_SAMPLE_STATIC: int = 0
TEMPORAL_SAMPLE_RELATIVE: int = 1
TEMPORAL_SAMPLE_DYNAMIC: int = 2


@dataclass
class TemporalFilter:
    window: Tuple[int, int]
    timestamps: torch.Tensor
    initial_state: torch.Tensor
    forward: bool = False
    mode: int = TEMPORAL_SAMPLE_STATIC
