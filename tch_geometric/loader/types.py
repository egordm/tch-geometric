from dataclasses import dataclass
from typing import Union, List, Dict, Tuple

import torch
from torch_geometric.typing import EdgeType

from tch_geometric.types import MixedData, validate_mixeddata

NumNeighbors = Union[List[int], Dict[EdgeType, List[int]]]


@dataclass
class EdgeSampler:
    def validate(self, hetero: bool = False) -> None:
        raise NotImplementedError


@dataclass
class UniformEdgeSampler(EdgeSampler):
    with_replacement: bool = False

    def validate(self, hetero: bool = False) -> None:
        pass


@dataclass
class WeightedEdgeSampler(EdgeSampler):
    weights: MixedData

    def validate(self, hetero: bool = False) -> None:
        validate_mixeddata(self.weights, hetero=hetero, dtype=torch.float64)


TEMPORAL_SAMPLE_STATIC: int = 0
TEMPORAL_SAMPLE_RELATIVE: int = 1
TEMPORAL_SAMPLE_DYNAMIC: int = 2


@dataclass
class EdgeFilter:
    def validate(self, hetero: bool = False) -> None:
        raise NotImplementedError


@dataclass
class TemporalEdgeFilter:
    window: Tuple[int, int]
    timestamps: MixedData
    forward: bool = False
    mode: int = TEMPORAL_SAMPLE_STATIC

    def validate(self, hetero: bool = False) -> None:
        validate_mixeddata(self.timestamps, hetero=hetero, dtype=torch.int64)