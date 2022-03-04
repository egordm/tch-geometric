from dataclasses import dataclass
from typing import Union, List, Dict, Tuple, Optional

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType, NodeType

from tch_geometric.loader.utils import to_hetero_csc, to_csc, edge_type_to_str, RelType

import tch_geometric.tch_geometric as lib

NumNeighbors = Union[List[int], Dict[EdgeType, List[int]]]

MixedData = Union[Tensor, Dict[str, Tensor]]


def validate_mixeddata(data: MixedData, hetero: bool = False, dtype=None):
    if hetero:
        assert isinstance(data, dict)
        for v in data.values():
            assert v.dtype == dtype
    else:
        assert data.dtype == dtype


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


class NeighborSampler:
    def __init__(
            self,
            data: Union[Data, HeteroData],
            num_neighbors: NumNeighbors,
            edge_sampler: Optional[EdgeSampler] = None,
            edge_filter: Optional[EdgeFilter] = None,
    ) -> None:
        super().__init__()
        self.data_cls = data.__class__
        self.num_neighbors = num_neighbors
        self.edge_sampler = edge_sampler
        self.edge_filter = edge_filter

        # Convert the graph data into a suitable format for sampling.
        if isinstance(data, Data):
            self.col_ptrs, self.row_indices, self.perm = to_csc(data)
            assert isinstance(num_neighbors, (list, tuple))

        elif isinstance(data, HeteroData):
            self.col_ptrs_dict, self.row_indices_dict, self.perm_dict = to_hetero_csc(data)
            self.node_types, self.edge_types = data.metadata()

            if isinstance(num_neighbors, (list, tuple)):
                num_neighbors = {key: num_neighbors for key in self.edge_types}
            assert isinstance(num_neighbors, dict)
            self.num_neighbors = {
                edge_type_to_str(key): value
                for key, value in num_neighbors.items()
            }

            self.num_hops = max([len(v) for v in self.num_neighbors.values()])
        else:
            raise TypeError(f'NeighborLoader found invalid type: {type(data)}')

    def __call__(self, inputs: MixedData, input_states: Optional[MixedData] = None):
        if issubclass(self.data_cls, Data):
            validate_mixeddata(inputs, hetero=False, dtype=torch.int64)

            sample_fn = lib.algo.neighbor_sampling_homogenous
            samples, rows, cols, edges, layer_offsets = sample_fn(
                self.col_ptrs,
                self.row_indices,
                inputs,
                self.num_neighbors,
                self.edge_sampler,
                self.edge_filter,
            )
            return samples, rows, cols, edges, layer_offsets, inputs.numel()

        elif issubclass(self.data_cls, HeteroData):
            validate_mixeddata(inputs, hetero=True, dtype=torch.int64)

            if input_states:
                validate_mixeddata(input_states, hetero=True, dtype=torch.int64)
            edge_filter = (self.edge_filter, input_states) if self.edge_filter else None

            sample_fn = lib.algo.neighbor_sampling_heterogenous
            samples, rows, cols, edges, layer_offsets = sample_fn(
                self.node_types,
                self.edge_types,
                self.col_ptrs_dict,
                self.row_indices_dict,
                inputs,
                self.num_neighbors,
                self.num_hops,
                self.edge_sampler,
                edge_filter,
            )
            batch_size = {key: value.numel() for key, value in inputs.items()}
            return samples, rows, cols, edges, layer_offsets, batch_size
