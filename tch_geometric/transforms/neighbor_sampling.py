from dataclasses import dataclass
from typing import Union, List, Dict, Optional, Tuple

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader.utils import filter_data, filter_hetero_data
from torch_geometric.typing import EdgeType

import tch_geometric.tch_geometric as native
from tch_geometric.data import to_csc, to_hetero_csc, edge_type_to_str
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


class NeighborSamplerTransform:
    def __init__(
            self,
            data: Union[Data, HeteroData],
            num_neighbors: NumNeighbors,
            edge_sampler: Optional[EdgeSampler] = None,
            edge_filter: Optional[EdgeFilter] = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.num_neighbors = num_neighbors
        self.edge_sampler = edge_sampler
        self.edge_filter = edge_filter

        # Convert the graph data into a suitable format for sampling.
        if isinstance(data, Data):
            self.col_ptrs, self.row_indices, self.perm, self.size = to_csc(data)
            assert isinstance(num_neighbors, (list, tuple))

        elif isinstance(data, HeteroData):
            self.col_ptrs_dict, self.row_indices_dict, self.perm_dict, self.size_dict = to_hetero_csc(data)
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

    def __call__(self, inputs: MixedData, input_states: Optional[MixedData] = None) -> Union[Data, HeteroData]:
        if isinstance(self.data, Data):
            validate_mixeddata(inputs, hetero=False, dtype=torch.int64)

            sample_fn = native.neighbor_sampling_homogenous
            node, row, col, edge, layer_offsets = sample_fn(
                self.col_ptrs,
                self.row_indices,
                inputs,
                self.num_neighbors,
                self.edge_sampler,
                self.edge_filter,
            )
            batch_size = inputs.numel()

            data = filter_data(self.data, node, row, col, edge, self.perm)
            data.batch_size = batch_size

            return data

        elif isinstance(self.data, HeteroData):
            validate_mixeddata(inputs, hetero=True, dtype=torch.int64)

            if input_states:
                validate_mixeddata(input_states, hetero=True, dtype=torch.int64)
            edge_filter = (self.edge_filter, input_states) if self.edge_filter else None

            sample_fn = native.neighbor_sampling_heterogenous
            nodes, rows, cols, edges, layer_offsets = sample_fn(
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

            data = filter_hetero_data(
                self.data, nodes, rows, cols, edges, self.perm_dict
            )
            for node_type, batch_size in batch_size.items():
                data[node_type].batch_size = batch_size

            return data
