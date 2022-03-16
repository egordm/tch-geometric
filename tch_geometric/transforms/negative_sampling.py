from typing import Union, Tuple

import torch
from torch_geometric.data import Data, HeteroData

import tch_geometric.tch_geometric as native
from tch_geometric.data import to_csc, to_csr, to_hetero_csc, to_hetero_csr
from tch_geometric.data.subgraph import create_subgraph
from tch_geometric.types import MixedData, validate_mixeddata


class NegativeSamplerTransform:
    def __init__(
            self,
            data: Union[Data, HeteroData],
            num_neg: int,
            try_count: int,
            inbound: bool = True,
    ) -> None:
        super().__init__()
        self.data = data
        self.num_neg = num_neg
        self.try_count = try_count
        self.inbound = inbound

        # Convert the graph data into a suitable format for sampling.
        if isinstance(data, Data):
            convert_fn = to_csc if inbound else to_csr
            self.ptrs, self.indices, self.perm, self.size = convert_fn(data)

        elif isinstance(data, HeteroData):
            convert_fn = to_hetero_csc if inbound else to_hetero_csr
            self.ptrs_dict, self.indices_dict, self.perm_dict, self.size_dict = convert_fn(data)
            self.node_types, self.edge_types = data.metadata()

        else:
            raise TypeError(f'Invalid graph type: {type(data)}')

    def __call__(self, inputs: MixedData) -> HeteroData:
        if isinstance(self.data, Data):
            validate_mixeddata(inputs, hetero=False, dtype=torch.int64)

            # Sample negative edges.
            sample_fn = native.negative_sample_neighbors_homogenous
            node, row, col, sample_count = sample_fn(
                self.ptrs,
                self.indices,
                self.size,
                inputs,
                self.num_neg,
                self.try_count,
            )

            # Build the subgraph
            subgraph = Data()
            subgraph.x = node
            subgraph.edge_index = torch.stack([row, col], dim=0)
            subgraph.sample_count = sample_count

            return subgraph

        elif isinstance(self.data, HeteroData):
            validate_mixeddata(inputs, hetero=True, dtype=torch.int64)

            # Sample negative edges.
            sample_fn = native.negative_sample_neighbors_heterogenous
            nodes, rows, cols, sample_counts = sample_fn(
                self.node_types,
                self.edge_types,
                self.ptrs_dict,
                self.indices_dict,
                self.size_dict,
                inputs,
                self.num_neg,
                self.try_count,
                self.inbound,
            )

            subgraph = create_subgraph(nodes, rows, cols, node_attrs=dict(sample_count=sample_counts))

            return subgraph
