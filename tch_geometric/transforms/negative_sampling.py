from typing import Union

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader.utils import filter_data, filter_hetero_data

from tch_geometric.data import to_csc, to_csr, to_hetero_csc, to_hetero_csr
from tch_geometric.types import MixedData, validate_mixeddata
import tch_geometric.tch_geometric as native


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

    def __call__(self, inputs: MixedData) -> Union[Data, HeteroData]:
        if isinstance(self.data, Data):
            validate_mixeddata(inputs, hetero=False, dtype=torch.int64)

            sample_fn = native.negative_sample_neighbors_homogenous
            node, row, col, sample_count = sample_fn(
                self.ptrs,
                self.indices,
                self.size,
                inputs,
                self.num_neg,
                self.try_count,
            )
            edge = torch.tensor([], dtype=torch.long)

            data = filter_data(self.data, node, row, col, edge, self.perm)
            data.batch_size = sample_count

            return data

        elif isinstance(self.data, HeteroData):
            validate_mixeddata(inputs, hetero=True, dtype=torch.int64)

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
            )

            edges = {k: torch.tensor([], dtype=torch.long) for k in rows.keys()}
            data = filter_hetero_data(self.data, nodes, rows, cols, edges, self.perm_dict)
            for node_type, batch_size in sample_counts.items():
                data[node_type].batch_size = batch_size

            return data
