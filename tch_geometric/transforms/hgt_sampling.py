from typing import Union, List, Dict

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader.utils import filter_hetero_data
from torch_geometric.typing import NodeType

import tch_geometric.tch_geometric as native
from tch_geometric.data import to_hetero_csc
from tch_geometric.types import MixedData, validate_mixeddata


class HGTSamplerTransform:
    def __init__(
            self,
            data: Union[Data, HeteroData],
            num_samples: Union[List[int], Dict[NodeType, List[int]]],
    ) -> None:
        super().__init__()
        assert isinstance(data, HeteroData)

        self.data = data
        self.num_samples = num_samples

        self.col_ptrs_dict, self.row_indices_dict, self.perm_dict, self.size_dict = to_hetero_csc(data)
        self.node_types, self.edge_types = data.metadata()

        if isinstance(num_samples, (list, tuple)):
            num_samples = {key: num_samples for key in self.node_types}
        assert isinstance(num_samples, dict)
        self.num_samples = num_samples

        self.num_hops = max([len(v) for v in self.num_samples.values()])

    def __call__(self, inputs_dict: MixedData) -> Union[Data, HeteroData]:
        validate_mixeddata(inputs_dict, hetero=True, dtype=torch.int64)

        # Correct amount of samples by the batch size
        num_inputs = sum([len(v) for v in inputs_dict.values() ])
        num_samples = {
            k: [n * num_inputs for n in v]
            for k, v in self.num_samples.items()
        }

        # Sample the data
        sample_fn = native.hgt_sampling
        nodes, rows, cols, edges = sample_fn(
            self.node_types,
            self.edge_types,
            self.col_ptrs_dict,
            self.row_indices_dict,
            inputs_dict,
            num_samples,
            self.num_hops,
        )
        batch_size = {key: value.numel() for key, value in inputs_dict.items()}

        # Transform data to HeteroData
        data = filter_hetero_data(
            self.data, nodes, rows, cols, edges, self.perm_dict
        )
        for node_type, batch_size in batch_size.items():
            data[node_type].batch_size = batch_size

        return data
