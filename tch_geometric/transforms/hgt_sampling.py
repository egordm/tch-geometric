from typing import Union, List, Dict, Optional

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader.utils import filter_hetero_data
from torch_geometric.typing import NodeType

import tch_geometric.tch_geometric as native
from tch_geometric.data import to_hetero_csc, to_hetero_sparse_attr
from tch_geometric.types import MixedData, validate_mixeddata, HeteroTensor, Timerange

NAN_TIMESTAMP = -1
NAN_TIMEDELTA = -99999

class HGTSamplerTransform:
    def __init__(
            self,
            data: Union[Data, HeteroData],
            num_samples: Union[List[int], Dict[NodeType, List[int]]],
            temporal: bool = False,
    ) -> None:
        super().__init__()
        assert isinstance(data, HeteroData)

        self.data = data
        self.num_samples = num_samples
        self.temporal = temporal

        self.col_ptrs_dict, self.row_indices_dict, self.perm_dict, self.size_dict = to_hetero_csc(data)
        self.node_types, self.edge_types = data.metadata()

        if temporal:
            assert 'timestamp' in data.keys
            self.row_timestamps_dict = to_hetero_sparse_attr(data, 'timestamp', self.perm_dict)

        if isinstance(num_samples, (list, tuple)):
            num_samples = {key: num_samples for key in self.node_types}
        assert isinstance(num_samples, dict)
        self.num_samples = num_samples

        self.num_hops = max([len(v) for v in self.num_samples.values()])

    def __call__(
            self,
            inputs_dict: HeteroTensor,
            inputs_timestamps_dict: Optional[HeteroTensor] = None,
            timerange: Optional[Timerange] = None,
    ) -> Union[Data, HeteroData]:
        validate_mixeddata(inputs_dict, hetero=True, dtype=torch.int64)

        # Correct amount of samples by the batch size
        num_inputs = sum([len(v) for v in inputs_dict.values() ])
        num_samples = {
            k: [n * num_inputs for n in v]
            for k, v in self.num_samples.items()
        }

        # Sample the data
        sample_fn = native.hgt_sampling
        nodes, nodes_timestamps, rows, cols, edges = sample_fn(
            self.node_types,
            self.edge_types,
            self.col_ptrs_dict,
            self.row_indices_dict,
            self.row_timestamps_dict if self.temporal else None,
            inputs_dict,
            inputs_timestamps_dict if self.temporal else None,
            num_samples,
            self.num_hops,
            timerange if self.temporal else None,
        )
        batch_size = {key: value.numel() for key, value in inputs_dict.items()}

        # Transform data to HeteroData
        data = filter_hetero_data(
            self.data, nodes, rows, cols, edges, self.perm_dict
        )
        for node_type, batch_size in batch_size.items():
            data[node_type].batch_size = batch_size

        if self.temporal:
            for store in data.node_stores:
                node_type = store._key
                if node_type in nodes_timestamps:
                    store.timestamp = nodes_timestamps[node_type]

            for store in data.edge_stores:
                (src, _, dst) = store._key
                timestamp_src = data[src].timestamp[store.edge_index[0, :]]
                timestamp_dst = data[dst].timestamp[store.edge_index[1, :]]
                store.timedelta = timestamp_dst - timestamp_src
                store.timedelta[torch.logical_or(timestamp_src == NAN_TIMESTAMP, timestamp_dst == NAN_TIMESTAMP)] = NAN_TIMEDELTA

        return data
