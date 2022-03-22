from typing import Union, List, Dict, Optional, Tuple

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader.utils import filter_hetero_data
from torch_geometric.typing import NodeType

import tch_geometric.tch_geometric as native
from tch_geometric.data import to_hetero_csc, to_hetero_sparse_attr
from tch_geometric.types import MixedData, validate_mixeddata, HeteroTensor, Timerange

NAN_TIMESTAMP = -1
NAN_TIMEDELTA = -99999


class BudgetSamplerTransform:
    def __init__(
            self,
            data: Union[Data, HeteroData],
            num_neighbors: Union[List[int], Dict[NodeType, List[int]]],
            window: Optional[Tuple[int, int]] = None,
            forward: bool = False,
            relative: bool = True,
    ) -> None:
        super().__init__()
        assert isinstance(data, HeteroData)

        self.data = data
        self.window = window
        self.forward = forward
        self.relative = relative
        self.temporal = window is not None

        self.col_ptrs_dict, self.row_indices_dict, self.perm_dict, self.size_dict = to_hetero_csc(data)
        self.node_types, self.edge_types = data.metadata()

        if self.temporal:
            assert 'timestamp' in data.keys
            self.row_timestamps_dict = to_hetero_sparse_attr(data, 'timestamp', self.perm_dict)

        if isinstance(num_neighbors, (list, tuple)):
            num_neighbors = {key: num_neighbors for key in self.node_types}
        assert isinstance(num_neighbors, dict)
        self.num_neighbors = num_neighbors

        self.num_hops = max([len(v) for v in self.num_neighbors.values()])

    def __call__(
            self,
            inputs_dict: HeteroTensor,
            inputs_timestamps_dict: Optional[HeteroTensor] = None,
            temporal: bool = True,
    ) -> Union[Data, HeteroData]:
        validate_mixeddata(inputs_dict, hetero=True, dtype=torch.int64)
        temporal = self.temporal

        # Sample the data
        sample_fn = native.budget_sampling
        nodes, nodes_timestamps, rows, cols, edges = sample_fn(
            self.node_types,
            self.edge_types,
            self.col_ptrs_dict,
            self.row_indices_dict,
            self.row_timestamps_dict if temporal else None,
            inputs_dict,
            inputs_timestamps_dict,
            self.num_neighbors,
            self.num_hops,
            self.window if temporal else None,
            self.forward,
            self.relative,
        )
        batch_size = {key: value.numel() for key, value in inputs_dict.items()}

        # Transform data to HeteroData
        data = filter_hetero_data(
            self.data, nodes, rows, cols, edges, self.perm_dict
        )
        for node_type, batch_size in batch_size.items():
            data[node_type].batch_size = batch_size

        if temporal:
            for store in data.node_stores:
                node_type = store._key
                if node_type in nodes_timestamps:
                    store.timestamp = nodes_timestamps[node_type]

        return data
