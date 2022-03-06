from typing import Union, Optional, Callable

import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, BatchSampler
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader.base import BaseDataLoader
from torch_geometric.loader.utils import filter_data, filter_hetero_data

import tch_geometric.tch_geometric as native
from tch_geometric.data import to_hetero_csc, to_csc, edge_type_to_str
from tch_geometric.transforms import EdgeFilter, EdgeSampler, NumNeighbors
from tch_geometric.types import MixedData, validate_mixeddata


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

            sample_fn = native.neighbor_sampling_homogenous
            nodes, rows, cols, edges, layer_offsets = sample_fn(
                self.col_ptrs,
                self.row_indices,
                inputs,
                self.num_neighbors,
                self.edge_sampler,
                self.edge_filter,
            )
            return nodes, rows, cols, edges, layer_offsets, inputs.numel()

        elif issubclass(self.data_cls, HeteroData):
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
            return nodes, rows, cols, edges, layer_offsets, batch_size


class SamplerDataset(Dataset):
    def __init__(
            self,
            inputs: MixedData,
            input_states: Optional[MixedData] = None
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.inputs_states = input_states

    def __getitem__(self, index):
        inputs = self.inputs[index]

        if self.inputs_states:
            states = self.inputs_states[index]
        else:
            states = None

        return inputs, states

    def __len__(self) -> int:
        return len(self.inputs)


class HeteroSamplerDataset(Dataset):
    def __init__(
            self,
            inputs: MixedData,
            input_states: Optional[MixedData] = None
    ) -> None:
        super().__init__()
        self.node_types = list(inputs.keys())

        node_type_map = {key: i for i, key in enumerate(self.node_types)}
        inputs_items = []
        types_items = []
        states_items = []
        for key, value in inputs.items():
            inputs_items.append(value)
            types_items.append(torch.full([value.shape[0]], node_type_map[key], dtype=torch.int64))
            if input_states:
                states_items.append(input_states[key])

        self.inputs = torch.cat(inputs_items, dim=0)
        self.types = torch.cat(types_items, dim=0)
        self.inputs_states = torch.cat(states_items, dim=0) if states_items else None

    def __getitem__(self, index):
        index = torch.tensor(index, dtype=torch.int64)
        types = self.types[index]
        inputs = {}
        states = {} if self.inputs_states else None

        used_types = torch.unique(types)
        for t in used_types:
            type_name = self.node_types[t]
            mask = index[types == t]
            inputs[type_name] = self.inputs[mask]
            if self.inputs_states:
                states[type_name] = self.inputs_states[mask]

        return inputs, states

    def __len__(self) -> int:
        return len(self.inputs)


class NeighborLoader(BaseDataLoader):
    def __init__(
            self,
            data: Union[Data, HeteroData],
            num_neighbors: NumNeighbors,
            dataset: Dataset,
            edge_sampler: Optional[EdgeSampler] = None,
            edge_filter: Optional[EdgeFilter] = None,
            inputs_states: Optional[MixedData] = None,
            neighbor_sampler: Optional[NeighborSampler] = None,
            transform: Callable = None,
            shuffle: bool = False,
            generator=None,
            batch_size: int = 1,
            drop_last: bool = False,
            **kwargs,
    ):
        self.data = data
        self.num_neighbors = num_neighbors
        self.edge_sampler = edge_sampler
        self.edge_filter = edge_filter
        self.inputs_states = inputs_states
        self.neighbor_sampler = neighbor_sampler
        self.transform = transform
        self._homogenous = isinstance(self.data, Data)
        self.batch_size = batch_size

        if neighbor_sampler is None:
            self.neighbor_sampler = NeighborSampler(
                data, num_neighbors, edge_sampler, edge_filter
            )

        # Default sampler to set autocollate to False
        if shuffle:
            sampler = RandomSampler(dataset, generator=generator)
        else:
            sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        super().__init__(
            dataset,
            collate_fn=self.sample,
            sampler=batch_sampler,
            batch_size=None,
            shuffle=shuffle,
            drop_last=drop_last,
            **kwargs,
        )

    def sample(self, inputs):
        inputs, states = inputs
        return self.neighbor_sampler(inputs, states)

    def transform_fn(self, out):
        if self._homogenous:
            node, row, col, edge, layer_offsets, batch_size = out
            data = filter_data(self.data, node, row, col, edge, self.neighbor_sampler.perm)
            data.batch_size = batch_size
        else:
            node_dict, row_dict, col_dict, edge_dict, layer_offsets, batch_size = out
            data = filter_hetero_data(
                self.data, node_dict, row_dict, col_dict, edge_dict, self.neighbor_sampler.perm_dict
            )
            for node_type, batch_size in batch_size.items():
                data[node_type].batch_size = batch_size

        return data if self.transform is None else self.transform(data)
