from time import perf_counter

import numpy as np
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
import tch_geometric as thg
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader.utils import to_csc, to_hetero_csc, filter_data

# dataset = pyg.datasets.FakeHeteroDataset()
dataset = pyg.datasets.FakeDataset()
data = dataset[0]
replace = False
directed = False

# colptr, row, perm = to_hetero_csc(data, device='cpu')
colptr, row, perm = to_csc(data, device='cpu')
index = torch.tensor([
    0, 1, 2, 3, 4, 5, 6, 7,
    # 0, 1, 2, 3, 4, 5, 6, 7,
    # 0, 1, 2, 3, 4, 5, 6, 7,
    # 0, 1, 2, 3, 4, 5, 6, 7,
    # 0, 1, 2, 3, 4, 5, 6, 7,
    # 0, 1, 2, 3, 4, 5, 6, 7,
    # 0, 1, 2, 3, 4, 5, 6, 7,
    # 0, 1, 2, 3, 4, 5, 6, 7,
], dtype=torch.long)

start = perf_counter()
for i in range(200):
    sp_node, sp_row, sp_col, sp_edge = torch.ops.torch_sparse.neighbor_sample(
        colptr,
        row,
        index,
        [4, 4],
        replace,
        directed,
    )
end = perf_counter()
print(f'Torch Sparse: {end - start}')

batch_data_og = filter_data(data, sp_node, sp_row, sp_col, sp_edge, perm)

start = perf_counter()
for i in range(200):
    sp_node, sp_row, sp_col, sp_edge = thg.sample_own_custom(
        colptr,
        row,
        index,
        [4, 4],
        replace,
        directed,
    )
end = perf_counter()
print(f'TCH Sparse Own: {end - start}')
# print(sp_node.shape, sp_row.shape, sp_col.shape, sp_edge.shape)

start = perf_counter()
for i in range(200):
    sp_node, sp_row, sp_col, sp_edge = thg.sample(
        colptr,
        row,
        index,
        [4, 4],
        replace,
        directed,
    )
end = perf_counter()
print(f'TCH Sparse Cloned: {end - start}')

batch_data_my = filter_data(data, sp_node, sp_row, sp_col, sp_edge, perm)

u = 0

loader = NeighborLoader(
    data=data,
    batch_size=32,
    input_nodes='v0',
    num_neighbors=[4, 4],
    directed=True,
    replace=False,
    num_workers=0,
)

batch: pyg.data.HeteroData = next(iter(loader))

layer = pyg_nn.SAGEConv((-1, -1), 32)

x = batch.get_node_store('v0').x
edge_index = batch.get_edge_store('v0', 'e0', 'v0').edge_index

output = layer(x=x, edge_index=edge_index)

u = 0

print(
    thg.sum_as_string(
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    )
)
