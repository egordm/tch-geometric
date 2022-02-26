import torch
import torch_geometric as pyg
from torch_geometric.loader.utils import filter_data
from torch_geometric.nn import SAGEConv
import tch_geometric as thg

device = 'cpu'
samples_per_node = 4
num_neighbors = [4, 3]

dataset = pyg.datasets.FakeDataset()
data = dataset[0]

col_ptrs, row_indices, perm = thg.data.to_csc(data.edge_index, len(data.x))

start = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
samples, rows, cols, edge_index, layer_offsets = thg.algo.neighbor_sampling_homogenous(
    col_ptrs, row_indices, start.repeat(samples_per_node), num_neighbors, False
)
batch = filter_data(data, samples, rows, cols, edge_index, perm)

layer = SAGEConv((-1, -1), 32)
output = layer(x=batch.x, edge_index=batch.edge_index)
