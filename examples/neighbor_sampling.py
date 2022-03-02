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

# Standard sampling
start = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
samples, rows, cols, edge_index, layer_offsets = thg.algo.neighbor_sampling_homogenous(
    col_ptrs, row_indices, start.repeat(samples_per_node), num_neighbors
)
batch = filter_data(data, samples, rows, cols, edge_index, perm)

layer = SAGEConv((-1, -1), 32)
output = layer(x=batch.x, edge_index=batch.edge_index)


# Weighted sampling
start = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
weights = torch.rand(row_indices.shape, dtype=torch.double)
samples, rows, cols, edge_index, layer_offsets = thg.algo.neighbor_sampling_homogenous(
    col_ptrs, row_indices, start.repeat(samples_per_node), num_neighbors, sampler=thg.WeightedSampler(weights)
)
batch = filter_data(data, samples, rows, cols, edge_index, perm)

layer = SAGEConv((-1, -1), 32)
output = layer(x=batch.x, edge_index=batch.edge_index)

# Temporal Filtering
start = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
initial_timestamps = torch.randint(size=start.shape, low=0, high=5, dtype=torch.long)
timestamps = torch.randint(size=row_indices.shape, low=0, high=5, dtype=torch.long)
samples, rows, cols, edge_index, layer_offsets = thg.algo.neighbor_sampling_homogenous(
    col_ptrs, row_indices, start.repeat(samples_per_node), num_neighbors,
    filter=thg.TemporalFilter((0, 3), timestamps, initial_timestamps)
)
batch = filter_data(data, samples, rows, cols, edge_index, perm)

layer = SAGEConv((-1, -1), 32)
output = layer(x=batch.x, edge_index=batch.edge_index)