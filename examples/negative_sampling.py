from time import perf_counter

import torch
import torch_geometric as pyg
from torch_geometric.data import Data

import tch_geometric as thg
from tch_geometric.data import to_csr
from tch_geometric.transforms import NegativeSamplerTransform

device = 'cpu'
samples_per_node = 4
num_neighbors = [4, 3]

dataset = pyg.datasets.FakeDataset()
data: Data = dataset[0]

row_ptrs, col_indices, perm, size = to_csr(data)

inputs = torch.arange(data.num_nodes, dtype=torch.long).to(device)
start = perf_counter()
samples, rows, cols, sample_count = thg.native.negative_sample_neighbors_homogenous(row_ptrs, col_indices, data.size(), inputs, 5, 5)
end = perf_counter()
print(f'Neg Sample Neighbors: {end - start}')

u = 0

transform = NegativeSamplerTransform(data, 5, 5, inbound=False)
neg_samples = transform(inputs)

u = 0