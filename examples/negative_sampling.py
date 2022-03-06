from time import perf_counter

import torch_geometric as pyg
import tch_geometric as thg
from tch_geometric.data import to_csr

device = 'cpu'
samples_per_node = 4
num_neighbors = [4, 3]

dataset = pyg.datasets.FakeDataset()
data = dataset[0]

row_ptrs, col_indices, perm = to_csr(data)

inputs = data.edge_index[0, :]
start = perf_counter()
samples, rows, cols, sample_count = thg.native.negative_sample_neighbors_homogenous(row_ptrs, col_indices, data.size(), inputs, 5, 5)
end = perf_counter()
print(f'Neg Sample Neighbors: {end - start}')

u = 0