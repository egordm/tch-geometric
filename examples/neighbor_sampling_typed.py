import torch
import torch_geometric as pyg
from torch_geometric.data import Data, HeteroData

from tch_geometric.transforms import NeighborSamplerTransform

device = 'cpu'
samples_per_node = 4
num_neighbors = [4, 3]

dataset = pyg.datasets.FakeDataset()
data: Data = dataset[0]

inputs = torch.arange(10, dtype=torch.long).to(device)

transform = NeighborSamplerTransform(data, num_neighbors=[4, 3])
batch = transform(inputs)

print('Sampled Homogenous')

dataset = pyg.datasets.FakeHeteroDataset()
data: HeteroData = dataset[0]

inputs = {'v0': inputs}

transform = NeighborSamplerTransform(data, num_neighbors=[4, 3])
batch = transform(inputs)

print('Sampled Heterogenous')

