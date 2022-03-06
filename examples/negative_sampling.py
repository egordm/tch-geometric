import torch
import torch_geometric as pyg
from torch_geometric.data import Data

from tch_geometric.transforms import NegativeSamplerTransform

device = 'cpu'
samples_per_node = 4
num_neighbors = [4, 3]

dataset = pyg.datasets.FakeDataset()
data: Data = dataset[0]

inputs = torch.arange(data.num_nodes, dtype=torch.long).to(device)

transform = NegativeSamplerTransform(data, 5, 5, inbound=False)
batch = transform(inputs)

u = 0
