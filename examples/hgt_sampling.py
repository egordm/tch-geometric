import torch
import torch_geometric as pyg
from torch_geometric.data import Data, HeteroData

from tch_geometric.transforms import NeighborSamplerTransform
from tch_geometric.transforms.hgt_sampling import HGTSamplerTransform

device = 'cpu'
samples_per_node = 4
num_samples = [40, 30]

dataset = pyg.datasets.FakeHeteroDataset()
data: HeteroData = dataset[0]
inputs = torch.arange(10, dtype=torch.long).to(device)

inputs = {'v0': inputs}

transform = HGTSamplerTransform(data, num_samples=num_samples)
batch = transform(inputs)

print('Sampled HGT')

