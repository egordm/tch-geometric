import torch
import torch_geometric as pyg
from torch_geometric.data import Data, HeteroData

from tch_geometric.transforms import NeighborSamplerTransform
from tch_geometric.transforms.hgt_sampling import HGTSamplerTransform

device = 'cpu'
samples_per_node = 4
num_samples = [4, 3]

dataset = pyg.datasets.FakeHeteroDataset()
data: HeteroData = dataset[0]
for store in data.edge_stores:
    store.timestamps = torch.rand(store.num_edges, dtype=torch.float64)

inputs = torch.arange(10, dtype=torch.long).to(device)
inputs_timestamps = torch.rand(10, dtype=torch.float64).to(device)

inputs = {'v0': inputs}
inputs_timestamps = {'v0': inputs_timestamps}

transform = HGTSamplerTransform(data, num_samples=num_samples)
batch1 = transform(inputs)

print('Sampled HGT')


transform = HGTSamplerTransform(data, num_samples=num_samples, temporal=True)
batch2 = transform(inputs, inputs_timestamps, (0.0, 0.5))

print('Sampled Temporal HGT')
