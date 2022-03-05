import torch
import torch_geometric as pyg

from tch_geometric.loader import NeighborSampler, NeighborLoader, SamplerDataset, HeteroSamplerDataset

dataset = pyg.datasets.FakeDataset()
data = dataset[0]
inputs = torch.tensor([3, 2, 1, 0])

sampler = NeighborSampler(data, [4, 3])
batch = sampler(inputs)

loader = NeighborLoader(data=data, num_neighbors=[4, 3], dataset=SamplerDataset(inputs), batch_size=4)
batch = next(iter(loader))


dataset = pyg.datasets.FakeHeteroDataset()
data = dataset[0]

sampler = NeighborSampler(data, [4, 3])
batch = sampler({'v0': inputs})

sampler_dataset = HeteroSamplerDataset({'v0': inputs, 'v1': inputs, 'v2': inputs})
loader = NeighborLoader(data=data, num_neighbors=[4, 3], dataset=sampler_dataset, batch_size=4)
batch = next(iter(loader))


print('done')