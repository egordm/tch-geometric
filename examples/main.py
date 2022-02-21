import torch_geometric as pyg
import tch_geometric as thg
import torch
from torch_geometric.loader.utils import to_csc, to_hetero_csc

dataset = pyg.datasets.FakeHeteroDataset()
data = dataset[0]

colptr, row, perm = to_hetero_csc(data, device='cpu')

u = 0

print(
    thg.sum_as_string(
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    )
)