import torch
import torch_geometric as pyg
from torch_geometric.nn import Node2Vec
import tch_geometric as thg

device = 'cpu'
walk_length = 10
walks_per_node = 4
p = 1.0
q = 1.5

dataset = pyg.datasets.FakeDataset()
data = dataset[0]

row_ptrs, col_indices, perm = thg.data.to_csr(data.edge_index, len(data.x))

model = Node2Vec(
    data.edge_index,
    embedding_dim=12,
    walk_length=walk_length,
    context_size=walk_length,
    walks_per_node=walks_per_node,
    num_negative_samples=3,
    p=p,
    q=q,
    sparse=True
).to(device)

start = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
pos_rw = thg.algo.random_walk(row_ptrs, col_indices, start.repeat(walks_per_node), walk_length - 1, p, q)
neg_rw = model.neg_sample(start.repeat(walks_per_node))

example = model.pos_sample(start)

assert pos_rw.shape == example.shape

loss = model.loss(pos_rw, neg_rw)
print(f'Initial loss {loss}')
