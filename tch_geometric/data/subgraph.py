import torch
from torch_geometric.data import HeteroData

from tch_geometric.utils import zip_dict


def build_subgraph(nodes_dict, rows_dict, cols_dict, edge_attrs=None, node_attrs=None):
    if edge_attrs is None:
        edge_attrs = {}
    if node_attrs is None:
        node_attrs = {}

    # Build the subgraph
    subgraph = HeteroData()
    for node_type, node in nodes_dict.items():
        subgraph[node_type].x = node
        for node_attr, node_attr_vals in node_attrs.items():
            setattr(subgraph[node_type], node_attr, node_attr_vals[node_type])

    for rel_type, (row, col) in zip_dict(rows_dict, cols_dict):
        edge_type = tuple(rel_type.split('__')) if isinstance(rel_type, str) else rel_type
        subgraph[edge_type].edge_index = torch.stack([row, col], dim=0)

        for edge_attr, edge_attr_vals in edge_attrs.items():
            setattr(subgraph[edge_type], edge_attr, edge_attr_vals[edge_type])

    return subgraph