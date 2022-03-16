from collections import defaultdict
from typing import Dict, Any
from copy import deepcopy

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType

from tch_geometric.utils import zip_dict


def subgraph_from_edgelist(
        edge_index_dict: Dict[EdgeType, Tensor],
        node_attrs: Dict[str, Dict[NodeType, Any]] = None,
        edge_attrs: Dict[str, Dict[EdgeType, Any]] = None,
):
    if node_attrs is None:
        node_attrs = {}
    if edge_attrs is None:
        edge_attrs = {}

    nodes: Dict[NodeType, Any] = defaultdict(list)
    rows = dict()
    cols = dict()
    node_counts = defaultdict(lambda: 0)
    sample_count = defaultdict(lambda: 0)

    for edge_type, edge_index in edge_index_dict.items():
        (src, _, _) = edge_type
        start = node_counts[src]
        edge_count = edge_index.shape[1]
        nodes[src].append(edge_index[0, :])
        rows[edge_type] = torch.arange(start, start + edge_count, dtype=torch.long)

        node_counts[src] += edge_count

    sample_count = deepcopy(node_counts)

    for edge_type, edge_index in edge_index_dict.items():
        (_, _, dst) = edge_type
        start = node_counts[dst]
        edge_count = edge_index.shape[1]
        nodes[dst].append(edge_index[1, :])
        cols[edge_type] = torch.arange(start, start + edge_count, dtype=torch.long)

        node_counts[dst] += edge_count

    for node_type, vals in nodes.items():
        nodes[node_type] = torch.cat(vals, dim=0)

    subgraph = create_subgraph(
        nodes, rows, cols,
        node_attrs=dict(sample_count=sample_count, **node_attrs),
        edge_attrs=edge_attrs
    )

    return subgraph


def create_subgraph(nodes_dict, rows_dict, cols_dict, edge_attrs=None, node_attrs=None):
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