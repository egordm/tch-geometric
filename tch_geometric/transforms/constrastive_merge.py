from collections import defaultdict

import torch
from torch_geometric.data import HeteroData, Data

from tch_geometric.utils import zip_dict


class ContrastiveMergeTransform:
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pos: HeteroData, neg: HeteroData) -> HeteroData:
        result = HeteroData()

        nodes_start_neg = {}
        for (node_type, (pos_store, neg_store)) in zip_dict(pos._node_store_dict, neg._node_store_dict):
            nodes_start_neg[node_type] = pos_store.num_nodes
            result[node_type].x = torch.cat([pos_store.x, neg_store.x], dim=0)

        for (edge_type, (pos_store, neg_store)) in zip_dict(pos._edge_store_dict, neg._edge_store_dict):
            (src, rel, dst) = edge_type

            pos_edge_store = result[(src, f'{rel}_pos', dst)]
            pos_edge_store.edge_index = pos_store.edge_index
            pos_edge_store.type = 'pos'

            neg_edge_store = result[(src, f'{rel}_neg', dst)]
            neg_edge_store.edge_index = neg_store.edge_index
            neg_edge_store.edge_index[1, :] += nodes_start_neg[dst]
            neg_edge_store.type = 'neg'

        return result


class EdgeTypeAggregateTransform:
    def __init__(self) -> None:
        super().__init__()

    def __call__(
            self,
            data: HeteroData
    ) -> HeteroData:
        result = HeteroData()
        node_type_default = 'n'

        # Merge all nodes into a single tensor while preserving the type specific offsets
        offset = 0
        node_offsets = {}
        xs = []
        for store in data.node_stores:
            node_type = store._key
            xs.append(store.x)
            node_offsets[node_type] = offset
            offset += store.num_nodes
        result[node_type_default].x = torch.cat(xs, dim=0)

        # Merge all edges into a single tensor and correct edges with correct node offsets
        edge_indexes_dict = defaultdict(list)
        for store in data.edge_stores:
            (src, _, dst) = store._key
            type = store.type

            edge_index = store.edge_index
            edge_index[0, :] += node_offsets[src]
            edge_index[1, :] += node_offsets[dst]

            edge_indexes_dict[(node_type_default, type, node_type_default)].append(edge_index)

        for edge_type, edge_indexes in edge_indexes_dict.items():
            result[edge_type].edge_index = torch.cat(edge_indexes, dim=1)

        return result