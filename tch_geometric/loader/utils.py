from typing import Union, Tuple, Dict

from torch import Tensor
from torch_geometric.data import HeteroData, Data
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.typing import EdgeType

import tch_geometric.tch_geometric as lib

RelType = str


def edge_type_to_str(edge_type: Union[EdgeType, str]) -> RelType:
    # It is faster to have keys consisting of single string
    return edge_type if isinstance(edge_type, str) else '__'.join(edge_type)


def to_csc(data: Union[Data, EdgeStorage]) -> Tuple[Tensor, Tensor, Tensor]:
    if not hasattr(data, 'edge_index'):
        raise AttributeError("Data object does not contain attribute 'edge_index'")

    size = data.size()
    col_ptrs, row_indices, perm = lib.data.to_csc(data.edge_index, size)
    return col_ptrs, row_indices, perm


def to_hetero_csc(data: HeteroData) -> Tuple[Dict[RelType, Tensor], Dict[RelType, Tensor], Dict[RelType, Tensor]]:
    col_ptrs_dict, row_indices_dict, perm_dict = {}, {}, {}

    for store in data.edge_stores:
        key = edge_type_to_str(store._key)
        col_ptrs_dict[key], row_indices_dict[key], perm_dict[key] = to_csc(store)

    return col_ptrs_dict, row_indices_dict, perm_dict
