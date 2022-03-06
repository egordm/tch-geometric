from typing import Union, Tuple, Dict

from torch import Tensor
from torch_geometric.data import HeteroData, Data
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.typing import EdgeType

import tch_geometric.tch_geometric as native

RelType = str
Size = Tuple[int, int]


def edge_type_to_str(edge_type: Union[EdgeType, str]) -> RelType:
    # It is faster to have keys consisting of single string
    return edge_type if isinstance(edge_type, str) else '__'.join(edge_type)


def to_sparse(data: Union[Data, EdgeStorage], sparse_fn) -> Tuple[Tensor, Tensor, Tensor, Size]:
    if not hasattr(data, 'edge_index'):
        raise AttributeError("Data object does not contain attribute 'edge_index'")

    size = data.size()
    ptrs, indices, perm = sparse_fn(data.edge_index, size)
    return ptrs, indices, perm, size


def to_csr(data: Union[Data, EdgeStorage]) -> Tuple[Tensor, Tensor, Tensor, Size]:
    return to_sparse(data, native.to_csr)


def to_csc(data: Union[Data, EdgeStorage]) -> Tuple[Tensor, Tensor, Tensor, Size]:
    return to_sparse(data, native.to_csc)


def to_hetero_sparse(data: HeteroData, sparse_fn) \
        -> Tuple[Dict[RelType, Tensor], Dict[RelType, Tensor], Dict[RelType, Tensor], Dict[RelType, Size]]:
    ptrs_dict, indices_dict, perm_dict, size_dict = {}, {}, {}, {}

    for store in data.edge_stores:
        key = edge_type_to_str(store._key)
        ptrs_dict[key], indices_dict[key], perm_dict[key], size_dict[key] = sparse_fn(store)

    return ptrs_dict, indices_dict, perm_dict, size_dict


def to_hetero_csc(data: HeteroData) \
        -> Tuple[Dict[RelType, Tensor], Dict[RelType, Tensor], Dict[RelType, Tensor], Dict[RelType, Size]]:
    return to_hetero_sparse(data, to_csc)


def to_hetero_csr(data: HeteroData) \
        -> Tuple[Dict[RelType, Tensor], Dict[RelType, Tensor], Dict[RelType, Tensor], Dict[RelType, Size]]:
    return to_hetero_sparse(data, to_csr)
