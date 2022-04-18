from typing import Union, Tuple, List, Optional, Dict

from torch import Tensor
from torch_geometric.typing import NodeType, EdgeType

from tch_geometric.transforms import EdgeSampler, EdgeFilter

LayerOffset = (int, int, int)
RelType = str


def to_csc(row_col: Tensor, size: Union[int, Tuple[int, int]]) -> Tuple[Tensor, Tensor, Tensor]:
    ...


def to_csr(row_col: Tensor, size: Union[int, Tuple[int, int]]) -> Tuple[Tensor, Tensor, Tensor]:
    ...


def neighbor_sampling_homogenous(
        col_ptrs: Tensor,
        row_indices: Tensor,
        inputs: Tensor,
        num_neighbors: List[int],
        sampler: Optional[EdgeSampler],
        filter: Optional[Tuple[EdgeFilter, Tensor]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[LayerOffset]]:
    ...


def neighbor_sampling_heterogenous(
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        col_ptrs: Dict[RelType, Tensor],
        row_indices: Dict[RelType, Tensor],
        inputs: Dict[NodeType, Tensor],
        num_neighbors: Dict[RelType, List[int]],
        num_hops: int,
        sampler: Optional[EdgeSampler],
        filter: Optional[Tuple[EdgeFilter, Tensor]],
) -> Tuple[
    Dict[NodeType, Tensor], Dict[RelType, Tensor], Dict[RelType, Tensor], Dict[RelType, Tensor], List[LayerOffset]
]:
    ...


def hgt_sampling(
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        col_ptrs: Dict[RelType, Tensor],
        row_indices: Dict[RelType, Tensor],
        row_timestamps: Optional[Dict[RelType, Tensor]],
        inputs: Dict[NodeType, Tensor],
        input_timestamps: Optional[Dict[NodeType, Tensor]],
        num_samples: Dict[NodeType, List[int]],
        num_hops: int,
        timerange: Optional[Tuple[int, int]] = None,
) -> Tuple[
    Dict[NodeType, Tensor], Dict[NodeType, Tensor], Dict[RelType, Tensor], Dict[RelType, Tensor], Dict[RelType, Tensor]
]:
    ...


def budget_sampling(
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        col_ptrs: Dict[RelType, Tensor],
        row_indices: Dict[RelType, Tensor],
        row_timestamps: Optional[Dict[RelType, Tensor]],
        inputs: Dict[NodeType, Tensor],
        input_timestamps: Optional[Dict[NodeType, Tensor]],
        num_neighbors: Dict[NodeType, List[int]],
        num_hops: int,
        window: Optional[Tuple[int, int]],
        forward: bool,
        relative: bool,
) -> Tuple[
    Dict[NodeType, Tensor], Dict[NodeType, Tensor], Dict[RelType, Tensor], Dict[RelType, Tensor], Dict[RelType, Tensor]
]:
    ...


def random_walk(
        row_ptrs: Tensor,
        col_indices: Tensor,
        start: Tensor,
        walk_length: int,
        p: float,
        q: float,
) -> Tensor:
    ...


def tempo_random_walk(
        row_ptrs: Tensor,
        col_indices: Tensor,
        node_timestamps: Tensor,
        edge_timestamps: Tensor,
        start: Tensor,
        start_timestamps: Tensor,
        walk_length: int,
        window: Tuple[int, int],
) -> Tuple[Tensor, Tensor]:
    ...


def negative_sample_neighbors_homogenous(
        row_ptrs: Tensor,
        col_indices: Tensor,
        graph_size: Tuple[int, int],
        inputs: Tensor,
        num_neg: int,
        try_count: int,
) -> Tuple[Tensor, Tensor, Tensor, int]:
    ...


def negative_sample_neighbors_heterogenous(
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        row_ptrs: Dict[RelType, Tensor],
        col_indices: Dict[RelType, Tensor],
        sizes: Dict[RelType, Tuple[int, int]],
        inputs: Dict[NodeType, Tensor],
        num_neg: int,
        try_count: int,
        inbound: bool,
) -> Tuple[
    Dict[NodeType, Tensor], Dict[RelType, Tensor], Dict[RelType, Tensor], Dict[NodeType, int]
]:
    ...
