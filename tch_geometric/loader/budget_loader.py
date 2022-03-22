from typing import Callable, Union, List, Dict, Optional, Tuple

from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from tch_geometric.transforms.budget_sampling import BudgetSamplerTransform
from tch_geometric.transforms.hgt_sampling import HGTSamplerTransform
from tch_geometric.loader.custom_loader import CustomLoader


class BudgetLoader(CustomLoader):
    def __init__(
            self,
            dataset: Dataset,
            neighbor_sampler: BudgetSamplerTransform = None,
            data: HeteroData = None,
            num_neighbors: Union[List[int], Dict[NodeType, List[int]]] = None,
            window: Optional[Tuple[int, int]] = None,
            forward: bool = False,
            relative: bool = True,
            batch_size_tmp: int = None,
            temporal: bool = True,
            **kwargs
    ):
        super().__init__(dataset, batch_size_tmp=batch_size_tmp, **kwargs)

        self.neighbor_sampler = neighbor_sampler
        if not self.neighbor_sampler:
            self.neighbor_sampler = BudgetSamplerTransform(data, num_neighbors, window, forward, relative)

        self.temporal = temporal

    def sample(self, inputs: HeteroData):
        inputs_dict = inputs.x_dict
        inputs_timestamps_dict = inputs.timestamp_dict if self.neighbor_sampler.window else None

        return self.neighbor_sampler(inputs_dict, inputs_timestamps_dict)
