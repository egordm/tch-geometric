from typing import Callable, Union, List, Dict, Optional

from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from tch_geometric.transforms.hgt_sampling import HGTSamplerTransform
from tch_geometric.loader.custom_loader import CustomLoader


class HGTLoader(CustomLoader):
    def __init__(
            self,
            dataset: Dataset,
            neighbor_sampler: HGTSamplerTransform = None,
            data: HeteroData = None,
            num_samples: Union[List[int], Dict[NodeType, List[int]]] = None,
            temporal: bool = False,
            batch_size_tmp: int = None,
            **kwargs
    ):
        super().__init__(dataset, batch_size_tmp=batch_size_tmp, **kwargs)

        self.neighbor_sampler = neighbor_sampler
        if not self.neighbor_sampler:
            self.neighbor_sampler = HGTSamplerTransform(data, num_samples, temporal)

    def sample(self, inputs: HeteroData):
        inputs_dict = inputs.x_dict
        inputs_timestamps_dict = inputs.timestamp_dict if self.neighbor_sampler.temporal else None
        timerange = None

        return self.neighbor_sampler(inputs_dict, inputs_timestamps_dict, timerange)
