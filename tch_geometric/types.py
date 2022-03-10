from typing import Union, Dict, Tuple

from torch import Tensor

MixedData = Union[Tensor, Dict[str, Tensor]]
HeteroTensor = Dict[str, Tensor]

Timerange = Tuple[int, int]

def validate_mixeddata(data: MixedData, hetero: bool = False, dtype=None):
    if hetero:
        assert isinstance(data, dict)
        for v in data.values():
            assert v.dtype == dtype
    else:
        assert data.dtype == dtype
