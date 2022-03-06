from typing import Union, Dict

from torch import Tensor

MixedData = Union[Tensor, Dict[str, Tensor]]


def validate_mixeddata(data: MixedData, hetero: bool = False, dtype=None):
    if hetero:
        assert isinstance(data, dict)
        for v in data.values():
            assert v.dtype == dtype
    else:
        assert data.dtype == dtype
