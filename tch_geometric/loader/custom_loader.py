from abc import abstractmethod
from typing import Callable

from torch.utils.data import Dataset, RandomSampler, SequentialSampler, BatchSampler
from torch_geometric.loader.base import BaseDataLoader


class CustomLoader(BaseDataLoader):
    def __init__(
            self,
            dataset: Dataset,
            transform: Callable = None,
            shuffle: bool = False,
            generator=None,
            batch_size: int = 1,
            drop_last: bool = False,
            **kwargs,
    ):
        self.transform = transform
        self.batch_size = batch_size

        # Default sampler to set autocollate to False
        if shuffle:
            sampler = RandomSampler(dataset, generator=generator)
        else:
            sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        super().__init__(
            dataset,
            collate_fn=self.sample,
            sampler=batch_sampler,
            batch_size=None,
            shuffle=shuffle,
            drop_last=drop_last,
            **kwargs,
        )

    @abstractmethod
    def sample(self, inputs):
        pass

    @abstractmethod
    def transform(self, out):
        pass

    def transform_fn(self, out):
        data = self.transform(out)
        return data if self.transform is None else self.transform(data)
