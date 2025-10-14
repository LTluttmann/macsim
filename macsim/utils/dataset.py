import os 
import math
import numpy as np
from typing import Union
from omegaconf import ListConfig

import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader

from macsim.envs.base import Environment


class SingleGeneratorDataset(Dataset):
    def __init__(
            self, 
            env: Environment,
            num_samples: int, 
            **kwargs
        ) -> None:
        
        super().__init__()
        assert len(env.generators) == 1
        self.dataset = env.generate(batch_size=num_samples)
        self.num_samples = num_samples 

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch, dim=0)
    

class MultiGeneratorDataset(Dataset):
    def __init__(
            self, 
            env: Environment,
            num_samples: int, 
            **kwargs
        ) -> None:
        
        super().__init__()
        self.num_samples = num_samples 
        self.datasets = {g.id: g(batch_size=num_samples) for g in env.generators}

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index_tuple):
        dataset_idx, sample_idx = index_tuple
        return self.datasets[dataset_idx][sample_idx]
    
    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch, dim=0)


class InstanceFilesDataset(Dataset):

    def __init__(
            self, 
            env: Environment, 
            path: str, 
            num_samples: int = None,
            **kwargs
        ) -> None:

        super().__init__()
        self.env = env
        self.instances = self.env.load_instances(path, **kwargs)
        if num_samples is not None:
            self.instances = self.instances[:num_samples]
        self.num_samples = len(self.instances)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        td = self.instances[idx]
        return td

    def collate_fn(self, batch):
        return torch.stack(batch, 0)


class SequentialSampler(Sampler[int]):

    def __init__(
            self, 
            data_source: MultiGeneratorDataset,
            batch_size: int,
            dataset_distribution = None,
            drop_last: bool = True,
        ) -> None:
        """
        Sequential sampler that samples mini-batches from different datasets.
        
        Args:
            data_source: MultiGeneratorDataset containing multiple datasets
            batch_size: Number of samples per batch
            dataset_distribution: Probability distribution for sampling datasets
            drop_last: Whether to drop the last incomplete batch
        """
        self.batch_size = batch_size
        self.datasets = data_source.datasets
        self.dataset_indices = list(self.datasets.keys())
        self.dataset_distribution = dataset_distribution
        self.drop_last = drop_last


    @property
    def num_batches(self) -> int:
        """
        Calculate the number of batches. 
        Ref: https://github.com/pytorch/pytorch/blob/ee21ccc81620e5e9ff798838d9243338547d60cb/torch/utils/data/sampler.py#L345
        """
        if self.drop_last:
            return len(self) // self.batch_size
        else:
            return (len(self) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        sampler_iter = iter(range(len(self)))
        if self.dataset_distribution is not None:
            dataset_iter = iter(np.random.choice(self.dataset_indices, size=self.num_batches, p=self.dataset_distribution))
        else:
            repeats = math.ceil(self.num_batches / len(self.dataset_indices))
            dataset_iter = iter(np.tile(self.dataset_indices, repeats)[:self.num_batches])
        while True:
            try:
                dataset_idx = next(dataset_iter)
                batch = [next(sampler_iter) for _ in range(self.batch_size)]
                yield from [(dataset_idx, i) for i in batch]
            except StopIteration:
                break

    def __len__(self) -> int:
        return min([len(x) for x in self.datasets.values()]) 


    
class EnvLoader(DataLoader):
    def __init__(
        self, 
        env: Environment, 
        batch_size: int = 1,
        dataset_size: int = None,
        path: str = None,
        shuffle: bool = False,
        sampler = None,
        batch_sampler = None,
        dataset_distribution = None,
        drop_last: bool = False,
        **kwargs
    ) -> None:

        if path is not None:
            dataset = InstanceFilesDataset(
                env, 
                path, 
                num_samples=dataset_size, 
                **kwargs
            )
            super().__init__(
                dataset=dataset, 
                batch_size=batch_size, 
                sampler=sampler, 
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                drop_last=drop_last,
                # num_workers=20,
                # pin_memory=True
            )

        elif len(env.generators) > 1:
            dataset = MultiGeneratorDataset(
                env=env,
                num_samples=dataset_size,
                **kwargs
            )
            if sampler is not None:
                raise ValueError(
                    """passing sampler externally for multiple generators is not implemented yet.
                    Most likely case why you see this error is using multiple GPUs for training."""
                )

            sampler = SequentialSampler(dataset, batch_size, dataset_distribution)
            super().__init__(
                dataset=dataset, 
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                sampler=sampler,
                batch_sampler=batch_sampler,
                shuffle=shuffle,
                drop_last=drop_last,
                # num_workers=20,
                # persistent_workers=True,
                # pin_memory=True
            )

        else:
            dataset = SingleGeneratorDataset(
                env=env,
                num_samples=dataset_size, 
                **kwargs
            )
            super().__init__(
                dataset=dataset, 
                batch_size=batch_size, 
                collate_fn=dataset.collate_fn,
                sampler=sampler,
                batch_sampler=batch_sampler,
                shuffle=shuffle,
                drop_last=drop_last,
                # num_workers=20,
                # persistent_workers=True,
                # pin_memory=True
            )
        
        
def get_file_dataloader(
        env: Environment, 
        batch_size: int, 
        file_dirs: Union[ListConfig, list, str] = None, 
        num_file_instances: int = None,
        **kwargs
    ):

    if file_dirs is None:
        return {}

    file_dirs = [file_dirs] if not isinstance(file_dirs, (list, ListConfig)) else file_dirs
    dataloader = {}
    for file_dir in file_dirs:
        dl_id = os.path.basename(file_dir)
        try:
            dataloader[dl_id] = EnvLoader(
                env=env,
                path=file_dir, 
                batch_size=batch_size,
                dataset_size=num_file_instances,
                **kwargs
            )
        except FileNotFoundError:
            continue

    return dataloader