import torch
from torch.utils.data import Sampler


class ResizedRandomSampler(Sampler):
    def __init__(self, n_items, epoch_size):
        super().__init__(None)
        self.n_items = n_items
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        return (i for i in torch.randperm(self.n_items)[:len(self)])
