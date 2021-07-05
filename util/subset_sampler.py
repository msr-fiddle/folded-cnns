import torch


class SubsetSampler(torch.utils.data.Sampler):
    """
    Samples sequentially from a given list of indices.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
