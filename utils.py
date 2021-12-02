import random
import numpy as np
import torch


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if self._until is None:
            return True
        return step < self._until


class Every:
    def __init__(self, every):
        self._every = every

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every
        if step % every == 0:
            return True
        return False
