import torch
import torch.utils.data

import datasets
import utils


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg

        utils.set_seed_everywhere(self.cfg.seed)

        self.train_dataset, self.train_dataloader, self.test_dataset, self.test_dataloader = datasets.load_stl10_data(cfg)

    @property
    def device(self):
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

