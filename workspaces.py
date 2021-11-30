from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import datasets
import utils
from loggers import Logger
from models import CMC


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(self.cfg.seed)

        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        self.train_dataset, self.train_dataloader = datasets.load_stl10_train_data(cfg)
        self.model = CMC(self.cfg, device=utils.device())
        self.global_epoch = 0
        self.global_min_loss = np.inf

    def train(self):
        self.model.train()

        train_until_epoch = utils.Until(self.cfg.num_epochs)
        while train_until_epoch(self.global_epoch):
            metrics = dict()
            epoch_losses = []
            loader = tqdm(self.train_dataloader)
            loader.set_postfix({'epoch': self.global_epoch})
            for images, labels in loader:
                loss = self.model.update(images)
                epoch_losses.append(loss)
                loader.set_postfix({'epoch': self.global_epoch, 'loss': np.mean(epoch_losses)})
            epoch_loss = np.mean(epoch_losses)
            metrics['epoch_loss'] = epoch_loss
            self.logger.log_metrics(metrics, self.global_epoch, ty='train')

            if self.cfg.save_snapshot:
                self.save_snapshot()

            if self.global_min_loss >= epoch_loss:
                self.global_min_loss = epoch_loss
                self.save_min_loss_snapshot()
            self.global_epoch += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['model', 'global_epoch', 'global_min_loss']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def save_min_loss_snapshot(self):
        snapshot = self.work_dir / 'min_loss_snapshot.pt'
        keys_to_save = ['model', 'global_epoch', 'global_min_loss']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        if not snapshot.exists():
            snapshot = self.work_dir / 'min_loss_snapshot.pt'

        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

