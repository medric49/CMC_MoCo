from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import datasets
import utils
from loggers import Logger
from models import Encoder, Classifier


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(self.cfg.seed)

        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        self.train_dataset, self.train_dataloader = datasets.load_stl10_train_data(cfg)
        self.encoder = Encoder(self.cfg, device=utils.device())
        self.global_enc_epoch = 0
        self.global_enc_min_loss = np.inf

        self.classifier = Classifier(self.cfg, device=utils.device(), feature_dim=self.encoder.output_dim(self.cfg.layer))
        self.global_class_epoch = 0
        self.global_class_min_loss = np.inf

    def train_encoder(self):
        self.encoder.train()
        train_until_epoch = utils.Until(self.cfg.num_enc_epochs)
        while train_until_epoch(self.global_enc_epoch):
            metrics = dict()
            n_samples = 0
            epoch_losses = []
            loader = tqdm(self.train_dataloader)
            loader.set_postfix({'epoch': self.global_enc_epoch})
            for images, labels in loader:
                loss = self.encoder.update(images)

                n_samples += images.shape[0]
                epoch_losses.append(loss * images.shape[0])
                loader.set_postfix({
                    'epoch': self.global_enc_epoch,
                    'loss': np.sum(epoch_losses) / n_samples
                })

            epoch_loss = np.sum(epoch_losses) / n_samples
            metrics['epoch_loss'] = epoch_loss
            self.logger.log_metrics(metrics, self.global_enc_epoch, ty='train_enc')

            if self.cfg.save_snapshot:
                self.save_snapshot()
            if self.global_enc_min_loss >= epoch_loss:
                self.global_enc_min_loss = epoch_loss
                self.save_min_loss_snapshot(ty='enc')
            self.global_enc_epoch += 1

    def train_classifier(self):
        self.classifier.train()
        train_class_until_epoch = utils.Until(self.cfg.num_class_epochs)
        while train_class_until_epoch(self.global_class_epoch):
            metrics = dict()
            n_samples = 0
            epoch_losses = []
            epoch_scores = []

            loader = tqdm(self.train_dataloader)
            loader.set_postfix({'epoch': self.global_class_epoch})

            for images, labels in loader:
                features = self.encoder(images, self.cfg.layer)
                loss, score = self.classifier.update(features, labels)

                n_samples += images.shape[0]
                epoch_losses.append(loss * images.shape[0])
                epoch_scores.append(score * images.shape[0])
                loader.set_postfix({
                    'epoch': self.global_class_epoch,
                    'loss': np.sum(epoch_losses)/n_samples,
                    'score': np.sum(epoch_scores)/n_samples
                })

            epoch_loss = np.sum(epoch_losses)/n_samples
            epoch_score = np.sum(epoch_scores)/n_samples
            metrics['epoch_loss'] = epoch_loss
            metrics['epoch_score'] = epoch_score
            self.logger.log_metrics(metrics, self.global_class_epoch, ty='train_class')

            if self.cfg.save_snapshot:
                self.save_snapshot()
            if self.global_class_min_loss >= epoch_loss:
                self.global_class_min_loss = epoch_loss
                self.save_min_loss_snapshot(ty='class')

            self.global_class_epoch += 1

    def train(self):
        if self.global_enc_epoch < self.cfg.num_enc_epochs:
            print('### ENCODER TRAINING')
            self.train_encoder()

        snapshot = self.work_dir / 'enc_min_loss_snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        self.__dict__['encoder'] = payload['encoder']

        self.encoder.eval()
        self.encoder.required_grad(False)

        if self.global_class_epoch < self.cfg.num_class_epochs:
            print('### CLASSIFIER TRAINING ###')
            self.train_classifier()

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['encoder', 'global_enc_epoch', 'global_enc_min_loss', 'classifier', 'global_class_epoch', 'global_class_min_loss']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def save_min_loss_snapshot(self, ty):
        snapshot = self.work_dir / f'{ty}_min_loss_snapshot.pt'
        keys_to_save = ['encoder', 'global_enc_epoch', 'global_enc_min_loss', 'classifier', 'global_class_epoch', 'global_class_min_loss']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'

        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

