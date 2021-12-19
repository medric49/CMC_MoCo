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
        self.enc_train_dataset, self.enc_train_dataloader = datasets.load_stl10_enc_train_data(cfg)
        self.class_train_dataset, self.class_train_dataloader = datasets.load_stl10_class_train_data(cfg)
        self.valid_dataset, self.valid_dataloader = datasets.load_stl10_test_data(cfg)

        self.encoder = Encoder(self.cfg).to(utils.device())
        self.global_enc_epoch = 0
        self.global_enc_min_loss = np.inf

        self.classifier = Classifier(self.cfg, feature_dim=self.encoder.output_dim(self.cfg.layer))
        self.global_class_epoch = 0
        self.global_class_min_loss = np.inf

    def train_encoder(self):
        self.encoder.train()
        train_until_epoch = utils.Until(self.cfg.num_enc_epochs)
        while train_until_epoch(self.global_enc_epoch):
            metrics = dict()
            n_samples = 0
            epoch_losses = []
            loader = tqdm(self.enc_train_dataloader)
            loader.set_postfix({'epoch': self.global_enc_epoch})
            for images, labels in loader:
                images = images.to(device=utils.device(), dtype=torch.float)
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
        train_class_until_epoch = utils.Until(self.cfg.num_class_epochs)
        while train_class_until_epoch(self.global_class_epoch):
            metrics = dict()
            n_samples = 0
            train_epoch_losses = []
            train_epoch_scores = []
            loader = tqdm(self.class_train_dataloader)
            loader.set_postfix({'epoch': self.global_class_epoch})
            self.classifier.train()
            for images, labels in loader:
                images = images.to(device=utils.device(), dtype=torch.float)
                labels = labels.to(device=utils.device(), dtype=torch.long)
                with torch.no_grad():
                    features = self.encoder(images, self.cfg.layer)
                loss, score = self.classifier.update(features, labels)

                n_samples += images.shape[0]
                train_epoch_losses.append(loss * images.shape[0])
                train_epoch_scores.append(score * images.shape[0])
                loader.set_postfix({
                    'epoch': self.global_class_epoch,
                    'loss': np.sum(train_epoch_losses)/n_samples,
                    'score': np.sum(train_epoch_scores)/n_samples
                })
            train_epoch_loss = np.sum(train_epoch_losses) / n_samples
            train_epoch_score = np.sum(train_epoch_scores) / n_samples
            metrics['epoch_loss'] = train_epoch_loss
            metrics['epoch_score'] = train_epoch_score

            n_samples = 0
            valid_epoch_losses = []
            valid_epoch_scores = []
            loader = tqdm(self.valid_dataloader, colour='green')
            loader.set_postfix({'epoch': self.global_class_epoch})
            self.classifier.eval()
            for images, labels in loader:
                images = images.to(device=utils.device(), dtype=torch.float)
                labels = labels.to(device=utils.device(), dtype=torch.long)
                with torch.no_grad():
                    features = self.encoder(images, self.cfg.layer)
                loss, score = self.classifier.evaluate(features, labels)

                n_samples += images.shape[0]
                valid_epoch_losses.append(loss * images.shape[0])
                valid_epoch_scores.append(score * images.shape[0])
                loader.set_postfix({
                    'epoch': self.global_class_epoch,
                    'loss': np.sum(valid_epoch_losses) / n_samples,
                    'score': np.sum(valid_epoch_scores) / n_samples
                })
            valid_epoch_loss = np.sum(valid_epoch_losses) / n_samples
            valid_epoch_score = np.sum(valid_epoch_scores) / n_samples
            metrics['valid_epoch_loss'] = valid_epoch_loss
            metrics['valid_epoch_score'] = valid_epoch_score

            self.logger.log_metrics(metrics, self.global_class_epoch, ty='train_class')
            if self.cfg.save_snapshot:
                self.save_snapshot()
            if self.global_class_min_loss >= valid_epoch_loss:
                self.global_class_min_loss = valid_epoch_loss
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

        if self.cfg.reset_classifier:
            self.classifier = Classifier(self.cfg,
                                         feature_dim=self.encoder.output_dim(self.cfg.layer)).to(utils.device())
            self.global_class_epoch = 0
            self.global_class_min_loss = np.inf

        if self.global_class_epoch < self.cfg.num_class_epochs:
            print('### CLASSIFIER TRAINING ###')
            self.train_classifier()

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = [
            'encoder', 'global_enc_epoch', 'global_enc_min_loss',
            'classifier', 'global_class_epoch', 'global_class_min_loss'
        ]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def save_min_loss_snapshot(self, ty):
        snapshot = self.work_dir / f'{ty}_min_loss_snapshot.pt'
        keys_to_save = [
            'encoder', 'global_enc_epoch', 'global_enc_min_loss',
            'classifier', 'global_class_epoch', 'global_class_min_loss'
        ]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'

        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

