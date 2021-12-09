import random
from typing import List

import hydra.utils
import torch
from torch import nn

from losses import SupConLoss
from nets import LinearClassifier


class Encoder:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        self.encoders: List[nn.Module] = [
            hydra.utils.instantiate(self.cfg.encoder, in_channel=depth).to(device) for depth in self.cfg.colorspace.view_depths
        ]

        self.optimizers: List[torch.optim.Optimizer] = [
            torch.optim.Adam(
                encoder.parameters(),
                lr=self.cfg.lr,
                betas=(self.cfg.beta_1, self.cfg.beta_2)
            ) for encoder in self.encoders
        ]

        self.criterion = SupConLoss(
            contrast_mode='all' if self.cfg.full_graph else 'one',
            temperature=self.cfg.temperature,
            base_temperature=self.cfg.temperature
        )

    def train(self):
        for encoder in self.encoders:
            encoder.train()

    def eval(self):
        for encoder in self.encoders:
            encoder.eval()

    def to(self, device):
        self.device = device
        for encoder in self.encoders:
            encoder.to(device)
        return self

    def output_dim(self, layer):
        return self.encoders[0].output_dim(layer) * len(self.encoders)

    def __call__(self, x: torch.Tensor, layer, pool_type='max'):
        return self.encode(x, layer=layer, concat=True)

    def encode(self, x: torch.Tensor, layer=None, concat=False):
        x = x.to(device=self.device, dtype=torch.float)
        views_list = list(torch.split(x, list(self.cfg.colorspace.view_depths), dim=1))
        vectors_list = [self.encoders[i](views, layer) for i, views in enumerate(views_list)]
        if concat:
            vectors_list = torch.cat(vectors_list, dim=1)
        return vectors_list

    def zero_grad_optimizers(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step_optimizers(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def update(self, x: torch.Tensor):
        self.zero_grad_optimizers()
        vectors_list = self.encode(x)
        loss = self.criterion(torch.stack(vectors_list, dim=1))
        # loss = self.full_graph_loss(vectors_list) if self.cfg.full_graph else self.core_view_loss(vectors_list)
        loss.backward()
        self.step_optimizers()
        return loss.item()

    def _contrast_loss(self, vectors_1, vectors_2):
        i = random.randint(0, vectors_1.shape[0] - 1)

        z1 = vectors_1[i].repeat((vectors_1.shape[0], 1))
        z2 = vectors_2
        similarities = torch.cosine_similarity(z1, z2, eps=0)
        critic = torch.log_softmax(similarities * self.cfg.temperature, dim=0)[i]
        return - critic

    def _two_view_loss(self, vectors_1, vectors_2):
        return self._contrast_loss(vectors_1, vectors_2) + self._contrast_loss(vectors_2, vectors_1)

    def core_view_loss(self, vectors_list):
        loss = torch.tensor(0, dtype=torch.float, device=self.device)
        for i in range(1, len(vectors_list)):
            loss += self._two_view_loss(vectors_list[0], vectors_list[i])
        return loss

    def full_graph_loss(self, vectors_list):
        loss = torch.tensor(0, dtype=torch.float, device=self.device)
        for i in range(len(vectors_list)-1):
            for j in range(i+1, len(vectors_list)):
                loss += self._two_view_loss(vectors_list[i], vectors_list[j])
        return loss


class Classifier:
    def __init__(self, cfg, device, feature_dim):
        self.cfg = cfg
        self.device = device
        self.classifier = LinearClassifier(feature_dim, 10).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta_1, self.cfg.beta_2),
        )
        self.criterion = nn.CrossEntropyLoss()

    def to(self, device):
        self.device = device
        self.classifier.to(device)

    def update(self, x: torch.Tensor, y: torch.Tensor):
        y = y.to(device=self.device, dtype=torch.long)
        self.optimizer.zero_grad()
        output = self.classifier(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()

        loss = loss.item()
        score = ((output.argmax(dim=1) == y) * 1.).mean().item()

        return loss, score

    def evaluate(self, x: torch.Tensor, y: torch.Tensor):
        y = y.to(device=self.device, dtype=torch.long)
        output = self.classifier(x)
        loss = self.criterion(output, y)
        loss = loss.item()
        score = ((output.argmax(dim=1) == y) * 1.).mean().item()
        return loss, score

    def train(self):
        self.classifier.train()

    def eval(self):
        self.classifier.eval()

