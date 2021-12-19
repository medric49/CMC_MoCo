import random
from typing import List

import hydra.utils
import torch
from torch import nn

import utils
from losses import SupConLoss
from nets import LinearClassifier


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        self._encoders = nn.ModuleList(
            [
                hydra.utils.instantiate(self.cfg.encoder, in_channel=depth) for depth in
                self.cfg.colorspace.view_depths
            ]
        )
        self._optimizer = torch.optim.Adam(self._encoders.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta_1, self.cfg.beta_2))

        self._criterion = SupConLoss(
            contrast_mode='all' if self.cfg.full_graph else 'one',
            temperature=self.cfg.temperature,
            base_temperature=self.cfg.temperature
        )

    def output_dim(self, layer):
        return self._encoders[0].output_dim(layer) * len(self._encoders)

    def forward(self, x, layer):
        return self.encode(x, layer, concat=True)

    def encode(self, x: torch.Tensor, layer=None, concat=False):
        views_list = list(torch.split(x, list(self.cfg.colorspace.view_depths), dim=1))
        vectors_list = [self._encoders[i](views, layer) for i, views in enumerate(views_list)]
        if concat:
            vectors_list = torch.cat(vectors_list, dim=1)
        return vectors_list

    def update(self, x: torch.Tensor):
        self._optimizer.zero_grad()
        vectors_list = self.encode(x)
        loss = self._criterion(torch.stack(vectors_list, dim=1))
        # loss = self.full_graph_loss(vectors_list) if self.cfg.full_graph else self.core_view_loss(vectors_list)
        loss.backward()
        self._optimizer.step()
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
        loss = torch.tensor(0, dtype=torch.float, device=utils.device())
        for i in range(1, len(vectors_list)):
            loss += self._two_view_loss(vectors_list[0], vectors_list[i])
        return loss

    def full_graph_loss(self, vectors_list):
        loss = torch.tensor(0, dtype=torch.float, device=utils.device())
        for i in range(len(vectors_list)-1):
            for j in range(i+1, len(vectors_list)):
                loss += self._two_view_loss(vectors_list[i], vectors_list[j])
        return loss


class Classifier(nn.Module):
    def __init__(self, cfg, feature_dim):
        super(Classifier, self).__init__()
        self.cfg = cfg
        self._classifier = LinearClassifier(feature_dim, 10)
        self._optimizer = torch.optim.Adam(
            self._classifier.parameters(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta_1, self.cfg.beta_2),
        )
        self._criterion = nn.CrossEntropyLoss()

    def update(self, x: torch.Tensor, y: torch.Tensor):
        self._optimizer.zero_grad()
        output = self._classifier(x)
        loss = self._criterion(output, y)
        loss.backward()
        self._optimizer.step()

        loss = loss.item()
        score = ((output.argmax(dim=1) == y) * 1.).mean().item()

        return loss, score

    def evaluate(self, x: torch.Tensor, y: torch.Tensor):
        output = self._classifier(x)
        loss = self._criterion(output, y)
        loss = loss.item()
        score = ((output.argmax(dim=1) == y) * 1.).mean().item()
        return loss, score

