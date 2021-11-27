import random

import torch
from torch import nn
from torch.nn import functional as F
from typing import List

import utils


class HalfAlexNet(nn.Module):
    def __init__(self, in_channel=1, feat_dim=128):
        super(HalfAlexNet, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, 96 // 2, 11, 4, 2, bias=False),  # 224 -> 111
            nn.BatchNorm2d(96 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # 111 -> 55
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(96 // 2, 256 // 2, 5, 1, 2, bias=False),  # 55 -> 27
            nn.BatchNorm2d(256 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # 27 -> 13
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(256 // 2, 384 // 2, 3, 1, 1, bias=False),  # 13 -> 13
            nn.BatchNorm2d(384 // 2),
            nn.ReLU(inplace=True),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(384 // 2, 384 // 2, 3, 1, 1, bias=False),  # 13 -> 13
            nn.BatchNorm2d(384 // 2),
            nn.ReLU(inplace=True),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(384 // 2, 256 // 2, 3, 1, 1, bias=False),  # 13 -> 13
            nn.BatchNorm2d(256 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # 13 ->  6
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256 * 6 * 6 // 2, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(4096 // 2, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc8 = nn.Sequential(
            nn.Linear(4096 // 2, feat_dim)
        )
        self.l2norm = Normalize(2)

    def pool_flatten(self, x, pool_size, pool_type):
        if pool_type == 'max':
            x = F.adaptive_max_pool2d(x, (pool_size, pool_size))
        elif pool_type == 'avg':
            x = F.adaptive_avg_pool2d(x, (pool_size, pool_size))
        else:
            raise NotImplementedError()
        x = x.view(x.shape[0], -1)
        return x

    def forward(self, x, layer, pool_type):
        if layer <= 0:
            raise NotImplementedError()

        x = self.conv_block_1(x)
        if layer == 1:
            x = self.pool_flatten(x, pool_size=10, pool_type=pool_type)
            return x

        x = self.conv_block_2(x)
        if layer == 2:
            x = self.pool_flatten(x, pool_size=6, pool_type=pool_type)
            return x

        x = self.conv_block_3(x)
        if layer == 3:
            x = self.pool_flatten(x, pool_size=5, pool_type=pool_type)
            return x

        x = self.conv_block_4(x)
        if layer == 4:
            x = self.pool_flatten(x, pool_size=5, pool_type=pool_type)
            return x

        x = self.conv_block_5(x)
        if layer == 5:
            x = self.pool_flatten(x, pool_size=6, pool_type=pool_type)
            return x

        x = x.view(x.shape[0], -1)

        x = self.fc6(x)
        if layer == 6:
            return x

        x = self.fc7(x)
        if layer == 7:
            return x

        x = self.fc8(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class CMC:
    def __init__(self, cfg):
        self.cfg = cfg

        self.encoders: List[HalfAlexNet] = [
            HalfAlexNet(depth) for depth in self.cfg.colorspace.view_depths
        ]

        self.optimizers: List[torch.optim.Optimizer] = [
            torch.optim.Adam(
                encoder.parameters(),
                lr=self.cfg.lr,
                betas=(self.cfg.beta_1, self.cfg.beta_2),
                weight_decay=self.cfg.weight_decay
            ) for encoder in self.encoders
        ]

    def train(self):
        for encoder in self.encoders:
            encoder.train()

    def eval(self):
        for encoder in self.encoders:
            encoder.eval()

    def to(self, device):
        for encoder in self.encoders:
            encoder.to(device)
        return self

    def __call__(self, x: torch.Tensor, layer=8, pool_type='max'):
        return self.encode(x, layer=layer, pool_type=pool_type, concat=True)

    def encode(self, x: torch.Tensor, layer=8, pool_type='max', concat=False):
        x = x.to(dtype=torch.float)
        views_list = list(torch.split(x, list(self.cfg.colorspace.view_depths), dim=1))
        vectors_list = [self.encoders[i](views, layer=layer, pool_type=pool_type) for i, views in enumerate(views_list)]
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
        x = x.to(device=utils.device(), dtype=torch.float)
        vectors_list = self.encode(x)
        loss = self.full_graph_loss(vectors_list)
        loss.backward()
        self.step_optimizers()
        return loss

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
        loss = None
        for i in range(1, len(vectors_list)):
            if loss is None:
                loss = self._two_view_loss(vectors_list[0], vectors_list[i])
            else:
                loss += self._two_view_loss(vectors_list[0], vectors_list[i])
        return loss

    def full_graph_loss(self, vectors_list):
        loss = None
        for i in range(len(vectors_list)-1):
            for j in range(i+1, len(vectors_list)):
                if loss is None:
                    loss = self._two_view_loss(vectors_list[i], vectors_list[j])
                else:
                    loss += self._two_view_loss(vectors_list[i], vectors_list[j])
        return loss


