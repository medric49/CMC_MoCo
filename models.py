import random

import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()


class CMC:

    def __init__(self, cfg):
        self.cfg = cfg

        self.encoders = [AlexNet() for i in range(self.cfg.num_views)]

        self.optimizers = [
            torch.optim.Adam(self.encoders[i].parameters(), lr=self.cfg.lr) for i in range(self.cfg.num_views)
        ]


    def train(self):
        for encoder in self.encoders:
            encoder.train()

    def eval(self):
        for encoder in self.encoders:
            encoder.eval()

    def encode(self, views_list):
        return [self.encoders[i][views] for i, views in enumerate(views_list)]

    def contrast_loss(self, vectors_1, vectors_2):
        i = random.randint(0, vectors_1.shape[0] - 1)

        z1 = vectors_1[i].repeat((vectors_1.shape[0], 1))
        z2 = vectors_2
        similarities = torch.cosine_similarity(z1, z2, eps=0)
        critic = torch.log_softmax(similarities * self.cfg.temperature, dim=0)[i]
        return - critic

    def two_view_loss(self, vectors_1, vectors_2):
        return self.contrast_loss(vectors_1, vectors_2) + self.contrast_loss(vectors_2, vectors_1)

    def core_view_loss(self, vectors_list):
        loss = 0
        for i in range(1, len(vectors_list)):
            loss = self.two_view_loss(vectors_list[0], vectors_list[i])
        return loss

    def full_graph_loss(self, vectors_list):
        pass


