import efficientnet_pytorch
import torch
from torch import nn
from torch.nn import functional as F


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class LinearClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearClassifier, self).__init__()
        self.fc_1 = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        return self.fc_1(x)


class HalfAlexNet(nn.Module):
    def __init__(self, in_channel=1, feat_dim=128, pool_type='max'):
        super(HalfAlexNet, self).__init__()
        self.pool_type = pool_type
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

    def pool_flatten(self, x, pool_size):
        if self.pool_type == 'max':
            x = F.adaptive_max_pool2d(x, (pool_size, pool_size))
        elif self.pool_type == 'avg':
            x = F.adaptive_avg_pool2d(x, (pool_size, pool_size))
        else:
            raise NotImplementedError()
        x = x.view(x.shape[0], -1)
        return x

    def forward(self, x, layer):
        x = self.conv_block_1(x)
        if layer == 1:
            x = self.pool_flatten(x, pool_size=10)
            return x

        x = self.conv_block_2(x)
        if layer == 2:
            x = self.pool_flatten(x, pool_size=6)
            return x

        x = self.conv_block_3(x)
        if layer == 3:
            x = self.pool_flatten(x, pool_size=5)
            return x

        x = self.conv_block_4(x)
        if layer == 4:
            x = self.pool_flatten(x, pool_size=5)
            return x

        x = self.conv_block_5(x)
        if layer == 5:
            x = torch.flatten(x, start_dim=1)
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

    def output_dim(self, layer):
        if layer == 1:
            pool_size = 10
            n_channels = 96
        elif layer == 2:
            pool_size = 6
            n_channels = 256
        elif layer == 3:
            pool_size = 5
            n_channels = 384
        elif layer == 4:
            pool_size = 5
            n_channels = 384
        elif layer == 5:
            pool_size = 6
            n_channels = 256
        else:
            raise NotImplementedError()
        return n_channels * pool_size * pool_size // 2


class EfficientNet(nn.Module):
    def __init__(self, in_channel=1, feat_dim=128):
        super(EfficientNet, self).__init__()
        self.in_channels = in_channel
        self.feat_dim = feat_dim
        self.encoder = efficientnet_pytorch.EfficientNet.from_name('efficientnet-b0', in_channels=in_channel, num_classes=feat_dim)
        self.l2norm = Normalize(2)
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x, layer):
        if layer == 15:
            x = self.encoder.extract_features(x)
            x = self.flatten(x)
            return x
        x = self.encoder(x)
        x = self.l2norm(x)
        return x

    def output_dim(self, layer):
        if layer == 15:
            return 1280
        else:
            return self.feat_dim




