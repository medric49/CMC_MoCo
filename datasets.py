from pathlib import Path

import hydra.utils
import torch
import torch.utils.data
import torchvision
from hydra.utils import to_absolute_path


def load_stl10_train_data(cfg):
    color_transfer = hydra.utils.instantiate(cfg.colorspace.translator)
    normalize = torchvision.transforms.Normalize(mean=cfg.im_norm_mean, std=cfg.im_norm_std)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(cfg.image_size, scale=(cfg.low_crop, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        color_transfer,
        torchvision.transforms.ToTensor(),
        normalize
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        to_absolute_path((Path(cfg.stl10_dataset_path) / 'train')),
        transform=transform
    )
    train_dataloader = torch.utils.data.dataloader.DataLoader(
        train_dataset, shuffle=True, batch_size=cfg.batch_size)
    return train_dataset, train_dataloader


def load_stl10_test_data(cfg):
    color_transfer = hydra.utils.instantiate(cfg.colorspace.translator)
    normalize = torchvision.transforms.Normalize(mean=cfg.im_norm_mean, std=cfg.im_norm_std)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(cfg.image_size),
        color_transfer,
        torchvision.transforms.ToTensor(),
        normalize
    ])
    test_dataset = torchvision.datasets.ImageFolder(
        to_absolute_path((Path(cfg.stl10_dataset_path) / 'test')),
        transform=transform
    )
    test_dataloader = torch.utils.data.dataloader.DataLoader(
        test_dataset, batch_size=cfg.batch_size)
    return test_dataset, test_dataloader

