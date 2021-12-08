import os

import hydra.utils
import torch
import torch.utils.data
import torchvision
from pathlib import Path
from hydra.utils import to_absolute_path

import colorspaces


def load_stl10_train_data(cfg):
    mean = cfg.im_norm_mean
    std = cfg.im_norm_std
    color_transfer = hydra.utils.instantiate(cfg.colorspace.translator)
    normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(cfg.image_size, scale=(cfg.crop, 1.)),
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
    test_dataset = torchvision.datasets.ImageFolder(
        to_absolute_path((Path(cfg.stl10_dataset_path) / 'test')),
    )
    test_dataloader = torch.utils.data.dataloader.DataLoader(
        test_dataset, batch_size=cfg.batch_size)
    return test_dataset, test_dataloader

