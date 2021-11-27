import torch
import torch.utils.data
import os
import torchvision


def train_transforms():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])


def test_transforms():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])


def load_stl10_data(cfg):
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(cfg.stl10_dataset_path, 'train'),
        transform=train_transforms()
    )
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(cfg.stl10_dataset_path, 'test'),
        transform=test_transforms()
    )

    train_dataloader = torch.utils.data.dataloader.DataLoader(
        train_dataset, shuffle=True, batch_size=cfg.num_negative_samples+1)
    test_dataloader = torch.utils.data.dataloader.DataLoader(
        test_dataset, batch_size=cfg.num_negative_samples+1)

    return train_dataset, train_dataloader, test_dataset, test_dataloader
