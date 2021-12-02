import hydra.utils
import numpy as np
import torch
import torch.utils.data
import os
import torchvision
from skimage import color

from hydra.utils import to_absolute_path


def load_stl10_train_data(cfg):
    mean = cfg.im_norm_mean
    std = cfg.im_norm_std
    color_transfer = hydra.utils.instantiate(cfg.color_space)
    normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(cfg.image_size, scale=(cfg.crop, 1.)),
        torchvision.transforms.RandomHorizontalFlip(),
        color_transfer,
        torchvision.transforms.ToTensor(),
        normalize
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        to_absolute_path(os.path.join(cfg.stl10_dataset_path, 'train')),
        transform=transform
    )
    train_dataloader = torch.utils.data.dataloader.DataLoader(
        train_dataset, shuffle=True, batch_size=cfg.batch_size)
    return train_dataset, train_dataloader


def load_stl10_test_data(cfg):
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(cfg.stl10_dataset_path, 'test'),
    )
    test_dataloader = torch.utils.data.dataloader.DataLoader(
        test_dataset, batch_size=cfg.batch_size)
    return test_dataset, test_dataloader


class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)
        return img


class RGB2HSV(object):
    """Convert RGB PIL image to ndarray HSV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2hsv(img)
        return img


class RGB2HED(object):
    """Convert RGB PIL image to ndarray HED."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2hed(img)
        return img


class RGB2LUV(object):
    """Convert RGB PIL image to ndarray LUV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2luv(img)
        return img


class RGB2YUV(object):
    """Convert RGB PIL image to ndarray YUV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2yuv(img)
        return img


class RGB2XYZ(object):
    """Convert RGB PIL image to ndarray XYZ."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2xyz(img)
        return img


class RGB2YCbCr(object):
    """Convert RGB PIL image to ndarray YCbCr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ycbcr(img)
        return img


class RGB2YDbDr(object):
    """Convert RGB PIL image to ndarray YDbDr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ydbdr(img)
        return img


class RGB2YPbPr(object):
    """Convert RGB PIL image to ndarray YPbPr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ypbpr(img)
        return img


class RGB2YIQ(object):
    """Convert RGB PIL image to ndarray YIQ."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2yiq(img)
        return img


class RGB2CIERGB(object):
    """Convert RGB PIL image to ndarray RGBCIE."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2rgbcie(img)
        return img
