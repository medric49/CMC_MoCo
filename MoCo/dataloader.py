import os
import sys
from PIL import Image
from PIL import ImageFilter
import random

import torchvision.datasets as Datasets
import torchvision.transforms as Transforms
from torchvision.datasets import VisionDataset, ImageFolder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F


import numpy as np
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, verify_str_arg

from colorspace import RGB2Lab, RGB2YCbCr, RGB2YDbDr


# The following class STL10_train is the method derived from torch.datasets and modified in such a way that it outputs two different augmentations of the image instead of one. 
# The modification is made in __getitem__ function

class STL10_train(VisionDataset):
    """`STL10 <https://cs.stanford.edu/~acoates/stl10/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``stl10_binary`` exists.
        split (string): One of {'train', 'test', 'unlabeled', 'train+unlabeled'}.
            Accordingly dataset is selected.
        folds (int, optional): One of {0-9} or None.
            For training, loads one of the 10 pre-defined folds of 1k samples for the
            standard evaluation procedure. If no value is passed, loads the 5k samples.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'stl10_binary'
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    tgz_md5 = '91f7769df0f17e558f3565bffb0c7dfb'
    class_names_file = 'class_names.txt'
    folds_list_file = 'fold_indices.txt'
    train_list = [
        ['train_X.bin', '918c2871b30a85fa023e0c44e0bee87f'],
        ['train_y.bin', '5a34089d4802c674881badbb80307741'],
        ['unlabeled_X.bin', '5242ba1fed5e4be9e1e742405eb56ca4']
    ]

    test_list = [
        ['test_X.bin', '7f263ba9f9e0b06b93213547f721ac82'],
        ['test_y.bin', '36f9794fa4beb8a2c72628de14fa638e']
    ]
    splits = ('train', 'train+unlabeled', 'unlabeled', 'test')

    def __init__(
            self,
            root: str,
            split: str = "train",
            folds: Optional[int] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(STL10_train, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.split = verify_str_arg(split, "split", self.splits)
        self.folds = self._verify_folds(folds)

        if download:
            self.download()
        elif not self._check_integrity():
            raise RuntimeError(
                'Dataset not found or corrupted. '
                'You can use download=True to download it')

        # now load the picked numpy arrays
        self.labels: Optional[np.ndarray]
        if self.split == 'train':
            self.data, self.labels = self.__loadfile(
                self.train_list[0][0], self.train_list[1][0])
            self.__load_folds(folds)

        elif self.split == 'train+unlabeled':
            self.data, self.labels = self.__loadfile(
                self.train_list[0][0], self.train_list[1][0])
            self.__load_folds(folds)
            unlabeled_data, _ = self.__loadfile(self.train_list[2][0])
            self.data = np.concatenate((self.data, unlabeled_data))
            self.labels = np.concatenate(
                (self.labels, np.asarray([-1] * unlabeled_data.shape[0])))

        elif self.split == 'unlabeled':
            self.data, _ = self.__loadfile(self.train_list[2][0])
            self.labels = np.asarray([-1] * self.data.shape[0])
        else:  # self.split == 'test':
            self.data, self.labels = self.__loadfile(
                self.test_list[0][0], self.test_list[1][0])

        class_file = os.path.join(
            self.root, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()

    def _verify_folds(self, folds: Optional[int]) -> Optional[int]:
        if folds is None:
            return folds
        elif isinstance(folds, int):
            if folds in range(10):
                return folds
            msg = ("Value for argument folds should be in the range [0, 10), "
                   "but got {}.")
            raise ValueError(msg.format(folds))
        else:
            msg = "Expected type None or int for argument folds, but got type {}."
            raise ValueError(msg.format(type(folds)))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (augmented_image_1, augmented_image_2, target) where target is index of the target class.
        """
        target: Optional[int]
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            aug_img_1 = self.transform(img)
            aug_img_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return aug_img_1, aug_img_2, target


    def __len__(self) -> int:
        return self.data.shape[0]

    def __loadfile(self, data_file: str, labels_file: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        labels = None
        if labels_file:
            path_to_labels = os.path.join(
                self.root, self.base_folder, labels_file)
            with open(path_to_labels, 'rb') as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = os.path.join(self.root, self.base_folder, data_file)
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
        self._check_integrity()

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def __load_folds(self, folds: Optional[int]) -> None:
        # loads one of the folds if specified
        if folds is None:
            return
        path_to_folds = os.path.join(
            self.root, self.base_folder, self.folds_list_file)
        with open(path_to_folds, 'r') as f:
            str_idx = f.read().splitlines()[folds]
            list_idx = np.fromstring(str_idx, dtype=np.int64, sep=' ')
            self.data = self.data[list_idx, :, :, :]
            if self.labels is not None:
                self.labels = self.labels[list_idx]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def data_loader(dataset_root='/datasets/STL-10', resize=84, crop=64, 
                batch_size=64, num_workers=4, type='train', view=None):    
    '''
    Data loader for MoCo. It is written assuming that 'ImageNet' dataset is used to train an encoder in 
    self-supervised manner, and 'STL-10' dataset is used to evaluate the encoder.

    Args:
        - dataset_root (str): Root directory consisting of subdirectories for each class. Each subdirectory 
                              contains images corresponding that specific class. Note that the class label
                              is not used in training, but this constraint is caused by the structure of 
                              Imagenet dataset.
        - resize (int) : Images are resized with this value.
        - crop (int) : Images are cropped with this value. This is a final size of image transformation.
        - batch_size (int) : Batch size
        - num_workers (int) : Number of workers for data loader
        - type (str) : Type of data loader.
                       1) encoder_train : data loader for training an encoder in self-supervised manner.
                       2) classifier_train : data loader for training a linear classifier to evaluate 
                                             the encoder.
                       3) classifier_test : data loader for evaluating the linear classifier.
                       
    Returns:
        - dloader : Data loader
        - dlen : Total number of data
    '''

    transform_list = []
    
    if type == 'encoder_train':
        
        if view is None:
            
            transform_list += [Transforms.RandomResizedCrop(size=crop),
                           Transforms.ColorJitter(0.1, 0.1, 0.1),
                           Transforms.RandomHorizontalFlip(),
                           Transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                           Transforms.RandomGrayscale()]
            
            transform_list += [Transforms.ToTensor(),
                       Transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        
        elif view == 'Lab':
            
            mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
            std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
            color_transfer = RGB2YDbDr()
            
            transform_list += [Transforms.RandomResizedCrop(size=crop),
                           Transforms.ColorJitter(0.1, 0.1, 0.1),
                           Transforms.RandomHorizontalFlip(),
                           Transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                           color_transfer,
                           Transforms.RandomGrayscale()]
            
            transform_list += [Transforms.ToTensor(),
                       Transforms.Normalize(mean=mean, std=std)]
        
        elif view == 'YCbCr':
            
            mean = [116.151, 121.080, 132.342]
            std = [109.500, 111.855, 111.964]
            color_transfer = RGB2YCbCr()
            
            transform_list += [Transforms.RandomResizedCrop(size=crop),
                           Transforms.RandomHorizontalFlip(),
                           color_transfer]
            
            transform_list += [Transforms.ToTensor(),
                       Transforms.Normalize(mean=mean, std=std)]
            
                                            
    elif type == 'classifier_train':
    
        mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
        std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
        color_transfer = RGB2YDbDr()
        
        transform_list += [Transforms.Resize(size=resize),
                           Transforms.RandomCrop(size=crop),
                           Transforms.RandomHorizontalFlip(),
                           color_transfer]
        
        transform_list += [Transforms.ToTensor(),
                       Transforms.Normalize(mean=mean,
                                            std=std)]
        
    elif type == 'classifier_test':
    
        mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
        std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
        color_transfer = RGB2YDbDr()
        
        transform_list += [Transforms.Resize(size=resize),
                           Transforms.CenterCrop(size=crop),
                           color_transfer]
        
        transform_list += [Transforms.ToTensor(),
                       Transforms.Normalize(mean=mean,
                                            std=std)]

    

    transform = Transforms.Compose(transform_list)
    
    if type == 'encoder_train':
        split = 'train+unlabeled' # 'train+unlabeled'
        #split = 'train'
        dset = STL10_train(root=dataset_root, split=split, transform=transform, download=True)
    elif type == 'classifier_train' or type == 'classifier_test':
        split = type.split('_')[-1] # 'train' or 'test'
        train_bool = True if split == 'train' else False
        
        dset = Datasets.STL10(root=dataset_root, split=split, transform=transform, download=True)
        #dset = Datasets.CIFAR100(root=dataset_root, train=train_bool, transform=transform, download=True)
        #dset = Datasets.CIFAR10(root=dataset_root, train=train_bool, transform=transform, download=True)
            
    dlen = len(dset)
    dloader = DataLoader(dataset=dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dloader, dlen