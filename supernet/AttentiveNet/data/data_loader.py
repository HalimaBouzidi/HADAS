# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from __future__ import print_function

import torch
import torchvision.transforms.functional as F
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torch.utils.data import Dataset
import math
import sys
import random
from PIL import Image
import pandas as pd

from typing import Any, Callable, cast, Dict, List, Optional, Tuple

from torch.utils.data.distributed import DistributedSampler
import os

from .data_transform import get_data_transform

# Children dataset classes to return index or path names 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class INDEXEDCIFAR10(CIFAR10):       # originally defined in ~/$anaconda3/envs/$env_name$/lib/python3.6/site-packages/torchvision/datasets/cifar.py

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class INDEXEDCIFAR100(CIFAR100):        # originally defined in ~/$anaconda3/envs/$env_name$/lib/python3.6/site-packages/torchvision/datasets/cifar.py

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class NamedImageFolder(ImageFolder):        # originally defined in ~/$anaconda3/envs/$env_name$/lib/python3.6/site-packages/torchvision/datasets/folder.py
    """A child class of ImageFolder to return the pathnames of data samples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(NamedImageFolder, self).__init__(root, loader=loader,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

def build_data_loader(args):
    if args.dataset == 'imagenet':
        return build_default_imagenet_data_loader(args)
    elif args.dataset == 'tiny-imagenet':
        return build_default_tiny_imagenet_data_loader(args)
    elif args.dataset == 'cifar-10':
        return build_default_CIFAR10_data_loader(args)
    elif args.dataset == 'cifar-100':
        return build_default_CIFAR100_data_loader(args)
    else:
        print(args.dataset)
        raise NotImplementedError
    
def build_default_imagenet_data_loader(args):
    traindir = os.path.join(args.dataset_dir, "train")
    valdir = os.path.join(args.dataset_dir, "val")

    #build transforms
    train_transform = get_data_transform(args, is_training=True, augment=args.augment)
    test_transform = get_data_transform(args, is_training=False, augment=args.augment)

    #build datasets
    if not getattr(args, 'data_loader_cross_validation', False):
        train_dataset = datasets.ImageFolder(traindir, train_transform)
        val_dataset = datasets.ImageFolder(valdir, test_transform)
   
    #build data loaders
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last = getattr(args, 'drop_last', True),
        num_workers=args.data_loader_workers_per_gpu,
        pin_memory=True,
    )

    if args.distributed and getattr(args, 'distributed_val', True):
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    eval_batch_size = min(args.batch_size, 16) \
        if not getattr(args, 'eval_only', False) else args.batch_size

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.data_loader_workers_per_gpu,
        drop_last=False,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader, train_sampler

def build_default_tiny_imagenet_data_loader(args):
    traindir = os.path.join(args.dataset_dir, "train")
    valdir = os.path.join(args.dataset_dir, "val")

    val_data = pd.read_csv(f'{valdir}/val_annotations.txt', sep='\t', header=None, names=['File', 'Class', 'X', 'Y', 'H', 'W'])
    val_img_dir = os.path.join(valdir, 'images')
    fp = open(os.path.join(valdir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    for img, folder in val_img_dict.items():
        newpath = (os.path.join(val_img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(val_img_dir, img)):
            os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))

    # Save class names (for corresponding labels) as dict from words.txt file
    class_to_name_dict = dict()
    fp = open(os.path.join(args.dataset_dir, 'words.txt'), 'r')
    data = fp.readlines()
    for line in data:
        words = line.strip('\n').split('\t')
        class_to_name_dict[words[0]] = words[1].split(',')[0]
    fp.close()

    #build transforms
    train_transform = get_data_transform(args, is_training=True, augment=args.augment)
    test_transform = get_data_transform(args, is_training=False, augment=args.augment)

    #build datasets
    if not getattr(args, 'data_loader_cross_validation', False):
        train_dataset = NamedImageFolder(traindir, train_transform)
        val_dataset = NamedImageFolder(val_img_dir, test_transform)

    #build data loaders
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last = getattr(args, 'drop_last', True),
        num_workers=args.data_loader_workers_per_gpu,
        pin_memory=True,
    )

    if args.distributed and getattr(args, 'distributed_val', True):
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    eval_batch_size = min(args.batch_size, 16) \
        if not getattr(args, 'eval_only', False) else args.batch_size

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.data_loader_workers_per_gpu,
        drop_last=False,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader, train_sampler

def build_default_CIFAR10_data_loader(args):

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])    

    non_train_transforms = transforms.Compose([
        transforms.Resize(40),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    train_dataset = INDEXEDCIFAR10(root="/home/mohanadodema/dataset/CIFAR10", train = True, transform=train_transforms, download = True)
    val_dataset = INDEXEDCIFAR10(root="/home/mohanadodema/dataset/CIFAR10", train = False, transform=non_train_transforms, download = True)

    train_sampler, val_sampler = None, None 
    eval_batch_size = min(args.batch_size, 16) \
        if not getattr(args, 'eval_only', False) else args.batch_size  

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False, #(train_sampler is None), 
        sampler=train_sampler,
        drop_last = getattr(args, 'drop_last', True),
        num_workers=args.data_loader_workers_per_gpu,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.data_loader_workers_per_gpu,
        drop_last=False,
        pin_memory=True,
        sampler=val_sampler,
        )

    return train_loader, val_loader, train_sampler

def build_default_CIFAR100_data_loader(args):
    #build transforms
    train_transform = get_data_transform(args, is_training=True, augment=args.augment)
    test_transform = get_data_transform(args, is_training=False, augment=args.augment)

    #build datasets    
    train_dataset = INDEXEDCIFAR100(root=args.dataset_dir, train=True,
                                        download=True, transform=train_transform)
    val_dataset = INDEXEDCIFAR100(root=args.dataset_dir, train=False,
                                        download=True, transform=test_transform)

    #build data loaders
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last = getattr(args, 'drop_last', True),
        num_workers=args.data_loader_workers_per_gpu,
        pin_memory=True,
    )

    if args.distributed and getattr(args, 'distributed_val', True):
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    eval_batch_size = min(args.batch_size, 16) \
        if not getattr(args, 'eval_only', False) else args.batch_size

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.data_loader_workers_per_gpu,
        drop_last=False,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader, train_sampler

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

def build_val_data_loader(args):
    if args.dataset == 'imagenet':
        return build_default_imagenet_val_data_loader(args)
    else:
        raise NotImplementedError
    
def build_default_imagenet_val_data_loader(args):
    valdir = os.path.join(args.dataset_dir, "val")

    #build transforms
    test_transform = get_data_transform(args, is_training=False, augment=args.augment)

    #build datasets
    if not getattr(args, 'data_loader_cross_validation', False):
        val_dataset = datasets.ImageFolder(valdir, test_transform)

    if args.distributed and getattr(args, 'distributed_val', True):
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    eval_batch_size = min(args.batch_size, 16) \
        if not getattr(args, 'eval_only', False) else args.batch_size

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.data_loader_workers_per_gpu,
        drop_last=False,
        pin_memory=True,
        sampler=val_sampler,
    )

    return val_loader