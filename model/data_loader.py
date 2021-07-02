"""
   CIFAR-10 data normalization reference:
   https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
"""
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
from torch.utils.data.dataset import random_split
from torch.utils.data.sampler import SubsetRandomSampler


def fetch_dataloader(types, params):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """
    if types == 'train':
        if params.augmentation == "yes":
            train_transformer = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        # data augmentation can be turned off
        else:
            train_transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        trainset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=True,
            download=True, transform=train_transformer) #50000

        if ('train_size' in params.dict) and ('collection_size' in params.dict) and ('day' in params.dict):
            if "_distill" in params.model_version:
                base_dataset, new_trainset = __edge_data_split(trainset, params)
                print(f"base_dataset: {len(base_dataset)}")
                print(f"day_dataset: {len(new_trainset)}")
            else:
                new_trainset = __cloud_data_split(trainset, params)
                print(f"train_dataset: {len(new_trainset)}")
        elif 'train_size' in params.dict:
            new_trainset, _ = __data_split(trainset, params.train_size)
            print(f"train_dataset: {len(new_trainset)}")
        else:
            new_trainset = trainset

        trainloader = torch.utils.data.DataLoader(new_trainset, batch_size=params.batch_size,
        shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda)
        dl = trainloader
    else:
        dev_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        devset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=False,
            download=True, transform=dev_transformer) #10000
        devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)
        dl = devloader

    return dl


def fetch_subset_dataloader(types, params):
    """
    Use only a subset of dataset for KD training, depending on params.subset_percent
    """

    # using random crops and horizontal flip for train set
    if params.augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    trainset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=True,
        download=True, transform=train_transformer)

    devset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=False,
        download=True, transform=dev_transformer)

    trainset_size = len(trainset)
    indices = list(range(trainset_size))
    split = int(np.floor(params.subset_percent * trainset_size))
    np.random.seed(230)
    np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:split])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        sampler=train_sampler, num_workers=params.num_workers, pin_memory=params.cuda)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl

def __data_split(dataset, main_size):
    torch.manual_seed(0)
    return tuple(random_split(dataset, [main_size, len(dataset) - main_size]))

def __cloud_data_split(dataset, params):
    base_dataset, remain_dataset = __data_split(dataset, params.train_size)
    collections = [base_dataset]
    for i in range(1, params.day):
        if i == 1 and 'first_collection_size' in params.dict:
            day_collection, remain_dataset = __data_split(remain_dataset, params.first_collection_size)
            continue
        day_collection, remain_dataset = __data_split(remain_dataset, params.collection_size)
        collections.append(day_collection)
    return ConcatDataset(collections)

def __edge_data_split(dataset, params):
    base_dataset, remain_dataset = __data_split(dataset, params.train_size)
    collections = []
    for i in range(1, params.day+1):
        if i == 1 and 'first_collection_size' in params.dict:
            day_collection, remain_dataset = __data_split(remain_dataset, params.first_collection_size)
            collections.append(day_collection)
            continue
        day_collection, remain_dataset = __data_split(remain_dataset, params.collection_size)
        collections.append(day_collection)
    return base_dataset, ConcatDataset(collections)
