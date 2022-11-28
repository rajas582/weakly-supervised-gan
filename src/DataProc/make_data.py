import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


# import matplotlib.pyplot as plt
# import numpy as np


class GANLoader:
    def __init__(self, path, download=True):
        """
        Give path for where to download the data
        """

        self.path = path
        self.download = download
        self.transform_original = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((0.5,), (0.5,))])

        self.transform_aug = transform1 = transforms.Compose([
            transforms.RandomResizedCrop(28),
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        self.trainset = datasets.FashionMNIST(self.path,
                                              download=self.download,
                                              train=True,
                                              transform=self.transform_original)
        self.testset = datasets.FashionMNIST(self.path,
                                             download=self.download,
                                             train=False,
                                             transform=self.transform_original)

    def train_val_split(self):
        train_ind, val_ind, _, _ = train_test_split(
            range(len(self.trainset)),
            self.trainset.targets,
            stratify=self.trainset.targets,
            test_size=0.2)

        return train_ind, val_ind

    def original_set(self, val=True):
        ## TODO - if val = False, return without Val set
        '''
        Returns original data
        '''
        train_ind, val_ind = self.train_val_split()

        train_data = Subset(self.trainset, train_ind)
        val_data = Subset(self.trainset, val_ind)
        test_data = self.testset

        return train_data, val_data, test_data

    def original_loader(self, val=True, batch_size=64):
        '''
        Returns original data loader
        '''
        train_data, val_data, test_data = self.original_set()

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                   shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                                 shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        return train_loader, val_loader, test_loader

    def augmented_set(self, val=True):
        ## TODO - if val = False, return without Val set
        '''
        Returns augmented data
        '''
        aug_set = datasets.FashionMNIST(self.path,
                                        download=self.download,
                                        train=True,
                                        transform=self.transform_aug)
        aug_test = datasets.FashionMNIST(self.path,
                                        download=self.download,
                                        train=False,
                                        transform=self.transform_aug)
        train_ind, val_ind = self.train_val_split()

        train_data = Subset(self.trainset, train_ind)
        aug_data = Subset(aug_set, train_ind)
        aug_val_data = Subset(aug_set, val_ind)
        val_data = Subset(self.trainset, val_ind)
        test_data = self.testset

        aug_train = AugmentedDataset(train_data, aug_data)
        aug_val = AugmentedDataset(val_data, aug_val_data)
        aug_test = AugmentedDataset(test_data, aug_test)
        return aug_train, aug_val, aug_test

    def augmented_loader(self, val=True, batch_size=64):
        '''
        Returns augmented data loader
        '''
        train_data, val_data, test_data = self.augmented_set()

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                   num_workers=3)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                                 shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        return train_loader, val_loader, test_loader


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, augmented_set_1, augmented_set_2):
        self.augmented_set_1 = augmented_set_1
        self.augmented_set_2 = augmented_set_2

    def __len__(self):
        return len(self.augmented_set_1)

    def __getitem__(self, item):
        return self.augmented_set_1[item], self.augmented_set_2[item]
