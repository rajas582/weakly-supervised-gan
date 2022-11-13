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
            transforms.RandomCrop(28, pad_if_needed=True),
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
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
        train_data, val_data, test_data = self.original_loader()

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

        train_ind, val_ind = self.train_val_split()

        train_data = Subset(self.trainset, train_ind)
        aug_data = Subset(aug_set, train_ind)
        val_data = Subset(self.trainset, val_ind)
        test_data = self.testset

        aug_train_data = torch.utils.data.ConcatDataset([[train_data[i], aug_data[i]] for i in range(len(train_data))])

        return aug_train_data, val_data, test_data

    def augmented_loader(self, val=True, batch_size=64):
        '''
        Returns augmented data loader
        '''
        train_data, val_data, test_data = self.augmented_set()

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                   num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                                 shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        return train_loader, val_loader, test_loader

