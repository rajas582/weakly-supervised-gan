from abc import ABC, abstractmethod

import torch

class Trainer(ABC):
    def __init__(
            self,
            model,
            device,
            train_loader,
            test_loader,
            epochs,
            optimizer,
            loss
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss

    @abstractmethod
    def train(self):
        pass
    @abstractmethod
    def test(self):
        pass
    @abstractmethod
    def model_trainer(self):
        pass

    def save_model(self, path):
        torch.save(self.model, path)
