import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Trainer import Trainer


class GANTrainer(Trainer):
    def __init__(
            self,
            discriminator,
            generator,
            device,
            train_loader,
            test_loader,
            epochs,
            optimizer_gen,
            optimizer_disc,
            loss_gen,
            loss_disc,
            noise_features,
            n_classes
    ):
        super(GANTrainer, self).__init__(
            discriminator,
            device,
            train_loader,
            test_loader,
            epochs,
            optimizer_disc,
            loss_disc
        )
        self.generator = generator
        self.optimizer_gen = optimizer_gen
        self.loss_gen = loss_gen
        self.noise_features = noise_features
        self.n_classes = n_classes

    def test(self):
        pass

    def train(self):
        train_loss = 0
        for i, (X, y) in enumerate(self.train_loader):
            noise = torch.randn(X.shape[0], self.noise_features)
            fake_labels = F.one_hot(torch.randint(high=self.n_classes, size=(X.shape[0],)),num_classes=self.n_classes)
            fake_labels_oh = F.pad(fake_labels, (self.noise_features - self.n_classes, 0))
            noise = noise + fake_labels_oh
            fake = self.generator(noise)
            inpt = torch.cat([X, fake], dim=0)
            labels = torch.cat([y, fake_labels])
            permutation = torch.randperm(inpt.shape[0])
            preds = self.model(inpt[permutation])
            labels = labels[permutation]
            gen_loss = self.loss_gen(preds, labels >= X.shape[0], labels)
            disc_loss = self.loss(preds, labels >= X.shape[0], labels)

    def model_trainer(self):
        pass

