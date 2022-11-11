import torch
from torch.utils.data import DataLoader
from torch import optim

import ContrastiveLearning
import CLLoss
from src.Trainer import Trainer
from src.utils import wsgan_loaders


class ContrastiveLearningTrainer(Trainer):

    def train(self):
        train_loss = 0
        self.model.train()

        # X are the fashionMNIST images as PIL
        for idx, (X, y) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            gen_im, (z1, z2) = self.model(X, self.device)
            y = torch.Tensor(X, device=self.device)
            l = self.loss(z1, z2, gen_im, y)
            l.backward()
            self.optimizer.step()
            train_loss += l.item()
        return train_loss * 100 / len(self.train_loader)

    def test(self):
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for X, _ in self.test_loader:
                gen_im, (z1, z2) = self.model(X, self.device)
                y = torch.Tensor(X, device=self.device)
                l = self.loss(z1, z2, gen_im, y)
                test_loss += l.item()
        return test_loss * 100 / len(self.test_loader)

    def model_trainer(self):
        train_loss = []
        test_loss = []
        for epoch in range(1, self.epochs + 1):
            train_loss.append(self.train())
            test_loss.append(self.test())
        return train_loss, test_loss


if __name__ == '__main__':
    clnet = ContrastiveLearning.CLNet(1)
    clloss = CLLoss.CLLoss()
    adam = optim.Adam(clnet.parameters())
    eps = 15
    train_ldr, test_ldr = wsgan_loaders()
    clnettrainer = ContrastiveLearningTrainer(
        clnet,
        'cuda' if torch.cuda.is_available() else 'cpu',
        train_ldr,
        test_ldr,
        eps,
        adam,
        clloss
    )
    train_loss, test_loss = clnettrainer.model_trainer()
    clnettrainer.save_model('model_dump/Contrastive_learning.pt')
