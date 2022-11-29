import torch
from torch.utils.data import DataLoader
from torch import optim

import os
import matplotlib.pyplot as plt
import tqdm
import ContrastiveLearning
import CLLoss
from src.Trainer import Trainer
from src.DataProc.make_data import GANLoader
print(os.getcwd(), 'hi')
class ContrastiveLearningTrainer(Trainer):

    def train(self):
        train_loss = 0
        self.model.train()

        # X are the fashionMNIST images as PIL

        loop = tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for idx, (X, y) in loop:
            self.optimizer.zero_grad()
            X1 = X[0]
            X2 = y[0]
            x = torch.zeros(X1.shape[0] * 2, *X1.shape[1:])
            x[::2] = X1
            x[1::2] = X2
            X, y = x.to(self.device), x.to(self.device)
            gen_im, (z1, z2) = self.model(X, self.device)
            l = self.loss(z1, z2, gen_im, y)
            l.backward()
            self.optimizer.step()
            loop.set_postfix({'loss': l.detach()})
            train_loss += l.item()
        plt.imshow(gen_im.cpu().detach()[0][0])
        plt.show()

        plt.imshow(X[0][0].cpu())
        plt.show()
        plt.imshow(X[1][0].cpu())
        plt.show()
        return train_loss * 100 / len(self.train_loader)

    def test(self):
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for X, y in self.test_loader:
                X1 = X[0]
                X2 = y[0]
                x = torch.zeros(X1.shape[0] * 2, *X1.shape[1:])
                x[::2] = X1
                x[1::2] = X2
                X, y = x.to(self.device), x.to(self.device)
                gen_im, (z1, z2) = self.model(X, self.device)
                l = self.loss(z1, z2, gen_im, y)
                test_loss += l.item()
        return test_loss * 100 / len(self.test_loader)

    def model_trainer(self):
        train_loss = []
        test_loss = []
        print(self.device)
        for epoch in range(1, self.epochs + 1):
            train_loss.append(self.train())
            test_loss.append(self.test())
        return train_loss, test_loss


if __name__ == '__main__':
    clnet = ContrastiveLearning.CLNet(512)
    clloss = CLLoss.CLLoss()
    adam = optim.Adam(clnet.parameters())
    eps = 10
    data_maker = GANLoader('../../data')
    train_ldr, _, test_ldr = data_maker.augmented_loader(batch_size=100)
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
