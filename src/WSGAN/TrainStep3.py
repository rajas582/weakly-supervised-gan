import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt

import WSGANClassifier
import WSGANGenerator
import GANLoss
from src.DataProc.make_data import GANLoader
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
            n_classes,
            mod
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
        self.generator = generator.to(device)
        self.generator.require_grad = False
        self.optimizer_gen = optimizer_gen
        self.loss_gen = loss_gen
        self.noise_features = noise_features
        self.n_classes = n_classes
        self.mod = mod.to(device)
        self.mod.require_grad = False

    def test(self):
        test_loss = 0
        self.model.eval()
        self.generator.eval()
        loop = tqdm.tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        for i, (X, y) in loop:
            X, y = X.to(self.device), y.to(self.device)
            noise = torch.randn(X.shape[0], self.noise_features).to(self.device)
            fake_labels = y
            fake_labels_hot = F.one_hot(fake_labels, num_classes=self.n_classes)
            fake_labels_oh = F.pad(fake_labels_hot, (self.noise_features - self.n_classes, 0))
            noise = noise
            fake = self.generator(noise, fake_labels)
            inpt = torch.cat([X, fake], dim=0)
            labels = torch.cat([y, fake_labels])
            permutation = torch.randperm(inpt.shape[0])
            preds = self.model(inpt[permutation])
            labels = labels[permutation]
            gen_loss = self.loss_gen(preds, permutation >= X.shape[0], labels)
            disc_loss = self.loss(preds, permutation >= X.shape[0], labels)
            test_loss += gen_loss.cpu().detach() + disc_loss.cpu().detach()
        return test_loss / len(self.train_loader)

    def train(self, epoch):
        train_loss = 0
        self.model.train()
        self.generator.train()
        loop = tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        torch.autograd.set_detect_anomaly(True)
        gen_loss = torch.Tensor([0])
        disc_loss = torch.Tensor([0])
        for i, (X, y) in loop:
            X, y = X.to(self.device), y.to(self.device)

            noise = torch.randn(X.shape[0], self.noise_features).to(self.device)
            fake_labels = y
            hacky_fix = torch.ones(X.shape[0]).bool().to(self.device)

            noise = noise

            fake = self.generator(noise, fake_labels)[:, :, 3: -3, 3: -3]
            # print(fake.shape)
            if i % 15 == 0:
                plt.imshow(fake[0][0].cpu().detach())
                plt.colorbar()
                plt.show()

            # Train with only True
            inpt = X
            labels = y
            preds = torch.log(self.model.sigmoid(self.model(inpt)))
            labels = labels
            disc_loss = self.loss(preds[: , :-1], labels)
            disc_loss.backward(retain_graph=True)

            # Train with false and accumulate gradients
            pred_fake = self.model.sigmoid(self.model(fake.detach()))
            disc_loss_fake = self.loss_gen(pred_fake[:, -1], 1 - hacky_fix.float()) + self.loss_gen(preds[:, -1], hacky_fix.float())
            disc_loss_fake.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            disc_loss = disc_loss + disc_loss_fake
            # Train generator


            loop.set_postfix({'gen loss': gen_loss.cpu().detach(), 'disc loss': disc_loss.cpu().detach()})
        return train_loss / len(self.train_loader)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path + '/discriminator3.pt')
        torch.save(self.generator.state_dict(), path + '/generator3.pt')

    def model_trainer(self):
        train_loss = []
        test_loss = []
        print(self.device)
        for epoch in range(self.epochs):
            print(epoch)
            train_loss.append(self.train(epoch))
            # test_loss.append(self.test())
        return train_loss, test_loss


if __name__ == '__main__':
    mod = torch.load('model_dump/Contrastive_learning.pt')
    res = mod.path.enc.resnet_conv
    torch.save(res, 'model_dump/resnet.pt')
    noise_features = 512
    n_classes = 10
    eps = 50
    data_maker = GANLoader('../../data')
    train_ldr, _, test_ldr = data_maker.original_loader(batch_size=256)

    classifier = WSGANClassifier.WSGANClassifier(1, 64, 64, 'model_dump/resnet.pt')
    generator = WSGANGenerator.Generator(9, 1, noise_features)

    generator.load_state_dict(torch.load('model_dump/generator2.pt'))
    class_loss = GANLoss.Supervisedloss()
    gen_loss = GANLoss.Unsupervisedloss()
    lr = .001
    gen_optimizer = optim.Adam(generator.parameters(), lr=lr / 100)
    class_optimizer = optim.Adam(classifier.parameters(), lr=lr)
    trainer = GANTrainer(
        classifier,
        generator,
        'cuda' if torch.cuda.is_available() else 'cpu',
        train_ldr,
        test_ldr,
        eps,
        gen_optimizer,
        class_optimizer,
        gen_loss,
        class_loss,
        noise_features,
        n_classes,
        mod
    )

    train_loss, test_loss = trainer.model_trainer()
    trainer.save_model('model_dump')
