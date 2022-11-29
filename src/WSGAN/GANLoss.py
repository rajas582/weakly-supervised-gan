import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self, n_classes):
        super(GANLoss, self).__init__()
        self.n_classes = n_classes

    def _generate_fake_real(
            self, X: torch.Tensor,
            gen_list: torch.BoolTensor,
            labels: torch.Tensor
    ):
        fake, real = X[gen_list], X[~gen_list]
        fakey, realy = labels[gen_list], labels[~gen_list]
        one_hot_f = F.one_hot(fakey, num_classes=self.n_classes + 1).float()
        one_hot_r = F.one_hot(realy, num_classes=self.n_classes + 1).float()
        one_hot_r[:, -1] = 1
        return (fake, one_hot_f), (real, one_hot_r)


class DiscriminatorLoss(GANLoss):
    def __init__(self, n_classes):
        super(DiscriminatorLoss, self).__init__(n_classes)

    def forward(self, X: torch.Tensor, gen_list: torch.BoolTensor, labels: torch.Tensor):
        # (fake, one_hot_f), (real, one_hot_r) = self._generate_fake_real(X, gen_list, labels)
        # fake_loss = F.mse_loss(fake[:, -1:], one_hot_f[:, -1:]) if fake.shape[0] > 0 else 0
        # real_loss = F.mse_loss(real[:, -1:], one_hot_r[:, -1:]) if real.shape[0] > 0 else 0
        loss = F.binary_cross_entropy(X[:, -1], 1 - gen_list.float())
        return loss / 2


class GeneratorLoss(GANLoss):
    def __init__(self, n_classes):
        super(GeneratorLoss, self).__init__(n_classes)

    def forward(self, X: torch.Tensor, gen_list: torch.BoolTensor, labels: torch.Tensor):
        # (fake, one_hot_f), _ = self._generate_fake_real(X, gen_list, labels)
        # one_hot_f[:, -1] = 1
        # loss = F.mse_loss(fake[:, -1:], one_hot_f[:, -1:])
        loss = F.binary_cross_entropy(X[:, -1], gen_list.float())
        return loss

class Supervisedloss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        loss = F.nll_loss(x, y)
        return loss

class Unsupervisedloss(nn.Module):
    def __init__(self):
        super(Unsupervisedloss, self).__init__()
    def forward(self, x, y):
        return F.mse_loss(x, y)