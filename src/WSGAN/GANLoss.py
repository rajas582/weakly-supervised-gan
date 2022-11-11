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
        one_hot_f = F.one_hot(fakey, num_classes=self.n_classes + 1)
        one_hot_r = F.one_hot(realy, num_classes=self.n_classes + 1)
        one_hot_r[:, -1] = 1
        return (fake, one_hot_f), (real, one_hot_r)


class DiscriminatorLoss(GANLoss):
    def __init__(self, n_classes):
        super(DiscriminatorLoss, self).__init__(n_classes)

    def forward(self, X: torch.Tensor, gen_list: torch.BoolTensor, labels: torch.Tensor):
        (fake, one_hot_f), (real, one_hot_r) = self._generate_fake_real(X, gen_list, labels)
        loss = 1 / fake.shape[0] * F.mse_loss(fake, one_hot_f) + 1 / real.shape[0] * F.mse_loss(real, one_hot_r)

        return loss


class GeneratorLoss(GANLoss):
    def __init__(self, n_classes):
        super(GeneratorLoss, self).__init__(n_classes)

    def forward(self, X: torch.Tensor, gen_list: torch.BoolTensor, labels: torch.Tensor):
        (fake, one_hot_f), _ = self._generate_fake_real(X, gen_list, labels)
        loss = 1 / fake.shape[0] * F.mse_loss(fake, one_hot_f)
        return loss
