import torch
import torch.nn as nn


class infoNCELoss(nn.Module):
    def __init__(self):
        """
        Implements the infoNCE loss in the paper
        :param T: hyperparameter
        """
        super().__init__()
        self.exp_s = None

        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def loss(self, i: int, j: int) -> torch.Tensor:
        """
        Computes the loss found in equation 2 of the paper
        :param i: the first index of an image
        :param j:  the second index of an image
        :return: the loss lij
        """

        loss = -torch.log(self.exp_s[i, j] / sum(self.exp_s[i]))
        return loss

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Calculate the infoLoss
        :param z1: torch.Tensor
            Shape: `(bsz, end_dim)'
        :param z2: torch.Tensor
            Shape: `(bsz, end_dim)'
        :return: loss
        """
        bsz, enc = z1.shape

        z = torch.cat([z1, z2], dim=0)
        s = self.cosine_similarity(z, z)
        s = s.flatten()[1:].view(bsz - 1, bsz + 1)[:, :-1].reshape(bsz, bsz - 1)  # cosine simliarity

        self.exp_s = torch.exp(s / torch.sqrt(bsz))
        L = 0
        for m in range(bsz):
            L += self.loss(m, bsz + m) + self.loss(bsz + m, m)
        return 1 / 2 * bsz * L


class CLLoss(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.info_loss = infoNCELoss(T)
        self.mse_loss = nn.MSELoss()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss shown in equation 3 of the paper
        :param z1: The state vectors produced by the projector
            Shape: `(bsz, enc_dim)'
        :param z2: The state vectors produced by the projector
            Shape: `(bsz, enc_dim)'
        :param X: The predicted images from the decoder
            Shape: `(bsz, in_channels, H, W)'
        :param Y: The original images
            Shape: `(bsz, in_channels, H, W)'
        :return: the loss of the batch
        """
        bsz, _, _, _ = X
        return self.info_loss(z1, z2) + 1 / bsz * self.mse_loss(X, Y)
