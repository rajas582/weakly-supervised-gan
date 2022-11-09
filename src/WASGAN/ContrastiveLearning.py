import torch
import torch.nn as nn
from torchvision.models import resnet50

from src.utils import AugmentationPipeline

"""
Implementation of the first network in https://arxiv.org/pdf/2111.14605.pdf. It is a contrastive learning 
network. The resnet feature extracting layers trained in this file will be used in the full classifier

Author: Jordan Axelrod
Date: 11.9.2022
"""


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        resnet = resnet50(weights='DEFAULT')
        self.resnet_conv = nn.Sequential(*(list(resnet.children())[:-1]))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.resnet_conv(X)


class DeconvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        deconv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=(4, 4))
        batchnorm = nn.BatchNorm2d(output_channels)
        relu = nn.ReLU()
        self.block = nn.Sequential(
            deconv,
            batchnorm,
            relu
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.block(X)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = nn.Sequential()
        for i in range(3):
            self.blocks.append(
                DeconvBlock(
                    int(2048 / 2 ** i),
                    int(1024 / 2 ** i)
                )
            )
        self.blocks.append(DeconvBlock(int(1024 / 2 ** 2), 1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.blocks(X)


class Projector(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.projection = nn.Linear(input_channels, output_channels)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Projects the hidden vector to different dimension
        :param X:
        :return:
        """
        X = X.flatten(1)
        return self.projection(X)


class Path(nn.Module):
    def __init__(self, output_channels):
        self.enc = Encoder()
        self.proj = Projector(2048, output_channels)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Implements one of the paths in figure 3
        :param X: tensor of images
            Shape: `(bsz, channels, H, W)
        :return:
        """
        h = self.enc(X)
        z = self.proj(h)
        return h, z


class CLNet(nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        self.path1 = Path(output_channels)
        self.path2 = Path(output_channels)
        self.dec = Decoder()

    def forward(self, X: list):
        """
        Implements the net found in figure 3
        :param X: tensor of images
            Shape: `(2 * bsz, channels, H, W)
        :return:
        """
        X = AugmentationPipeline(X)
        x1 = X[::2]
        x2 = X[1::2]
        h1, z1 = self.path1(x1)
        h2, z2 = self.path2(x2)

        gen_image = self.dec(h1 + h2)

        return gen_image, (z1, z2)


class infoNCELoss(nn.Module):
    def __init__(self, T):
        """
        Implements the infoNCE loss in the paper
        :param T: hyperparameter
        """
        super().__init__()
        self.exp_s = None
        self.T = T
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
        s = self.cosine_similarity(z, z).flatten()[1:].view(bsz - 1, bsz + 1)[:, :-1].reshape(bsz,
                                                                                              bsz - 1)  # cosine simliarity of different vectors
        self.exp_s = torch.exp(s / self.T)
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


if __name__ == '__main__':
    enc = Encoder()
    dec = Decoder()
    input = torch.randn(2, 3, 32, 32)
    h = enc(input)
    Z = dec(h)
    print(Z.shape)
