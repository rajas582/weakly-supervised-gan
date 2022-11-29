import torch
import torch.nn as nn
from torchvision.models import resnet18


"""
Implementation of the first network in https://arxiv.org/pdf/2111.14605.pdf. It is a contrastive learning 
network. The resnet feature extracting layers trained in this file will be used in the full classifier

Author: Jordan Axelrod
Date: 11.9.2022
"""


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        resnet = resnet18(weights='DEFAULT')
        self.resnet_conv = nn.Sequential(*(list(resnet.children())[:-1]))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.resnet_conv(X.expand(-1, 3, -1, -1))


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
        for i in range(9):
            self.blocks.append(
                DeconvBlock(
                    int(512 / 2 ** i),
                    int(256 / 2 ** i)
                )
            )

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
        super(Path, self).__init__()
        self.enc = Encoder()
        self.proj = Projector(512, output_channels)

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
        self.path = Path(output_channels)
        self.dec = Decoder()

    def forward(self, X: torch.Tensor, device: str):
        """
        Implements the net found in figure 3
        :param X: tensor of images
            Shape: `(2 * bsz, channels, H, W)
        :return:
        """
        x1 = X[::2]
        x2 = X[1::2]
        h1, z1 = self.path(x1)
        h2, z2 = self.path(x2)

        gen_image = self.dec(h1 + h2)

        return gen_image, (z1, z2)


if __name__ == '__main__':
    enc = Encoder()
    dec = Decoder()
    input = torch.randn(2, 3, 32, 32)
    h = enc(input)
    Z = dec(h)
    print(Z.shape)
