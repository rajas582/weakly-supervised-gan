import torch
import torch.nn as nn
from torchvision.models import resnet50

from utils import AugmentationPipeline

"""
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

if __name__ == '__main__':
    enc = Encoder()
    dec = Decoder()
    input = torch.randn(2, 3, 32, 32)
    h = enc(input)
    Z = dec(h)
    print(Z.shape)
