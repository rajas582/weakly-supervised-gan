import torch

import torch.nn as nn


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, depth, out_channels, noise_size):
        super(Generator, self).__init__()
        self.gen = nn.Sequential()
        for i in range(depth - 1):
            self.gen.append(
                DeconvBlock(
                    noise_size,
                    noise_size // 2 ** i
                )
            )
        self.gen.append(DeconvBlock(noise_size // depth - 2, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, -1, 1, 1)
        return self.gen(x)
