import torch

import torch.nn as nn

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
