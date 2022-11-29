import torch

import torch.nn as nn
import torch.nn.functional as F

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation, out_padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=(4, 4),
                stride=stride,
                dilation=dilation,
                output_padding=out_padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.block(x)
        return x


# 1 -> 2 -> 4 -> 8 -> 16 -> 32
# 28 -> 13  -> 6 -> 4 -> 1
# s 2,  2,   1,   1
class Generator(nn.Module):
    def __init__(self, depth, out_channels, noise_size):
        super(Generator, self).__init__()
        self.gen = nn.Sequential()
        strides = [2, 2, 2, 1, 1]
        dilation = [1, 4, 2, 1, 1]
        out_padding = [0, 0, 0, 0, 0]
        depth = len(strides)
        for i in range(depth - 1):
            self.gen.append(
                DeconvBlock(
                    noise_size // 2 ** i,
                    noise_size // 2 ** (i + 1),
                    stride=strides[i],
                    dilation=dilation[i],
                    out_padding=out_padding[i]
                )
            )
        self.gen.append(DeconvBlock(noise_size // 2 ** (i + 1), noise_size // 2 ** (i + 2), stride=strides[-1], dilation=dilation[-1], out_padding=out_padding[-1]))
        self.noise_size = noise_size
        self.label_proj = nn.Linear(1, noise_size)
        self.gen.append(nn.Conv2d(noise_size // 2 ** (i + 2), out_channels, kernel_size=3, stride=2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, fake_labels: torch.Tensor) -> torch.Tensor:
        fake_labels = self.label_proj(fake_labels.unsqueeze(1).float() / 9)
        x = x + fake_labels
        x = x.reshape(-1, self.noise_size, 1, 1)
        x = self.gen(x)

        x = self.sigmoid(x)
        x = F.pad(x, (2, 2, 2, 2))
        return x
