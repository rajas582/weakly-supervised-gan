import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Implementation of the classifier from section 3.2 of https://arxiv.org/pdf/2111.14605.pdf

Author: Jordan Axelrod
"""


class _ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=2)
        self.spectral_conv = nn.utils.parametrizations.spectral_norm(conv)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Interior block of a convolutional block
        :param x: Images
            Shape: `(bsz, in_channels, H, W)
        :return:
        """
        x = self.spectral_conv(x)
        x = self.leaky_relu(x)
        return x


class ConvolutionBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        hidden_channels = output_channels // 2
        self.conv_block1 = _ConvolutionalBlock(input_channels, hidden_channels)
        self.conv_block2 = _ConvolutionalBlock(hidden_channels, output_channels)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Convolutional block
        :param X: images
            Shape: `(bsz, input_channes, H, W)
        :return: torch.Tensor
            Shape: `(bsz, output_channels, H - 2, W - 2)'
        """
        block1 = self.conv_block1(X)
        block2 = self.conv_block2(block1)
        return block2 + self.avg_pool(torch.cat([block1, block1], dim=1))


class SpatialAttention(nn.Module):
    def __init__(self, in_chans):
        super().__init__()
        self.q = nn.Conv2d(in_chans, in_chans, kernel_size=1)
        self.k = nn.Conv2d(in_chans, in_chans, kernel_size=1)
        self.v = nn.Conv2d(in_chans, in_chans, kernel_size=1)
        self.soft_max = nn.Softmax(dim=2)
        self.out = nn.Linear(in_chans, in_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        bsz, in_chans, h, w = x.shape
        q = self.q(x).reshape(bsz, in_chans, h * w)
        k = self.k(x).reshape(bsz, in_chans, h * w)
        v = self.v(x).reshape(bsz, in_chans, h * w)
        qk = torch.matmul(q.permute(0, 2, 1), k) / (h * w) ** .5  # (bsz, h * w, h * w)
        attn = torch.matmul(self.soft_max(qk), v.permute(0, 2, 1))  # (bsz, h * w, in_chans)
        return self.out(attn).permute(0, 2, 1).reshape(bsz, in_chans, h, w)


class PixelAttention(nn.Module):
    def __init__(self, in_chans):
        super().__init__()
        self.q = nn.Conv2d(in_chans, in_chans, kernel_size=1)
        self.k = nn.Conv2d(in_chans, in_chans, kernel_size=1)
        self.v = nn.Conv2d(in_chans, in_chans, kernel_size=1)
        self.soft_max = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        bsz, in_chans, h, w = x.shape
        q = self.q(x).reshape(bsz, in_chans, h * w)
        k = self.k(x).reshape(bsz, in_chans, h * w)
        v = self.v(x).reshape(bsz, in_chans, h * w)
        qk = torch.matmul(q, k.permute(0, 2, 1)) / (h * w) ** .5  # (bsz, h * w, h * w)
        attn = torch.matmul(self.soft_max(qk), v)  # (bsz, h * w, in_chans)
        return attn.permute(0, 2, 1).reshape(bsz, in_chans, h, w)


class ClassifierBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        hidden_chans = out_chans // 2
        conv_block1 = ConvolutionBlock(in_chans, hidden_chans)
        conv_block2 = ConvolutionBlock(hidden_chans, out_chans)
        self.block = nn.Sequential(
            conv_block1,
            conv_block2
        )
        self.spatial_attn = SpatialAttention(out_chans)
        self.pixel_attn = PixelAttention(out_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        out = self.block(x)
        pix = self.pixel_attn(out)
        spat = self.spatial_attn(out)

        return pix + spat


class Classifier(nn.Module):

    def __init__(self, in_chans, out_chans, spec_chans, depth=2):
        super().__init__()
        self.classifier_block1 = ClassifierBlock(in_chans, out_chans)
        if depth == 2:
            self.classifier_block2 = ClassifierBlock(spec_chans, spec_chans)
        self.depth = depth
        conv_out = nn.Conv2d(out_chans, spec_chans, kernel_size=2)
        self.spect_conv = nn.utils.parametrizations.spectral_norm(conv_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.classifier_block1(x)
        if self.depth == 2:
            out = self.classifier_block2(out)
        return self.spect_conv(out)


class WSGANClassifier(nn.Module):
    def __init__(self, in_chans, out_chans, spec_chans, resnet_path: str = 'model_dump/resnet_conv.pt', n_classes=10):
        super().__init__()
        self.resnet_convert = nn.Linear(1, 3)
        self.resnet_conv = torch.load(resnet_path)
        self.resnet_fc = nn.Linear(512, n_classes + 1)
        self.local_class = Classifier(in_chans, out_chans, spec_chans, depth=1)
        self.global_class = Classifier(in_chans, out_chans, spec_chans)
        self.glob_dense = nn.Linear(spec_chans, n_classes + 1)
        self.loc_dense = nn.Linear(spec_chans, n_classes + 1)

    def forward(self, x: torch.Tensor):
        res_in = self.resnet_convert(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        res_out = self.resnet_conv(res_in)
        res_out = self.resnet_fc(res_out.flatten(1))
        loc_out = self.local_class(x)
        loc_avg = F.avg_pool2d(loc_out, loc_out.shape[-1])
        loc_max = F.max_pool2d(loc_out, loc_out.shape[-1])
        loc_out = self.loc_dense((loc_avg + loc_max).flatten(1))
        glob_out = self.global_class(x)
        glob_avg = F.avg_pool2d(glob_out, glob_out.shape[-1])
        glob_max = F.max_pool2d(glob_out, glob_out.shape[-1])
        glob_out = self.glob_dense((glob_max + glob_avg).flatten(1))
        out = loc_out + glob_out + res_out
        out[:, :-1] = F.softmax(out[:, :-1], dim=1)
        out[:, -1] = F.sigmoid(out[:, -1])
        return out


class SimpleClassifier(nn.Module):
    def __init__(self, in_chans, n_classes):
        super().__init__()

        self.simpclass = nn.Sequential(
            nn.Conv2d(in_chans, 8, kernel_size=4, stride=2),  # bsz, 8, 13, 13
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),  # bsz, 16, 5, 5
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # bsz, 32, 1, 1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(64, n_classes + 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.simpclass(x)
        x[:, :-1] = F.softmax(x[:, :-1], dim=-1)
        x[:, -1] = self.sigmoid(x[:, -1])
        return x
