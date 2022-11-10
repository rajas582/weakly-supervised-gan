
import torch
import torch.nn as nn

class FID(nn.Module):
    """
    Implements the Frechet Inception Distance https://arxiv.org/pdf/1706.08500.pdf
    """
    def __init__(self):
        super(FID, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """

        :param x:
        :param y:
        :return:
        """
        x = x.flatten(1)
        y = y.flatten(1)
        x_mean = x.mean(0)
        y_mean = y.mean(0)
        x_cov = x.cov()
        y_cov = y.cov()
        out = torch.linalg.norm(x_mean - y_mean) + torch.trace(
            x_cov + y_cov - 2 * torch.sqrt(torch.matmul(x_cov, y_cov)))
        return out
