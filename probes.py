import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super(LinearProbe, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, n_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)