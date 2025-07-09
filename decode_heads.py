import torch
import torch.nn as nn


## TODO
class LinearClassifier(nn.Module):
    def __init__(self, feat_dim: int, n_classes: int):
        super(LinearClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(feat_dim, n_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)


## TODO
class Segmenter(nn.Module):
    pass