import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, feat_dim: int, n_classes: int):
        super(LinearClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(feat_dim, n_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)


class SegmentationHead(nn.Module):
    def __init__(self, in_dim=1152, n_class=19, patch_size=16, out_size=224):
        super().__init__()
        self.in_dim = in_dim
        self.n_class = n_class
        self.patch_size = patch_size
        self.resolution = out_size // patch_size
        
        ## Modules
        self.project = nn.Sequential(
            nn.Linear(in_dim, n_class),
            nn.SiLU()
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(n_class, n_class, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.BatchNorm2d(n_class)
        )
        
    def forward(self, x: Tensor):
        B, N, d = x.shape                     ## (B, 196, 1152)
        H = W = self.resolution               ## 224

        x = self.project(x).permute(0, 2, 1)  ## (B, n_class, 14, 14)
        x = x.view(B, self.n_class, H, W)     ## (B, n_class, 14, 14)
        
        return self.upsample(x)               ## (B, n_class, 224, 224)