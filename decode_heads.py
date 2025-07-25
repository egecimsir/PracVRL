import torch
import torch.nn as nn
from torch import Tensor


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
    def __init__(self, in_dim=1152, n_class=19, out_size=224):
        super().__init__()
        self.n_class = n_class
        self.out_size = out_size

        # Reshape flat vector to feature map
        self.feature_dim = (128, 3, 3)  # 128×3×3 = 1152

        # Convolutional decoder to upsample to 224×224
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 6×6
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),    # 12×12
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 24×24
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 48×48
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, n_class, kernel_size=4, stride=2, padding=1), # 96×96
            nn.Upsample(size=(out_size, out_size), mode='bilinear', align_corners=False)  # final 224×224
        )

    def forward(self, x):
        B = x.size(0)                     ## x: (B, 1152)
        x = x.view(B, *self.feature_dim)  ## (B, 128, 3, 3)
        logits = self.decoder(x)          ## (B, n_class, 224, 224)
        return logits



class PatchedSegmentationHead(nn.Module):
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
    
