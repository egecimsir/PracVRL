import os
import pickle

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
from extract_features import SegImagesWithLabels
from PIL import Image


FEAT_ROOT = "cityscapes_features"
TRGT_ROOT = "cityscapes"


def get_label_id(y_map: Tensor):
    y_map = y_map.long().squeeze(0)
        
    mask = (y_map != 255)
    if mask.sum() == 0:
        return torch.full((1000,), 0, dtype=torch.long)  # fallback to class 0 if all unlabeled
        
    ids, counts = torch.unique(y_map[mask], return_counts=True)
    label_id = int(ids[counts.argmax()])

    return label_id


class FeatureSegmentationDataset(Dataset):
    """
    Loads extracted features from OUTPUT_DIR and matches them with segmentation maps from Cityscapes.
    Assumes features are saved as batches in .pt files: features_00000.pt, etc.
    """
    def __init__(
            self, 
            split="val", 
            timestep: float = 0.95,
            features_root: str = FEAT_ROOT,
            cityscapes_root: str = TRGT_ROOT, 
            trgt_type="instance", 
            transform=None, 
            trgt_transform=None, 
        ):
        super().__init__()
        
        self.split = split
        self.trgt_type = trgt_type
        self.transform = transform if transform else T.Lambda(lambda x: x)
        
        if trgt_transform is not None:
            self.trgt_transform = trgt_transform
        else:
            self.trgt_transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor()
            ])
        
        ## Build directory path for features
        self.features_dir = os.path.join(features_root, split, f"timestep_{int(timestep * 100)}")
        self.feature_files = sorted([
            os.path.join(self.features_dir, f) for f in os.listdir(self.features_dir) if f.endswith(".pt")
        ])

        ## Features
        loaded_features = []
        for fp in self.feature_files:
            data = torch.load(fp, map_location="cpu")
            if isinstance(data, list):
                loaded_features.extend(data)
            else:
                loaded_features.append(data)
        
        features = torch.cat(loaded_features, dim=0)
        self.features = [feat.unsqueeze(0) for feat in features]

        ## Targets, Labels
        cityscapes = Cityscapes(root=cityscapes_root,split=split, target_type=trgt_type)
        self.num_samples = min(len(self.features), len(cityscapes))

        self.targets = [self.trgt_transform(cityscapes[i][-1]) for i in range(len(self))]
        self.label_ids = [get_label_id(self.targets[i]) for i in range(len(self))]

        self.idx2label = {c.train_id: c.name for c in Cityscapes.classes if c.train_id != 255}

        del features, loaded_features, cityscapes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        feat = self.transform(self.features[idx])
        y_map = self.targets[idx]
        y_id = self.label_ids[idx]
        
        return feat, (y_id, y_map)
    

if __name__ == "__main__":
    """
    ## Save dummy files

    EXT_B_SIZE = 64
    SAVED_EVERY = 10
    for timestep in [0.95, 0.50]:
        for split in ["val", "test", "train"]:

            features_dir = os.path.join(FEAT_ROOT, split, f"timestep_{int(timestep * 100)}")
            file_path = os.path.join(features_dir, "feature.pt")
            if os.path.exists(file_path):
                os.remove(file_path)

            feat = torch.rand(EXT_B_SIZE * SAVED_EVERY, 196, 1152, dtype=torch.float32)
            torch.save(feat, file_path)
    """
    from sklearn.model_selection import train_test_split
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    dataset = FeatureSegmentationDataset(
        features_root="cityscapes_features",
        split="val",
        timestep=0.95,
        trgt_type="semantic",
        cityscapes_root="cityscapes"
    )

    # Stack features and flatten for TSNE
    features_tensor = torch.cat(dataset.features, dim=0)  # shape: (N, ...)
    features_flat = features_tensor.view(features_tensor.size(0), -1).numpy()
    label_ids = dataset.label_ids

    ## ----
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(features_flat)

    print("Done!")
