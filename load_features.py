import os
import pickle
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
from extract_features import SegImagesWithLabels


FEAT_ROOT = "cityscapes_features"
TRGT_ROOT = "cityscapes"


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
        self.trgt_transform = trgt_transform if trgt_transform else T.ToTensor()
        
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

        ## Targets
        cityscapes = SegImagesWithLabels(split=split, trgt_type=trgt_type)
        self.targets = [self.trgt_transform(cityscapes.dataset[i][-1]) for i in range(len(self))]
        
        ## Labels
        self.label_ids = [cityscapes.get_label_id(y_map) for y_map in self.targets]

        del features, loaded_features, cityscapes

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat, y_id, y_map = self.features[idx], self.label_ids[idx], self.targets[idx]
        return self.transform(feat), (y_id, y_map)
    


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

    x, y = dataset[0]
    print(x.shape, y.shape) ## torch.Size([1, 196, 1152]) torch.Size([1, 1024, 2048])

    plt.figure()
    plt.imshow(y.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.show()

    features = dataset.features
    ## TODO: ...
    # Stack features into a 2D array for t-SNE: [num_samples, feature_dim]
    features_flat = torch.cat([f.flatten() for f in dataset.features], dim=0).view(len(dataset.features), -1)
    # Optionally, get a simple label for coloring (e.g., mean of target mask)
    labels = [y.numpy().mean() for _, y in dataset]

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(features_flat)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="viridis", s=10)
    plt.colorbar(scatter, label="Mean target value")
    plt.title("t-SNE of Features")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()

    ## ----
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(features)


    print("Done!")
