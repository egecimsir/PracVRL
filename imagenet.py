import os
import torch

from datasets import load_dataset
from torch.utils.data import IterableDataset
from torchvision.transforms import transforms as T
from utils import parse_class_labels


def get_token():
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise ValueError(
            "HF_TOKEN environment variable not found.\nExpoert as export HF_TOKEN=...\n"
        )
    return token


class ImageNet1K(IterableDataset):
    
    ## Class variables
    label_map = parse_class_labels()
    split_lens = {
        "train": 1_281_167,
        "validation": 50_000,
        "test": 100_000
    }

    def __init__(self, split: str = "validation", normalize=True, transform=None):
        token = get_token()
        try:
            ## Dataset from HF
            print("Trying to access dataset on HF...\n")
            self.dataset = load_dataset(
                "benjamin-paine/imagenet-1k-256x256", 
                split=split, 
                token=token,
                streaming=True
            ).shuffle(buffer_size=1000).map(self.preprocess, batched=True, batch_size=32)
        
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ImageNet dataset. Make sure your token has access. \nError: {str(e)}"
            )
        
        ## Transformations
        trafos = [T.ToTensor()]
        if normalize: trafos.append(T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]))  
        if transform is not None: trafos.append(list(transform.transforms))
        self.transform = T.Compose(trafos)

        self.split = split
        self.label_map = ImageNet1K.label_map
        self.idx2label = {v:k for k,v in self.label_map.items()}


    def __iter__(self):
        for item in self.dataset:
            yield item["imgs"], item["label"]


    def __len__(self):
        return ImageNet1K.split_lens[self.split]
    
    
    def preprocess(self, batch):
        batch["imgs"] = [self.transform(img.convert("RGB")) for img in batch["image"]]
        return batch
