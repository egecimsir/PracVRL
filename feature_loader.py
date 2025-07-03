import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as T
from utils import parse_class_labels


## TODO
class TimeStepData(Dataset):
    def __init__(self, activations: dict, label_file: str):
        super(TimeStepData, self).__init__()
        self.label_map = parse_class_labels(filename=label_file)
        self.labels = list(self.label_map.keys())
        self.targets = list(self.label_map.items())

    def __len__(self):
        pass

    def __getitem__(self):
        pass
    
