import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as T
from utils import parse_class_labels


## TODO
class TimeStepData(Dataset):
    def __init__(self, activations: dict, label_file: str):
        super(TimeStepData, self).__init__()
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass
    
