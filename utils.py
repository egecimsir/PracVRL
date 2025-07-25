import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import colormaps
from torch import Tensor



def overlay_mask(img, seg_mask, title: str = "Overlay", alpha = 0.5):
    if isinstance(img, Tensor):
        if img.ndim == 4:
            img.squeeze(0)
        img = img.permute(1, 2, 0)
    if isinstance(seg_mask, Tensor):
        if seg_mask.ndim == 4:
            seg_mask.squeeze(0)
        seg_mask = seg_mask.permute(1, 2, 0)

    img, seg_mask = np.asarray(img), np.asarray(seg_mask)
    cmap = colormaps['jet']
    colored_mask = cmap(seg_mask / seg_mask.max())[:, :, :3]  # Drop alpha channel

    return (1 - alpha) * img / 255.0 + alpha * colored_mask


def load_weights(model, checkpoint, prefix="ema_denoiser."):
    state_dict = checkpoint["state_dict"]
    loaded, total = 0, len(model.state_dict())

    for name, param in model.state_dict().items():
        full_name = prefix + name
        if full_name in state_dict:
            try:
                param.copy_(state_dict[full_name])
                loaded += 1
            except Exception as e:
                print(f"Failed to load {full_name}: {e}")
        else:
            print(f"Missing key in checkpoint: {full_name}")

    print(f"Loaded {loaded}/{total} weights.")

    return model


def tensor_to_image(x):
    if x is None:
        raise ValueError("Input tensor is None")
    return torch.clamp((x + 1.0) * 127.5 + 0.5, 0, 255).to(torch.uint8)


def parse_class_labels(filename="imagenet_classlabels.txt"):

    label_map = {}
    try:
        with open(filename, "r") as f:
            for line in f:
                match = re.match(r'\|\s*(\d+)\s*\|\s*(.*?)\s*\|', line)
                if match:
                    label_map[match.group(2).strip().lower()] = int(match.group(1))
    except FileNotFoundError:
        print("Label file not found.")
        exit()

    return label_map


