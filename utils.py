import re
import torch


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


