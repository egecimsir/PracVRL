import os
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn

from datasets import load_dataset
from torch.utils.data import IterableDataset
from torchvision.transforms import transforms as T

from huggingface_hub import login
from omegaconf import OmegaConf
from PIL import Image

from utils import load_weights, tensor_to_image, parse_class_labels
from src.diffusion.base.guidance import simple_guidance_fn
from src.diffusion.stateful_flow_matching.scheduling import LinearScheduler
from src.diffusion.stateful_flow_matching.sampling import EulerSampler



class ImageNet1K(IterableDataset):
    split_lens = {
        "train": 1_281_167,
        "validation": 50_000,
        "test": 100_000
    }

    def __init__(self, split: str = "train", normalize=True):
        login(token=os.getenv("HF_TOKEN"))
        dataset = load_dataset("imagenet-1k", split=split, token=True, streaming=True)
        trafos = [
            T.Resize(256), T.CenterCrop(224),
            T.ToTensor(),
        ]
        if normalize:
            trafos.append(T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]))
                
        self.dataset = dataset.shuffle(buffer_size=1000).map(self.preprocess, batched=True, batch_size=32)
        self.transform = T.Compose(trafos)
        
        self.split = split
        self._label_map = parse_class_labels()
        self._idx2label = {v:k for k,v in self.label_map.items()}


    def __iter__(self):
        for item in self.dataset:
            yield item["imgs"], item["label"]


    def __len__(self):
        return ImageNet1K.split_lens[self.split]
    
    
    def preprocess(self, batch):
        batch["imgs"] = [self.transform(img.convert("RGB")) for img in batch["image"]]
        return batch



def instantiate_from_config(config):
    module_path, class_name = config["class_path"].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)(**config.get("init_args", {}))


def initialise_models(config_path: str, ckpt_path: str, device: str):
    cfg = OmegaConf.load(config_path)
    vae = instantiate_from_config(cfg.model.vae)
    denoiser = instantiate_from_config(cfg.model.denoiser)
    conditioner = instantiate_from_config(cfg.model.conditioner)

    # Load weights
    ckpt = torch.load(ckpt_path)
    denoiser = load_weights(denoiser, ckpt).to(device).eval()
    vae = vae.to(device).eval()

    return vae, denoiser, conditioner


def store_activations(
        config_path: str,
        ckpt_path: str,
        device: str,
        cls_name: str,
        pkl_save_path: str = None,
        img_save_path: str = None,
        resolution: int = 256, 
        num_steps = 100,
        guidance = 8.5,
        guidance_min = 0.02, 
        guidance_max = 0.98,
        last_step = 0.005, 
        timeshift = 0.9,
        img_seed=1234,
        verbose=False,
    ) -> dict:

    ## Initialise models
    vae, denoiser, conditioner = initialise_models(
        config_path=config_path, 
        ckpt_path=ckpt_path, 
        device=device
    )

    # Use multiple GPUs if available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        denoiser = torch.nn.DataParallel(denoiser)

    sampler = EulerSampler(
        scheduler=LinearScheduler(),
        w_scheduler=LinearScheduler(),
        guidance_fn=simple_guidance_fn,
        num_steps=num_steps,
        guidance=guidance,
        state_refresh_rate=1,
        guidance_interval_min=guidance_min,
        guidance_interval_max=guidance_max,
        timeshift=timeshift,
        last_step=last_step
    )

    ## Create hook
    activations = {}
    current_timestep = {'t': None}  # Use dict for mutability in closure

    def fwrd_hook(name):
        def hook_fn(layer, input, output):
            t = current_timestep['t']
            if t is not None:
                t_val = float(t[0].item()) if isinstance(t, torch.Tensor) else float(t)
                if t_val not in activations:
                    activations[t_val] = {}
                activations[t_val][name] = output.detach().cpu()
        return hook_fn

    ## Register hooks to encoder
    for name, layer in denoiser.blocks.named_modules():
        if name.count(".") == 1:
            if verbose: print(f"Registered hook for: {name}")
            layer.register_forward_hook(fwrd_hook(name))

    ## Adjust denoiser.forward() to track timestep
    orig_forward = denoiser.forward
    def forward_with_timestep(*args, **kwargs):
        ## Capture time argument (t)
        t = args[1] if len(args) > 1 else kwargs.get('t', None)
        current_timestep['t'] = t
        return orig_forward(*args, **kwargs)
    denoiser.forward = forward_with_timestep

    ## Pass forward through denoiser
    cls_id = ImageNet1K.label_map[cls_name.lower()]
    generator = torch.Generator().manual_seed(img_seed)
    noise = torch.randn((1, 4, resolution//8, resolution//8), generator=generator).to(device)
    with torch.no_grad():
        cond, uncond = conditioner([cls_id])
        output = sampler(denoiser, noise, cond.to(device), uncond.to(device))
        decoded = vae.decode(output.to(device))

    ## Save if specified
    if pkl_save_path is not None:
        with open(pkl_save_path, "wb") as f:
            pickle.dump(activations, f)

    if img_save_path is not None:
        img_tensor = tensor_to_image(decoded.cpu())[0].permute(1, 2, 0).numpy()
        img_tensor = img_tensor[:, :, :3] if img_tensor.shape[2] > 3 else img_tensor
        img = Image.fromarray(img_tensor)
        fname = f"{cls_name.replace(' ', '_')}_seed{img_seed}.png"
        img.save(os.path.join(img_save_path, fname))

    return activations, decoded


def get_encoder_activations(activations: dict, config_path: str):
    l0: int = len(activations)
    n_encoders: int = OmegaConf.load(config_path).model.denoiser.init_args.num_encoder_blocks
    
    activations = {int(float(f"{k:.3f}")*1000): v for k, v in activations.items()}
    assert l0 == len(activations)

    encoder_activations = {}
    for t, t_act in activations.items():
        encoder_activations[t] = {name: act for name, act in t_act.items() if f"{n_encoders}." in name}

    assert l0 == len(encoder_activations)
    
    return encoder_activations



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Store and print timestep-dependent activations.")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--class', dest='cls_name', type=str, required=True, help='Class name (e.g. "goldfish")')
    parser.add_argument('--out', type=str, default=None, help='Path to save activations (pickle)')
    parser.add_argument('--img_out', type=str, default=None, help='Directory to save generated image')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()


    activations, _ = store_activations(
        config_path=args.config,
        ckpt_path=args.ckpt,
        device=args.device,
        cls_name=args.cls_name,
        pkl_save_path=args.out,
        img_save_path=args.img_out,
        verbose=args.verbose
    )
    
    ## TODO: Logging instead printing
    print(f"Stored activations for {len(activations)} timesteps.")
    for t, layers in sorted(activations.items()):
        print(f"Timestep {t:.5f}: {list(layers.keys())}")

