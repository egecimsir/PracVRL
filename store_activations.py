import os
import pickle
import numpy as np

import torch
import torch.nn as nn

from omegaconf import OmegaConf
from PIL import Image

from utils import load_weights, tensor_to_image, parse_class_labels
from src.diffusion.base.guidance import simple_guidance_fn
from src.diffusion.stateful_flow_matching.scheduling import LinearScheduler
from src.diffusion.stateful_flow_matching.sampling import EulerSampler


label_map: dict = parse_class_labels()


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

    def fwrd_hook(name):
        def hook_fn(layer, input, output):
            activations[name] = output.detach().cpu()
        return hook_fn

    ## Register hooks to encoder
    for name, layer in denoiser.blocks.named_modules():
        if name.count(".") == 1:
            if verbose: print(f"Registered hook for: {name}")
            layer.register_forward_hook(fwrd_hook(name))


    ## Pass forward through denoiser
    cls_id = label_map[cls_name.lower()]
    generator = torch.Generator().manual_seed(img_seed)
    noise = torch.randn((1, 4, resolution//8, resolution//8), generator=generator).to(device)
    with torch.no_grad():
        cond, uncond = conditioner([cls_id])
        output = sampler(denoiser, noise, cond.to(device), uncond.to(device))
        decoded = vae.decode(output.to(device))

    ## Save if specified
    if pkl_save_path is not None:
        with open(pkl_save_path, "w") as f:
            pickle.dump(activations, f)

    if img_save_path is not None:
        img_tensor = tensor_to_image(decoded.cpu())[0].permute(1, 2, 0).numpy()
        img_tensor = img_tensor[:, :, :3] if img_tensor.shape[2] > 3 else img_tensor
        img = Image.fromarray(img_tensor)
        fname = f"{cls_name.replace(' ', '_')}_seed{img_seed}.png"
        img.save(os.path.join(img_save_path, fname))

    return activations, decoded

        