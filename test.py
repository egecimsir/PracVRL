OUTPUT_DIR = "imagenet_features"

import gc
import os
import copy
import yaml

import torch
import torch.nn as nn
from src.lightning_model import LightningModel as MyLightningModel


if __name__ == "__main__":

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    def instantiate(cfg):
        if not isinstance(cfg, dict) or "class_path" not in cfg:
            raise ValueError("Config must have a 'class_path'")
        
        module_path, cls_name = cfg["class_path"].rsplit(".", 1)
        module = __import__(module_path, fromlist=[cls_name])
        
        return getattr(module, cls_name)(**cfg.get("init_args", {}))


    # load config and components
    with open("configs/repa_improved_ddt_xlen22de6_256.yaml") as f:
        cfg = yaml.safe_load(f)


    print("Instantiating model parts...")
    vae = instantiate(cfg["model"]["vae"])
    conditioner = instantiate(cfg["model"]["conditioner"])
    denoiser = instantiate(cfg["model"]["denoiser"])

    # shared scheduler
    sched_path = (cfg["model"]["diffusion_trainer"]["init_args"].get("scheduler"))
    assert isinstance(sched_path, str)
    scheduler = instantiate({"class_path": sched_path, "init_args": {}})

    # Deep copy to preserve original config details
    trainer_cfg = copy.deepcopy(cfg["model"]["diffusion_trainer"])
    trainer_cfg["init_args"]["scheduler"] = scheduler
    trainer = instantiate(trainer_cfg)

    sampler_cfg = copy.deepcopy(cfg["model"]["diffusion_sampler"])
    sampler_cfg["init_args"]["scheduler"] = scheduler
    sampler = instantiate(sampler_cfg)

    # load lightning model
    assert os.path.isfile("model.ckpt"), "Checkpoint not found."
    model_cpu = MyLightningModel.load_from_checkpoint(
        "model.ckpt",
        vae=vae, 
        conditioner=conditioner,
        denoiser=denoiser, 
        diffusion_trainer=trainer,
        diffusion_sampler=sampler, 
        strict=False,
        map_location="cpu"
    )
    print("Loaded checkpoint on CPU")


    # move to muGPU(s) if available
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model_cpu).to(device)
            model.module.eval()
            model.module.freeze()
        else:
            model = model_cpu.to(device)
            model.eval()
            model.freeze()
    else:
        model = model_cpu
        model.eval()
        model.freeze()

    print(f"Model ready on {device}")

    ddt = model.module.denoiser if isinstance(model, nn.DataParallel) else model.denoiser
    vae = model.module.vae if isinstance(model, nn.DataParallel) else model.vae
    print(f"DDT model device: {next(ddt.parameters()).device}")


