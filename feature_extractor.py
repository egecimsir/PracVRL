import os
import gc
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from src.models.vae import BaseVAE, fp2uint8
from src.models.conditioner import BaseConditioner
from src.utils.model_loader import ModelLoader
from src.callbacks.simple_ema import SimpleEMA
from src.diffusion.base.sampling import BaseSampler
from src.diffusion.base.training import BaseTrainer
from src.diffusion.stateful_flow_matching.scheduling import LinearScheduler

from src.lightning_model import LightningModel
from imagenet import ImageNet1K
from utils import load_weights


class Model:
    def __init__(
            self,
            vae: BaseVAE,
            conditioner: BaseConditioner,
            denoiser: nn.Module,
            diffusion_trainer: BaseTrainer,
            diffusion_sampler: BaseSampler,
    ):
        self.vae = vae
        self.conditioner = conditioner
        self.denoiser = denoiser
        self.diffusion_trainer = diffusion_trainer
        self.diffusion_sampler = diffusion_sampler


def instantiate_from_config(config):
    module_path, class_name = config["class_path"].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)(**config.get("init_args", {}))


def initialize_models(config_path: str, ckpt_path: str, device: str, lighning=True):
    cfg = OmegaConf.load(config_path)

    vae = instantiate_from_config(cfg["model"]["vae"])
    denoiser = instantiate_from_config(cfg["model"]["denoiser"])
    conditioner = instantiate_from_config(cfg["model"]["conditioner"])

    # Load weights
    if device == "cuda":
        ckpt = torch.load(ckpt_path)
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    denoiser = load_weights(denoiser, ckpt).to(device).eval()
    vae = vae.to(device).eval()

    # Get scheduler class path
    sched_path = cfg["model"]["diffusion_trainer"]["init_args"]["scheduler"]
    
    # Create scheduler config
    scheduler = instantiate_from_config({
        "class_path": sched_path,
        "init_args": {}
    })
    # Set up trainer with scheduler
    trainer_cfg = {
        "class_path": cfg["model"]["diffusion_trainer"]["class_path"],
        "init_args": {**cfg["model"]["diffusion_trainer"]["init_args"], "scheduler": scheduler}
    }
    diff_trainer = instantiate_from_config(trainer_cfg)

    # Set up sampler with scheduler
    sampler_cfg = {
        "class_path": cfg["model"]["diffusion_sampler"]["class_path"],
        "init_args": {**cfg["model"]["diffusion_sampler"]["init_args"], "scheduler": scheduler}
    }
    sampler = instantiate_from_config(sampler_cfg)

    if lighning:
        model = LightningModel.load_from_checkpoint(
            checkpoint_path="model.ckpt",
            vae=vae,
            conditioner=conditioner,
            denoiser=denoiser,
            diffusion_trainer=diff_trainer,
            diffusion_sampler=sampler,
            strict=False
        )
        return model, scheduler
    
    return Model(vae, conditioner, denoiser, diff_trainer, sampler), scheduler


def register_encoder_hook(activations: dict, model: nn.Module, trgt_layer: int):
    assert len(activations) == 0, "Activation dict is full!\n"

    current_timestep = {'t': None}

    def fwrd_hook(name):
        def hook_fn(layer, input, output):
            t = current_timestep['t']
            if t is not None:
                t_val = float(t[0].item()) if isinstance(t, torch.Tensor) else float(t)
                if t_val not in activations:
                    activations[t_val] = {}
                activations[t_val][name] = output.detach().cpu()
        
        return hook_fn
    
    ## Register hooks to target layer
    for n, block in model.blocks.named_children():  ## 0-27
        if n == str(trgt_layer-1):
            for name, layer in block.named_children():
                layer.register_forward_hook(fwrd_hook(f"{n}.{name}"))
                print(f"Hook registered for output of layer: {n}.{name}\n")
    
    ## Adjust model.forward() to track timestep
    orig_forward = model.forward
    def forward_with_timestep(*args, **kwargs):
        ## Capture time argument (t)
        t = args[1] if len(args) > 1 else kwargs.get('t', None)
        assert t is not None, "Could't extract time-step when running model.forward()\n"
        current_timestep['t'] = t
        return orig_forward(*args, **kwargs)
    
    model.forward = forward_with_timestep

    return activations, model


def extract_features(
        vae: nn.Module,                 ## Latent space encoder
        ddt: nn.Module,                 ## Diffusion Transformer
        scheduler: LinearScheduler,
        time_steps: iter,
        loader: DataLoader,
        device: str, 
        n_samples=50_000,
        n_classes=1000,
        one_batch=True
    ):
    n_batches = n_samples // loader.batch_size
    
    vae = vae.eval().to(device)
    ddt = ddt.eval().to(device)

    ## Extraction loop
    for t_step in time_steps:
        with torch.no_grad():
            for i, (img_b, y_b) in enumerate(tqdm(loader, desc="Extracting Features: ", total=n_batches)):
                img_b, y_b = img_b.to(device), y_b.to(device)

                z_raw = vae.encode(img_b)
                z = z_raw.sample() if hasattr(z_raw, "sample") else z_raw

                noise = torch.rand_like(z)
                uncond = torch.full_like(y_b, n_classes)

                t = torch.full((z.shape[0],), t_step, device=device).to(device)
                a_t = scheduler.alpha(torch.tensor([t_step])).view(-1,1,1,1).to(device)
                s_t = scheduler.sigma(torch.tensor([t_step])).view(-1,1,1,1).to(device)
                z_t = a_t * z + s_t * noise
                
                # No need to reshape - keep the 4D format (B, C, H, W)
                out = ddt(z_t, t, uncond)

            if one_batch:
                ## Test run
                break


def main(config_path: str, ckpt_path: str, device: str, hf_token: str, lightning=False):
    
    num_enc = OmegaConf.load(config_path)["model"]["denoiser"]["init_args"]["num_encoder_blocks"]
    print(f"Number of encoder blocks: {num_enc}")

    activations = {}
    try:
        ## Initialize models with hooks
        model, scheduler = initialize_models(
            config_path=config_path,
            ckpt_path=ckpt_path,
            device=device,
            lighning=lightning
        )
        register_encoder_hook(activations, model.denoiser, trgt_layer=num_enc)

        ## Load ImageNet
        os.environ["HF_TOKEN"] = hf_token  # Set token in environment
        dataset = ImageNet1K(split="validation", token=hf_token)
        dataloader = DataLoader(
            dataset, 
            batch_size=8,
            num_workers=0,  ## Disable multiprocessing for CPU
            pin_memory=device == "cuda"
        )

        ## Extract time dependent features to activations
        extract_features(
            vae=model.vae,
            ddt=model.denoiser,
            scheduler=scheduler,
            time_steps=[0.95],
            loader=dataloader,
            device=device,
            n_samples=len(dataset),
            n_classes=len(dataset.idx2label)
        )

        ## Clean up
        del dataloader
        del dataset
        torch.cuda.empty_cache()
    
    
    ##except Exception as e:
    ##    print(f"Error: {str(e)}\n")
    
    ## Cleanup
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    

    return activations



if __name__ == "__main__":

    config_path = "configs/repa_improved_ddt_xlen22de6_256.yaml"
    ckpt_path = "model.ckpt"
    hf_token = os.getenv("HF_TOKEN")  # Get token from environment variable
    
    if not hf_token:
        raise ValueError("Please set HF_TOKEN environment variable")

    gc.collect()
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    activations = main(
        config_path=config_path,
        ckpt_path=ckpt_path,
        device=device,
        hf_token=hf_token,
        lightning=False  # Use custom model instead of Lightning
    )
    print(f"Collected activations for {len(activations)} timesteps")
