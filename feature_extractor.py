import os
import gc
import torch
import torch.nn as nn

from torch import Tensor
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
from src.models.denoiser.decoupled_improved_dit import DDT, DDTBlock
from utils import load_weights



def instantiate_from_config(config):
    module_path, class_name = config["class_path"].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)(**config.get("init_args", {}))


class DDTWrapper:
    def __init__(
        self, 
        config_path: str = "configs/repa_improved_ddt_xlen22de6_256.yaml", 
        ckpt_path: str = "model.ckpt", 
        device: str = "cpu"
    ):
        self.config = OmegaConf.load(config_path)
        self.ckpt_path = ckpt_path
        self.device = device

        ## Models
        self.vae: BaseVAE = None
        self.conditioner: BaseConditioner = None
        self.ddt: DDT = None
        self.diff_trainer: BaseTrainer = None
        self.sampler: BaseSampler = None
        self.scheduler: LinearScheduler = None

        ## Initialization
        self.num_encoder_blocks = self.config["model"]["denoiser"]["init_args"]["num_encoder_blocks"]
        self.initialize_models(self.config, ckpt_path)


    def initialize_models(self, cfg: OmegaConf, ckpt_path: str):
        print("\n---------------\nInitializing ddt, vae, conditioner")
        self.vae = instantiate_from_config(cfg["model"]["vae"])
        self.ddt = instantiate_from_config(cfg["model"]["denoiser"])
        self.conditioner = instantiate_from_config(cfg["model"]["conditioner"])

        print("Loading weights...")
        if self.device == "cuda":
            ckpt = torch.load(ckpt_path)
        else:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)

        self.ddt = load_weights(self.ddt, ckpt).to(self.device).eval()
        self.vae = self.vae.to(self.device).eval()

        ## Create scheduler
        print("Initializing diffusion trainer, sampler, scheduler")
        self.scheduler = instantiate_from_config({
            "class_path": cfg["model"]["diffusion_trainer"]["init_args"]["scheduler"],
            "init_args": {}
        })
        # Set up trainer with scheduler
        trainer_cfg = {
            "class_path": cfg["model"]["diffusion_trainer"]["class_path"],
            "init_args": {**cfg["model"]["diffusion_trainer"]["init_args"], "scheduler": self.scheduler}
        }
        self.diff_trainer = instantiate_from_config(trainer_cfg)

        # Set up sampler with scheduler
        sampler_cfg = {
            "class_path": cfg["model"]["diffusion_sampler"]["class_path"],
            "init_args": {**cfg["model"]["diffusion_sampler"]["init_args"], "scheduler": self.scheduler}
        }
        self.sampler = instantiate_from_config(sampler_cfg)
        print(f"Number of encoder blocks: {model.num_encoder_blocks}")
        print("\nCompleted!\n---------------\n")
        

    def register_encoder_hook(self, activations: list):
        assert len(activations) == 0, "Input list must be empty!\n"

        def hook_fn(layer: DDTBlock, input: Tensor, output: Tensor):
            activations.append(output.detach().cpu().clone())    
        
        ## Register hooks to target layer
        for n, block in self.ddt.blocks.named_children():  ## 0-27
            if n == str(self.num_encoder_blocks-1):
                block.register_forward_hook(hook_fn)
                print(f"Hook registered for layer: ddt.blocks.{n}")
        


def extract_features(
        vae: nn.Module,
        ddt: nn.Module,
        scheduler: LinearScheduler,
        time_steps: iter,
        loader: DataLoader,
        device: str, 
        n_samples: int,
        n_classes: int,
        one_batch=True
    ):
    n_batches = n_samples // loader.batch_size

    vae = vae.eval().to(device)
    ddt = ddt.eval().to(device)

    for t_step in time_steps:
        with torch.no_grad():
            for _, (img_b, y_b) in enumerate(tqdm(loader, desc="Extracting Features", total=n_batches)):
                img_b, y_b = img_b.to(device), y_b.to(device)

                z_raw = vae.encode(img_b)
                z = z_raw.sample() if hasattr(z_raw, "sample") else z_raw

                noise = torch.rand_like(z)
                uncond = torch.full_like(y_b, n_classes)

                t = torch.full((z.shape[0],), t_step, device=device)
                a_t = scheduler.alpha(t_step).view(-1, 1, 1, 1).to(device)
                s_t = scheduler.sigma(t_step).view(-1, 1, 1, 1).to(device)
                z_t = a_t * z + s_t * noise

                ddt(z_t, t, uncond)

                if one_batch:
                    break


if __name__ == "__main__":

    CONF_PATH = "configs/repa_improved_ddt_xlen22de6_256.yaml"
    CKPT_PATH = "model.ckpt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device {device.upper()}")

    
    activations = []

    model = DDTWrapper(
        config_path=CONF_PATH,
        ckpt_path=CKPT_PATH,
        device=device
    )
    model.register_encoder_hook(activations)
    

    print("Done!")
