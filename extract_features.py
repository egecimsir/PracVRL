import os
import gc
import torch
import torch.nn as nn
import torchvision.transforms as T

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import Cityscapes
from omegaconf import OmegaConf
from tqdm import tqdm

from src.models.vae import BaseVAE
from src.models.conditioner import BaseConditioner
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
    """
    Instantializes DDT Model with all dependencies.
    """
    def __init__(
        self, 
        config_path: str = "configs/repa_improved_ddt_xlen22de6_256.yaml", 
        ckpt_path: str = "model.ckpt", 
        device: str = "cpu",
        n_classes: int = 35
    ):
        self.config = OmegaConf.load(config_path)
        self.ckpt_path = ckpt_path
        self.device = device
        self.n_classes = n_classes

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
        self.vae.precompute = False
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

        print(f"Number of encoder blocks: {self.num_encoder_blocks}")
        print("\nCompleted!\n---------------\n")
        
    def register_encoder_hook(self, activations: list = []) -> list:
        assert len(activations) == 0, "Input list must be empty!\n"

        def hook_fn(layer: DDTBlock, input: Tensor, output: Tensor):
            activations.append(output.detach().cpu())    
        
        ## Register hooks to target layer
        for n, block in self.ddt.blocks.named_children():  ## 0-27
            if n == str(self.num_encoder_blocks-1):
                block.register_forward_hook(hook_fn)
                print(f"Hook registered for layer: ddt.blocks.{n}")

        return activations

    def to(self, dev: str):
        self.device = dev
        self.ddt.to(dev)
        self.vae.to(dev)
        self.sampler.to(dev)
        self.diff_trainer.to(dev)

    def to_eval(self):
        self.ddt.eval()
        self.vae.eval()
        self.sampler.eval()
        self.diff_trainer.eval()

    def to_train(self):
        self.ddt.train()
        self.vae.train()
        self.sampler.train()
        self.diff_trainer.train()


class SegImagesWithLabels(Dataset):
    """
    Wrap datasets so that, returns (img, y) instead of (img, y_map);
    to be fed into DDT.forward(x, t, y)
    """
    def __init__(self, split: str = "train", transform=None, trgt_transform=None, trgt_type: str = "semantic"):
        assert trgt_type in ["semantic", "instance"]
        super().__init__()
        self.dataset = Cityscapes(
            root="cityscapes",
            split=split,
            mode="fine",
            target_type=trgt_type
        )
        # Handle default transforms correctly
        if transform is None:
            self.transform = T.ToTensor()
        elif isinstance(transform, list):
            self.transform = T.Compose(transform)
        else:
            self.transform = transform

        if trgt_transform is None:
            self.trgt_transform = T.ToTensor()
        elif isinstance(trgt_transform, list):
            self.trgt_transform = T.Compose(trgt_transform)
        else:
            self.trgt_transform = trgt_transform

        self.train_id_to_name = {c.train_id: c.name for c in Cityscapes.classes if c.train_id != 255}
        self.n_classes = len(self.train_id_to_name)
        self.max_classes = 1000  # match config/ckpt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, y_map = self.dataset[index]

        y_map = self.trgt_transform(y_map)
        if not torch.is_tensor(y_map):
            y_map = T.ToTensor()(y_map)

        if self.dataset.target_type == "semantic":
            return self.transform(img), self.process_sem_mask(y_map)
        else:
            return self.transform(img), self.process_ins_mask(y_map)

    def process_sem_mask(self, y_map: Tensor) -> Tensor:
        """
        y is one-hot encoding of label or label combination, rest is not useful.
        """
        y_sem = y_map.long().squeeze(0)  # shape: (H, W)
        mask = (y_sem != 255)

        # Find all unique class ids in the mask (excluding 255)
        class_ids = torch.unique(y_sem[mask])
        y = torch.zeros(self.max_classes, dtype=torch.long)

        # one-hot for all present classes
        for cid in class_ids:
            y[int(cid)] = 1

        return y

    def process_ins_mask(self, y_map: Tensor) -> Tensor:
        """
        y should be a tensor of shape (..., 1000);
        first n_classes are class indices, rest is not useful
        """
        label_id = self.get_label_id(y_map)
        y = torch.full((self.max_classes,), 0, dtype=torch.long)
        y[label_id] = 1  # one-hot for the class

        return y

    def get_label_id(self, y_map):
        y_map = y_map.long().squeeze(0)
        mask = (y_map != 255)
        if mask.sum() == 0:
            return torch.full((self.max_classes,), 0, dtype=torch.long)  # fallback to class 0 if all unlabeled
        ids, counts = torch.unique(y_map[mask], return_counts=True)
        
        return int(ids[counts.argmax()])


def extract_features(
        model: DDTWrapper, 
        t_step: int, 
        loader: DataLoader,
        output_dir: str,
        save_every: int = 10,
        device: str = "cuda",
    ):
    print(f"Extracting features using {device.upper()}")
    os.makedirs(output_dir, exist_ok=True)
    
    file_counter = 0

    ddt = model.ddt
    vae = model.vae
    scheduler = model.scheduler

    t0 = torch.tensor([t_step], device=device)
    a0 = scheduler.alpha(t0).view(-1,1,1,1).to(device)
    s0 = scheduler.sigma(t0).view(-1,1,1,1).to(device)

    activations = model.register_encoder_hook([])
    for _, (x, y) in enumerate(tqdm(loader, 
                    desc="Extracting Features", 
                    total=len(loader.dataset) // loader.batch_size, 
                    leave=True)
        ):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            z = vae.encode(x)
            noise = torch.randn_like(z)
            zt = a0 * z + s0 * noise
            y_idx = y.argmax(dim=1)

            _ = ddt(
                x=zt,
                t=torch.full((z.shape[0],), t_step, device=device, dtype=torch.float32),
                y=y_idx
            )
        
            del z, zt, noise
            torch.cuda.empty_cache()

        if len(activations) >= save_every:
            ## Save every act in activations list and empty list
            act_chunk = torch.cat([act.cpu() for act in activations[:save_every]], dim=0)
            save_path = os.path.join(output_dir, f"features_{file_counter:05d}.pt")
            torch.save(act_chunk, save_path)

            ## delete items
            activations[:] = activations[save_every:]
            file_counter += 1
    
    ## Save any remaining activations
    if len(activations) > 0:
        save_path = os.path.join(output_dir, f"features_{file_counter:05d}.pt")
        torch.save(activations, save_path)
    
    print("Extracted!")



if __name__ == "__main__":
    CONF_PATH = "configs/repa_improved_ddt_xlen22de6_256.yaml"
    CKPT_PATH = "model.ckpt"
    device = "cuda" if torch.cuda.is_available() else "cpu"


    gc.collect()
    torch.cuda.empty_cache()
    print(f"Device: {device.upper()}")


    model = DDTWrapper(
        config_path=CONF_PATH,
        ckpt_path=CKPT_PATH,
        device=device
    )

    if device == "cuda" and torch.cuda.device_count() > 1:
        ## Fully sharded data parallel
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        model.vae = FSDP(model.vae)
        model.ddt = FSDP(model.ddt)
        model.conditioner = FSDP(model.conditioner)
        model.sampler = FSDP(model.sampler)
        model.diff_trainer = FSDP(model.diff_trainer)
        print("Model -> FSDP")

        model.to_eval() 

        for SPLIT in ("val", "test", "train"):

            ## ARGUMENTS
            TIME_STEPS = [0.95, 0.50]
            TRGT_TYPE = "semantic"
            OUTPUT_DIR = os.path.join("cityscapes_features", SPLIT)

            if not os.path.exists(OUTPUT_DIR):

                B_SIZE = 64
                N_SAMPLES = 5000
                N_CLASSES = 1000
                SAVE_EVERY = 5
                TRAFOS = [
                    T.Resize(256), 
                    T.CenterCrop(224),
                    T.ToTensor(),
                ]
                TRGT_TRAFOS = [
                    T.Resize(256), 
                    T.CenterCrop(224),
                    T.ToTensor(),
                ]

                ## Preperation
                os.makedirs(OUTPUT_DIR)
                gc.collect()
                torch.cuda.empty_cache()

                dataset = SegImagesWithLabels(
                    split=SPLIT, 
                    transform=TRAFOS, 
                    trgt_transform=TRGT_TRAFOS, 
                    trgt_type=TRGT_TYPE
                )
                dataloader = DataLoader(dataset, batch_size=B_SIZE)

                ## MAIN LOOP
                for t in TIME_STEPS:
                    out_path = os.path.join(OUTPUT_DIR, f"timestep_{int(t*100)}")
                    os.makedirs(out_path, exist_ok=True)
                    print(f"Saving to: {out_path}")

                    ## Extract whole dataset
                    extract_features(
                        model=model,
                        t_step=t,
                        loader=dataloader,
                        output_dir=out_path,
                        save_every=SAVE_EVERY,
                        device=device
                    )
                    print(f"Extracted for time-step {t} !\n")

                print("Done!")

            else:
                print("Folder already exits!")
    print("Done!")
