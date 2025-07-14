import torch
import torch.nn as nn
import lightning.pytorch as pl

from typing import Callable, List
from torch import Tensor
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from src.models.denoiser.decoupled_improved_dit import DDT
from src.models.vae import BaseVAE, fp2uint8
from src.models.conditioner import BaseConditioner
from src.utils.model_loader import ModelLoader
from src.diffusion.base.scheduling import BaseScheduler
from src.diffusion.base.sampling import BaseSampler
from src.diffusion.base.training import BaseTrainer
from src.diffusion.stateful_flow_matching.scheduling import LinearScheduler
from src.utils.no_grad import no_grad, filter_nograd_tensors



class DDTExtractor(BaseSampler):
    def __init__(
            self, 
            scheduler: BaseScheduler, 
            loss_weight_fn: Callable,
            lognorm_t=False,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.lognorm_t = lognorm_t
        self.loss_weight_fn = loss_weight_fn
        pass

    def _impl_sampling(self, ddt, x, t, cond, uncond):
        
        out = ddt(x, t, uncond)

        return

    
class DDTEncoder(pl.LightningModule):
    def __init__(
            self,
            vae: BaseVAE,
            denoiser: DDT,
            conditioner: BaseConditioner,
            diffusion_sampler: DDTExtractor,
            diffusion_trainer: BaseTrainer = None,
            optimizer = None,
            lr_scheduler = None,
        
        ):
        super().__init__()
        self.vae = vae
        self.conditioner = conditioner
        self.denoiser = denoiser
        self.diffusion_sampler = diffusion_sampler
        self.diffusion_trainer = diffusion_trainer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.timesteps: List[float] = [0.95, 0.5]
        
    def training_step(self, batch, batch_idx):
        raw_images, x, y = batch
        with torch.no_grad():
            x = self.vae.encode(x)
            cond, uncond = self.conditioner(y)
        loss = self.diffusion_trainer(self.denoiser, raw_images, x, cond, uncond)
        self.log_dict(loss, prog_bar=True, on_step=True)

        return loss["loss"]
    
    def predict_step(self, batch, batch_idx):
        """Extraction Method"""
        
        raw_images, img_b, y_b = batch
        activations = {t_step: None for t_step in self.timesteps}
        
        z_b = self.vae.encode(img_b)
        cond, uncond = self.conditioner(y_b)

        for t_step in self.timesteps:    
            t = torch.full_like(z_b, fill_value=t_step)
            _, s_t = self.diffusion_sampler(ddt=self.denoiser(), x=z_b, t=t, cond=cond, uncond=uncond)
            activations[t_step] = s_t

        return activations

    def validation_step(self, batch, batch_idx):
        return self.predict_step(batch, batch_idx)
    
    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        self.denoiser.state_dict(
            destination=destination,
            prefix=prefix+"denoiser.",
            keep_vars=keep_vars)
        self.diffusion_trainer.state_dict(
            destination=destination,
            prefix=prefix+"diffusion_trainer.",
            keep_vars=keep_vars)
        
        return destination
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        params_denoiser = filter_nograd_tensors(self.denoiser.parameters())
        params_trainer = filter_nograd_tensors(self.diffusion_trainer.parameters())
        optimizer: torch.optim.Optimizer = self.optimizer([*params_trainer, *params_denoiser])
        if self.lr_scheduler is None:
            return dict(
                optimizer=optimizer
            )
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return dict(
                optimizer=optimizer,
                lr_scheduler=lr_scheduler
            )
