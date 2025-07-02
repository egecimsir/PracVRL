import torch
import torch.nn as nn



import torch
from omegaconf import OmegaConf
from src.models.vae import fp2uint8
from src.diffusion.base.guidance import simple_guidance_fn
from src.diffusion.stateful_flow_matching.sharing_sampling import EulerSampler
from src.diffusion.stateful_flow_matching.scheduling import LinearScheduler
from PIL import Image
import gradio as gr
from huggingface_hub import snapshot_download


def instantiate_class(config):
    kwargs = config.get("init_args", {})
    class_module, class_name = config["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(**kwargs)


def load_model(weight_dict, denosier):
    prefix = "ema_denoiser."
    for k, v in denoiser.state_dict().items():
        try:
            v.copy_(weight_dict["state_dict"][prefix + k])
        except:
            print(f"Failed to copy {prefix + k} to denoiser weight")
    return denoiser



class HookedPipeline:
    pass

