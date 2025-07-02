import argparse
import torch
import torch.nn as nn
import gradio as gr

from omegaconf import OmegaConf
from PIL import Image
from huggingface_hub import snapshot_download

from src.models.vae import fp2uint8
from src.diffusion.base.guidance import simple_guidance_fn
from src.diffusion.stateful_flow_matching.sharing_sampling import EulerSampler
from src.diffusion.stateful_flow_matching.scheduling import LinearScheduler



def instantiate_class(config):
    kwargs = config.get("init_args", {})
    class_module, class_name = config["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    
    return args_class(**kwargs)


def load_model(weight_dict, denoiser):
    prefix = "ema_denoiser."
    for k, v in denoiser.state_dict().items():
        try:
            v.copy_(weight_dict["state_dict"][prefix + k])
        except:
            print(f"Failed to copy {prefix + k} to denoiser weight")
    
    return denoiser



class HookedPipeline:
    def __init__(
            self,
            vae,
            denoiser,
            conditioner,
            diffusion_sampler,
            resolution,
            cls2idx,
    ):
        self.vae = vae
        self.denoiser = denoiser
        self.conditioner = conditioner
        self.diffusion_sampler = diffusion_sampler
        self.resolution = resolution
        self.cls2idx = cls2idx

        self._activations = {}
        self._hook_denoiser()

        
    @activations.getter
    def activations(self):
        return self._activations
    
    
    def _hook_denoiser(self):
        """
        Registers forward hooks to capture activations of the denoiser layers 
        and stores activations in self.activations.
        """
        self.activations = {}

        def fwd_hook(name):
            def hook(layer, input, output):
                self.activations[name] = output.detach().cpu()
            
            return hook
        
        for name, layer in self.denoiser.named_children():
            layer.register_forward_hook(fwd_hook(name))


    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def __call__(
            self, 
            y, 
            n_images, 
            seed, 
            n_steps, 
            guidance, 
            state_refresh_rate, 
            guidance_interval_min,
            guidance_interval_max,
            timeshift,
    ):
        self.diffusion_sampler.n_steps = n_steps
        self.diffusion_sampler.guidance = guidance
        self.diffusion_sampler.state_refresh_rate = state_refresh_rate
        self.diffusion_sampler.guidance_interval_min = guidance_interval_min
        self.diffusion_sampler.guidance_interval_max = guidance_interval_max
        self.diffusion_sampler.timeshift = timeshift

        generator = torch.Generator(device="cpu").manual_seed(seed)
        size = (n_images, 4, self.resolution // 8, self.resolution // 8)
        xT = torch.randn(size, device="cpu", dtype=torch.float32, generator=generator).to("cuda")

        with torch.no_grad():
            condition, uncondition = self.conditioner([self.cls2idx[y],] * n_images)

        # Clear previous activations
        self.activations.clear()
        # Sample images:
        samples = self.diffusion_sampler(self.denoiser, xT, condition, uncondition)
        samples = self.vae.decode(samples)

        samples = fp2uint8(samples)
        samples = samples.permute(0, 2, 3, 1).cpu().numpy()

        images = []
        for i in range(n_images):
            image = Image.fromarray(samples[i])
            images.append(image)
        
        # Return images and activations
        return images, self.activations


def store_activations(
        config: str = "configs/repa_improved_ddt_xlen22de6_256.yaml", 
        resolution: int = 512, 
        ckpt_path: str = "models",
    ):
    
    args = {
        "config": config,
        "resolution": resolution,
        "ckpt_path": ckpt_path
    }

    config = OmegaConf.load(config)
    vae_config = config.model.vae
    diffusion_sampler_config = config.model.diffusion_sampler
    denoiser_config = config.model.denoiser
    conditioner_config = config.model.conditioner

    vae = instantiate_class(vae_config)
    denoiser = instantiate_class(denoiser_config)
    conditioner = instantiate_class(conditioner_config)

    diffusion_sampler = EulerSampler(
       scheduler=LinearScheduler(),
       w_scheduler=LinearScheduler(),
       guidance_fn=simple_guidance_fn,
       num_steps=50,
       guidance=3.0,
       state_refresh_rate=1,
       guidance_interval_min=0.3,
       guidance_interval_max=1.0,
       timeshift=1.0
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    denoiser = load_model(ckpt, denoiser)
    denoiser = denoiser.cuda()
    vae = vae.cuda()
    denoiser.eval()
    print("\nModels initialized\n")


    # read imagenet classlabels
    with open("imagenet_classlabels.txt", "r") as f:
        classlabels = f.readlines()
        classlabels = [label.strip() for label in classlabels]
    print("Labels initialized\n")

    classlabels2id = {label: i for i, label in enumerate(classlabels)}
    id2classlabels = {i: label for i, label in enumerate(classlabels)}

    pipeline = HookedPipeline(vae, denoiser, conditioner, diffusion_sampler, resolution, classlabels2id)
    print("Pipeline initialized\n")

    with gr.Blocks() as demo:
        gr.Markdown("DDT")
        with gr.Row():
            with gr.Column(scale=1):
                num_steps = gr.Slider(minimum=1, maximum=100, step=1, label="num steps", value=50)
                guidance = gr.Slider(minimum=0.1, maximum=10.0, step=0.1, label="CFG", value=4.0)
                num_images = gr.Slider(minimum=1, maximum=10, step=1, label="num images", value=4)
                label = gr.Dropdown(choices=classlabels, value=id2classlabels[948], label="label")
                seed = gr.Slider(minimum=0, maximum=1000000, step=1, label="seed", value=0)
                state_refresh_rate = gr.Slider(minimum=1, maximum=10, step=1, label="encoder reuse", value=1)
                guidance_interval_min = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="interval guidance min",
                                                  value=0.0)
                guidance_interval_max = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, label="interval guidance max",
                                                  value=1.0)
                timeshift = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, label="timeshift", value=1.0)
            with gr.Column(scale=2):
                btn = gr.Button("Generate")
                output = gr.Gallery(label="Images")

        btn.click(fn=pipeline,
                  inputs=[
                      label,
                      num_images,
                      seed,
                      num_steps,
                      guidance,
                      state_refresh_rate,
                      guidance_interval_min,
                      guidance_interval_max,
                      timeshift
                  ], outputs=[output])
    demo.launch(server_name="0.0.0.0", server_port=7861)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/repa_improved_ddt_xlen22de6_256.yaml")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--ckpt_path", type=str, default="model.ckpt")
    args = parser.parse_args()

    store_activations(args.config, args.resolution, args.ckpt_path)