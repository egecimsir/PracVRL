import re
import torch

from omegaconf import OmegaConf
from src.diffusion.base.guidance import simple_guidance_fn
from src.diffusion.stateful_flow_matching.scheduling import LinearScheduler
from src.diffusion.stateful_flow_matching.sampling import EulerSampler


def instantiate_from_config(config):
    module_path, class_name = config["class_path"].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)(**config.get("init_args", {}))


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


def initialize_DDT(
        config_path = "configs/repa_improved_ddt_xlen22de6_512.yaml",
        checkpoint_path = "model.ckpt",
        classes = ["tiger cat"],
        num_steps = 100,
        guidance = 8.5,
        guidance_min = 0.02, 
        guidance_max = 0.98,
        last_step = 0.005, 
        timeshift = 0.9,
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ):
    
    # Load configuration and models
    cfg = OmegaConf.load(config_path)
    vae = instantiate_from_config(cfg.model.vae)
    denoiser = instantiate_from_config(cfg.model.denoiser)
    conditioner = instantiate_from_config(cfg.model.conditioner)

    # Load weights
    print(f"Loading weights from {checkpoint_path}...")
    ckpt = torch.load(
        checkpoint_path, 
        map_location=torch.device("cpu") if device != "cuda" else device
    )
    denoiser = load_weights(denoiser, ckpt).to(device).eval()
    vae = vae.to(device).eval()

    # Use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        denoiser = torch.nn.DataParallel(denoiser)

    # Set up sampler
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

    # Process class names
    label_map = parse_class_labels()
    valid_classes = [cls for cls in classes if cls.lower() in label_map]

    return vae, denoiser, conditioner, sampler


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

