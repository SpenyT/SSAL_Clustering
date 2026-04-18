import torch
import torch.nn as nn
from glob_config import DEVICE, N_GPUS

# I might be overengineering but I'm assuming you might like
# the convenience of having the possibility of using multiple GPUs


def prepare_model(model: torch.nn.Module) -> torch.nn.Module:
    if N_GPUS > 1:
        print(f"Using {N_GPUS} GPUs")
        model = torch.nn.DataParallel(model)
    else:
        print(f"Using device: {DEVICE}")
    return model.to(DEVICE)


# torch.compile only works on 20 series or newer NVIDIA gpus
def try_compile(model: nn.Module) -> nn.Module:
    if DEVICE.type != "cuda" or torch.cuda.get_device_capability()[0] < 7:
        return model
    try:
        return torch.compile(model, mode="reduce-overhead")
    except Exception:
        return model
