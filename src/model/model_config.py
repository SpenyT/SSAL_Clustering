import torch
from glob_config import DEVICE, N_GPUS

# I might be overengineering but I'm assuming you might like the convenience of having
# the possibility of using multiple GPUs
def prepare_model(model: torch.nn.Module) -> torch.nn.Module:
    if N_GPUS > 1:
        print(f"Using {N_GPUS} GPUs")
        model = torch.nn.DataParallel(model)
    else:
        print(f"Using device: {DEVICE}")
    return model.to(DEVICE)