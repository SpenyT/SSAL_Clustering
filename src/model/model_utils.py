import torch
import torch.nn as nn
from glob_config import DEVICE, N_GPUS

# I might be overengineering but I'm assuming you might like
# the convenience of having the possibility of using multiple GPUs


def prepare_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Move a model to the target device, wrapping in DataParallel if needed.

    If more than one GPU is available, wraps the model in DataParallel
    before transferring. Otherwise moves directly to DEVICE.

    Arguments
    ---------
    model : torch.nn.Module
        The model to prepare.

    Returns
    -------
    torch.nn.Module
        The model on the target device, optionally wrapped in DataParallel.

    Example
    -------
    >>> model = prepare_model(load_resnet18())
    """
    if N_GPUS > 1:
        print(f"Using {N_GPUS} GPUs")
        model = torch.nn.DataParallel(model)
    return model.to(DEVICE)


def try_compile(model: nn.Module) -> nn.Module:
    """
    Attempt to compile the model with torch.compile for faster inference.

    Compilation is only attempted on CUDA devices with compute capability
    >= 7.0 (Volta and later). Falls back to the original model silently
    if compilation fails or the device is unsupported.

    Arguments
    ---------
    model : nn.Module
        The model to compile.

    Returns
    -------
    nn.Module
        The compiled model, or the original model if compilation was
        skipped or failed.

    Example
    -------
    >>> model = try_compile(prepare_model(load_resnet18()))
    """
    if DEVICE.type != "cuda" or torch.cuda.get_device_capability()[0] < 7:
        return model
    try:
        return torch.compile(model, mode="reduce-overhead")
    except Exception:
        return model
