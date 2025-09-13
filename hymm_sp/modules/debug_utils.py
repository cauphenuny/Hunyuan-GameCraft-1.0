from loguru import logger
import torch
import numpy as np

def inspect_tensor(tensor, name="tensor", stop=False):
    if tensor is None:
        logger.info(f"{name} is None")
    else:
        logger.info(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}, min={tensor.min().item():.6f}, max={tensor.max().item():.6f}")
    if stop:
        input("Press Enter to continue...")

def inspect_nparray(array, name="array", stop=False):
    if array is None:
        logger.info(f"{name} is None")
    else:
        logger.info(f"{name}: shape={array.shape}, dtype={array.dtype}, mean={array.mean():.6f}, std={array.std():.6f}, min={array.min():.6f}, max={array.max():.6f}")
    if stop:
        input("Press Enter to continue...")

def inspect_list(lst, name="list", stop=False):
    if lst is None:
        logger.info(f"{name} is None")
    else:
        for i, item in enumerate(lst):
            if isinstance(item, np.ndarray):
                inspect_nparray(item, f"{name}[{i}]")
            elif isinstance(item, torch.Tensor):
                inspect_tensor(item, f"{name}[{i}]")
            else:
                logger.info(f"{name}[{i}]: type={type(item)}, value={item}")
    if stop:
        input("Press Enter to continue...")