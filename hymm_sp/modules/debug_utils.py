from loguru import logger
import torch
import numpy as np

tensor_dict = {}

def inspect_tensor(tensor, name="tensor", stop=False, depth=1):
    tensor_dict[name] = tensor
    if tensor is None:
        logger.opt(depth=depth).info(f"{name} is None")
    else:
        logger.opt(depth=depth).info(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, mean={tensor.float().mean():.8f}, std={tensor.float().std():.8f}, min={tensor.min().item():.8f}, max={tensor.max().item():.8f}")
    if stop:
        with open("tensor_dict.pt", "wb") as f:
            torch.save(tensor_dict, f)
        input("Press Enter to continue...")

def inspect_nparray(array, name="array", stop=False, depth=1):
    if array is None:
        logger.opt(depth=depth).info(f"{name} is None")
    else:
        logger.opt(depth=depth).info(f"{name}: shape={array.shape}, dtype={array.dtype}, mean={array.mean():.8f}, std={array.std():.8f}, min={array.min():.8f}, max={array.max():.8f}")
    if stop:
        input("Press Enter to continue...")

def inspect_list(lst, name="list", stop=False, depth=1):
    if lst is None:
        logger.opt(depth=depth).info(f"{name} is None")
    else:
        for i, item in enumerate(lst):
            if isinstance(item, np.ndarray):
                inspect_nparray(item, f"{name}[{i}]", depth=depth+1)
            elif isinstance(item, torch.Tensor):
                inspect_tensor(item, f"{name}[{i}]", depth=depth+1)
            else:
                logger.opt(depth=depth).info(f"{name}[{i}]: type={type(item)}, value={item}")
    if stop:
        input("Press Enter to continue...")

def check_same(dict1, dict2):
    for name, tensor in dict1.items():
        if name in dict2:
            tensor2 = dict2[name]
            if tensor.shape != tensor2.shape:
                logger.warning(f"Shape mismatch for {name}: {tensor.shape} vs {tensor2.shape}")
            else:
                if not torch.allclose(tensor, tensor2, atol=1e-5, rtol=1e-3):
                    diff = (tensor - tensor2).abs()
                    logger.warning(f"Value mismatch for {name}: max diff={diff.max().item():.8f}, mean diff={diff.mean().item():.8f}")
                else:
                    logger.info(f"{name} matches")
        else:
            logger.warning(f"{name} not found in second dict")