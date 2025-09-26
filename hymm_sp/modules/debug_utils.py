from loguru import logger
import torch
import numpy as np
import os

tensor_dict = {}

disable_inspect = os.getenv("DISABLE_INSPECT", "0") == "1"
dump_path = os.getenv("DUMP_PATH", "tensor_dict.pt")

def inspect_tensor(tensor, name="tensor", stop=False, depth=1):
    if disable_inspect:
        return
    if name in tensor_dict:
        logger.error(f"Tensor name '{name}' already exists in tensor_dict. Please use a unique name.")
        exit(1)
    tensor_dict[name] = tensor.clone().to("cpu") if tensor is not None else None
    if tensor is None:
        logger.opt(depth=depth).info(f"{name} is None")
    else:
        logger.opt(depth=depth).info(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, mean={tensor.float().mean():.8f}, std={tensor.float().std():.8f}, min={tensor.min().item():.8f}, max={tensor.max().item():.8f}")
    if stop:
        with open(dump_path, "wb") as f:
            torch.save(tensor_dict, f)
            logger.info(f"Dumped tensor_dict to {dump_path}")
        input("Press Enter to continue...")

def inspect_nparray(array, name="array", stop=False, depth=1):
    if disable_inspect:
        return
    if array is None:
        logger.opt(depth=depth).info(f"{name} is None")
    else:
        logger.opt(depth=depth).info(f"{name}: shape={array.shape}, dtype={array.dtype}, mean={array.mean():.8f}, std={array.std():.8f}, min={array.min():.8f}, max={array.max():.8f}")
    if stop:
        input("Press Enter to continue...")

def inspect_list(lst, name="list", stop=False, depth=1):
    if disable_inspect:
        return
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

def diff_tensors(tensor1, tensor2, name: str | None = None):
    diff: torch.Tensor = (tensor1 - tensor2).abs()
    flat_diff = diff.reshape(-1)
    max_diff, flat_idx = torch.max(flat_diff, dim=0)
    max_idx = torch.unravel_index(flat_idx, tensor1.shape)
    idx_str = ', '.join([str(i.item()) for i in max_idx])
    val1 = tensor1[max_idx].item()
    val2 = tensor2[max_idx].item()
    logger.warning(
        f"Value mismatch{f' for {name}' if name else ''}: max diff={max_diff.item():.8f}, mean diff={diff.mean().item():.8f}, "
        f"max diff index=({idx_str}), value1={val1:.8f}, value2={val2:.8f}"
    )

def check_same(dict1, dict2, atol=1e-5, rtol=1e-3):
    for name, tensor in dict1.items():
        # logger.debug(f"checking {name}...")
        if name in dict2:
            tensor2 = dict2[name]
            if tensor is None and tensor2 is None:
                logger.info(f"{name} both None")
                continue
            if tensor.shape != tensor2.shape:
                logger.warning(f"Shape mismatch for {name}: {tensor.shape} vs {tensor2.shape}")
            else:
                if not torch.allclose(tensor, tensor2, atol=atol, rtol=rtol):
                    diff_tensors(tensor, tensor2, name=name)
                else:
                    logger.info(f"{name} matches")
        else:
            logger.warning(f"{name} not found in second dict")

# check_same(dict1, dict2, atol=1e-2, rtol=1e-2)

def test_pad(x):
    return torch.nn.functional.pad(x, (1, 1, 1, 1, 2, 0), mode='replicate')
