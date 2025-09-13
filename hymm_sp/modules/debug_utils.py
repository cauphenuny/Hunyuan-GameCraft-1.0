from loguru import logger

def inspect_tensor(tensor, name="tensor", stop=False):
    if tensor is None:
        logger.info(f"{name} is None")
    else:
        logger.info(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}, min={tensor.min().item():.6f}, max={tensor.max().item():.6f}")
    if stop:
        input("Press Enter to continue...")