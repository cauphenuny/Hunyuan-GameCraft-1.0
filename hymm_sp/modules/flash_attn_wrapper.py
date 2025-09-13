import math
from loguru import logger
from jaxtyping import Float, Int
import torch
import torch.nn.functional as F
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_gpu
except ImportError:
    flash_attn_varlen_func_gpu = None
    logger.warning("Use NPU interface")
    import torch_npu
    def flash_attn_varlen_func_npu(
        q: Float[torch.Tensor, " total_q nheads headdim"],
        k: Float[torch.Tensor, " total_k nheads_k headdim"],
        v: Float[torch.Tensor, " total_k nheads_k headdim"],
        cu_seqlens_q: Int[torch.Tensor, " (batch_size + 1)"],
        cu_seqlens_k: Int[torch.Tensor, " (batch_size + 1)"],
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = False,
        #window_size: tuple[int, int] = (-1, -1),  # -1 means infinite context window
        #softcap=0.0, # 0.0 means deactivated
        #alibi_slopes=None,
        #deterministic=False,
        #return_attn_probs=False,
        #block_table=None,
    ):
        nheads = q.shape[-2]
        headdim = q.shape[-1]
        if softmax_scale is None:
            softmax_scale = 1 / math.sqrt(headdim)

        max_seqlen = max(max_seqlen_q, max_seqlen_k)
        # logger.info(f"{max_seqlen_q = }, {max_seqlen_q = }")

        if not causal:
            return torch_npu.npu_fusion_attention(
                q, k, v, nheads,
                pse=None,
                atten_mask=None,
                scale=softmax_scale,
                keep_prob=1 - dropout_p,
                input_layout="TND",
                actual_seq_qlen=cu_seqlens_q[1:].cpu().numpy().tolist(),
                actual_seq_kvlen=cu_seqlens_k[1:].cpu().numpy().tolist()
            )[0]
        else:
            atten_mask_npu = torch.triu(torch.ones([max_seqlen, max_seqlen]), diagonal=1).bool().to(q.device)
            logger.info(f"Mask size: {atten_mask_npu.numel():,}")

            return torch_npu.npu_fusion_attention(
                q, k, v, nheads,
                pse=None,
                padding_mask=None,
                atten_mask=atten_mask_npu,
                scale=softmax_scale,
                keep_prob=1 - dropout_p,
                input_layout="TND",
                actual_seq_qlen=cu_seqlens_q[1:].cpu().numpy().tolist(),
                actual_seq_kvlen=cu_seqlens_k[1:].cpu().numpy().tolist(),
                sparse_mode=3
            )[0]

def inspect_tensor(tensor, name="tensor"):
    if tensor is None:
        logger.info(f"{name} is None")
    else:
        logger.info(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}, min={tensor.min().item():.6f}, max={tensor.max().item():.6f}")
    input("Press Enter to continue...")

def flash_attn_varlen_func(*args, **kwargs):
    if flash_attn_varlen_func_gpu:
        q, k, v = args[:3]
        inspect_tensor(q, "flash_attn_varlen_func_gpu q")
        inspect_tensor(k, "flash_attn_varlen_func_gpu k")
        inspect_tensor(v, "flash_attn_varlen_func_gpu v")
        result =  flash_attn_varlen_func_gpu(*args, **kwargs)
        inspect_tensor(result, "flash_attn_varlen_func_gpu result")
    else:
        q, k, v = args[:3]
        inspect_tensor(q, "flash_attn_varlen_func_npu q")
        inspect_tensor(k, "flash_attn_varlen_func_npu k")
        inspect_tensor(v, "flash_attn_varlen_func_npu v")
        result = flash_attn_varlen_func_npu(*args, **kwargs)
        inspect_tensor(result, "flash_attn_varlen_func_npu result")