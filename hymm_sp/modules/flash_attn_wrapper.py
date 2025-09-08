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

def flash_attn_varlen_func_torch(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    return_attn_probs=False
):
    """
    PyTorch 实现的变长序列 FlashAttention 函数。
    输入形状:
        q: (total_q, num_heads, head_dim)
        k, v: (total_k, num_heads, head_dim)
        cu_seqlens_q/k: (batch_size + 1,)
    """
    assert q.dtype == k.dtype == v.dtype, "输入数据类型必须一致"

    total_q, num_heads, head_dim = q.shape
    total_k = k.shape[0]
    batch_size = cu_seqlens_q.shape[0] - 1

    # 1. 分割变长序列为单独的注意力计算
    outputs = []
    attn_probs = [] if return_attn_probs else None

    for i in range(batch_size):
        # 获取当前序列的起止位置
        q_start, q_end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
        k_start, k_end = cu_seqlens_k[i], cu_seqlens_k[i + 1]

        # 提取当前序列的 q, k, v
        qi = q[q_start:q_end]  # (seqlen_q, num_heads, head_dim)
        ki = k[k_start:k_end]  # (seqlen_k, num_heads, head_dim)
        vi = v[k_start:k_end]  # (seqlen_k, num_heads, head_dim)

        # 2. 计算注意力分数
        scores = torch.einsum("qhd,khd->hqk", qi, ki)  # (num_heads, seqlen_q, seqlen_k)

        if softmax_scale is None:
            softmax_scale = 1.0 / (head_dim ** 0.5)
        scores = scores * softmax_scale

        # 3. 因果掩码（如果是解码器）
        if causal:
            causal_mask = torch.triu(
                torch.ones(qi.shape[0], ki.shape[0]), diagonal=1
            ).bool().to(device=qi.device)  # (seqlen_q, seqlen_k)
            scores.masked_fill_(causal_mask, float("-inf"))

        # 4. Softmax 和 Dropout
        attn_prob = F.softmax(scores, dim=-1)  # (num_heads, seqlen_q, seqlen_k)
        if dropout_p > 0.0:
            attn_prob = F.dropout(attn_prob, p=dropout_p)

        # 5. 加权求和
        out = torch.einsum("hqk,khd->qhd", attn_prob, vi)  # (seqlen_q, num_heads, head_dim)
        outputs.append(out)

        if return_attn_probs:
            assert attn_probs is not None
            attn_probs.append(attn_prob)

    # 6. 合并所有序列的输出
    output = torch.cat(outputs, dim=0)  # (total_q, num_heads, head_dim)

    if return_attn_probs:
        return output, attn_probs
    return output

def flash_attn_varlen_func(*args, **kwargs):
    if flash_attn_varlen_func_gpu:
        return flash_attn_varlen_func_gpu(*args, **kwargs)
    else:
        return flash_attn_varlen_func_npu(*args, **kwargs)