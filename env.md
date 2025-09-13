# environment settings

## flash_attn

npu 不支持 flash_attn

将 npu 的 torch_npu.npu_fusion_attention 包装成 flash_attn_varlen_func:

[document: FlashAttentionScore](https://www.hiascend.com/document/detail/zh/Pytorch/710/ptmoddevg/trainingmigrguide/performance_tuning_0034.html)

## triton

安装 triton_ascend

## fp8

好像现在 ascend 不支持 torch.float8_e4m3fn?
