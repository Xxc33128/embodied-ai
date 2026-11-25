#!/usr/bin/env python3
"""NPU 最小探针：区分卡选择问题 vs 算子编译子系统问题。用法: probe.py <card>"""
import sys

import torch
import torch_npu  # noqa: F401

card = sys.argv[1] if len(sys.argv) > 1 else "0"
x = torch.randn(64, 64, device="npu")
print(f"PROBE card={card} basic_matmul_ok={float((x @ x).sum()) == float((x @ x).sum())}")
# 触发一个需要在线编译的非平凡算子路径（conv1d+bn 类融合更接近 pi05 实际负载）
conv = torch.nn.Conv1d(8, 16, 3).to("npu")
y = conv(torch.randn(2, 8, 32, device="npu"))
print(f"PROBE card={card} conv1d_ok={y.shape}")
print(f"PROBE card={card} DONE")
