#!/usr/bin/env python3
"""W6/W7：JAX 参考 vs NPU 输出按维对比，产出转换验证结论。"""
import json
import numpy as np

jax = json.load(open("/workspace/data/w6_jax_ref.json"))
npu = json.load(open("/workspace/data/w7_npu_baseline.json"))
jm = np.array(jax["per_dim_mean"])
nm = np.array(npu["per_dim_mean"])
abs_diff = np.abs(jm - nm)
rel = abs_diff / (np.abs(nm) + 1e-6)
res = {
    "per_dim_absdiff": [round(float(x), 6) for x in abs_diff],
    "per_dim_rel": [round(float(x), 4) for x in rel],
    "max_absdiff": float(abs_diff.max()),
    "mean_absdiff": float(abs_diff.mean()),
}
print(json.dumps(res, indent=1))
open("/workspace/data/w6_compare.json", "w").write(json.dumps(res, indent=1))
print("COMPARE_SAVED")
