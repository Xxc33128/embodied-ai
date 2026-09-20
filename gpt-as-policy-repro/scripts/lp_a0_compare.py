#!/usr/bin/env python3
"""LP-A0 比对：三腿 npz → 成对差值 + 阈值判定 → evidence JSON。"""
import json
import sys

import numpy as np

OUT = "/workspace/data/lp_a0_formal"
TH = {"per_draw_max_abs": 0.03, "per_draw_mean_abs": 0.005}


def cmp(a, b):
    d = np.abs(a - b)
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean())}


def main():
    j = np.load(OUT + "/jax.npz")["actions"].astype(np.float64)
    c = np.load(OUT + "/torch_cpu.npz")["actions"].astype(np.float64)
    n = np.load(OUT + "/torch_npu.npz")["actions"].astype(np.float64)
    draws = []
    for i in range(len(j)):
        r = {"draw": i,
             "jax_vs_cpu": cmp(j[i], c[i]),
             "npu_vs_cpu": cmp(n[i], c[i]),
             "jax_vs_npu": cmp(j[i], n[i])}
        r["gate"] = (r["npu_vs_cpu"]["max_abs"] <= TH["per_draw_max_abs"]
                     and r["npu_vs_cpu"]["mean_abs"] <= TH["per_draw_mean_abs"])
        draws.append(r)
        print(f"DRAW {i} gate={r['gate']} npu_vs_cpu={r['npu_vs_cpu']} jax_vs_cpu={r['jax_vs_cpu']}")
    passes = sum(d["gate"] for d in draws)
    out = {"schema": "gap_repro.lp_a0_formal.v1", "thresholds": TH,
           "draws": draws, "passes": passes, "total": len(draws)}
    json.dump(out, open(OUT + "/evidence.json", "w"), indent=1)
    print(f"COMPARE_DONE passes={passes}/{len(draws)}")


if __name__ == "__main__":
    main()
