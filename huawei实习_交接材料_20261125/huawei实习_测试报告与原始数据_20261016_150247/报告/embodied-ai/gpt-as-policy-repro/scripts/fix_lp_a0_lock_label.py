#!/usr/bin/env python3
"""按 LP-A0 对抗审查结论修正 lock 口径（PASS→Torch 段通过，JAX leg 未闭合）。"""
import json
from pathlib import Path

p = Path("configs/libero_pro.lock.json")
d = json.loads(p.read_text())
d["lp_a0"]["status"] = ("torch-segment pass (development-diagnostic evidence); "
                        "JAX leg + preregistered thresholds OPEN (review 2026-09-18: P1 x2)")
d["lp_a0"]["label_correction"] = (
    "实测数字经攻击存活（noise 注入贯穿去噪循环、determinism=0 可信、证据链闭合），"
    "但按计划 §2 定义 LP-A0=JAX CPU→Torch CPU→NPU 逐张量对比，本轮仅完成 Torch 段"
    "且阈值未预注册——PASS 标签撤回")
p.write_text(json.dumps(d, ensure_ascii=False, indent=1))
print("lock reclassified")
