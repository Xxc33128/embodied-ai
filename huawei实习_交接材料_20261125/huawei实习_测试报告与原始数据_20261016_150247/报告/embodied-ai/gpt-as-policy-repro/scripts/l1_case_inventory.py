#!/usr/bin/env python3
"""L1：LIBERO-PRO 四套件 × 六条件 case 池清单生成（gap-sim 执行）。

产出 /workspace/data/l1_case_inventory.json：
- 24 个 benchmark（4 套件 × {Ori,Obj,Pos,Sem,Task,Env}）× 10 任务；
- 每条 case：problem_folder/bddl_file/任务名/benchmark.language/bddl 内嵌
  :language/bddl sha256/init 存在性；
- 已知缺口如实登记（Env 条件 BDDL+init 需 EnvironmentReplacePerturbator
  生成，本轮缺席）；
- 条件映射依据：钉定 PRO eafdb809 evaluation_config.yaml perturbation_mapping
  + perturbation.py 扰动器语义（SwapPerturbator=位置互换 → 论文 Pos）。
"""
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, "/workspace/repo/src")
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("LIBERO_CONFIG_PATH", "/workspace/data/libero_config")
from gap_repro.sim.libero_session import _ensure_runtime_env  # noqa: E402
_ensure_runtime_env()

from libero.libero import benchmark  # noqa: E402

BASE = ["libero_goal", "libero_spatial", "libero_10", "libero_object"]
VARIANTS = {"Ori": "", "Obj": "_object", "Pos": "_swap", "Sem": "_lan",
            "Task": "_task", "Env": "_env"}
ROOT = "/workspace/repo/upstream/LIBERO-PRO/libero/libero"

d = benchmark.get_benchmark_dict()
out = {"schema": "gap_repro.lp_l1_case_inventory.v1",
       "condition_mapping_basis": {
           "source": "Zxy-MLlab/LIBERO-PRO@eafdb809",
           "mapping": {"Ori": "baseline suite",
                        "Obj": "_object (ObjectReplacePerturbator)",
                        "Pos": "_swap (SwapPerturbator: 位置互换)",
                        "Sem": "_lan (LanguagePerturbator, :language 段)",
                        "Task": "_task (TaskPerturbator)",
                        "Env": "_env (EnvironmentReplacePerturbator, 需生成)"},
           "note": "_temp benchmark 为 init 生成的临时产物目录，非论文条件"},
       "known_gaps": [],
       "cases": []}
for b in BASE:
    for cond, suf in VARIANTS.items():
        name = b + suf
        inst = d[name]()
        for i in range(10):
            t = inst.get_task(i)
            bddl_path = f"{ROOT}/bddl_files/{name}/{t.bddl_file}"
            init_path = f"{ROOT}/init_files/{name}/{t.name}.pruned_init"
            entry = {"benchmark": name, "suite": b, "condition": cond,
                     "task_index": i, "task_name": t.name,
                     "problem_folder": t.problem_folder, "bddl_file": t.bddl_file,
                     "benchmark_language": t.language}
            try:
                content = open(bddl_path).read()
                m = re.search(r"\(:language\s+(.*?)\)", content)
                entry["bddl_language"] = m.group(1).strip() if m else None
                entry["bddl_sha256"] = hashlib.sha256(content.encode()).hexdigest()
                entry["bddl_exists"] = True
            except FileNotFoundError:
                entry["bddl_exists"] = False
                entry["bddl_language"] = None
                out["known_gaps"].append(f"bddl missing: {bddl_path}")
            entry["init_exists"] = os.path.isfile(init_path)
            if not entry["init_exists"]:
                out["known_gaps"].append(f"init missing: {init_path}")
            out["cases"].append(entry)

counts = {}
for c in out["cases"]:
    ok = c["bddl_exists"] and c["init_exists"]
    counts[(c["condition"], ok)] = counts.get((c["condition"], ok), 0) + 1
out["coverage_counts"] = {f"{k[0]}:{'ok' if k[1] else 'missing'}": v
                          for k, v in sorted(counts.items())}
out["totals"] = {"cases": len(out["cases"]),
                 "complete": sum(1 for c in out["cases"]
                                 if c["bddl_exists"] and c["init_exists"])}
json.dump(out, open("/workspace/data/l1_case_inventory.json", "w"),
          indent=1, ensure_ascii=False)
print(f"INVENTORY_DONE totals={out['totals']} gaps={len(out['known_gaps'])}")
