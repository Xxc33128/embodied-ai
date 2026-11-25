#!/usr/bin/env python3
"""L1 收尾：生成 Env 条件（{suite}_env）的 BDDL + 50 init（gap-sim 执行）。

复用钉定 PRO@eafdb809 的官方实现（零语义漂移）：
- perturbation.py::EnvironmentReplacePerturbator（确定性文本变换：
  main_table → living_room_table，作者硬编码 new_env，随机被注释）
- notebooks/generate_init_states.py::generate_init_states（50×get_sim_state
  → pickle → torch zip 格式 .pruned_init；按 bddl 目录整批生成）

v2 加固（首次运行 ENV_GEN_ERRORED 后；诊断前先行防御）：
1. yaml 覆盖预检：ood_environment.yaml 必须覆盖全部 (suite, task)，且当前环境
   在作者允许集合内——配置缺失在 v1 会静默写出未修改的 BDDL（= 基线冒充 Env）。
2. vacuous 检测：基线环境已是 living_room_table 的任务，作者硬编码使其 Env 条件
   等价基线（candidates 排除自身 + new_env 恒为 living_room_table）。此类 case
   如实登记 vacuous=true，不伪造差异。
3. 签名适配：generate_init_states 以运行时 inspect 过滤 kwargs；num_inits 必须
   被支持（50 = 论文协议），不支持则硬失败——不静默降级到默认值。
4. 写后断言：每个 _env BDDL 与基线比对；非 vacuous 却无差异 → 硬失败。

工程决策（显式偏离登记）：官方脚本不设 seed（不可复现）；本管线在生成前
np.random.seed(42) 并记录——资产生成确定性不影响评测协议（评测消费冻结的
init 文件）。manifest（每文件 sha256）落 /workspace/data/l1_env_gen_manifest.json。
幂等：已存在的 BDDL/init 跳过；重复执行只补缺。
"""
import hashlib
import inspect
import json
import os
import sys

sys.path.insert(0, "/workspace/repo/src")
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("LIBERO_CONFIG_PATH", "/workspace/data/libero_config")
from gap_repro.sim.libero_session import _ensure_runtime_env  # noqa: E402
_ensure_runtime_env()

import numpy as np  # noqa: E402
import yaml  # noqa: E402

PRO = "/workspace/repo/upstream/LIBERO-PRO"
sys.path.insert(0, PRO)
sys.path.insert(0, PRO + "/notebooks")

from perturbation import BDDLParser, EnvironmentReplacePerturbator  # noqa: E402
import generate_init_states as gen  # noqa: E402

SEED = 42
NUM_INITS = 50
ROOT = PRO + "/libero/libero"
SUITES = ["libero_goal", "libero_spatial", "libero_10", "libero_object"]
ENV_CONFIG = PRO + "/libero_ood/ood_environment.yaml"
ALLOWED_ENVS = {"main_table", "kitchen_table", "living_room_table",
                "study_table", "floor"}
NEW_ENV = "living_room_table"  # 作者硬编码（perturbation.py:410），钉定语义

from libero.libero.benchmark import get_benchmark_dict  # noqa: E402
d = get_benchmark_dict()

manifest_path = "/workspace/data/l1_env_gen_manifest.json"
manifest = (json.load(open(manifest_path)) if os.path.exists(manifest_path)
            else {"schema": "gap_repro.lp_l1_env_gen.v2", "seed": SEED,
                  "num_inits": NUM_INITS,
                  "generator": "PRO@eafdb809 EnvironmentReplacePerturbator "
                               "+ generate_init_states",
                  "vacuous_cases": [], "files": {}})
manifest.setdefault("vacuous_cases", [])  # v1 manifest 升级路径

# ---- 0) 签名预检（fail loud，打印真实签名供一次性修补） ----
gen_params = set(inspect.signature(gen.generate_init_states).parameters)
print(f"[sig] generate_init_states{inspect.signature(gen.generate_init_states)}",
      flush=True)
for req in ("num_inits",):
    if req not in gen_params:
        raise RuntimeError(f"generate_init_states 不支持 {req}（论文协议 50 集必需），"
                           f"实际签名参数={sorted(gen_params)}")
if not os.path.isfile(ENV_CONFIG):
    raise FileNotFoundError(f"ood_environment.yaml 缺失: {ENV_CONFIG}")
ood_cfg = yaml.safe_load(open(ENV_CONFIG)) or {}

for suite in SUITES:
    inst = d[suite]()
    tasks = [inst.get_task(i) for i in range(10)]

    # ---- 1) yaml 覆盖预检（先于任何写入） ----
    for t in tasks:
        cur = (ood_cfg.get(suite, {}) or {}).get(t.name)
        if cur is None:
            raise RuntimeError(f"ood_environment.yaml 缺 {suite}/{t.name} "
                               f"——将静默写出基线冒充 Env，拒绝生成")
        cur = cur[0] if isinstance(cur, list) else cur
        if cur not in ALLOWED_ENVS:
            raise RuntimeError(f"ood_environment.yaml {suite}/{t.name} 环境 "
                               f"'{cur}' 不在作者允许集合 {ALLOWED_ENVS}")

    out_bddl_dir = f"{ROOT}/bddl_files/{suite}_env"
    out_init_dir = f"{ROOT}/init_files/{suite}_env"
    os.makedirs(out_bddl_dir, exist_ok=True)
    os.makedirs(out_init_dir, exist_ok=True)

    # ---- 2) BDDL：确定性文本变换（缺才写）+ vacuous/差异断言 ----
    for i in range(10):
        t = tasks[i]
        dst = f"{out_bddl_dir}/{t.bddl_file}"
        base_content = open(f"{ROOT}/bddl_files/{suite}/{t.bddl_file}").read()
        cur = (ood_cfg.get(suite, {}) or {}).get(t.name)
        cur = cur[0] if isinstance(cur, list) else cur
        vacuous = cur == NEW_ENV
        if not os.path.exists(dst):
            if vacuous:
                # 作者代码语义：candidates 排除自身且 new_env 恒定 → 内容不变。
                # 仍写副本（benchmark 加载需要该文件），但登记 vacuous。
                new_content = base_content
            else:
                parser = BDDLParser(base_content)
                new_content = EnvironmentReplacePerturbator(
                    parser, ENV_CONFIG).perturb(suite, t.name)
            with open(dst, "w") as f:
                f.write(new_content)
            print(f"[{suite}] bddl written: {t.bddl_file}"
                  f"{' (vacuous==baseline)' if vacuous else ''}", flush=True)
        if vacuous and f"{suite}/{t.name}" not in manifest["vacuous_cases"]:
            manifest["vacuous_cases"].append(f"{suite}/{t.name}")
        final_content = open(dst).read()
        if not vacuous and final_content == base_content:
            raise RuntimeError(
                f"非 vacuous 任务 {suite}/{t.name} 的 _env BDDL 与基线无差异"
                f"——变换器未生效，拒绝登记")
        manifest["files"][f"bddl_files/{suite}_env/{t.bddl_file}"] = \
            hashlib.sha256(final_content.encode()).hexdigest()

    # ---- 3) init：整目录批量（官方函数语义）；全部齐则跳过 ----
    missing = [t for t in tasks
               if not os.path.exists(f"{out_init_dir}/{t.name}.pruned_init")]
    if missing:
        np.random.seed(SEED)
        print(f"[{suite}] generating {len(missing)} init files "
              f"(batch over dir; ~{len(missing)*NUM_INITS} env builds)",
              flush=True)
        call = {"bddl_base_dir": out_bddl_dir, "output_dir": out_init_dir,
                "num_inits": NUM_INITS}
        call = {k: v for k, v in call.items() if k in gen_params}
        gen.generate_init_states(**call)
    for i in range(10):
        t = tasks[i]
        p = f"{out_init_dir}/{t.name}.pruned_init"
        manifest["files"][f"init_files/{suite}_env/{t.name}.pruned_init"] = \
            hashlib.sha256(open(p, "rb").read()).hexdigest() \
            if os.path.exists(p) else None
    print(f"[{suite}] SUITE_DONE", flush=True)

manifest["files_nonnull"] = sum(
    1 for v in manifest["files"].values() if v is not None)
json.dump(manifest, open(manifest_path, "w"), indent=1)
n = len(manifest["files"])
none_n = sum(1 for v in manifest["files"].values() if v is None)
print(f"ENV_GEN_DONE files={n} missing={none_n} "
      f"vacuous={len(manifest['vacuous_cases'])}", flush=True)
