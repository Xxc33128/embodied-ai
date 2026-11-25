#!/usr/bin/env python3
"""
checks/run_gate_tests.py — N3.1 最小测试 T1..T7（计划 §N2 + 审查 N3.1 M6/M7）

T1  正常 G0b 输入                     -> PASS，严格 24 行
T2  删除任意一个 state 字段           -> FAIL
T3  空 NPZ                           -> FAIL
T4  G0c 缺 actual target             -> FAIL
T5  28-name metadata 走真实 validation path 且失败  -> FAIL（ValueError）
T6  cycle 0 对齐、cycle 1 分叉        -> 接口 G0c PASS，R4-A diagnostic 有差异(cycle1)
T7  G0a time input（golden 缺失 -> FAIL，不允许静默 skip）

路径策略（N3.1-P1.3）：
  - 优先 ZIP 根结构（scripts/ + checks/ + raw_evidence/ 同级）→ 全包内相对路径；
  - 仓库 dev 模式（scripts/experiments/checks/）→ 仓库资产路径；
  - 任一台模式下缺 golden -> T7 直接 FAIL（不是 skip）。
输出同时写 stderr/console 与 UTF-8 文件的 test_results.txt（若指定 --out）。
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ZIPROOT = HERE.parent  # checks/ 的上级
OUT_FILE = None


def log(msg=""):
    print(msg, flush=True)
    if OUT_FILE is not None:
        with open(OUT_FILE, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")


def locate_paths():
    """返回 (scripts_dir, policy, motion, config, golden) 路径，全部包内相对优先。"""
    if (ZIPROOT / "raw_evidence" / "configs_used").exists() and (ZIPROOT / "scripts").exists():
        a = ZIPROOT / "raw_evidence"
        return (ZIPROOT / "scripts",
                a / "configs_used" / "motion_anchor_obs_model_23500.onnx",
                a / "configs_used" / "dance_9.npz",
                a / "configs_used" / "mimic_dance_9.yaml",
                a / "isaac_g0_golden" / "isaac_golden_states.npz")
    # 仓库 dev：scripts/experiments/checks
    exp = HERE.parent
    repo = exp.parent.parent  # E:\humanoid-lab
    golden = Path(r"E:/sim2sim-week-2026-08-26/01_lab_interface_gate/aligned_v1/isaac_g0_golden/isaac_golden_states.npz")
    return (exp,
            repo / "deploy/policy/dance_9/motion_anchor_obs_model_23500.onnx",
            repo / "motions/dance_9.npz",
            repo / "deploy/src/era_rl_controller/configs/mimic_dance_9.yaml",
            golden)


SCRIPTS_DIR, POLICY, MOTION, CONFIG, GOLDEN = None, None, None, None, None

RESULTS = []
FAILS = []


def check(name, cond, info=""):
    RESULTS.append((name, bool(cond)))
    tag = "PASS" if cond else "FAIL"
    log(f"  [{tag}] {name} {info}")
    if not cond:
        FAILS.append(name)


class StubPolicy:
    joint_names = [f"joint_{i}" for i in range(29)]


def make_adapter_pair(tmp, n_states=4, n_fields=6, drop=None):
    states = ["id0_identity", "id1_yaw_p90", "id2_yaw_m90", "id3_roll_pitch"][:n_states]
    fields = ["joint_pos", "joint_vel", "root_pos", "root_quat", "root_lin_vel_w", "root_ang_vel_b"][:n_fields]
    ia, ma = {}, {}
    v = 0.0
    for s in states:
        for f in fields:
            dim = 29 if f in ("joint_pos", "joint_vel") else (4 if f == "root_quat" else 3)
            ia[f"state_{s}_{f}"] = np.full(dim, v, dtype=np.float32)
            ma[f"state_{s}_{f}"] = np.full(dim, v, dtype=np.float32)
            v += 0.001
    if drop is not None:
        ia.pop(drop, None)
        ma.pop(drop, None)
    p_ia = tmp / "adapter_isaac.npz"
    p_ma = tmp / "adapter_mujoco.npz"
    np.savez(p_ia, **ia)
    np.savez(p_ma, **ma)
    return p_ia, p_ma


def make_trace_pair(tmp, div_cycle=1, drop=None, cycles=5):
    keys = ["obs(770)", "action(29)", "q_des(29)", "actual_processed_q_target(29)", "q(29)", "dq(29)"]
    itr, mtr = {}, {}
    for fk in keys:
        dim = 770 if fk.startswith("obs") else 29
        itr[fk] = np.zeros((cycles, dim), dtype=np.float32)
        mtr[fk] = np.zeros((cycles, dim), dtype=np.float32)
    itr["policy_step"] = np.arange(cycles)
    mtr["policy_step"] = np.arange(cycles)
    if div_cycle is not None:
        for c in range(div_cycle, cycles):
            mtr["dq(29)"][c, 22] = 5.0
            mtr["q(29)"][c, 22] = 0.5
    if drop is not None:
        itr.pop(drop, None)
        mtr.pop(drop, None)
    np.savez(tmp / "isaac_startup.npz", **itr)
    np.savez(tmp / "mj_startup.npz", **mtr)
    return tmp / "isaac_startup.npz", tmp / "mj_startup.npz"


def run():
    tmp = Path(tempfile.mkdtemp(prefix="gate_tests_"))
    # ---- T1 ----
    log("T1 G0b 正常输入:")
    ia, ma = make_adapter_pair(tmp)
    class A1:
        isaac_adapter, mujoco_adapter, g0c_tol = str(ia), str(ma), 1e-4
    p, rows, _ = ci.gate_g0b(A1(), tmp, None, log)
    check("T1 G0b PASS", p, f"rows={len(rows)}")
    check("T1 恰好 24 行", len(rows) == 24)

    # ---- T2 ----
    log("T2 G0b 删字段:")
    ia2, ma2 = make_adapter_pair(tmp, drop="state_id0_identity_root_quat")
    class A2:
        isaac_adapter, mujoco_adapter, g0c_tol = str(ia2), str(ma2), 1e-4
    p2, rows2, _ = ci.gate_g0b(A2(), tmp, None, log)
    check("T2 G0b FAIL on missing key", not p2)
    check("T2 缺字段行存在", any(not r["pass"] for r in rows2 if r["field_name"] == "root_quat"))

    # ---- T3 ----
    log("T3 空 NPZ:")
    empty = tmp / "empty.npz"
    np.savez(empty, state_id0_identity_joint_pos=np.zeros((0,), dtype=np.float32))
    class A3:
        isaac_adapter = mujoco_adapter = str(empty)
        g0c_tol = 1e-4
    p3, rows3, _ = ci.gate_g0b(A3(), tmp, None, log)
    check("T3 G0b FAIL on empty array", not p3)

    # ---- T4 ----
    log("T4 G0c 缺 actual_processed_q_target:")
    is4, mj4 = make_trace_pair(tmp, drop="actual_processed_q_target(29)")
    class A4:
        isaac_startup, mujoco_startup, g0c_tol, r4a_cycles = str(is4), str(mj4), 1e-4, 5
    p4, rows4, det4 = ci.gate_g0c_startup(A4(), tmp, StubPolicy(), log)
    check("T4 G0c FAIL on missing key", not p4)
    check("T4 missing-key detail", any(d["kind"] == "missing-key" for d in det4))

    # ---- T5 (N3.1-M6)：真实 validation path 拒绝 28-name ----
    log("T5 28-name metadata -> validate_policy_metadata FAIL:")
    from run_mujoco_onnx import validate_policy_metadata  # noqa: E402
    names28 = [f"j{i}" for i in range(28)]
    arrays29 = np.zeros(29, dtype=np.float32)
    try:
        validate_policy_metadata(names28, arrays29, arrays29, arrays29, arrays29)
        check("T5 28-name 被拒绝", False, "未抛 ValueError")
    except ValueError as e:
        check("T5 28-name 被拒绝", True, f"ValueError: {str(e)[:60]}")
    # 29-name 正常
    names29 = validate_policy_metadata([f"j{i}" for i in range(29)], arrays29, arrays29, arrays29, arrays29)
    check("T5 29-name 通过", len(names29) == 29)

    # ---- T6 ----
    log("T6 cycle0 对齐 cycle1 分叉:")
    is6, mj6 = make_trace_pair(tmp, div_cycle=1)
    class A6:
        isaac_startup, mujoco_startup, g0c_tol, r4a_cycles = str(is6), str(mj6), 1e-4, 5
    p6, rows6, det6 = ci.gate_g0c_startup(A6(), tmp, StubPolicy(), log)
    check("T6 G0c startup PASS (cycle0 一致)", p6)
    s6, rows6b, det6b = ci.gate_r4a_prefix(A6(), tmp, StubPolicy(), log)
    check("T6 R4A equal=False", s6["equal"] is False)
    check("T6 first_dynamic_divergence_cycle=1", s6["first_divergence_cycle"] == 1)

    # ---- T7 (N3.1-M7)：golden 缺失 -> FAIL（不 skip）；存在 -> 跑 G0a ----
    log("T7 G0a time input:")
    if not GOLDEN.is_file():
        check("T7 golden 缺失 -> FAIL（不允许静默 skip）", False, f"golden 不存在: {GOLDEN}")
    else:
        from run_mujoco_onnx import MimicOnnxPolicy  # noqa: E402
        import yaml
        pol = MimicOnnxPolicy(str(POLICY), log=lambda s: None)
        motion = {k: np.load(MOTION)[k] for k in ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w"]}
        obs_cfg = yaml.safe_load(Path(CONFIG).read_text(encoding="utf-8"))["rl_model"]["observations"]
        class A7:
            golden = str(GOLDEN)
            atol = rtol = 1e-5
            compare_start_idx = 4
            g0c_tol = 1e-4
        p7, rows7, det7, extra7 = ci.gate_g0a(A7(), tmp, pol, motion, obs_cfg, log)
        check("T7 time_step_sensitivity 已报告", "time_step_sensitivity_max_abs" in extra7,
              f"sens={extra7.get('time_step_sensitivity_max_abs')}")
        check("T7 G0a 整体 PASS", p7, f"rows={len(rows7)}")

    n_pass = sum(1 for _, ok in RESULTS if ok)
    log("")
    log(f"=== 汇总 {n_pass}/{len(RESULTS)} ===")
    if FAILS:
        log("FAILED: " + "; ".join(FAILS))
        return 1
    return 0


if __name__ == "__main__":
    if "--out" in sys.argv:
        OUT_FILE = Path(sys.argv[sys.argv.index("--out") + 1])
        if OUT_FILE.exists():
            OUT_FILE.unlink()
    SCRIPTS_DIR, POLICY, MOTION, CONFIG, GOLDEN = locate_paths()
    sys.path.insert(0, str(SCRIPTS_DIR))
    import compare_interface as ci  # noqa: E402
    sys.exit(run())