#!/usr/bin/env python3
"""
compare_interface.py — Humanoid Lab 三层 G0 门禁 + R4-A 动力学前缀诊断（N2 加固版，2026-08-27）

分层：
  G0a policy-contract gate：同一组 canonical states 分别经过
       S1  golden obs（Isaac 侧导出的 full_isaac_observation_770）
       S2  deploy MimicObsBuilder（WSL 重放）
       S3  clean-room IndependentObsBuilder（本文件内独立实现，防共享实现缺陷）
      startup(0..3) 与 steady(4..) 分窗比较；字段级 atol/rtol；action 重推理使用 golden
      每帧 reference_time（真实 time input，不再固定 0.0）；并输出 time_step 敏感性探针。
  G0b simulator-adapter gate：--isaac-adapter / --mujoco-adapter 两份 npz（各 4 组注入状态：
       id0_identity / id1_yaw_p90 / id2_yaw_m90 / id3_roll_pitch），强制 4×6=24 行，
       缺失/空/shape 不一致直接 FAIL（不再静默 continue / all([]) 空 PASS）。
  G0c-startup-interface gate：--isaac-startup / --mujoco-startup 两份 policy_trace 只比较
       cycle 0（首次物理执行前的启动接口链），缺失字段直接 FAIL；阈值 g0c_tol=1e-4。
  R4-A closed-loop prefix（诊断，不计入接口门禁）：比较前 N 个策略周期的 obs/action/q_des/
       target/q/dq 逐周期最大差，输出 first_dynamic_divergence_cycle —— 用于定位动力学分叉起点。

输出：field_comparison.csv、interface_summary.json（含
       g0a_policy_contract_pass / g0b_simulator_adapter_pass / g0c_startup_interface_pass /
       overall_interface_gate_pass / r4a_closed_loop_prefix_equal / first_dynamic_divergence_cycle）。
       overall=false -> 退出码 1（供自动化门禁）。

用法（WSL .venv_lab_sim 或 Windows env51，均无 Mujoco 依赖）：
  python scripts/experiments/compare_interface.py \
      --policy deploy/policy/dance_9/motion_anchor_obs_model_23500.onnx \
      --motion deploy/policy/dance_9/dance_9.npz \
      --config deploy/src/era_rl_controller/configs/mimic_dance_9.yaml \
      --golden <aligned_v1>/isaac/isaac_golden_states.npz \
      [--isaac-adapter <npz>] [--mujoco-adapter <npz>] \
      [--isaac-startup <policy_trace.npz>] [--mujoco-startup <policy_trace.npz>] \
      --output <aligned_v1>/g0_report_v2/
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_mujoco_onnx import (  # noqa: E402
    MimicObsBuilder,
    MimicOnnxPolicy,
    matrix_from_quat,
    quat_inv,
    quat_mul,
)

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "deploy"
DEFAULT_CONFIG = DEPLOY / "src/era_rl_controller/configs/mimic_dance_9.yaml"

OBS_LAYOUT = [
    ("curr_motion_data", 58),
    ("motion_anchor_ori_b", 6),
    ("base_ang_vel", 3),
    ("qpos_rel", 29),
    ("qvel", 29),
    ("action", 29),
]


def slices_770():
    out, off = [], 0
    for name, dim in OBS_LAYOUT:
        out.append((name, off, off + dim * 5))
        off += dim * 5
    assert off == 770
    return out


def rmse(a, b):
    """修正版 RMSE = sqrt(mean((a-b)^2))（旧版误写成 sqrt(mean(a-b)^2)）。"""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def cos(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return None
    return float(np.dot(a, b) / (na * nb))


# ================================================================ clean-room 独立观测构造器（G0a 的 S3）
class IndependentObsBuilder:
    """不依赖 deploy 类；按计划 §4.4 的 770 布局独立实现（含独立四元数/矩阵/ring 逻辑）。"""

    def __init__(self, default_q, history_init="repeat-first"):
        self.default_q = np.asarray(default_q, dtype=np.float32)
        self.history_init = history_init
        self.buf = {n: np.zeros((5, d), dtype=np.float32) for n, d in OBS_LAYOUT}
        self.init_yaw = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.yaw_done = False
        self._started = False

    # ---- 独立数学（与 deploy 不同的公式路径）----
    @staticmethod
    def _qinv(q):
        q = np.asarray(q, dtype=np.float64)
        return np.array([q[0], -q[1], -q[2], -q[3]]) / float(np.dot(q, q))

    @staticmethod
    def _qmul(a, b):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                         w1*x2 + x1*w2 + y1*z2 - z1*y2,
                         w1*y2 - x1*z2 + y1*w2 + z1*x2,
                         w1*z2 + x1*y2 - y1*x2 + z1*w2])

    @staticmethod
    def _yaw(q):
        q = np.asarray(q, dtype=np.float64)
        w, x, y, z = q
        yaw = np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
        h = 0.5 * yaw
        r = np.array([np.cos(h), 0.0, 0.0, np.sin(h)])
        return r / np.linalg.norm(r)

    @staticmethod
    def _mat2(q):
        q = np.asarray(q, dtype=np.float64)
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])
        return R[:, :2]  # 与 deploy/isaac 相同：前两列 (3×2=6)

    # ---- ring（列表式，第一个元素为最旧；repeat-first 在首次 build 时对全部 ring 预填）----
    def _push(self, name, vec):
        row = np.asarray(vec, dtype=np.float32).ravel()
        assert row.shape == (dict(OBS_LAYOUT)[name],)
        buf = np.roll(self.buf[name], -1, axis=0)  # 丢最旧，新值放末尾（=deploy 时序）
        buf[-1] = row
        self.buf[name] = buf

    def build(self, root_quat, root_ang_vel, q, dq, prev_action, ref_jpos, ref_jvel, ref_anchor_quat):
        root_quat = np.asarray(root_quat, dtype=np.float64)
        if not self.yaw_done:
            self.yaw_done = True
            self.init_yaw = self._qmul(self._yaw(root_quat),
                                       self._qinv(self._yaw(ref_anchor_quat)))
        qrel = self._qmul(self._qmul(self._qinv(root_quat), self.init_yaw), np.asarray(ref_anchor_quat, dtype=np.float64))
        terms = {
            "curr_motion_data": np.concatenate([ref_jpos, ref_jvel]),
            "motion_anchor_ori_b": self._mat2(qrel).astype(np.float32).ravel(),
            "base_ang_vel": np.asarray(root_ang_vel, dtype=np.float32).ravel(),
            "qpos_rel": (np.asarray(q, dtype=np.float32) - self.default_q),
            "qvel": np.asarray(dq, dtype=np.float32).ravel(),
            "action": np.asarray(prev_action, dtype=np.float32).ravel(),
        }
        if not self._started:
            self._started = True
            for name, vec in terms.items():
                self.buf[name][:] = np.asarray(vec, dtype=np.float32).ravel()[None, :]  # repeat-first 预填
        for name, vec in terms.items():
            self._push(name, vec)
        return np.concatenate([self.buf[n].ravel() for n, _ in OBS_LAYOUT]).astype(np.float32)


# ================================================================ G0a
G0A_GOLDEN_REQUIRED_KEYS = [
    "policy_step", "reference_time", "root_quaternion_wxyz", "root_angular_velocity_body",
    "joint_position_in_policy_order", "joint_velocity_in_policy_order", "previous_action",
    "full_isaac_observation_770",
]


def gate_g0a(args, out_dir, policy, motion, obs_cfg, log):
    gold = np.load(args.golden, allow_pickle=True)
    missing_keys = [k for k in G0A_GOLDEN_REQUIRED_KEYS if k not in gold.files]
    N = int(gold["policy_step"].shape[0])
    rows, details = [], []
    if missing_keys:
        for k in missing_keys:
            rows.append({"pair": "G0a", "field_name": f"golden_missing_{k}", "pass": False,
                         "note": f"golden 缺 key: {k}"})
            details.append({"gate": "G0a", "kind": "missing-key", "key": k})
        log(f"G0a: FAIL golden 缺 {missing_keys}")
        return False, rows, details, {}
    if N == 0:
        rows.append({"pair": "G0a", "field_name": "golden_frames", "pass": False,
                     "note": "golden policy_step 帧数 = 0"})
        details.append({"gate": "G0a", "kind": "empty-golden"})
        return False, rows, details, {}

    root_quats = gold["root_quaternion_wxyz"].astype(np.float32)
    root_ang = [np.asarray(v, dtype=np.float32) for v in gold["root_angular_velocity_body"]]
    q_pol = [np.asarray(v, dtype=np.float32) for v in gold["joint_position_in_policy_order"]]
    dq_pol = [np.asarray(v, dtype=np.float32) for v in gold["joint_velocity_in_policy_order"]]
    prev_act = [np.asarray(v, dtype=np.float32) for v in gold["previous_action"]]
    ref_time = gold["reference_time"].astype(np.float32)  # 真实 time input（秒）
    # N3.1-P1.4：若 golden 已记录实际送入 ONNX 的 policy_time_input（runner motion frame index），
    # 则原样重放；否则用 reference_time（秒）并注明推断。
    time_input_source = "golden reference_time (seconds)"
    if "policy_time_input" in gold.files:
        policy_time_in = gold["policy_time_input"].astype(np.float32)
        if policy_time_in.shape[0] == N:
            ref_time = policy_time_in
            time_input_source = "golden policy_time_input (runner 实际传入值)"
    ref_t = np.arange(N, dtype=np.int64)
    obs_golden = gold["full_isaac_observation_770"].astype(np.float32)
    log(f"G0a golden obs present: {obs_golden.shape} (policy_step 0 = {int(gold['policy_step'][0])})")

    # S2：deploy builder 重放（与 runner 相同：先 repeat-first 预填，再逐帧 build）
    deploy = MimicObsBuilder(policy, motion, obs_cfg)
    deploy.prefill_repeat_first(root_quats[0], root_ang[0], q_pol[0], dq_pol[0])
    s2 = []
    for i in range(N):
        policy.action_buffer[:] = prev_act[i]
        o, _ = deploy.build_obs(root_quats[i], root_ang[i], q_pol[i], dq_pol[i])
        s2.append(o)
        deploy.advance()
    s2 = np.stack(s2).astype(np.float32)

    # S3：clean-room 独立 builder
    indep = IndependentObsBuilder(policy.default_q)
    ref_jpos = motion["joint_pos"].astype(np.float32)
    ref_jvel = motion["joint_vel"].astype(np.float32)
    ref_aq = motion["body_quat_w"].astype(np.float32)
    s3 = []
    for i in range(N):
        o = indep.build(root_quats[i], root_ang[i], q_pol[i], dq_pol[i], prev_act[i],
                        ref_jpos[ref_t[i]], ref_jvel[ref_t[i]], ref_aq[ref_t[i], policy.anchor_body_id])
        s3.append(o)
    s3 = np.stack(s3).astype(np.float32)

    pairs = [("s2_deploy_wsl", s2), ("s1_golden_isaac", obs_golden)]
    atol, rtol = args.atol, args.rtol

    # action 重推理：使用 golden 每帧真实 reference_time（T7：不再固定 0.0）
    act2 = np.stack([policy.infer(o, float(ref_time[i])) for i, o in enumerate(s2)]).astype(np.float32)
    act3 = np.stack([policy.infer(o, float(ref_time[i])) for i, o in enumerate(s3)]).astype(np.float32)
    # time_step 敏感性探针：同一 obs 用 t=0 vs t=ref 重推理的最大差（判断 ONNX time 输入是否参与输出）
    probe_max = 0.0
    for i in range(min(5, N)):
        a0 = policy.infer(s3[i], 0.0)
        a1 = policy.infer(s3[i], float(ref_time[i]))
        probe_max = max(probe_max, float(np.max(np.abs(a0 - a1))))

    def _compare_window(pname, ps, r, lo, hi, kind):
        g = ps[lo:hi]
        rr = r[lo:hi]
        for name, s, e in slices_770():
            gs, rs = g[:, s:e], rr[:, s:e]
            dmax = float(np.max(np.abs(gs - rs))) if gs.size else 0.0
            tol = atol + rtol * np.abs(gs)
            bad = np.abs(gs - rs) > tol
            n_bad = int(bad.sum())
            first_bad_frame = -1
            if n_bad:
                fr = np.where(bad.any(axis=1))[0]
                first_bad_frame = int(fr[0]) + lo
            passed = n_bad == 0 and gs.size > 0
            rows.append({
                "pair": f"{pname}_{kind}", "field_name": name, "start_index": s, "end_index": e - 1,
                "n_bad_entries": n_bad, "max_abs_error": round(dmax, 8), "rmse": round(rmse(gs, rs), 8),
                "cosine_similarity": "NA" if cos(gs, rs) is None else round(cos(gs, rs), 8),
                "first_bad_frame": first_bad_frame, "threshold": None, "first_bad_cycle": first_bad_frame,
                "expected": None, "actual": None, "index": None, "joint_name": None,
                "note": "" if passed else "obs-block",
                "pass": passed,
            })
            if not passed:
                details.append({"gate": "G0a", "pair": pname, "field": name, "kind": kind,
                                "first_bad_cycle": first_bad_frame, "n_bad_entries": n_bad, "max_abs": dmax,
                                "threshold": None})

    for pname, ps in pairs:
        _compare_window(pname, ps, s3, 0, min(4, N), "startup")
        _compare_window(pname, ps, s3, args.compare_start_idx, N, "steady")

    # action / q_des（S2 vs S3，全帧）
    v = slice(None)
    a_max = float(np.max(np.abs(act2[v] - act3[v])))
    q2 = act2 * policy.action_scale + policy.default_q
    q3 = act3 * policy.action_scale + policy.default_q
    q_max = float(np.max(np.abs(q2[v] - q3[v])))
    act_pass = a_max <= max(atol * 10, 1e-4)
    q_pass = q_max <= max(atol * 10, 1e-4)
    rows.append({"pair": "action(s2_vs_s3)", "field_name": "raw_action", "start_index": -1, "end_index": -1,
                 "n_bad_entries": 0 if act_pass else 1, "max_abs_error": round(a_max, 8),
                 "rmse": round(rmse(act2[v], act3[v]), 8), "cosine_similarity": "NA",
                 "first_bad_frame": -1, "threshold": max(atol * 10, 1e-4), "first_bad_cycle": -1,
                 "expected": None, "actual": None, "index": None, "joint_name": None,
                 "note": "", "pass": act_pass})
    rows.append({"pair": "q_des(s2_vs_s3)", "field_name": "q_des", "start_index": -1, "end_index": -1,
                 "n_bad_entries": 0 if q_pass else 1, "max_abs_error": round(q_max, 8),
                 "rmse": round(rmse(q2[v], q3[v]), 8), "cosine_similarity": "NA",
                 "first_bad_frame": -1, "threshold": max(atol * 10, 1e-4), "first_bad_cycle": -1,
                 "expected": None, "actual": None, "index": None, "joint_name": None,
                 "note": "", "pass": q_pass})
    if not act_pass:
        details.append({"gate": "G0a", "kind": "action", "max_abs": a_max, "threshold": max(atol * 10, 1e-4)})
    if not q_pass:
        details.append({"gate": "G0a", "kind": "q_des", "max_abs": q_max, "threshold": max(atol * 10, 1e-4)})

    assert len(rows) > 0, "G0a 输出行为空（不应发生）"
    g0a_pass = len(missing_keys) == 0 and all(r["pass"] for r in rows)
    extra = {"time_step_sensitivity_max_abs": probe_max,
             "action_reinference_time_input": time_input_source,
             "action_reinference_time_fixed": False,
             "time_input_note": "P1-4: runner 实传 motion frame index；golden 带 policy_time_input 时原样重放，"
                                 f"本包 golden 无该字段，用 {time_input_source}。sensitivity=0 → 对本模型无差别。"}
    log(f"G0a: {'PASS' if g0a_pass else 'FAIL'} rows={len(rows)} bad={sum(1 for r in rows if not r['pass'])} "
        f"time_sens={probe_max:.2e}")
    return g0a_pass, rows, details, extra


# ================================================================ G0b
EXPECTED_ADAPTER_STATES = ["id0_identity", "id1_yaw_p90", "id2_yaw_m90", "id3_roll_pitch"]
ADAPTER_FIELDS = ["joint_pos", "joint_vel", "root_pos", "root_quat", "root_lin_vel_w", "root_ang_vel_b"]


def _split_adapter_key(k):
    """state_id0_identity_root_quat -> ('id0_identity', 'root_quat')；字段名含下划线，须从後缀匹配。"""
    s = k[len("state_"):]
    for f in sorted(ADAPTER_FIELDS, key=len, reverse=True):
        if s.endswith(f):
            return s[: -len(f) - 1], f
    return s, ""


def gate_g0b(args, out_dir, policy, log):
    ia = np.load(args.isaac_adapter, allow_pickle=True)
    ma = np.load(args.mujoco_adapter, allow_pickle=True)
    expected_keys = [f"state_{s}_{f}" for s in EXPECTED_ADAPTER_STATES for f in ADAPTER_FIELDS]
    rows, details = [], []

    def _add_fail(pair, fld, note, kind):
        rows.append({"pair": pair, "field_name": fld, "pass": False, "note": note,
                     "start_index": -1, "end_index": -1, "max_abs_error": None, "rmse": None,
                     "cosine_similarity": "NA", "first_bad_frame": -1, "threshold": 1e-4,
                     "first_bad_cycle": -1, "expected": None, "actual": None, "index": None,
                     "joint_name": None})
        details.append({"gate": "G0b", "state": pair, "field": fld, "kind": kind})

    for k in expected_keys:
        state, fld = _split_adapter_key(k)
        if k not in ia.files or k not in ma.files:
            _add_fail(state, fld, f"missing key {k}", "missing-key")
            continue
        a = np.asarray(ia[k], dtype=np.float64)
        b = np.asarray(ma[k], dtype=np.float64)
        if a.size == 0 or b.size == 0:
            _add_fail(state, fld, "empty array", "empty")
            continue
        if a.shape != b.shape:
            _add_fail(state, fld, f"shape {a.shape} vs {b.shape}", "shape-mismatch")
            continue
        dmax = float(np.max(np.abs(a - b)))
        tol = 1e-4  # 回读/类型转换阈值（预注册）
        ok = dmax <= tol
        rows.append({"pair": state, "field_name": fld, "pass": ok, "note": "",
                     "start_index": -1, "end_index": len(a) - 1,
                     "max_abs_error": round(dmax, 8), "rmse": round(rmse(a, b), 8),
                     "cosine_similarity": "NA" if np.linalg.norm(a) < 1e-10 else round(cos(a, b), 8),
                     "first_bad_frame": -1, "threshold": tol, "first_bad_cycle": -1,
                     "expected": None, "actual": None, "index": None, "joint_name": None})
        if not ok:
            details.append({"gate": "G0b", "state": state, "field": fld, "kind": "adapter", "max_abs": dmax,
                            "threshold": tol})
    extra_keys = (set(ia.files) | set(ma.files)) - set(expected_keys)
    if extra_keys:
        log(f"G0b: 额外 keys（不参与判定）: {sorted(extra_keys)[:8]} ...")
    g0b_pass = len(expected_keys) == 24 and len(rows) == 24 and all(r["pass"] for r in rows)
    log(f"G0b: {'PASS' if g0b_pass else 'FAIL'} rows={len(rows)} (期望 24) prefixes={EXPECTED_ADAPTER_STATES}")
    return g0b_pass, rows, details


# ================================================================ G0c-startup-interface（只查 cycle 0）
G0C_FIELDS = {"obs(770)": "obs", "action(29)": "action", "q_des(29)": "q_des",
              "actual_processed_q_target(29)": "target", "q(29)": "q", "dq(29)": "dq"}


def _joint_name(policy, flat_idx, a_shape):
    if policy is None or not hasattr(policy, "joint_names"):
        return None
    dims = len(a_shape)
    if (dims == 3 and a_shape[2] == 29) or (dims == 2 and a_shape[1] == 29):
        return str(policy.joint_names[flat_idx])
    return None


def gate_g0c_startup(args, out_dir, policy, log):
    itr = np.load(args.isaac_startup, allow_pickle=True)
    mtr = np.load(args.mujoco_startup, allow_pickle=True)
    tol = args.g0c_tol
    rows, details = [], []
    for fk, name in G0C_FIELDS.items():
        if fk not in itr.files or fk not in mtr.files:
            rows.append({"pair": "G0c-startup", "field_name": name, "pass": False,
                         "note": f"missing key: {fk}", "start_index": 0, "end_index": 0,
                         "max_abs_error": None, "rmse": None, "cosine_similarity": "NA",
                         "first_bad_frame": -1, "threshold": tol, "first_bad_cycle": -1,
                         "expected": None, "actual": None, "index": None, "joint_name": None})
            details.append({"gate": "G0c-startup", "field": name, "kind": "missing-key", "key": fk})
            continue
        a = np.asarray(itr[fk][:1], dtype=np.float64)  # 仅 cycle 0
        b = np.asarray(mtr[fk][:1], dtype=np.float64)
        if a.shape != b.shape:
            rows.append({"pair": "G0c-startup", "field_name": name, "pass": False,
                         "note": f"shape {a.shape} vs {b.shape}", "start_index": 0, "end_index": 0,
                         "max_abs_error": None, "rmse": None, "cosine_similarity": "NA",
                         "first_bad_frame": -1, "threshold": tol, "first_bad_cycle": -1,
                         "expected": None, "actual": None, "index": None, "joint_name": None})
            details.append({"gate": "G0c-startup", "field": name, "kind": "shape"})
            continue
        dmax = float(np.max(np.abs(a - b)))
        per_cycle = [dmax]
        ok = dmax <= tol
        flat_bad = int(np.argmax(np.abs(a - b))) if not ok else -1
        exp = act = idx = jn = None
        if not ok:
            idx = flat_bad % (a.shape[-1] if a.ndim == 2 else a.shape[-1])
            exp = float(a.reshape(-1)[flat_bad])
            act = float(b.reshape(-1)[flat_bad])
            jn = _joint_name(policy, idx, a.shape)
            details.append({"gate": "G0c-startup", "field": name, "kind": "startup",
                            "first_bad_cycle": 0, "index": idx, "joint_name": jn, "expected": exp,
                            "actual": act, "max_abs": dmax, "threshold": tol, "per_cycle_max": per_cycle})
        rows.append({"pair": "G0c-startup", "field_name": name, "pass": ok, "note": "",
                     "start_index": 0, "end_index": 0, "max_abs_error": round(dmax, 8),
                     "rmse": round(rmse(a, b), 8), "cosine_similarity": "NA",
                     "first_bad_frame": -1, "threshold": tol, "first_bad_cycle": 0 if not ok else -1,
                     "expected": exp, "actual": act, "index": idx, "joint_name": jn,
                     "per_cycle_max": per_cycle})
    g0c_pass = len(rows) == len(G0C_FIELDS) and all(r["pass"] for r in rows)
    log(f"G0c-startup: {'PASS' if g0c_pass else 'FAIL'} (仅 cycle 0, tol={tol}) fields={len(rows)}/{len(G0C_FIELDS)}")
    return g0c_pass, rows, details


# ================================================================ R4-A：闭环前缀动力学诊断（不计入门禁）
def gate_r4a_prefix(args, out_dir, policy, log):
    itr = np.load(args.isaac_startup, allow_pickle=True)
    mtr = np.load(args.mujoco_startup, allow_pickle=True)
    n = min(args.r4a_cycles, int(itr["policy_step"].shape[0]), int(mtr["policy_step"].shape[0]))
    tol = args.g0c_tol
    rows, details = [], []
    first_div_cycle, first_div_field = None, None
    for fk, name in G0C_FIELDS.items():
        if fk not in itr.files or fk not in mtr.files:
            rows.append({"pair": "R4A", "field_name": name, "pass": False,
                         "note": f"missing key: {fk}", "threshold": tol,
                         "first_bad_cycle": -1, "per_cycle_max": "NA"})
            continue
        a = np.asarray(itr[fk][:n], dtype=np.float64)
        b = np.asarray(mtr[fk][:n], dtype=np.float64)
        if a.shape != b.shape:
            rows.append({"pair": "R4A", "field_name": name, "pass": False, "note": "shape mismatch",
                         "threshold": tol, "first_bad_cycle": -1, "per_cycle_max": "NA"})
            continue
        per_cycle = np.max(np.abs(a - b), axis=1).tolist()
        bad = np.array(per_cycle) > tol
        first_bad = int(np.where(bad)[0][0]) if bad.any() else -1
        exp = act = idx = jn = None
        if first_bad >= 0:
            dm = np.abs(a - b)[first_bad]
            flat = int(np.argmax(dm))
            idx = flat % a.shape[-1]
            row_flat = first_bad * a.shape[-1] + flat  # 多周期数组的全局拍平索引
            exp = float(a.reshape(-1)[row_flat])
            act = float(b.reshape(-1)[row_flat])
            jn = _joint_name(policy, idx, a.shape)
            details.append({"gate": "R4A", "field": name, "kind": "dynamics",
                            "first_bad_cycle": first_bad, "index": idx, "joint_name": jn,
                            "expected": exp, "actual": act,
                            "max_abs": float(np.max(per_cycle)), "threshold": tol, "per_cycle_max": per_cycle})
            if first_div_cycle is None or first_bad < first_div_cycle:
                first_div_cycle, first_div_field = first_bad, name
        rows.append({"pair": "R4A", "field_name": name, "pass": first_bad < 0, "note": "",
                     "max_abs_error": round(float(np.max(per_cycle)) if per_cycle else 0.0, 8),
                     "rmse": round(rmse(a, b), 8), "cosine_similarity": "NA",
                     "first_bad_frame": first_bad, "threshold": tol,
                     "first_bad_cycle": first_bad, "expected": exp, "actual": act,
                     "index": idx, "joint_name": jn, "per_cycle_max": per_cycle})
    equal = len(rows) == len(G0C_FIELDS) and all(r["pass"] for r in rows)
    summary = {"equal": equal,
               "first_divergence_cycle": first_div_cycle if first_div_cycle is not None else -1,
               "first_divergence_field": first_div_field}
    log(f"R4A: equal={equal} first_divergence_cycle={summary['first_divergence_cycle']} "
        f"field={summary['first_divergence_field']} cycles={n} tol={tol}")
    return summary, rows, details


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description="Lab 三层 G0 门禁 + R4-A（N2 加固版）")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--golden", default=None, help="isaac_golden_states.npz（G0a）")
    ap.add_argument("--isaac-adapter", default=None, help="Isaac adapter readback npz（G0b）")
    ap.add_argument("--mujoco-adapter", default=None, help="MuJoCo adapter readback npz（G0b）")
    ap.add_argument("--isaac-startup", default=None, help="Isaac policy_trace.npz（G0c/R4A）")
    ap.add_argument("--mujoco-startup", default=None, help="MuJoCo policy_trace.npz（G0c/R4A）")
    ap.add_argument("--output", required=True)
    ap.add_argument("--compare-start-idx", type=int, default=4, help="G0a steady-state 比较起点")
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--g0c-tol", type=float, default=1e-4)
    ap.add_argument("--r4a-cycles", type=int, default=5, help="R4-A 前缀诊断比较的周期数")
    ap.add_argument("--partial-ok", action="store_true",
                    help="允许只跑部分门禁（默认：缺任何一组输入直接退出码 2）")
    args = ap.parse_args()

    run_gate_groups = [bool(args.golden), bool(args.isaac_adapter and args.mujoco_adapter),
                       bool(args.isaac_startup and args.mujoco_startup)]
    if not all(run_gate_groups) and not args.partial_ok:
        ap.error("必须同时提供 G0a(golden)/G0b(adapters)/G0c(startups) 三组输入，或用 --partial-ok 显式允许部分")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    def log(m=""):
        print(m, flush=True)

    policy = MimicOnnxPolicy(args.policy, log=log)
    motion = {k: np.load(args.motion)[k] for k in ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w"]}
    obs_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["rl_model"]["observations"]

    results = {}
    all_rows = []
    details = []
    time_info = {}
    if args.golden:
        p, rows, det, extra = gate_g0a(args, out_dir, policy, motion, obs_cfg, log)
        results["g0a_policy_contract_pass"] = p
        all_rows += rows
        details += det
        time_info = extra
    if args.isaac_adapter and args.mujoco_adapter:
        p, rows, det = gate_g0b(args, out_dir, policy, log)
        results["g0b_simulator_adapter_pass"] = p
        all_rows += rows
        details += det
    if args.isaac_startup and args.mujoco_startup:
        p, rows, det = gate_g0c_startup(args, out_dir, policy, log)
        results["g0c_startup_interface_pass"] = p
        all_rows += rows
        details += det
        r4a, r4a_rows, r4a_det = gate_r4a_prefix(args, out_dir, policy, log)
        results["r4a_closed_loop_prefix_equal"] = r4a["equal"]
        results["first_dynamic_divergence_cycle"] = r4a["first_divergence_cycle"]
        results["first_dynamic_divergence_field"] = r4a["first_divergence_field"]
        all_rows += r4a_rows
        details += r4a_det

    run_gates = [k for k in ("g0a_policy_contract_pass", "g0b_simulator_adapter_pass",
                             "g0c_startup_interface_pass") if k in results]
    if not run_gates:
        raise SystemExit("至少需要一组完整输入（--golden 或 adapter 对或 startup 对，且 --partial-ok）")
    overall = all(results[k] for k in run_gates)
    results["overall_interface_gate_pass"] = overall
    results["gates_run"] = run_gates
    results["thresholds"] = {"atol": args.atol, "rtol": args.rtol, "g0c_tol": args.g0c_tol,
                             "r4a_cycles": args.r4a_cycles,
                             "action_qdes_tol": max(args.atol * 10, 1e-4)}
    results["time_input"] = time_info or {"note": "G0a 未运行，无 time input"}
    results["details"] = details

    if all_rows:
        fieldnames = []
        for r in all_rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(out_dir / "field_comparison.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
    with open(out_dir / "interface_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log("=== gate summary ===")
    for k in run_gates:
        log(f"{k} = {results[k]}")
    log(f"overall_interface_gate_pass = {overall}")
    if "r4a_closed_loop_prefix_equal" in results:
        log(f"r4a_closed_loop_prefix_equal = {results['r4a_closed_loop_prefix_equal']} "
            f"(first_dynamic_divergence_cycle={results['first_dynamic_divergence_cycle']})")
    log(f"written: {out_dir/'field_comparison.csv'} {out_dir/'interface_summary.json'}")
    if not overall:
        for d in details:
            log(f"  BAD: {d}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()