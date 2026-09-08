#!/usr/bin/env python3
"""run_mujoco_step_test.py — N5-R 单关节 step + parity（MuJoCo 侧，smoke v2 修订）

v2 相对 v1 的变化（依据 2026-08-28 smoke v1 验收）：
- CLEAN_ISOLATED 改用**真硬约束模型**（mj_isolation_model.py 生成）：
  root free joint 删除（pelvis 固定世界），其余 28 hinge 删除（body frame 折叠 q0），
  仅被测 hinge 保留为唯一动态 DOF → other-joint/root 漂移在动力学上不存在。
  不再依赖"每步写回"投影近似。
- 输出保持 29 维契约：被测关节用真实值，其余 28 维 = q0 / dq=0 / tau=0（锁定语义）。
- 记录 other_joint_drift_max（=0，无 DOF）与 root_drift_max（xpos 数值残余）。
- --isolation coupled 保留 v1 的官方模型+写回路径（工程补充协议）。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "deploy"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_mujoco_onnx import (  # noqa: E402
    load_contract,
    map_by_name,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_STEP_RAD = 0.10
FALLBACK_STEP_RAD = 0.05


def sha256_bytes(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_intervention_xml(src_xml: Path, joint: str, armature: float, out_root: Path):
    """coupled 模式：复制官方 XML，仅在被测关节 <joint> 行注入 armature（不改其他）。"""
    text = src_xml.read_text(encoding="utf-8")
    marker = f'name="{joint}"'
    idx = text.find(marker)
    if idx < 0:
        raise ValueError(f"XML 中未找到关节 {joint}")
    line_start = text.rfind("\n", 0, idx) + 1
    line_end = text.find("\n", idx)
    line = text[line_start:line_end]
    if "armature=" in line:
        raise ValueError(f"关节行已有 armature：{line}")
    stripped = line.strip()
    if not stripped.endswith("/>"):
        raise ValueError(f"关节行不以 /> 结尾（带子元素？）：{line}")
    new_line = stripped[:-2].rstrip() + f' armature="{armature}"/>'
    new_text = text[:line_start] + new_line + text[line_end:]
    out = out_root / f"{src_xml.stem}_INTERVENTION_ARMATURE_MATCHED_{joint}.xml"
    out.write_text(new_text, encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--joint", required=True)
    ap.add_argument("--mode", choices=["parity", "step"], required=True)
    ap.add_argument("--step-rad", type=float, default=DEFAULT_STEP_RAD)
    ap.add_argument("--armature-override", type=float, default=None,
                    help="clean 模式：生成 ISOLATED_MATCHED（被测 armature=此值）；coupled 模式：旧 intervention 副本")
    ap.add_argument("--initial-state", required=True)
    ap.add_argument("--duration", type=float, default=0.5)
    ap.add_argument("--isolation", choices=["coupled", "clean"], default="clean",
                    help="clean=真硬约束隔离模型（默认）；coupled=官方模型+substep 写回（工程补充）")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "stdout.log", "w", encoding="utf-8")
    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    import mujoco
    from mj_isolation_model import build as build_isolation_xml, self_check as iso_self_check
    c = load_contract(args.config)
    cfg_path = Path(args.config) if Path(args.config).is_file() else DEPLOY / args.config
    src_xml = Path(c["xml_path"])
    s2s = c["sim2sim_order"]
    from run_mujoco_onnx import MimicOnnxPolicy  # noqa: E402
    pol = MimicOnnxPolicy(str(c["policy_path"]), log=log)
    joint_names = list(pol.joint_names)
    assert args.joint in joint_names, f"{args.joint} 不在 policy joint_names"

    # ---------------- 模型选择 ----------------
    clean = args.isolation == "clean"
    xml_official = src_xml
    condition = "ARMATURE_MATCHED" if args.armature_override is not None else "NATIVE"
    if clean:
        xml_used = build_isolation_xml(src_xml, args.joint, Path(args.initial_state),
                                       src_xml.parent, args.armature_override, check=True)
        log(f"ISOLATED_MODEL: {xml_used.name}（固定根+唯一 DOF={args.joint}）")
    else:
        xml_used = src_xml
        if args.armature_override is not None:
            xml_used = make_intervention_xml(src_xml, args.joint, args.armature_override, src_xml.parent)
            log(f"coupled intervention XML: {xml_used.name}")

    m = mujoco.MjModel.from_xml_path(str(xml_used))
    d = mujoco.MjData(m)
    physics_dt = float(m.opt.timestep)

    contacts_off_method = "not_off"
    if clean:
        m.geom_contype[:] = 0
        m.geom_conaffinity[:] = 0
        contacts_off_method = "geom_contype/conaffinity=0 (runtime model-level)"
        m.opt.gravity[:] = np.zeros(3)
        log("CLEAN: gravity=0 + geom_contype=0（运行时，XML 未改）")

    # ---------------- 关节/执行器映射 ----------------
    hinge_ids = [jid for jid in range(m.njnt) if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE]
    hinge_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) for jid in hinge_ids]
    name2jid = dict(zip(hinge_names, hinge_ids))
    act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    assert args.joint in hinge_names, "被测关节不在模型中（isolation 模型应只有被测 hinge）"
    jid = name2jid[args.joint]
    did = int(m.jnt_dofadr[jid])
    arm_read = float(m.dof_armature[did])

    # ---------------- 初态 ----------------
    init = np.load(args.initial_state)
    assert list(init["joint_names"]) == list(joint_names)
    q0_pol = np.asarray(init["joint_pos"], dtype=np.float64)
    root_pos0 = np.asarray(init["root_pos"], dtype=np.float64)
    root_quat0 = np.asarray(init["root_quat"], dtype=np.float64)
    jidx = joint_names.index(args.joint)
    q0_j = float(q0_pol[jidx])
    others_pol = [i for i in range(29) if i != jidx]

    if clean:
        # 1-DOF 模型：只有被测 hinge
        assert m.njnt == 1 and m.nu == 1, f"isolation 模型期望 1 joint/1 actuator: {m.njnt}/{m.nu}"
        d.qpos[:] = q0_j
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        q0_s2s = None
        others_s2s_idx = []
        need_lock = False
        root_free = False
    else:
        # 官方模型：29 hinge + free root
        assert set(hinge_names) == set(s2s) and set(act_names) == set(s2s)
        def s2j(arr): return map_by_name(arr, s2s, hinge_names, "s2s->j")
        def s2a(arr): return map_by_name(arr, s2s, act_names, "s2s->a")
        def j2s(arr): return map_by_name(arr, hinge_names, s2s, "j->s2s")
        d.qpos[:7] = np.concatenate([root_pos0, root_quat0])
        d.qpos[7:] = s2j(map_by_name(q0_pol, joint_names, s2s, "q0->s2s"))
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        q0_s2s = map_by_name(q0_pol, joint_names, s2s, "q0->s2s")
        others_s2s_idx = [list(s2s).index(n) for i, n in enumerate(joint_names) if i in others_pol]
        need_lock = True
        root_free = True

    # 控制参数（policy order -> 当前模型 hinge/actuator 序；isolation 下仅被测）
    kp_pol = pol.kp.astype(np.float64)
    kd_pol = pol.kd.astype(np.float64)
    ctrl_range_j = m.actuator_ctrlrange[act_names.index(args.joint)].copy()

    # step 预检：触 range 回退
    jrange = np.array([m.jnt_range[jid][0], m.jnt_range[jid][1]])
    step_rad = args.step_rad
    fallback_decided = False
    if q0_j + step_rad > jrange[1] - 0.02 or q0_j + step_rad < jrange[0] + 0.02:
        log(f"预检：step {step_rad} 触 joint range {jrange} -> 回退 {FALLBACK_STEP_RAD}")
        step_rad = FALLBACK_STEP_RAD
        fallback_decided = True

    # ---------------- parity 模式 ----------------
    if args.mode == "parity":
        q_des_j = q0_j + step_rad
        q_pol29, dq_pol29 = _state29(m, d, clean, jidx, others_pol, q0_pol, j2s if not clean else None,
                                     s2s if not clean else None, joint_names, hinge_names)
        q_j = float(q_pol29[jidx])
        tau_u29 = (q_des_j - q_pol29[jidx]) * kp_pol[jidx] - dq_pol29[jidx] * kd_pol[jidx]
        tau_c29 = float(np.clip(tau_u29, ctrl_range_j[0], ctrl_range_j[1]))
        d.ctrl[:] = tau_c29
        mujoco.mj_forward(m, d)
        applied_j = float(d.actuator_force[0])
        _runner_sha_par = sha256_bytes(__file__)
        _canon_sha_par = sha256_bytes(args.initial_state)
        _xml_used_sha_par = sha256_bytes(xml_used)
        _xml_official_sha_par = sha256_bytes(xml_official)
        # 层 1 契约：29 维（锁定关节 tau=0）
        contract = {
            "engine": "mujoco", "mode": "pd_contract_at_common_state",
            "state": "q=q0, dq=0（静止初态，不推进物理）", "joint": args.joint,
            "condition": condition, "isolation": args.isolation, "isolation_model": clean,
            "step_rad_used": step_rad, "q_des_j": q_des_j, "q_j": q_j,
            "kp_j/kd_j": [float(kp_pol[jidx]), float(kd_pol[jidx])],
            "tau_unclipped_j": float(tau_u29), "tau_clipped_j": tau_c29,
            "ctrlrange_j": ctrl_range_j.tolist(),
            "tau_unclipped_pol(29)": (np.zeros(29, dtype=np.float64) if False else _tau29(tau_u29, jidx)).tolist(),
            "tau_clipped_pol(29)": _tau29(tau_c29, jidx).tolist(),
            "clip_triggered_any": bool(abs(tau_u29 - tau_c29) > 1e-9),
            "clip_triggered_joint": args.joint if abs(tau_u29 - tau_c29) > 1e-9 else None,
            "runner_sha256": _runner_sha_par,
            "model_used_sha256": _xml_used_sha_par,
            "canonical_npz_sha256": _canon_sha_par,
            "xml_used_sha256": _xml_used_sha_par,
            "xml_official_sha256": _xml_official_sha_par,
            "hashes": {"runner": _runner_sha_par, "model": _xml_used_sha_par, "canonical": _canon_sha_par, "initial_state": _canon_sha_par},
        }
        readback = {
            "engine": "mujoco", "mode": "engine_torque_readback",
            "advance": "mj_forward 一次（不推进时间），读 d.actuator_force",
            "joint": args.joint, "condition": condition, "isolation": args.isolation,
            "q_j": q_j, "dq_j": float(dq_pol29[jidx]),
            "applied_j_actuator_force": applied_j, "armature_j_readback": arm_read,
            "runner_sha256": _runner_sha_par,
            "model_used_sha256": _xml_used_sha_par,
            "canonical_npz_sha256": _canon_sha_par,
            "hashes": {"runner": _runner_sha_par, "model": _xml_used_sha_par, "canonical": _canon_sha_par},
        }
        (out_dir / "pd_contract_at_common_state.json").write_text(
            json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "engine_torque_readback.json").write_text(
            json.dumps(readback, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"parity: contract tau_u_j={tau_u29:.5f} tau_c_j={tau_c29:.5f} applied_j={applied_j:.5f} arm={arm_read}")
        logf.close()
        return

    # ---------------- step 模式 ----------------
    n_steps = int(round(args.duration / physics_dt))
    n_states = n_steps + 1
    t_switch = 0.10

    rec = {"t_state": [], "t_command_start": [], "q_pol(29)": [], "dq_pol(29)": [],
           "q_des_pol(29)": [], "tau_unclipped_pol(29)": [], "tau_clipped_pol(29)": [],
           "applied_pol(29)": [], "saturated(29)": [],
           "other_joint_drift_max": [], "root_drift_max": [], "ncon": [], "max_contact_force": []}

    def rec0(t, q29, dq29, qdes29, tau_u, tau_c, sat):
        rec["t_state"].append(t)
        rec["t_command_start"].append(t)
        rec["q_pol(29)"].append(q29.astype(np.float32))
        rec["dq_pol(29)"].append(dq29.astype(np.float32))
        rec["q_des_pol(29)"].append(np.asarray(qdes29, dtype=np.float32))
        rec["tau_unclipped_pol(29)"].append(np.asarray(tau_u, dtype=np.float32))
        rec["tau_clipped_pol(29)"].append(np.asarray(tau_c, dtype=np.float32))
        rec["applied_pol(29)"].append(np.zeros(29, dtype=np.float32))
        rec["saturated(29)"].append(np.asarray(sat, dtype=np.float32) if np.ndim(sat) else
                                     np.zeros(29, dtype=np.float32))
        rec["other_joint_drift_max"].append(0.0)
        rec["root_drift_max"].append(0.0)
        rec["ncon"].append(int(d.ncon))
        rec["max_contact_force"].append(0.0)

    def state29():
        if clean:
            q29 = q0_pol.copy(); dq29 = np.zeros(29)
            q29[jidx] = float(d.qpos[0]); dq29[jidx] = float(d.qvel[0])
            return q29, dq29
        q_pol = map_by_name(j2s(d.qpos[7:].copy()), s2s, joint_names, "q")
        dq_pol = map_by_name(j2s(d.qvel[6:].copy()), s2s, joint_names, "dq")
        return np.asarray(q_pol, dtype=np.float64), np.asarray(dq_pol, dtype=np.float64)

    def tau29(jval, q_j, dq_j):
        u = np.zeros(29)
        u[jidx] = (jval - q_j) * kp_pol[jidx] - dq_j * kd_pol[jidx]
        cc = float(np.clip(u[jidx], ctrl_range_j[0], ctrl_range_j[1]))
        return u, _tau29(cc, jidx)

    def anchor_lock():
        if not clean:
            d.qpos[0:7] = np.concatenate([root_pos0, root_quat0])
            d.qvel[0:6] = 0.0
            q_s2s = j2s(d.qpos[7:].copy())
            dq_s2s = j2s(d.qvel[6:].copy())
            q_s2s[others_s2s_idx] = q0_s2s[others_s2s_idx]
            dq_s2s[others_s2s_idx] = 0.0
            d.qpos[7:] = s2j(q_s2s)
            d.qvel[6:] = s2j(dq_s2s)

    # t=0 真初态
    q_pol29, dq_pol29 = state29()
    rec0(0.0, q_pol29, dq_pol29, q0_pol.copy(),
         np.zeros(29), np.zeros(29), np.zeros(29, dtype=bool))

    max_drift_others = 0.0
    max_root_drift = 0.0
    max_ncon = int(d.ncon)
    max_cf = 0.0
    step_saturated = False
    lo_j, hi_j = ctrl_range_j[0], ctrl_range_j[1]

    for k in range(1, n_states):
        t_state = k * physics_dt
        t_cmd = (k - 1) * physics_dt
        anchor_lock()
        q_des_j = q0_j + (step_rad if t_cmd >= t_switch - 1e-9 else 0.0)
        q_pol29, dq_pol29 = state29()
        u29, c29 = tau29(q_des_j, q_pol29[jidx], dq_pol29[jidx])
        d.ctrl[:] = c29[jidx]
        sat = bool(abs(u29[jidx] - c29[jidx]) > 1e-9)
        step_saturated = step_saturated or sat
        mujoco.mj_step(m, d)
        # post-step
        q_pol29b, dq_pol29b = state29()
        applied = np.zeros(29)
        applied[jidx] = float(d.actuator_force[0])
        ncon = int(d.ncon)
        cf = 0.0
        if ncon > 0:
            cf = float(np.max(np.linalg.norm(d.cforce[:ncon], axis=1)))
        max_ncon = max(max_ncon, ncon)
        max_cf = max(max_cf, cf)
        # drift（clean：无其他 DOF => 0；coupled：step 后瞬时）
        if clean:
            drift_others = 0.0
            pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            root_drift = float(max(np.linalg.norm(d.xpos[pid] - root_pos0),
                                   min(np.linalg.norm(d.xquat[pid] - root_quat0),
                                       np.linalg.norm(d.xquat[pid] + root_quat0))))
        else:
            q_now_s = j2s(d.qpos[7:].copy())
            drift_others = float(np.max(np.abs(q_now_s[others_s2s_idx] - q0_s2s[others_s2s_idx])))
            root_drift = float(max(np.linalg.norm(d.qpos[0:3] - root_pos0),
                                   np.linalg.norm(d.qpos[3:7] - root_quat0)))
        max_drift_others = max(max_drift_others, drift_others)
        max_root_drift = max(max_root_drift, root_drift)
        qdes29 = q0_pol.copy(); qdes29[jidx] = q_des_j
        rec["t_state"].append(t_state)
        rec["t_command_start"].append(t_cmd)
        rec["q_pol(29)"].append(q_pol29b.astype(np.float32))
        rec["dq_pol(29)"].append(dq_pol29b.astype(np.float32))
        rec["q_des_pol(29)"].append(np.asarray(qdes29, dtype=np.float32))
        rec["tau_unclipped_pol(29)"].append(u29.astype(np.float32))
        rec["tau_clipped_pol(29)"].append(c29.astype(np.float32))
        rec["applied_pol(29)"].append(applied.astype(np.float32))
        satarr = np.zeros(29, dtype=np.float32); satarr[jidx] = 1.0 if sat else 0.0
        rec["saturated(29)"].append(satarr)
        rec["other_joint_drift_max"].append(drift_others)
        rec["root_drift_max"].append(root_drift)
        rec["ncon"].append(ncon)
        rec["max_contact_force"].append(cf)

    lens = {kk: len(v) for kk, v in rec.items()}
    assert len(set(lens.values())) == 1, f"记录长度不一致: {lens}"
    n_rec = lens["t_state"]

    npz = {kk: np.asarray(v, dtype=(np.float32 if kk != "ncon" else np.int32)) for kk, v in rec.items()}
    np.savez_compressed(out_dir / "step_trace.npz", **npz)

    _runner_sha = sha256_bytes(__file__)
    _canon_sha = sha256_bytes(args.initial_state)
    _xml_official_sha = sha256_bytes(xml_official)
    _xml_used_sha = sha256_bytes(xml_used)
    manifest = {
        "engine": "mujoco", "mode": "step", "joint": args.joint, "condition": condition,
        "isolation": args.isolation, "isolation_model": clean,
        "root_constraint_method": ("isolated_model: free joint removed, pelvis fixed to canonical frame"
                                   if clean else "root pose/vel 每 physics 步写回 canonical"),
        "locked_joint_method": ("28 hinge removed (body frame folded q0)" if clean
                                 else "28 关节 q/dq 每 physics 步写回 q0/0"),
        "locked_joint_names": [n for i, n in enumerate(joint_names) if i != jidx] if clean else "n/a(coupled)",
        "constraint_runtime_readback": {"njnt": int(m.njnt), "nu": int(m.nu),
                                        "hinge_names": hinge_names},
        "gravity": (0.0, 0.0, 0.0) if clean else None,
        "contact_disable_method": contacts_off_method,
        "step_rad_used": step_rad, "fallback_decided": fallback_decided,
        "duration_s": args.duration, "physics_dt": physics_dt, "n_steps": n_steps, "n_states": n_rec,
        "t_switch_s": t_switch, "t0_is_true_initial": True,
        "t_final_is_0_500": abs(float(rec["t_state"][-1]) - 0.5) < 1e-9,
        "sampling": "状态在推进物理后记录标 t+dt；t=0 为真实共同初态；命令 [t_cmd,t_state) 生效",
        "initial_state_npz_sha256": _canon_sha,
        "canonical_npz_sha256": _canon_sha,
        "xml_official_sha256": _xml_official_sha,
        "xml_used_sha256": _xml_used_sha,
        "xml_used_name": xml_used.name,
        "runner_sha256": _runner_sha,
        "model_used_sha256": _xml_used_sha,
        "model_sha256": _xml_used_sha,
        "armature_j_readback": arm_read,
        "kp_j/kd_j": [float(kp_pol[jidx]), float(kd_pol[jidx])],
        "ctrlrange_j": ctrl_range_j.tolist(),
        "joint_range_j": jrange.tolist(),
        "q0_j": q0_j,
        "max_drift_other_joints": max_drift_others, "max_root_drift": max_root_drift,
        "max_ncon": max_ncon, "max_contact_force": max_cf,
        "step_saturated_any": step_saturated,
        "hashes": {"policy": sha256_bytes(c["policy_path"]), "motion": sha256_bytes(c["motion_path"]),
                   "config": sha256_bytes(cfg_path),
                   "runner": _runner_sha,
                   "model": _xml_used_sha,
                   "initial_state": _canon_sha,
                   "canonical": _canon_sha},
    }
    (out_dir / "run_manifest.yaml").write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
                                               encoding="utf-8")
    log(f"OK: {out_dir} n_states={n_rec} drift_others={max_drift_others:.6f} root_drift={max_root_drift:.2e} "
        f"ncon_max={max_ncon} cf={max_cf:.3e} sat={step_saturated}")
    logf.close()


def _state29(m, d, clean, jidx, others_pol, q0_pol, j2s, s2s, joint_names, hinge_names):
    """返回 (q29, dq29)：clean 时被测真实、其余 q0/0；coupled 时全真实。"""
    if clean:
        q29 = q0_pol.copy(); dq29 = np.zeros(29)
        q29[jidx] = float(d.qpos[0]); dq29[jidx] = float(d.qvel[0])
        return q29, dq29
    q_pol = map_by_name(j2s(d.qpos[7:].copy()), s2s, joint_names, "q")
    dq_pol = map_by_name(j2s(d.qvel[6:].copy()), s2s, joint_names, "dq")
    return np.asarray(q_pol, dtype=np.float64), np.asarray(dq_pol, dtype=np.float64)


def _tau29(val, jidx):
    arr = np.zeros(29, dtype=np.float64)
    arr[jidx] = val
    return arr


if __name__ == "__main__":
    main()