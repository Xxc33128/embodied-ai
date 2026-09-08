#!/usr/bin/env python3
"""
N4：生成 29 行 policy-order actuator_runtime_contract.csv + 停止条件检查
输入：
  - isaac_runtime_dump.json（--runtime-dump 产物）
  - MuJoCo smoke run_manifest.yaml（runtime_mjmodel_actuator_params + policy_metadata）
输出：
  - actuator_runtime_contract.csv（29 行，policy order）
  - n4_stop_condition_check.json（逐条停止条件判定）
规则：拿不到的 runtime 值写 UNAVAILABLE；requested 值与 readback 分开列，禁止混用。
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ISAAC = Path(r"E:/sim2sim-week-2026-08-26/04_actuator_isolation/n4_runtime_dump/isaac_runtime_dump.json")
MJ_MANIFEST = Path(r"E:/sim2sim-week-2026-08-26/04_actuator_isolation/n4_runtime_dump/mj_smoke_manifest.yaml")
OUT_CSV = Path(r"E:/sim2sim-week-2026-08-26/04_actuator_isolation/actuator_runtime_contract.csv")
OUT_JSON = Path(r"E:/sim2sim-week-2026-08-26/04_actuator_isolation/n4_stop_condition_check.json")
TOL = 1e-6


def main():
    isaac = json.loads(ISAAC.read_text(encoding="utf-8"))
    mj = yaml.safe_load(MJ_MANIFEST.read_text(encoding="utf-8"))
    mj_params = mj["runtime_mjmodel_actuator_params"]              # keyed by s2s name
    mj_policy_meta = mj["policy_metadata"]
    pol_names = isaac["policy_joint_names"]                        # policy order (29)
    rt = isaac["rt_joint_names"]
    assert rt == pol_names, f"Isaac rt order != policy order: {rt != pol_names}"
    idx_rt = {n: i for i, n in enumerate(rt)}

    def arr(tag):
        v = isaac[f"{tag}_runtime_readback"]
        return None if v == "UNAVAILABLE" else np.asarray(v, dtype=np.float64)

    ikp = arr("kp"); ikd = arr("kd"); iarm = arr("armature"); ieff = arr("effort_limit"); ivel = arr("velocity_limit")
    # requested (metadata)
    rkp = np.asarray(isaac["kp_requested_from_metadata"], dtype=np.float64)
    rkd = np.asarray(isaac["kd_requested_from_metadata"], dtype=np.float64)

    rows = []
    problems = []
    for j in pol_names:
        i = idx_rt[j]
        mp = mj_params[j]
        mj_kp = mj_policy_meta["kp"][mj_policy_meta["joint_names"].index(j)]
        mj_kd = mj_policy_meta["kd"][mj_policy_meta["joint_names"].index(j)]
        rows.append({
            "joint_name": j,
            "isaac_armature_runtime": f"{iarm[i]:.6f}" if iarm is not None else "UNAVAILABLE",
            "mujoco_armature_runtime": f"{mp['dof_armature']:.6f}",
            "isaac_kp_requested": f"{rkp[i]:.4f}",
            "isaac_kp_runtime": f"{ikp[i]:.4f}" if ikp is not None else "UNAVAILABLE",
            "mujoco_kp_runtime": "UNAVAILABLE(PD 在 runner 外计算,用 metadata kp)",  # MuJoCo 无 runtime PD 增益
            "mujoco_kp_metadata": f"{mj_kp:.4f}",
            "isaac_kd_requested": f"{rkd[i]:.4f}",
            "isaac_kd_runtime": f"{ikd[i]:.4f}" if ikd is not None else "UNAVAILABLE",
            "mujoco_kd_metadata": f"{mj_kd:.4f}",
            "isaac_effort_limit": f"{ieff[i]:.2f}" if ieff is not None else "UNAVAILABLE",
            "mujoco_ctrl_range": str(mp["actuator_ctrlrange"]),
            "isaac_velocity_limit": f"{ivel[i]:.4f}" if ivel is not None else "UNAVAILABLE",
            "mujoco_dof_velocity_limit": "UNAVAILABLE",
            "mujoco_joint_id": mp["mujoco_joint_id"],
            "mujoco_dof_id": mp["mujoco_dof_id"],
            "mujoco_actuator_id": mp["mujoco_actuator_id"],
            "mujoco_gainprm0": f"{mp['actuator_gainprm0']:.4f}",
            "mujoco_gear": f"{mp['actuator_gear']:.4f}",
            "mujoco_dof_damping": f"{mp['dof_damping']:.5f}",
            "mujoco_dof_frictionloss": f"{mp['dof_frictionloss']:.5f}",
            "isaac_kp_runtime_status": "OK" if ikp is not None else "UNAVAILABLE",
            "isaac_armature_runtime_status": "OK" if iarm is not None else "UNAVAILABLE",
            "parameter_source": "runtime readback",
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV written: {OUT_CSV} rows={len(rows)}")

    # ---------------- N4 停止条件 ----------------
    sc = {}
    # SC1: Kp/Kd/armature 任一侧读不到
    sc["sc1_kpkd_armature_readable"] = {"ok": iarm is not None and ikp is not None and ikd is not None,
                                        "detail": f"isaac arm/kp/kd readback: {[x is not None for x in (iarm, ikp, ikd)]}"}
    # SC2: requested vs runtime Kp/Kd 不一致且原因不明
    kp_diff = float(np.max(np.abs(rkp - ikp))) if ikp is not None else None
    kd_diff = float(np.max(np.abs(rkd - ikd))) if ikd is not None else None
    sc["sc2_requested_vs_runtime_gain"] = {"ok": (kp_diff is not None and kd_diff is not None
                                                  and kp_diff < 1e-3 and kd_diff < 1e-3),
                                           "max_kp_diff": kp_diff, "max_kd_diff": kd_diff}
    # SC3: right elbow/hip 的 joint/dof id 可追溯
    for j in ("right_elbow_pitch_joint", "right_hip_pitch_joint"):
        mp = mj_params[j]
        ok = isinstance(mp["mujoco_joint_id"], int) and isinstance(mp["mujoco_dof_id"], int)
        sc[f"sc3_ids_{j}"] = {"ok": ok, "joint_id": mp["mujoco_joint_id"], "dof_id": mp["mujoco_dof_id"]}
    # SC4: effort/ctrl clamp 语义
    elbow_i_eff = ieff[idx_rt["right_elbow_pitch_joint"]] if ieff is not None else None
    elbow_mj_ctrl = mj_params["right_elbow_pitch_joint"]["actuator_ctrlrange"]
    sc["sc4_effort_clamp_semantics"] = {
        "ok": elbow_i_eff is not None and abs(elbow_i_eff - 95.0) < 1e-6 and elbow_mj_ctrl == [-95.0, 95.0],
        "isaac_effort_elbow": elbow_i_eff, "mujoco_ctrlrange_elbow": elbow_mj_ctrl,
        "note": "Isaac effort_limit=95 == MuJoCo ctrlrange ±95 -> 同一 clamp 语义"}
    # SC5: right elbow armature 匹配预期 Isaac 0.01 / MuJoCo 0.0685
    el_a_i = iarm[idx_rt["right_elbow_pitch_joint"]] if iarm is not None else None
    el_a_m = mj_params["right_elbow_pitch_joint"]["dof_armature"]
    sc["sc5_elbow_armature_expected"] = {"ok": el_a_i is not None and abs(el_a_i - 0.01) < 1e-6
                                          and abs(el_a_m - 0.0685) < 1e-6,
                                          "isaac": el_a_i, "mujoco": el_a_m}
    # 汇总
    all_ok = all(v.get("ok", False) for v in sc.values())
    sc["overall_n4_stop_condition"] = "NO_STOP" if all_ok else "STOP"
    OUT_JSON.write_text(json.dumps(sc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(sc, indent=2, ensure_ascii=False))
    print("RESULT:", sc["overall_n4_stop_condition"])
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())