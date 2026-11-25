#!/usr/bin/env python3
"""
p3_summarize_r2.py — R5 offline metrics/plots (self-contained, fail-closed)

Usage: python 04_analysis/p3_summarize_r2.py --root .
(reads ONLY R4 package relative paths; no E:/ or /mnt/ paths; no simulation)

Execution order (R5 plan §4.2):
 1. read 04_analysis/p3_final_gate.json; require validity_pass + VALID_P3
 2. recompute the four frozen trace SHA256 (must match §2.2 pins) -> else INVALID_R5_INPUT_HASH
 3. per trace: 500 cycles, finite, time/joint/body consistency
 4. joint order + groups (12/3/14) from body_mapping.json (assert disjoint+complete)
 5. D00 requested/effective must equal actual_processed_q_target point-by-point, else stop
 6. compute metrics, write json/long/wide/schema, plots + plot CSVs, crosscheck §6 (rel 1e-4)
 7. any failure: decision != VALID_R5, exit 1
"""
import argparse, hashlib, json, pathlib, sys

import numpy as np
import yaml

TRACE_PINS = {
    "isaac_d00_10s": "be64b024e206b0221d17a6135e3d2e5a75ab583db98e827d98c6c965dcb2e79d",
    "isaac_d20_10s_r2": "d4714acc68caa669fdb27c30c5ee608238a57e455276b2e93076432d8f5920c3",
    "mj_elbow_d00_10s": "ba9276f78b24759612441ec939ac0aefb287de6a2d9877d99224904e76474a84",
    "mj_elbow_d20_10s": "df965d24dc47f38036de61d5563e6dfc754702ed308dbd23d7f14a7f0156ec09",
}

CROSSCHECK = {
    ("within","isaac","q_sensitivity"): 0.030929,
    ("within","mujoco","q_sensitivity"): 0.048023,
    ("within","isaac","q_des_sensitivity"): 0.033234,
    ("within","mujoco","q_des_sensitivity"): 0.049301,
    ("within","isaac","body_position_world_sensitivity"): 0.026946,
    ("within","mujoco","body_position_world_sensitivity"): 0.051533,
    ("within","isaac","body_position_pelvis_relative_sensitivity"): 0.009635,
    ("within","mujoco","body_position_pelvis_relative_sensitivity"): 0.017089,
    ("within","isaac","body_orientation_sensitivity"): 0.052106,
    ("within","mujoco","body_orientation_sensitivity"): 0.104533,
    ("within","isaac","requested_tracking_rms_D00"): 0.131353,
    ("within","mujoco","requested_tracking_rms_D00"): 0.122301,
    ("within","isaac","requested_tracking_rms_D20"): 0.145288,
    ("within","mujoco","requested_tracking_rms_D20"): 0.139712,
    ("within","isaac","effective_tracking_rms_D20"): 0.130011,
    ("within","mujoco","effective_tracking_rms_D20"): 0.123062,
    ("within","isaac","group_LEGS_q_sensitivity"): 0.034088,
    ("within","mujoco","group_LEGS_q_sensitivity"): 0.057339,
    ("within","isaac","group_WAIST_q_sensitivity"): 0.012869,
    ("within","mujoco","group_WAIST_q_sensitivity"): 0.019398,
    ("within","isaac","group_ARMS_q_sensitivity"): 0.030822,
    ("within","mujoco","group_ARMS_q_sensitivity"): 0.043342,
    ("cross","q_gap_D00"): 0.055946,
    ("cross","q_gap_D20"): 0.067324,
    ("cross","body_position_world_gap_D00"): 0.104610,
    ("cross","body_position_world_gap_D20"): 0.074709,
    ("cross","body_position_pelvis_relative_gap_D00"): 0.017400,
    ("cross","body_position_pelvis_relative_gap_D20"): 0.023317,
    ("cross","body_orientation_gap_D00"): 0.104374,
    ("cross","body_orientation_gap_D20"): 0.150575,
    ("cross","pelvis_position_gap_D00"): 0.103292,
    ("cross","pelvis_position_gap_D20"): 0.070367,
    ("cross","pelvis_orientation_gap_D00"): 0.040747,
    ("cross","pelvis_orientation_gap_D20"): 0.084814,
}

LEGS = ["left_hip_roll_joint","left_hip_yaw_joint","left_hip_pitch_joint","left_knee_joint",
        "left_ankle_pitch_joint","left_ankle_roll_joint","right_hip_roll_joint","right_hip_yaw_joint",
        "right_hip_pitch_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint"]
WAIST = ["waist_yaw_joint","waist_roll_joint","waist_pitch_joint"]
ARMS = ["left_shoulder_pitch_joint","left_shoulder_roll_joint","left_arm_yaw_joint","left_elbow_pitch_joint",
        "left_elbow_yaw_joint","left_wrist_pitch_joint","left_wrist_roll_joint",
        "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_arm_yaw_joint","right_elbow_pitch_joint",
        "right_elbow_yaw_joint","right_wrist_pitch_joint","right_wrist_roll_joint"]


def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def rms(a): return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2)))

def geodesic(q1, q2):
    q1 = np.asarray(q1, dtype=np.float64); q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    return 2 * np.arccos(np.clip(dot, 0, 1))

def roll_pitch(qw):  # wxyz
    w, x, y, z = qw[:, 0], qw[:, 1], qw[:, 2], qw[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return roll, pitch

def fail(out_dir, decision, errors, validation_extra=None):
    v = {"decision": decision, "validity_pass": False, "errors": errors}
    if validation_extra:
        v.update(validation_extra)
    (out_dir / "p3_r5_validation.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
    print(json.dumps(v, indent=2))
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    args = ap.parse_args()
    root = pathlib.Path(args.root).resolve()
    out_dir = root / "04_analysis"
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    media_dir = root / "05_media"

    # step 1: upstream gate
    fg = out_dir / "p3_final_gate.json"
    if not fg.is_file():
        fail(out_dir, "INVALID_R5_NO_GATE", ["missing p3_final_gate.json"])
    gate = json.loads(fg.read_text(encoding="utf-8"))
    if not (gate.get("validity_pass") and gate.get("decision") == "VALID_P3"):
        fail(out_dir, "INVALID_R5_GATE_NOT_VALID_P3", [json.dumps(gate.get("decision"))])

    idx = json.loads((root / "00_inputs/p3_condition_index.json").read_text(encoding="utf-8"))
    conds = idx["conditions"]

    # step 2: trace hashes
    errs = []
    for name, pin in TRACE_PINS.items():
        p = root / conds[name]["trace"]
        got = sha(p)
        if got != pin:
            errs.append(f"{name} trace sha {got} != frozen {pin}")
    if errs:
        fail(out_dir, "INVALID_R5_INPUT_HASH", errs)

    # step 3/4: load + structure + order
    bm = json.loads((root / "00_inputs/body_mapping.json").read_text(encoding="utf-8"))
    joints = bm["joint_names_policy_order"]
    if len(joints) != 29:
        fail(out_dir, "INVALID_R5_ORDER", ["body_mapping joint count != 29"])
    groups = {"LEGS": [joints.index(j) for j in LEGS], "WAIST": [joints.index(j) for j in WAIST],
              "ARMS": [joints.index(j) for j in ARMS]}
    allidx = sorted(groups["LEGS"] + groups["WAIST"] + groups["ARMS"])
    if allidx != list(range(29)) or len(groups["LEGS"]) != 12 or len(groups["WAIST"]) != 3 or len(groups["ARMS"]) != 14:
        fail(out_dir, "INVALID_R5_GROUPS", [str({k: len(v) for k, v in groups.items()})])

    T = {}
    for name, c in conds.items():
        d = np.load(str(root / c["trace"]), allow_pickle=True)
        if d["q(29)"].shape != (500, 29):
            fail(out_dir, "INVALID_R5_TRACE_SHAPE", [f"{name} q {d['q(29)'].shape}"])
        if not np.array_equal(d["policy_step"].astype(int), np.arange(500)):
            fail(out_dir, "INVALID_R5_TRACE_ORDER", [f"{name} policy_step"])
        for f in ["q(29)", "dq(29)", "q_des(29)", "action(29)", "root_pos(3)", "root_quat(4)",
                  "actual_body_pos_w(14,3)", "actual_body_quat_w(14,4)",
                  "actual_body_lin_vel_w(14,3)", "actual_body_ang_vel_w(14,3)",
                  "reference_body_pos_w(14,3)", "reference_body_quat_w(14,4)"]:
            if f in d.files and not np.isfinite(d[f]).all():
                fail(out_dir, "INVALID_R5_TRACE_FINITE", [f"{name} {f}"])
        if "d20" in name:
            for f in ["q_des_requested(29)", "q_des_effective(29)"]:
                if not np.isfinite(d[f]).all():
                    fail(out_dir, "INVALID_R5_TRACE_FINITE", [f"{name} {f}"])
        if name.startswith("mj"):
            for f in ["tau_pd_raw_policy(29)", "applied_policy(29)", "qfrc_actuator_policy(29)"]:
                if not np.isfinite(d[f]).all():
                    fail(out_dir, "INVALID_R5_TRACE_FINITE", [f"{name} {f}"])
        T[name] = d

    # step 5: D00 requested==effective==q_des via actual_processed_q_target
    for name in ["isaac_d00_10s", "mj_elbow_d00_10s"]:
        d = T[name]
        if "actual_processed_q_target(29)" not in d.files:
            fail(out_dir, "INVALID_R5_D00_FIELDS", [f"{name} missing actual_processed_q_target"])
        diff = float(np.max(np.abs(d["actual_processed_q_target(29)"] - d["q_des(29)"])))
        if diff > 1e-6:
            fail(out_dir, "INVALID_R5_D00_EFFECTIVE_MISMATCH", [f"{name} proc!=q_des {diff}"])
    def req_field(name):
        d = T[name]
        return d["q_des_requested(29)"] if "q_des_requested(29)" in d.files else d["q_des(29)"]
    def eff_field(name):
        d = T[name]
        return d["q_des_effective(29)"] if "q_des_effective(29)" in d.files else d["actual_processed_q_target(29)"]

    # ---- metrics
    # survival/fall/reset derived from trace counters + summaries + upstream gate (R5.1 hygiene: no hardcode)
    cond_stats = {}
    for name, c in conds.items():
        d = T[name]
        summ = json.loads((root / c["summary"]).read_text(encoding="utf-8"))
        cycles = int(d["policy_step"].shape[0])
        done_max = float(np.max(d["done"]))
        fall_trace = [str(x) for x in np.asarray(d["fall_reason"]).reshape(-1) if str(x) not in ("", "None")]
        fall_summ = summ.get("fall_reason")
        # independent reset inference (mirrors p3_validate): counters continuous
        cont = (np.array_equal(d["policy_step"].astype(int), np.arange(cycles))
                and np.array_equal(d["time_step"].astype(int), np.arange(cycles)))
        if name == "isaac_d20_10s_r2":
            pf = d["actuator_physics_step_index_substep"].reshape(-1)
            cont = cont and np.array_equal(pf.astype(np.int64), np.arange(1, 4 * cycles + 1))
        reset_obs = (not cont) or done_max > 0
        # cross-check trace vs summary vs gate
        if cycles != 500 or done_max != 0.0 or fall_trace or fall_summ not in (None, "", "null") or reset_obs:
            fail(out_dir, "INVALID_R5_SURVIVAL_MISMATCH",
                 [f"{name} cycles={cycles} done={done_max} fall_trace={fall_trace[:2]} fall_summ={fall_summ} reset={reset_obs}"])
        if gate.get("survival", {}).get("isaac_d20") not in (None, 500):
            fail(out_dir, "INVALID_R5_GATE_SURVIVAL", [json.dumps(gate.get("survival"))])
        cond_stats[name] = {"trace": c["trace"], "trace_sha256": TRACE_PINS[name],
                            "cycles": cycles, "survival_cycles": cycles,
                            "survival_time_s": round(cycles * 0.02, 3),
                            "fall_reason": None, "reset_observed": False,
                            "reset_evidence": "independent_trace_counters_and_summary"}
    M = {"conditions": cond_stats,
         "groups": {k: {"indices": v, "size": len(v), "names": [joints[i] for i in v]} for k, v in groups.items()},
         "units": {"q": "rad", "position": "m", "orientation": "rad (geodesic 2*acos|dot|)", "torque": "N*m",
                   "linear_velocity": "m/s", "angular_velocity": "rad/s", "saturation_fraction": "1",
                   "action_variation": "policy_action_unit", "survival_cycles": "cycle", "survival_time": "s"}}

    eng_map = {"isaac": ("isaac_d00_10s", "isaac_d20_10s_r2"), "mujoco": ("mj_elbow_d00_10s", "mj_elbow_d20_10s")}
    within = {}
    for eng, (d00, d20) in eng_map.items():
        a, b = T[d00], T[d20]
        w = {}
        w["q_sensitivity"] = rms(b["q(29)"] - a["q(29)"])
        w["q_des_sensitivity"] = rms(req_field(d20) - a["q_des(29)"])
        w["body_position_world_sensitivity"] = rms(b["actual_body_pos_w(14,3)"] - a["actual_body_pos_w(14,3)"])
        bp_rel_b = b["actual_body_pos_w(14,3)"] - b["actual_body_pos_w(14,3)"][:, :1, :]
        bp_rel_a = a["actual_body_pos_w(14,3)"] - a["actual_body_pos_w(14,3)"][:, :1, :]
        w["body_position_pelvis_relative_sensitivity"] = rms(bp_rel_b - bp_rel_a)
        w["body_orientation_sensitivity"] = rms(geodesic(b["actual_body_quat_w(14,4)"], a["actual_body_quat_w(14,4)"]))
        w["requested_tracking_rms_D00"] = rms(a["q_des(29)"] - a["q(29)"])
        w["requested_tracking_rms_D20"] = rms(req_field(d20) - b["q(29)"])
        w["effective_tracking_rms_D20"] = rms(eff_field(d20) - b["q(29)"])
        for g, gi in groups.items():
            w[f"group_{g}_q_sensitivity"] = rms(b["q(29)"][:, gi] - a["q(29)"][:, gi])
        r0z = a["root_pos(3)"][:, 2] - b["root_pos(3)"][:, 2]
        w["root_height_sensitivity"] = rms(r0z)
        w["root_xy_drift_sensitivity"] = rms(b["root_pos(3)"][:, :2] - a["root_pos(3)"][:, :2])
        rp_a = roll_pitch(a["root_quat(4)"]); rp_b = roll_pitch(b["root_quat(4)"])
        w["root_roll_sensitivity"] = rms(rp_b[0] - rp_a[0])
        w["root_pitch_sensitivity"] = rms(rp_b[1] - rp_a[1])
        w["body_linear_vel_sensitivity"] = rms(b["actual_body_lin_vel_w(14,3)"] - a["actual_body_lin_vel_w(14,3)"])
        w["body_angular_vel_sensitivity"] = rms(b["actual_body_ang_vel_w(14,3)"] - a["actual_body_ang_vel_w(14,3)"])
        w["action_variation_D00"] = rms(np.diff(a["action(29)"], axis=0))
        w["action_variation_D20"] = rms(np.diff(b["action(29)"], axis=0))
        w["q_des_variation_D00"] = rms(np.diff(a["q_des(29)"], axis=0))
        w["q_des_variation_D20"] = rms(np.diff(b["q_des(29)"], axis=0))
        within[eng] = w

    # cross engine
    cross = {}
    def pair_gap(f00a, f20a, f00b, f20b, fn):
        return {"D00": fn(T[f00a], T[f00b]), "D20": fn(T[f20a], T[f20b])}
    cross["q_gap"] = pair_gap("isaac_d00_10s", "isaac_d20_10s_r2", "mj_elbow_d00_10s", "mj_elbow_d20_10s",
                              lambda x, y: rms(x["q(29)"] - y["q(29)"]))
    cross["body_position_world_gap"] = pair_gap("isaac_d00_10s", "isaac_d20_10s_r2", "mj_elbow_d00_10s", "mj_elbow_d20_10s",
                                                lambda x, y: rms(x["actual_body_pos_w(14,3)"] - y["actual_body_pos_w(14,3)"]))
    def relpos(d):
        return d["actual_body_pos_w(14,3)"] - d["actual_body_pos_w(14,3)"][:, :1, :]
    cross["body_position_pelvis_relative_gap"] = pair_gap("isaac_d00_10s", "isaac_d20_10s_r2", "mj_elbow_d00_10s", "mj_elbow_d20_10s",
                                                          lambda x, y: rms(relpos(x) - relpos(y)))
    cross["body_orientation_gap"] = pair_gap("isaac_d00_10s", "isaac_d20_10s_r2", "mj_elbow_d00_10s", "mj_elbow_d20_10s",
                                             lambda x, y: rms(geodesic(x["actual_body_quat_w(14,4)"], y["actual_body_quat_w(14,4)"])))
    cross["pelvis_position_gap"] = pair_gap("isaac_d00_10s", "isaac_d20_10s_r2", "mj_elbow_d00_10s", "mj_elbow_d20_10s",
                                            lambda x, y: rms(x["actual_body_pos_w(14,3)"][:, 0] - y["actual_body_pos_w(14,3)"][:, 0]))
    cross["pelvis_orientation_gap"] = pair_gap("isaac_d00_10s", "isaac_d20_10s_r2", "mj_elbow_d00_10s", "mj_elbow_d20_10s",
                                               lambda x, y: rms(geodesic(x["actual_body_quat_w(14,4)"][:, 0], y["actual_body_quat_w(14,4)"][:, 0])))
    aux = bm["body_names"]
    anchors = {"pelvis": 0, "left_ankle_roll_link": aux.index("left_ankle_roll_link"),
               "right_ankle_roll_link": aux.index("right_ankle_roll_link"),
               "left_wrist_roll_link": aux.index("left_wrist_roll_link"),
               "right_wrist_roll_link": aux.index("right_wrist_roll_link")}
    cross["anchor_position_gap"] = {
        k: pair_gap("isaac_d00_10s", "isaac_d20_10s_r2", "mj_elbow_d00_10s", "mj_elbow_d20_10s",
                    lambda x, y, i=i: rms(x["actual_body_pos_w(14,3)"][:, i] - y["actual_body_pos_w(14,3)"][:, i]))
        for i, k in [(v, k) for k, v in anchors.items()]}
    cross["anchor_orientation_gap"] = {
        k: pair_gap("isaac_d00_10s", "isaac_d20_10s_r2", "mj_elbow_d00_10s", "mj_elbow_d20_10s",
                    lambda x, y, i=i: rms(geodesic(x["actual_body_quat_w(14,4)"][:, i], y["actual_body_quat_w(14,4)"][:, i])))
        for i, k in [(v, k) for k, v in anchors.items()]}
    cross["body_linear_vel_gap"] = pair_gap("isaac_d00_10s", "isaac_d20_10s_r2", "mj_elbow_d00_10s", "mj_elbow_d20_10s",
                                            lambda x, y: rms(x["actual_body_lin_vel_w(14,3)"] - y["actual_body_lin_vel_w(14,3)"]))
    cross["body_angular_vel_gap"] = pair_gap("isaac_d00_10s", "isaac_d20_10s_r2", "mj_elbow_d00_10s", "mj_elbow_d20_10s",
                                             lambda x, y: rms(x["actual_body_ang_vel_w(14,3)"] - y["actual_body_ang_vel_w(14,3)"]))

    # absolute task tracking (vs reference motion) per condition
    task = {}
    TASK_NOTE = ("diagnostic only: MJ reference body_pos_w uses motion/COM-frame semantics and unaligned world frame; "
                 "raw MJ-vs-motion body error is NOT comparable to Isaac without the P0/P1 anchor-alignment algorithm; "
                 "excluded from conclusions (robot-vs-robot metrics above are the report basis)")
    for name in conds:
        d = T[name]
        rel_act = d["actual_body_pos_w(14,3)"] - d["actual_body_pos_w(14,3)"][:, :1, :]
        rel_ref = d["reference_body_pos_w(14,3)"] - d["reference_body_pos_w(14,3)"][:, :1, :]
        task[name] = {
            "requested_tracking_rms_vs_q": rms(req_field(name) - d["q(29)"]),
            "diagnostic_unaligned_body_position_world_error_vs_motion_rms": rms(d["actual_body_pos_w(14,3)"] - d["reference_body_pos_w(14,3)"]),
            "diagnostic_unaligned_body_position_pelvis_relative_error_vs_motion_rms": rms(rel_act - rel_ref),
            "diagnostic_unaligned_pelvis_height_mean_actual": float(np.mean(d["actual_body_pos_w(14,3)"][:, 0, 2])),
            "diagnostic_unaligned_body_orientation_error_vs_motion_rms": rms(geodesic(d["actual_body_quat_w(14,4)"], d["reference_body_quat_w(14,4)"])),
        }

    # mujoco torque
    mj = {"isaac_torque": "UNAVAILABLE"}
    for name, tag in [("mj_elbow_d00_10s", "D00"), ("mj_elbow_d20_10s", "D20")]:
        d = T[name]
        raw = d["tau_pd_raw_policy(29)"].astype(np.float64)
        app = d["applied_policy(29)"].astype(np.float64)
        qf = d["qfrc_actuator_policy(29)"].astype(np.float64)
        sat = float(np.mean(np.abs(raw) > np.abs(app) + 1e-6))
        mj[f"applied_torque_rms_{tag}"] = rms(app)
        mj[f"applied_torque_peak_{tag}"] = float(np.max(np.abs(app)))
        mj[f"raw_pd_torque_rms_{tag}"] = rms(raw)
        mj[f"torque_saturation_fraction_{tag}"] = sat
        mj[f"qfrc_actuator_rms_{tag}"] = rms(qf)
        # saturation vs ctrlrange: compare raw vs applied
    M.update({"engine_within_D20_vs_D00": within, "cross_engine_isaac_vs_mujoco": cross,
              "task_tracking_per_condition": task, "task_tracking_note": TASK_NOTE, "mujoco_torque": mj,
              "scope": "dance_9 / seed 42 / 10 s / gain 1.0 / current policy-motion-canonical-model / explicit position-target FIFO / 0 ms vs 20 ms"})

    # ---- step 6 crosscheck
    mismatches = []
    def chk(keys, got):
        ref = CROSSCHECK.get(tuple(keys))
        if ref is None:
            return
        rel = abs(got - ref) / max(abs(ref), 1e-12)
        if rel > 1e-4:
            mismatches.append({"key": "/".join(keys), "got": got, "ref": ref, "rel_err": rel})
    for eng, w in within.items():
        for k, v in w.items():
            if isinstance(v, float):
                chk(["within", eng, k], v)
    for k, v in cross.items():
        if isinstance(v, dict) and "D00" in v:
            chk(["cross", k + "_D00"], v["D00"]); chk(["cross", k + "_D20"], v["D20"])
    if mismatches:
        fail(out_dir, "CROSSCHECK_FAIL", mismatches, {"metrics_partial": M})

    # ---- write outputs
    (out_dir / "p3_metrics.json").write_text(json.dumps(M, indent=2), encoding="utf-8")

    # ---- R5.1 explicit unit maps (no mixed units anywhere)
    WITHIN_UNITS = {
        "q_sensitivity": "rad", "q_des_sensitivity": "rad",
        "body_position_world_sensitivity": "m", "body_position_pelvis_relative_sensitivity": "m",
        "body_orientation_sensitivity": "rad",
        "requested_tracking_rms_D00": "rad", "requested_tracking_rms_D20": "rad", "effective_tracking_rms_D20": "rad",
        "group_LEGS_q_sensitivity": "rad", "group_WAIST_q_sensitivity": "rad", "group_ARMS_q_sensitivity": "rad",
        "root_height_sensitivity": "m", "root_xy_drift_sensitivity": "m",
        "root_roll_sensitivity": "rad", "root_pitch_sensitivity": "rad",
        "body_linear_vel_sensitivity": "m/s", "body_angular_vel_sensitivity": "rad/s",
        "action_variation_D00": "policy_action_unit", "action_variation_D20": "policy_action_unit",
        "q_des_variation_D00": "rad", "q_des_variation_D20": "rad",
    }
    CROSS_UNITS = {
        "q_gap": "rad", "body_position_world_gap": "m", "body_position_pelvis_relative_gap": "m",
        "body_orientation_gap": "rad", "pelvis_position_gap": "m", "pelvis_orientation_gap": "rad",
        "body_linear_vel_gap": "m/s", "body_angular_vel_gap": "rad/s",
    }
    TORQUE_UNITS = {
        "applied_torque_rms_D00": "N*m", "applied_torque_rms_D20": "N*m",
        "raw_pd_torque_rms_D00": "N*m", "raw_pd_torque_rms_D20": "N*m",
        "qfrc_actuator_rms_D00": "N*m", "qfrc_actuator_rms_D20": "N*m",
        "applied_torque_peak_D00": "N*m", "applied_torque_peak_D20": "N*m",
        "torque_saturation_fraction_D00": "1", "torque_saturation_fraction_D20": "1",
    }
    TASK_UNITS = {
        "requested_tracking_rms_vs_q": "rad",
        "diagnostic_unaligned_body_position_world_error_vs_motion_rms": "m",
        "diagnostic_unaligned_body_position_pelvis_relative_error_vs_motion_rms": "m",
        "diagnostic_unaligned_pelvis_height_mean_actual": "m",
        "diagnostic_unaligned_body_orientation_error_vs_motion_rms": "rad",
    }
    assert set(WITHIN_UNITS) >= set(within["isaac"].keys()), sorted(set(within["isaac"]) - set(WITHIN_UNITS))
    for m in (CROSS_UNITS, TORQUE_UNITS, TASK_UNITS):
        pass

    schema = {"generated_by": "04_analysis/p3_summarize_r2.py", "revision": "R5.1", "inputs": TRACE_PINS,
              "definitions": {
                "q_sensitivity": {"formula": "sqrt(mean((q_D20-q_D00)^2)) over cycles*29", "unit": "rad", "coords": "joint space, policy order", "primary": True, "source": "q(29)"},
                "q_des_sensitivity": {"formula": "sqrt(mean((q_des_requested_D20-q_des_D00)^2))", "unit": "rad", "primary": True, "source": "q_des_requested(29) or q_des(29)"},
                "body_position_world_sensitivity": {"formula": "RMS over (500,14,3) of actual_body_pos_w difference", "unit": "m", "coords": "world frame", "primary": True},
                "body_position_pelvis_relative_sensitivity": {"formula": "per-frame pos - pos_pelvis then RMS", "unit": "m", "coords": "pelvis-translated (not rotated)", "primary": True},
                "body_orientation_sensitivity": {"formula": "geodesic 2*acos(|dot(q1n,q2n)|) RMS", "unit": "rad", "primary": True},
                "requested_tracking_rms": {"formula": "RMS(q_des_requested - q)", "unit": "rad", "primary": True},
                "effective_tracking_rms": {"formula": "RMS(q_des_effective - q)", "unit": "rad", "primary": True},
                "q_des_effective_D00": {"formula": "= actual_processed_q_target(29) == q_des(29) pointwise (validated <=1e-6)", "unit": "rad"},
                "group_*": {"formula": "q_sensitivity restricted to group indices", "unit": "rad", "groups": "LEGS12/WAIST3/ARMS14 from body_mapping.json", "primary": True},
                "cross_engine_position_gaps": {"formula": "RMS(isaac_field - mujoco_field) at same delay and policy_step pairing", "unit": "m", "fields": "body world/pelvis-relative/pelvis/anchor positions", "primary": True},
                "cross_engine_orientation_gaps": {"formula": "RMS geodesic between isaac and mujoco quaternions", "unit": "rad", "fields": "14-body, pelvis, feet/hands anchors", "primary": True},
                "cross_engine_q_gap": {"formula": "RMS(isaac q - mujoco q)", "unit": "rad", "primary": True},
                "cross_engine_velocity_gaps": {"formula": "RMS difference of body lin/angular velocities", "unit": "m/s and rad/s"},
                "mujoco applied_torque": {"formula": "RMS/max|.| applied_policy(29) (post-actuator force, policy order)", "unit": "N*m", "note": "qfrc_actuator_policy reported separately; isaac torque UNAVAILABLE"},
                "torque_saturation_fraction": {"formula": "mean(|tau_pd_raw_policy| > |applied_policy| + 1e-6)", "unit": "1"},
                "root_height/xy_drift": {"formula": "RMS difference of root_pos z / xy", "unit": "m"},
                "root_roll/pitch": {"formula": "RMS difference of euler roll/pitch from root_quat wxyz", "unit": "rad"},
                "action_variation": {"formula": "RMS(diff over cycles) of action(29)", "unit": "policy_action_unit"},
                "q_des_variation": {"formula": "RMS(diff over cycles) of q_des(29)", "unit": "rad"},
                "survival": {"formula": "cycles/time completed without fall or reset; derived from trace counters + summary + upstream gate", "unit": "cycle / s", "primary": True},
                "diagnostic_unaligned_*": {"formula": "robot-vs-motion raw values", "unit": "m or rad", "primary": False,
                                            "note": "MJ reference uses motion/COM-frame semantics without P0/P1 anchor alignment; diagnostic only, never a headline metric"},
              }}
    (out_dir / "p3_metric_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    import csv as _csv
    with open(out_dir / "p3_metrics_long.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f, lineterminator="\n")
        w.writerow(["scope", "engine_or_pair", "metric", "value", "unit"])
        for eng, mt in within.items():
            for k, v in mt.items():
                if isinstance(v, float):
                    w.writerow(["within_D20_vs_D00", eng, k, f"{v:.6f}", WITHIN_UNITS[k]])
        for k, v in cross.items():
            if k in ("anchor_position_gap", "anchor_orientation_gap"):
                unit = "m" if k == "anchor_position_gap" else "rad"
                for an, dd in v.items():
                    w.writerow(["cross_engine_anchor", "isaac_vs_mujoco", f"{k}:{an}_D00", f'{dd["D00"]:.6f}', unit])
                    w.writerow(["cross_engine_anchor", "isaac_vs_mujoco", f"{k}:{an}_D20", f'{dd["D20"]:.6f}', unit])
            elif isinstance(v, dict) and "D00" in v:
                w.writerow(["cross_engine", "isaac_vs_mujoco", f"{k}_D00", f'{v["D00"]:.6f}', CROSS_UNITS[k]])
                w.writerow(["cross_engine", "isaac_vs_mujoco", f"{k}_D20", f'{v["D20"]:.6f}', CROSS_UNITS[k]])
        for name, tt in task.items():
            for k, v in tt.items():
                w.writerow(["task_tracking_diagnostic", name, k, f"{v:.6f}", TASK_UNITS[k]])
        for k, v in mj.items():
            if isinstance(v, float):
                w.writerow(["torque", "mujoco", k, f"{v:.6f}", TORQUE_UNITS[k]])
        for name, st in cond_stats.items():
            w.writerow(["survival", name, "survival_cycles", st["survival_cycles"], "cycle"])
            w.writerow(["survival", name, "survival_time_s", f'{st["survival_time_s"]:.3f}', "s"])
            w.writerow(["survival", name, "reset_observed", int(st["reset_observed"]), "1"])
    # R5.1: previous "wide" csv was actually tidy long; renamed honestly (plan option 1)
    with open(out_dir / "p3_metrics_engine_tidy.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f, lineterminator="\n")
        w.writerow(["engine", "metric", "value", "unit"])
        keys = [k for k in within["isaac"] if isinstance(within["isaac"][k], float)]
        for eng, mt in within.items():
            for k in keys:
                w.writerow([eng, k, f"{mt[k]:.6f}", WITHIN_UNITS[k]])
    old_wide = out_dir / "p3_metrics_wide.csv"
    if old_wide.is_file():
        old_wide.unlink()

    def plot_csv(name, header, rows):
        with open(plot_dir / name, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f, lineterminator="\n")
            w.writerow(header)
            for r in rows:
                w.writerow(r)

    # ---- plots (10+)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    TITLE = "dance_9 / seed=42 / 10 s / gain=1.0"
    t = np.arange(500) * 0.02
    I00, I20, M00, M20 = T["isaac_d00_10s"], T["isaac_d20_10s_r2"], T["mj_elbow_d00_10s"], T["mj_elbow_d20_10s"]

    def save(fig, name):
        fig.tight_layout(); fig.savefig(plot_dir / name, dpi=150, bbox_inches="tight"); plt.close(fig)

    # 01 survival
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(["Isaac D00", "Isaac D20", "MJ D00", "MJ D20"], [10, 10, 10, 10], color=["#2b6cb0", "#63b3ed", "#276749", "#68d391"])
    ax.axvline(10, color="k", ls=":")
    ax.set_xlabel("sim time (s)"); ax.set_title(f"Survival timeline (no fall/reset) {TITLE}")
    save(fig, "01_survival_timeline.png")
    plot_csv("data_01_survival.csv", ["condition", "survival_cycles", "survival_time_s", "fall_reason", "reset_observed", "unit"],
             [[n, cond_stats[n]["survival_cycles"], f"{cond_stats[n]['survival_time_s']:.2f}", "null", "false", "cycle/s"] for n in cond_stats])

    # 02 q sensitivity groups bar
    fig, ax = plt.subplots(figsize=(8, 4))
    cats = ["all29", "LEGS12", "WAIST3", "ARMS14"]
    nj = [29, 12, 3, 14]
    iva = [within["isaac"]["q_sensitivity"], within["isaac"]["group_LEGS_q_sensitivity"], within["isaac"]["group_WAIST_q_sensitivity"], within["isaac"]["group_ARMS_q_sensitivity"]]
    mva = [within["mujoco"]["q_sensitivity"], within["mujoco"]["group_LEGS_q_sensitivity"], within["mujoco"]["group_WAIST_q_sensitivity"], within["mujoco"]["group_ARMS_q_sensitivity"]]
    x = np.arange(4)
    ax.bar(x - 0.18, iva, 0.36, label="Isaac")
    ax.bar(x + 0.18, mva, 0.36, label="MuJoCo")
    ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("q trajectory sensitivity D20 vs D00 (rad)")
    ax.set_title(f"q sensitivity all+groups {TITLE}"); ax.legend(); ax.grid(alpha=0.3)
    save(fig, "02_q_sensitivity_all_and_groups.png")
    plot_csv("data_02_q_sensitivity_groups.csv", ["group", "isaac", "mujoco", "unit", "n_joints", "aggregation"],
             [[cats[i], f"{iva[i]:.6f}", f"{mva[i]:.6f}", "rad", nj[i], "RMS over cycles x joints"] for i in range(4)])

    # 03 tracking requested vs effective (aggregate RMS over 29 joints per cycle)
    fig, ax = plt.subplots(figsize=(10, 4))
    def agg(d, rf):
        return np.sqrt(np.mean((d[rf] - d["q(29)"]) ** 2, axis=1))
    ax.plot(t, agg(I00, "q_des(29)"), color="black", label="Isaac D00 requested")
    ax.plot(t, agg(I20, "q_des_requested(29)"), color="black", ls="--", label="Isaac D20 requested")
    ax.plot(t, agg(I20, "q_des_effective(29)"), color="#2b6cb0", ls="-.", label="Isaac D20 effective")
    ax.plot(t, agg(M00, "q_des(29)"), color="green", label="MJ D00 requested")
    ax.plot(t, agg(M20, "q_des_requested(29)"), color="green", ls="--", label="MJ D20 requested")
    ax.plot(t, agg(M20, "q_des_effective(29)"), color="#68d391", ls="-.", label="MJ D20 effective")
    ax.set_xlabel("time (s)"); ax.set_ylabel("RMS over 29 joints of (q_des - q) (rad)")
    ax.set_title(f"requested vs effective tracking {TITLE}"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    save(fig, "03_q_tracking_requested_effective.png")
    plot_csv("data_03_tracking_timeseries.csv",
             ["time_s", "isaac_d00_requested", "isaac_d20_requested", "isaac_d20_effective",
              "mujoco_d00_requested", "mujoco_d20_requested", "mujoco_d20_effective", "unit", "aggregation"],
             [[f"{t[i]:.2f}", f"{agg(I00,'q_des(29)')[i]:.6f}", f"{agg(I20,'q_des_requested(29)')[i]:.6f}",
               f"{agg(I20,'q_des_effective(29)')[i]:.6f}", f"{agg(M00,'q_des(29)')[i]:.6f}",
               f"{agg(M20,'q_des_requested(29)')[i]:.6f}", f"{agg(M20,'q_des_effective(29)')[i]:.6f}",
               "rad", "per-cycle RMS over 29 joints of (q_des - q)"] for i in range(500)])

    # 04 requested vs effective selected joints (elbow pitch R, hip pitch R, waist yaw)
    sel = [("right_elbow_pitch_joint", joints.index("right_elbow_pitch_joint")),
           ("right_hip_pitch_joint", joints.index("right_hip_pitch_joint")),
           ("waist_yaw_joint", joints.index("waist_yaw_joint"))]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    for axj, (jn, ji) in zip(axes, sel):
        axj.plot(t, I20["q_des_requested(29)"][:, ji], color="black", label="requested")
        axj.plot(t, I20["q_des_effective(29)"][:, ji], color="#2b6cb0", ls="--", label="effective(20ms)")
        axj.plot(t, M20["q_des_requested(29)"][:, ji], color="green", label="requested MJ")
        axj.plot(t, M20["q_des_effective(29)"][:, ji], color="#68d391", ls="--", label="effective MJ")
        axj.set_title(jn.replace("_joint", "")); axj.set_xlabel("time (s)"); axj.set_ylabel("q_des (rad)"); axj.legend(fontsize=6); axj.grid(alpha=0.3)
    fig.suptitle(f"requested vs effective selected joints {TITLE}", y=1.02)
    save(fig, "04_q_des_requested_vs_effective_selected_joints.png")
    plot_csv("data_04_req_eff_joints.csv",
             ["time_s"] + sum([[f"isaac_d20_requested_{jn}", f"isaac_d20_effective_{jn}",
                                f"mujoco_d20_requested_{jn}", f"mujoco_d20_effective_{jn}"] for jn, _ in sel], ["unit"]),
             [[f"{t[i]:.2f}"] + sum([[f"{I20['q_des_requested(29)'][i,ji]:.6f}", f"{I20['q_des_effective(29)'][i,ji]:.6f}",
                                      f"{M20['q_des_requested(29)'][i,ji]:.6f}", f"{M20['q_des_effective(29)'][i,ji]:.6f}"]
                                     for _, ji in sel], ["rad"]) for i in range(500)])

    # 05 root position/height
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(t, I00["root_pos(3)"][:, 2], "k", label="Isaac D00"); axes[0].plot(t, I20["root_pos(3)"][:, 2], "k--", label="Isaac D20")
    axes[0].plot(t, M00["root_pos(3)"][:, 2], "g", label="MJ D00"); axes[0].plot(t, M20["root_pos(3)"][:, 2], "g--", label="MJ D20")
    axes[0].set_title("root height (m)"); axes[0].legend(fontsize=6)
    axes[1].plot(t, I00["root_pos(3)"][:, 0] - I00["root_pos(3)"][0, 0], "k"); axes[1].plot(t, I20["root_pos(3)"][:, 0] - I20["root_pos(3)"][0, 0], "k--")
    axes[1].plot(t, M00["root_pos(3)"][:, 0] - M00["root_pos(3)"][0, 0], "g"); axes[1].plot(t, M20["root_pos(3)"][:, 0] - M20["root_pos(3)"][0, 0], "g--")
    axes[1].set_title("root x drift (m)")
    axes[2].plot(t, I00["root_pos(3)"][:, 1] - I00["root_pos(3)"][0, 1], "k"); axes[2].plot(t, I20["root_pos(3)"][:, 1] - I20["root_pos(3)"][0, 1], "k--")
    axes[2].plot(t, M00["root_pos(3)"][:, 1] - M00["root_pos(3)"][0, 1], "g"); axes[2].plot(t, M20["root_pos(3)"][:, 1] - M20["root_pos(3)"][0, 1], "g--")
    axes[2].set_title("root y drift (m)")
    for a in axes:
        a.set_xlabel("time (s)"); a.grid(alpha=0.3)
    fig.suptitle(f"root position {TITLE}", y=1.02)
    save(fig, "05_root_position_and_height.png")
    plot_csv("data_05_root_position.csv",
             ["time_s", "isaac_d00_height", "isaac_d20_height", "mujoco_d00_height", "mujoco_d20_height",
              "isaac_d00_x_drift", "isaac_d20_x_drift", "mujoco_d00_x_drift", "mujoco_d20_x_drift",
              "isaac_d00_y_drift", "isaac_d20_y_drift", "mujoco_d00_y_drift", "mujoco_d20_y_drift", "unit"],
             [[f"{t[i]:.2f}", f"{I00['root_pos(3)'][i,2]:.6f}", f"{I20['root_pos(3)'][i,2]:.6f}",
               f"{M00['root_pos(3)'][i,2]:.6f}", f"{M20['root_pos(3)'][i,2]:.6f}",
               f"{I00['root_pos(3)'][i,0]-I00['root_pos(3)'][0,0]:.6f}", f"{I20['root_pos(3)'][i,0]-I20['root_pos(3)'][0,0]:.6f}",
               f"{M00['root_pos(3)'][i,0]-M00['root_pos(3)'][0,0]:.6f}", f"{M20['root_pos(3)'][i,0]-M20['root_pos(3)'][0,0]:.6f}",
               f"{I00['root_pos(3)'][i,1]-I00['root_pos(3)'][0,1]:.6f}", f"{I20['root_pos(3)'][i,1]-I20['root_pos(3)'][0,1]:.6f}",
               f"{M00['root_pos(3)'][i,1]-M00['root_pos(3)'][0,1]:.6f}", f"{M20['root_pos(3)'][i,1]-M20['root_pos(3)'][0,1]:.6f}",
               "m"] for i in range(500)])

    # 06 roll pitch
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for d, lab, c, ls in [(I00, "Isaac D00", "k", "-"), (I20, "Isaac D20", "k", "--"), (M00, "MJ D00", "g", "-"), (M20, "MJ D20", "g", "--")]:
        r, p_ = roll_pitch(d["root_quat(4)"])
        axes[0].plot(t, r, color=c, ls=ls, label=lab)
        axes[1].plot(t, p_, color=c, ls=ls, label=lab)
    axes[0].set_title("root roll (rad)"); axes[1].set_title("root pitch (rad)")
    for a in axes:
        a.set_xlabel("time (s)"); a.legend(fontsize=6); a.grid(alpha=0.3)
    fig.suptitle(f"root orientation {TITLE}", y=1.02)
    save(fig, "06_root_roll_pitch.png")
    rp_series = {lab: roll_pitch(d["root_quat(4)"]) for d, lab, _, _ in
                 [(I00, "isaac_d00", "k", "-"), (I20, "isaac_d20", "k", "--"), (M00, "mujoco_d00", "g", "-"), (M20, "mujoco_d20", "g", "--")]}
    plot_csv("data_06_root_roll_pitch.csv",
             ["time_s"] + sum([[f"{lab}_roll", f"{lab}_pitch"] for lab in rp_series], ["unit"]),
             [[f"{t[i]:.2f}"] + sum([[f"{rp_series[lab][0][i]:.6f}", f"{rp_series[lab][1][i]:.6f}"] for lab in rp_series], ["rad"]) for i in range(500)])

    # 07 world vs pelvis-relative cross gap bars
    fig, ax = plt.subplots(figsize=(9, 4))
    cats = ["14-body world gap (m)", "14-body pelvis-rel gap (m)", "pelvis pos gap (m)"]
    d00 = [cross["body_position_world_gap"]["D00"], cross["body_position_pelvis_relative_gap"]["D00"], cross["pelvis_position_gap"]["D00"]]
    d20 = [cross["body_position_world_gap"]["D20"], cross["body_position_pelvis_relative_gap"]["D20"], cross["pelvis_position_gap"]["D20"]]
    x = np.arange(3)
    ax.bar(x - 0.18, d00, 0.36, label="D00"); ax.bar(x + 0.18, d20, 0.36, label="D20")
    for i, (a, b) in enumerate(zip(d00, d20)):
        ax.text(i + 0.18, b, f"{(b - a) / a * 100:+.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("cross-engine gap (m)")
    ax.set_title(f"world vs pelvis-relative decomposition {TITLE}"); ax.legend(); ax.grid(alpha=0.3)
    save(fig, "07_body_position_world_vs_pelvis_relative.png")
    plot_csv("data_07_world_vs_pelvisrel.csv",
             ["metric", "D00", "D20", "change_pct", "unit", "aggregation", "coordinate_frame"],
             [["body_position_world_gap", f"{d00[0]:.6f}", f"{d20[0]:.6f}", f"{(d20[0]-d00[0])/d00[0]*100:.2f}", "m", "RMS over 500x14x3", "world"],
              ["body_position_pelvis_relative_gap", f"{d00[1]:.6f}", f"{d20[1]:.6f}", f"{(d20[1]-d00[1])/d00[1]*100:.2f}", "m", "RMS over 500x14x3", "pelvis-translated"],
              ["pelvis_position_gap", f"{d00[2]:.6f}", f"{d20[2]:.6f}", f"{(d20[2]-d00[2])/d00[2]*100:.2f}", "m", "RMS over 500x3", "world"]])

    # 08 (R5.1) two valid comparisons, both geodesic rad over 14 actual bodies; robot-vs-motion diagnostic removed from plots
    def sens_q_series(a, b):
        return np.sqrt(np.mean(geodesic(b["actual_body_quat_w(14,4)"], a["actual_body_quat_w(14,4)"]) ** 2, axis=1))
    def cross_q_series(x, y):
        return np.sqrt(np.mean(geodesic(x["actual_body_quat_w(14,4)"], y["actual_body_quat_w(14,4)"]) ** 2, axis=1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].plot(t, sens_q_series(I00, I20), color="black", label="Isaac D20 vs D00")
    axes[0].plot(t, sens_q_series(M00, M20), color="green", ls="--", label="MuJoCo D20 vs D00")
    axes[0].set_title("within-engine orientation sensitivity"); axes[0].set_ylabel("geodesic RMS over 14 bodies (rad)")
    axes[1].plot(t, cross_q_series(I00, M00), color="black", label="D00 isaac-vs-mujoco")
    axes[1].plot(t, cross_q_series(I20, M20), color="#c53030", ls="--", label="D20 isaac-vs-mujoco")
    axes[1].set_title("cross-engine orientation gap"); axes[1].set_ylabel("geodesic RMS over 14 bodies (rad)")
    for a in axes:
        a.set_xlabel("time (s)"); a.legend(fontsize=7); a.grid(alpha=0.3)
    fig.suptitle(f"body orientation geodesic (actual-body only) {TITLE}", y=1.02)
    save(fig, "08_body_orientation_geodesic.png")
    plot_csv("data_08_orientation_geodesic.csv",
             ["time_s", "isaac_d20_vs_d00", "mujoco_d20_vs_d00", "cross_d00", "cross_d20", "unit", "aggregation"],
             [[f"{t[i]:.2f}", f"{sens_q_series(I00,I20)[i]:.6f}", f"{sens_q_series(M00,M20)[i]:.6f}",
               f"{cross_q_series(I00,M00)[i]:.6f}", f"{cross_q_series(I20,M20)[i]:.6f}",
               "rad", "per-cycle geodesic RMS over 14 actual bodies"] for i in range(500)])

    # 09 (R5.1) four single-unit absolute panels + one relative panel (no shared mixed axis)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    cats9 = [("q_gap", "q gap", "rad"), ("body_position_world_gap", "body world gap", "m"),
             ("body_position_pelvis_relative_gap", "body pelvis-rel gap", "m"), ("body_orientation_gap", "body quat gap", "rad")]
    rel = []
    for axj, (ck, lab, unit) in zip(axes[:4], cats9):
        v0 = cross[ck]["D00"]; v2 = cross[ck]["D20"]
        axj.bar([0, 1], [v0, v2], color=["#4a5568", "#c53030"])
        axj.set_xticks([0, 1]); axj.set_xticklabels(["D00", "D20"])
        axj.set_ylabel(f"{lab} ({unit})"); axj.set_title(lab, fontsize=9); axj.grid(alpha=0.3)
        for xi, vv in enumerate([v0, v2]):
            axj.text(xi, vv, f"{vv:.4f}", ha="center", va="bottom", fontsize=7)
        rel.append((v2 - v0) / v0 * 100)
    axes[4].bar(np.arange(4), rel, color=["#4a5568" if r < 0 else "#c53030" for r in rel])
    axes[4].set_xticks(np.arange(4)); axes[4].set_xticklabels([c[1] for c in cats9], rotation=25, fontsize=7)
    axes[4].axhline(0, color="k", lw=0.8); axes[4].set_ylabel("D20 vs D00 change (%)")
    axes[4].set_title("relative change (dimensionless %)", fontsize=9); axes[4].grid(alpha=0.3)
    for i, rr in enumerate(rel):
        axes[4].text(i, rr, f"{rr:+.1f}%", ha="center", va="bottom" if rr > 0 else "top", fontsize=7)
    fig.suptitle(f"cross-engine gap D00 vs D20, one unit per panel {TITLE}", y=1.03)
    save(fig, "09_cross_engine_gap_d00_vs_d20.png")
    plot_csv("data_09_cross_gap.csv", ["metric", "D00", "D20", "change_pct", "unit", "aggregation", "coordinate_frame"],
             [[ck, f"{cross[ck]['D00']:.6f}", f"{cross[ck]['D20']:.6f}", f"{rel[i]:.2f}", unit, "RMS over 500 cycles",
               {"q_gap": "joint space", "body_position_world_gap": "world", "body_position_pelvis_relative_gap": "pelvis-translated", "body_orientation_gap": "SO3 geodesic"}[ck]]
              for i, (ck, lab, unit) in enumerate(cats9)])

    # 10 (R5.1) three panels: RMS / peak / saturation (units separated)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    rms_labs = ["applied", "raw_PD", "qfrc_actuator"]
    rms0 = [mj["applied_torque_rms_D00"], mj["raw_pd_torque_rms_D00"], mj["qfrc_actuator_rms_D00"]]
    rms2 = [mj["applied_torque_rms_D20"], mj["raw_pd_torque_rms_D20"], mj["qfrc_actuator_rms_D20"]]
    axes[0].bar(np.arange(3) - 0.18, rms0, 0.36, label="D00"); axes[0].bar(np.arange(3) + 0.18, rms2, 0.36, label="D20")
    axes[0].set_xticks(np.arange(3)); axes[0].set_xticklabels(rms_labs, fontsize=7); axes[0].set_ylabel("torque RMS (N*m)")
    axes[0].set_title("RMS (N*m)", fontsize=9); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    pk0 = [mj["applied_torque_peak_D00"], mj["raw_pd_torque_rms_D00"]]
    pk2 = [mj["applied_torque_peak_D20"], mj["raw_pd_torque_rms_D20"]]
    axes[1].bar(np.arange(2) - 0.18, pk0, 0.36, label="D00"); axes[1].bar(np.arange(2) + 0.18, pk2, 0.36, label="D20")
    axes[1].set_xticks(np.arange(2)); axes[1].set_xticklabels(["applied peak", "raw PD RMS"], fontsize=7)
    axes[1].set_ylabel("N*m"); axes[1].set_title("peak (N*m)", fontsize=9); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    s0 = mj["torque_saturation_fraction_D00"]; s2 = mj["torque_saturation_fraction_D20"]
    axes[2].bar([0, 1], [s0, s2], color=["#4a5568", "#c53030"])
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["D00", "D20"]); axes[2].set_ylabel("saturation fraction (1)")
    axes[2].set_title("saturation fraction (1)", fontsize=9); axes[2].grid(alpha=0.3)
    fig.suptitle(f"MuJoCo torque panels (Isaac torque UNAVAILABLE) {TITLE}", y=1.03)
    save(fig, "10_mujoco_torque_rms_peak.png")
    plot_csv("data_10_torque.csv", ["metric", "D00", "D20", "unit", "aggregation"],
             [["applied_torque_rms", f"{rms0[0]:.6f}", f"{rms2[0]:.6f}", "N*m", "RMS over 500x29"],
              ["raw_pd_torque_rms", f"{rms0[1]:.6f}", f"{rms2[1]:.6f}", "N*m", "RMS over 500x29"],
              ["qfrc_actuator_rms", f"{rms0[2]:.6f}", f"{rms2[2]:.6f}", "N*m", "RMS over 500x29"],
              ["applied_torque_peak", f"{pk0[0]:.6f}", f"{pk2[0]:.6f}", "N*m", "max abs"],
              ["torque_saturation_fraction", f"{s0:.6f}", f"{s2:.6f}", "1", "mean(|raw|>|applied|+1e-6)"]])

    # 11 (R5.1) dimensionless sensitivity ratio MuJoCo/Isaac
    fig, ax = plt.subplots(figsize=(9, 3.8))
    labels = ["q", "q_des", "body world", "body pelvis-rel", "body orientation"]
    keys11 = ["q_sensitivity", "q_des_sensitivity", "body_position_world_sensitivity",
              "body_position_pelvis_relative_sensitivity", "body_orientation_sensitivity"]
    ratio = [within["mujoco"][k] / within["isaac"][k] for k in keys11]
    ax.bar(np.arange(5), ratio, color="#2b6cb0")
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.text(4.4, 1.02, "ratio=1", fontsize=8)
    for i, rr in enumerate(ratio):
        ax.text(i, rr, f"{rr:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MuJoCo sensitivity / Isaac sensitivity (dimensionless)")
    ax.set_title(f"dimensionless engine sensitivity ratio {TITLE}"); ax.grid(alpha=0.3)
    save(fig, "11_metric_tradeoff_summary.png")
    plot_csv("data_11_tradeoff.csv",
             ["metric", "ratio_mujoco_over_isaac", "isaac_value", "mujoco_value", "isaac_unit", "mujoco_unit", "aggregation"],
             [[k, f"{ratio[i]:.4f}", f"{within['isaac'][k]:.6f}", f"{within['mujoco'][k]:.6f}",
               WITHIN_UNITS[k], WITHIN_UNITS[k], "D20-vs-D00 trajectory sensitivity"] for i, k in enumerate(keys11)])

    # ---- validation json
    v = {"decision": "VALID_R5", "revision": "R5.1", "validity_pass": True,
         "upstream": {"p3_final_gate": "VALID_P3", "trace_hashes_verified": TRACE_PINS},
         "crosscheck": {"reference": "R5 plan section 6", "max_rel_err_allowed": 1e-4, "mismatches": 0},
         "groups_sizes": {k: len(v2) for k, v2 in groups.items()},
         "isaac_torque": "UNAVAILABLE",
         "orientation": "geodesic 2*acos(|dot|)",
         "units_policy": "R5.1: every CSV row has a definite unit; no mixed units; nested anchor metrics expanded into long CSV",
         "anchor_orientation_gap_added": sorted(anchors.keys()),
         "media_time_semantics": "container playback duration differs from covered simulation time (10 s); see 05_media/media_manifest.json",
         "world_vs_pelvis_relative_note": "world body gap fell (pelvis/root translation closer) while pelvis-relative body gap increased; report both separately",
         "scope": M["scope"]}
    (out_dir / "p3_r5_validation.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
    print("VALID_R5 written; metrics/plots done")


if __name__ == "__main__":
    main()
