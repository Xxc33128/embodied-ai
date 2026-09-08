#!/usr/bin/env python3
"""Generate 4 minimal curves for Lab 10s baseline (and 5s)."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE = Path("E:/sim2sim-week-2026-08-26/02_lab_dance9")
ISAAC_10 = BASE / "isaac_10s"
MJ_10 = BASE / "mj_10s"
ISAAC_5 = BASE / "isaac_5s"
MJ_5 = BASE / "mj_5s"
OUT = BASE / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# Load canonical for joint order
CANON = np.load(BASE.parent / "01_lab_interface_gate/aligned_v1/canonical_initial_state.npz")
joint_names = list(CANON["joint_names"])
def jidx(name): return joint_names.index(name)

# Joints of interest
J_HIP = jidx("right_hip_pitch_joint")
J_KNEE = jidx("right_knee_joint")
J_ANKLE = jidx("right_ankle_pitch_joint")
J_ELBOW = jidx("right_elbow_pitch_joint")

def load_isaac(path):
    d = dict(np.load(path / "policy_trace.npz"))
    # Isaac uses keys: "t", "q(29)", "q_des(29)", "action(29)", "root_pos(3)", "root_quat(4)"
    t = d["t"].astype(float) if "t" in d else d["policy_step"].astype(float)*0.02
    q = d["q(29)"].astype(float) if "q(29)" in d else d["q"].astype(float)
    qdes = d["q_des(29)"].astype(float)
    action = d["action(29)"].astype(float) if "action(29)" in d else None
    root_pos = d["root_pos(3)"].astype(float) if "root_pos(3)" in d else None
    root_quat = d["root_quat(4)"].astype(float) if "root_quat(4)" in d else None
    # torque: try computed/applied
    tau = None
    if "tau_pd_raw_policy(29)" in d:
        tau = d["tau_pd_raw_policy(29)"].astype(float)
    elif "tau" in d:
        tau = d["tau"].astype(float)
    return {"t": t, "q": q, "qdes": qdes, "action": action, "root_pos": root_pos, "root_quat": root_quat, "tau": tau, "raw": d}

def load_mj(path):
    d = dict(np.load(path / "policy_trace.npz"))
    t = d["t"].astype(float) if "t" in d else d["policy_step"].astype(float)*0.02
    # MJ uses "q(29)" etc but via policy order
    # In run_mujoco_onnx, keys are "q(29)", "q_des(29)", etc. but also "tau_pd_raw(29)" etc.
    q = d["q(29)"].astype(float) if "q(29)" in d else d["q"].astype(float)
    qdes = d["q_des(29)"].astype(float) if "q_des(29)" in d else d["q_des"].astype(float)
    # torque
    tau = d["tau_pd_raw(29)"].astype(float) if "tau_pd_raw(29)" in d else None
    if tau is None and "tau_pd_raw_policy(29)" in d:
        tau = d["tau_pd_raw_policy(29)"].astype(float)
    root_pos = d["root_pos(3)"].astype(float) if "root_pos(3)" in d else None
    root_quat = d["root_quat(4)"].astype(float) if "root_quat(4)" in d else None
    action = None
    if "action(29)" in d:
        action = d["action(29)"].astype(float)
    return {"t": t, "q": q, "qdes": qdes, "action": action, "root_pos": root_pos, "root_quat": root_quat, "tau": tau, "raw": d}

def quat_to_rpy(q):
    # q is wxyz
    w,x,y,z = q
    # roll
    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2*(w*y - z*x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

for tag, isaac_p, mj_p, dur in [("10s", ISAAC_10, MJ_10, 10), ("5s", ISAAC_5, MJ_5, 5)]:
    print(f"Generating {tag} plots...")
    try:
        isaac = load_isaac(isaac_p)
        mj = load_mj(mj_p)
    except Exception as e:
        print(f"  skip {tag}: {e}")
        import traceback; traceback.print_exc()
        continue
    t_i = isaac["t"]
    t_m = mj["t"]
    # 1. root height / roll / pitch
    fig, axs = plt.subplots(3,1, figsize=(10,8), sharex=True)
    # height
    if isaac["root_pos"] is not None and mj["root_pos"] is not None:
        axs[0].plot(t_i, isaac["root_pos"][:,2], label="Isaac", color="black")
        axs[0].plot(t_m, mj["root_pos"][:,2], label="MuJoCo", color="red", ls="--")
        axs[0].set_ylabel("root height (m)")
        axs[0].legend(); axs[0].grid(alpha=0.3)
        # roll/pitch from quat
        rpy_i = np.array([quat_to_rpy(q) for q in isaac["root_quat"]])
        rpy_m = np.array([quat_to_rpy(q) for q in mj["root_quat"]])
        axs[1].plot(t_i, rpy_i[:,0], label="Isaac roll", color="black")
        axs[1].plot(t_m, rpy_m[:,0], label="MuJoCo roll", color="red", ls="--")
        axs[1].set_ylabel("roll (rad)"); axs[1].legend(); axs[1].grid(alpha=0.3)
        axs[2].plot(t_i, rpy_i[:,1], label="Isaac pitch", color="black")
        axs[2].plot(t_m, rpy_m[:,1], label="MuJoCo pitch", color="red", ls="--")
        axs[2].set_ylabel("pitch (rad)"); axs[2].set_xlabel("t (s)"); axs[2].legend(); axs[2].grid(alpha=0.3)
    else:
        print("  root pos/quat missing")
    fig.suptitle(f"Lab dance_9 {tag} - root height/roll/pitch")
    plt.tight_layout(); plt.savefig(OUT / f"root_{tag}.png", dpi=150); plt.close()
    print(f"  saved root_{tag}.png")

    # 2. leg joints q/qdes
    fig, axs = plt.subplots(3,1, figsize=(10,9), sharex=True)
    for ax, j, name in zip(axs, [J_HIP, J_KNEE, J_ANKLE], ["right_hip_pitch", "right_knee", "right_ankle_pitch"]):
        ax.plot(t_i, isaac["q"][:,j], label="Isaac q", color="black")
        ax.plot(t_i, isaac["qdes"][:,j], label="Isaac q_des", color="gray", ls=":")
        ax.plot(t_m, mj["q"][:,j], label="MuJoCo q", color="red", ls="--")
        ax.plot(t_m, mj["qdes"][:,j], label="MuJoCo q_des", color="orange", ls=":")
        ax.set_ylabel(f"{name} (rad)"); ax.grid(alpha=0.3)
        if ax==axs[0]: ax.legend(ncol=2, fontsize=8)
    axs[-1].set_xlabel("t (s)")
    fig.suptitle(f"Lab dance_9 {tag} - leg q/q_des")
    plt.tight_layout(); plt.savefig(OUT / f"leg_q_{tag}.png", dpi=150); plt.close()
    print(f"  saved leg_q_{tag}.png")

    # 3. elbow q/qdes/action
    fig, axs = plt.subplots(3,1, figsize=(10,8), sharex=True)
    # q/qdes
    axs[0].plot(t_i, isaac["q"][:,J_ELBOW], label="Isaac q", color="black")
    axs[0].plot(t_i, isaac["qdes"][:,J_ELBOW], label="Isaac q_des", color="gray", ls=":")
    axs[0].plot(t_m, mj["q"][:,J_ELBOW], label="MuJoCo q", color="red", ls="--")
    axs[0].plot(t_m, mj["qdes"][:,J_ELBOW], label="MuJoCo q_des", color="orange", ls=":")
    axs[0].set_ylabel("elbow q (rad)"); axs[0].legend(); axs[0].grid(alpha=0.3)
    # action
    if isaac["action"] is not None:
        axs[1].plot(t_i, isaac["action"][:,J_ELBOW], label="Isaac action", color="black")
    if mj["action"] is not None:
        axs[1].plot(t_m, mj["action"][:,J_ELBOW], label="MuJoCo action", color="red", ls="--")
    else:
        # derive action from q_des if needed
        pass
    axs[1].set_ylabel("elbow action"); axs[1].legend(); axs[1].grid(alpha=0.3)
    # qdes - q
    axs[2].plot(t_i, isaac["qdes"][:,J_ELBOW]-isaac["q"][:,J_ELBOW], label="Isaac qdes-q", color="black")
    axs[2].plot(t_m, mj["qdes"][:,J_ELBOW]-mj["q"][:,J_ELBOW], label="MuJoCo qdes-q", color="red", ls="--")
    axs[2].set_ylabel("elbow qdes-q"); axs[2].set_xlabel("t (s)"); axs[2].legend(); axs[2].grid(alpha=0.3)
    fig.suptitle(f"Lab dance_9 {tag} - elbow q/qdes/action")
    plt.tight_layout(); plt.savefig(OUT / f"elbow_{tag}.png", dpi=150); plt.close()
    print(f"  saved elbow_{tag}.png")

    # 4. torque
    fig, axs = plt.subplots(2,1, figsize=(10,6), sharex=True)
    # use tau if available, else try applied
    if isaac["tau"] is not None:
        axs[0].plot(t_i, isaac["tau"][:,J_HIP], label="Isaac hip torque", color="black")
        axs[1].plot(t_i, isaac["tau"][:,J_ELBOW], label="Isaac elbow torque", color="black")
    else:
        # try to load from raw
        isaac_raw = isaac["raw"]
        if "qfrc_actuator(29)" in isaac_raw:
            axs[0].plot(t_i, isaac_raw["qfrc_actuator(29)"][:,J_HIP], label="Isaac qfrc hip", color="black")
            axs[1].plot(t_i, isaac_raw["qfrc_actuator(29)"][:,J_ELBOW], label="Isaac qfrc elbow", color="black")
    if mj["tau"] is not None:
        axs[0].plot(t_m, mj["tau"][:,J_HIP], label="MuJoCo hip torque", color="red", ls="--")
        axs[1].plot(t_m, mj["tau"][:,J_ELBOW], label="MuJoCo elbow torque", color="red", ls="--")
    else:
        mj_raw = mj["raw"]
        if "tau_pd_raw(29)" in mj_raw:
            axs[0].plot(t_m, mj_raw["tau_pd_raw(29)"][:,J_HIP], label="MuJoCo hip torque", color="red", ls="--")
            axs[1].plot(t_m, mj_raw["tau_pd_raw(29)"][:,J_ELBOW], label="MuJoCo elbow torque", color="red", ls="--")
    for ax in axs:
        ax.legend(); ax.grid(alpha=0.3)
    axs[0].set_ylabel("hip torque (Nm)"); axs[1].set_ylabel("elbow torque (Nm)"); axs[1].set_xlabel("t (s)")
    fig.suptitle(f"Lab dance_9 {tag} - torque (hip/elbow)")
    plt.tight_layout(); plt.savefig(OUT / f"torque_{tag}.png", dpi=150); plt.close()
    print(f"  saved torque_{tag}.png")

print("All plots generated to", OUT)
