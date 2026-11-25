#!/usr/bin/env python3
"""sim2sim_record.py — Humanoid Gym 100Hz MuJoCo baseline with logging (self-contained).

Self-contained 12-DOF constants (no `from humanoid.envs` import, no pip install -e .).
Reuses the proven loop from /mnt/e/tmp/hg_wsl/record.py but adds:
  - CLI: --duration --control-hz --fixed-delay-ms --gain-scale --output --record-video --offscreen
  - Baseline fixed: vx=0.4, vy=0, dyaw=0; obs noise OFF (patched XML, hash recorded); physics dt=0.001; action_scale=0.25
  - Logging of q/dq/action/target_q/tau/root etc. + manifest with hashes

WSL usage (MuJoCo EGL):
  export MUJOCO_GL=egl
  python humanoid/scripts/sim2sim_record.py --duration 10 --control-hz 100 --output /mnt/e/sim2sim-week-2026-08-26/05_gym/mj_100hz --record-video

Do NOT overwrite humanoid/scripts/sim2sim.py (kept as viewer-only).
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import mujoco
import torch

# Self-contained constants (from XBotLCfg, humanoid-gym)
DT = 0.001
DEFAULT_CONTROL_HZ = 100
DEFAULT_DURATION = 10.0
WIDTH, HEIGHT = 1280, 720
VIDEO_FPS = 30

# 12-DOF config
NUM_ACTIONS = 12
NUM_SINGLE_OBS = 47
FRAME_STACK = 15
NUM_OBSERVATIONS = FRAME_STACK * NUM_SINGLE_OBS  # 705
CLIP_OBS = 18.0
CLIP_ACT = 18.0
ACTION_SCALE = 0.25
KPS = np.array([200, 200, 350, 350, 15, 15, 200, 200, 350, 350, 15, 15], dtype=np.double)
KDS = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.double)
TAU_LIMIT = 200.0 * np.ones(NUM_ACTIONS, dtype=np.double)
OBS_SCALES = {"lin_vel": 2.0, "ang_vel": 1.0, "dof_pos": 1.0, "dof_vel": 0.05}

# Baseline command (fixed)
CMD_VX = 0.4
CMD_VY = 0.0
CMD_DYAW = 0.0

# Paths (repo-relative)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = REPO_ROOT / "logs/XBot_ppo/exported/policies/policy_example.pt"
DEFAULT_MJCF = REPO_ROOT / "resources/robots/XBot/mjcf/XBot-L.xml"

FB_XML = f"""<visual>
        <global offwidth="{WIDTH}" offheight="{HEIGHT}"/>"""

def sha256_bytes(p: Path):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else None

def quaternion_to_euler_array(quat):
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    from scipy.spatial.transform import Rotation as R
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def make_patched_model(mjcf_path: Path, out_dir: Path):
    src = Path(mjcf_path)
    xml = src.read_text(encoding="utf-8")
    xml = re.sub(r'\s+sensornoise="[^"]*"', "", xml)
    assert "<worldbody>" in xml and xml.count("<worldbody>") == 1
    xml = xml.replace("<visual>", FB_XML, 1)
    # Use same dir as src for meshdir relative
    dst = src.parent / "XBot-L-patched.xml"
    # Also copy to out_dir for provenance
    dst_out = Path(out_dir) / "XBot-L-patched.xml"
    Path(dst).write_text(xml, encoding="utf-8")
    Path(dst_out).write_text(xml, encoding="utf-8")
    return dst

def aim_camera(renderer, eye, target):
    cam = renderer.scene.camera[0]
    fwd = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    fwd /= np.linalg.norm(fwd)
    cam.pos = np.asarray(eye, dtype=np.float64)
    cam.forward = fwd
    cam.up = np.array([0.0, 0.0, 1.0])

def open_video(path, fps, width, height):
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if vw.isOpened():
            return ("cv2", vw)
        vw.release()
    except Exception as e:
        print(f"  cv2 unavailable ({e}), falling back to imageio")
    import imageio.v2 as iio
    w = iio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    return ("imageio", w)

def write_frame(video, img):
    kind, w = video
    if kind == "cv2":
        w.write(img[:, :, ::-1])
    else:
        w.append_data(img)

def close_video(video):
    kind, w = video
    if kind == "cv2":
        w.release()
    else:
        w.close()

def main():
    ap = argparse.ArgumentParser(description="Humanoid Gym 100Hz MuJoCo baseline with logging")
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="sim duration seconds")
    ap.add_argument("--control-hz", type=int, default=DEFAULT_CONTROL_HZ, help="control frequency (100 or 50)")
    ap.add_argument("--fixed-delay-ms", type=float, default=0.0, help="fixed delay ms (0 for baseline, 10/20 for sweep, but baseline must be 0)")
    ap.add_argument("--gain-scale", type=float, default=1.0, help="gain scale")
    ap.add_argument("--output", type=str, required=True, help="output directory")
    ap.add_argument("--record-video", action="store_true", help="record video via EGL")
    ap.add_argument("--offscreen", action="store_true", help="alias for record-video (EGL)")
    ap.add_argument("--policy", type=str, default=str(DEFAULT_POLICY), help="policy pt path")
    ap.add_argument("--mjcf", type=str, default=str(DEFAULT_MJCF), help="mjcf xml path")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    control_hz = int(args.control_hz)
    decimation = max(1, int(round((1.0/control_hz) / DT)))
    control_dt = DT * decimation
    print(f"Config: duration={args.duration}s control_hz={control_hz} decimation={decimation} control_dt={control_dt} dt={DT} fixed_delay={args.fixed_delay_ms}ms gain_scale={args.gain_scale}")

    # Make patched model
    patched_mjcf = make_patched_model(Path(args.mjcf), out_dir)
    patched_hash = sha256_bytes(patched_mjcf)
    orig_hash = sha256_bytes(Path(args.mjcf))
    policy_path = Path(args.policy)
    policy_hash = sha256_bytes(policy_path)
    script_hash = sha256_bytes(Path(__file__))

    print(f"[1/4] Loading policy: {policy_path}")
    policy = torch.jit.load(str(policy_path))
    policy.eval()

    print(f"[2/4] Building model {patched_mjcf}")
    model = mujoco.MjModel.from_xml_path(str(patched_mjcf))
    model.opt.timestep = DT
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    print(f"      nq={model.nq} nu={model.nu} nbody={model.nbody}")

    if args.record_video or args.offscreen:
        print(f"[3/4] Offscreen renderer {WIDTH}x{HEIGHT} (EGL)...")
        renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
        renderer.update_scene(data, camera=-1)
        video_path = out_dir / "video.mp4"
        video = open_video(video_path, VIDEO_FPS, WIDTH, HEIGHT)
    else:
        renderer = None
        video = None
        video_path = None

    # Logging buffers
    total_steps = int(args.duration / DT)
    total_control_steps = int(args.duration / control_dt)
    print(f"[4/4] Simulating {args.duration}s ({total_steps} physics steps, {total_control_steps} control steps) cmd vx={CMD_VX}...")
    # buffers for logging (control rate)
    log_t = []
    log_q = []
    log_dq = []
    log_action = []
    log_target_q = []
    log_tau = []
    log_root_pos = []
    log_root_quat = []
    log_root_lin_vel = []
    log_root_ang_vel = []

    action = np.zeros(NUM_ACTIONS, dtype=np.double)
    hist_obs = [np.zeros([1, NUM_SINGLE_OBS], dtype=np.double) for _ in range(FRAME_STACK)]
    target_q = np.zeros(NUM_ACTIONS, dtype=np.double)
    count_lowlevel = 0
    frames = 0
    first_img = None
    last_img = None

    # For fixed delay simulation (optional)
    delay_steps = int(round(args.fixed_delay_ms / (DT*1000))) if args.fixed_delay_ms !=0 else 0
    if delay_steps>0:
        print(f"  fixed delay {args.fixed_delay_ms}ms = {delay_steps} physics steps (not used in baseline, but logged)")

    try:
        for i in range(total_steps):
            q, dq, quat, v, omega, gvec = get_obs(data)
            q = q[-NUM_ACTIONS:]
            dq = dq[-NUM_ACTIONS:]

            if count_lowlevel % decimation == 0:
                t_ctrl = count_lowlevel * DT
                obs = np.zeros([1, NUM_SINGLE_OBS], dtype=np.float32)
                eu_ang = quaternion_to_euler_array(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi
                obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * DT / 0.64)
                obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * DT / 0.64)
                obs[0, 2] = CMD_VX * OBS_SCALES["lin_vel"]
                obs[0, 3] = CMD_VY * OBS_SCALES["lin_vel"]
                obs[0, 4] = CMD_DYAW * OBS_SCALES["ang_vel"]
                obs[0, 5:17] = q * OBS_SCALES["dof_pos"]
                obs[0, 17:29] = dq * OBS_SCALES["dof_vel"]
                obs[0, 29:41] = action
                obs[0, 41:44] = omega
                obs[0, 44:47] = eu_ang
                obs = np.clip(obs, -CLIP_OBS, CLIP_OBS)
                hist_obs.pop(0)
                hist_obs.append(obs)
                policy_input = np.zeros([1, NUM_OBSERVATIONS], dtype=np.float32)
                for fi in range(FRAME_STACK):
                    policy_input[0, fi * NUM_SINGLE_OBS:(fi + 1) * NUM_SINGLE_OBS] = hist_obs[fi][0, :]
                with torch.no_grad():
                    action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -CLIP_ACT, CLIP_ACT)
                target_q = action * ACTION_SCALE

                # Log at control rate
                log_t.append(t_ctrl)
                log_q.append(q.copy())
                log_dq.append(dq.copy())
                log_action.append(action.copy())
                log_target_q.append(target_q.copy())
                # root
                root_pos = data.qpos[0:3].copy()
                root_quat = data.qpos[3:7].copy()  # wxyz? data.qpos 3:7 is quat wxyz
                # For Gym, qpos[3:7] is quat (x,y,z,w) ??? Actually Gym uses same as mujoco: qpos[3:7] is quat wxyz? Check record.py uses data.qpos[0:3] and sensor orientation for quat.
                # We'll use sensor quat for root orientation as in get_obs
                log_root_pos.append(root_pos.copy())
                log_root_quat.append(quat.copy())
                log_root_lin_vel.append(data.qvel[0:3].copy())
                log_root_ang_vel.append(omega.copy())
                # tau will be computed after pd

            target_dq = np.zeros(NUM_ACTIONS, dtype=np.double)
            tau = pd_control(target_q, q, KPS * args.gain_scale, target_dq, dq, KDS * args.gain_scale)
            tau = np.clip(tau, -TAU_LIMIT, TAU_LIMIT)
            if count_lowlevel % decimation == 0:
                log_tau.append(tau.copy())
            data.ctrl = tau
            mujoco.mj_step(model, data)
            count_lowlevel += 1

            if renderer is not None and count_lowlevel % 33 == 0:  # ~30 fps
                base_xyz = data.qpos[0:3]
                target = (base_xyz[0], base_xyz[1] * 0.5, 0.75)
                eye = (base_xyz[0] + 0.35, base_xyz[1] * 0.5 - 2.9, 1.25)
                renderer.update_scene(data, camera=-1)
                aim_camera(renderer, eye, target)
                img = renderer.render()
                if first_img is None:
                    first_img = img.copy()
                last_img = img.copy()
                write_frame(video, img)
                frames += 1
                if frames % 150 == 0:
                    print(f"      t={data.time:5.1f}s base=({data.qpos[0]:+.2f},{data.qpos[1]:+.2f},{data.qpos[2]:+.2f}) | {frames} frames")
    finally:
        if video is not None:
            close_video(video)

    print(f"[Done] {frames} frames -> {video_path if video_path else 'no video'}")
    if first_img is not None:
        import imageio.v2 as iio
        iio.imwrite(str(out_dir / "frame0.png"), first_img)
        iio.imwrite(str(out_dir / "frame_end.png"), last_img)
        print(f"      stills saved")

    # Save logs
    log_data = {
        "t": np.array(log_t, dtype=np.float32),
        "q": np.array(log_q, dtype=np.float32),
        "dq": np.array(log_dq, dtype=np.float32),
        "action": np.array(log_action, dtype=np.float32),
        "target_q": np.array(log_target_q, dtype=np.float32),
        "tau": np.array(log_tau, dtype=np.float32),
        "root_pos": np.array(log_root_pos, dtype=np.float32),
        "root_quat": np.array(log_root_quat, dtype=np.float32),
        "root_lin_vel": np.array(log_root_lin_vel, dtype=np.float32),
        "root_ang_vel": np.array(log_root_ang_vel, dtype=np.float32),
    }
    np.savez_compressed(out_dir / "trace.npz", **log_data)
    # Also save as npz with manifest
    # P1-2 extra fields
    import platform, subprocess as _sp
    try:
        _git_commit = _sp.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
        _git_dirty = _sp.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], capture_output=True, text=True, timeout=5).stdout.strip()
    except: _git_commit="unavailable"; _git_dirty=""
    # history time span both definitions
    hist_window_samples = FRAME_STACK
    hist_window_dt = FRAME_STACK * control_dt
    hist_first_to_last = (FRAME_STACK - 1) * control_dt
    manifest = {
        "policy": str(policy_path),
        "policy_sha256": policy_hash,
        "mjcf_original": str(Path(args.mjcf)),
        "mjcf_original_sha256": orig_hash,
        "mjcf_patched": str(patched_mjcf),
        "mjcf_patched_sha256": patched_hash,
        "script": str(Path(__file__)),
        "script_sha256": script_hash,
        "duration": args.duration,
        "control_hz": control_hz,
        "control_dt": control_dt,
        "physics_dt": DT,
        "decimation": decimation,
        "fixed_delay_ms": args.fixed_delay_ms,
        "gain_scale": args.gain_scale,
        "cmd": {"vx": CMD_VX, "vy": CMD_VY, "dyaw": CMD_DYAW},
        "obs_scales": OBS_SCALES,
        "kps": KPS.tolist(),
        "kds": KDS.tolist(),
        "action_scale": ACTION_SCALE,
        "frame_stack": FRAME_STACK,
        "history_length": FRAME_STACK,
        "history_time_span_samples": hist_window_samples,
        "history_time_span_window": hist_window_dt,
        "history_time_span_first_to_last": hist_first_to_last,
        "history_time_span_note": f"{hist_window_samples} samples at {control_hz} Hz: window {hist_window_dt:.3f}s, first-to-last {hist_first_to_last:.3f}s",
        "joint_names_12": ["left_leg_roll_joint","left_leg_yaw_joint","left_leg_pitch_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint","right_leg_roll_joint","right_leg_yaw_joint","right_leg_pitch_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint"],
        "joint_order": ["left_leg_roll_joint","left_leg_yaw_joint","left_leg_pitch_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint","right_leg_roll_joint","right_leg_yaw_joint","right_leg_pitch_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint"],
        "default_q_12": [0.0]*12,
        "init_state": {"pos": [0.0,0.0,0.88], "quat": [1,0,0,0], "note": "MJCF base_link pos 0,0,0.88 + free joint 0,0,0; qpos[2] is free joint z, body height = qpos[2]+0.88"},
        "history_init": "zeros (15 frames of zeros, first obs at t=0 uses 0)",
        "seed": 42,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "mujoco": mujoco.__version__, "numpy": np.__version__},
        "repo": {"commit": _git_commit, "dirty": bool(_git_dirty), "dirty_files": len(_git_dirty.splitlines()) if _git_dirty else 0},
        "failure_predicate": "none (Gym baseline has no termination; only clip and PD)",
        "fall": False,
        "failure_time": None,
        "reset_count": 0,
        "root_pos_semantics": "data.qpos[0:3] is free joint translation (world xz + free z); base_link body height = qpos[2] + 0.88",
        "n_control_steps": len(log_t),
        "n_physics_steps": total_steps,
        "video_frames": frames,
        "video_path": str(video_path) if video_path else None,
    }
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  trace.npz + run_manifest.json saved to {out_dir}")
    print(f"  history time span: {FRAME_STACK * control_dt:.2f}s ({FRAME_STACK} frames * {control_dt}s)")

if __name__ == "__main__":
    main()
