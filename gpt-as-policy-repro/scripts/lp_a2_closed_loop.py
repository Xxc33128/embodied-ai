#!/usr/bin/env python3
"""LP-A2 闭环客户端（gap-sim 容器）：官方 openpi 客户端契约 × LiberoSession。

用法: lp_a2_closed_loop.py <arm> <device_tag> <init,init,...> [pin_ref.json]
- arm ∈ {paper_faithful, scene_pinned}：进噪声播种与证据文件名；
  scene_pinned 需给 pin_ref.json（含 fixture_poses，取自参考 episode）。
- 契约 pin（configs/lp_a2_validation.json）：wait10/replan5/max300/旋转180°/
  state=eef_pos+axisangle(eef_quat)+gripper_qpos/prompt=task.language/
  actions[:5] 执行；噪声按 (arm,init,replan_idx) 稳定播种显式注入两设备。
- 交换目录 /workspace/data/lp_a2/{req,resp}（与 gap-repro 的 policy server 共享卷）。
"""
import collections
import hashlib
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, "/workspace/repo/src")

import numpy as np

from gap_repro.sim.libero_session import LiberoSession

BASE = pathlib.Path("/workspace/data/lp_a2")
REQ, RESP = BASE / "req", BASE / "resp"
MAX_STEPS = 300
REPLAN = 5
POLL_TIMEOUT_S = 3600

ARM = sys.argv[1]
DEVICE_TAG = sys.argv[2]
INITS = [int(x) for x in sys.argv[3].split(",")]
PIN = json.load(open(sys.argv[4]))["fixture_poses"] if len(sys.argv) > 4 else None


def _quat2axisangle(quat):
    # 逐字对齐 openpi examples/libero/main.py L199-214（robosuite 拷贝源）
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def noise_for(arm, init, replan_idx):
    # 形状 (10,32)：pi05 模型内部动作维 32（LP-A0 JAX 腿同款语义，输出解码到 7）
    seed = int(hashlib.sha256(
        f"{arm}|{init}|{replan_idx}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).standard_normal((10, 32)).astype(np.float32)


def request(obs, arm, init, replan_idx, prompt):
    # 官方客户端预处理：旋转 180°（训练预处理对齐）；224 捕获下 resize 恒等
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate([
        obs["robot0_eef_pos"],
        _quat2axisangle(np.array(obs["robot0_eef_quat"], dtype=np.float64)),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)
    assert state.shape == (8,), f"state 契约 8 维，实得 {state.shape}"
    name = f"{arm}_{DEVICE_TAG}_init{init}_r{replan_idx}_{int(time.time()*1000)}.npz"
    # 原子落位：经文件句柄写 .tmp（np.savez 对路径会自动追加 .npz 后缀——
    # 曾两次踩坑：直接写最终名被 server 半写抢读；tmp 路径名被加缀成
    # .tmp.npz 且 rename 落空），再 rename 到最终名。
    import os
    tmp = REQ / (name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, agentview=img, wrist=wrist, state=state,
                 prompt=np.array(str(prompt)),
                 noise=noise_for(arm, init, replan_idx),
                 meta=np.array(f"{arm}|{init}|{replan_idx}"))
    os.rename(tmp, REQ / name)
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        resp = RESP / name
        if resp.exists():
            d = np.load(resp)
            resp.unlink()
            return np.asarray(d["actions"], dtype=np.float32), json.loads(str(d["meta"]))
        err = RESP / (name[:-4] + ".err")
        if err.exists():
            raise RuntimeError(f"server error: {err.read_text()}")
        time.sleep(0.1)
    raise TimeoutError(f"no response for {name}")


def run_episode(session, init):
    obs = session.reset_to(init, fixture_poses=PIN)
    start = {
        "fingerprint_full": session.fingerprint_full(obs),
        "fingerprint_state": session.fingerprint_state(obs),
        "fixture_poses": session.fixture_poses(),
    }
    prompt = session.task_description
    plan = collections.deque()
    executed, chunks, t = [], [], 0
    t_wall0 = time.time()
    while t < MAX_STEPS:
        if not plan:
            actions, meta = request(obs, ARM, init, len(chunks), prompt)
            assert len(actions) >= REPLAN, f"chunk {len(actions)} < replan {REPLAN}"
            plan.extend(actions[:REPLAN])
            chunks.append({"replan_idx": len(chunks), "t": t,
                           "actions": actions.tolist(), "server": meta})
            print(f"init{init} r{len(chunks)-1} t={t} "
                  f"{meta['infer_s']}s sha={meta['actions_sha']}", flush=True)
        action = plan.popleft()
        obs, _, done, _ = session.step(action)
        executed.append(np.asarray(action, dtype=np.float32))
        t += 1
        if done:
            break
    return {
        "success": bool(done), "done_step": t,
        "wall_s": round(time.time() - t_wall0, 1),
        "n_replans": len(chunks),
        "start_identity": start,
        "chunks": chunks,
    }, np.stack(executed)


def main():
    REQ.mkdir(parents=True, exist_ok=True)
    RESP.mkdir(parents=True, exist_ok=True)
    session = LiberoSession(suite="libero_goal", task_index=0, seed=0)
    for init in INITS:
        rec, executed = run_episode(session, init)
        tag = f"ep_{ARM}_{DEVICE_TAG}_init{init}"
        np.savez(BASE / f"{tag}.npz", executed=executed)
        out = {"schema": "gap_repro.lp_a2_episode.v1", "arm": ARM,
               "device": DEVICE_TAG, "init": init,
               "max_steps": MAX_STEPS, "replan_steps": REPLAN,
               "pinned": PIN is not None, "prompt": session.task_description,
               **rec}
        json.dump(out, open(BASE / f"{tag}.json", "w"), indent=1)
        print(f"EP_DONE {tag} success={rec['success']} step={rec['done_step']} "
              f"replans={rec['n_replans']} wall={rec['wall_s']}s", flush=True)
    session.env.close()


if __name__ == "__main__":
    main()
