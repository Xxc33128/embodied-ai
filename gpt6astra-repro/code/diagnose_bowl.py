"""Task 2/3 诊断（规划 2026-09-14 §5-§6）：正确组装 + 分阶段诊断 + 子实验。

功能：
  1. 分阶段回归（默认）：staged scripted_solve 跑 seeds 1000-1009，每阶段
     门控（S1 空载到位 → S2 闭合 → S3 提升 → S4 搬运 → S5 转腕 → S6 释放），
     输出逐阶段记录 + 逐步诊断 JSONL。
  2. --pacing fast：同一路标点按工具链节奏（无沉降）复跑对照——规划 §6
     "慢速与接近正式工具执行速度的版本"，只在慢速脚本成功不能放行。
  3. --battery：空载到位位置网格（中心 + y±0.03 + 悬停/下探高度）与悬停
     ±10° 转腕，独立新鲜 reset，按拟议标准（3mm/2°/0.3s）判定。
  4. --series：代表 seed 的逐步诊断导出 CSV（滑移/接触力时序分析用）。
  5. --assembly：旧组装单因素消融（Task 2 证据，已记录，可复跑）。

用法：
  .venv/bin/python diagnose_bowl.py                        # fixed 组装 staged 慢速
  .venv/bin/python diagnose_bowl.py --pacing fast          # 工具链节奏对照
  .venv/bin/python diagnose_bowl.py --battery              # 空载/转腕电池
  .venv/bin/python diagnose_bowl.py --series 1000 1001     # CSV 时序
输出：diag-bowl/<mode>/… + diag-bowl/summary-<mode>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from mujoco_bowl import (_BOWL_XY, MuJoCoBowlEmbodiment, _cube_in_hand, _go,
                         build_xml, scripted_solve)

CONFIGS = {
    "old": dict(equality=False, walls_radian=False),
    "eq_only": dict(equality=True, walls_radian=False),
    "walls_only": dict(equality=False, walls_radian=True),
    "fixed": dict(equality=True, walls_radian=True),
}


def _blank_render(emb) -> None:
    emb._render = lambda cam: np.zeros((emb._img_h, emb._img_w, 3), dtype=np.uint8)


def _contact_diagnostics(emb) -> dict:
    """按左右指聚合与 cube 的接触；法向/切向力取自接触系（force[0]=法向）。"""
    mj, model, data = emb._mj, emb._model, emb._data
    cube_gid = model.geom("cube_geom").id
    finger_bids = {model.body("left_finger").id: "left",
                   model.body("right_finger").id: "right"}
    out = {s: {"n": 0, "fn": 0.0, "ft": 0.0, "pen": 0.0} for s in finger_bids.values()}
    force = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if cube_gid not in (c.geom1, c.geom2):
            continue
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        side = next((s for b, s in finger_bids.items() if b in (b1, b2)), None)
        if side is None:
            continue
        mj.mj_contactForce(model, data, i, force)
        out[side]["n"] += 1
        out[side]["fn"] += float(force[0])
        out[side]["ft"] += float(np.hypot(force[1], force[2]))
        out[side]["pen"] = min(out[side]["pen"], float(c.dist))
    return out


def step_diagnostics(emb, action: np.ndarray, res) -> dict:
    """单控制步诊断记录（规划 §5 诊断字段清单）。"""
    model, data = emb._model, emb._data
    tcp = emb._site_pos()
    cube = emb._cube_pos()
    cube_hand = _cube_in_hand(emb)
    q = data.qpos[emb._qadr]
    margin = np.minimum(q - emb._ctrl_lo, emb._ctrl_hi - q)
    contacts = _contact_diagnostics(emb)
    q1, q2 = float(data.qpos[emb._finger_qadr[0]]), float(data.qpos[emb._finger_qadr[1]])
    return {
        "t": round(float(data.time), 3),
        "phase": int(res.info["phase"]),
        "cmd": np.round(action[:5], 4).tolist(),
        "tcp": np.round(tcp, 4).tolist(),
        "pos_err": round(float(np.linalg.norm(action[:3] - tcp)), 5),
        "grip_cmd": round(float(action[4]), 3),
        "grip_meas": round(emb._grip_measured(), 3),
        "finger_q": [round(q1, 5), round(q2, 5)],
        "finger_asym": round(abs(q1 - q2), 5),
        "joint_margin_min": round(float(margin.min()), 4),
        "act_force_grip": round(float(data.actuator_force[emb._act_grip]), 2),
        "act_force_arm_max": round(float(np.abs(data.actuator_force[emb._act_arm]).max()), 1),
        "cube": np.round(cube, 4).tolist(),
        "cube_in_hand": np.round(cube_hand, 4).tolist(),
        "contacts": {s: {k: round(v, 5) for k, v in d.items()} for s, d in contacts.items()},
        "solver_niter": int(np.max(data.solver_niter)) if data.solver_niter.size else 0,
        "nwarn": int(np.sum(data.warning.number)),
    }


def run_seed(cfg_xml: str, seed: int, *, pacing: str = "slow",
             render: bool = False) -> tuple[dict, list[dict]]:
    """跑一个 seed 的分阶段 scripted_solve，返回 (汇总, 逐步诊断行)。"""
    emb = MuJoCoBowlEmbodiment(execution_noise=(0.0, 0.0, 0.0), xml=cfg_xml)
    if not render:
        _blank_render(emb)
    rows: list[dict] = []
    orig_step = emb.step

    def logged_step(action):
        res = orig_step(action)
        rows.append(step_diagnostics(emb, np.asarray(action.data), res))
        return res

    emb.step = logged_step
    try:
        result = scripted_solve(emb, seed, pacing=pacing)
    finally:
        emb.step = orig_step
        emb.close()
    summary = {**result, "n_steps": len(rows),
               "max_warn": max((r["nwarn"] for r in rows), default=0)}
    return summary, rows


def battery(seed: int) -> dict:
    """空载到位位置网格 + 悬停 ±10° 转腕（新鲜 reset，拟议标准 3mm/2°/0.3s）。"""
    emb = MuJoCoBowlEmbodiment(execution_noise=(0.0, 0.0, 0.0))
    _blank_render(emb)
    from inspect_robots.scene import Scene
    emb.reset(Scene(id=f"bat-{seed}", instruction="t", init_seed=seed), seed=seed)
    cube = emb._cube_pos()
    qw, qx, qy, qz = emb._data.qpos[emb._cube_qadr + 3:emb._cube_qadr + 7]
    yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    pts = {
        "hover_center": (cube[0], cube[1], cube[2] + 0.12),
        "descend_center": (cube[0], cube[1], cube[2] + 0.018),
        "descend_y+0.03": (cube[0], cube[1] + 0.03, cube[2] + 0.018),
        "descend_y-0.03": (cube[0], cube[1] - 0.03, cube[2] + 0.018),
    }
    approach = {}
    for name, (x, y, z) in pts.items():
        emb.reset(Scene(id=f"bat-{seed}", instruction="t", init_seed=seed), seed=seed)
        approach[name] = _go(emb, x, y, z, yaw=yaw, grip=1.0)
    # 转腕：悬停按转移门（40mm/6°，边界姿态实测极限）进入，±10° 只严测 yaw
    emb.reset(Scene(id=f"bat-{seed}", instruction="t", init_seed=seed), seed=seed)
    hov = _go(emb, cube[0], cube[1], cube[2] + 0.12, yaw=0.0, grip=1.0,
              tol_pos=0.040, tol_yaw=np.radians(6.0))
    wrist = {"hover": hov}
    for name, ty in [("plus10", 0.0 + np.radians(10)), ("back", 0.0),
                     ("minus10", 0.0 - np.radians(10))]:
        wrist[name] = _go(emb, cube[0], cube[1], cube[2] + 0.12, yaw=ty, grip=1.0,
                          tol_pos=0.040, tol_yaw=np.radians(2.0))
    emb.close()
    return {"seed": seed, "approach": approach, "wrist": wrist}


def freefall_bowl(cfg_xml: str, drop_z: float = 0.25, settle_s: float = 2.0) -> dict:
    """自由落体入碗对照：cube 从碗正上方静止释放，检查静置容纳。"""
    import mujoco
    from mujoco_bowl import _CUBE_HALF

    model = mujoco.MjModel.from_xml_string(cfg_xml)
    data = mujoco.MjData(model)
    cube_qadr = model.jnt_qposadr[model.joint("cube_joint").id]
    data.qpos[cube_qadr:cube_qadr + 3] = [_BOWL_XY[0], _BOWL_XY[1], drop_z]
    for _ in range(int(settle_s / model.opt.timestep)):
        mujoco.mj_step(model, data)
    cube = np.array(data.xpos[model.body("cube").id])
    r_xy = float(np.hypot(cube[0] - _BOWL_XY[0], cube[1] - _BOWL_XY[1]))
    v = float(np.linalg.norm(data.qvel[cube_qadr + 0:cube_qadr + 3]))
    return {"inside": bool(r_xy < 0.055 - 0.008 and cube[2] < _CUBE_HALF + 0.006),
            "r_xy": round(r_xy, 4), "z": round(float(cube[2]), 4), "speed": round(v, 4)}


_CSV_COLS = ["t", "phase", "pos_err", "grip_cmd", "grip_meas", "finger_asym",
             "joint_margin_min", "act_force_grip", "act_force_arm_max",
             "solver_niter", "nwarn",
             "tcp_x", "tcp_y", "tcp_z", "cube_x", "cube_y", "cube_z",
             "hand_x", "hand_y", "hand_z", "asym_L", "asym_R",
             "nL", "nR", "fnL", "fnR", "ftL", "ftR", "penL", "penR"]


def write_series(rows: list[dict], path: str) -> None:
    """把逐步诊断行展平成 CSV（时序分析输入）。"""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_CSV_COLS)
        for r in rows:
            c = r["contacts"]
            w.writerow([r["t"], r["phase"], r["pos_err"], r["grip_cmd"], r["grip_meas"],
                        r["finger_asym"], r["joint_margin_min"], r["act_force_grip"],
                        r["act_force_arm_max"], r["solver_niter"], r["nwarn"],
                        *r["tcp"], *r["cube"], *r["cube_in_hand"],
                        r["cube_in_hand"][0], r["cube_in_hand"][1], 0.0,
                        c["left"]["n"], c["right"]["n"],
                        c["left"]["fn"], c["right"]["fn"],
                        c["left"]["ft"], c["right"]["ft"],
                        c["left"]["pen"], c["right"]["pen"]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembly", default="fixed", choices=list(CONFIGS))
    ap.add_argument("--pacing", default="slow", choices=["slow", "fast"])
    ap.add_argument("--seeds", type=int, default=10, help="从 1000 起的连续 seed 数")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--battery", action="store_true")
    ap.add_argument("--series", type=int, nargs="*", default=None,
                    help="导出这些 seed 的逐步 CSV")
    ap.add_argument("--no-freefall", action="store_true")
    args = ap.parse_args()

    os.makedirs("diag-bowl", exist_ok=True)
    cfg_xml = build_xml(**CONFIGS[args.assembly])
    tag = f"{args.assembly}-{args.pacing}"

    if args.battery:
        out = [battery(1000 + i) for i in range(args.seeds)]
        with open(f"diag-bowl/battery-{tag}.json", "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        for b in out:
            bad = [k for k, v in b["approach"].items() if not v["reached"]]
            print(f"battery seed={b['seed']}: approach FAIL={bad or '无'} "
                  f"wrist={[k for k, v in b['wrist'].items() if not v['reached']] or '全过'}")
        print(f"-> diag-bowl/battery-{tag}.json")
        return

    out_dir = os.path.join("diag-bowl", tag)
    os.makedirs(out_dir, exist_ok=True)
    sums = []
    for i in range(args.seeds):
        seed = 1000 + i
        s, rows = run_seed(cfg_xml, seed, pacing=args.pacing, render=args.render)
        sums.append(s)
        with open(os.path.join(out_dir, f"seed-{seed}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    if args.series:
        os.makedirs("diag-bowl/series", exist_ok=True)
        for seed in args.series:
            tag2 = f"{tag}-seed-{seed}"
            _, rows = run_seed(cfg_xml, seed, pacing=args.pacing, render=False)
            write_series(rows, f"diag-bowl/series/{tag2}.csv")
        print(f"series -> diag-bowl/series/ ({args.series})")

    n_ok = sum(s["success"] for s in sums)
    n_held = sum(s["held"] for s in sums)
    stages_fail = {}
    for s in sums:
        stages_fail[s["fail_stage"] or "none"] = stages_fail.get(s["fail_stage"] or "none", 0) + 1
    summary_all = {"held": n_held, "success": n_ok, "fail_stages": stages_fail,
                   "seeds": sums}
    if not args.no_freefall:
        summary_all["freefall"] = freefall_bowl(cfg_xml)
    print(f"\n=== {tag} ===")
    print(f"  held: {n_held}/{args.seeds}   success: {n_ok}/{args.seeds}")
    print(f"  fail_stages: {stages_fail}")
    if "freefall" in summary_all:
        print(f"  freefall: {summary_all['freefall']}")
    for s in sums:
        st = " | ".join(f"{x['stage']}:{'ok' if x.get('ok', x.get('reached', x.get('success'))) else 'FAIL'}"
                        for x in s["stages"])
        print(f"    seed={s['seed']} fail={s['fail_stage'] or '-'} steps={s['n_steps']} {st}")
    with open(f"diag-bowl/summary-{tag}.json", "w") as f:
        json.dump(summary_all, f, indent=1, ensure_ascii=False)
    print(f"summary -> diag-bowl/summary-{tag}.json")


if __name__ == "__main__":
    main()
