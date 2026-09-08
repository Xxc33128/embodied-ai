#!/usr/bin/env python3
"""asset_equivalence_check.py — G0 资产等价门禁（2026-08-28 审查 Step B.1，独立实现）

不 import urdf_isolation.py / mj_isolation_model.py 的任何变换函数，独立解析 URDF / MJCF：

mode=urdf  纯 python（Windows）：解析官方+隔离 URDF，正向运动学（canonical root **一次**）、
           目标关节世界位置/轴、下游 link 世界位姿、COM、质量、等效惯量 I_eff（解析公式）。
mode=mj    WSL mujoco：加载隔离 XML，设被测 hinge=q0_j，mj_forward；下游 body 世界位姿、
           质心、质量；I_eff = d.qM[0]（1-dof 广义质量，引擎解析，含全部刚体惯量传递）。

比较口径（B2/B3 阈值）：
  max_body_pos_err <= 1e-5 m；max_body_rot_err <= 1e-5 rad
  target joint pos err <= 1e-5 m；axis err <= 1e-5 rad（点乘 -> 角度）
  downstream mass 相对差 <= 1%；rigid I_eff 相对差 <= 1%

B.1 新增：
  - 归一化拓扑：允许忽略 fixed leaf dummy（mass<=1e-8, inertia_max<=1e-9, 无动态子树），
    其余非豁免 link 集合必须完全一致；最终 pass = normalized_topology_pass AND FK/COM pass AND mass/Ieff.
  - G0 JSON 绑定模型 hash：urdf_sha256 / mj_native_xml_sha256 / mj_matched_xml_sha256 / canonical_npz_sha256 /
    asset_equivalence_check_sha256 / asset_generator_sha256 (urdf_isolation + mj_isolation_model).
  - compare 失败时进程返回非零退出码（fail-closed）。

用法:
  # 生成 URDF 侧
  python asset_equivalence_check.py --mode urdf --joint right_elbow_pitch_joint \
      --urdf <isolation.urdf> --canonical <npz> --out <dir>
  # 生成 MJ 侧
  python asset_equivalence_check.py --mode mj --joint right_elbow_pitch_joint \
      --xml <isolation.xml> --canonical <npz> --out <dir>
  # 比对（B.1 推荐：同时传入资产路径以绑定 hash）
  python asset_equivalence_check.py --mode compare --joint <joint> --canonical <npz> \
      --urdf-json <u.json> --mj-json <m.json> --out <dir> \
      [--urdf <isolation.urdf> --xml-native <native.xml> --xml-matched <matched.xml>]
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


# ---------------- 通用 ----------------

def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else None


def rpy_to_mat(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def quat_to_mat(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def axis_rotate_mat(axis, angle):
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-30)
    x, y, z = a
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def parse_vec(s, n=3):
    return np.array([float(x) for x in s.strip().split()][:n])


def angle_between(a, b):
    a = a / (np.linalg.norm(a) + 1e-30)
    b = b / (np.linalg.norm(b) + 1e-30)
    return math.acos(min(1.0, max(-1.0, float(np.dot(a, b)))))


def load_canonical(path):
    d = np.load(path)
    return (list(d["joint_names"]), np.asarray(d["joint_pos"], dtype=np.float64),
            np.asarray(d["root_pos"], dtype=np.float64), np.asarray(d["root_quat"], dtype=np.float64))


# ---------------- URDF（mode=urdf） ----------------

def parse_urdf(urdf_path):
    import xml.etree.ElementTree as ET
    root = ET.parse(str(urdf_path)).getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    joints = []
    for j in root.findall("joint"):
        joints.append({
            "name": j.get("name"), "type": j.get("type"),
            "parent": j.find("parent").get("link") if j.find("parent") is not None else None,
            "child": j.find("child").get("link") if j.find("child") is not None else None,
            "origin": _origin(j.find("origin")),
            "axis": _axis(j.find("axis")),
        })
    return links, joints


def _origin(o):
    if o is None:
        return {"xyz": np.zeros(3), "R": np.eye(3)}
    return {"xyz": parse_vec(o.get("xyz", "0 0 0")), "R": rpy_to_mat(parse_vec(o.get("rpy", "0 0 0")))}


def _axis(a):
    if a is None:
        return np.array([1.0, 0, 0])
    return parse_vec(a.get("xyz", "1 0 0"))


def urdf_fk(urdf_path, qmap, root_pos, root_quat):
    """只把 canonical root 世界变换施加在根 link 一次（修复语义）。返回 M[link]=4x4 世界。"""
    links, joints = parse_urdf(urdf_path)
    children = {j["child"] for j in joints}
    roots = [n for n in links if n not in children]
    assert len(roots) == 1, f"URDF 期望单一根 link，实际 {roots}"
    T_root = np.eye(4)
    T_root[:3, :3] = quat_to_mat(*root_quat)
    T_root[:3, 3] = root_pos
    M = {roots[0]: T_root}
    queue = [roots[0]]
    by_parent = {}
    for j in joints:
        by_parent.setdefault(j["parent"], []).append(j)
    while queue:
        cur = queue.pop(0)
        for j in by_parent.get(cur, []):
            R = j["origin"]["R"]
            jt = j["type"]
            if jt in ("revolute", "continuous") and j["name"] in qmap:
                R = R @ axis_rotate_mat(j["axis"], qmap[j["name"]])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = j["origin"]["xyz"]
            M[j["child"]] = M[cur] @ T
            queue.append(j["child"])
    return M, links, joints, roots[0]


def urdf_downstream(urdf_path, joint, qmap, root_pos, root_quat):
    M, links, joints, root_link = urdf_fk(urdf_path, qmap, root_pos, root_quat)
    target = next(j for j in joints if j["name"] == joint)
    parent_L = target["parent"]
    Rp = M[parent_L][:3, :3]
    p_j = M[parent_L] @ np.concatenate([target["origin"]["xyz"], [1.0]])
    a_w = Rp @ target["axis"]
    a_w = a_w / (np.linalg.norm(a_w) + 1e-30)

    by_parent = {}
    for j in joints:
        by_parent.setdefault(j["parent"], []).append(j)
    down = []
    stack = [target["child"]]
    while stack:
        n = stack.pop(0)
        down.append(n)
        for j in by_parent.get(n, []):
            stack.append(j["child"])
    down_set = set(down)
    # joint by child for metadata
    joint_by_child = {j["child"]: j for j in joints}
    rows = []
    masses = []
    for ln in down:
        link = links[ln]
        inertial = link.find("inertial")
        mass = 0.0
        ixyz = np.zeros(3)
        I = np.zeros((3, 3))
        inertia_max = 0.0
        if inertial is not None:
            mass = float(inertial.find("mass").get("value"))
            iel = inertial.find("inertia")
            ixx = float(iel.get("ixx")); iyy = float(iel.get("iyy")); izz = float(iel.get("izz"))
            ixy = float(iel.get("ixy")); ixz = float(iel.get("ixz")); iyz = float(iel.get("iyz"))
            I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
            io = inertial.find("origin")
            if io is not None:
                ixyz = parse_vec(io.get("xyz", "0 0 0"))
                oR = rpy_to_mat(parse_vec(io.get("rpy", "0 0 0")))
                I = oR @ I @ oR.T
            inertia_max = float(np.max(np.abs(I)))
        ML = M[ln]
        p_com = ML @ np.concatenate([ixyz, [1.0]])
        Rl = ML[:3, :3]
        # joint type and leaf
        jinfo = joint_by_child.get(ln)
        joint_type = jinfo["type"] if jinfo is not None else "root"
        # is leaf: no child joint where parent == ln and child in down_set
        is_leaf = True
        for jj in joints:
            if jj["parent"] == ln and jj["child"] in down_set:
                is_leaf = False
                break
        row = {"link": ln, "mass": mass, "inertia_max": inertia_max, "joint_type": joint_type,
               "is_leaf": bool(is_leaf),
               "pos_w": M[ln][:3, 3].tolist(),
               "com_w": p_com[:3].tolist(), "rot_w": M[ln][:3, :3].tolist()}
        rows.append(row)
        if mass > 0:
            masses.append((ln, mass, p_com[:3].copy(), Rl @ I @ Rl.T))
    I_eff = 0.0
    for ln, mass, p_com, RIRt in masses:
        r = p_com - p_j[:3]
        I_eff += float(a_w @ RIRt @ a_w) + mass * float(np.linalg.norm(np.cross(a_w, r)) ** 2)
    return {
        "target_joint_world_pos": p_j[:3].tolist(), "target_axis_world": a_w.tolist(),
        "downstream_links": down, "rows": rows, "downstream_mass": float(sum(m[1] for m in masses)),
        "I_eff": float(I_eff)}


# ---------------- MJCF（mode=mj，WSL mujoco） ----------------

def mj_downstream(xml_path, joint, canonical):
    import mujoco
    _, q0_pol, root_pos, root_quat = canonical
    pol_names, q0_pol = canonical[0], canonical[1]
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    qadr = None
    for i in range(model.njnt):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) == joint:
            qadr = model.jnt_qposadr[i]
            break
    assert qadr is not None, f"被测 hinge {joint} 不在隔离 XML"
    data.qpos[qadr] = q0_pol[pol_names.index(joint)]
    mujoco.mj_forward(model, data)

    jid = None
    for i in range(model.njnt):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) == joint:
            jid = i
            break
    axis_local = model.jnt_axis[jid]
    child_body = model.jnt_bodyid[jid]
    p_j = data.xanchor[jid].copy()
    a_w = data.xmat[child_body].reshape(3, 3) @ axis_local
    a_w = a_w / (np.linalg.norm(a_w) + 1e-30)

    def subtree(b):
        out = [b]
        for i in range(model.nbody):
            if model.body_parentid[i] == b:
                out.extend(subtree(i))
        return out
    down_ids = subtree(child_body)
    rows = []
    masses = []
    for b in down_ids:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        mass = float(model.body_mass[b])
        p_com = data.xipos[b].copy()
        Rl = data.xmat[b].reshape(3, 3).copy()
        I_body = model.body_inertia[b]
        # inertia_max for dummy check: max diag (body frame diag)
        inertia_max = float(np.max(np.abs(I_body)))
        RIRt = Rl @ np.diag(I_body) @ Rl.T
        # is_leaf: no child body in down_ids
        is_leaf = True
        for i in range(model.nbody):
            if model.body_parentid[i] == b and i in down_ids and i != b:
                is_leaf = False
                break
        # joint type: check if body has a joint that is revolute/fixed
        # In isolation model only target joint remains, so others are fixed (no joint)
        has_joint = False
        joint_type = "fixed"
        for j in range(model.njnt):
            if model.jnt_bodyid[j] == b:
                has_joint = True
                joint_type = "revolute" if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE else "fixed"
        if not has_joint and b != child_body:
            joint_type = "fixed"
        elif b == child_body:
            joint_type = "revolute"
        rows.append({"link": name, "mass": mass, "inertia_max": inertia_max, "joint_type": joint_type,
                     "is_leaf": bool(is_leaf),
                     "pos_w": data.xipos[b].tolist(),
                     "com_w": p_com.tolist(), "rot_w": Rl.tolist()})
        if mass > 0:
            masses.append((name, mass, p_com, RIRt))
    I_eff_mass = 0.0
    for name, mass, p_com, RIRt in masses:
        r = p_com - p_j
        I_eff_mass += float(a_w @ RIRt @ a_w) + mass * float(np.linalg.norm(np.cross(a_w, r)) ** 2)
    qM0 = float(data.qM[0])
    armature0 = float(model.dof_armature[0])
    return {
        "target_joint_world_pos": p_j.tolist(), "target_axis_world": a_w.tolist(),
        "downstream_links": [r["link"] for r in rows], "rows": rows,
        "downstream_mass": float(sum(m[1] for m in masses)),
        "I_eff": float(I_eff_mass), "I_eff_qM(1dof)": qM0, "armature_j": armature0,
        "I_eff_rigid(qM-arm)": float(qM0 - armature0),
        "root_pos_npz_reference": np.asarray(canonical[2]).tolist()}


# ---------------- 汇总比较 ----------------

DUMMY_MASS_THRESH = 1e-8
DUMMY_INERTIA_THRESH = 1e-9

def _is_dummy_link(row, tol_mass=DUMMY_MASS_THRESH, tol_inertia=DUMMY_INERTIA_THRESH):
    # 规则：只允许忽略 fixed leaf；不得包含动态 joint 或动态子树；mass<=1e-8；inertia_max<=1e-9
    if row.get("joint_type") not in ("fixed", "root"):
        return False, f"joint_type={row.get('joint_type')} not fixed"
    if not row.get("is_leaf", False):
        return False, "not leaf"
    mass = float(row.get("mass", 0))
    inertia_max = float(row.get("inertia_max", 0))
    if mass > tol_mass + 1e-12:
        return False, f"mass {mass} > {tol_mass}"
    if inertia_max > tol_inertia + 1e-12:
        return False, f"inertia_max {inertia_max} > {tol_inertia}"
    return True, "ok"


def compare(urdf_json: dict, mj_json: dict, joint: str, tol: dict) -> dict:
    urdf_set = set(urdf_json["downstream_links"])
    mj_set = set(mj_json["downstream_links"])
    raw_topology_pass = bool(len(urdf_set & mj_set) >= 1 and urdf_set == mj_set)
    # 归一化拓扑：检测 extra links 是否为 dummy
    urdf_extra = sorted(list(urdf_set - mj_set))
    mj_extra = sorted(list(mj_set - urdf_set))
    urdf_rows_by = urdf_json.get("rows_by_name", {r["link"]: r for r in urdf_json.get("rows", [])})
    mj_rows_by = mj_json.get("rows_by_name", {r["link"]: r for r in mj_json.get("rows", [])})
    # For compatibility, rebuild rows_by_name if not present in mj/urdf json rows
    if not urdf_rows_by and "rows" in urdf_json:
        urdf_rows_by = {r["link"]: r for r in urdf_json["rows"]}
    if not mj_rows_by and "rows" in mj_json:
        mj_rows_by = {r["link"]: r for r in mj_json["rows"]}
    ignored_dummy_links = []
    ignored_mass_max = 0.0
    ignored_inertia_max = 0.0
    non_exempt_urdf = []
    non_exempt_mj = []
    # check urdf extra
    for ln in urdf_extra:
        row = urdf_rows_by.get(ln, {})
        # fallback: if inertia_max missing, try to infer from mass only (legacy); then treat as non-dummy
        if "inertia_max" not in row:
            non_exempt_urdf.append(ln)
            continue
        ok, reason = _is_dummy_link(row)
        if ok:
            ignored_dummy_links.append(ln)
            ignored_mass_max = max(ignored_mass_max, float(row.get("mass", 0)))
            ignored_inertia_max = max(ignored_inertia_max, float(row.get("inertia_max", 0)))
        else:
            non_exempt_urdf.append(ln)
    for ln in mj_extra:
        row = mj_rows_by.get(ln, {})
        if "inertia_max" not in row:
            non_exempt_mj.append(ln)
            continue
        ok, reason = _is_dummy_link(row)
        if ok:
            ignored_dummy_links.append(ln)
            ignored_mass_max = max(ignored_mass_max, float(row.get("mass", 0)))
            ignored_inertia_max = max(ignored_inertia_max, float(row.get("inertia_max", 0)))
        else:
            non_exempt_mj.append(ln)
    normalized_topology_pass = (len(non_exempt_urdf) == 0 and len(non_exempt_mj) == 0)
    # core matched set after ignoring dummies
    n_body_matched = len(urdf_set & mj_set)
    # Also need to ensure remaining core sets equal after ignoring dummies
    core_urdf = urdf_set - set(ignored_dummy_links)
    core_mj = mj_set - set(ignored_dummy_links)
    if core_urdf != core_mj:
        normalized_topology_pass = False

    keys = [k for k in urdf_json["rows_by_name"] if k in mj_json["rows_by_name"]]
    # Also include only core matched keys (exclude ignored)
    keys = [k for k in keys if k not in ignored_dummy_links]
    max_pos = max_rot = max_com = 0.0
    worst = None
    fk_rows = []
    for k in keys:
        u, m = urdf_json["rows_by_name"][k], mj_json["rows_by_name"][k]
        pe = float(np.max(np.abs(np.asarray(u["pos_w"]) - np.asarray(m["pos_w"]))))
        R = np.asarray(u["rot_w"]) @ np.asarray(m["rot_w"]).T
        re = float(np.linalg.norm(R - np.eye(3), ord="fro") / math.sqrt(3) / 2)
        ce = float(np.max(np.abs(np.asarray(u["com_w"]) - np.asarray(m["com_w"]))))
        fk_rows.append({"link": k, "pos_frame_origin_semantic_m": pe, "rot_err_rad": re,
                        "com_err_m": ce,
                        "pos_frame_urdf": u["pos_w"], "pos_frame_mj": m["pos_w"],
                        "mass_urdf": u["mass"], "mass_mj": m["mass"],
                        "com_urdf": u["com_w"], "com_mj": m["com_w"]})
        if pe > max_pos:
            max_pos = pe
        if re > max_rot:
            max_rot, worst = re, (k, "rot")
        if ce > max_com:
            max_com, worst = ce, (k, "com")
    axis_err = angle_between(urdf_json["target_axis_world"], mj_json["target_axis_world"])
    tpos_err = float(np.max(np.abs(np.asarray(urdf_json["target_joint_world_pos"])
                                   - np.asarray(mj_json["target_joint_world_pos"]))))
    m_rel = abs(urdf_json["downstream_mass"] - mj_json["downstream_mass"]) / max(
        abs(urdf_json["downstream_mass"]), 1e-12)
    mj_ieff = mj_json.get("I_eff_rigid(qM-arm)", mj_json["I_eff"])
    i_rel = abs(urdf_json["I_eff"] - mj_ieff) / max(abs(urdf_json["I_eff"]), 1e-12)
    fk_pass = bool(max_rot <= tol["rot"] and max_com <= tol["pos"]
                   and tpos_err <= tol["pos"] and axis_err <= tol["axis"])
    mass_pass = bool(m_rel <= tol["mass_rel"])
    ieff_pass = bool(i_rel <= tol["ieff_rel"])
    overall_pass = bool(normalized_topology_pass and fk_pass and mass_pass and ieff_pass)
    result = {
        "joint": joint,
        "raw_topology_pass": bool(raw_topology_pass),
        "topology_pass": bool(raw_topology_pass),  # legacy alias
        "normalized_topology_pass": bool(normalized_topology_pass),
        "ignored_dummy_links": ignored_dummy_links,
        "ignored_dummy_mass_max": float(ignored_mass_max),
        "ignored_dummy_inertia_max": float(ignored_inertia_max),
        "non_exempt_urdf_extra": non_exempt_urdf,
        "non_exempt_mj_extra": non_exempt_mj,
        "n_body_matched": len(keys), "downstream_urdf": urdf_json["downstream_links"],
        "downstream_mj": mj_json["downstream_links"],
        "max_body_frame_origin_semantic_diff_m": max_pos,
        "max_body_rot_err_rad": max_rot, "max_com_err_m": max_com, "worst": worst,
        "target_joint_pos_err_m": tpos_err, "target_axis_err_rad": axis_err,
        "downstream_mass_urdf": urdf_json["downstream_mass"],
        "downstream_mass_mj": mj_json["downstream_mass"], "mass_rel_diff": m_rel,
        "Ieff_urdf": urdf_json["I_eff"], "Ieff_mj(qM-arm)": float(mj_ieff),
        "Ieff_mj_qM": mj_json.get("I_eff_qM(1dof)"), "armature_mj": mj_json.get("armature_j"),
        "Ieff_rel_diff": i_rel,
        "fk_pass": bool(fk_pass),
        "mass_pass": bool(mass_pass),
        "ieff_pass": bool(ieff_pass),
        "pass": bool(overall_pass),
        "tol": tol,
        "dummy_criteria": {"mass_thresh": DUMMY_MASS_THRESH, "inertia_thresh": DUMMY_INERTIA_THRESH,
                           "rule": "fixed leaf, mass<=1e-8, inertia_max<=1e-9, no dynamic subtree"},
    }
    return result, fk_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["urdf", "mj", "compare"], required=True)
    ap.add_argument("--joint", required=True)
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--urdf", default="", help="隔离 URDF（mode=urdf 或 compare 的 URDF 路径用于 hash）")
    ap.add_argument("--xml", default="", help="MJ 隔离 XML（mode=mj）")
    ap.add_argument("--xml-native", default="", help="MJ native XML（compare 绑定）")
    ap.add_argument("--xml-matched", default="", help="MJ matched XML（compare 绑定）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--urdf-json", default="", help="mode=compare 的 URDF 侧 json")
    ap.add_argument("--mj-json", default="", help="mode=compare 的 MJ 侧 json")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical = load_canonical(args.canonical)
    tol = {"pos": 1e-5, "rot": 1e-5, "axis": 1e-5, "mass_rel": 0.01, "ieff_rel": 0.01}

    if args.mode == "urdf":
        if not args.urdf:
            print("ERROR --urdf required for mode=urdf", file=sys.stderr)
            sys.exit(2)
        pol_names, q0_pol, rp, rq = canonical
        qmap = {n: float(q0_pol[i]) for i, n in enumerate(pol_names)}
        res = urdf_downstream(args.urdf, args.joint, qmap, rp, rq)
        doc = {"engine": "urdf", "joint": args.joint, **res,
               "rows_by_name": {r["link"]: r for r in res["rows"]},
               "canonical_root_pos": rp.tolist(), "canonical_root_quat": rq.tolist(),
               "I_eff_formula": "I_eff = Σ[aᵀ R I Rᵀ a + m‖a×(p_com−p_joint)‖²]",
               "urdf_sha256": sha256_file(Path(args.urdf)),
               "canonical_npz_sha256": sha256_file(Path(args.canonical)),
               "asset_equivalence_check_sha256": sha256_file(Path(__file__))}
        (out_dir / f"asset_urdf_{args.joint}.json").write_text(
            json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / f"fk_urdf_{args.joint}.csv").write_text(_rows_csv(res["rows"]), encoding="utf-8")
        print(f"URDF: downstream_mass={res['downstream_mass']:.6f} I_eff={res['I_eff']:.6f} "
              f"target_pos={np.round(res['target_joint_world_pos'], 6)} axis={np.round(res['target_axis_world'], 6)}")
    elif args.mode == "mj":
        if not args.xml:
            print("ERROR --xml required for mode=mj", file=sys.stderr)
            sys.exit(2)
        res = mj_downstream(args.xml, args.joint, canonical)
        doc = {"engine": "mj", "joint": args.joint, **res,
               "rows_by_name": {r["link"]: r for r in res["rows"]},
               "I_eff_formula": "解析(平行轴项) + I_eff_qM(1dof) 引擎广义质量",
               "xml_sha256": sha256_file(Path(args.xml)),
               "canonical_npz_sha256": sha256_file(Path(args.canonical)),
               "asset_equivalence_check_sha256": sha256_file(Path(__file__))}
        (out_dir / f"asset_mj_{args.joint}.json").write_text(
            json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / f"fk_mj_{args.joint}.csv").write_text(_rows_csv(res["rows"]), encoding="utf-8")
        print(f"MJ: downstream_mass={res['downstream_mass']:.6f} I_eff={res['I_eff']:.6f} "
              f"I_eff_qM={res['I_eff_qM(1dof)']:.6f} target_pos={np.round(res['target_joint_world_pos'], 6)} "
              f"axis={np.round(res['target_axis_world'], 6)}")
    elif args.mode == "compare":
        if not args.urdf_json or not args.mj_json:
            print("ERROR --urdf-json and --mj-json required for mode=compare", file=sys.stderr)
            sys.exit(2)
        uj = json.load(open(args.urdf_json, encoding="utf-8"))
        mj = json.load(open(args.mj_json, encoding="utf-8"))
        # Rebuild rows_by_name if not present (legacy)
        if "rows_by_name" not in uj:
            uj["rows_by_name"] = {r["link"]: r for r in uj["rows"]}
        if "rows_by_name" not in mj:
            mj["rows_by_name"] = {r["link"]: r for r in mj["rows"]}
        result, fk_rows = compare(uj, mj, args.joint, tol)
        result["canonical_root_pos"] = uj.get("canonical_root_pos")
        result["canonical_root_quat"] = uj.get("canonical_root_quat")
        # hash binding B.1
        checker_hash = sha256_file(Path(__file__))
        # generator hashes
        gen_dir = Path(__file__).parent
        urdf_gen_hash = sha256_file(gen_dir / "urdf_isolation.py")
        mj_gen_hash = sha256_file(gen_dir / "mj_isolation_model.py")
        # B1-F1: asset hashes — require explicit paths, no fallback reuse (fail-closed)
        urdf_path = Path(args.urdf) if args.urdf else None
        xml_native_path = Path(args.xml_native) if args.xml_native else None
        xml_matched_path = Path(args.xml_matched) if args.xml_matched else None
        # 旧单参数 --xml 保持兼容但不允许单独满足 matched 绑定
        if not xml_native_path and args.xml:
            xml_native_path = Path(args.xml)
        urdf_sha = sha256_file(urdf_path) if urdf_path and urdf_path.is_file() else None
        mj_native_sha = sha256_file(xml_native_path) if xml_native_path and xml_native_path.is_file() else None
        mj_matched_sha = sha256_file(xml_matched_path) if xml_matched_path and xml_matched_path.is_file() else None
        canonical_sha = sha256_file(Path(args.canonical)) if Path(args.canonical).is_file() else None
        result["urdf_sha256"] = urdf_sha
        result["mj_native_xml_sha256"] = mj_native_sha
        result["mj_matched_xml_sha256"] = mj_matched_sha
        result["canonical_npz_sha256"] = canonical_sha
        result["asset_equivalence_check_sha256"] = checker_hash
        result["asset_generator_sha256"] = {"urdf_isolation.py": urdf_gen_hash, "mj_isolation_model.py": mj_gen_hash}
        result["hash_sources"] = {
            "urdf": str(urdf_path) if urdf_path else None,
            "xml_native": str(xml_native_path) if xml_native_path else None,
            "xml_matched": str(xml_matched_path) if xml_matched_path else None,
            "canonical": str(args.canonical),
            "checker": str(Path(__file__))
        }
        # B1-F1 required hash list (matched 强制)
        missing_hashes = []
        if not urdf_sha:
            missing_hashes.append("urdf_sha256")
        if not mj_native_sha:
            missing_hashes.append("mj_native_xml_sha256")
        if not mj_matched_sha:
            missing_hashes.append("mj_matched_xml_sha256")
        if not canonical_sha:
            missing_hashes.append("canonical_npz_sha256")
        if not checker_hash:
            missing_hashes.append("asset_equivalence_check_sha256")
        if not urdf_gen_hash or not mj_gen_hash:
            missing_hashes.append("asset_generator_sha256")
        # B1-F1 额外审计：hip 要求 native==matched；elbow 要求仅目标 armature 差异
        asset_audit = {}
        if not missing_hashes:
            if "hip" in args.joint:
                if mj_native_sha != mj_matched_sha:
                    asset_audit["hip_hash_mismatch"] = f"native {mj_native_sha[:12]} != matched {mj_matched_sha[:12]} (hip 预期 no-op 相同)"
                    missing_hashes.append("hip_native_matched_should_be_equal")
                else:
                    asset_audit["hip_native_matched_equal"] = True
            elif "elbow" in args.joint:
                if mj_native_sha == mj_matched_sha:
                    asset_audit["elbow_hash_should_differ"] = "native == matched (elbow 预期 armature 干预不同)"
                    missing_hashes.append("elbow_native_matched_should_differ")
                else:
                    # 严格审计：仅目标关节 armature 变化（ElementTree 精确比对）
                    try:
                        import xml.etree.ElementTree as ET
                        def _audit(native_p, matched_p, joint_name):
                            n_root = ET.parse(str(native_p)).getroot()
                            m_root = ET.parse(str(matched_p)).getroot()
                            n_joints = {el.get("name"): el.get("armature") for el in n_root.iter("joint") if el.get("name")}
                            m_joints = {el.get("name"): el.get("armature") for el in m_root.iter("joint") if el.get("name")}
                            if set(n_joints.keys()) != set(m_joints.keys()):
                                return False, {"reason": "joint name set differs", "native_keys": sorted(n_joints.keys()), "matched_keys": sorted(m_joints.keys())}
                            diff_keys = [k for k in n_joints if n_joints[k] != m_joints[k]]
                            # 只允许目标关节不同
                            if diff_keys != [joint_name]:
                                return False, {"diff_keys": diff_keys, "native_armatures": {k: n_joints[k] for k in diff_keys[:3]}, "matched_armatures": {k: m_joints[k] for k in diff_keys[:3]}}
                            return True, {"native_armature": n_joints[joint_name], "matched_armature": m_joints[joint_name], "diff_keys": diff_keys}
                        only_arm, detail = _audit(xml_native_path, xml_matched_path, args.joint)
                        asset_audit["elbow_armature_audit"] = detail
                        if not only_arm:
                            asset_audit["elbow_extra_diff"] = "native/matched 除目标 armature 外存在其他差异"
                            missing_hashes.append("elbow_should_only_differ_armature")
                    except Exception as e:
                        asset_audit["audit_error"] = str(e)
                        missing_hashes.append("elbow_audit_error")
        result["asset_audit"] = asset_audit
        if missing_hashes:
            result["hash_missing"] = missing_hashes
            result["pass"] = False
        (out_dir / f"asset_equivalence_{args.joint}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        if fk_rows:
            fn = out_dir / f"fk_compare_{args.joint}.csv"
            with open(fn, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(fk_rows[0].keys()))
                w.writeheader()
                w.writerows(fk_rows)
            print("fk_compare:", fn)
        print("G0_PASS" if result["pass"] else "G0_FAIL",
              json.dumps({k: result.get(k) for k in
                          ["raw_topology_pass", "normalized_topology_pass", "ignored_dummy_links",
                           "max_com_err_m", "max_body_rot_err_rad", "target_joint_pos_err_m",
                           "target_axis_err_rad", "mass_rel_diff", "Ieff_rel_diff",
                           "downstream_mass_urdf", "downstream_mass_mj", "Ieff_urdf", "Ieff_mj(qM-arm)"]}))
        if not result["pass"]:
            sys.exit(1)


def _rows_csv(rows):
    import io
    buf = io.StringIO()
    # ensure all rows have same keys; use first row's keys
    if not rows:
        return ""
    # filter to common display keys
    fieldnames = list(rows[0].keys())
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


if __name__ == "__main__":
    main()
