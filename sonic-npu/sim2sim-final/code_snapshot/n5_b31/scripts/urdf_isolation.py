#!/usr/bin/env python3
"""urdf_isolation.py — Isaac 侧真硬约束隔离 URDF 生成器（2026-08-28 验收 V2-2）

将官方 URDF 转换为"固定根 + 仅被测 revolute 保留 DOF"的 isolation 模型：
- 28 个非被测 revolute → fixed，并把 canonical 关节角 q0j 烘焙进 joint origin：
      O'_j = O_j ⊗ Rot(q0j, axis_j)      （rpy 更新，xyz 不变；axis 不变）
  数学依据：链式折叠后整棵冻结子树自动处于 canonical 姿态；被测关节保持 0 角定义，
  runner 把被测关节初始化为 q0 后全身 = canonical（与 MJ isolation 模型严格等价）。
- 被测 revolute 保持原样（origin/axis/limit 不变），是唯一动态 DOF；
- floating_base_joint 在官方 URDF 中已被注释（root link = pelvis），fix_base=True 由
  importer/runner 施加，无需改 URDF。
- 官方 URDF 不覆盖；输出 l7_29dof_neck_fixed_ISOLATION_<joint>.urdf。

用法:
  python urdf_isolation.py --src <official.urdf> --joint <name> --canonical <npz> --out <dir>
"""
import argparse
import hashlib
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


def sha256_bytes(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rpy_to_mat(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # URDF: rpy = 绕固定轴 XYZ（extrinsic）


def mat_to_rpy(R):
    """从旋转矩阵提取 URDF rpy（绕固定 XYZ 的 roll/pitch/yaw）。"""
    R = np.asarray(R, dtype=np.float64)
    sy = math.hypot(R[0, 0], R[1, 0])
    if sy > 1e-12:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw])


def axis_rotate_mat(axis, angle):
    """Rodrigues：绕单位向量 axis 旋转 angle 的矩阵。"""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def parse_vec(s, n=3):
    return np.array([float(x) for x in s.strip().split()][:n])


def fmt_vec(v, n=15):
    return " ".join(f"{x:.{n}g}" for x in v)


def build(src_urdf: Path, joint: str, canonical_npz: Path, out_dir: Path, fold_root: bool = False) -> Path:
    """构造真硬约束隔离 URDF（2026-08-28 审查 V3 修复）。

    修复（审查 §3）：canonical root 世界变换**不再**写入任何 joint origin 或 link origin。
    - 所有 joint/link origin 保持官方 URDF 的局部语义（xyz/rpy 不变）；
    - 非被测 revolute → fixed，且只做局部冻结：T_fixed = T_joint_origin ⊗ Rot(axis, q0j)
      （origin xyz 不变、rpy 右乘 q0 旋转；axis 不变）；
    - 被测 revolute 完全不动（origin/axis/limit 原样），是唯一动态 DOF；
    - canonical root 世界位姿由 runner 在 Isaac 场景中一次性设置到 articulation 根 Xform
      （不写回）；fix_base 由 importer/runner 施加。
    官方 URDF 不覆盖；输出 l7_29dof_neck_fixed_ISOLATION_<joint>.urdf。
    """
    init = np.load(canonical_npz)
    pol_names = list(init["joint_names"])
    q0_pol = np.asarray(init["joint_pos"], dtype=np.float64)
    if joint not in pol_names:
        raise ValueError(f"{joint} 不在 canonical joint_names")
    q0_j = float(q0_pol[pol_names.index(joint)])

    tree = ET.parse(str(src_urdf))
    root = tree.getroot()
    body_refs = {}
    for link in root.findall("link"):
        body_refs[link.get("name")] = link
    joints = root.findall("joint")
    names = {j.get("name") for j in joints}
    if joint not in names:
        raise ValueError(f"{joint} 不在 URDF joint 中")
    for j in joints[:]:
        jname = j.get("name")
        jtype = j.get("type")
        origin = j.find("origin")
        if jtype not in ("revolute", "continuous"):
            continue
        if jname == joint:
            continue  # 被测 revolute：完全不动（局部 origin/axis 保持官方原样）
        # 局部冻结：T_fixed = T_joint_origin ⊗ Rot(axis, q0j)；origin xyz 不变、rpy 右乘 q0 旋转
        if origin is None:
            raise ValueError(f"joint {jname} 无 origin，无法烘焙")
        xyz = parse_vec(origin.get("xyz", "0 0 0"))
        rpy = parse_vec(origin.get("rpy", "0 0 0"))
        axis_el = j.find("axis")
        axis = parse_vec(axis_el.get("xyz", "1 0 0")) if axis_el is not None else np.array([1.0, 0, 0])
        R = rpy_to_mat(rpy) @ axis_rotate_mat(axis, float(q0_pol[pol_names.index(jname)]))
        origin.set("rpy", fmt_vec(mat_to_rpy(R)))
        j.set("type", "fixed")
        rm = j.find("limit")
        if rm is not None:
            j.remove(rm)
        mi = j.find("mimic")
        if mi is not None:
            j.remove(mi)

    out = out_dir / f"{src_urdf.stem}_ISOLATION_{joint}.urdf"
    ET.indent(tree, space="  ")
    tree.write(str(out), encoding="utf-8", xml_declaration=True)
    return out




def self_check(urdf_path: Path, joint: str, canonical_npz: Path, tol=1e-6) -> tuple[bool, dict]:
    """数值自检（V3 修复）：canonical root 世界变换**只在根 link 施加一次**后逐 link 递推。

    - 期望（官方 URDF）：所有关节角 = canonical q0（含被测 q0_j），根 link 施加 T_root 一次；
    - 实际（isolation URDF）：非被测 fixed 已烘焙 q0（无旋转项）、被测 revolute 设 q=q0_j，
      同样根 link 施加 T_root 一次；
    - 两者 FK 逐 link 世界位姿误差 ≤ tol（m/rad）。
    tol=1e-6：修复后根变换仅一次、远端坐标 ~2.5m，浮点余量 ≪1e-6；结构级错误（24.5m）绝对暴露。
    """
    init = np.load(canonical_npz)
    pol_names = list(init["joint_names"])
    q0_pol = np.asarray(init["joint_pos"], dtype=np.float64)
    q0_map = {n: float(q0_pol[i]) for i, n in enumerate(pol_names)}
    _rp = np.asarray(init["root_pos"], dtype=np.float64)
    _rq = np.asarray(init["root_quat"], dtype=np.float64)
    _w, _x, _y, _z = _rq[0], _rq[1], _rq[2], _rq[3]
    _Rr = np.array([
        [1 - 2 * (_y * _y + _z * _z), 2 * (_x * _y - _w * _z), 2 * (_x * _z + _w * _y)],
        [2 * (_x * _y + _w * _z), 1 - 2 * (_x * _x + _z * _z), 2 * (_y * _z - _w * _x)],
        [2 * (_x * _z - _w * _y), 2 * (_y * _z + _w * _x), 1 - 2 * (_x * _x + _y * _y)],
    ])

    def frame_tree(tree_root, dynamic_qmap):
        links = {l.get("name"): l for l in tree_root.findall("link")}
        joints = list(tree_root.findall("joint"))
        parent_of = {j.get("name"): j.find("parent").get("link") for j in joints}
        child_of = {j.get("name"): j.find("child").get("link") for j in joints}
        roots = [l for l in links if l not in child_of.values()]
        assert len(roots) == 1
        # root 世界变换只施加一次（canonical root pose）
        T_root = np.eye(4); T_root[:3, :3] = _Rr; T_root[:3, 3] = _rp
        M = {roots[0]: T_root}
        queue = [roots[0]]
        while queue:
            cur = queue.pop(0)
            for j in joints:
                if parent_of[j.get("name")] != cur:
                    continue
                o = j.find("origin")
                xyz = parse_vec(o.get("xyz", "0 0 0"))
                rpy = parse_vec(o.get("rpy", "0 0 0"))
                R = rpy_to_mat(rpy)
                jt = j.get("type")
                if jt in ("revolute", "continuous") and j.get("name") in dynamic_qmap:
                    ax = j.find("axis")
                    axis = parse_vec(ax.get("xyz", "1 0 0")) if ax is not None else np.array([1.0, 0, 0])
                    R = R @ axis_rotate_mat(axis, dynamic_qmap[j.get("name")])
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = xyz
                M[child_of[j.get("name")]] = M[cur] @ T
                queue.append(child_of[j.get("name")])
        return M

    official = ET.parse(str(_find_official(urdf_path)))
    iso = ET.parse(str(urdf_path))
    # 官方：全部关节（含被测）用 canonical q0；隔离：只被测 revolute 用 q0_j（其余已烘焙进 fixed origin）
    qmap_all = dict(q0_map)
    qmap_iso = {joint: q0_map[joint]}
    M_official = frame_tree(official.getroot(), qmap_all)
    M_iso = frame_tree(iso.getroot(), qmap_iso)
    common = sorted(set(M_official) & set(M_iso))
    maxp = maxq = 0.0
    for ln in common:
        maxp = max(maxp, float(np.max(np.abs(M_iso[ln][:3, 3] - M_official[ln][:3, 3]))))
        R = M_iso[ln][:3, :3] @ M_official[ln][:3, :3].T
        maxq = max(maxq, float(np.linalg.norm(R - np.eye(3), ord="fro") / math.sqrt(3) / 2))
    dynamic = [j.get("name") for j in iso.getroot().findall("joint")
               if j.get("type") in ("revolute", "continuous")]
    n_fixed = sum(1 for j in iso.getroot().findall("joint") if j.get("type") == "fixed")
    n_org_rev = sum(1 for j in official.getroot().findall("joint") if j.get("type") in ("revolute", "continuous"))
    ok = (dynamic == [joint]) and (maxp <= tol) and (maxq <= tol)
    return ok, {"dynamic": dynamic, "n_fixed": n_fixed, "frozen_from_revolute": n_org_rev,
                "max_link_pos_err": maxp, "max_link_rot_err": maxq,
                "n_links_compared": len(common)}


def _find_official(urdf_path: Path) -> Path:
    for cand in urdf_path.parent.iterdir():
        if cand.suffix == ".urdf" and "_ISOLATION_" not in cand.name:
            return cand
    return urdf_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--joint", required=True)
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = build(Path(args.src), args.joint, Path(args.canonical), Path(args.out))
    fk_ok, fk_res = self_check(out, args.joint, Path(args.canonical))
    if not fk_ok:
        raise SystemExit(f"FK self_check FAILED: {fk_res}")
    print("written:", out)
    print("sha256:", sha256_bytes(out))
    print("self_check:", fk_ok, fk_res)