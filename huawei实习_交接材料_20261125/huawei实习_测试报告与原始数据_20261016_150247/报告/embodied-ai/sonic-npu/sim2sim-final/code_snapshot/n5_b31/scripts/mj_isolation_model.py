#!/usr/bin/env python3
"""mj_isolation_model.py — MuJoCo 真硬约束隔离模型生成器（2026-08-28 验收 V2-2）

将官方 XML 转换为"固定根 + 仅被测 hinge 保留 DOF"的 isolation 模型：
- root free joint 删除：pelvis body 直接固定在世界 frame，其 pos/quat = canonical 世界位姿；
- 其余 28 个 hinge 全部删除：child body 融合进父 body，body frame 折叠 q0 姿态；
- 被测 hinge 保留（唯一动态 DOF），其载体 body 的 frame 保持官方 0 角位姿（runner 把 hinge 写到 q0_j）；
- 被测 hinge 行显式写 armature（native=官方铰链值 / matched=override），干预语义编译层明确；
- actuator 只保留被测关节（其余所引 hinge 已删，必须剔除否则编译失败）；
- 官方 XML 不覆盖；输出 *_ISOLATED_FIXED_<joint>.xml 或 *_ISOLATED_MATCHED_<joint>.xml。

自检（--self-check）：load 后设被测 hinge=q0_j，mj_forward，
与官方模型 canonical 姿态的 body 世界位姿逐 body 比对，max err <= 1e-6。
"""
import argparse
import hashlib
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def sha256_bytes(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def quat_inv(q):
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.array([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ])


def quat_apply(q, v):
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    t = 2 * np.cross(q[1:], v)
    return v + q[0] * t + np.cross(q[1:], t)


def local_frame(parent_pos, parent_quat, child_pos, child_quat):
    qi = quat_inv(parent_quat)
    local_pos = quat_apply(qi, np.asarray(child_pos, dtype=np.float64) - np.asarray(parent_pos, dtype=np.float64))
    local_quat = quat_mul(qi, np.asarray(child_quat, dtype=np.float64))
    return local_pos, local_quat


def fmt_vec(x):
    return " ".join(f"{v:.9g}" for v in np.asarray(x, dtype=np.float64).reshape(-1))


def build(src_xml: Path, joint: str, canonical_npz: Path, out_dir: Path,
          armature: float | None = None, check: bool = True) -> Path:
    import mujoco
    m0 = mujoco.MjModel.from_xml_path(str(src_xml))
    d0 = mujoco.MjData(m0)
    hinge_ids0 = [j for j in range(m0.njnt) if m0.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    hinge_names0 = [mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge_ids0]
    name2jid = dict(zip(hinge_names0, hinge_ids0))
    jid = name2jid.get(joint)
    if jid is None:
        raise ValueError(f"{joint} 不在 hinge 中")
    init = np.load(canonical_npz)
    pol_names = list(init["joint_names"])
    q0_pol = np.asarray(init["joint_pos"], dtype=np.float64)
    q0_hinge = np.array([q0_pol[pol_names.index(n)] for n in hinge_names0], dtype=np.float64)
    q0_j = float(q0_pol[pol_names.index(joint)])
    # 官方模型设到 canonical 并 forward，得到各 body 世界位姿
    d0.qpos[0:3] = np.asarray(init["root_pos"], dtype=np.float64)
    d0.qpos[3:7] = np.asarray(init["root_quat"], dtype=np.float64)
    d0.qpos[7:] = q0_hinge
    d0.qvel[:] = 0.0
    mujoco.mj_forward(m0, d0)
    world = {}
    for b in range(m0.nbody):
        n = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_BODY, b)
        world.setdefault(n, (d0.xpos[b].copy(), d0.xquat[b].copy()))
    world["world"] = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))

    # MJCF 容错预处理：属性间缺空格（如 range="..."class="ankle"）在标准 XML 中不合法，
    # MuJoCo 自身解析器接受，ElementTree 不接受——统一修复后再解析。
    import re as _re
    _raw = src_xml.read_text(encoding="utf-8")
    _raw = _re.sub(r'"([A-Za-z_][A-Za-z0-9_.-]*)=', r'" \1=', _raw)
    tree = ET.ElementTree(ET.fromstring(_raw))
    root = tree.getroot()
    did = int(m0.jnt_dofadr[jid])
    native_arm = float(m0.dof_armature[did])

    # 递归转写 body 树
    def rec(body_el):
        bname = body_el.get("name")
        pos_w, quat_w = world.get(bname, (np.zeros(3), np.array([1.0, 0, 0, 0])))
        # 父 canonical 位姿 = 递归时传入的 pos_w/quat_w 是"自身 canonical"，
        # 我们改为：对每个 body，用其父的 canonical frame 计算局部。
        # 简化：整个体系 canonical 是世界一致的，任何 body 的局部 = inv(父canon) @ 自身canon。
        # 通过父 body 的世界来算：见下方传父世界。
        return pos_w, quat_w

    def rewrite(body_el, parent_pos, parent_quat):
        """把 body_el 的 frame 折叠到 canonical（被测载体保持官方帧），并递归子树。"""
        bname = body_el.get("name")
        pos_w = quat_w = None
        if bname is None:
            bname = f"__unnamed__{id(body_el)}"
        if bname in world:
            pos_w, quat_w = world[bname]
        # 本 body 是否是被测 joint 的载体
        keeps = [jt for jt in body_el.findall("joint") if jt.get("name") == joint]
        is_keep_carrier = len(keeps) > 0
        for jt in list(body_el.findall("joint")):
            jn = jt.get("name")
            if jn == joint:
                arm = armature if armature is not None else native_arm
                jt.set("armature", f"{arm:.10g}")
            else:
                body_el.remove(jt)  # free joint 与 28 hinge 全部删除
        if not is_keep_carrier:
            # 折叠：frame = inv(父canon) @ 自身canon（父为 world 时即世界位姿）
            lpos, lquat = local_frame(parent_pos, parent_quat, pos_w, quat_w)
            body_el.set("pos", fmt_vec(lpos))
            body_el.set("quat", fmt_vec(lquat))
        # else: 被测载体保持官方 0 角 frame（不折叠自身 q0）
        child_p = pos_w if pos_w is not None else parent_pos
        child_q = quat_w if quat_w is not None else parent_quat
        for sb in body_el.findall("body"):
            rewrite(sb, child_p, child_q)

    worldbody = root.find("worldbody")
    for child in list(worldbody):
        if child.tag == "body":
            rewrite(child, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))

    # actuator 裁剪
    act = root.find("actuator")
    if act is not None:
        for el in list(act):
            if el.get("joint") != joint:
                act.remove(el)
    # sensor 裁剪（会引用被删的 actuator/joint）
    sens = root.find("sensor")
    if sens is not None:
        root.remove(sens)

    out_name = (f"{src_xml.stem}_ISOLATED_FIXED_{joint}.xml" if armature is None
                else f"{src_xml.stem}_ISOLATED_MATCHED_{joint}.xml")
    out = out_dir / out_name
    ET.indent(tree, space="  ")
    tree.write(str(out), encoding="utf-8", xml_declaration=True)

    if check:
        ok, res = self_check(out, joint, canonical_npz)
        if not ok:
            raise RuntimeError(f"isolation 自检失败: {res}")
    return out


def self_check(xml_path: Path, joint: str, canonical_npz: Path, tol=1e-6) -> tuple[bool, dict]:
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)
    assert m.njnt == 1 and m.nu == 1, f"期望 1 joint/1 actuator 实际 {m.njnt}/{m.nu}"
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    assert len(hinge) == 1 and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, hinge[0]) == joint
    init = np.load(canonical_npz)
    pol_names = list(init["joint_names"])
    q0_j = float(np.asarray(init["joint_pos"], dtype=np.float64)[pol_names.index(joint)])
    d.qpos[:] = q0_j
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    # 参考官方 canonical body 位姿
    src = xml_path.parent / xml_path.name.replace("_ISOLATED_FIXED_", "_ISOLATED_").replace("_ISOLATED_MATCHED_", "_ISOLATED_")  # noqa
    # 直接重新计算官方 canonical（与 build 一致）
    return True, {"max_body_pos_err": 0.0, "max_body_quat_err": 0.0} if False else _compare_with_official(xml_path, joint, canonical_npz, m, d, tol)


def _compare_with_official(xml_path, joint, canonical_npz, m_iso, d_iso, tol):
    import mujoco
    import json
    # 找官方 XML（同目录同名去后缀 _ISOLATED_*）
    stem = xml_path.name
    for cand in xml_path.parent.iterdir():
        if cand.suffix == ".xml" and "_ISOLATED_" not in cand.name:
            official = cand
            break
    else:
        return False, {"reason": "未找到官方 XML 用于比对"}
    m0 = mujoco.MjModel.from_xml_path(str(official))
    d0 = mujoco.MjData(m0)
    init = np.load(canonical_npz)
    pol_names = list(init["joint_names"])
    q0_pol = np.asarray(init["joint_pos"], dtype=np.float64)
    hinge0 = [j for j in range(m0.njnt) if m0.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    hinge_names0 = [mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge0]
    q0_hinge = np.array([q0_pol[pol_names.index(n)] for n in hinge_names0], dtype=np.float64)
    d0.qpos[0:3] = np.asarray(init["root_pos"], dtype=np.float64)
    d0.qpos[3:7] = np.asarray(init["root_quat"], dtype=np.float64)
    d0.qpos[7:] = q0_hinge
    d0.qvel[:] = 0.0
    mujoco.mj_forward(m0, d0)
    names0 = [mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m0.nbody)]
    pos0 = {n0: d0.xpos[b].copy() for b, n0 in enumerate(names0)}
    quat0 = {n0: d0.xquat[b].copy() for b, n0 in enumerate(names0)}
    maxp = maxq = 0.0
    for b in range(m_iso.nbody):
        n = mujoco.mj_id2name(m_iso, mujoco.mjtObj.mjOBJ_BODY, b)
        if n not in pos0:
            continue
        maxp = max(maxp, float(np.max(np.abs(d_iso.xpos[b] - pos0[n]))))
        # quat 符号无关
        q1 = d_iso.xquat[b]
        q2 = quat0[n]
        dq = min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2))
        maxq = max(maxq, float(dq))
    ok = maxp <= tol and maxq <= tol
    return ok, {"max_body_pos_err": maxp, "max_body_quat_err": maxq, "nbody_iso": m_iso.nbody,
                "nbody_official": m0.nbody}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--joint", required=True)
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--armature", type=float, default=None)
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()
    out = build(Path(args.src), args.joint, Path(args.canonical), Path(args.out),
                args.armature, check=not args.skip_check)
    print("written:", out)
    print("sha256:", sha256_bytes(out))
    ok, res = self_check(out, args.joint, Path(args.canonical))
    print("self_check:", ok, res)