#!/usr/bin/env python3
"""W11 前置：USDZ → OBJ → MJCF <mesh> 转换管线（场景对象物理化）。

在容器/服务器执行（需 trimesh + usd-core；本脚本依赖两者仅做 mesh 提取）。
方法：usdz 打开 → 遍历 mesh prim → 输出 OBJ（trimesh 合并）→ 生成
MJCF <asset><mesh> + <body><freejoint/><geom> 模板 → environment 加载时
按 layout JSON 注入 pos/quat。

用法：python3 scripts/w11_convert_scene_objects.py --object-dir <usdz根目录> --out-dir <OBJ输出>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def convert_usdz_to_obj(usdz_path: Path, out_obj: Path) -> dict:
    """usd(z) → 合并 mesh → OBJ。返回 {n_meshes, n_verts, n_faces}。

    点烘焙 local→world 变换（camera_stand 等资产把缩放放在 prim 层级上）；
    同一资产常含 visual+collision 两套相同网格，有 visual 时跳过 collision。
    """
    from pxr import Usd, UsdGeom
    import trimesh

    stage = Usd.Stage.Open(str(usdz_path))
    xf = UsdGeom.XformCache(Usd.TimeCode.Default())
    prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    has_visual = any("visual" in str(p.GetPath()).lower() for p in prims)
    meshes = []
    for prim in prims:
        if has_visual and "collision" in str(prim.GetPath()).lower():
            continue
        mesh = UsdGeom.Mesh(prim)
        pts_attr = mesh.GetPointsAttr().Get()
        # F05 修复：USD 的 Gf 矩阵为行向量约定（p_world = p_local @ M，
        # 平移在最后一行）。此前误用 GetTranspose()，等价于列向量左乘，
        # 父节点平移整段丢失（真实反例：translate(1,2,3) 导出 bbox 仍在
        # 原点）。oracle 测试见 tests/test_scene_assets.py（Gf.Transform 对照）。
        M = np.array(xf.GetLocalToWorldTransform(prim))
        pts = (np.c_[np.array(pts_attr, dtype=np.float64),
                     np.ones(len(pts_attr))] @ M)[:, :3]
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        faces = []
        idx = 0
        for cnt in counts:
            for k in range(1, cnt - 1):
                faces.append([indices[idx], indices[idx + k], indices[idx + k + 1]])
            idx += cnt
        meshes.append({"vertices": pts, "faces": faces})

    if not meshes:
        return {"n_meshes": 0, "n_verts": 0, "n_faces": 0}

    all_v = [v for m in meshes for v in m["vertices"]]
    all_f = []
    offset = 0
    for m in meshes:
        all_f.extend([[a + offset, b + offset, c + offset] for a, b, c in m["faces"]])
        offset += len(m["vertices"])

    tm = trimesh.Trimesh(vertices=np.array(all_v), faces=np.array(all_f))
    tm.export(str(out_obj))
    return {"n_meshes": len(meshes), "n_verts": len(tm.vertices), "n_faces": len(tm.faces)}


def generate_object_mjcf(name: str, obj_path: Path, mass: float = 0.05) -> str:
    return f'''<mujoco model="{name}">
  <asset><mesh name="{name}_mesh" file="{obj_path.name}"/></asset>
  <worldbody>
    <body name="{name}" pos="0 0 0">
      <freejoint/>
      <geom type="mesh" mesh="{name}_mesh" mass="{mass}" condim="3" friction="1.0 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object-dir", required=True, help="USDZ 根目录")
    ap.add_argument("--out-dir", required=True, help="OBJ/MJCF 输出目录")
    ap.add_argument("--mass", type=float, default=0.05)
    args = ap.parse_args()
    obj_dir = Path(args.object_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    usdzs = sorted(list(obj_dir.rglob("*.usdz")) + list(obj_dir.rglob("*.usd")))
    print(f"found {len(usdzs)} usd(z) files")
    results = {}
    for u in usdzs:
        rel = u.relative_to(obj_dir)
        stem = str(rel.with_suffix("")).replace("/", "_")
        obj_out = out_dir / f"{stem}.obj"
        info = convert_usdz_to_obj(u, obj_out)
        if info["n_verts"] > 0:
            mjcf = generate_object_mjcf(stem, obj_out, args.mass)
            (out_dir / f"{stem}.xml").write_text(mjcf)
        results[str(rel)] = info
    (out_dir / "conversion_report.json").write_text(json.dumps(results, indent=1))
    ok = sum(1 for v in results.values() if v["n_verts"] > 0)
    print(f"converted {ok}/{len(results)}; report: {out_dir / 'conversion_report.json'}")


if __name__ == "__main__":
    main()
