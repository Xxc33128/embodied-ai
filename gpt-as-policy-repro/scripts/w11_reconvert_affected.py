#!/usr/bin/env python3
"""F05 修复后重建：检出受 USD 转置 bug 影响的资产，重转并保存新旧证据。

背景（audit F05）：旧转换用 `p @ M.T`（列向量语义），父节点平移/旋转
丢失。修复后需重转受影响资产，但不得覆盖历史产物。

流程：
1. 扫描 object-dir 下全部 usd(z)，逐 mesh prim 比较 `p @ M` 与 `p @ M.T`
   是否一致；不一致者记为受影响。
2. 对每个受影响资产：记录旧 OBJ 的 sha256 与 bbox → 用修复后的转换器
   输出到 new-dir → 记录新 sha256 与 bbox。
3. 写出 JSON 证据（含扫描总数、受影响清单、逐资产新旧哈希/bbox/顶点数）。

用法：
  python3 scripts/w11_reconvert_affected.py \
      --object-dir data/organize_objects \
      --old-dir data/w11_scene_obj \
      --new-dir data/w11_scene_obj_v2 \
      --evidence docs/acceptance/w11-usd-transform-rebuild.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_CONVERTER_PATH = Path(__file__).resolve().parent / "w11_convert_scene_objects.py"


def _load_converter():
    spec = importlib.util.spec_from_file_location("w11_convert_scene_objects",
                                                  _CONVERTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def obj_bbox(path: Path):
    import trimesh

    tm = trimesh.load(str(path), force="mesh")
    return {"min": [float(x) for x in tm.bounds[0]],
            "max": [float(x) for x in tm.bounds[1]],
            "n_verts": int(len(tm.vertices)),
            "n_faces": int(len(tm.faces))}


def detect_affected(root: Path):
    """返回 (checked, raw_affected, included_affected, no_mesh, all_files)。

    raw = 任一 mesh prim 的 local-to-world 矩阵转置后结果不同。
    included = 上述且该 prim 真会被转换器收录（有 visual 时跳过 collision）。
    二者分开：影响面按 raw 记录，实际产物变化按 included 记录。
    """
    from pxr import Usd, UsdGeom

    files = sorted(root.rglob("*.usdz")) + sorted(root.rglob("*.usd"))
    raw, included, checked, no_mesh = [], [], 0, []
    for f in files:
        stage = Usd.Stage.Open(str(f))
        xf = UsdGeom.XformCache(Usd.TimeCode.Default())
        prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        if not prims:
            no_mesh.append(str(f.relative_to(root)))
            continue
        checked += 1
        has_visual = any("visual" in str(p.GetPath()).lower() for p in prims)
        raw_bad = inc_bad = False
        for prim in prims:
            M = np.array(xf.GetLocalToWorldTransform(prim))
            pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),
                             dtype=np.float64)
            if len(pts) == 0:
                continue
            sample = np.c_[pts[: min(len(pts), 32)],
                           np.ones(min(len(pts), 32))]
            if not np.allclose(sample @ M, sample @ M.T, atol=1e-9):
                raw_bad = True
                if not (has_visual and "collision" in str(prim.GetPath()).lower()):
                    inc_bad = True
        rel = str(f.relative_to(root))
        if raw_bad:
            raw.append(rel)
        if inc_bad:
            included.append(rel)
    return checked, raw, included, no_mesh, [str(p.relative_to(root)) for p in files]


def usd_included_bbox(usd_path: Path):
    """转换器收录策略下该资产的 world-space bbox（新产物 oracle）。"""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    xf = UsdGeom.XformCache(Usd.TimeCode.Default())
    prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    has_visual = any("visual" in str(p.GetPath()).lower() for p in prims)
    lo = np.array([np.inf] * 3)
    hi = np.array([-np.inf] * 3)
    for prim in prims:
        if has_visual and "collision" in str(prim.GetPath()).lower():
            continue
        pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),
                         dtype=np.float64)
        if len(pts) == 0:
            continue
        M = np.array(xf.GetLocalToWorldTransform(prim))
        w = (np.c_[pts, np.ones(len(pts))] @ M)[:, :3]
        lo = np.minimum(lo, w.min(axis=0))
        hi = np.maximum(hi, w.max(axis=0))
    return {"min": [float(x) for x in lo], "max": [float(x) for x in hi]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object-dir", required=True)
    ap.add_argument("--old-dir", required=True)
    ap.add_argument("--new-dir", required=True)
    ap.add_argument("--evidence", required=True)
    args = ap.parse_args()

    root, old_dir, new_dir = Path(args.object_dir), Path(args.old_dir), Path(
        args.new_dir)
    new_dir.mkdir(parents=True, exist_ok=True)
    converter = _load_converter()

    checked, raw_affected, included_affected, no_mesh, all_files = detect_affected(root)
    from collections import Counter
    from pxr import Usd, UsdGeom

    up_axis, meters = Counter(), Counter()
    for rel in all_files:
        st = Usd.Stage.Open(str(root / rel))
        up_axis[str(UsdGeom.GetStageUpAxis(st))] += 1
        meters[str(UsdGeom.GetStageMetersPerUnit(st))] += 1
    report = {
        "unit_audit": {"up_axis": dict(up_axis), "meters_per_unit": dict(meters),
                       "note": "MuJoCo 为 Z-up/米；与资产一致时无需换算"},
        "generated_by": "scripts/w11_reconvert_affected.py",
        "note": ("F05：旧实现 p @ M.T 丢失父节点变换。raw_affected = 任一 mesh "
                 "prim 的 local-to-world 矩阵转置后结果不同；"
                 "included_affected = 其中会被转换器实际收录的 prim（有 visual "
                 "时跳过 collision）。历史产物保持不动，候选资产新产物在 "
                 "new_dir。old_dir 产物若来自其他管线版本/源树，新旧差异不能"
                 "单独归因于本修复，见 per-asset oracle_bbox 与 provenance 说明。"),
        "object_dir": str(root),
        "old_dir": str(old_dir),
        "new_dir": str(new_dir),
        "files_scanned": len(all_files),
        "assets_checked": checked,
        "assets_without_mesh": no_mesh,
        "raw_affected": raw_affected,
        "included_affected": included_affected,
        "affected": [],
    }

    for rel in raw_affected:
        old_obj = old_dir / (rel.replace("/", "_").replace(".usdz", ".obj")
                             .replace(".usd", ".obj"))
        new_obj = new_dir / old_obj.name
        entry = {"asset": rel, "old_obj": str(old_obj), "new_obj": str(new_obj)}
        if old_obj.exists():
            entry["old_sha256"] = sha256_file(old_obj)
            entry["old_bbox"] = obj_bbox(old_obj)
        else:
            entry["old_sha256"] = None
            entry["old_bbox"] = None
            entry["warning"] = "old product missing (nothing to compare)"
        info = converter.convert_usdz_to_obj(root / rel, new_obj)
        entry["new_sha256"] = sha256_file(new_obj)
        entry["new_bbox"] = obj_bbox(new_obj)
        entry["converter_info"] = info
        oracle = usd_included_bbox(root / rel)
        entry["oracle_bbox"] = oracle
        entry["oracle_ok"] = bool(
            np.allclose(entry["new_bbox"]["min"], oracle["min"], atol=1e-5) and
            np.allclose(entry["new_bbox"]["max"], oracle["max"], atol=1e-5))
        if entry["old_bbox"] and entry["new_bbox"]:
            entry["bbox_delta_min"] = [entry["new_bbox"]["min"][i] -
                                       entry["old_bbox"]["min"][i]
                                       for i in range(3)]
            entry["bbox_delta_max"] = [entry["new_bbox"]["max"][i] -
                                       entry["old_bbox"]["max"][i]
                                       for i in range(3)]
        entry["included_in_conversion"] = rel in included_affected
        report["affected"].append(entry)

    Path(args.evidence).parent.mkdir(parents=True, exist_ok=True)
    Path(args.evidence).write_text(json.dumps(report, indent=1))
    print(f"checked {checked} assets; raw_affected {len(raw_affected)}; "
          f"included_affected {len(included_affected)}; evidence: {args.evidence}")
    for e in report["affected"]:
        print(f"  {e['asset']}: old={e.get('old_sha256')} "
              f"new={e['new_sha256']}")
        if e.get("bbox_delta_min"):
            print(f"    bbox delta min={e['bbox_delta_min']} "
                  f"max={e['bbox_delta_max']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
