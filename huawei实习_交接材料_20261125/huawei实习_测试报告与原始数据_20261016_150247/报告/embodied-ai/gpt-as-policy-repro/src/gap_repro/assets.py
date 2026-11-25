"""W1 资产引用闭包：从 50 个原布局 + 机器人配置解析实际资产文件。

路径规则（E1，upstream/RoboDojo env/scene_manager/layout_manager.py）：
- 物体类 section（Rigid/Dynamic/Geometry/Articulation/Garment/Fluid）：
  Assets/Object/RoboDojo/{section}/{cat}/{idx:05d}/object.usdz，缺则 object.usd；
  inst["type"]=="cluttered" 时 section 段替换为 Clutter；inst["visual"].
  visual_usd_path 附加。
- Room：Assets/Room/{default}/ 第一个 .usd。
- Table：Assets/Material/{default}/ 第一个 .mdl。
- Background：Assets/Background/{category_name}。
- Ground：程序化几何，无资产。

本模块只输出"引用→实际文件→哈希→使用 case"表；不判定渲染等价。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

OBJECT_SECTIONS = ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid")
ASSETS_ROOT = "Assets"


def extract_layout_refs(layout: dict) -> list[dict]:
    """从单个布局 JSON 提取资产引用。返回 [{kind, candidates, ref}]。"""
    refs: list[dict] = []
    for key, value in layout.items():
        if key in OBJECT_SECTIONS and isinstance(value, dict):
            for cat, inst_list in value.items():
                if not isinstance(inst_list, list):
                    continue
                for inst in inst_list:
                    if not isinstance(inst, dict):
                        continue
                    idx = inst.get("category_idx")
                    if idx is None:
                        continue
                    section = "Clutter" if inst.get("type") == "cluttered" else key
                    base = (f"{ASSETS_ROOT}/Object/RoboDojo/{section}/{cat}/{int(idx):05d}")
                    refs.append({"kind": key.lower(), "ref": f"{section}/{cat}/{int(idx):05d}",
                                 "candidates": [f"{base}/object.usdz", f"{base}/object.usd"]})
                    visual = inst.get("visual")
                    if isinstance(visual, dict) and visual.get("visual_usd_path"):
                        vpath = str(visual["visual_usd_path"]).replace(
                            "$Robodojo_ASSETS", ASSETS_ROOT)
                        refs.append({"kind": key.lower() + "_visual",
                                     "ref": vpath, "candidates": [vpath]})
        elif key == "Room" and isinstance(value, dict) and value.get("default"):
            cat = value["default"]
            refs.append({"kind": "room", "ref": f"Room/{cat}",
                         "candidates": [f"{ASSETS_ROOT}/Room/{cat}/__DIR__"]})
        elif key == "Table" and isinstance(value, dict) and value.get("default"):
            cat = value["default"]
            refs.append({"kind": "table", "ref": f"Material/{cat}",
                         "candidates": [f"{ASSETS_ROOT}/Material/{cat}/__DIR__"]})
        elif key == "Background" and isinstance(value, dict) and value.get("category_name"):
            name = value["category_name"]
            refs.append({"kind": "background", "ref": name,
                         "candidates": [f"{ASSETS_ROOT}/Background/{name}"]})
    return refs


def build_tree_index(tree: list[dict]) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """HF tree 列表 → (path→{sha256,size}, dir→[子文件路径])。仅 file 项。"""
    by_path: dict[str, dict] = {}
    by_dir: dict[str, list[str]] = defaultdict(list)
    for f in tree:
        if f.get("type") != "file":
            continue
        p = f["path"]
        lfs = f.get("lfs") or {}
        by_path[p] = {"sha256": lfs.get("oid") or f.get("sha256") or None,
                      "size": lfs.get("size", f.get("size", 0))}
        parent = p.rsplit("/", 1)[0] if "/" in p else ""
        by_dir[parent].append(p)
    return by_path, by_dir


def resolve_refs(refs_per_case: dict[str, list[dict]],
                 tree_index: tuple[dict, dict]) -> dict:
    """把每个 case 的引用解析到实际文件。__DIR__ 规则：目录内按字典序第一个
    具备目标后缀的文件（Room→.usd，Material→.mdl）。"""
    by_path, by_dir = tree_index
    files: dict[str, dict] = {}
    unresolved: list[dict] = []
    for case_id, refs in refs_per_case.items():
        for r in refs:
            hit = None
            for cand in r["candidates"]:
                if cand.endswith("__DIR__"):
                    d = cand[:-len("__DIR__")].rstrip("/")
                    suffix = ".usd" if "/Room/" in d else (".mdl" if "/Material/" in d else "")
                    subs = sorted(p for p in by_dir.get(d, []) if p.endswith(suffix))
                    hit = subs[0] if subs else None
                elif cand in by_path:
                    hit = cand
                if hit:
                    break
            if hit is None:
                unresolved.append({"case_id": case_id, "kind": r["kind"], "ref": r["ref"],
                                   "candidates": r["candidates"]})
                continue
            ent = files.setdefault(hit, {"path": hit, "sha256": by_path[hit]["sha256"],
                                         "size": by_path[hit]["size"], "kinds": set(),
                                         "used_by": set(), "refs": set()})
            ent["kinds"].add(r["kind"])
            ent["used_by"].add(case_id)
            ent["refs"].add(r["ref"])
    return {"files": files, "unresolved": unresolved}


def add_robot_closure(files: dict[str, dict], tree_index: tuple[dict, dict],
                      all_case_ids: list[str], support_case_ids: list[str]) -> int:
    """把双 X5（全部 case）与第三 Franka（支持臂 case）的机器人资产并入闭包。

    机器人目录整体纳入（URDF/USD/mesh/configuration 互相引用，逐文件判使用面
    会漏依赖）；返回新增文件数。x5 身份依据计划 §2.2（X5A.urdf/ARX.usd/meshes）。
    """
    by_path, by_dir = tree_index
    added = 0

    def add_dir(prefix: str, case_ids: set[str], kind: str) -> None:
        nonlocal added
        for p, meta in by_path.items():
            if p.startswith(prefix):
                ent = files.setdefault(p, {"path": p, "sha256": meta["sha256"],
                                           "size": meta["size"], "kinds": set(),
                                           "used_by": set(), "refs": set()})
                if not ent["used_by"]:
                    added += 1
                ent["kinds"].add(kind)
                ent["used_by"].update(case_ids)
                ent["refs"].add(prefix)

    add_dir(f"{ASSETS_ROOT}/Robots/x5/", set(all_case_ids), "robot_x5")
    add_dir(f"{ASSETS_ROOT}/Robots/franka", set(support_case_ids), "robot_franka")
    return added


def summarize_closure(resolved: dict) -> dict:
    files = resolved["files"]
    by_kind: dict[str, int] = defaultdict(int)
    for ent in files.values():
        for k in ent["kinds"]:
            by_kind[k] += 1
    return {"n_files": len(files), "n_bytes": sum(e["size"] for e in files.values()),
            "by_kind": dict(by_kind), "n_unresolved": len(resolved["unresolved"])}
