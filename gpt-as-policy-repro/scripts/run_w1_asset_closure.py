#!/usr/bin/env python3
"""W1 资产引用闭包 runner：解析 50 布局 + 机器人 → 实际文件 → 哈希 → 使用 case。

产出 docs/acceptance/w1-asset-closure.json（引用闭包表）+
data/hf_assets_tree.json（Assets 树缓存）。下载闭包文件用 --download。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gap_repro import assets, inputs  # noqa: E402

REV = "91f76c28d93dd20c5fa46ce6a5a1d96a4f384acd"
# 容器里数据挂载在 /workspace/data，与 repo 分离；Mac 上是 ROOT/data。
DATA_DIR = Path(os.environ.get("GAP_REPRO_DATA", ROOT / "data"))
TREE_CACHE = DATA_DIR / "hf_assets_tree.json"
OUT = ROOT / "docs" / "acceptance" / "w1-asset-closure.json"
LAYOUT_DIR = "Assets/Eval_Layout/RoboDojo/arx_x5/0"


def hf_tree_paginated(path: str) -> list[dict]:
    """HF tree API 递归拉取（Link 头分页）。"""
    out: list[dict] = []
    cursor = None
    while True:
        url = (f"https://huggingface.co/api/datasets/{inputs.HF_DATASET}"
               f"/tree/{REV}/{path}?recursive=true&limit=1000")
        if cursor:
            url += f"&cursor={cursor}"
        req = urllib.request.Request(url, headers={"User-Agent": "gap-repro/0.1"})
        with urllib.request.urlopen(req, timeout=180) as r:
            out.extend(json.load(r))
            link = r.headers.get("Link")
        if not link or 'rel="next"' not in link:
            break
        cursor = link.split("cursor=")[1].split(">")[0].split("&")[0]
    return out


def download_closure(files: list[dict], target: Path, kinds: set[str] | None = None) -> int:
    """把闭包文件下载到 target/<path>；已存在且哈希一致的跳过。返回失败数。"""
    if kinds:
        files = [e for e in files if kinds & set(e["kinds"])]
    failed = []
    for i, ent in enumerate(files):
        dest = target / ent["path"]
        ok = False
        if dest.is_file():
            sha, _ = inputs.sha256_file(dest)
            ok = sha == ent["sha256"]
            if not ok:
                dest.unlink()
        if not ok:
            try:
                got = inputs.fetch_dataset_file(REV, ent["path"], dest, timeout=600)
                ok = got["sha256"] == ent["sha256"]
            except Exception as e:  # noqa: BLE001 — 记录后继续，最后统一报失败
                print(f"FETCH_ERR {ent['path']}: {e}", flush=True)
        if not ok:
            failed.append(ent["path"])
        if (i + 1) % 25 == 0:
            print(f"... {i + 1}/{len(files)} failed_so_far={len(failed)}", flush=True)
    print(f"DOWNLOAD_DONE ok={len(files) - len(failed)}/{len(files)}", flush=True)
    for p in failed:
        print("FAILED:", p, flush=True)
    return len(failed)


def main() -> int:
    download_target = None
    if len(sys.argv) > 2 and sys.argv[1] == "--download":
        download_target = Path(sys.argv[2])
        download_target.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "hf_cache"
    if TREE_CACHE.exists():
        tree = json.loads(TREE_CACHE.read_text())
    else:
        tree = hf_tree_paginated("Assets")
        TREE_CACHE.write_text(json.dumps(tree))
    by_path, _ = assets.build_tree_index(tree)
    n_dirs = sum(1 for f in tree if f.get("type") == "directory")
    print(f"tree: {len(by_path)} files, {n_dirs} dirs")

    panel = inputs.load_json(ROOT / "upstream/GPT-as-Policy/hybrid_rollout/"
                             "robodojo/eval_panels/robodojo_panel60_v1.json")
    scope = inputs.load_json(ROOT / "upstream/GPT-as-Policy/hybrid_rollout/"
                             "robodojo/eval_panels/robodojo_panel50_scope_v2.json")
    cases = inputs.selected_cases(panel, scope)

    refs_per_case: dict[str, list[dict]] = {}
    for c in cases:
        layout = inputs.load_json(cache / c["layout"]["path"])
        refs_per_case[c["case_id"]] = assets.extract_layout_refs(layout)
    all_case_ids = list(refs_per_case)
    support_ids = [cid for cid in all_case_ids
                   if "imitate_sorting_sequence" in cid or "make_kong" in cid]

    resolved = assets.resolve_refs(refs_per_case, (by_path, _))
    added = assets.add_robot_closure(resolved["files"], (by_path, _),
                                     all_case_ids, support_ids)
    summary = assets.summarize_closure(resolved)
    franka_dirs = sorted({p.split("/")[2] for p in by_path
                          if p.startswith("Assets/Robots/") and "franka" in p.lower()})

    for ent in resolved["files"].values():
        ent["kinds"] = sorted(ent["kinds"])
        ent["used_by"] = sorted(ent["used_by"])
        ent["refs"] = sorted(ent["refs"])

    out = {"date": "2026-09-17", "hf_revision": REV,
           "cases": len(all_case_ids), "support_cases": len(support_ids),
           "summary": summary, "robot_dirs_seen": franka_dirs,
           "robot_files_added_by_closure": added,
           "files": sorted(resolved["files"].values(), key=lambda e: e["path"]),
           "unresolved": resolved["unresolved"]}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"summary": summary, "franka_dirs": franka_dirs,
                      "unresolved": resolved["unresolved"][:5],
                      "out": str(OUT.relative_to(ROOT))}, ensure_ascii=False, indent=1))
    rc = 0 if not resolved["unresolved"] else 1
    if download_target is not None:
        kinds = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
        n_fail = download_closure(out["files"], download_target, kinds)
        rc = rc or (1 if n_fail else 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
