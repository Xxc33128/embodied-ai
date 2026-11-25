#!/usr/bin/env python3
"""W1 输入身份全量核查 runner（本机可执行部分）。

产出 docs/acceptance/w1-input-identity-evidence.json，包含：
1) 169 个 native 源文件内容哈希 vs panel60 清单；
2) 所选 50 case 的布局 JSON 下载并逐字节验证（含与上轮记录回比）；
3) 9 条支持轨迹内容级下载验证（上轮仅有 LFS 元数据）；
4) checkpoint 推理身份 17 文件的远端 LFS 元数据核对与字节数校验；
5) 聚合哈希（上游原配方）对作者侧参照值的复现校验 + 集中式验收判定。

用法：python3 scripts/run_w1_verification.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gap_repro import inputs  # noqa: E402
from gap_repro.acceptance import decide_w1_ok  # noqa: E402

PINNED = {
    "gpt_as_policy_commit": "8f3d362b077d8efb77e2a7274d5b2c20e2243846",
    "robodojo_commit": "ee67a1468510da7624a089164402359f2afc72c8",
    "xpolicylab_commit": "432f82b1758c5b1202e42a3dfe014546dbc50871",
    "hf_revision": "91f76c28d93dd20c5fa46ce6a5a1d96a4f384acd",
}
PANEL_REL = "hybrid_rollout/robodojo/eval_panels/robodojo_panel60_v1.json"
SCOPE_REL = "hybrid_rollout/robodojo/eval_panels/robodojo_panel50_scope_v2.json"
PRIOR_INPUT_JSON = Path(
    "/Users/xerxes3/Documents/huawei实习/embodied-ai/weeks/2026_0914-0920_GPT6Astra评测复现/"
    "2026-09-17-GPT-as-Policy输入一致性核查.json")
PRIOR_WEIGHT_JSON = Path(
    "/Users/xerxes3/Documents/huawei实习/embodied-ai/weeks/2026_0914-0920_GPT6Astra评测复现/"
    "2026-09-17-GPT-as-Policy权重身份元数据核查.json")
EXPECTED_PARAM_BYTES = 12_440_988_569
EXPECTED_PREV_AGGREGATE = "d15fb8bd1d29cb30b69f01b71c66596cb0293c1a8a94111b343c1580dd3e3e5b"
CKPT_SUBDIR = f"ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999"


def hf_tree(revision: str, dirpath: str) -> list[dict]:
    url = (f"https://huggingface.co/api/datasets/{inputs.HF_DATASET}"
           f"/tree/{revision}/{dirpath}?recursive=true")
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def canonical_hash_variants(obj: dict, drop_key: str) -> dict[str, str]:
    """对去掉 drop_key 后的 dict 尝试多种 JSON 规范化，供配方比对。"""
    core = {k: v for k, v in obj.items() if k != drop_key}
    outs = {}
    for name, kwargs in {
        "sort_keys": dict(sort_keys=True),
        "sort_keys_compact": dict(sort_keys=True, separators=(",", ":")),
        "sort_keys_ensure_ascii_false": dict(sort_keys=True, ensure_ascii=False),
        "sort_keys_indent2": dict(sort_keys=True, indent=2),
        "insertion_order": dict(),
    }.items():
        outs[name] = hashlib.sha256(
            json.dumps(core, **kwargs).encode("utf-8")).hexdigest()
    return outs


def main() -> int:
    report_repo = ROOT / "upstream" / "GPT-as-Policy"
    native_root = ROOT / "upstream" / "RoboDojo"
    cache = ROOT / "data" / "hf_cache"
    cache.mkdir(parents=True, exist_ok=True)

    # 1) revision 与 panel/scope 一致性
    pins_ok = {}
    import subprocess
    for name, d in [("gpt_as_policy_commit", "GPT-as-Policy"),
                    ("robodojo_commit", "RoboDojo"),
                    ("xpolicylab_commit", "XPolicyLab")]:
        head = subprocess.run(["git", "-C", str(ROOT / "upstream" / d), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        pins_ok[name] = (head == PINNED[name])
    submod = subprocess.run(
        ["git", "-C", str(native_root), "ls-tree", "HEAD", "XPolicyLab"],
        capture_output=True, text=True).stdout.split()
    submodule_pin = submod[2] if len(submod) >= 3 else None

    panel = inputs.load_json(report_repo / PANEL_REL)
    scope = inputs.load_json(report_repo / SCOPE_REL)
    identity = {
        "panel_sha256_equal_scope_rows": all(
            s["panel_sha256"] == panel["panel_sha256"] for s in scope["cases"]),
        "panel_native_source_commit_matches_pin":
            panel["native_source_commit"] == PINNED["robodojo_commit"],
        "scope_count": len(scope["cases"]),
        "panel_case_count": len(panel["cases"]),
    }

    # 2) 169 源文件内容核查
    src = inputs.verify_native_sources(native_root, panel["native_source_files"])
    # 与上轮 75 文件记录回比
    prior = json.loads(PRIOR_INPUT_JSON.read_text())
    prior_src_index = {r["path"]: r["actual_sha256"] for r in prior["results"]
                       if r["kind"] == "native_source"}
    src_regression = [
        {"path": r["path"], "prior_actual": prior_src_index[r["path"]],
         "clone_now": r["actual"], "consistent": prior_src_index[r["path"]] == r["actual"]}
        for r in src["results"] if r["path"] in prior_src_index]

    # 3) 50 布局 + 9 轨迹内容级下载验证
    prior_layout_index = {r["path"]: r["actual_sha256"] for r in prior["results"]
                          if r["kind"] == "layout"}
    prior_traj_index = {t["path"]: t["expected_sha256"]
                        for t in prior["support_trajectory_metadata"]}
    case_inputs = inputs.verify_case_inputs(
        panel, scope, PINNED["hf_revision"], cache,
        prior={"layout_index": prior_layout_index, "trajectory_index": prior_traj_index})

    # 4) checkpoint 17 文件元数据核对
    tree = hf_tree(PINNED["hf_revision"], CKPT_SUBDIR)
    ckpt_files = inputs.checkpoint_identity_files(tree, CKPT_SUBDIR)
    prior_w = json.loads(PRIOR_WEIGHT_JSON.read_text())
    prior_w_index = {f["path"]: f["sha256"] for f in prior_w["files"]}
    ckpt_cmp = []
    for f in ckpt_files:
        exp = prior_w_index.get(f["path"])
        ckpt_cmp.append({"path": f["path"], "lfs_sha256": f["lfs_sha256"],
                         "expected_prior": exp,
                         "match": (exp == f["lfs_sha256"]) if exp else None,
                         "size": f["size"]})
    params_bytes = sum(f["size"] for f in ckpt_files if f["path"].startswith("params/"))
    identity_17 = [f["path"] for f in ckpt_files]
    # 上游聚合配方（{path: sha256} 映射 + sort_keys JSON），作用于元数据哈希；
    # 必须精确复现作者侧 previous_load_sha256，否则配方漂移，立即失败。
    aggregate = inputs.aggregate_identity(
        [(f["path"], f["lfs_sha256"]) for f in ckpt_files if f["lfs_sha256"]])
    aggregate_reproduced = aggregate == EXPECTED_PREV_AGGREGATE

    # 4b) 汇总验收证据
    evidence = {
        "date": "2026-09-17",
        "pins": {"expected": PINNED, "clone_heads_match": pins_ok,
                 "robodojo_xpolicylab_submodule_pin": submodule_pin},
        "panel_scope_identity": identity,
        "panel_bytes_sha256": inputs.sha256_file(report_repo / PANEL_REL)[0],
        "scope_bytes_sha256": inputs.sha256_file(report_repo / SCOPE_REL)[0],
        "panel_canonical_variants": canonical_hash_variants(panel, "panel_sha256"),
        "scope_canonical_variants": canonical_hash_variants(scope, "scope_sha256"),
        "native_sources": {"total": len(src["results"]), "passed": src["passed"],
                           "failed": src["failed"],
                           "failures": [r for r in src["results"] if not r["match"]],
                           "regression_vs_prior": {
                               "n": len(src_regression),
                               "all_consistent": all(r["consistent"] for r in src_regression)}},
        "case_inputs": {
            "layouts": {"total": len(case_inputs["layouts"]),
                        "passed": sum(r["match"] for r in case_inputs["layouts"]),
                        "prior_regression_consistent": all(
                            r.get("prior_match") is True for r in case_inputs["layouts"])},
            "trajectories": {"total": len(case_inputs["trajectories"]),
                             "passed": sum(r["match"] for r in case_inputs["trajectories"]),
                             "records": case_inputs["trajectories"]}},
        "checkpoint_metadata": {
            "subdir": CKPT_SUBDIR,
            "identity_files_n": len(identity_17),
            "params_bytes": params_bytes,
            "params_bytes_expected": EXPECTED_PARAM_BYTES,
            "params_bytes_match": params_bytes == EXPECTED_PARAM_BYTES,
            "all_match_prior": all(c["match"] for c in ckpt_cmp),
            "per_file": ckpt_cmp,
            "aggregate": aggregate,
            "aggregate_recipe": ("上游原配方：sha256(json.dumps({path: sha256}, "
                                 "sort_keys=True))，经作者 previous_load_sha256 验证"),
            "aggregate_reproduced": aggregate_reproduced,
            "expected_prior_aggregate": EXPECTED_PREV_AGGREGATE,
            "note": "17 文件内容尚未下载（下载在 NPU 服务器执行）；此为 LFS 元数据级核对。"
                    "内容级验证在服务器下载完成后重跑，聚合配方不变。",
        },
    }
    out = ROOT / "docs" / "acceptance" / "w1-input-identity-evidence.json"
    out.write_text(json.dumps(evidence, ensure_ascii=False, indent=1))
    print(json.dumps({
        "pins_ok": pins_ok, "submodule_pin_ok": submodule_pin == PINNED["xpolicylab_commit"],
        "native_sources": f'{src["passed"]}/{len(src["results"])}',
        "layouts": evidence["case_inputs"]["layouts"],
        "trajectories": {k: v for k, v in evidence["case_inputs"]["trajectories"].items()
                         if k != "records"},
        "params_bytes_match": params_bytes == EXPECTED_PARAM_BYTES,
        "ckpt_all_match_prior": evidence["checkpoint_metadata"]["all_match_prior"],
        "evidence": str(out.relative_to(ROOT)),
    }, ensure_ascii=False, indent=1))
    # 5) 验收判定（集中式，见 gap_repro.acceptance；不得在 runner 里临时拼布尔式）
    ok, reasons = decide_w1_ok(
        pins_ok=pins_ok,
        submodule_pin=submodule_pin,
        expected_submodule_pin=PINNED["xpolicylab_commit"],
        native_sources={"total": len(src["results"]), "failed": src["failed"]},
        expected_source_total=len(panel["native_source_files"]),
        layouts=evidence["case_inputs"]["layouts"],
        expected_layout_total=len(scope["cases"]),
        trajectories=evidence["case_inputs"]["trajectories"],
        checkpoint={"all_match_prior": evidence["checkpoint_metadata"]["all_match_prior"],
                    "params_bytes_match": evidence["checkpoint_metadata"]["params_bytes_match"]},
        aggregate_reproduced=aggregate_reproduced,
    )
    if identity["panel_sha256_equal_scope_rows"] is not True:
        ok = False
        reasons.append("scope 各行的 panel_sha256 与源 panel 不一致")
    print(json.dumps({"ok": ok, "reasons": reasons}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
