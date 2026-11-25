#!/usr/bin/env python3
"""compare_repeats.py — N5-R G6 三重复确定性检查（2026-08-28 Step B.1 fail-closed 扩展）

B.1 相对 v2/v3 的修复（审查 P0-4）：
  - 每个条件 n_repeats 必须严格等于 3，否则 G6 FAIL。
  - 全部 REQUIRED_TRACE_FIELDS 必须存在且 shape 一致，缺失/shape不同/数值超阈 -> G6 FAIL。
  - 全部 REQUIRED_HASH_FIELDS 必须存在且三次一致，任一缺失/不一致 -> G6 FAIL。
  - 缺字段不再仅记 checked=false，而是直接导致该条件 FAIL。
  - smoke 单次运行时应由 summarize 给 G6=NOT_EVALUATED，不应伪造 PASS。

字段口径同 v3：t/q/dq/q_des/torque/sat/ncon/cf 全字段阈值 1e-6，整数精确相等。

用法:
  python compare_repeats.py --joint right_elbow_pitch_joint \
    --isaac-dirs r1,r2,r3 --mj-native-dirs r1,r2,r3 --mj-matched-dirs r1,r2,r3 \
    --out repeats_gates.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

def dig(d, path):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

CONT_THRESH = 1e-6
# (字段对列表, 阈值, 是否整数, 是否必需)
PAIRS = [
    (("t_state", "t_state"), CONT_THRESH, False, True),
    (("t_command_start", "t_command_start"), CONT_THRESH, False, True),
    (("q(29)", "q_pol(29)"), CONT_THRESH, False, True),
    (("dq(29)", "dq_pol(29)"), CONT_THRESH, False, True),
    (("q_des(29)", "q_des_pol(29)"), CONT_THRESH, False, True),
    (("computed_torque(29)", "tau_unclipped_pol(29)"), CONT_THRESH, False, True),
    (("applied_torque(29)", "tau_clipped_pol(29)"), CONT_THRESH, False, True),
    (("saturated(29)", "saturated(29)"), 0, True, True),
    (("ncon", "ncon"), 0, True, True),
    (("max_contact_force", "max_contact_force"), CONT_THRESH, False, True),
]
# B.1 要求全部存在且一致的 hash 字段（fail-closed）
MANIFEST_HASH_KEYS = ["hashes.runner", "hashes.model", "hashes.config", "hashes.policy",
                     "hashes.initial_state", "xml_official_sha256", "xml_used_sha256",
                     "initial_state_npz_sha256"]

# 针对 Isaac/MJ 的差异：Isaac 没有 xml_* ， MJ 没有 hashes.model 可能？B.1 要求运行器补齐后这些都应存在。
# 我们对不同引擎做条件豁免：Isaac 允许 xml_* 缺省为 NOT_APPLICABLE，但 runner/model/config/这种必须存在。
# 为严格 fail-closed，我们定义每引擎的 required hash 集合。
REQUIRED_HASH_PER_COND = {
    "isaac": ["hashes.runner", "hashes.model", "hashes.initial_state", "initial_state_npz_sha256"],
    "mj_native": ["hashes.runner", "hashes.config", "hashes.policy", "xml_official_sha256", "xml_used_sha256", "initial_state_npz_sha256"],
    "mj_matched": ["hashes.runner", "hashes.config", "hashes.policy", "xml_official_sha256", "xml_used_sha256", "initial_state_npz_sha256"],
}

def load_trace(d):
    candidates = list(d.glob("step_trace*.npz"))
    if (d / "step_trace.npz").exists():
        npz = d / "step_trace.npz"
    elif candidates:
        npz = candidates[0]
    else:
        raise FileNotFoundError(f"{d} 无 step_trace*.npz")
    data = {k: np.asarray(v) for k, v in np.load(npz, allow_pickle=False).items()}
    # also try to keep original dtype for integer checks
    return data

def manifest_of(d):
    p = d / "run_manifest.yaml"
    if p.exists():
        # Use yaml safe_load if available, otherwise fallback to manual parse
        try:
            import yaml
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
        out = {}
        cur = None
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = re.match(r"(\s+)?(\w[\w./-]*):\s*(.*)", line)
            if not m:
                continue
            indent, key, val = m.group(1) or "", m.group(2), m.group(3).strip()
            # Remove quotes
            if val.startswith("'") and val.endswith("'"):
                val = val[1:-1]
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            if indent:
                if cur is not None:
                    out.setdefault(cur, {})[key] = val
            else:
                cur = key if val == "" else None
                out[key] = {} if val == "" else val
        return out
    p = d / "run_manifest.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    mj = list(d.glob("run_manifest_*.json"))
    if mj:
        return json.loads(mj[0].read_text(encoding="utf-8"))
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", required=True)
    ap.add_argument("--isaac-dirs", required=True)
    ap.add_argument("--mj-native-dirs", required=True)
    ap.add_argument("--mj-matched-dirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest-hash-keys", default="", help="逗号分隔额外 hash 字段路径（追加到必需集合）")
    args = ap.parse_args()

    def dirs(s):
        return [Path(x.strip()) for x in s.split(",") if x.strip()]

    extra_keys = [k for k in args.manifest_hash_keys.split(",") if k.strip()]
    result = {}
    overall_ok = True
    for cond, d_list in [("isaac", dirs(args.isaac_dirs)),
                         ("mj_native", dirs(args.mj_native_dirs)),
                         ("mj_matched", dirs(args.mj_matched_dirs))]:
        cond_ok = True
        cond_detail = {}
        # B.1: n_repeats 严格 ==3
        n = len(d_list)
        cond_detail["n_repeats"] = n
        cond_detail["n_repeats_ok"] = bool(n == 3)
        if n != 3:
            print(f"[G6] FAIL {cond}: n_repeats={n} !=3 (B.1 严格三重复)")
            cond_ok = False
        for d in d_list:
            if not d.exists():
                print(f"[G6] FAIL {cond}: dir {d} 不存在")
                cond_ok = False
        # Load traces and manifests
        try:
            traces = [load_trace(d) for d in d_list]
        except Exception as e:
            print(f"[G6] FAIL {cond}: load trace error {e}")
            traces = []
            cond_ok = False
            max_diffs = {"_load_error": str(e)}
        else:
            max_diffs = {}
            is_isaac = cond == "isaac"
            # 检查每个 required field
            for (fname_pair, thr, integer, required) in PAIRS:
                key = fname_pair[0] if is_isaac else fname_pair[1]
                vals = []
                missing = False
                shape_mismatch = False
                for tr in traces:
                    if key not in tr:
                        missing = True
                        break
                    vals.append(tr[key])
                if missing:
                    max_diffs[f"{key}_checked"] = False
                    max_diffs[f"{key}_missing"] = True
                    if required:
                        cond_ok = False
                        print(f"[G6] FAIL {cond}: 缺字段 {key} (required)")
                    continue
                # shape consistency
                shapes = [v.shape for v in vals]
                if len(set(shapes)) != 1:
                    max_diffs[f"{key}_shape_mismatch"] = shapes
                    max_diffs[f"{key}_consistent"] = False
                    cond_ok = False
                    print(f"[G6] FAIL {cond}: {key} shape 不一致 {shapes}")
                    continue
                # pairwise diff
                m = 0.0
                for i in range(len(vals)):
                    for j in range(i+1, len(vals)):
                        a, b = vals[i], vals[j]
                        if integer:
                            try:
                                diff = np.max(np.abs(a.astype(np.int64) - b.astype(np.int64)))
                            except Exception:
                                diff = float("inf")
                            m = max(m, float(diff))
                        else:
                            if a.shape != b.shape:
                                m = float("inf")
                            else:
                                m = max(m, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
                max_diffs[f"max_{key}_diff"] = float(m)
                ok = (m <= (0 if integer else CONT_THRESH))
                max_diffs[f"{key}_consistent"] = bool(ok)
                max_diffs[f"{key}_checked"] = True
                if not ok:
                    cond_ok = False
                    print(f"[G6] FAIL {cond}: {key} diff {m:.3e} > thr {thr}")
                # also record missing as false
                max_diffs[f"{key}_missing"] = False
            # aggregate field check
        # hash 一致性 B.1 fail-closed
        try:
            mans = [manifest_of(d) for d in d_list]
        except Exception as e:
            mans = [{} for _ in d_list]
            cond_ok = False
            print(f"[G6] FAIL {cond}: manifest load error {e}")
        required_hashes = REQUIRED_HASH_PER_COND.get(cond, MANIFEST_HASH_KEYS) + extra_keys
        # Deduplicate
        required_hashes = list(dict.fromkeys(required_hashes))
        hash_report = {}
        hash_ok_all = True
        for hk in required_hashes:
            hv = [dig(m, hk) for m in mans]
            present = all(v is not None and str(v).strip() != "" for v in hv)
            same = False
            if present:
                try:
                    same = len(set(str(v).strip() for v in hv)) == 1
                except Exception:
                    same = False
            hash_report[hk] = {"present": bool(present), "same": bool(same),
                               "values": [str(v)[:16] if v is not None else None for v in hv],
                               "required": True}
            if not present or not same:
                hash_ok_all = False
                cond_ok = False
                print(f"[G6] FAIL {cond}: hash {hk} present={present} same={same} values={hash_report[hk]['values']}")
        # Also report any extra manifest hashes for completeness (optional)
        for hk in MANIFEST_HASH_KEYS:
            if hk not in hash_report:
                hv = [dig(m, hk) for m in mans]
                present = all(v is not None for v in hv)
                same = present and len(set(str(v) for v in hv)) == 1
                hash_report[hk] = {"present": bool(present), "same": bool(same),
                                   "values": [str(v)[:16] if v is not None else None for v in hv],
                                   "required": False}
        # Determine condition pass: must have all required fields consistent AND hashes ok AND n==3
        # flag for field consistency: all required PAIRS must be consistent
        field_consist_ok = True
        for (fname_pair, thr, integer, required) in PAIRS:
            if not required:
                continue
            key = fname_pair[0] if cond == "isaac" else fname_pair[1]
            if not max_diffs.get(f"{key}_consistent", False):
                field_consist_ok = False
        # Overall cond pass
        cond_pass = bool(cond_ok and field_consist_ok and hash_ok_all and n == 3)
        result[cond] = {
            "n_repeats": n,
            "n_repeats_ok": bool(n == 3),
            "field_max_diffs": max_diffs,
            "all_field_consistent": bool(field_consist_ok),
            "manifest_hash": hash_report,
            "all_hash_present_and_same": bool(hash_ok_all),
            "pass": bool(cond_pass),
        }
        if not cond_pass:
            overall_ok = False

    out = {"G6_repeat_determinism": result, "joint": args.joint,
           "threshold_continuous": CONT_THRESH, "threshold_integer": 0,
           "pass": bool(overall_ok),
           "b1_strict": True,
           "note": "B.1 fail-closed: n_repeats==3, all fields present/shape/threshold, all required hashes present/same"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({c: {"pass": result[c]["pass"], "n": result[c]["n_repeats"], "field_consistent": result[c]["all_field_consistent"], "hash_ok": result[c]["all_hash_present_and_same"]} for c in result}, ensure_ascii=False))
    if not overall_ok:
        for c in result:
            if not result[c]["pass"]:
                print(f"  {c} FAIL detail: n_ok={result[c]['n_repeats_ok']} field_ok={result[c]['all_field_consistent']} hash_ok={result[c]['all_hash_present_and_same']}")
        sys.exit(1)

if __name__ == "__main__":
    main()
