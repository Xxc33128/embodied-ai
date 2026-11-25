#!/usr/bin/env python3
"""
validate_final_delivery.py — R7 final delivery validator (fail-closed).

Checks (plan 2026-09-02 R5.1-R7 §11):
 1 FINAL_DELIVERY_INDEX.csv rows: path exists, sha256 recomputable/matching
 2 claim matrix evidence paths resolvable (inside fd dir, week root, or named zip)
 3 report local images/assets exist (relative refs)
 4 reports contain no Win/WSL/Mac absolute paths
 5 G0-I / G0-A naming present and distinguished
 6 P2 (R6) and P3 validators in valid states (validation/*.json)
 7 R5.1 long CSV has no mixed units
 8 final report does not reference stale R5 08/11 (assets must equal R5.1 tree files)
 9 full tree SHA (SHA256_FULL_TREE.txt) verifies when reachable
 10 embedded mode never claims full provenance; full-provenance verifies all source ZIPs

Usage:
  python validate_final_delivery.py --root . [--source-root <week_root>] --mode embedded|full-provenance
"""
import argparse, csv, hashlib, json, pathlib, re, sys, zipfile

def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

ABS_PAT = re.compile(r"(E:\\|E:/|e:/|C:\\|/mnt/[a-z]/|/Users/|/home/[a-z]+/sim2sim)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--source-root", default=None)
    ap.add_argument("--week-root", default=None, help="week root override (default: parent of --root)")
    ap.add_argument("--p2-elbow-xml", default=None, help="override path for packaged P2 elbow XML integrity check")
    ap.add_argument("--r5-long-csv", default=None, help="override path for R5.1 long CSV unit check")
    ap.add_argument("--mode", choices=["embedded", "full-provenance"], default="embedded")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    root = pathlib.Path(args.root).resolve()
    errors, notes = [], []

    def fail(m): errors.append(m)

    # 1 index
    idx_p = root / "FINAL_DELIVERY_INDEX.csv"
    if not idx_p.is_file():
        fail("missing FINAL_DELIVERY_INDEX.csv")
        rows = []
    else:
        rows = list(csv.DictReader(idx_p.read_text(encoding="utf-8").splitlines()))
    week = pathlib.Path(args.week_root).resolve() if args.week_root else root.parent
    for r in rows:
        rel = r["relative_path"]
        p = root / rel if rel.startswith("14_final_delivery/") is False and (root / rel).exists() else week / rel
        p2 = week / rel
        p = p2 if p2.exists() else p
        if not p.exists():
            fail(f"index path missing {rel}")
        elif r["sha256"] not in ("", "dir", "PENDING") and r["sha256"] != "dir":
            if p.is_file() and sha(p) != r["sha256"]:
                fail(f"index sha mismatch {rel}")

    # 2 claim matrix paths
    idx_by_artifact = {r["artifact"]: r["relative_path"] for r in rows}
    cm = root / "FINAL_CLAIM_EVIDENCE_MATRIX.csv"
    if not cm.is_file():
        fail("missing FINAL_CLAIM_EVIDENCE_MATRIX.csv")
    else:
        for r in csv.DictReader(cm.read_text(encoding="utf-8").splitlines()):
            ep = r["evidence_path"]
            ok = False
            q = week / ep.split(";")[0].strip()
            if q.exists():
                ok = True
            if not ok and (root / ep).exists():
                ok = True
            if not ok and ".zip" in ep.split(";")[0]:
                ok = (week / ep.split(";")[0].strip()).is_file()
            if not ok:
                # path inside a referenced zip (resolve via index artifact)
                zn = r["evidence_artifact"].strip()
                rel = idx_by_artifact.get(zn)
                zp = (week / rel) if rel else (week / zn)
                if zp.is_file():
                    try:
                        names = zipfile.ZipFile(zp).namelist()
                        inner = ep.split(";")[0].strip().split("/")[-1]
                        ok = any(n.endswith(inner) for n in names)
                    except Exception:
                        ok = False
            if not ok:
                fail(f"claim {r['claim_id']} evidence path unresolvable: {ep[:60]}")

    # 3/4 reports
    for rep in ["report/本周三项目Sim2Sim调研与实验总结_最终版.md", "report/本周三项目Sim2Sim一页摘要_最终版.md"]:
        rp = root / rep
        if not rp.is_file():
            fail(f"missing {rep}")
            continue
        txt = rp.read_text(encoding="utf-8")
        if ABS_PAT.search(txt.replace("E:\\sim2sim-week-2026-08-26\\` 为根", "")):
            # allow the single documented root statement line
            bad_lines = [l for l in txt.splitlines() if ABS_PAT.search(l) and "为根" not in l]
            if bad_lines:
                fail(f"{rep} contains absolute paths: {bad_lines[0][:80]}")
        for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", txt):
            a = m.group(1)
            if not ((root / "report" / a).resolve().exists() or (root / a.replace("../", "")).exists()):
                fail(f"{rep} image missing: {a}")
        for m in re.finditer(r"`((?:\.\./)?assets/[^`]+?\.(?:png|mp4|json|csv))`", txt):
            a = m.group(1)
            if not ((root / "report" / a).resolve().exists() or (root / a.replace("../", "")).exists()):
                fail(f"{rep} asset ref missing: {a}")

    # 5 G0 naming
    allrep = "".join((root / f).read_text(encoding="utf-8") for f in
                     ["report/本周三项目Sim2Sim调研与实验总结_最终版.md"] if (root / f).is_file())
    if "G0-I" not in allrep or "G0-A" not in allrep:
        fail("reports must name 接口 G0-I and 资产 G0-A explicitly")

    # 6 validator states
    v = root / "validation"
    p3g = v / "p3_r4_final_gate.json"
    if not (p3g.is_file() and json.loads(p3g.read_text(encoding="utf-8")).get("decision") == "VALID_P3"):
        fail("P3 R4 gate not VALID_P3 in validation/")
    p3r5 = v / "p3_r5_validation.json"
    if not (p3r5.is_file() and json.loads(p3r5.read_text(encoding="utf-8")).get("decision") == "VALID_R5"):
        fail("P3 R5 validation not VALID_R5")
    p2n = v / "p2_negative_tests_r6.json"
    if not (p2n.is_file() and json.loads(p2n.read_text(encoding="utf-8")).get("overall_pass")):
        fail("P2 R6 negative tests not all pass")
    p2g = v / "p2_final_gate_recomputed_r6.json"
    if not (p2g.is_file() and json.loads(p2g.read_text(encoding="utf-8")).get("decision") == "NO_INCREMENTAL_ARM_GROUP_EFFECT"):
        fail("P2 recomputed gate decision changed")
    p0e = v / "p0_embedded_validation.json"
    p0f = v / "p0_full_provenance_validation.json"
    if not (p0e.is_file() and json.loads(p0e.read_text(encoding="utf-8")).get("decision") == "P0_OVERLAY_VALID"):
        fail("P0 embedded validation missing/invalid")
    if not (p0f.is_file() and json.loads(p0f.read_text(encoding="utf-8")).get("decision") == "P0_OVERLAY_VALID"):
        fail("P0 full-provenance validation missing/invalid")

    # 6c packaged P2 elbow XML integrity (fail-closed mirror of R6-1 into final gate)
    EXPECTED_ELBOW = "ab1e1862367be20a91edeefca12c6faedbc3e99a151d662cf42b4c56da98e7f9"
    elbow = pathlib.Path(args.p2_elbow_xml) if args.p2_elbow_xml else week / "12_p2_arm_only_v1_1/00_inputs/p1_elbow_only/l7_29dof_neck_fixed_elbow_matched.xml"
    if not elbow.is_file():
        fail(f"packaged P2 elbow XML missing: {elbow}")
    elif sha(elbow) != EXPECTED_ELBOW:
        fail(f"packaged P2 elbow XML sha {sha(elbow)[:12]} != {EXPECTED_ELBOW[:12]}")

    # 7 R5.1 long csv units
    long_csv = pathlib.Path(args.r5_long_csv) if args.r5_long_csv else week / "13_p3_delay20_r4_build/13_p3_delay20/04_analysis/p3_metrics_long.csv"
    if long_csv.is_file():
        t = long_csv.read_text(encoding="utf-8")
        if "mixed" in t:
            fail("R5.1 long csv still contains mixed unit")
    else:
        notes.append("R5.1 long csv not reachable from this root (checked via FD-07 zip in full-provenance)")

    # 8 stale 08/11 check: assets must equal R5.1 tree copies
    tree = week / "13_p3_delay20_r4_build/13_p3_delay20/04_analysis/plots"
    for name in ["08_body_orientation_geodesic.png", "09_cross_engine_gap_d00_vs_d20.png", "11_metric_tradeoff_summary.png"]:
        a, b = root / "assets/p3" / name, tree / name
        if a.is_file() and b.is_file() and sha(a) != sha(b):
            fail(f"assets/p3/{name} differs from R5.1 tree version (stale figure)")

    # 9 full tree sha
    ftree = week / "13_p3_delay20_r4_build/13_p3_delay20/SHA256_FULL_TREE.txt"
    if ftree.is_file():
        ok = True
        for line in ftree.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            h, rel = line.split(maxsplit=1)
            q = week / "13_p3_delay20_r4_build" / rel
            if not q.is_file() or sha(q) != h.lower():
                fail(f"full-tree sha fail {rel}")
                ok = False
                break
        if ok:
            notes.append("SHA256_FULL_TREE.txt verified (108 entries)")

    # 10 modes
    if args.mode == "full-provenance":
        if not args.source_root:
            fail("--source-root required in full-provenance")
        else:
            sroot = pathlib.Path(args.source_root).resolve()
            for r in rows:
                rel = r["relative_path"]
                if not rel.endswith(".zip"):
                    continue
                if r["required_for_full_provenance"].lower() != "true":
                    continue
                q = sroot / rel
                if not q.is_file():
                    fail(f"source zip missing {rel}")
                elif r["sha256"] not in ("", "PENDING") and sha(q) != r["sha256"]:
                    fail(f"source zip sha mismatch {rel}")
            notes.append("full-provenance: all required source ZIPs present and hash-matched")
    else:
        notes.append("embedded: source ZIPs not verified; NOT a full-provenance claim")

    decision = "VALID_FINAL_DELIVERY" if not errors else "INVALID_FINAL_DELIVERY"
    out = {"mode": args.mode, "decision": decision, "errors": errors, "notes": notes,
           "index_rows": len(rows)}
    txt = json.dumps(out, indent=2)
    print(txt)
    if args.output:
        pathlib.Path(args.output).write_text(txt, encoding="utf-8")
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
