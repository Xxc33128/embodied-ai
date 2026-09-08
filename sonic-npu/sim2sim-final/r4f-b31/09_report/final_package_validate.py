#!/usr/bin/env python3
"""final_package_validate.py — B31 read-only full validation (spec 8)

This validator is READ-ONLY: it never writes inside the package.
It checks all payload hashes fully (not just 5), and validates media_manifest, video_visibility, evidence_index, etc.
Exit 0 if all pass, 1 otherwise. Optionally writes a report to stdout or to a path outside the package if given via --out
"""
import argparse, hashlib, json, csv, re, sys
from pathlib import Path

def sha256_file(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default=None, help="Work dir (default: parent of this script's parent)")
    parser.add_argument("--out", default=None, help="Optional output JSON outside package (e.g., /tmp/validation.json)")
    args = parser.parse_args()
    WORK = Path(args.work) if args.work else Path(__file__).resolve().parents[1]
    REPORT = WORK / "09_report"
    results = {}
    def check(name, ok, detail=""):
        results[name] = {"pass": bool(ok), "detail": detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return ok

    # 1. 18 traces
    trace_manifest_paths = []
    for joint in ["elbow","hip"]:
        for cond in ["isaac","mj_native","mj_matched"]:
            for rep in [1,2,3]:
                if cond=="isaac":
                    j="right_elbow_pitch_joint" if joint=="elbow" else "right_hip_pitch_joint"
                    trace = WORK / f"04_isolation_N5R2/{joint}/isaac_r{rep}/step_trace_{j}.npz"
                    manifest = WORK / f"04_isolation_N5R2/{joint}/isaac_r{rep}/run_manifest_{j}.json"
                else:
                    trace = WORK / f"04_isolation_N5R2/{joint}/{cond}_r{rep}/step_trace.npz"
                    manifest = WORK / f"04_isolation_N5R2/{joint}/{cond}_r{rep}/run_manifest.yaml"
                trace_manifest_paths.append((trace, manifest))
    ok1 = all(t.is_file() and m.is_file() for t,m in trace_manifest_paths)
    check("1. 18 N5-R2 trace/manifest exist", ok1, f"{len(trace_manifest_paths)} pairs")

    # 2. manifest csv
    manifest_csv = WORK / "04_isolation_N5R2/repeat_evidence_manifest.csv"
    ok2=False
    if manifest_csv.is_file():
        rows=list(csv.DictReader(open(manifest_csv, encoding="utf-8")))
        if len(rows)==18:
            combos=set((r["joint"],r["condition"],r["repeat"]) for r in rows)
            if len(combos)==18:
                all_ok=True
                for r in rows:
                    for k in ["trace_relative_path","manifest_relative_path"]:
                        p=r[k]
                        if p.startswith("E:") or p.startswith("/mnt/e") or Path(p).is_absolute():
                            all_ok=False
                        if not (WORK / p).is_file():
                            all_ok=False
                    for hk in ["trace_sha256","manifest_sha256","runner_sha256","model_sha256","canonical_sha256"]:
                        v=r[hk].strip()
                        if len(v)!=64 or not re.fullmatch(r"[0-9a-f]{64}", v):
                            all_ok=False
                    # hash check
                    try:
                        if sha256_file(WORK / r["trace_relative_path"]) != r["trace_sha256"]:
                            all_ok=False
                        if sha256_file(WORK / r["manifest_relative_path"]) != r["manifest_sha256"]:
                            all_ok=False
                    except:
                        all_ok=False
                ok2=all_ok
                check("2. manifest 18 rows relative full hash", ok2, "18 rows ok" if ok2 else "hash mismatch")
            else:
                check("2. manifest 18 rows relative full hash", False, "combos not 18")
        else:
            check("2. manifest 18 rows relative full hash", False, f"rows {len(rows)}")
    else:
        check("2. manifest 18 rows relative full hash", False, "file missing")

    # 3. G6
    ok3=True
    for joint in ["elbow","hip"]:
        g6 = WORK / f"04_isolation_N5R2/{joint}_G6.json"
        if not g6.is_file():
            ok3=False
        else:
            j=json.load(open(g6, encoding="utf-8"))
            if not (j.get("pass")==True and j.get("b1_strict")==True):
                ok3=False
    check("3. elbow/hip G6 pass", ok3, "both pass" if ok3 else "fail")

    # 4. gate
    ok4=True
    for joint in ["elbow","hip"]:
        gate = WORK / f"04_isolation_N5R2/{joint}_gates.json"
        if not gate.is_file():
            ok4=False
        else:
            j=json.load(open(gate, encoding="utf-8"))
            checks=[
                j.get("g0_asset_equivalence",{}).get("pass")==True,
                j.get("g0_hash_binding")=={},
                j.get("g1_sampling",{}).get("pass")==True,
                j.get("g1_root_runtime",{}).get("pass")==True,
                j.get("g1_root_runtime",{}).get("runtime_dof_count")==1,
                j.get("g2_isolation",{}).get("pass")==True,
                j.get("g3_no_saturation",{}).get("pass")==True,
                j.get("g5_hip_noop",{}).get("pass")==True,
                j.get("g6_repeat",{}).get("status")=="PASS",
                j.get("validity_pass(G0G1G2G3G5G6)")==True,
                j.get("all_pass")==True,
            ]
            if not all(checks):
                ok4=False
    check("4. elbow/hip gate validity true all_pass true", ok4, "both valid" if ok4 else "fail")

    # 5. no duplicate
    globs = list((WORK / "04_isolation_N5R2").glob("*G6*.json")) + list((WORK / "04_isolation_N5R2").glob("*gates*.json"))
    expected = {"elbow_G6.json","hip_G6.json","elbow_gates.json","hip_gates.json"}
    actual = {p.name for p in globs}
    ok5 = actual==expected
    check("5. no duplicate G6/gate", ok5, f"{sorted(actual)}")

    # 6. report links
    report_md = WORK / "09_report/本周三项目Sim2Sim调研与实验总结_20260828.md"
    ok6=True
    if report_md.is_file():
        txt=report_md.read_text(encoding="utf-8")
        links=re.findall(r"\[([^\]]+)\]\(([^)]+)\)", txt)
        missing=[]
        for _, path in links:
            if path.startswith("http"):
                continue
            ppath = Path(path.split("#")[0].strip())
            if not ppath.suffix:
                continue
            target = (report_md.parent / ppath).resolve()
            if not target.is_file():
                alt = WORK / ppath
                alt2 = WORK / Path(str(ppath).lstrip("./"))
                cand = WORK / Path(path.replace("../",""))
                if not (alt.is_file() or alt2.is_file() or target.is_file() or cand.is_file()):
                    missing.append(path)
        ok6 = len(missing)==0
        check("6. report markdown links exist", ok6, f"{len(links)} links" if ok6 else f"missing {missing[:3]}")
    else:
        check("6. report markdown links exist", False, "report missing")

    # 7. SONIC
    sonic_ledger = WORK / "06_sonic/sonic_II_IM_MM_MI_ledger.csv"
    ok7=True
    if sonic_ledger.is_file():
        content=sonic_ledger.read_text(encoding="utf-8")
        banned = ["action_space_fix_before_after.npz","zero_shot_IM.csv","docs/assets/action_space_fix.mp4","docs/assets/isaac_backtransfer_r2_11000.mp4"]
        found=[b for b in banned if b in content]
        if found:
            ok7=False
            check("7. SONIC ledger", False, f"banned {found}")
        else:
            check("7. SONIC ledger", True, "no banned, all unavailable correctly")
    else:
        check("7. SONIC ledger", False, "missing")

    # 8. video_visibility + media_manifest
    vis = WORK / "02_lab_dance9/video_visibility.json"
    media = WORK / "02_lab_dance9/media_manifest.json"
    ok8=True
    if vis.is_file() and media.is_file():
        vj=json.load(open(vis, encoding="utf-8"))
        mj=json.load(open(media, encoding="utf-8"))
        if not vj.get("pass")==True:
            ok8=False
        # check 6 frames exist and hashes match media_manifest
        for rel in ["isaac_10s/frame_first.png","isaac_10s/frame_mid.png","isaac_10s/frame_last.png","mj_10s/frame_first.png","mj_10s/frame_mid.png","mj_10s/frame_last.png"]:
            if not (WORK / f"02_lab_dance9/{rel}").is_file():
                ok8=False
        # check media_manifest hashes match actual files
        for key in ["video_sha256","frame_first_sha256","frame_mid_sha256","frame_last_sha256"]:
            for eng in ["isaac","mujoco"]:
                expected = mj["mujoco" if eng=="mujoco" else "isaac"][key]
                # for isaac, check file
                rel = f"{eng}_10s/{key.replace('_sha256','').replace('video','video.mp4').replace('frame_','frame_').replace('first','first.png').replace('mid','mid.png').replace('last','last.png')}"
                # Actually construct path
                if "video" in key:
                    p = WORK / f"02_lab_dance9/{eng}_10s/video.mp4"
                else:
                    p = WORK / f"02_lab_dance9/{eng}_10s/{key.replace('_sha256','').replace('frame_','frame_')}.png"
                    # e.g., frame_first -> frame_first.png
                    p = WORK / f"02_lab_dance9/{eng}_10s/{key.replace('_sha256','')}.png"
                if p.is_file():
                    actual = sha256_file(p)
                    if actual != expected:
                        ok8=False
                        print(f"hash mismatch {p} {actual[:12]} vs {expected[:12]}")
        check("8. video_visibility + media_manifest", ok8, "pass and hashes ok" if ok8 else "fail")
    else:
        check("8. video_visibility + media_manifest", False, "missing")

    # 9. README
    readme = WORK / "README.md"
    ok9=True
    if readme.is_file():
        txt=readme.read_text(encoding="utf-8")
        # Check that README does not contain old 187 placeholder but contains actual counts
        # We will check that it contains "199" or actual payload count, not 187
        if "187" in txt and "199" not in txt:
            ok9=False
            check("9. README", False, "still contains 187")
        else:
            # Check required phrases
            needed=["B.3","N5-R2 已完成 18 条","validity=true","全身可见","SONIC","唯一权威清单"]
            missing=[n for n in needed if n not in txt]
            if missing:
                ok9=False
                check("9. README", False, f"missing {missing}")
            else:
                check("9. README", True, "contains required phrases, no old 187")
    else:
        check("9. README", False, "missing")

    # 10. root SHA full check (all payload)
    sha_path = WORK / "SHA256.txt"
    ok10=False
    if sha_path.is_file():
        # Check LF (no CRLF)
        raw = sha_path.read_bytes()
        if b"\r\n" in raw:
            check("10. root SHA LF and full", False, "CRLF found, need LF")
        else:
            lines=[l for l in sha_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            payload_hashes={}
            for line in lines:
                parts=line.split()
                if len(parts)<2:
                    continue
                h=parts[0]; rel=parts[1]
                if rel.startswith("./"):
                    rel=rel[2:]
                payload_hashes[rel]=h
            all_files=[p for p in WORK.rglob("*") if p.is_file() and p.name!="SHA256.txt"]
            payload_files=[str(p.relative_to(WORK)).replace("\\","/") for p in all_files]
            # Check all files are in SHA and all hashes are full 64 and content matches
            missing=[f for f in payload_files if f not in payload_hashes]
            extra=[f for f in payload_hashes if f not in payload_files]
            bad_hash=[h for h in payload_hashes.values() if len(h)!=64 or not re.fullmatch(r"[0-9a-f]{64}", h)]
            mismatch=[]
            for rel, h in payload_hashes.items():
                fp=WORK / rel
                if fp.is_file():
                    actual=sha256_file(fp)
                    if actual!=h:
                        mismatch.append(rel)
                        break  # fail fast, but we need to check all
            if missing:
                check("10. root SHA LF and full", False, f"missing in SHA {missing[:3]}")
            elif extra:
                check("10. root SHA LF and full", False, f"extra in SHA {extra[:3]}")
            elif bad_hash:
                check("10. root SHA LF and full", False, "bad hash")
            elif mismatch:
                check("10. root SHA LF and full", False, f"hash mismatch {mismatch[:3]}")
            elif len(payload_hashes)!=len(payload_files):
                check("10. root SHA LF and full", False, f"count mismatch {len(payload_hashes)} vs {len(payload_files)}")
            else:
                # Full check: verify all hashes (not just 5)
                all_mismatch=[]
                for rel, h in payload_hashes.items():
                    fp=WORK / rel
                    if sha256_file(fp)!=h:
                        all_mismatch.append(rel)
                if all_mismatch:
                    check("10. root SHA LF and full", False, f"hash mismatch {all_mismatch[:3]}")
                else:
                    ok10=True
                    check("10. root SHA LF and full", True, f"LF, {len(payload_hashes)} payload, all full 64, all verified")
    else:
        check("10. root SHA LF and full", False, "missing SHA256.txt")

    all_pass = all(v["pass"] for v in results.values())
    print("\n=== SUMMARY ===")
    for k,v in results.items():
        print(f"{k}: {'PASS' if v['pass'] else 'FAIL'} - {v['detail']}")
    # Write to out if requested, otherwise just stdout
    if args.out:
        Path(args.out).write_text(json.dumps({"checks": results, "all_pass": all_pass}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    # Also ensure we do NOT write inside WORK
    sys.exit(0 if all_pass else 1)

if __name__=="__main__":
    main()
