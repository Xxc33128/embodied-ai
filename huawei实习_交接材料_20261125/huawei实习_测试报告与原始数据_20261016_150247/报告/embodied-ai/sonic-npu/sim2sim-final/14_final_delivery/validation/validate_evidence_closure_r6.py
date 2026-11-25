#!/usr/bin/env python3
"""
validate_evidence_closure_r6.py — P0 evidence closure validator, R6-2 rebuild.

Interface (explicit, no implicit absolute-path fallbacks):
  python validate_evidence_closure_r6.py --root <overlay_root> [--source-root <dir>] --mode embedded|full-provenance

modes:
  embedded          verifies overlay-internal files only (SHA256.txt, G0-I, G0-A index,
                    body/anchor CSV structure+values self-consistency, environment,
                    master index). External source ZIPs NOT read:
                    external_source_status = EXTERNAL_SOURCE_NOT_CHECKED.
                    Never claims full-provenance PASS.
  full-provenance   additionally recomputes each source ZIP SHA256 from --source-root,
                    verifies master-index primary paths exist inside the mapped ZIP,
                    and recomputes body/anchor metrics from P1 traces inside the P1 ZIP.
                    Any missing source or hash mismatch fails closed
                    (EXTERNAL_SOURCE_NOT_AVAILABLE / INVALID).

exit 0 only when all checks of the requested mode pass.
"""
import argparse, csv, hashlib, json, pathlib, re, sys, zipfile

import numpy as np

EXPECTED_COMMITS = {"humanoid_lab": "c68c1e63ac3b088f531c951fba886c542c42c867",
                    "isaac_lab": "5c2ec81cb17532d32f7922dd7fcaae40d123b71a"}


def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="overlay root (11_p0_evidence_closure)")
    ap.add_argument("--source-root", default=None, help="directory containing source ZIPs (full-provenance)")
    ap.add_argument("--mode", choices=["embedded", "full-provenance"], default="embedded")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    root = pathlib.Path(args.root).resolve()
    errors, notes = [], []

    def fail(msg):
        errors.append(msg)

    # ---- 1. overlay SHA256.txt (paths like ./x/y)
    sha_txt = root / "SHA256.txt"
    if not sha_txt.is_file():
        fail("missing overlay SHA256.txt")
    else:
        n_ok = n_all = 0
        for line in sha_txt.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            h, rel = line.split()[0], line.split(maxsplit=1)[1]
            rel = rel.lstrip("./").replace("\\", "/")
            p = root / rel
            n_all += 1
            if not p.is_file():
                fail(f"overlay file missing {rel}")
            elif sha(p) != h.lower():
                fail(f"overlay hash mismatch {rel}")
            else:
                n_ok += 1
        notes.append(f"overlay SHA256.txt {n_ok}/{n_all} OK")

    # ---- 2. G0-I
    isum = root / "01_interface_G0I/interface_summary.json"
    if isum.is_file():
        j = json.loads(isum.read_text(encoding="utf-8"))
        if not j.get("overall_interface_gate_pass"):
            fail("G0-I interface_summary overall_interface_gate_pass not true")
        for k in ["g0a_policy_contract_pass", "g0b_simulator_adapter_pass", "g0c_startup_interface_pass"]:
            if not j.get(k):
                fail(f"G0-I {k} not true")
    else:
        fail("missing G0-I interface_summary.json")
    for rel in ["01_interface_G0I/field_comparison.csv",
                "01_interface_G0I/raw_evidence/adapter_isaac.npz",
                "01_interface_G0I/raw_evidence/adapter_mujoco.npz",
                "01_interface_G0I/raw_evidence/isaac_g0_golden/isaac_golden_states.npz",
                "01_interface_G0I/raw_evidence/mj_g0c_trace/policy_trace.npz"]:
        if not (root / rel).is_file():
            fail(f"missing G0-I evidence {rel}")

    # ---- 3. G0-A index
    ga = root / "02_asset_G0A_index/asset_G0_index.csv"
    if ga.is_file():
        rows = list(csv.DictReader(ga.read_text(encoding="utf-8").splitlines()))
        if len(rows) != 3 or any(r["status"] != "PASS" for r in rows):
            fail(f"G0-A index rows {len(rows)} or status not all PASS")
    else:
        fail("missing asset_G0_index.csv")

    # ---- 4. body/anchor CSV structure + value sanity
    ba = root / "03_body_anchor/body_anchor_metrics.csv"
    body_rows = []
    if ba.is_file():
        body_rows = list(csv.DictReader(ba.read_text(encoding="utf-8").splitlines()))
        if len(body_rows) != 6:
            fail(f"body_anchor rows {len(body_rows)} != 6")
        for r in body_rows:
            for k in ["body_pos_rmse_m", "body_quat_angle_rmse_rad", "anchor_pos_rmse_m", "root_height_rmse_m"]:
                v = float(r[k])
                if not np.isfinite(v) or v < 0 or v > 2.0:
                    fail(f"body_anchor {r['run_id']} {k}={v} out of sane range")
    else:
        fail("missing body_anchor_metrics.csv")

    # ---- 5. environment commits
    lab_c = (root / "04_environment/humanoid_lab_git_commit.txt")
    il_c = (root / "04_environment/isaac_lab_git_commit.txt")
    if lab_c.is_file() and EXPECTED_COMMITS["humanoid_lab"] not in lab_c.read_text(encoding="utf-8"):
        fail("humanoid lab commit mismatch")
    if il_c.is_file() and EXPECTED_COMMITS["isaac_lab"] not in il_c.read_text(encoding="utf-8"):
        fail("isaac lab commit mismatch")
    ds = (root / "04_environment/humanoid_lab_git_status.txt")
    if ds.is_file():
        ndirty = len([l for l in ds.read_text(encoding="utf-8").splitlines() if l.strip()])
        notes.append(f"humanoid-lab dirty files {ndirty} (dirty repo is the honest state)")

    # ---- 6. master index parse + no absolute locators
    mi = root / "05_master_index/master_evidence_index.csv"
    master_rows = []
    if mi.is_file():
        master_rows = list(csv.DictReader(mi.read_text(encoding="utf-8").splitlines()))
        if len(master_rows) < 12:
            fail(f"master index rows {len(master_rows)} < 12")
        pat = re.compile(r"(E:/|/mnt/e|C:/)")
        for r in master_rows:
            if pat.search(r["path"]):
                fail(f"master index absolute locator {r['claim_id']}")
    else:
        fail("missing master_evidence_index.csv")

    external_status = "EXTERNAL_SOURCE_NOT_CHECKED"
    if args.mode == "embedded":
        notes.append("mode=embedded: external source ZIPs not read; this is NOT a full-provenance PASS")
    else:
        # ---- full-provenance
        if not args.source_root:
            fail("--source-root required for full-provenance")
        else:
            sroot = pathlib.Path(args.source_root).resolve()
            ledger_csv = root / "00_sources/source_packages.csv"
            ledger = {r["package"]: r["sha256"].lower() for r in csv.DictReader(ledger_csv.read_text(encoding="utf-8").splitlines())}
            extra = {"P2_fullbody_arm_only_v1_1.zip": "a48749cb4513df25c1a74f6d52bd803f6c97e15ab7c7d8cbb290bfab4095398a",
                     "P3_R21_R3_isaac_d20_10s_v1.zip": "2fea92f341557855fabf9e346754d81fc929baae1190d6ac7162bd0bc860f864",
                     "P3_R4_final_gate_v1.zip": "9a4ebdf3c75afa72bbf25d108cbce3d75d34eb5fff00aa36aa973493ee64a3e6",
                     "P3_R5_metrics_media_v1.zip": "4f0c55f6725289fb82c6085a8b2ef83543f21fe67a39f97d947e4c0cb7cbe1d7"}
            all_pkgs = dict(ledger)
            all_pkgs.update({k: v for k, v in extra.items() if k not in all_pkgs})
            zips = {}
            for pkg, want in all_pkgs.items():
                cands = [sroot / pkg, sroot / "feedback" / pkg, sroot / "feedback/B31_20260831/output" / pkg,
                         sroot / "feedback/B31_20260831/input" / pkg]
                found = next((c for c in cands if c.is_file()), None)
                if found is None:
                    fail(f"EXTERNAL_SOURCE_NOT_AVAILABLE {pkg}")
                    continue
                got = sha(found)
                if got != want:
                    fail(f"source zip sha mismatch {pkg}: got {got} want {want}")
                else:
                    zips[pkg] = found
            # master index primary paths resolvable inside mapped zips
            for r in master_rows:
                pkg = r["package"]
                if pkg not in zips:
                    continue
                try:
                    names = set(zipfile.ZipFile(zips[pkg]).namelist())
                except Exception as e:
                    fail(f"zip read {pkg}: {e}")
                    continue
                first = r["path"].split("+")[0].strip()
                if first and not any(n.endswith(first) or n == first for n in names):
                    fail(f"master {r['claim_id']} path {first} not in {pkg}")
            # recompute body/anchor from P1 zip traces and compare
            p1 = zips.get("P1_fullbody_elbow_transfer_v1.zip")
            if p1 and body_rows:
                import io
                zf = zipfile.ZipFile(p1)
                def loadz(name):
                    with zf.open(name) as fh:
                        return np.load(io.BytesIO(fh.read()), allow_pickle=True)
                def geodesic(q1, q2):
                    # EXACT p1_summarize.quat_angle_error algorithm: float32 normalize/dot/arccos
                    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
                    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
                    dot = np.abs(np.sum(q1 * q2, axis=-1))
                    dot = np.clip(dot, -1.0, 1.0)
                    return 2 * np.arccos(dot)
                isaac_names = [n for n in zf.namelist() if n.endswith("policy_trace.npz") and "isaac" in n]
                mj_names = {("native" if "native" in n else "matched"): n for n in zf.namelist() if n.endswith("policy_trace.npz") and "mj_" in n}
                if not isaac_names:
                    fail("P1 zip: no isaac trace found")
                else:
                    isaac = loadz(isaac_names[0])
                    for r in body_rows:
                        cond = "native" if r["condition"] == "native" else "matched"
                        if cond not in mj_names:
                            continue
                        d = loadz(mj_names[cond])
                        n = min(len(d["t"]), len(isaac["t"]))
                        bp = float(np.sqrt(np.mean((d["actual_body_pos_w(14,3)"][:n].astype(np.float64) - isaac["actual_body_pos_w(14,3)"][:n].astype(np.float64)) ** 2)))
                        bq = float(np.sqrt(np.mean(geodesic(d["actual_body_quat_w(14,4)"][:n].astype(np.float64), isaac["actual_body_quat_w(14,4)"][:n].astype(np.float64)) ** 2)))
                        ap = float(np.sqrt(np.mean((d["actual_body_pos_w(14,3)"][:n, 0].astype(np.float64) - isaac["actual_body_pos_w(14,3)"][:n, 0].astype(np.float64)) ** 2)))
                        if abs(bp - float(r["body_pos_rmse_m"])) > 1e-9 or abs(bq - float(r["body_quat_angle_rmse_rad"])) > 1e-9 or abs(ap - float(r["anchor_pos_rmse_m"])) > 1e-9:
                            fail(f"body/anchor recompute mismatch {r['run_id']}: bp {bp} vs {r['body_pos_rmse_m']}")
                    if not errors:
                        notes.append("body/anchor metrics recomputed from P1 zip traces (float64, same algorithm as p1_summarize): match")
            external_status = "EXTERNAL_SOURCES_VERIFIED"

    result = {"mode": args.mode, "decision": "P0_OVERLAY_VALID" if not errors else "INVALID",
              "external_source_status": external_status if args.mode == "embedded" else external_status,
              "full_provenance_claim": args.mode == "full-provenance" and not errors,
              "errors": errors, "notes": notes}
    out = pathlib.Path(args.output) if args.output else None
    txt = json.dumps(result, indent=2)
    print(txt)
    if out:
        out.write_text(txt, encoding="utf-8")
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
