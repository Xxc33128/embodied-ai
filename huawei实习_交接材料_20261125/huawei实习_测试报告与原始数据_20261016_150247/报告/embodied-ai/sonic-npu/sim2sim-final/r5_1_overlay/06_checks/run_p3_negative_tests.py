#!/usr/bin/env python3
"""
run_p3_negative_tests.py — R4 five-stage: 8 REAL tampering negative tests.

Each case:
  1. copies the whole R4 root (13_p3_delay20) to a temp dir;
  2. tampers exactly one file/field;
  3. invokes the SAME validator: python 04_analysis/p3_validate.py --root <temp>;
  4. requires non-zero exit AND expected decision;
  5. stores command/exit/stdout-stderr summary in p3_negative_tests.json.

NO mock validator: the real fail-closed gate must catch each tamper.
"""
import argparse, json, pathlib, shutil, subprocess, sys, tempfile

import numpy as np
import yaml

VAL_REL = "04_analysis/p3_validate.py"

EXPECT = {
    "T1_effective_tampered": "INVALID_DELAY_IMPLEMENTATION",
    "T2_missing_effective_substep": "INVALID_TRACE",
    "T3_requested_delay_15ms": "INVALID_DELAY_IMPLEMENTATION",
    "T4_canonical_hash": "INVALID_ARTIFACT_HASH",
    "T5_model_hash": "INVALID_ARTIFACT_HASH",
    "T6_deleted_cycle": "INVALID_TRACE",
    "T7_phys_index_broken": "INVALID_DELAY_IMPLEMENTATION",
    "T8_seed_parity": "INVALID_BASELINE_PAIRING",
}


def rewrite_npz(path, fn):
    d = dict(np.load(str(path), allow_pickle=True))
    fn(d)
    np.savez_compressed(str(path), **d)


def tamperers():
    T = {}

    def t1(r):
        p = r / "03_runs/isaac_d20_10s_r2/policy_trace.npz"
        def f(d):
            d["q_des_effective_substep"][50, 0, 0] += 0.5
            d["q_des_effective(29)"][50, 0] += 0.5
        rewrite_npz(p, f)
    T["T1_effective_tampered"] = t1

    def t2(r):
        p = r / "03_runs/isaac_d20_10s_r2/policy_trace.npz"
        def f(d):
            del d["q_des_effective_substep"]
        rewrite_npz(p, f)
    T["T2_missing_effective_substep"] = t2

    def t3(r):
        p = r / "03_runs/isaac_d20_10s_r2/run_manifest.yaml"
        m = yaml.safe_load(p.read_text(encoding="utf-8"))
        m["requested_delay_ms"] = 15.0
        p.write_text(yaml.safe_dump(m, allow_unicode=True, sort_keys=False), encoding="utf-8")
    T["T3_requested_delay_15ms"] = t3

    def t4(r):
        p = r / "03_runs/isaac_d20_10s_r2/run_manifest.yaml"
        m = yaml.safe_load(p.read_text(encoding="utf-8"))
        m["hashes"]["canonical"] = "f" * 64
        m["initial_state_npz_sha256"] = "f" * 64
        p.write_text(yaml.safe_dump(m, allow_unicode=True, sort_keys=False), encoding="utf-8")
    T["T4_canonical_hash"] = t4

    def t5(r):
        p = r / "03_runs/isaac_d20_10s_r2/run_manifest.yaml"
        m = yaml.safe_load(p.read_text(encoding="utf-8"))
        m["model_asset_sha256"] = "0" * 64
        m["hashes"]["model"] = "0" * 64
        p.write_text(yaml.safe_dump(m, allow_unicode=True, sort_keys=False), encoding="utf-8")
    T["T5_model_hash"] = t5

    def t6(r):
        p = r / "03_runs/isaac_d20_10s_r2/policy_trace.npz"
        def f(d):
            for k, v in list(d.items()):
                a = np.asarray(v)
                if a.ndim >= 1 and a.shape[0] == 500:
                    d[k] = a[:499]
        rewrite_npz(p, f)
    T["T6_deleted_cycle"] = t6

    def t7(r):
        p = r / "03_runs/isaac_d20_10s_r2/policy_trace.npz"
        def f(d):
            idx = d["actuator_physics_step_index_substep"].copy()
            idx.reshape(-1)[10], idx.reshape(-1)[11] = idx.reshape(-1)[11], idx.reshape(-1)[10]
            # ensure a real discontinuity
            idx2 = d["actuator_physics_step_index_substep"]
            idx2[100, 0] = 999999
        rewrite_npz(p, f)
    T["T7_phys_index_broken"] = t7

    def t8(r):
        p = r / "03_runs/mj_elbow_d00_10s_reference/run_manifest.yaml"
        m = yaml.safe_load(p.read_text(encoding="utf-8"))
        m["seed"] = 41
        p.write_text(yaml.safe_dump(m, allow_unicode=True, sort_keys=False), encoding="utf-8")
    T["T8_seed_parity"] = t8

    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--python", type=str, default=None)
    args = ap.parse_args()
    root = pathlib.Path(args.root).resolve()
    py = args.python or sys.executable
    results = []
    T = tamperers()
    # sanity: unmodified root must be VALID_P3 with the same validator
    for name in EXPECT:
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="p3neg_"))
        try:
            dest = tmp / "13_p3_delay20"
            shutil.copytree(str(root), str(dest))
            T[name](dest)
            cmd = [py, str(dest / VAL_REL), "--root", str(dest)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            decision = None
            fg = dest / "04_analysis/p3_final_gate.json"
            if fg.is_file():
                try:
                    decision = json.loads(fg.read_text(encoding="utf-8")).get("decision")
                except Exception:
                    decision = "UNREADABLE"
            # a control sanity check on the FIRST iteration: validator must also PASS on untouched copy?
            ok_nonzero = res.returncode != 0
            ok_reason = decision == EXPECT[name]
            passed = bool(ok_nonzero and ok_reason)
            stdout_s = res.stdout[-600:]
            stderr_s = res.stderr[-300:]
            results.append({"test": name, "expected": EXPECT[name], "decision": decision,
                            "returncode": res.returncode, "pass": passed,
                            "stdout_tail": stdout_s, "stderr_tail": stderr_s})
            print(f"{name}: {'PASS' if passed else 'FAIL'} decision={decision} rc={res.returncode}")
        except Exception as e:
            results.append({"test": name, "expected": EXPECT[name], "pass": False, "error": str(e)})
            print(f"{name}: EXCEPTION {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # separate control run: untouched copy must be VALID_P3 (proves validator not always-fail)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="p3pos_"))
    try:
        dest = tmp / "13_p3_delay20"
        shutil.copytree(str(root), str(dest))
        res = subprocess.run([py, str(dest / VAL_REL), "--root", str(dest)], capture_output=True, text=True)
        control_ok = res.returncode == 0 and json.loads((dest / "04_analysis/p3_final_gate.json").read_text(encoding="utf-8")).get("decision") == "VALID_P3"
        results.append({"test": "CONTROL_untouched_VALID_P3", "expected": "VALID_P3",
                        "returncode": res.returncode, "pass": bool(control_ok)})
        print(f"CONTROL: {'PASS' if control_ok else 'FAIL'} rc={res.returncode}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    overall = all(r["pass"] for r in results)
    out = {"tests": results, "overall_pass": bool(overall),
           "validator": VAL_REL + " --root <temp> (real, no mocks)",
           "note": "8 real tampers + untouched control; each tamper: non-zero exit AND expected decision"}
    (root / "06_checks/p3_negative_tests.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # stdout/stderr summaries as txt per plan
    lines = ["R4 negative test logs (validator stdout/stderr tails)"]
    for r in results:
        lines.append(f"\n== {r['test']} expect {r['expected']} -> {'PASS' if r['pass'] else 'FAIL'} rc={r.get('returncode')}")
        if r.get("stdout_tail"):
            lines.append("--stdout tail--\n" + r["stdout_tail"])
        if r.get("stderr_tail"):
            lines.append("--stderr tail--\n" + r["stderr_tail"])
    (root / "06_checks/negative_test_logs.txt").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"overall_pass": overall}, indent=2))
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
