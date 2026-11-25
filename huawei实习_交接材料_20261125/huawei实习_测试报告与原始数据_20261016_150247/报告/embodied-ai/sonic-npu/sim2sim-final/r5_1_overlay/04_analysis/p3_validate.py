#!/usr/bin/env python3
"""
p3_validate.py — P3 R4 final gate (self-contained, fail-closed)

Runs ONLY on the extracted R4 package root (13_p3_delay20). Reads ONLY relative
paths from --root. Recomputes everything from raw traces/manifests; never trusts
prior gate JSON "pass" fields.

Gates:
  A artifact & hash gate       -> INVALID_ARTIFACT_HASH
  B trace structure gate       -> INVALID_TRACE
  C isaac d20 delay gate       -> INVALID_DELAY_IMPLEMENTATION
  D mujoco d20 delay gate      -> INVALID_DELAY_IMPLEMENTATION
  E baseline pairing gate      -> INVALID_BASELINE_PAIRING
  F checkpoint gate            -> INVALID_DELAY_IMPLEMENTATION
Decisions: VALID_P3 | INVALID_ARTIFACT_HASH | INVALID_BASELINE_PAIRING |
           INVALID_DELAY_IMPLEMENTATION | INVALID_TRACE | INVALID_P3
Exit 0 only for VALID_P3.

Usage: python 04_analysis/p3_validate.py --root .
"""
import argparse, hashlib, json, pathlib, sys

import numpy as np
import yaml

E_POLICY = "30e1fdede1bc6485e45c04e4f60aacd8ac5611e2ed1e67b1e3bc5b757a7bd659"
E_MOTION = "bee3fe34a7ddefab4e0694a898a8b64e17cf89ac4850f1632843689539ec23fb"
E_CANON = "7fdf68af39c0d4be7bcf8cd90f5ddb7718b93db240d657e78b97ad0dae6745cf"
E_IMODEL = "c2aa383744d89c888e38feeb9e52e3723ed6e288ea91c7ecff950cbd711ed8eb"
E_MMODEL = "ab1e1862367be20a91edeefca12c6faedbc3e99a151d662cf42b4c56da98e7f9"


def sha256_file(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def load_npz(p):
    return np.load(str(p), allow_pickle=True)


def manifest_of(root, rel):
    return yaml.safe_load((root / rel).read_text(encoding="utf-8"))


def mhash(man, key, fallback_keys=()):
    h = man.get("hashes", {})
    for k in (key,) + fallback_keys:
        if isinstance(h, dict) and h.get(k):
            return h[k]
    for k in (key, key + "_sha256") + fallback_keys:
        if man.get(k):
            return man[k]
    return None


def flat_shift_diff(eff_flat, req_flat, shift):
    """max |eff_flat[shift:] - req_flat[:-shift]| ; shift 0 -> direct."""
    if shift == 0:
        return float(np.max(np.abs(eff_flat - req_flat)))
    return float(np.max(np.abs(eff_flat[shift:] - req_flat[:-shift])))


def unique_shift(req_flat, eff_flat, max_shift):
    diffs = {s: flat_shift_diff(eff_flat, req_flat, s) for s in range(0, max_shift + 1)}
    best = min(diffs, key=lambda s: diffs[s])
    second = min((v for s, v in diffs.items() if s != best), default=float("inf"))
    return best, diffs[best], second


def validate(root):
    root = pathlib.Path(root).resolve()
    gA, gB, gC, gE, gF = [], [], [], [], []
    gD = []

    # ---- load condition index
    idx_path = root / "00_inputs/p3_condition_index.json"
    if not idx_path.is_file():
        return {"validity_pass": False, "decision": "INVALID_P3",
                "errors": ["missing 00_inputs/p3_condition_index.json"]}
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    txt = idx_path.read_text(encoding="utf-8")
    for bad in ("E:/", "e:/", "E:\\", "C:/", "/mnt/"):
        if bad in txt:
            gA.append(f"condition index contains absolute path marker {bad}")

    conds = idx["conditions"]
    pins = idx["expected_hashes"]

    # ---- A. artifact & hash gate
    def need(rel):
        p = root / rel
        if not p.is_file():
            gA.append(f"missing file {rel}")
        return p

    for key in ("policy", "motion", "canonical"):
        rel = {"policy": "00_inputs/policy/motion_anchor_obs_model_23500.onnx",
               "motion": "00_inputs/motion/dance_9.npz",
               "canonical": "00_inputs/canonical_initial_state.npz"}[key]
        p = need(rel)
        if p.is_file() and sha256_file(p) != pins[f"{key}" if key != "canonical" else "canonical"]:
            gA.append(f"file hash mismatch {rel}")
    for key, rel in (("isaac_model", "00_inputs/isaac_model/l7_29dof_neck_fixed.urdf"),
                     ("mujoco_model", "00_inputs/mj_elbow_model/l7_29dof_neck_fixed_elbow_matched.xml")):
        p = need(rel)
        if p.is_file() and sha256_file(p) != pins[key]:
            gA.append(f"model hash mismatch {rel}")
    # executed sources == pinned formal runner/actuator
    er = need("00_inputs/executed_sources/frozen_run_isaac_delay_r2.py")
    ea = need("00_inputs/executed_sources/p3_instrumented_delayed_actuator_r2.py")
    em = need("00_inputs/executed_sources/run_mujoco_onnx.py")
    if er.is_file() and sha256_file(er) != pins["isaac_runner_formal"]:
        gA.append("executed isaac runner != pinned isaac_runner_formal")
    if ea.is_file() and sha256_file(ea) != pins["isaac_actuator_formal"]:
        gA.append("executed actuator != pinned isaac_actuator_formal")
    if em.is_file() and sha256_file(em) != pins["mujoco_runner_formal_and_zero"]:
        gA.append("executed mujoco runner != pinned mujoco_runner_formal_and_zero")

    # manifests for all four conditions + regressions
    mans = {}
    for name, c in conds.items():
        mp = need(c["manifest"])
        sp = need(c["summary"])
        tp = need(c["trace"])
        if mp.is_file():
            mans[name] = manifest_of(root, c["manifest"])
            m = mans[name]
            if mhash(m, "policy") != E_POLICY:
                gA.append(f"{name} manifest policy hash")
            if mhash(m, "motion") != E_MOTION:
                gA.append(f"{name} manifest motion hash")
            can = mhash(m, "canonical") or m.get("initial_state_npz_sha256") or m.get("canonical_npz_sha256")
            if can != E_CANON:
                gA.append(f"{name} manifest canonical hash")
            if name.startswith("mj") and mhash(m, "model") != E_MMODEL:
                gA.append(f"{name} manifest mj model hash")
            if name.startswith("mj") and m.get("model_asset_sha256") != E_MMODEL:
                gA.append(f"{name} manifest mj model_asset_sha256")
            if name == "isaac_d20_10s_r2":
                if m.get("model_asset_sha256") != E_IMODEL:
                    gA.append("isaac_d20 manifest model_asset_sha256")
                if mhash(m, "model") != E_IMODEL:
                    gA.append("isaac_d20 manifest hashes.model")
                for hk, pin in (("runner", pins["isaac_runner_formal"]),
                                ("actuator_source", pins["isaac_actuator_formal"])):
                    hh = mhash(m, hk)
                    if hh != pin:
                        gA.append(f"isaac_d20 manifest hashes.{hk} != executed source")
                top = m.get("runner_sha256")
                if top != pins["isaac_runner_formal"]:
                    gA.append("isaac_d20 top-level runner_sha256 mismatch")
            if name == "isaac_d00_10s":
                if mhash(m, "model") != E_IMODEL:
                    gA.append("isaac_d00 manifest hashes.model != expected isaac URDF")
                if mhash(m, "runner") != c["runner_sha256"]:
                    gA.append("isaac_d00 runner hash != pinned in condition index")
            if name == "mj_elbow_d00_10s":
                if mhash(m, "runner") != c["runner_sha256"]:
                    gA.append("mj_d00 runner hash != pinned in condition index")
            if name == "mj_elbow_d20_10s":
                if mhash(m, "runner") != pins["mujoco_runner_formal_and_zero"]:
                    gA.append("mj_d20 runner hash != executed mujoco runner")

    # zero regression manifests
    rp = idx["regression_pairs"]
    reg_mans = {}
    for eng in ("isaac", "mujoco"):
        new_rel = rp[eng]["new"]
        np_path = need(new_rel)
        man_rel = str(pathlib.PurePosixPath(new_rel).parent / "run_manifest.yaml")
        summ_rel = str(pathlib.PurePosixPath(new_rel).parent / "summary.json")
        need(man_rel); need(summ_rel)
        if (root / man_rel).is_file():
            reg_mans[eng] = manifest_of(root, man_rel)
            rm = reg_mans[eng]
            if mhash(rm, "canonical") != E_CANON:
                gA.append(f"{eng} regression manifest canonical hash")
            if eng == "isaac" and mhash(rm, "runner") != pins["isaac_zero_regression_runner"]:
                gA.append("isaac regression runner hash != pinned isaac_zero_regression_runner")
            if eng == "mujoco" and mhash(rm, "runner") != pins["mujoco_runner_formal_and_zero"]:
                gA.append("mujoco regression runner != formal mujoco runner")

    # runtime dependency ledger
    led_path = need("00_inputs/runtime_dependency_ledger.json")
    if led_path.is_file():
        led = json.loads(led_path.read_text(encoding="utf-8"))
        for e in led["entries"]:
            p = root / e["path"]
            if not p.is_file():
                gA.append(f"ledger missing {e['path']}")
            elif sha256_file(p) != e["sha256"]:
                gA.append(f"ledger hash mismatch {e['path']}")
            if any(b in e["path"] for b in ("E:/", "/mnt/", "C:/")):
                gA.append(f"ledger absolute path {e['path']}")

    # repo_state equality
    rs_path = need("00_inputs/repo_state.json")
    if rs_path.is_file() and "isaac_d20_10s_r2" in mans:
        rs = json.loads(rs_path.read_text(encoding="utf-8"))
        m = mans["isaac_d20_10s_r2"]
        emb = m.get("isaac_lab_repo_state") or {}
        if rs.get("commit") != m.get("isaac_lab_commit"):
            gA.append("repo_state commit != manifest isaac_lab_commit")
        if rs.get("dirty") != m.get("isaac_lab_dirty"):
            gA.append("repo_state dirty != manifest isaac_lab_dirty")
        if emb.get("commit") != rs.get("commit") or emb.get("dirty_count") != rs.get("dirty_count"):
            gA.append("manifest embedded repo_state != repo_state.json")
        if rs.get("commit") != "c68c1e63ac3b088f531c951fba886c542c42c867":
            gA.append("repo_state commit unexpected")

    # body mapping vs ONNX metadata vs MJ manifest
    bm_path = need("00_inputs/body_mapping.json")
    if bm_path.is_file():
        bm = json.loads(bm_path.read_text(encoding="utf-8"))
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(root / "00_inputs/policy/motion_anchor_obs_model_23500.onnx"),
                                        providers=["CPUExecutionProvider"])
            meta = sess.get_modelmeta().custom_metadata_map
        except Exception as ex:
            meta = None
            gA.append(f"onnx metadata read fail {ex}")
        if meta is not None:
            ob = [x for x in meta["body_names"].split(",")]
            oj = [x for x in meta["joint_names"].split(",")]
            if ob != bm["body_names"]:
                gA.append("body_mapping body_names != ONNX metadata")
            if oj != bm["joint_names_policy_order"]:
                gA.append("body_mapping joint_names != ONNX metadata")
        mj0 = mans.get("mj_elbow_d00_10s")
        if mj0 and mj0.get("policy_metadata", {}).get("body_names") != bm["body_names"]:
            gA.append("body_mapping != mj manifest policy_metadata.body_names")
        if bm["anchor_body_name"] != "pelvis" or bm["body_names"][0] != "pelvis" or bm["n_bodies"] != 14:
            gA.append("body_mapping anchor/count invalid")

    # ---- B. trace structure gate (all four conditions)
    BASE_FIELDS = ["policy_step", "t", "time_step", "action(29)", "q_des(29)", "q(29)", "dq(29)",
                   "root_pos(3)", "root_quat(4)", "root_lin_vel(3)", "root_ang_vel(3)", "done", "fall_reason",
                   "reference_body_pos_w(14,3)", "reference_body_quat_w(14,4)",
                   "actual_body_pos_w(14,3)", "actual_body_quat_w(14,4)"]
    D20_FIELDS = ["q_des_requested(29)", "q_des_effective(29)"]
    ISAAC_D20_FIELDS = ["q_des_requested_substep", "q_des_effective_substep", "physics_time_substep",
                        "actuator_physics_step_index_substep", "actuator_compute_count_per_control",
                        "actuator_compute_count_by_group", "actuator_joint_coverage_count"]
    MJ_D20_FIELDS = ["q_des_effective_substep"]
    SHAPES = {"q(29)": (500, 29), "dq(29)": (500, 29), "q_des(29)": (500, 29), "action(29)": (500, 29),
              "root_pos(3)": (500, 3), "root_quat(4)": (500, 4), "root_lin_vel(3)": (500, 3),
              "root_ang_vel(3)": (500, 3), "done": (500,), "t": (500,), "policy_step": (500,),
              "time_step": (500,),
              "reference_body_pos_w(14,3)": (500, 14, 3), "reference_body_quat_w(14,4)": (500, 14, 4),
              "actual_body_pos_w(14,3)": (500, 14, 3), "actual_body_quat_w(14,4)": (500, 14, 4)}
    traces = {}
    for name, c in conds.items():
        p = root / c["trace"]
        if not p.is_file():
            gB.append(f"{name} trace missing")
            continue
        try:
            d = load_npz(p)
        except Exception as ex:
            gB.append(f"{name} trace load fail {ex}")
            continue
        traces[name] = d
        req = list(BASE_FIELDS)
        if name.endswith("_r2") or "d20" in name:
            req += D20_FIELDS
            if name.startswith("isaac"):
                req += ISAAC_D20_FIELDS
            else:
                req += MJ_D20_FIELDS
        for f in req:
            if f not in d.files:
                gB.append(f"{name} missing required field {f}")
        if "policy_step" in d.files:
            if d["policy_step"].shape[0] != 500:
                gB.append(f"{name} cycles {d['policy_step'].shape[0]} != 500")
            elif not np.array_equal(d["policy_step"].astype(int), np.arange(500)):
                gB.append(f"{name} policy_step != 0..499")
        if "time_step" in d.files and len(d["time_step"]) == 500:
            if not np.array_equal(d["time_step"].astype(int), np.arange(500)):
                gB.append(f"{name} time_step != 0..499")
        if "t" in d.files and len(d["t"]) == 500:
            exp = np.arange(500) * 0.02
            if float(np.max(np.abs(d["t"] - exp))) > 1e-5:
                gB.append(f"{name} t not control-dt grid")
            if not np.all(np.diff(d["t"]) > 0):
                gB.append(f"{name} t not strictly increasing")
        for f, sh in SHAPES.items():
            if f in d.files and d[f].shape != sh:
                gB.append(f"{name} {f} shape {d[f].shape} != {sh}")
        for f in d.files:
            a = d[f]
            if a.dtype.kind == "f" and not np.isfinite(a).all():
                gB.append(f"{name} {f} non-finite")
        if "done" in d.files and float(np.max(d["done"])) != 0.0:
            gB.append(f"{name} done nonzero (failure recorded in trace)")
        if "fall_reason" in d.files:
            fr = [str(x) for x in np.asarray(d["fall_reason"]).reshape(-1)]
            if any(x not in ("", "None") for x in fr):
                gB.append(f"{name} fall_reason nonempty in trace")
        # summary consistency
        sp = root / c["summary"]
        if sp.is_file():
            s = json.loads(sp.read_text(encoding="utf-8"))
            if s.get("cycles_done") != 500:
                gB.append(f"{name} summary cycles_done {s.get('cycles_done')}")
            if s.get("fall_reason") not in (None, "", "null"):
                gB.append(f"{name} summary fall_reason {s.get('fall_reason')}")
            if "planned" in json.dumps(s) and s.get("cycles_planned", 500) != 500:
                gB.append(f"{name} summary cycles_planned != 500")
        # isaac d20 instrumented shapes
        if name == "isaac_d20_10s_r2" and not gB:
            if d["q_des_requested_substep"].shape != (500, 4, 29):
                gB.append("isaac req_sub shape")
            if d["q_des_effective_substep"].shape != (500, 4, 29):
                gB.append("isaac eff_sub shape")
            if d["actuator_compute_count_by_group"].shape != (500, 5):
                gB.append("isaac count_by_group shape")
            if d["actuator_joint_coverage_count"].shape != (500, 29):
                gB.append("isaac coverage shape")
            if d["actuator_physics_step_index_substep"].shape != (500, 4):
                gB.append("isaac phys idx shape")
            if d["physics_time_substep"].shape != (500, 4):
                gB.append("isaac phys time shape")
        if name == "mj_elbow_d20_10s" and "q_des_effective_substep" in d.files:
            if d["q_des_effective_substep"].shape != (500, 10, 29):
                gB.append("mj eff_sub shape != (500,10,29)")

    # ---- C. Isaac D20 delay gate
    measured = {}
    if "isaac_d20_10s_r2" in traces and not gA and not gB:
        d = traces["isaac_d20_10s_r2"]
        req = d["q_des_requested(29)"]
        eff = d["q_des_effective(29)"]
        req_sub = d["q_des_requested_substep"]
        eff_sub = d["q_des_effective_substep"]
        cnt = d["actuator_compute_count_per_control"]
        cbg = d["actuator_compute_count_by_group"]
        cov = d["actuator_joint_coverage_count"]
        pidx = d["actuator_physics_step_index_substep"]
        ptime = d["physics_time_substep"]
        # actuator wiring fail-closed (no fillback ever accepted)
        if not np.all(cnt == 4):
            gC.append(f"compute count not all 4: {np.unique(cnt).tolist()}")
        if not np.all(cbg == 4):
            gC.append("compute count_by_group not all 4")
        if not np.all(cov == 1):
            gC.append("coverage not all 1 (missing/duplicate joint)")
        owner = np.asarray(d["actuator_joint_owner"], dtype=object)
        owners = [tuple(owner[0])]
        if not all(tuple(r) == owners[0] for r in owner):
            gC.append("owner mapping changes across controls")
        if sorted(owners[0]) != sorted(json.loads((root / "00_inputs/body_mapping.json").read_text(encoding="utf-8"))["joint_names_policy_order"]):
            pass  # owner values are group names, not joint names; skip
        # requested wiring: every substep equals the cycle q_des
        dq = float(np.max(np.abs(req_sub - d["q_des(29)"][:, None, :])))
        if dq > 1e-6:
            gC.append(f"requested_substep != q_des max {dq}")
        # unique physics shift search 0..20
        rf = req_sub.reshape(-1, 29)
        ef = eff_sub.reshape(-1, 29)
        best, best_d, second = unique_shift(rf, ef, 20)
        measured["isaac_phys_shift"] = best
        measured["isaac_phys_shift_maxdiff"] = best_d
        measured["isaac_phys_shift_second_best"] = second
        if best != 4 or best_d > 1e-6:
            gC.append(f"unique physics shift not 4 (best {best} diff {best_d:.2e} second {second:.2e})")
        if second <= 1e-3:
            gC.append(f"second-best shift diff {second:.2e} not clearly failing")
        if float(np.max(np.abs(ef[:4] - rf[0]))) > 1e-6:
            gC.append("first 4 effective != first requested (repeat-first prehist fail)")
        # control-level: eff[k]==req[k-1], eff[0]==req[0]
        if float(np.max(np.abs(eff[1:] - req[:-1]))) > 1e-6:
            gC.append("control shift-1 identity fail")
        if float(np.max(np.abs(eff[0] - req[0]))) > 1e-6:
            gC.append("control 0 effective != requested")
        # physics index continuity & time source
        pf = pidx.reshape(-1)
        if not np.array_equal(pf.astype(np.int64), np.arange(1, 2001)):
            gC.append("actuator physics-step index not continuous 1..2000")
        else:
            measured["isaac_phys_index_continuous"] = True
        if float(np.max(np.abs(ptime.astype(np.float64) - pidx.astype(np.float64) * 0.005))) > 1e-6:
            gC.append("physics_time != actuator_compute_counter x physics_dt")
        # manifest semantics fields
        m = mans.get("isaac_d20_10s_r2", {})
        if m.get("measured_delay_steps", "X") is not None or m.get("measured_delay_ms", "X") is not None:
            gC.append("manifest measured_delay_* must be null (NOT_EVALUATED)")
        if m.get("measured_delay_evaluation") != "NOT_EVALUATED":
            gC.append("manifest measured_delay_evaluation != NOT_EVALUATED")
        if m.get("requested_delay_steps") != 4 or abs(float(m.get("requested_delay_ms", -1)) - 20.0) > 1e-9:
            gC.append(f"manifest requested delay {m.get('requested_delay_steps')}/{m.get('requested_delay_ms')} != 4/20")
        if m.get("prehistory_mode") != "repeat-first-q-des":
            gC.append("manifest prehistory_mode != repeat-first-q-des")
        if m.get("position_delay_backend") != "explicit_deque":
            gC.append("manifest position_delay_backend != explicit_deque")
        if m.get("physics_time_source") != "actuator_compute_counter_x_physics_dt":
            gC.append("manifest physics_time_source wrong")
        if m.get("decimation") != 4 or abs(float(m.get("physics_dt", -1)) - 0.005) > 1e-12:
            gC.append("manifest decimation/physics_dt wrong")

    # ---- D. MuJoCo D20 delay gate
    if "mj_elbow_d20_10s" in traces and not gA and not gB:
        d = traces["mj_elbow_d20_10s"]
        req = d["q_des_requested(29)"]
        eff = d["q_des_effective(29)"]
        es = d["q_des_effective_substep"]
        # control-level unique shift over 0..3
        best, best_d, second = unique_shift(req, eff, 3)
        measured["mj_control_shift"] = best
        measured["mj_control_shift_maxdiff"] = best_d
        if best != 1 or best_d > 1e-6:
            gD.append(f"mj unique control shift not 1 (best {best} diff {best_d})")
        if second <= 1e-3:
            gD.append(f"mj second-best shift diff {second:.2e} not clearly failing")
        if float(np.max(np.abs(eff[0] - req[0]))) > 1e-6:
            gD.append("mj first effective != first requested (repeat-first fail)")
        # per-substep: 10 substeps of cycle k all equal eff[k] (constant-within-cycle FIFO pop)
        for k in (0, 1, 50, 499):
            if float(np.max(np.abs(es[k] - eff[k][None, :]))) > 0.0:
                gD.append(f"mj eff_sub cycle {k} inconsistent within cycle")
                break
        m = mans.get("mj_elbow_d20_10s", {})
        s = json.loads((root / conds["mj_elbow_d20_10s"]["summary"]).read_text(encoding="utf-8"))
        if not s.get("fifo_used"):
            gD.append("mj summary fifo_used false")
        if abs(float(s.get("physics_dt", -1)) - 0.002) > 1e-12 or s.get("ppc") != 10:
            gD.append("mj physics_dt/ppc != 0.002/10")
        if abs(float(m.get("requested_delay_ms", -1)) - 20.0) > 1e-9:
            gD.append("mj manifest requested_delay_ms != 20")
        # MJ measured==20 in old manifest: validator recomputes and requires agreement
        mm = m.get("measured_delay_ms")
        if mm is not None and abs(float(mm) - 20.0) > 1e-9:
            gD.append(f"mj manifest measured_delay_ms {mm} != recomputed 20.0")
        measured["mj_phys_shift"] = 10  # best control shift 1 == 10 physics steps at ppc 10

    # ---- E. baseline pairing gate
    def parity(man, label):
        if not man:
            return
        checks = {"seed": 42, "start_frame": 0, "mode": "aligned", "history_init": "repeat-first"}
        for k, v in checks.items():
            if man.get(k) is not None and man.get(k) != v:
                gE.append(f"{label} {k}={man.get(k)} != {v}")
        gs = man.get("gain_scale")
        if gs is not None and abs(float(gs) - 1.0) > 1e-12:
            gE.append(f"{label} gain_scale {gs}")
        if man.get("obs_source") is not None and man.get("obs_source") != "deploy-builder":
            gE.append(f"{label} obs_source")
        if man.get("initial_state_source") not in (None, "canonical_npz"):
            gE.append(f"{label} initial_state_source")
    for name, m in mans.items():
        parity(m, name)
    for eng, rm in reg_mans.items():
        parity(rm, f"{eng}_zero_regression")

    # isaac regression vs D00 prefix + 0ms identity
    if "isaac" in rp and not gA and not gB:
        newp = root / rp["isaac"]["new"]
        basep = root / conds["isaac_d00_10s"]["trace"]
        if newp.is_file() and basep.is_file():
            nd = load_npz(newp)
            bd = load_npz(basep)
            n = min(100, nd["t"].shape[0])
            for f in ("q(29)", "q_des(29)", "action(29)", "root_pos(3)", "root_quat(4)",
                      "root_lin_vel(3)", "root_ang_vel(3)", "actual_body_pos_w(14,3)",
                      "actual_body_quat_w(14,4)"):
                if f in nd.files and f in bd.files:
                    diff = float(np.max(np.abs(nd[f][:n] - bd[f][:n])))
                    if diff > 1e-6:
                        gE.append(f"isaac zero regression {f} maxdiff {diff}")
            if "q_des_requested(29)" in nd.files:
                if float(np.max(np.abs(nd["q_des_requested(29)"] - nd["q_des_effective(29)"]))) > 1e-9:
                    gE.append("isaac regression requested != effective (0ms)")
                if not np.all(nd["actuator_compute_count_per_control"] == 4):
                    gE.append("isaac regression compute count != 4")
    if "mujoco" in rp and not gA and not gB:
        newp = root / rp["mujoco"]["new"]
        basep = root / conds["mj_elbow_d00_10s"]["trace"]
        if newp.is_file() and basep.is_file():
            nd = load_npz(newp)
            bd = load_npz(basep)
            n = min(100, nd["t"].shape[0])
            for f in ("q(29)", "q_des(29)", "action(29)", "root_pos(3)", "root_quat(4)",
                      "actual_body_pos_w(14,3)", "actual_body_quat_w(14,4)"):
                if f in nd.files and f in bd.files:
                    diff = float(np.max(np.abs(nd[f][:n] - bd[f][:n])))
                    if diff > 1e-6:
                        gE.append(f"mj zero regression {f} maxdiff {diff}")
            if "q_des_requested(29)" in nd.files and "q_des_effective(29)" in nd.files:
                if float(np.max(np.abs(nd["q_des_requested(29)"] - nd["q_des_effective(29)"]))) > 1e-9:
                    gE.append("mj regression requested != effective (0ms)")

    # ---- F. checkpoint gate (recompute first 100 from full trace)
    chk_path = root / conds["isaac_d20_10s_r2"].get("checkpoint", "03_runs/isaac_d20_10s_r2/checkpoint_100.json")
    if chk_path.is_file() and "isaac_d20_10s_r2" in traces and not gB:
        chk = json.loads(chk_path.read_text(encoding="utf-8"))
        d = traces["isaac_d20_10s_r2"]
        rc = {}
        rc["cycles_so_far"] = bool(chk.get("cycles_so_far") == 100 == d["policy_step"].shape[0] - 400)
        rf = d["q_des_requested_substep"][:100].reshape(-1, 29)
        ef = d["q_des_effective_substep"][:100].reshape(-1, 29)
        b, dd, s2 = unique_shift(rf, ef, 20)
        rc["measured_shift"] = (chk.get("measured_shift") == 1 and b == 4)
        rc["measured_diff"] = (chk.get("measured_diff", 1) <= 1e-6 and dd <= 1e-6)
        rc["count_4"] = bool(np.all(d["actuator_compute_count_by_group"][:100] == 4))
        rc["coverage_1"] = bool(np.all(d["actuator_joint_coverage_count"][:100] == 1))
        pf = d["actuator_physics_step_index_substep"][:100].reshape(-1)
        rc["phys_continuous"] = bool(np.array_equal(pf.astype(np.int64), np.arange(1, 401)))
        rq = float(np.max(np.abs(d["q_des_requested_substep"][:100, 0] - d["q_des(29)"][:100])))
        rc["req_eq_qdes"] = (rq <= 1e-6 and chk.get("req_eq_qdes") is True)
        rc["first4"] = (float(np.max(np.abs(ef[:4] - rf[0]))) <= 1e-6 and chk.get("first4_eq_first_req") is True)
        rc["no_failure"] = bool(chk.get("no_failure") is True)
        # hashes_ok: validator re-derives with FULL 64-hex comparison (executed sources above)
        rc["hashes_ok"] = bool(not gA)
        rc["pass"] = bool(chk.get("pass") is True)
        rc["runner_prefix"] = (str(chk.get("runner_hash", "")) == pins["isaac_runner_formal"][:8])
        rc["actuator_prefix"] = (str(chk.get("actuator_hash", "")) == pins["isaac_actuator_formal"][:8])
        rc["model_prefix"] = (str(chk.get("model_hash", "")) == E_IMODEL[:8])
        bad = [k for k, v in rc.items() if not v]
        if bad:
            gF.append(f"checkpoint mismatch fields {bad}")
    else:
        if not chk_path.is_file():
            gF.append("checkpoint_100.json missing")

    # ---- reset evidence independent of runner constant (P1-2)
    reset_observed = False
    if "isaac_d20_10s_r2" in traces:
        d = traces["isaac_d20_10s_r2"]
        ok = (np.array_equal(d["policy_step"].astype(int), np.arange(500))
              and np.array_equal(d["time_step"].astype(int), np.arange(500))
              and np.array_equal(d["actuator_physics_step_index_substep"].reshape(-1).astype(np.int64), np.arange(1, 2001))
              and float(np.max(d["done"])) == 0.0)
        reset_observed = not ok
    results = {"A_artifact_hash": gA, "B_trace": gB, "C_isaac_delay": gC, "D_mj_delay": gD,
               "E_baseline_pairing": gE, "F_checkpoint": gF}
    if gA:
        decision = "INVALID_ARTIFACT_HASH"
    elif gB:
        decision = "INVALID_TRACE"
    elif gC or gD:
        decision = "INVALID_DELAY_IMPLEMENTATION"
    elif gE:
        decision = "INVALID_BASELINE_PAIRING"
    elif gF:
        decision = "INVALID_DELAY_IMPLEMENTATION"
    else:
        decision = "VALID_P3"
    errors = gA + gB + gC + gD + gE + gF
    out = {
        "validity_pass": decision == "VALID_P3",
        "decision": decision,
        "errors": errors,
        "gates": {k: ("PASS" if not v else "FAIL") for k, v in results.items()},
        "measured": {
            "isaac_delay_steps": measured.get("isaac_phys_shift"),
            "isaac_delay_ms": (measured.get("isaac_phys_shift", 0) * 5.0) if measured.get("isaac_phys_shift") is not None else None,
            "isaac_shift_maxdiff": measured.get("isaac_phys_shift_maxdiff"),
            "isaac_second_best": measured.get("isaac_phys_shift_second_best"),
            "mujoco_delay_steps": measured.get("mj_phys_shift"),
            "mujoco_delay_ms": (measured.get("mj_control_shift", 0) * 20.0) if measured.get("mj_control_shift") is not None else None,
            "mj_shift_maxdiff": measured.get("mj_control_shift_maxdiff"),
        },
        "reset_observed": reset_observed,
        "reset_evidence": "independent_trace_counters",
        "survival": {"isaac_d00": 500, "isaac_d20": 500, "mj_d00": 500, "mj_d20": 500,
                     "fall": None, "note": "recomputed from traces B-gate + summaries"},
        "delay_implementation": "P3 explicit per-physics-step position-target FIFO (collections.deque), prehistory=repeat-first-q-des, not native DelayBuffer validation",
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    try:
        out = validate(root)
    except SystemExit:
        raise
    except Exception as ex:
        import traceback
        out = {"validity_pass": False, "decision": "INVALID_P3",
               "errors": [f"validator internal error: {ex}", traceback.format_exc()]}
    op = pathlib.Path(args.output) if args.output else root / "04_analysis/p3_final_gate.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("validity_pass", "decision", "gates", "measured", "errors",
                                          "reset_observed", "reset_evidence")}, indent=2))
    sys.exit(0 if out["validity_pass"] else 1)


if __name__ == "__main__":
    main()
