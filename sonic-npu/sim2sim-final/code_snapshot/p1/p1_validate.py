#!/usr/bin/env python3
"""
p1_validate.py — Pure validate(root) for P1 fail-closed gate
"""
import hashlib, json, pathlib
import numpy as np
import yaml

TARGET_JOINT = "right_elbow_pitch_joint"
CONTROL_DT = 0.02

def sha256(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load_manifest(path): return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
def quat_to_roll_pitch(q):
    w=q[:,0]; x=q[:,1]; y=q[:,2]; z=q[:,3]
    roll=np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch=np.arcsin(np.clip(2*(w*y-z*x), -1,1))
    return roll,pitch

def validate(root: pathlib.Path) -> dict:
    import csv, traceback, copy
    try:
        root=pathlib.Path(root)
        isaac_ref=root/"00_inputs/isaac_reference/policy_trace.npz"
        native_runs=[root/f"03_runs/mj_native_r{i}/policy_trace.npz" for i in [1,2,3]]
        matched_runs=[root/f"03_runs/mj_matched_r{i}/policy_trace.npz" for i in [1,2,3]]
        native_manifests=[root/f"03_runs/mj_native_r{i}/run_manifest.yaml" for i in [1,2,3]]
        matched_manifests=[root/f"03_runs/mj_matched_r{i}/run_manifest.yaml" for i in [1,2,3]]
        isaac=np.load(str(isaac_ref))
        t_isaac=isaac["t"].astype(float)
        native_man0=load_manifest(native_manifests[0])
        joint_names=native_man0["policy_metadata"]["joint_names"]
        if TARGET_JOINT not in joint_names:
            raise ValueError(f"target {TARGET_JOINT} not in joint_names")
        target_idx=joint_names.index(TARGET_JOINT)
        def check_run(path, manifest_path):
            trace=np.load(str(path))
            man=load_manifest(manifest_path)
            summary=json.loads((pathlib.Path(manifest_path).parent/"summary.json").read_text(encoding="utf-8"))
            t=trace["t"].astype(float)
            cycles_ok=(summary["cycles_done"]==500 and len(t)==500)
            if len(t)!=len(t_isaac):
                time_ok=False; t_maxdiff=float('nan')
            else:
                t_maxdiff=float(np.max(np.abs(t-t_isaac))); time_ok=t_maxdiff<=1e-6
            has_nan=False
            for k in trace.files:
                arr=trace[k]
                if arr.dtype.kind in 'f' and (np.isnan(arr).any() or np.isinf(arr).any()):
                    has_nan=True
            nan_ok=not has_nan
            strict_inc=bool(np.all(np.diff(t)>0)) if len(t)>1 else False
            joint_ok=(man["policy_metadata"]["joint_names"]==joint_names)
            fall_ok=(summary["fall_reason"] is None)
            return {"cycles_ok": cycles_ok, "time_ok": time_ok, "t_maxdiff": t_maxdiff, "nan_ok": nan_ok, "strict_inc": strict_inc, "joint_ok": joint_ok, "fall_ok": fall_ok, "summary": summary, "manifest": man, "trace": trace}
        all_checks=[]
        for i,(p,m) in enumerate(zip(native_runs, native_manifests)):
            chk=check_run(p,m); chk["run_id"]=f"mj_native_r{i+1}"; chk["condition"]="native"; all_checks.append(chk)
        for i,(p,m) in enumerate(zip(matched_runs, matched_manifests)):
            chk=check_run(p,m); chk["run_id"]=f"mj_matched_r{i+1}"; chk["condition"]="matched"; all_checks.append(chk)
        # repeat
        def repeat_check_for_group(group_runs, group_name, manifests):
            traces=[np.load(str(p)) for p in group_runs]
            mans=[load_manifest(m) for m in manifests]
            fields=["t","q(29)","dq(29)","q_des(29)","action(29)","root_pos(3)","root_quat(4)"]
            body_fields=[]
            if "actual_body_pos_w(14,3)" in traces[0].files:
                body_fields=["actual_body_pos_w(14,3)","actual_body_quat_w(14,4)","reference_body_pos_w(14,3)","reference_body_quat_w(14,4)"]
            all_fields=fields+body_fields
            maxdiffs={}; ok=True
            for f in all_fields:
                arrs=[tr[f] for tr in traces]
                md=max(float(np.max(np.abs(arrs[a]-arrs[b]))) for a in range(3) for b in range(a+1,3))
                maxdiffs[f]=md
                if md>1e-7: ok=False
            hashes=[m["hashes"]["runner"] for m in mans]
            if not (hashes[0]==hashes[1]==hashes[2]): ok=False
            model_hashes=[m["hashes"]["model"] for m in mans]
            if not (model_hashes[0]==model_hashes[1]==model_hashes[2]): ok=False
            for k in ["policy","motion","canonical"]:
                hk=[m["hashes"][k] for m in mans]
                if not (hk[0]==hk[1]==hk[2]): ok=False
            return {"pass": bool(ok)}
        rep_native=repeat_check_for_group(native_runs, "mj_native", native_manifests)
        rep_matched=repeat_check_for_group(matched_runs, "mj_matched", matched_manifests)
        repeat_pass=bool(rep_native["pass"] and rep_matched["pass"])
        # metrics
        qdes_isaac_target=isaac["q_des(29)"][:, target_idx].astype(float)
        window=int(round(1.0/CONTROL_DT))
        best_std=-1; best_start=0
        for start in range(len(qdes_isaac_target)-window+1):
            std=float(np.std(qdes_isaac_target[start:start+window]))
            if std>best_std: best_std=std; best_start=start
        per_run_data=[]
        for chk in all_checks:
            trace=chk["trace"]
            q_target=trace["q(29)"][:, target_idx].astype(float)
            iq_target=isaac["q(29)"][:, target_idx].astype(float)
            mask_0_2=(t_isaac>=-1e-9)&(t_isaac<=2.0+1e-9)
            def mae(a,b,m): return float(np.mean(np.abs(a[m]-b[m])))
            mae_q_0_2=mae(q_target, iq_target, mask_0_2)
            root_pos=trace["root_pos(3)"].astype(float); iroot_pos=isaac["root_pos(3)"].astype(float)
            root_height_rmse=float(np.sqrt(np.mean((root_pos[:,2]-iroot_pos[:,2])**2)))
            root_quat=trace["root_quat(4)"].astype(float); iroot_quat=isaac["root_quat(4)"].astype(float)
            roll,pitch=quat_to_roll_pitch(root_quat); iroll,ipitch=quat_to_roll_pitch(iroot_quat)
            roll_rmse=float(np.sqrt(np.mean((roll-iroll)**2))); pitch_rmse=float(np.sqrt(np.mean((pitch-ipitch)**2)))
            q_all=trace["q(29)"].astype(float); iq_all=isaac["q(29)"].astype(float)
            joint_rmses=np.sqrt(np.mean((q_all-iq_all)**2, axis=0)); mean_joint_q_rmse=float(np.mean(joint_rmses))
            per_run_data.append({"run_id": chk["run_id"], "condition": chk["condition"], "mae_q_0_2": mae_q_0_2, "root_height_rmse": root_height_rmse, "root_roll_rmse": roll_rmse, "root_pitch_rmse": pitch_rmse, "mean_joint_q_rmse": mean_joint_q_rmse, "t_maxdiff": chk["t_maxdiff"], "fall_reason": chk["summary"]["fall_reason"], "fall_cycle": chk["summary"]["fall_cycle"], "hashes": chk["manifest"]["hashes"], "summary": chk["summary"], "manifest": chk["manifest"]})
        # validity
        def check_input_hashes_inner():
            p=root/"00_inputs/input_hashes.csv"
            if not p.exists(): return False, "missing input_hashes.csv", {}
            ok=True; details=[]; recomputed_map={}
            with open(p, newline="", encoding="utf-8") as f:
                reader=__import__("csv").DictReader(f)
                rows=list(reader)
                REQUIRED=["policy","motion","native_mujoco_xml","native_yaml","canonical_fullbody","isaac_10s_trace","mujoco_10s_trace"]
                present=set(r["input"] for r in rows if r["expected_sha256"].strip())
                missing=[r for r in REQUIRED if r not in present]
                if missing: ok=False; details.append(f"missing required rows {missing}")
                for row in rows:
                    expected=row["expected_sha256"].strip().lower(); actual=row["actual_sha256"].strip().lower(); status=row["status"].strip(); path=row["path"].strip(); inp=row["input"].strip()
                    if not expected:
                        if inp=="frozen_runner" and not actual: ok=False; details.append("frozen_runner hash empty")
                        continue
                    if status!="PASS": ok=False; details.append(f"{inp} status {status} != PASS"); continue
                    pp=pathlib.Path(path)
                    if not pp.exists():
                        alt=pathlib.Path(str(pp).replace("E:/","/mnt/e/").replace("E:\\","/mnt/e/"))
                        if alt.exists(): pp=alt
                        else:
                            found=list(root.rglob(pathlib.Path(path).name))
                            if found: pp=found[0]
                    if pp.exists():
                        try:
                            recomputed=hashlib.sha256(pp.read_bytes()).hexdigest().lower()
                            recomputed_map[inp]=recomputed
                            if recomputed != expected: ok=False; details.append(f"{inp} recomputed {recomputed[:8]} != expected {expected[:8]}")
                            if recomputed != actual: ok=False; details.append(f"{inp} recomputed != actual")
                        except Exception as e: ok=False; details.append(f"{inp} hash error {e}")
                    else: ok=False; details.append(f"{inp} path not found {path}")
            # matched XML binding
            try:
                matched_xml_path=root/"01_asset/l7_29dof_neck_fixed_elbow_matched.xml"
                if matched_xml_path.exists():
                    matched_recomputed=hashlib.sha256(matched_xml_path.read_bytes()).hexdigest().lower()
                    for chk in all_checks:
                        if chk["condition"]=="matched" and chk["manifest"]["hashes"]["model"].lower() != matched_recomputed:
                            ok=False; details.append(f"{chk['run_id']} model hash != matched XML recomputed")
                            break
                else: ok=False; details.append("matched XML missing")
            except Exception as e: ok=False; details.append(f"matched XML check error {e}")
            # config hash binding
            try:
                for chk in all_checks:
                    man=chk["manifest"]
                    yaml_path=root/"01_asset/mimic_dance_9_native.yaml" if chk["condition"]=="native" else root/"01_asset/mimic_dance_9_elbow_matched.yaml"
                    if yaml_path.exists():
                        recomputed_cfg=hashlib.sha256(yaml_path.read_bytes()).hexdigest().lower()
                        if man["hashes"]["config"].lower() != recomputed_cfg:
                            ok=False; details.append(f"{chk['run_id']} config hash != yaml recomputed")
            except Exception as e: ok=False; details.append(f"config hash check error {e}")
            return ok, "; ".join(details) if details else "all PASS", recomputed_map
        input_hash_ok, input_hash_detail, recomputed_hashes = check_input_hashes_inner()
        try:
            asset_audit=json.loads((root/"01_asset/fullbody_elbow_asset_audit.json").read_text(encoding="utf-8"))
            native_recomputed=hashlib.sha256((root/"01_asset/l7_29dof_neck_fixed_native.xml").read_bytes()).hexdigest().lower()
            matched_recomputed=hashlib.sha256((root/"01_asset/l7_29dof_neck_fixed_elbow_matched.xml").read_bytes()).hexdigest().lower()
            asset_ok = bool(asset_audit.get("pass") and asset_audit.get("changed_model_fields")==["dof_armature[right_elbow_pitch_joint]"] and abs(asset_audit.get("other_armature_max_diff",1))<1e-12 and native_recomputed==asset_audit.get("native_xml_sha256","").lower() and matched_recomputed==asset_audit.get("matched_xml_sha256","").lower())
            asset_detail=f"pass {asset_audit.get('pass')} changed {asset_audit.get('changed_model_fields')} recomputed ok" if asset_ok else f"asset fail recomputed {native_recomputed[:8]}/{matched_recomputed[:8]} vs audit"
            if not (root/"01_asset/xml_target_diff.patch").exists(): asset_ok=False; asset_detail+="; diff patch missing"
        except Exception as e: asset_ok=False; asset_detail=f"asset recompute error {e}"
        def check_config_inner():
            ok=True; details=[]
            try:
                ny=yaml.safe_load((root/"01_asset/mimic_dance_9_native.yaml").read_text(encoding="utf-8"))
                my=yaml.safe_load((root/"01_asset/mimic_dance_9_elbow_matched.yaml").read_text(encoding="utf-8"))
                if abs(float(ny.get("control_dt",0))-0.02)>1e-9: ok=False; details.append("native control_dt")
                if abs(float(my.get("control_dt",0))-0.02)>1e-9: ok=False; details.append("matched control_dt")
                expected_seq = ["left_hip_roll_joint", "left_hip_yaw_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_hip_pitch_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint", "waist_yaw_joint","waist_roll_joint","waist_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_arm_yaw_joint", "left_elbow_pitch_joint", "left_elbow_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint", "right_shoulder_pitch_joint","right_shoulder_roll_joint", "right_arm_yaw_joint","right_elbow_pitch_joint","right_elbow_yaw_joint","right_wrist_pitch_joint","right_wrist_roll_joint"]
                if ny["robot"]["sim2sim"]["joint_sequence"] != expected_seq: ok=False; details.append("native joint_sequence")
                if my["robot"]["sim2sim"]["joint_sequence"] != expected_seq: ok=False; details.append("matched joint_sequence")
                if ny["robot"]["sim2sim"]["xml_path"] != "l7_29dof_neck_fixed/l7_29dof_neck_fixed.xml": ok=False; details.append("native xml_path")
                if my["robot"]["sim2sim"]["xml_path"] != "l7_29dof_neck_fixed/p1_generated/l7_29dof_neck_fixed_elbow_matched.xml": ok=False; details.append("matched xml_path")
                import copy
                ncpy=copy.deepcopy(ny); mcpy=copy.deepcopy(my)
                ncpy["robot"]["sim2sim"]["xml_path"]="X"; mcpy["robot"]["sim2sim"]["xml_path"]="X"
                if ncpy != mcpy: ok=False; details.append("yaml diff beyond xml_path")
            except Exception as e: ok=False; details.append(f"yaml error {e}")
            joint_ok_manifest = all(chk["joint_ok"] for chk in all_checks)
            if not joint_ok_manifest: ok=False; details.append("joint order")
            return ok, "; ".join(details) if details else "ok"
        config_ok, config_detail = check_config_inner()
        def check_runtime_inner():
            ok=True; details=[]
            expected_policy="30e1fdede1bc6485e45c04e4f60aacd8ac5611e2ed1e67b1e3bc5b757a7bd659"
            expected_motion="bee3fe34a7ddefab4e0694a898a8b64e17cf89ac4850f1632843689539ec23fb"
            expected_canonical="7fdf68af39c0d4be7bcf8cd90f5ddb7718b93db240d657e78b97ad0dae6745cf"
            expected_runner="2ea7d4efdd4fe90fe515a2a8538fca4f444fee6a929f6b87f3a25966f1c20e8f"
            expected_native_model="69e975bad858be6be4cedd0c91a6d98882a48dc419a5aab0c19b94c99e3bba22"
            expected_matched_model="ab1e1862367be20a91edeefca12c6faedbc3e99a151d662cf42b4c56da98e7f9"
            for chk in all_checks:
                man=chk["manifest"]; summ=chk["summary"]; cond=chk["condition"]; rid=chk["run_id"]
                if man["hashes"]["policy"].lower()!=expected_policy: ok=False; details.append(f"{rid} policy")
                if man["hashes"]["motion"].lower()!=expected_motion: ok=False; details.append(f"{rid} motion")
                if man["hashes"]["canonical"].lower()!=expected_canonical: ok=False; details.append(f"{rid} canonical")
                if man["hashes"]["runner"].lower()!=expected_runner: ok=False; details.append(f"{rid} runner")
                exp_model=expected_native_model if cond=="native" else expected_matched_model
                if man["hashes"]["model"].lower()!=exp_model: ok=False; details.append(f"{rid} model")
                if abs(float(summ["control_dt"])-0.02)>1e-9 or abs(float(summ["physics_dt"])-0.002)>1e-9 or int(summ["ppc"])!=10: ok=False; details.append(f"{rid} dt/ppc")
                if int(summ["seed"])!=42 or summ["mode"]!="aligned" or summ["history_init"]!="repeat-first" or abs(float(summ["fixed_delay_ms"]))>1e-9 or abs(float(summ["gain_scale"])-1.0)>1e-9: ok=False; details.append(f"{rid} seed/mode")
                rt=man.get("runtime_mjmodel_actuator_params",{})
                if not rt: ok=False; details.append(f"{rid} missing runtime")
                else:
                    if cond=="native" and abs(float(rt.get("right_elbow_pitch_joint",{}).get("dof_armature",-1))-0.0685)>1e-9: ok=False; details.append(f"{rid} native arm")
                    if cond=="matched" and abs(float(rt.get("right_elbow_pitch_joint",{}).get("dof_armature",-1))-0.01)>1e-9: ok=False; details.append(f"{rid} matched arm")
                    if abs(float(rt.get("left_elbow_pitch_joint",{}).get("dof_armature",-1))-0.0685)>1e-9: ok=False; details.append(f"{rid} left elbow")
                if abs(float(summ["initial_q_max_err"]))>1e-6 or abs(float(summ["initial_dq_max_err"]))>1e-6: ok=False; details.append(f"{rid} init")
                if not (chk["t_maxdiff"]<=1e-6 and chk["fall_ok"] and chk["nan_ok"]): ok=False; details.append(f"{rid} t/fall/nan")
                for hk in ["policy","motion","model","runner","canonical"]:
                    if hk not in man["hashes"] or not man["hashes"][hk]: ok=False; details.append(f"{rid} missing {hk}")
            return ok, "; ".join(details) if details else "ok"
        runtime_ok, runtime_detail = check_runtime_inner()
        repeat_ok = repeat_pass
        time_ok = all(chk["time_ok"] for chk in all_checks)
        joint_ok = all(chk["joint_ok"] for chk in all_checks)
        shape_ok = all(chk["cycles_ok"] and chk["strict_inc"] for chk in all_checks)
        required_fields=["t","q(29)","dq(29)","q_des(29)","action(29)","root_pos(3)","root_quat(4)"]
        fields_ok=True
        for chk in all_checks:
            for rf in required_fields:
                if rf not in chk["trace"].files:
                    fields_ok=False
        shape_ok = shape_ok and fields_ok
        validity_checks={"input_hash": input_hash_ok, "asset": asset_ok, "config": config_ok, "runtime": runtime_ok, "repeat": repeat_ok, "time": time_ok, "joint": joint_ok, "shape": shape_ok}
        validity_details={"input_hash": input_hash_detail, "asset": asset_detail, "config": config_detail, "runtime": runtime_detail, "repeat": "pass" if repeat_ok else "fail", "time": "pass" if time_ok else "fail", "joint": "pass" if joint_ok else "fail", "shape": "pass" if shape_ok else "fail"}
        validity_pass=all(validity_checks.values())
        native_maes=[r["mae_q_0_2"] for r in per_run_data if r["condition"]=="native"]
        matched_maes=[r["mae_q_0_2"] for r in per_run_data if r["condition"]=="matched"]
        improvements=[(n-m) for n,m in zip(native_maes, matched_maes)]
        rel_improvements=[(n-m)/n if n!=0 else 0 for n,m in zip(native_maes, matched_maes)]
        abs_improve_mean=float(np.mean(improvements)) if improvements else 0
        rel_improve_mean=float(np.mean(rel_improvements)) if rel_improvements else 0
        direction_consistent=all(m<n for n,m in zip(native_maes, matched_maes))
        each_pass=all(rel>=0.10 and abs>=1e-3 for rel,abs in zip(rel_improvements, improvements))
        elbow_effect_pass=bool(rel_improve_mean>=0.10 and abs_improve_mean>=1e-3 and direction_consistent and each_pass)
        system_guards=[]
        for i in range(3):
            n_row=per_run_data[i]; m_row=per_run_data[i+3]
            n_fall=all_checks[i]["summary"]["fall_reason"]; m_fall=all_checks[i+3]["summary"]["fall_reason"]
            n_cycle=all_checks[i]["summary"]["fall_cycle"]; m_cycle=all_checks[i+3]["summary"]["fall_cycle"]
            n_c=n_cycle if n_cycle is not None else 9999; m_c=m_cycle if m_cycle is not None else 9999
            if m_fall is not None and n_fall is None: fall_guard=False
            elif m_fall is not None and n_fall is not None: fall_guard= not (m_c < n_c)
            else: fall_guard=True
            height_inc=m_row["root_height_rmse"]-n_row["root_height_rmse"]; height_thresh=max(n_row["root_height_rmse"]*0.10,0.002); height_guard=height_inc<=height_thresh
            roll_inc=m_row["root_roll_rmse"]-n_row["root_roll_rmse"]; roll_thresh=max(n_row["root_roll_rmse"]*0.10,0.005); roll_guard=roll_inc<=roll_thresh
            pitch_inc=m_row["root_pitch_rmse"]-n_row["root_pitch_rmse"]; pitch_thresh=max(n_row["root_pitch_rmse"]*0.10,0.005); pitch_guard=pitch_inc<=pitch_thresh
            joint_inc=m_row["mean_joint_q_rmse"]-n_row["mean_joint_q_rmse"]; joint_thresh=n_row["mean_joint_q_rmse"]*0.10; joint_guard=joint_inc<=joint_thresh
            guards_ok=bool(fall_guard and height_guard and roll_guard and pitch_guard and joint_guard)
            system_guards.append({"repeat": i+1, "fall_guard": fall_guard, "n_fall": n_fall, "m_fall": m_fall, "n_cycle": n_c, "m_cycle": m_c, "height_inc": height_inc, "height_thresh": height_thresh, "height_guard": height_guard, "roll_inc": roll_inc, "roll_thresh": roll_thresh, "roll_guard": roll_guard, "pitch_inc": pitch_inc, "pitch_thresh": pitch_thresh, "pitch_guard": pitch_guard, "joint_inc": joint_inc, "joint_thresh": joint_thresh, "joint_guard": joint_guard, "overall": guards_ok})
        system_guard_pass=all(g["overall"] for g in system_guards)
        if not validity_pass: decision="INVALID_RERUN"
        elif elbow_effect_pass and system_guard_pass: decision="SUPPORTED_TRANSFER_TO_FULLBODY"
        elif elbow_effect_pass and not system_guard_pass: decision="LOCAL_EFFECT_WITH_SYSTEM_TRADEOFF"
        elif not elbow_effect_pass: decision="ISOLATED_EFFECT_NOT_TRANSFERRED"
        else: decision="UNKNOWN"
        gate={"validity": {"checks": validity_checks, "details": validity_details, "pass": bool(validity_pass)}, "elbow_effect": {"target_joint": TARGET_JOINT, "window": "0-2s", "native_maes": native_maes, "matched_maes": matched_maes, "abs_improvements": improvements, "rel_improvements": rel_improvements, "mean_abs_improve": abs_improve_mean, "mean_rel_improve": rel_improve_mean, "direction_consistent": direction_consistent, "each_repeat_pass": bool(each_pass), "thresholds": {"rel":0.10, "abs":1e-3}, "require_each_pass": True, "pass": bool(elbow_effect_pass)}, "system_guard": {"per_repeat": system_guards, "pass": bool(system_guard_pass), "note": "no earlier failure, root height/roll/pitch increment <= max(10%,0.002/0.005), 29 joint mean RMSE not worse >10%; body/pelvis position shows small worsening but not gated"}, "decision": decision, "validity_pass": bool(validity_pass), "elbow_effect_pass": bool(elbow_effect_pass), "system_guard_pass": bool(system_guard_pass), "note": "conclusion limited to dance_9, current canonical 7fdf68, current control stack; do not claim full-body aligned or all joints 0.01; body position tradeoff noted"}
        return gate
    except Exception as e:
        import traceback
        return {"validity": {"checks": {}, "details": {"exception": str(e), "traceback": traceback.format_exc()}, "pass": False}, "elbow_effect": {"pass": False}, "system_guard": {"pass": False}, "decision": "INVALID_RERUN", "validity_pass": False, "elbow_effect_pass": False, "system_guard_pass": False, "error": str(e), "traceback": traceback.format_exc()}
