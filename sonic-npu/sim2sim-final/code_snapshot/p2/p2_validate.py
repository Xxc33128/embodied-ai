#!/usr/bin/env python3
"""p2_validate.py — pure validate(root) for P2 ARM_ONLY"""
import hashlib, json, pathlib
import numpy as np
import yaml
ARM_JOINTS_14 = ["left_shoulder_pitch_joint","left_shoulder_roll_joint","left_arm_yaw_joint","left_elbow_pitch_joint","left_elbow_yaw_joint","left_wrist_pitch_joint","left_wrist_roll_joint","right_shoulder_pitch_joint","right_shoulder_roll_joint","right_arm_yaw_joint","right_elbow_pitch_joint","right_elbow_yaw_joint","right_wrist_pitch_joint","right_wrist_roll_joint"]
ARM_INCREMENTAL_13 = [j for j in ARM_JOINTS_14 if j != "right_elbow_pitch_joint"]
TARGET_ELBOW="right_elbow_pitch_joint"
CONTROL_DT=0.02
def sha256(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load_manifest(p): return yaml.safe_load(pathlib.Path(p).read_text(encoding="utf-8"))
def quat_to_roll_pitch(q):
    w=q[:,0]; x=q[:,1]; y=q[:,2]; z=q[:,3]
    roll=np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch=np.arcsin(np.clip(2*(w*y-z*x), -1,1))
    return roll,pitch
def validate(root: pathlib.Path) -> dict:
    import csv, traceback
    try:
        root=pathlib.Path(root)
        isaac_ref=root/"00_inputs/isaac_reference/policy_trace.npz"
        native_runs=[root/f"00_inputs/mj_native_r{i}/policy_trace.npz" for i in [1,2,3]]
        elbow_runs=[root/f"00_inputs/mj_elbow_r{i}/policy_trace.npz" for i in [1,2,3]]
        arm_runs=[root/f"03_runs/mj_arm_r{i}/policy_trace.npz" for i in [1,2,3]]
        native_manifests=[root/f"00_inputs/mj_native_r{i}/run_manifest.yaml" for i in [1,2,3]]
        elbow_manifests=[root/f"00_inputs/mj_elbow_r{i}/run_manifest.yaml" for i in [1,2,3]]
        arm_manifests=[root/f"03_runs/mj_arm_r{i}/run_manifest.yaml" for i in [1,2,3]]
        # Check existence
        for p in [isaac_ref]+native_runs+elbow_runs+arm_runs+native_manifests+elbow_manifests+arm_manifests:
            if not p.exists():
                raise FileNotFoundError(f"missing {p}")
        isaac=np.load(str(isaac_ref))
        t_isaac=isaac["t"].astype(float)
        native_man0=load_manifest(native_manifests[0])
        joint_names=native_man0["policy_metadata"]["joint_names"]
        if TARGET_ELBOW not in joint_names: raise ValueError("elbow not in joint_names")
        idx_elbow=joint_names.index(TARGET_ELBOW)
        idx_13=[joint_names.index(j) for j in ARM_INCREMENTAL_13]

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
        for i,(p,m) in enumerate(zip(elbow_runs, elbow_manifests)):
            chk=check_run(p,m); chk["run_id"]=f"mj_elbow_r{i+1}"; chk["condition"]="elbow-only"; all_checks.append(chk)
        for i,(p,m) in enumerate(zip(arm_runs, arm_manifests)):
            chk=check_run(p,m); chk["run_id"]=f"mj_arm_r{i+1}"; chk["condition"]="arm-only"; all_checks.append(chk)

        def repeat_group(runs, manifests):
            traces=[np.load(str(p)) for p in runs]
            mans=[load_manifest(m) for m in manifests]
            fields=["t","q(29)","dq(29)","q_des(29)","action(29)","root_pos(3)","root_quat(4)"]
            body_fields=[]
            if "actual_body_pos_w(14,3)" in traces[0].files:
                body_fields=["actual_body_pos_w(14,3)","actual_body_quat_w(14,4)"]
            all_fields=fields+body_fields
            ok=True
            for f in all_fields:
                arrs=[tr[f] for tr in traces]
                md=max(float(np.max(np.abs(arrs[a]-arrs[b]))) for a in range(3) for b in range(a+1,3))
                if md>1e-7: ok=False
            for hk in ["runner","policy","motion","canonical"]:
                hs=[m["hashes"][hk] for m in mans]
                if not (hs[0]==hs[1]==hs[2]): ok=False
            model_hs=[m["hashes"]["model"] for m in mans]
            if not (model_hs[0]==model_hs[1]==model_hs[2]): ok=False
            return ok
        rep_native=repeat_group(native_runs, native_manifests)
        rep_elbow=repeat_group(elbow_runs, elbow_manifests)
        rep_arm=repeat_group(arm_runs, arm_manifests)
        repeat_pass=bool(rep_native and rep_elbow and rep_arm)

        # input hash check: use p2_input_manifest.csv
        def check_input_hash():
            p=root/"00_inputs/p2_input_manifest.csv"
            if not p.exists():
                return False, "missing p2_input_manifest.csv"
            try:
                rows=list(csv.DictReader(open(p, encoding="utf-8")))
                ok=True; details=[]
                for row in rows:
                    f=row["file"]; src=row["source_relative"]; h=row["sha256"]
                    pp=root/f
                    if not pp.exists():
                        ok=False; details.append(f"missing {f}")
                        continue
                    recomputed=hashlib.sha256(pp.read_bytes()).hexdigest().lower()
                    if recomputed != h.lower():
                        ok=False; details.append(f"{f} hash mismatch")
                    if len(h)!=64:
                        ok=False; details.append(f"{f} hash len")
                return ok, "; ".join(details) if details else "all PASS"
            except Exception as e:
                return False, str(e)
        input_hash_ok, input_hash_detail = check_input_hash()

        # asset check
        try:
            audit=json.loads((root/"01_asset/fullbody_arm_asset_audit.json").read_text(encoding="utf-8"))
            native_recomputed=hashlib.sha256((root/"01_asset/l7_29dof_neck_fixed_native.xml").read_bytes()).hexdigest().lower()
            arm_recomputed=hashlib.sha256((root/"01_asset/l7_29dof_neck_fixed_arm_matched.xml").read_bytes()).hexdigest().lower()
            # R6-1 (fail-closed): ONLY the packaged elbow XML is accepted; no absolute-path fallbacks.
            EXPECTED_ELBOW_SHA="ab1e1862367be20a91edeefca12c6faedbc3e99a151d662cf42b4c56da98e7f9"
            elbow_path=root/"00_inputs/p1_elbow_only/l7_29dof_neck_fixed_elbow_matched.xml"
            if not elbow_path.exists():
                asset_ok=False
                asset_detail="INVALID_P2 reason=PACKAGED_ELBOW_XML_MISSING"
                raise FileNotFoundError("PACKAGED_ELBOW_XML_MISSING: 00_inputs/p1_elbow_only/l7_29dof_neck_fixed_elbow_matched.xml")
            elbow_recomputed=hashlib.sha256(elbow_path.read_bytes()).hexdigest().lower()
            asset_ok = bool(audit.get("pass") and set(audit.get("changed_model_fields_native_vs_arm", audit.get("changed_model_fields", [])))==set(f"dof_armature[{j}]" for j in ARM_JOINTS_14) and audit.get("other_armature_max_diff",1)<1e-12 and native_recomputed==audit.get("native_xml_sha256","").lower() and arm_recomputed==audit.get("arm_xml_sha256","").lower())
            # R6-1: elbow hash must be bound into validity (recomputed == audit == pinned constant)
            if not (elbow_recomputed==audit.get("elbow_xml_sha256","").lower()==EXPECTED_ELBOW_SHA):
                asset_ok=False
                asset_detail=f"elbow hash binding fail recomputed={elbow_recomputed[:12]} audit={str(audit.get('elbow_xml_sha256'))[:12]} expected={EXPECTED_ELBOW_SHA[:12]}"
            # also check elbow vs arm if present
            if "changed_model_fields_elbow_vs_arm" in audit:
                if set(audit["changed_model_fields_elbow_vs_arm"]) != set(f"dof_armature[{j}]" for j in ARM_INCREMENTAL_13):
                    asset_ok=False
            if asset_ok:
                asset_detail = "pass"
            elif not asset_detail.startswith("elbow hash"):
                asset_detail = f"asset fail changed {audit.get('changed_model_fields')}"
            if not (root/"01_asset/xml_arm_target_diff.patch").exists():
                asset_ok=False; asset_detail+="; diff missing"
        except Exception as e:
            asset_ok=False; asset_detail=f"asset error {e}"

        def check_config():
            try:
                ny=yaml.safe_load((root/"01_asset/mimic_dance_9_native.yaml").read_text(encoding="utf-8"))
                ay=yaml.safe_load((root/"01_asset/mimic_dance_9_arm_matched.yaml").read_text(encoding="utf-8"))
                if abs(float(ny.get("control_dt",0))-0.02)>1e-9: return False, "native control_dt"
                if abs(float(ay.get("control_dt",0))-0.02)>1e-9: return False, "arm control_dt"
                if ny["robot"]["sim2sim"]["xml_path"] != "l7_29dof_neck_fixed/l7_29dof_neck_fixed.xml": return False, "native xml_path"
                if ay["robot"]["sim2sim"]["xml_path"] != "l7_29dof_neck_fixed/p2_generated/l7_29dof_neck_fixed_arm_matched.xml": return False, "arm xml_path"
                import copy
                ncpy=copy.deepcopy(ny); acpy=copy.deepcopy(ay)
                ncpy["robot"]["sim2sim"]["xml_path"]="X"; acpy["robot"]["sim2sim"]["xml_path"]="X"
                if ncpy != acpy: return False, "yaml diff beyond xml_path"
                return True, "ok"
            except Exception as e:
                return False, str(e)
        config_ok, config_detail = check_config()

        def check_runtime():
            try:
                expected_policy="30e1fdede1bc6485e45c04e4f60aacd8ac5611e2ed1e67b1e3bc5b757a7bd659"
                expected_motion="bee3fe34a7ddefab4e0694a898a8b64e17cf89ac4850f1632843689539ec23fb"
                expected_canonical="7fdf68af39c0d4be7bcf8cd90f5ddb7718b93db240d657e78b97ad0dae6745cf"
                expected_runner="2ea7d4efdd4fe90fe515a2a8538fca4f444fee6a929f6b87f3a25966f1c20e8f"
                expected_native_model="69e975bad858be6be4cedd0c91a6d98882a48dc419a5aab0c19b94c99e3bba22"
                expected_elbow_model="ab1e1862367be20a91edeefca12c6faedbc3e99a151d662cf42b4c56da98e7f9"
                expected_arm_model="2b8a0313479a054cf2639d52785b8a926d70a5b5d97b9021be566b223e865b8c"
                ok=True; details=[]
                for chk in all_checks:
                    man=chk["manifest"]; summ=chk["summary"]; cond=chk["condition"]; rid=chk["run_id"]
                    if man["hashes"]["policy"].lower()!=expected_policy: ok=False; details.append(f"{rid} policy")
                    if man["hashes"]["motion"].lower()!=expected_motion: ok=False; details.append(f"{rid} motion")
                    if man["hashes"]["canonical"].lower()!=expected_canonical: ok=False; details.append(f"{rid} canonical")
                    if man["hashes"]["runner"].lower()!=expected_runner: ok=False; details.append(f"{rid} runner")
                    exp_model=expected_native_model if cond=="native" else expected_elbow_model if cond=="elbow-only" else expected_arm_model
                    if man["hashes"]["model"].lower()!=exp_model: ok=False; details.append(f"{rid} model")
                    if abs(float(summ["control_dt"])-0.02)>1e-9 or abs(float(summ["physics_dt"])-0.002)>1e-9 or int(summ["ppc"])!=10: ok=False; details.append(f"{rid} dt/ppc")
                    if int(summ["seed"])!=42 or summ["mode"]!="aligned" or summ["history_init"]!="repeat-first": ok=False; details.append(f"{rid} seed/mode")
                    rt=man.get("runtime_mjmodel_actuator_params",{})
                    if not rt: ok=False; details.append(f"{rid} missing runtime")
                    else:
                        # check 14 arm joints 0.01 for arm, elbow only has 1, native 0
                        if cond=="arm-only":
                            for j in ARM_JOINTS_14:
                                if abs(float(rt.get(j,{}).get("dof_armature",-1))-0.01)>1e-9: ok=False; details.append(f"{rid} arm {j}")
                        elif cond=="elbow-only":
                            for j in ARM_JOINTS_14:
                                exp = 0.01 if j==TARGET_ELBOW else (0.0685 if "shoulder" in j or "arm_yaw" in j or j=="left_elbow_pitch_joint" else 0.03 if "wrist" in j or "elbow_yaw" in j else 0)
                                # Actually for elbow-only, only right_elbow_pitch is 0.01, others are native
                                if j==TARGET_ELBOW:
                                    if abs(float(rt.get(j,{}).get("dof_armature",-1))-0.01)>1e-9: ok=False; details.append(f"{rid} elbow arm")
                                else:
                                    # check not 0.01 (except maybe coincident? but native shouldn't be 0.01)
                                    if abs(float(rt.get(j,{}).get("dof_armature",-1))-0.01)<1e-9: ok=False; details.append(f"{rid} elbow other {j} should not be 0.01")
                        else: # native
                            for j in ARM_JOINTS_14:
                                if abs(float(rt.get(j,{}).get("dof_armature",-1))-0.01)<1e-9: ok=False; details.append(f"{rid} native {j} should not be 0.01")
                return ok, "; ".join(details) if details else "ok"
            except Exception as e:
                return False, str(e)
        runtime_ok, runtime_detail = check_runtime()

        repeat_ok=repeat_pass
        time_ok=all(chk["time_ok"] for chk in all_checks)
        joint_ok=all(chk["joint_ok"] for chk in all_checks)
        shape_ok=all(chk["cycles_ok"] and chk["strict_inc"] for chk in all_checks)
        # check required fields
        required=["t","q(29)","dq(29)","q_des(29)","action(29)","root_pos(3)","root_quat(4)","actual_body_pos_w(14,3)","actual_body_quat_w(14,4)"]
        for chk in all_checks:
            for rf in required:
                if rf not in chk["trace"].files:
                    shape_ok=False

        nan_ok=all(chk["nan_ok"] for chk in all_checks)
        validity_checks={"input_hash": input_hash_ok, "asset": asset_ok, "config": config_ok, "runtime": runtime_ok, "repeat": repeat_pass, "time": time_ok, "joint": joint_ok, "shape": shape_ok, "nan": nan_ok}
        validity_details={"input_hash": input_hash_detail, "asset": asset_detail, "config": config_detail, "runtime": runtime_detail, "repeat": "pass" if repeat_pass else "fail", "time": "pass" if time_ok else "fail", "joint": "pass" if joint_ok else "fail", "shape": "pass" if shape_ok else "fail", "nan": "pass" if nan_ok else "fail"}
        validity_pass=all(validity_checks.values())

        # compute metrics for gate
        # E13 per run
        per_run_E13={}
        per_run_elbow={}
        per_run_root_h={}
        per_run_root_roll={}
        per_run_root_pitch={}
        per_run_joint_mean={}
        per_run_body_pos={}
        per_run_anchor_pos={}
        per_run_body_quat={}
        for chk in all_checks:
            trace=chk["trace"]
            q=trace["q(29)"].astype(float); iq=isaac["q(29)"].astype(float)
            mask_0_2=(t_isaac>=-1e-9)&(t_isaac<=2.0+1e-9)
            # E13
            q13=q[:, idx_13]; iq13=iq[:, idx_13]
            per_joint_rmse_0_2=np.sqrt(np.mean((q13[mask_0_2]-iq13[mask_0_2])**2, axis=0))
            E13=float(np.mean(per_joint_rmse_0_2))
            per_run_E13[chk["run_id"]]=E13
            # elbow
            qe=q[:, idx_elbow]; iqe=iq[:, idx_elbow]
            elbow_mae=float(np.mean(np.abs(qe[mask_0_2]-iqe[mask_0_2])))
            per_run_elbow[chk["run_id"]]=elbow_mae
            # root
            rp=trace["root_pos(3)"].astype(float); irp=isaac["root_pos(3)"].astype(float)
            per_run_root_h[chk["run_id"]]=float(np.sqrt(np.mean((rp[:,2]-irp[:,2])**2)))
            rq=trace["root_quat(4)"].astype(float); irq=isaac["root_quat(4)"].astype(float)
            roll,pitch=quat_to_roll_pitch(rq); iroll,ipitch=quat_to_roll_pitch(irq)
            per_run_root_roll[chk["run_id"]]=float(np.sqrt(np.mean((roll-iroll)**2)))
            per_run_root_pitch[chk["run_id"]]=float(np.sqrt(np.mean((pitch-ipitch)**2)))
            # joint mean
            joint_rmses=np.sqrt(np.mean((q-iq)**2, axis=0))
            per_run_joint_mean[chk["run_id"]]=float(np.mean(joint_rmses))
            # body
            if "actual_body_pos_w(14,3)" in trace.files:
                bpos=trace["actual_body_pos_w(14,3)"].astype(float); ibpos=isaac["actual_body_pos_w(14,3)"].astype(float)
                per_run_body_pos[chk["run_id"]]=float(np.sqrt(np.mean((bpos-ibpos)**2)))
                per_run_anchor_pos[chk["run_id"]]=float(np.sqrt(np.mean((bpos[:,0,:]-ibpos[:,0,:])**2)))
                bquat=trace["actual_body_quat_w(14,4)"].astype(float); ibquat=isaac["actual_body_quat_w(14,4)"].astype(float)
                # quat error
                def qangle(q1,q2):
                    q1=q1/np.linalg.norm(q1, axis=-1, keepdims=True)
                    q2=q2/np.linalg.norm(q2, axis=-1, keepdims=True)
                    dot=np.abs(np.sum(q1*q2, axis=-1))
                    return 2*np.arccos(np.clip(dot, -1,1))
                angs=qangle(bquat, ibquat)
                per_run_body_quat[chk["run_id"]]=float(np.sqrt(np.mean(angs**2)))

        # arm group effect: D vs C per repeat
        # For each repeat i (1..3): compare mj_elbow_ri vs mj_arm_ri
        rel_improvements=[]; abs_improvements=[]; direction_consistent=True
        per_repeat_pass=[]
        perch_joint_improvements=[] # for median
        # need per-joint rmse for each repeat to check 8/13 and median
        # compute per_joint rmse for elbow and arm per repeat
        import collections
        # Build per joint rmse per run for 13 joints
        per_run_joint_rmse_13={}
        for chk in all_checks:
            if chk["condition"] in ["elbow-only","arm-only"]:
                trace=chk["trace"]
                q=trace["q(29)"].astype(float); iq=isaac["q(29)"].astype(float)
                mask_0_2=(t_isaac>=-1e-9)&(t_isaac<=2.0+1e-9)
                rmses={}
                for j, idx in zip(ARM_INCREMENTAL_13, idx_13):
                    qj=q[:, idx]; iqj=iq[:, idx]
                    rmses[j]=float(np.sqrt(np.mean((qj[mask_0_2]-iqj[mask_0_2])**2)))
                per_run_joint_rmse_13[chk["run_id"]]=rmses

        for i in [1,2,3]:
            e_id=f"mj_elbow_r{i}"; a_id=f"mj_arm_r{i}"
            E_e=per_run_E13[e_id]; E_a=per_run_E13[a_id]
            abs_imp=E_e - E_a
            rel_imp=abs_imp/E_e if E_e!=0 else 0
            rel_improvements.append(rel_imp); abs_improvements.append(abs_imp)
            if not (E_a < E_e): direction_consistent=False
            # per repeat pass: rel>=0.10 and abs>=1e-3
            per_repeat_pass.append(rel_imp>=0.10 and abs_imp>=1e-3)
            # per joint improvement for this repeat
            joint_imps=[]
            for j in ARM_INCREMENTAL_13:
                re_rmse=per_run_joint_rmse_13[e_id][j]
                ra_rmse=per_run_joint_rmse_13[a_id][j]
                joint_imps.append(re_rmse - ra_rmse)
            perch_joint_improvements.append(joint_imps)

        # Overall arm_group_effect_pass requires:
        # each repeat rel>=10% and abs>=1e-3, 3/3 direction, at least 8 joints improve, median >0
        # For median and 8/13, we can average over repeats or check per repeat? Spec says "13个新增关节中至少8个 joint_rmse 改善" and "中位数 >0" — likely overall mean across repeats or per repeat? We'll check using mean over repeats.
        # Compute mean per joint improvement across 3 repeats
        mean_per_joint=[]
        for j_idx, j in enumerate(ARM_INCREMENTAL_13):
            vals=[perch_joint_improvements[rep][j_idx] for rep in range(3)]
            mean_per_joint.append(float(np.mean(vals)))
        n_improved = sum(1 for v in mean_per_joint if v>0)
        median_improvement = float(np.median(mean_per_joint))
        arm_group_effect_pass = bool(all(per_repeat_pass) and direction_consistent and n_improved>=8 and median_improvement>0)

        # right elbow preservation: D vs C elbow MAE increment <= max(C*10%,1e-3)
        elbow_preservation_pass=True
        for i in [1,2,3]:
            e_id=f"mj_elbow_r{i}"; a_id=f"mj_arm_r{i}"
            e_mae=per_run_elbow[e_id]; a_mae=per_run_elbow[a_id]
            inc=a_mae - e_mae
            thresh=max(e_mae*0.10, 1e-3)
            if inc > thresh:
                elbow_preservation_pass=False

        # system guard: D vs C per repeat
        system_guards=[]
        for i in [1,2,3]:
            e_id=f"mj_elbow_r{i}"; a_id=f"mj_arm_r{i}"
            # fall
            # find all_checks entries for these ids
            e_chk=next(c for c in all_checks if c["run_id"]==e_id)
            a_chk=next(c for c in all_checks if c["run_id"]==a_id)
            e_fall=e_chk["summary"]["fall_reason"]; a_fall=a_chk["summary"]["fall_reason"]
            e_cycle=e_chk["summary"]["fall_cycle"]; a_cycle=a_chk["summary"]["fall_cycle"]
            e_c=e_cycle if e_cycle is not None else 9999; a_c=a_cycle if a_cycle is not None else 9999
            if a_fall is not None and e_fall is None: fall_guard=False
            elif a_fall is not None and e_fall is not None: fall_guard= not (a_c < e_c)
            else: fall_guard=True
            # root height
            eh=per_run_root_h[e_id]; ah=per_run_root_h[a_id]
            height_inc=ah - eh; height_thresh=max(eh*0.10, 0.002); height_guard=height_inc<=height_thresh
            er=per_run_root_roll[e_id]; ar=per_run_root_roll[a_id]
            roll_inc=ar-er; roll_thresh=max(er*0.10,0.005); roll_guard=roll_inc<=roll_thresh
            ep=per_run_root_pitch[e_id]; ap=per_run_root_pitch[a_id]
            pitch_inc=ap-ep; pitch_thresh=max(ep*0.10,0.005); pitch_guard=pitch_inc<=pitch_thresh
            ej=per_run_joint_mean[e_id]; aj=per_run_joint_mean[a_id]
            joint_inc=aj-ej; joint_thresh=ej*0.10; joint_guard=joint_inc<=joint_thresh
            eb=per_run_body_pos[e_id]; ab=per_run_body_pos[a_id]
            body_inc=ab-eb; body_thresh=max(eb*0.10,0.010); body_guard=body_inc<=body_thresh
            ea=per_run_anchor_pos[e_id]; aa=per_run_anchor_pos[a_id]
            anchor_inc=aa-ea; anchor_thresh=max(ea*0.10,0.010); anchor_guard=anchor_inc<=anchor_thresh
            # body quat
            ebq=per_run_body_quat[e_id]; abq=per_run_body_quat[a_id]
            body_q_inc=abq-ebq; body_q_thresh=max(ebq*0.10,0.010); body_q_guard=body_q_inc<=body_q_thresh
            # overall
            overall=bool(fall_guard and height_guard and roll_guard and pitch_guard and joint_guard and body_guard and anchor_guard and body_q_guard and elbow_preservation_pass)
            system_guards.append({"repeat": i, "fall_guard": fall_guard, "height_inc": height_inc, "height_thresh": height_thresh, "height_guard": height_guard, "roll_inc": roll_inc, "roll_thresh": roll_thresh, "roll_guard": roll_guard, "pitch_inc": pitch_inc, "pitch_thresh": pitch_thresh, "pitch_guard": pitch_guard, "joint_inc": joint_inc, "joint_thresh": joint_thresh, "joint_guard": joint_guard, "body_inc": body_inc, "body_thresh": body_thresh, "body_guard": body_guard, "anchor_inc": anchor_inc, "anchor_thresh": anchor_thresh, "anchor_guard": anchor_guard, "body_q_inc": body_q_inc, "body_q_thresh": body_q_thresh, "body_q_guard": body_q_guard, "overall": overall})
        system_guard_pass=all(g["overall"] for g in system_guards) and elbow_preservation_pass

        if not validity_pass: decision="INVALID_P2"
        elif arm_group_effect_pass and system_guard_pass: decision="SUPPORTED_ARM_GROUP_TRANSFER"
        elif arm_group_effect_pass and not system_guard_pass: decision="ARM_GROUP_EFFECT_WITH_SYSTEM_TRADEOFF"
        elif not arm_group_effect_pass: decision="NO_INCREMENTAL_ARM_GROUP_EFFECT"
        else: decision="UNKNOWN"

        gate={"validity": {"checks": validity_checks, "details": validity_details, "pass": bool(validity_pass)}, "arm_group_effect": {"E13_elbow": [per_run_E13[f"mj_elbow_r{i}"] for i in [1,2,3]], "E13_arm": [per_run_E13[f"mj_arm_r{i}"] for i in [1,2,3]], "rel_improvements": rel_improvements, "abs_improvements": abs_improvements, "mean_rel": float(np.mean(rel_improvements)) if rel_improvements else 0, "mean_abs": float(np.mean(abs_improvements)) if abs_improvements else 0, "direction_consistent": direction_consistent, "per_repeat_pass": per_repeat_pass, "n_improved_joints": n_improved, "median_improvement": median_improvement, "mean_per_joint": mean_per_joint, "pass": bool(arm_group_effect_pass)}, "right_elbow_preservation": {"elbow_elbow": [per_run_elbow[f"mj_elbow_r{i}"] for i in [1,2,3]], "elbow_arm": [per_run_elbow[f"mj_arm_r{i}"] for i in [1,2,3]], "pass": bool(elbow_preservation_pass)}, "system_guard": {"per_repeat": system_guards, "pass": bool(system_guard_pass)}, "decision": decision, "validity_pass": bool(validity_pass), "arm_group_effect_pass": bool(arm_group_effect_pass), "right_elbow_preservation_pass": bool(elbow_preservation_pass), "system_guard_pass": bool(system_guard_pass)}
        return gate
    except Exception as e:
        import traceback
        return {"validity": {"checks": {}, "details": {"exception": str(e), "traceback": traceback.format_exc()}, "pass": False}, "arm_group_effect": {"pass": False}, "right_elbow_preservation": {"pass": False}, "system_guard": {"pass": False}, "decision": "INVALID_P2", "validity_pass": False, "arm_group_effect_pass": False, "right_elbow_preservation_pass": False, "system_guard_pass": False, "error": str(e), "traceback": traceback.format_exc()}

if __name__=="__main__":
    import argparse, json, pathlib, sys
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="package root")
    args=ap.parse_args()
    root=pathlib.Path(args.root)
    gate=validate(root)
    print(json.dumps(gate, indent=2))
    sys.exit(0 if gate.get("validity_pass") else 1)
