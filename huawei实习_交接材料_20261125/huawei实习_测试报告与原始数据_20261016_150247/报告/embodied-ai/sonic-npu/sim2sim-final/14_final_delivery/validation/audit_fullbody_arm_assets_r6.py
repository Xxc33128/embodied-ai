#!/usr/bin/env python3
"""audit_fullbody_arm_assets.py — P2 ARM_ONLY 14 joints asset audit (native vs arm, elbow vs arm) with --root"""
import hashlib, json, pathlib, sys, argparse
ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_arm_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_arm_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"
]
TARGET_ELBOW = "right_elbow_pitch_joint"

def sha256(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def load_model(path):
    import mujoco
    try:
        return mujoco.MjModel.from_xml_path(str(path))
    except Exception as e:
        print(f"LOAD FAIL {path}: {e}")
        text = pathlib.Path(path).read_text(encoding="utf-8")
        if 'meshdir="meshes/"' in text:
            import tempfile, os
            # fallback to absolute meshdir inside asset
            # Try to find meshes relative to asset dir
            # Use the asset's meshes folder
            asset_dir = pathlib.Path(path).parent
            mesh_src = asset_dir / "meshes"
            if mesh_src.exists():
                alt_text = text.replace('meshdir="meshes/"', f'meshdir="{mesh_src.as_posix()}/"')
            else:
                alt_text = text.replace('meshdir="meshes/"', 'meshdir="/mnt/e/humanoid-lab/deploy/l7_29dof_neck_fixed/meshes/"')
            with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8") as tf:
                tf.write(alt_text); tf_path=tf.name
            try:
                m=mujoco.MjModel.from_xml_path(tf_path)
                os.unlink(tf_path)
                return m
            except Exception as e2:
                print(f"PATCH FAIL {e2}")
                raise
        raise

def joint_to_dof(m, name):
    import mujoco
    for i in range(m.njnt):
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)==name:
            return int(m.jnt_dofadr[i])
    return None

def dof_map(m):
    import mujoco
    mp={}
    for dof in range(m.nv):
        jid=int(m.dof_jntid[dof])
        mp[dof]=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) if jid>=0 else f"free_{dof}"
    return mp

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None, help="package root (contains 00_inputs, 01_asset)")
    args=parser.parse_args()
    if args.root:
        root=pathlib.Path(args.root).resolve()
    else:
        # default: 01_asset's parent's parent is root if script is in 01_asset, else use parent
        this=pathlib.Path(__file__).resolve()
        if this.parent.name=="01_asset":
            root=this.parent.parent
        else:
            root=this.parent
    P2_ASSET=root/"01_asset"
    # R6-1 (fail-closed): ONLY the packaged P1 elbow XML under 00_inputs/p1_elbow_only is accepted.
    # Absolute-path fallbacks (/mnt/e/..., E:/...) removed: missing packaged XML => FAIL exit 1.
    NATIVE_XML=P2_ASSET/"l7_29dof_neck_fixed_native.xml"
    ELBOW_XML=root/"00_inputs/p1_elbow_only/l7_29dof_neck_fixed_elbow_matched.xml"
    if not ELBOW_XML.exists():
        print(f"FAIL: PACKAGED_ELBOW_XML_MISSING {ELBOW_XML}")
        sys.exit(1)
    ARM_XML=P2_ASSET/"l7_29dof_neck_fixed_arm_matched.xml"
    OUTPUT_JSON=P2_ASSET/"fullbody_arm_asset_audit.json"
    for p in [NATIVE_XML, ELBOW_XML, ARM_XML]:
        if not p.exists():
            print(f"FAIL: missing {p}")
            sys.exit(1)
        # check hash for elbow
        if p==ELBOW_XML:
            h=sha256(p)
            expected="ab1e1862367be20a91edeefca12c6faedbc3e99a151d662cf42b4c56da98e7f9"
            if h.lower()!=expected:
                print(f"FAIL: elbow hash {h} != expected {expected}")
                sys.exit(1)
            else:
                print(f"elbow hash {h[:12]} OK")
    import mujoco, numpy as np
    mn=load_model(NATIVE_XML)
    me=load_model(ELBOW_XML)
    ma=load_model(ARM_XML)
    checks={}; passed=True
    for attr in ["nq","nv","nu","nbody","njnt","ngeom"]:
        eq=getattr(mn,attr)==getattr(me,attr)==getattr(ma,attr)
        checks[f"{attr}_equal"]=bool(eq)
        if not eq:
            print(f"FAIL {attr}: native {getattr(mn,attr)} elbow {getattr(me,attr)} arm {getattr(ma,attr)}")
            passed=False
    def get_names(m, obj_type):
        import mujoco
        names=[]
        if obj_type==mujoco.mjtObj.mjOBJ_JOINT:
            for i in range(m.njnt):
                n=mujoco.mj_id2name(m, obj_type, i)
                if n: names.append(n)
        elif obj_type==mujoco.mjtObj.mjOBJ_BODY:
            for i in range(m.nbody):
                n=mujoco.mj_id2name(m, obj_type, i)
                if n: names.append(n)
        elif obj_type==mujoco.mjtObj.mjOBJ_GEOM:
            for i in range(m.ngeom):
                n=mujoco.mj_id2name(m, obj_type, i)
                if n: names.append(n)
        elif obj_type==mujoco.mjtObj.mjOBJ_ACTUATOR:
            for i in range(m.nu):
                n=mujoco.mj_id2name(m, obj_type, i)
                if n: names.append(n)
        return names
    import mujoco
    for obj, label in [(mujoco.mjtObj.mjOBJ_JOINT, "joint"), (mujoco.mjtObj.mjOBJ_BODY, "body"), (mujoco.mjtObj.mjOBJ_GEOM, "geom"), (mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator")]:
        n_names=get_names(mn,obj); e_names=get_names(me,obj); a_names=get_names(ma,obj)
        eq=n_names==e_names==a_names
        checks[f"{label}_names_equal"]=bool(eq)
        if not eq:
            print(f"FAIL {label} names differ")
            passed=False
    dm_n=dof_map(mn); dm_e=dof_map(me); dm_a=dof_map(ma)
    diff_na=mn.dof_armature - ma.dof_armature
    changed_na=[]
    for dof in range(mn.nv):
        if abs(float(diff_na[dof]))>1e-12:
            joint=dm_n.get(dof, f"dof{dof}")
            changed_na.append(f"dof_armature[{joint}]")
    expected_na=set(f"dof_armature[{j}]" for j in ARM_JOINTS)
    changed_na_set=set(changed_na)
    checks["native_vs_arm_changed_14"]=(changed_na_set==expected_na)
    if changed_na_set!=expected_na:
        print(f"FAIL native_vs_arm changed {changed_na_set} vs expected {expected_na}")
        passed=False
    else:
        print(f"native_vs_arm changed 14 PASS")
    for joint in ARM_JOINTS:
        dof=joint_to_dof(ma,joint)
        val=float(ma.dof_armature[dof])
        if abs(val-0.01)>1e-9:
            print(f"FAIL arm runtime {joint} {val} !=0.01")
            passed=False
            checks["arm_runtime_0.01"]=False
            break
    else:
        checks["arm_runtime_0.01"]=True
        print("arm runtime 14*0.01 PASS")
    other_max=0.0
    for dof in range(mn.nv):
        joint=dm_n.get(dof)
        if joint not in ARM_JOINTS:
            other_max=max(other_max, abs(float(diff_na[dof])))
    checks["other_15_armature_0"]=other_max<1e-12
    if other_max>=1e-12:
        print(f"FAIL other 15 armature max {other_max}")
        passed=False
    else:
        print(f"other 15 armature 0 PASS max {other_max}")
    mass_max=float(np.max(np.abs(mn.body_mass - ma.body_mass)))
    inertia_max=float(np.max(np.abs(mn.body_inertia - ma.body_inertia)))
    body_pos_max=float(np.max(np.abs(mn.body_pos - ma.body_pos)))
    body_quat_max=float(np.max(np.abs(mn.body_quat - ma.body_quat)))
    checks["mass_0"]=mass_max<1e-12; checks["inertia_0"]=inertia_max<1e-12; checks["body_pos_0"]=body_pos_max<1e-12; checks["body_quat_0"]=body_quat_max<1e-12
    if mass_max>=1e-12 or inertia_max>=1e-12 or body_pos_max>=1e-12 or body_quat_max>=1e-12:
        print(f"FAIL body mass {mass_max} inertia {inertia_max} pos {body_pos_max} quat {body_quat_max}")
        passed=False
    jnt_axis_max=float(np.max(np.abs(np.array(mn.jnt_axis) - np.array(ma.jnt_axis))))
    jnt_range_max=float(np.max(np.abs(np.array(mn.jnt_range) - np.array(ma.jnt_range))))
    jnt_type_eq=bool(np.all(np.array(mn.jnt_type)==np.array(ma.jnt_type)))
    checks["jnt_axis_0"]=jnt_axis_max<1e-12; checks["jnt_range_0"]=jnt_range_max<1e-12; checks["jnt_type_eq"]=jnt_type_eq
    if jnt_axis_max>=1e-12 or jnt_range_max>=1e-12 or not jnt_type_eq:
        print(f"FAIL jnt axis {jnt_axis_max} range {jnt_range_max} type {jnt_type_eq}")
        passed=False
    damp_max=float(np.max(np.abs(mn.dof_damping - ma.dof_damping)))
    fric_max=float(np.max(np.abs(mn.dof_frictionloss - ma.dof_frictionloss)))
    checks["damping_0"]=damp_max<1e-12; checks["frictionloss_0"]=fric_max<1e-12
    if damp_max>=1e-12 or fric_max>=1e-12:
        print(f"FAIL damp {damp_max} fric {fric_max}")
        passed=False
    gear_max=float(np.max(np.abs(mn.actuator_gear - ma.actuator_gear)))
    ctrl_max=float(np.max(np.abs(np.array(mn.actuator_ctrlrange).flatten() - np.array(ma.actuator_ctrlrange).flatten())))
    force_max=float(np.max(np.abs(np.array(mn.actuator_forcerange).flatten() - np.array(ma.actuator_forcerange).flatten())))
    gainprm_max=float(np.max(np.abs(np.array(mn.actuator_gainprm) - np.array(ma.actuator_gainprm))))
    checks["actuator_gear_0"]=gear_max<1e-12; checks["actuator_ctrlrange_0"]=ctrl_max<1e-12; checks["actuator_forcerange_0"]=force_max<1e-12; checks["actuator_gainprm_0"]=gainprm_max<1e-12
    if gear_max>=1e-12 or ctrl_max>=1e-12 or force_max>=1e-12 or gainprm_max>=1e-12:
        print(f"FAIL actuator gear {gear_max} ctrl {ctrl_max} force {force_max} gain {gainprm_max}")
        passed=False
    timestep_eq=abs(float(mn.opt.timestep)-float(ma.opt.timestep))<1e-12
    iter_eq=int(mn.opt.iterations)==int(ma.opt.iterations)
    solver_eq=int(mn.opt.solver)==int(ma.opt.solver)
    tol_eq=abs(float(mn.opt.tolerance)-float(ma.opt.tolerance))<1e-12
    checks["timestep_eq"]=timestep_eq; checks["iterations_eq"]=iter_eq; checks["solver_eq"]=solver_eq; checks["tolerance_eq"]=tol_eq
    if not (timestep_eq and iter_eq and solver_eq and tol_eq):
        print(f"FAIL timestep {mn.opt.timestep} {ma.opt.timestep} iter {mn.opt.iterations} {ma.opt.iterations} solver {mn.opt.solver} {ma.opt.solver}")
        passed=False
    diff_ea=me.dof_armature - ma.dof_armature
    changed_ea=[]
    for dof in range(me.nv):
        if abs(float(diff_ea[dof]))>1e-12:
            joint=dm_e.get(dof, f"dof{dof}")
            changed_ea.append(f"dof_armature[{joint}]")
    dof_elbow=joint_to_dof(me, TARGET_ELBOW)
    elbow_diff=float(diff_ea[dof_elbow])
    checks["elbow_diff_0"]=abs(elbow_diff)<1e-12
    if abs(elbow_diff)>=1e-12:
        print(f"FAIL elbow diff {elbow_diff}")
        passed=False
    else:
        print(f"elbow diff 0 PASS")
    expected_ea=set(f"dof_armature[{j}]" for j in ARM_JOINTS if j != TARGET_ELBOW)
    changed_ea_set=set(changed_ea)
    checks["elbow_vs_arm_changed_13"]=(changed_ea_set==expected_ea)
    if changed_ea_set!=expected_ea:
        print(f"FAIL elbow_vs_arm changed {changed_ea_set} vs expected {expected_ea}")
        passed=False
    else:
        print(f"elbow_vs_arm changed 13 PASS")
    mass_ea=float(np.max(np.abs(me.body_mass - ma.body_mass)))
    if mass_ea>=1e-12:
        print(f"FAIL elbow vs arm mass {mass_ea}")
        passed=False
        checks["elbow_vs_arm_mass_0"]=False
    else:
        checks["elbow_vs_arm_mass_0"]=True
    result={
        "pass": bool(passed),
        "target_joints_14": ARM_JOINTS,
        "native_armatures": {j: float(mn.dof_armature[joint_to_dof(mn,j)]) for j in ARM_JOINTS},
        "elbow_armatures": {j: float(me.dof_armature[joint_to_dof(me,j)]) for j in ARM_JOINTS},
        "arm_armatures": {j: float(ma.dof_armature[joint_to_dof(ma,j)]) for j in ARM_JOINTS},
        "changed_model_fields_native_vs_arm": sorted(changed_na),
        "changed_model_fields_elbow_vs_arm": sorted(changed_ea),
        "other_armature_max_diff": float(other_max),
        "mass_max_diff": mass_max,
        "inertia_max_diff": inertia_max,
        "damping_max_diff": damp_max,
        "frictionloss_max_diff": fric_max,
        "actuator_param_max_diff": float(max(gear_max, ctrl_max, force_max, gainprm_max)),
        "native_xml_sha256": sha256(NATIVE_XML),
        "elbow_xml_sha256": sha256(ELBOW_XML),
        "arm_xml_sha256": sha256(ARM_XML),
        "checks": checks,
        "mujoco_version": mujoco.__version__,
        "nq": int(mn.nq), "nv": int(mn.nv), "nu": int(mn.nu), "nbody": int(mn.nbody), "njnt": int(mn.njnt), "ngeom": int(mn.ngeom)
    }
    OUTPUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"audit {'PASS' if passed else 'FAIL'} -> {OUTPUT_JSON}")
    if not passed:
        sys.exit(1)

if __name__=="__main__": main()
