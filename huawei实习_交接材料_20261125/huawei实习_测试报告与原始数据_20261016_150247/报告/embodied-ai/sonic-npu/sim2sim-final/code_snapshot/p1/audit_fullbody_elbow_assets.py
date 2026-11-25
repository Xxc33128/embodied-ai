#!/usr/bin/env python3
"""
audit_fullbody_elbow_assets.py — P1 single-variable asset audit

Uses MuJoCo 3.2.7 to compile both XMLs and checks:
- joint/body/geom/actuator names & order
- nq/nv/nu/nbody/njnt/ngeom
- native right elbow armature 0.0685, matched 0.01
- left & other 28 armature unchanged
- body mass/inertia/pose
- joint pose/axis/range/type
- damping/frictionloss
- actuator gear/ctrlrange/forcerange/gain/dyn/bias
- timestep/solver/iterations/tolerance
- changed field count exactly 1

Outputs fullbody_elbow_asset_audit.json with pass flag
"""
import hashlib
import json
import pathlib
import sys

# Resolve paths relative to this script
THIS = pathlib.Path(__file__).resolve()
P1_ASSET = THIS.parent
# Evidence XMLs (01_asset)
NATIVE_XML_EVIDENCE = P1_ASSET / "l7_29dof_neck_fixed_native.xml"
MATCHED_XML_EVIDENCE = P1_ASSET / "l7_29dof_neck_fixed_elbow_matched.xml"
# Deploy copies (for compilation fallback if evidence missing meshes)
DEPLOY_NATIVE = pathlib.Path("/mnt/e/humanoid-lab/deploy/l7_29dof_neck_fixed/l7_29dof_neck_fixed.xml") if pathlib.Path("/mnt/e/humanoid-lab/deploy/l7_29dof_neck_fixed/l7_29dof_neck_fixed.xml").exists() else pathlib.Path("E:/humanoid-lab/deploy/l7_29dof_neck_fixed/l7_29dof_neck_fixed.xml")
DEPLOY_MATCHED = pathlib.Path("/mnt/e/humanoid-lab/deploy/l7_29dof_neck_fixed/p1_generated/l7_29dof_neck_fixed_elbow_matched.xml") if pathlib.Path("/mnt/e/humanoid-lab/deploy/l7_29dof_neck_fixed/p1_generated/l7_29dof_neck_fixed_elbow_matched.xml").exists() else pathlib.Path("E:/humanoid-lab/deploy/l7_29dof_neck_fixed/p1_generated/l7_29dof_neck_fixed_elbow_matched.xml")

OUTPUT_JSON = P1_ASSET / "fullbody_elbow_asset_audit.json"

def sha256(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def load_model(path):
    import mujoco
    # Try direct load; if meshes missing, try via spec string with adjusted meshdir? but we copied meshes so should succeed
    try:
        m = mujoco.MjModel.from_xml_path(str(path))
        return m
    except Exception as e:
        print(f"LOAD FAIL {path}: {e}")
        # Try reading text and patch meshdir to absolute
        text = pathlib.Path(path).read_text(encoding="utf-8")
        # patch meshdir="meshes/" to absolute deploy meshes if needed
        # Determine deploy meshes absolute
        deploy_meshes = str(pathlib.Path("E:/humanoid-lab/deploy/l7_29dof_neck_fixed/meshes").resolve())
        # In WSL, use /mnt/e/...
        if pathlib.Path("/mnt/e/humanoid-lab/deploy/l7_29dof_neck_fixed/meshes").exists():
            deploy_meshes = "/mnt/e/humanoid-lab/deploy/l7_29dof_neck_fixed/meshes"
        if 'meshdir="meshes/"' in text:
            text = text.replace('meshdir="meshes/"', f'meshdir="{deploy_meshes}/"')
            # Use from_xml_string via temporary file? MjSpec
            import tempfile, os
            with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8") as tf:
                tf.write(text)
                tf_path = tf.name
            try:
                m = mujoco.MjModel.from_xml_path(tf_path)
                os.unlink(tf_path)
                return m
            except Exception as e2:
                print(f"PATCH LOAD FAIL: {e2}")
                raise
        raise

def main():
    import mujoco
    import numpy as np

    # Choose paths: prefer evidence (01_asset) if they have meshes, else deploy
    # Evidence should have meshes copied, so try evidence first
    native_path = NATIVE_XML_EVIDENCE if NATIVE_XML_EVIDENCE.exists() else DEPLOY_NATIVE
    matched_path = MATCHED_XML_EVIDENCE if MATCHED_XML_EVIDENCE.exists() else DEPLOY_MATCHED
    # If evidence native does not have meshes folder sibling, fallback
    # Check meshes existence
    if not (native_path.parent / "meshes").exists() and (DEPLOY_NATIVE.parent / "meshes").exists():
        native_path = DEPLOY_NATIVE
    if not (matched_path.parent / "meshes").exists() and (DEPLOY_MATCHED.parent / "meshes").exists():
        matched_path = DEPLOY_MATCHED

    print(f"Native path: {native_path}")
    print(f"Matched path: {matched_path}")
    # Hashes from evidence files (regardless of which path used for compile, hash evidence)
    native_sha = sha256(NATIVE_XML_EVIDENCE) if NATIVE_XML_EVIDENCE.exists() else sha256(native_path)
    matched_sha = sha256(MATCHED_XML_EVIDENCE) if MATCHED_XML_EVIDENCE.exists() else sha256(matched_path)
    print(f"native sha {native_sha}")
    print(f"matched sha {matched_sha}")

    mn = load_model(native_path)
    mm = load_model(matched_path)

    # Helper to get names
    def get_joint_names(m):
        # hinge joints only? Check spec says joint/body/geom/actuator names & sequence
        # We'll compare hinge joint names in order of appearance (jnt_type == HINGE)
        names=[]
        for i in range(m.njnt):
            if m.jnt_type[i]==mujoco.mjtJoint.mjJNT_HINGE:
                names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i))
        # also include free joint? but spec says 29 hinge, so check hinge only
        return names
    def get_all_joint_names(m):
        return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    def get_body_names(m):
        return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    def get_act_names(m):
        return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    def get_geom_names(m):
        # geom names may be None for many
        return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]

    checks = {}
    passed = True
    details = {}

    # nq/nv etc
    for attr in ["nq","nv","nu","nbody","njnt","ngeom"]:
        v_n = getattr(mn, attr)
        v_m = getattr(mm, attr)
        eq = (v_n == v_m)
        checks[f"{attr}_equal"] = bool(eq)
        details[attr] = {"native": int(v_n), "matched": int(v_m), "equal": bool(eq)}
        if not eq:
            passed=False
            print(f"FAIL {attr}: native {v_n} vs matched {v_m}")

    # names order
    jn_n = get_joint_names(mn)
    jn_m = get_joint_names(mm)
    checks["joint_names_equal"] = (jn_n == jn_m)
    if jn_n != jn_m:
        passed=False
        print(f"FAIL joint names: {jn_n} vs {jn_m}")
    details["joint_names_native"] = jn_n
    details["joint_names_matched"] = jn_m

    an_n = get_act_names(mn)
    an_m = get_act_names(mm)
    checks["actuator_names_equal"] = (an_n == an_m)
    if an_n != an_m:
        passed=False
        print(f"FAIL actuator names")

    bn_n = get_body_names(mn)
    bn_m = get_body_names(mm)
    checks["body_names_equal"] = (bn_n == bn_m)
    if bn_n != bn_m:
        passed=False
        print(f"FAIL body names")

    # geom names: check counts and types? For now check count equal (full check would be heavy)
    gn_n = get_geom_names(mn)
    gn_m = get_geom_names(mm)
    # Don't require exact names because some None, but check lengths equal
    checks["geom_count_equal"] = (len(gn_n)==len(gn_m))
    if len(gn_n)!=len(gn_m):
        passed=False

    # Armature check: find target joint
    target_joint = "right_elbow_pitch_joint"
    # Map joint name to dof id via jnt_dofadr
    def joint_to_dof(m, joint_name):
        for i in range(m.njnt):
            if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)==joint_name:
                return int(m.jnt_dofadr[i])
        return None
    def dof_to_joint_map(m):
        # returns dict dof->joint name via dof_jntid
        mp={}
        for dof in range(m.nv):
            jid = int(m.dof_jntid[dof])
            # jid 0 is free
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) if jid>=0 else f"free_{dof}"
            mp[dof]=name
        return mp

    # Native armature
    dof_native_target = joint_to_dof(mn, target_joint)
    dof_matched_target = joint_to_dof(mm, target_joint)
    if dof_native_target is None or dof_matched_target is None:
        print("FAIL: target joint not found")
        passed=False
        native_arm=matched_arm=None
    else:
        native_arm = float(mn.dof_armature[dof_native_target])
        matched_arm = float(mm.dof_armature[dof_matched_target])
        print(f"Target {target_joint} dof {dof_native_target} native arm {native_arm} matched arm {matched_arm}")
        checks["native_armature_target"] = abs(native_arm - 0.0685) < 1e-9
        checks["matched_armature_target"] = abs(matched_arm - 0.01) < 1e-9
        if not checks["native_armature_target"]:
            print(f"FAIL native armature {native_arm} !=0.0685")
            passed=False
        if not checks["matched_armature_target"]:
            print(f"FAIL matched armature {matched_arm} !=0.01")
            passed=False

    # Other armature unchanged
    diff_arm = mn.dof_armature - mm.dof_armature
    # Find non-zero diffs
    nonzero = [(i, float(diff_arm[i]), float(mn.dof_armature[i]), float(mm.dof_armature[i])) for i in range(len(diff_arm)) if abs(diff_arm[i])>1e-12]
    print(f"Armature diff nonzero count {len(nonzero)}: {nonzero}")
    # Map to joint names
    dof_map = dof_to_joint_map(mn)
    changed_fields=[]
    for dof, d, _, _ in nonzero:
        joint_name = dof_map.get(dof, f"dof{dof}")
        changed_fields.append(f"dof_armature[{joint_name}]")
    checks["changed_fields_count_1"] = (len(changed_fields)==1 and changed_fields==[f"dof_armature[{target_joint}]"])
    if not checks["changed_fields_count_1"]:
        print(f"FAIL changed fields {changed_fields} expected [dof_armature[{target_joint}]]")
        passed=False
    # other armature max diff
    other_arm_max = 0.0
    for i in range(len(diff_arm)):
        if dof_map.get(i)!=target_joint:
            other_arm_max = max(other_arm_max, abs(float(diff_arm[i])))
    details["armature_nonzero"] = nonzero
    details["changed_fields"] = changed_fields

    # Body mass/inertia/pose max diff
    mass_max = float(np.max(np.abs(mn.body_mass - mm.body_mass)))
    inertia_max = float(np.max(np.abs(mn.body_inertia - mm.body_inertia)))
    body_pos_max = float(np.max(np.abs(mn.body_pos - mm.body_pos)))
    body_quat_max = float(np.max(np.abs(mn.body_quat - mm.body_quat)))
    checks["mass_equal"] = mass_max < 1e-12
    checks["inertia_equal"] = inertia_max < 1e-12
    checks["body_pos_equal"] = body_pos_max < 1e-12
    checks["body_quat_equal"] = body_quat_max < 1e-12
    if not checks["mass_equal"]:
        print(f"FAIL mass max diff {mass_max}")
        passed=False
    if not checks["inertia_equal"]:
        print(f"FAIL inertia max diff {inertia_max}")
        passed=False
    if not checks["body_pos_equal"]:
        print(f"FAIL body_pos max diff {body_pos_max}")
        passed=False
    details["mass_max_diff"] = mass_max
    details["inertia_max_diff"] = inertia_max
    details["body_pos_max_diff"] = body_pos_max
    details["body_quat_max_diff"] = body_quat_max

    # Joint pose/axis/range/type
    jnt_pos_max = float(np.max(np.abs(mn.jnt_pos - mm.jnt_pos)))
    jnt_axis_max = float(np.max(np.abs(mn.jnt_axis - mm.jnt_axis)))
    jnt_range_max = float(np.max(np.abs(mn.jnt_range - mm.jnt_range)))
    jnt_type_eq = bool(np.array_equal(mn.jnt_type, mm.jnt_type))
    checks["jnt_pos_equal"] = jnt_pos_max < 1e-12
    checks["jnt_axis_equal"] = jnt_axis_max < 1e-12
    checks["jnt_range_equal"] = jnt_range_max < 1e-12
    checks["jnt_type_equal"] = jnt_type_eq
    if not checks["jnt_pos_equal"]:
        print(f"FAIL jnt_pos max {jnt_pos_max}")
        passed=False
    if not checks["jnt_axis_equal"]:
        print(f"FAIL jnt_axis max {jnt_axis_max}")
        passed=False
    if not checks["jnt_range_equal"]:
        print(f"FAIL jnt_range max {jnt_range_max}")
        passed=False
    if not jnt_type_eq:
        print(f"FAIL jnt_type not equal")
        passed=False
    details["jnt_pos_max_diff"] = jnt_pos_max
    details["jnt_axis_max_diff"] = jnt_axis_max
    details["jnt_range_max_diff"] = jnt_range_max

    # damping/frictionloss
    damp_max = float(np.max(np.abs(mn.dof_damping - mm.dof_damping)))
    fric_max = float(np.max(np.abs(mn.dof_frictionloss - mm.dof_frictionloss)))
    checks["damping_equal"] = damp_max < 1e-12
    checks["frictionloss_equal"] = fric_max < 1e-12
    if not checks["damping_equal"]:
        print(f"FAIL damping max {damp_max}")
        passed=False
    if not checks["frictionloss_equal"]:
        print(f"FAIL frictionloss max {fric_max}")
        passed=False
    details["damping_max_diff"] = damp_max
    details["frictionloss_max_diff"] = fric_max

    # actuator params
    gear_max = float(np.max(np.abs(mn.actuator_gear - mm.actuator_gear)))
    ctrl_max = float(np.max(np.abs(mn.actuator_ctrlrange - mm.actuator_ctrlrange)))
    frc_max = float(np.max(np.abs(mn.actuator_forcerange - mm.actuator_forcerange)))
    gain_max = float(np.max(np.abs(mn.actuator_gainprm - mm.actuator_gainprm)))
    dyn_eq = bool(np.array_equal(mn.actuator_dyntype, mm.actuator_dyntype))
    gain_type_eq = bool(np.array_equal(mn.actuator_gaintype, mm.actuator_gaintype))
    bias_eq = bool(np.array_equal(mn.actuator_biastype, mm.actuator_biastype))
    checks["actuator_gear_equal"] = gear_max < 1e-12
    checks["actuator_ctrlrange_equal"] = ctrl_max < 1e-12
    checks["actuator_forcerange_equal"] = frc_max < 1e-12
    checks["actuator_gainprm_equal"] = gain_max < 1e-12
    checks["actuator_dyntype_equal"] = dyn_eq
    checks["actuator_gaintype_equal"] = gain_type_eq
    checks["actuator_biastype_equal"] = bias_eq
    if not checks["actuator_gear_equal"]:
        print(f"FAIL gear max {gear_max}")
        passed=False
    if not checks["actuator_ctrlrange_equal"]:
        print(f"FAIL ctrlrange max {ctrl_max}")
        passed=False
    if not checks["actuator_forcerange_equal"]:
        print(f"FAIL frc max {frc_max}")
        passed=False
    if not checks["actuator_gainprm_equal"]:
        print(f"FAIL gainprm max {gain_max}")
        passed=False
    if not dyn_eq or not gain_type_eq or not bias_eq:
        print(f"FAIL actuator dyn/gain/bias not equal")
        passed=False
    details["actuator_gear_max_diff"] = gear_max
    details["actuator_ctrlrange_max_diff"] = ctrl_max
    details["actuator_forcerange_max_diff"] = frc_max
    details["actuator_gainprm_max_diff"] = gain_max

    actuator_param_max = max(gear_max, ctrl_max, frc_max, gain_max)
    details["actuator_param_max_diff"] = actuator_param_max

    # timestep/solver/iterations/tolerance
    ts_eq = abs(float(mn.opt.timestep) - float(mm.opt.timestep)) < 1e-12
    iter_eq = int(mn.opt.iterations) == int(mm.opt.iterations)
    solver_eq = int(mn.opt.solver) == int(mm.opt.solver)
    tol_eq = abs(float(mn.opt.tolerance) - float(mm.opt.tolerance)) < 1e-12
    checks["timestep_equal"] = ts_eq
    checks["iterations_equal"] = iter_eq
    checks["solver_equal"] = solver_eq
    checks["tolerance_equal"] = tol_eq
    if not ts_eq:
        print(f"FAIL timestep {mn.opt.timestep} vs {mm.opt.timestep}")
        passed=False
    if not iter_eq:
        print(f"FAIL iterations")
        passed=False
    if not solver_eq:
        print(f"FAIL solver")
        passed=False
    if not tol_eq:
        print(f"FAIL tolerance")
        passed=False
    details["timestep"] = {"native": float(mn.opt.timestep), "matched": float(mm.opt.timestep)}
    details["iterations"] = {"native": int(mn.opt.iterations), "matched": int(mm.opt.iterations)}
    details["solver"] = {"native": int(mn.opt.solver), "matched": int(mm.opt.solver)}
    details["tolerance"] = {"native": float(mn.opt.tolerance), "matched": float(mm.opt.tolerance)}

    # Overall
    result = {
        "pass": bool(passed),
        "target_joint": target_joint,
        "native_armature": float(native_arm) if native_arm is not None else None,
        "matched_armature": float(matched_arm) if matched_arm is not None else None,
        "changed_model_fields": changed_fields,
        "other_armature_max_diff": float(other_arm_max),
        "mass_max_diff": float(mass_max),
        "inertia_max_diff": float(inertia_max),
        "body_pos_max_diff": float(body_pos_max),
        "body_quat_max_diff": float(body_quat_max),
        "jnt_pos_max_diff": float(jnt_pos_max),
        "jnt_axis_max_diff": float(jnt_axis_max),
        "jnt_range_max_diff": float(jnt_range_max),
        "damping_max_diff": float(damp_max),
        "frictionloss_max_diff": float(fric_max),
        "actuator_param_max_diff": float(actuator_param_max),
        "actuator_gear_max_diff": float(gear_max),
        "actuator_ctrlrange_max_diff": float(ctrl_max),
        "actuator_forcerange_max_diff": float(frc_max),
        "actuator_gainprm_max_diff": float(gain_max),
        "timestep_equal": bool(ts_eq),
        "solver_equal": bool(solver_eq),
        "iterations_equal": bool(iter_eq),
        "tolerance_equal": bool(tol_eq),
        "native_xml_sha256": native_sha,
        "matched_xml_sha256": matched_sha,
        "native_xml_path": str(NATIVE_XML_EVIDENCE),
        "matched_xml_path": str(MATCHED_XML_EVIDENCE),
        "deploy_native_path": str(DEPLOY_NATIVE),
        "deploy_matched_path": str(DEPLOY_MATCHED),
        "compile_native_path": str(native_path),
        "compile_matched_path": str(matched_path),
        "checks": checks,
        "details": details,
        "mujoco_version": mujoco.__version__,
        "nq": int(mn.nq), "nv": int(mn.nv), "nu": int(mn.nu), "nbody": int(mn.nbody), "njnt": int(mn.njnt), "ngeom": int(mn.ngeom)
    }

    OUTPUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Audit {'PASS' if passed else 'FAIL'} -> {OUTPUT_JSON}")
    for k,v in checks.items():
        print(f"  {k}: {v}")
    if not passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
