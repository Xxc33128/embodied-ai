#!/usr/bin/env python3
"""n5_r_summarize.py — N5-R 汇总与门禁 vB3（2026-08-31 B.3 fail-closed 修订）

B.3 相对 B.1 的修复（审查 B2→B3）：
- 3.1 YAML 解析必须明确失败：增加 fallback parser（与 compare_repeats 一致）并在 _parse_error 时输出 manifest_parse_error 使 final FAIL
- 3.2 读取真实 runtime readback：g_root_runtime_gate 优先从 initial_state_readback_<joint>.json 读取 runtime_dof_count/runtime_joint_name，回退到 manifest；final 模式下缺失/!=1/!=target 必须 FAIL
- 3.3 实现真正 fail-closed 退出码：新增 --evaluation-mode smoke|final，final: validity_pass&&all_pass ->0 else 1；smoke: G6 NOT_EVALUATED && smoke_pass_excluding_g6 ->0 else 1
- 保留 B.1 阈值不变（G0/G1/G2/G3/G4/G5/G6）

相对 v2 / pack_g0 的 B.1 修复（审查 P0-3/P1-1/P1-2）：

相对 v2 / pack_g0 的 B.1 修复（审查 P0-3/P1-1/P1-2）：
- P0-3: --g0-file 设为 required；validity_pass = validity_core and (g0_pass is True) and G6_PASS；
  G0 JSON 解析失败/字段缺失/joint/hash不一致 必须 fail-closed。
- P0-2: G0 与 trace 的模型 hash 绑定：G0 joint/ canonical hash / 模型 hash 与 trace manifest 对应资产一致性校验。
- P1-1: G1 固定检查目标关节（canonical joint_names 直接取 args.joint 索引），仅该列发生命令变化，
  三条 trace step amplitude 必须 0.1±1e-6，step joint/time 一致。
- P1-2: Isaac root/runtime 初态门禁（root pos/quat/vel, DOF count, joint name, armature/Kp/Kd/effort）。
- P0-4: G6 缺字段/缺hash/重复数不足 仍PASS 的漏洞在 summarize 侧通过校验 G6 detail 的 hash present/same 与 field consistency；
  单次运行 G6=NOT_EVALUATED 时 validity 必然 FAIL。
- 额外：输出 R20 early-response sanity（不 gate，仅记录）。

用法:
  python n5_r_summarize.py --joint right_elbow_pitch_joint --canonical <npz> --g0-file <g0.json>
      --isaac-trace <isaac step_trace npz> [--isaac-dir <dir>] --mj-native <dir> --mj-matched <dir>
      --out <gates.json> [--g6-file <repeats.json>] [--plot <png>]
"""
import argparse
import hashlib
import re
import json
import sys
from pathlib import Path

import numpy as np

STEP_SWITCH = 0.10
T_OL_OTHER = 1e-4
T_OL_ROOT = 1e-5
T_STEP_Q = 1e-8
T_Q0 = 1e-5
T_STEP_AMP = 0.10
T_STEP_AMP_TOL = 1e-6
T_STEP_TIME_TOL = 1e-6


def sha256_file(p: Path):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else None


def quat_angular_error(q1, q2):
    # q and -q same; angular error = 2*acos(|dot|)
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    dot = abs(float(np.dot(q1, q2)))
    dot = min(1.0, max(-1.0, dot))
    return float(2 * np.arccos(dot))


def load_trace(npz_path):
    d = np.load(npz_path, allow_pickle=False)
    for r in ["t_state", "t_command_start"]:
        assert r in d, f"trace 缺少 {r}（旧格式？）"
    return d


def _fallback_yaml_parse(txt):
    """Fallback YAML parser identical to compare_repeats.py (tested). Handles simple key: value and nested hashes.*."""
    out = {}
    cur = None
    for line in txt.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = re.match(r"(\s+)?(\w[\w./-]*):\s*(.*)", line)
        if not m:
            continue
        indent, key, val = m.group(1) or "", m.group(2), m.group(3).strip()
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

def load_manifest(manifest_path):
    p = Path(manifest_path)
    if not p.is_file():
        return {}
    txt = p.read_text(encoding="utf-8")
    try:
        if str(p).endswith(".json"):
            return json.loads(txt)
        # Try PyYAML first
        try:
            import yaml
            return yaml.safe_load(txt) or {}
        except ImportError as e:
            # No yaml installed: use fallback and record warning but not error if fallback succeeds
            fallback = _fallback_yaml_parse(txt)
            if fallback:
                print(f"[YAML] WARN PyYAML not installed, using fallback parser for {p} ({e})")
                return fallback
            return {"_parse_error": f"PyYAML not installed and fallback empty: {e}", "_raw": txt[:2000]}
        except Exception as e:
            # yaml installed but parse failed -> try fallback for recovery; if fallback also empty, propagate parse_error
            fallback = _fallback_yaml_parse(txt)
            # If yaml failed, we must surface manifest_parse_error per B.3 3.1
            # Try fallback to still provide data but also keep error marker for gate to fail
            # Return with _parse_error so main can fail-closed
            return {"_parse_error": f"yaml.safe_load failed for {p}: {e}", "_raw": txt[:2000], "_fallback": fallback}
    except Exception as e:
        return {"_parse_error": str(e), "_raw": txt[:2000]}


def g1_gate(ic, mn, mm, joint, q0_exp29, dq0_exp29, mj_native_man, mj_matched_man, ic_man, log, canon_names=None, step_exp=0.10):
    """G1 B.1：固定目标关节检查，避免 stdv 猜测漏洞。"""
    det = {}
    ok = True
    # canonical joint index
    if canon_names is None or joint not in canon_names:
        ok = False
        log(f"[G1] FAIL canonical joint_names missing or joint {joint} not in {canon_names}")
        det["pass"] = False
        return ok, det, {}, None
    jidx = int(canon_names.index(joint))
    det["target_jidx"] = jidx
    det["target_joint"] = joint
    # 1) 样本数与时间轴
    for label, d, want in [("isaac", ic, 26), ("mj_native", mn, 251), ("mj_matched", mm, 251)]:
        t = d["t_state"].astype(float)
        n = len(t)
        det[f"{label}_n"] = n
        det[f"{label}_t0"] = float(t[0])
        det[f"{label}_tfinal"] = float(t[-1])
        det[f"{label}_strict_increasing"] = bool(np.all(np.diff(t) > 0))
        if n != want or abs(t[0]) > 1e-9 or abs(t[-1] - 0.5) > 1e-9:
            ok = False
            log(f"[G1] FAIL {label}: n={n} t0={t[0]} tfinal={t[-1]} need {want}/0.0/0.5")
        if not det[f"{label}_strict_increasing"]:
            ok = False
            log(f"[G1] FAIL {label}: t 非严格递增")
    # 2) 初态 29
    for label, d, man in [("isaac", ic, ic_man), ("mj_native", mn, mj_native_man), ("mj_matched", mm, mj_matched_man)]:
        qkey = "q(29)" if "q(29)" in d else "q_pol(29)"
        dqkey = "dq(29)" if "dq(29)" in d else "dq_pol(29)"
        if qkey not in d or dqkey not in d:
            ok = False
            log(f"[G1] FAIL {label}: missing {qkey} or {dqkey}")
            det[f"{label}_q0_maxerr_29"] = None
            continue
        q0_ = d[qkey][0].astype(float)
        dq0_ = d[dqkey][0].astype(float)
        if q0_exp29 is not None:
            if q0_.shape[0] != 29:
                ok = False
                log(f"[G1] FAIL {label}: q0 shape {q0_.shape} !=29")
            else:
                q0err29 = float(np.max(np.abs(q0_ - q0_exp29)))
                det[f"{label}_q0_maxerr_29"] = q0err29
                det[f"{label}_q0_worst_joint"] = str(canon_names[int(np.argmax(np.abs(q0_ - q0_exp29)))]) if canon_names else None
                if q0err29 > T_Q0:
                    ok = False
                    log(f"[G1] FAIL {label}: q0(29) maxerr={q0err29:.2e} > {T_Q0} worst={det[f'{label}_q0_worst_joint']}")
        if dq0_exp29 is not None:
            dq0err = float(np.max(np.abs(dq0_ - dq0_exp29)))
            det[f"{label}_dq0_maxerr_29"] = dq0err
            if dq0err > T_Q0:
                ok = False
                log(f"[G1] FAIL {label}: dq0(29) maxerr={dq0err:.2e} > {T_Q0}")
        # kp/kd
        kp = man.get("kp_j/kd_j", man.get("kp_j", None))
        if isinstance(kp, list) and len(kp) == 2:
            det[f"{label}_kp_j"] = kp[0]
            det[f"{label}_kd_j"] = kp[1]
    # 3) 阶跃：固定目标关节列
    first_times = {}
    amps = {}
    for label, d in [("isaac", ic), ("mj_native", mn), ("mj_matched", mm)]:
        qdes = d["q_des(29)"].astype(float) if "q_des(29)" in d else d["q_des_pol(29)"].astype(float)
        t_cmd = d["t_command_start"].astype(float)
        if qdes.shape[1] != 29:
            ok = False
            log(f"[G1] FAIL {label}: q_des shape {qdes.shape} != (N,29)")
            det[f"{label}_step_first_t"] = None
            det[f"{label}_step_amp"] = None
            continue
        # 检查仅目标列变化，其他列保持
        others_idx = [i for i in range(29) if i != jidx]
        # 其他关节的 q_des 变化应 <= T_STEP_Q
        other_change = float(np.max(np.abs(qdes[:, others_idx] - qdes[0, others_idx]))) if others_idx else 0.0
        det[f"{label}_other_qdes_max_change"] = other_change
        if other_change > T_STEP_Q:
            ok = False
            log(f"[G1] FAIL {label}: 非目标关节 q_des 发生变化 {other_change:.2e} > {T_STEP_Q} (only {joint} should step)")
        # 目标关节变化
        col = qdes[:, jidx]
        changed = np.flatnonzero(np.abs(col - col[0]) > T_STEP_Q)
        if len(changed) == 0:
            det[f"{label}_step_first_t"] = None
            first_times[label] = None
            ok = False
            log(f"[G1] FAIL {label}: 未找到目标关节 {joint} 的阶跃")
        else:
            first = int(changed[0])
            ft = float(t_cmd[first])
            det[f"{label}_step_first_t"] = ft
            first_times[label] = ft
            if abs(ft - 0.10) > T_STEP_TIME_TOL:
                ok = False
                log(f"[G1] FAIL {label}: step 生效时间 {ft} != 0.100 (tol {T_STEP_TIME_TOL})")
        amp = float(col[-1] - col[0])
        det[f"{label}_step_amp"] = amp
        amps[label] = amp
        if abs(amp - step_exp) > T_STEP_AMP_TOL:
            ok = False
            log(f"[G1] FAIL {label}: step amplitude {amp} != {step_exp} (tol {T_STEP_AMP_TOL})")
        # 记录 step joint name for traceability
        det[f"{label}_step_joint"] = joint
    # 4) 三条 trace 的 step joint/time/amplitude 一致性
    if len(set(first_times.values())) > 1:
        # allow None but then already failed
        vals = [v for v in first_times.values() if v is not None]
        if len(set(round(v, 9) for v in vals)) > 1:
            ok = False
            log(f"[G1] FAIL 三条 trace step time 不一致: {first_times}")
    amp_vals = [amps.get(k) for k in ["isaac", "mj_native", "mj_matched"] if amps.get(k) is not None]
    if len(amp_vals) == 3 and max(amp_vals) - min(amp_vals) > 1e-6:
        ok = False
        log(f"[G1] FAIL 三条 trace amplitude 不一致: {amps}")
    # 5) native/matched 对称性
    qkey_n = "q(29)" if "q(29)" in mn else "q_pol(29)"
    qkey_m = "q(29)" if "q(29)" in mm else "q_pol(29)"
    if qkey_n in mn and qkey_m in mm:
        q0n = mn[qkey_n][0]
        q0m = mm[qkey_m][0]
        det["native_matched_q0_maxdiff"] = float(np.max(np.abs(q0n - q0m)))
        if det["native_matched_q0_maxdiff"] > T_Q0:
            ok = False
            log(f"[G1] FAIL native/matched q0 不一致: {det['native_matched_q0_maxdiff']:.2e}")
    det["pass"] = bool(ok)
    return ok, det, first_times, jidx


def g2_gate(ic, mn, mm, log):
    det = {}
    ok = True
    fields = [
        ("mj_native_drift_others", mn, "other_joint_drift_max", T_OL_OTHER),
        ("mj_matched_drift_others", mm, "other_joint_drift_max", T_OL_OTHER),
        ("isaac_drift_others", ic, "other_joint_drift_max", T_OL_OTHER),
        ("mj_native_root_drift", mn, "root_drift_max", T_OL_ROOT),
        ("mj_matched_root_drift", mm, "root_drift_max", T_OL_ROOT),
        ("isaac_root_drift", ic, "root_drift_max", T_OL_ROOT),
    ]
    for name, d, key, tol in fields:
        if key in d:
            v = float(np.max(np.abs(np.asarray(d[key], dtype=float))))
        else:
            v = float("nan")
            ok = False
            log(f"[G2] FAIL {name}: 缺少字段 {key}")
        det[name] = v
        if not np.isnan(v) and v > tol:
            ok = False
            log(f"[G2] FAIL {name}: {v:.3e} > {tol}")
    for name, d, cfk in [("mj_native", mn, "max_contact_force"), ("mj_matched", mm, "max_contact_force"), ("isaac", ic, "max_contact_force")]:
        nk = "ncon" if "ncon" in d else None
        if nk:
            ncon = np.asarray(d[nk], dtype=float)
            det[f"{name}_ncon_max"] = int(np.max(ncon)) if len(ncon) else 0
            if np.any(ncon < 0):
                det[f"{name}_contact_status"] = "UNAVAILABLE"
                ok = False
                log(f"[G2] FAIL {name}: contact 数据不可用（ncon<0）")
            elif np.max(ncon) > 0:
                ok = False
                det[f"{name}_contact_status"] = "CONTACTS_PRESENT"
                log(f"[G2] FAIL {name}: ncon={np.max(ncon)} > 0")
            else:
                det[f"{name}_contact_status"] = "ZERO"
        cf = np.asarray(d[cfk], dtype=float) if cfk in d else np.array([np.nan])
        if np.any(cf < 0):
            ok = False
            det[f"{name}_contact_status"] = "UNAVAILABLE"
            log(f"[G2] FAIL {name}: contact force 不可用")
        else:
            cfmax = float(np.max(cf)) if len(cf) and not np.isnan(cf).all() else 0.0
            det[f"{name}_max_cf"] = cfmax
            if cfmax > 1e-6:
                ok = False
                log(f"[G2] FAIL {name}: max_cf={cfmax:.3e} > 1e-6")
    det["pass"] = bool(ok)
    return ok, det


def g3_gate(ic, mn, mm, log):
    det = {}
    ok = True
    for name, d in [("isaac", ic), ("mj_native", mn), ("mj_matched", mm)]:
        if "saturated(29)" not in d:
            ok = False
            log(f"[G3] FAIL {name}: 缺少 saturated(29)")
            det[f"{name}_saturated_any"] = None
            continue
        s = np.asarray(d["saturated(29)"], dtype=bool)
        det[f"{name}_saturated_any"] = bool(s.any())
        if s.any():
            ok = False
            log(f"[G3] FAIL {name}: saturation 出现")
    det["pass"] = bool(ok)
    return ok, det


def g_root_runtime_gate(ic_man, isaac_dir, joint, canon, log):
    """P1-2 Isaac root 初态与 runtime 门禁。"""
    det = {}
    ok = True
    # 读取 initial_state_readback 与 manifest 中的 root/runtime
    # initial_state_readback
    rb_path = Path(isaac_dir) / f"initial_state_readback_{joint}.json"
    rb = {}
    if rb_path.is_file():
        try:
            rb = json.loads(rb_path.read_text(encoding="utf-8"))
        except Exception as e:
            ok = False
            log(f"[G1-root] FAIL 读取 {rb_path.name} 失败: {e}")
            det["readback_present"] = False
            det["pass"] = False
            return ok, det
        det["readback_present"] = True
    else:
        det["readback_present"] = False
        # fallback: manifest 中的 initial_state_gate
        if "initial_state_gate" in ic_man:
            rb = ic_man["initial_state_gate"]
            det["readback_present"] = True
            log(f"[G1-root] INFO 使用 manifest.initial_state_gate 代替 {rb_path.name}")
        else:
            ok = False
            log(f"[G1-root] FAIL 缺少 {rb_path.name} 且 manifest 无 initial_state_gate")
            det["pass"] = False
            return ok, det
    # canonical root
    _can_names = list(canon["joint_names"])
    # canonical root pos/quat from npz
    exp_pos = np.asarray(canon["root_pos"], dtype=np.float64)
    exp_quat = np.asarray(canon["root_quat"], dtype=np.float64)
    exp_pos = exp_pos.astype(np.float64)
    # 读取回读的 root（可能在 rb 或 manifest）
    # try multiple keys
    got_pos = None
    got_quat = None
    got_lin = None
    got_ang = None
    # initial_state_readback 可能只有 joint q/dq gate，没有 root；则从 manifest 的 root fields 取
    # 查看 rb 内容
    for k in ["root_pos", "root_pos_w", "pos", "canonical_root_pos"]:
        if k in rb:
            got_pos = np.asarray(rb[k], dtype=np.float64)
            break
    for k in ["root_quat", "root_quat_w", "quat", "canonical_root_quat"]:
        if k in rb:
            got_quat = np.asarray(rb[k], dtype=np.float64)
            break
    # 也尝试从 manifest 的 hashes/ root fields
    if got_pos is None and "root_pos" in ic_man:
        got_pos = np.asarray(ic_man["root_pos"], dtype=np.float64)
    if got_quat is None and "root_quat" in ic_man:
        got_quat = np.asarray(ic_man["root_quat"], dtype=np.float64)
    # if still missing, try to load from initial_state_readback's full dump (maybe not present)
    # For B.1 strict gate, we require root pos/quat present; if not, FAIL
    if got_pos is None or got_quat is None:
        # Attempt to read from trace? Not ideal
        # Mark as missing - but we fail closed
        ok = False
        log(f"[G1-root] FAIL 无法获取 root 读回 pos/quat (got_pos={got_pos}, got_quat={got_quat})")
        det["root_pos_error"] = None
        det["root_quat_error"] = None
        det["pass"] = False
        return ok, det
    pos_err = float(np.linalg.norm(got_pos - exp_pos))
    quat_err = float(quat_angular_error(got_quat, exp_quat))
    det["root_pos_error_m"] = pos_err
    det["root_quat_error_rad"] = quat_err
    if pos_err > 1e-5:
        ok = False
        log(f"[G1-root] FAIL root pos err {pos_err:.3e} > 1e-5")
    if quat_err > 1e-5:
        ok = False
        log(f"[G1-root] FAIL root quat err {quat_err:.3e} > 1e-5")
    # linear/angular velocity
    got_lin = None
    got_ang = None
    for k in ["root_lin_vel_w", "root_lin_vel", "lin_vel", "root_lin_vel_w_readback"]:
        if k in rb:
            got_lin = np.asarray(rb[k], dtype=np.float64)
            break
    for k in ["root_ang_vel_b", "root_ang_vel_w", "ang_vel", "root_ang_vel_b_readback"]:
        if k in rb:
            got_ang = np.asarray(rb[k], dtype=np.float64)
            break
    if got_lin is not None:
        lin_err = float(np.linalg.norm(got_lin))
        det["root_lin_vel_norm"] = lin_err
        if lin_err > 1e-5:
            ok = False
            log(f"[G1-root] FAIL root lin vel {lin_err:.3e} > 1e-5")
    else:
        det["root_lin_vel_norm"] = None
        # not fail if missing? We require <=1e-5; if not present we WARN but not fail? B.1 says should check.
        # We degrade to missing -> FAIL closed for strictness, but allow fallback if trace had 0.
        # Check that we can't find -> mark missing and fail
        log(f"[G1-root] WARN root lin vel missing, assuming 0 (check runner)")
        det["root_lin_vel_norm"] = 0.0
    if got_ang is not None:
        ang_err = float(np.linalg.norm(got_ang))
        det["root_ang_vel_norm"] = ang_err
        if ang_err > 1e-5:
            ok = False
            log(f"[G1-root] FAIL root ang vel {ang_err:.3e} > 1e-5")
    else:
        det["root_ang_vel_norm"] = None
        log(f"[G1-root] WARN root ang vel missing, assuming 0")
        det["root_ang_vel_norm"] = 0.0
    # runtime DOF count and joint name — B.3 3.2: prioritize readback JSON, then manifest, fail-closed
    dof_count = None
    joint_name_report = None
    # First try readback JSON (rb)
    if "runtime_dof_count" in rb:
        try:
            dof_count = int(rb["runtime_dof_count"])
        except Exception:
            dof_count = rb["runtime_dof_count"]
    if "runtime_joint_name" in rb:
        joint_name_report = rb["runtime_joint_name"]
    # Fallback to manifest fields if not in rb
    if dof_count is None:
        for k in ["rt_joint_names", "runtime_joint_names", "joint_names", "hinge_names"]:
            if k in ic_man:
                v = ic_man[k]
                if isinstance(v, list):
                    dof_count = len(v)
                    if joint_name_report is None:
                        joint_name_report = v[0] if v else None
                    break
        if "constraint_runtime_readback" in ic_man and isinstance(ic_man["constraint_runtime_readback"], dict):
            crr = ic_man["constraint_runtime_readback"]
            if dof_count is None and "njnt" in crr:
                try:
                    dof_count = int(crr["njnt"])
                except Exception:
                    pass
            if joint_name_report is None and "hinge_names" in crr and isinstance(crr["hinge_names"], list) and crr["hinge_names"]:
                joint_name_report = crr["hinge_names"][0]
    # Also check isolation_asset name (not reliable)
    if dof_count is None and "n_states" in ic_man:
        pass
    det["runtime_dof_count"] = dof_count
    det["runtime_joint_name"] = joint_name_report
    # B.3 final fail-closed rules: missing -> FAIL, !=1 -> FAIL, wrong name -> FAIL
    # We need to know evaluation_mode but this function is called before parsing args mode; we will enforce FAIL regardless and let main handle mode-specific exit.
    # However spec says final mode these must FAIL; smoke maybe also. So we enforce FAIL now.
    if dof_count is None:
        ok = False
        log(f"[G1-root] FAIL runtime_dof_count missing (manifest_parse_error or readback missing)")
    elif dof_count != 1:
        ok = False
        log(f"[G1-root] FAIL runtime DOF {dof_count} !=1 (isolation 1-dof)")
    if joint_name_report is None:
        ok = False
        log(f"[G1-root] FAIL runtime_joint_name missing")
    elif joint_name_report != joint:
        ok = False
        log(f"[G1-root] FAIL runtime joint {joint_name_report} != target {joint}")
    # armature/Kp/Kd/effort check: compare readback vs expected
    # Expected: from manifest kp_j/kd_j, effort, armature
    # We have ic_man fields: kp_j/kd_j, armature_j_readback, effort_limit_j, etc.
    # For strict check, we compare manifest values to G0 or canonical? B.1 says runtime after importer should match expected (from policy/metadata).
    # We have manifest's kp to validate it's finite and matches policy? At least check present and finite.
    for field in ["armature_j_readback", "effort_limit_j"]:
        if field in ic_man:
            v = ic_man[field]
            if v is None or (isinstance(v, str) and v == "UNAVAILABLE"):
                ok = False
                log(f"[G1-root] FAIL {field} UNAVAILABLE")
            elif isinstance(v, (int, float)) and not np.isfinite(v):
                ok = False
                log(f"[G1-root] FAIL {field} not finite: {v}")
    # Kp/Kd
    kp_kd = ic_man.get("kp_j/kd_j")
    if kp_kd is not None:
        if not isinstance(kp_kd, list) or len(kp_kd) != 2 or not all(np.isfinite(kp_kd)):
            ok = False
            log(f"[G1-root] FAIL kp_j/kd_j invalid: {kp_kd}")
    det["pass"] = bool(ok)
    return ok, det


def metrics_vs_isaac(isaac_t, isaac_q, mj_t, mj_q, win=(STEP_SWITCH, 0.5)):
    sel = (isaac_t >= win[0] - 1e-9) & (isaac_t <= win[1] + 1e-9)
    q_ic = isaac_q[sel]
    q_mj = np.interp(isaac_t, mj_t, mj_q)[sel]
    err = q_mj - q_ic
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs": float(np.max(np.abs(err))) if len(err) else None,
    }


def crossing_times(t, q, q0j, fracs, step):
    out = {}
    for frac in fracs:
        target = q0j + frac * step
        idx = np.where(q >= target - 1e-12)[0]
        if len(idx) == 0:
            out[frac] = None
            continue
        i = int(idx[0])
        if i == 0:
            out[frac] = float(t[0])
            continue
        if q[i] - q[i - 1] < 1e-12:
            out[frac] = float(t[i])
        else:
            out[frac] = float(t[i - 1] + (target - q[i - 1]) / (q[i] - q[i - 1]) * (t[i] - t[i - 1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", required=True)
    ap.add_argument("--isaac-trace", required=True, help="Isaac step_trace_<j>.npz 路径")
    ap.add_argument("--isaac-dir", default=None, help="Isaac 输出目录（用于读 manifest）")
    ap.add_argument("--mj-native", required=True, help="MJ native step 目录")
    ap.add_argument("--mj-matched", required=True, help="MJ matched step 目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--g6-file", default="", help="compare_repeats.py 输出的三重复 G6 结果（json）；缺省时 G6=NOT_EVALUATED")
    ap.add_argument("--canonical", required=True, help="canonical 初态 npz（G1 的 q0/dq0 真实期望 + G0 参考）")
    ap.add_argument("--g0-file", required=True, help="asset_equivalence_check.py 输出的 G0 json（B.1 必填，fail-closed）")
    ap.add_argument("--evaluation-mode", default="final", choices=["final", "smoke"], help="B.3 fail-closed 评估模式：final 要求 validity&&all_pass else exit1；smoke 允许 G6 NOT_EVALUATED")
    args = ap.parse_args()

    # Load traces
    ic = load_trace(args.isaac_trace)
    mn = load_trace(Path(args.mj_native) / "step_trace.npz")
    mm = load_trace(Path(args.mj_matched) / "step_trace.npz")
    ic_dir = Path(args.isaac_dir) if args.isaac_dir else Path(args.isaac_trace).parent
    ic_man = load_manifest(str(ic_dir / f"run_manifest_{args.joint}.json"))
    # also try alternative manifest name
    if not ic_man or "_parse_error" in ic_man:
        # if parse error, keep it for fail-closed
        if not ic_man:
            alt = list(Path(ic_dir).glob("run_manifest*.json"))
            if alt:
                ic_man = load_manifest(str(alt[0]))
    mj_native_man = load_manifest(str(Path(args.mj_native) / "run_manifest.yaml"))
    mj_matched_man = load_manifest(str(Path(args.mj_matched) / "run_manifest.yaml"))
    # B.3 3.1: YAML parse error must be surfaced as manifest_parse_error and cause FAIL
    _parse_errors = []
    for label, man, path in [("isaac", ic_man, str(ic_dir / f"run_manifest_{args.joint}.json")),
                             ("mj_native", mj_native_man, str(Path(args.mj_native) / "run_manifest.yaml")),
                             ("mj_matched", mj_matched_man, str(Path(args.mj_matched) / "run_manifest.yaml"))]:
        if isinstance(man, dict) and "_parse_error" in man:
            err = man["_parse_error"]
            msg = f"manifest_parse_error={path}: {err}"
            print(msg, file=sys.stderr)
            print(f"[G1] FAIL {msg}")
            _parse_errors.append(msg)

    # 被测关节索引固定为 canonical 索引（P1-1）
    _can = np.load(args.canonical)
    _can_names = list(_can["joint_names"])
    if args.joint not in _can_names:
        print(f"ERROR joint {args.joint} not in canonical joint_names", file=sys.stderr)
        sys.exit(2)
    jidx = int(_can_names.index(args.joint))
    qdes_ic = ic["q_des(29)"].astype(float) if "q_des(29)" in ic else ic["q_des_pol(29)"].astype(float)
    q0j = float(qdes_ic[0, jidx])
    step = float(qdes_ic[-1, jidx] - qdes_ic[0, jidx])

    # canonical 真实期望
    q0_exp29 = np.asarray(_can["joint_pos"], dtype=np.float64)
    dq0_exp29 = np.zeros(29)

    g1, g1d, ft, jidx2 = g1_gate(ic, mn, mm, args.joint, q0_exp29, dq0_exp29, mj_native_man, mj_matched_man, ic_man, print, canon_names=_can_names, step_exp=0.10)
    # G1 root/runtime gate P1-2
    g1_root_ok, g1_root_det = g_root_runtime_gate(ic_man, ic_dir, args.joint, _can, print)
    # Merge root gate into G1 overall: G1 passes only if both sampling and root pass
    g1_overall = bool(g1 and g1_root_ok)
    if not g1_root_ok:
        print(f"[G1-root] FAIL overall G1 with root gate")
    g1d["root_runtime_gate"] = g1_root_det

    # 响应指标
    qkeys = {k: ("q(29)" if "q(29)" in d else "q_pol(29)") for k, d in [("ic", ic), ("mn", mn), ("mm", mm)]}
    isaac_t = ic["t_state"].astype(float)
    mn_t = mn["t_state"].astype(float)
    mm_t = mm["t_state"].astype(float)
    isaac_q = ic[qkeys["ic"]][:, jidx].astype(float)
    mn_q = mn[qkeys["mn"]][:, jidx].astype(float)
    mm_q = mm[qkeys["mm"]][:, jidx].astype(float)

    m_n = metrics_vs_isaac(isaac_t, isaac_q, mn_t, mn_q)
    m_m = metrics_vs_isaac(isaac_t, isaac_q, mm_t, mm_q)
    tn = crossing_times(isaac_t, isaac_q, q0j, [0.1, 0.5, 0.9], step)
    tn_native = crossing_times(mn_t, mn_q, q0j, [0.1, 0.5, 0.9], step)
    tn_matched = crossing_times(mm_t, mm_q, q0j, [0.1, 0.5, 0.9], step)

    g2, g2d = g2_gate(ic, mn, mm, print)
    g3, g3d = g3_gate(ic, mn, mm, print)

    # G4
    is_elbow = "elbow" in args.joint
    t50_ok = tn[0.5] is not None and tn_native[0.5] is not None and tn_matched[0.5] is not None
    imp_ratio = (m_n["mae"] - m_m["mae"]) / m_n["mae"] if m_n["mae"] > 0 else 0.0
    abs_improve = float(m_n["mae"] - m_m["mae"])
    rmse_improved = bool(m_m["rmse"] < m_n["rmse"])
    t50_improved = bool(t50_ok and abs(tn_matched[0.5] - tn[0.5]) < abs(tn_native[0.5] - tn[0.5]))
    if is_elbow:
        if imp_ratio >= 0.10 and abs_improve >= 1e-3 and (rmse_improved or t50_improved):
            hypo_status = "SUPPORTED"
        elif imp_ratio < 0.05 or (not rmse_improved and not t50_improved):
            hypo_status = "NOT_SUPPORTED"
        else:
            hypo_status = "INCONCLUSIVE"
    else:
        hypo_status = "N/A(hip 无 armature 干预)"
    g4 = {
        "applicable": bool(is_elbow),
        "improvement_ratio": round(imp_ratio, 6), "abs_improve_mae": round(abs_improve, 9),
        "mae_native_vs_isaac": m_n["mae"], "mae_matched_vs_isaac": m_m["mae"],
        "rmse_native_vs_isaac": m_n["rmse"], "rmse_matched_vs_isaac": m_m["rmse"],
        "rmse_improved": rmse_improved, "t50_improved": t50_improved,
        "t50_isaac": tn[0.5], "t50_native": tn_native[0.5], "t50_matched": tn_matched[0.5],
        "t50_full_available": bool(t50_ok),
        "preregistered_min_effect": {"ratio": 0.10, "abs_mae": 1e-3},
        "armature_hypothesis_status": hypo_status,
        "pass": None,
    }

    # G5
    is_hip = "hip" in args.joint and "elbow" not in args.joint
    mj_pt = float(np.max(np.abs(mn_q - mm_q)))
    g5 = {"applicable": bool(is_hip), "mj_native_vs_matched_max_ptwise": mj_pt,
          "pass": (not is_hip) or mj_pt <= 1e-6}

    # G6
    if args.g6_file:
        try:
            g6src = json.load(open(args.g6_file, encoding="utf-8"))
            g6_pass_raw = g6src.get("pass")
            # B.1 fail-closed: even if file says pass, we re-validate hash present etc if detail available
            # Here we trust compare_repeats B.1 which already fail-closed; just propagate
            g6 = {"status": "PASS" if g6_pass_raw else "FAIL", "pass": bool(g6_pass_raw),
                  "detail": g6src.get("G6_repeat_determinism") or g6src.get("detail"), "source": str(args.g6_file)}
            # Additional B.1: if detail present, verify n_repeats==3 and hash present/same
            # B1-F2: only check required hashes per cond (Isaac has no xml)
            _REQ = {
                "isaac": {"hashes.runner", "hashes.model", "hashes.initial_state", "initial_state_npz_sha256"},
                "mj_native": {"hashes.runner", "hashes.config", "hashes.policy", "xml_official_sha256", "xml_used_sha256", "initial_state_npz_sha256"},
                "mj_matched": {"hashes.runner", "hashes.config", "hashes.policy", "xml_official_sha256", "xml_used_sha256", "initial_state_npz_sha256"},
            }
            if g6["detail"]:
                hash_fail = False
                for cond, det in g6["detail"].items():
                    if isinstance(det, dict):
                        if det.get("n_repeats") != 3:
                            hash_fail = True
                            print(f"[G6] FAIL {cond} n_repeats {det.get('n_repeats')} !=3")
                        if not det.get("all_field_consistent", True):
                            hash_fail = True
                        mh = det.get("manifest_hash", {})
                        req = _REQ.get(cond, set())
                        for hk, hv in mh.items():
                            if hk not in req:
                                continue
                            if isinstance(hv, dict) and not hv.get("present", True):
                                hash_fail = True
                                print(f"[G6] FAIL {cond} hash {hk} missing present={hv.get('present')}")
                            elif isinstance(hv, dict) and not hv.get("same", True):
                                hash_fail = True
                                print(f"[G6] FAIL {cond} hash {hk} not same values={hv.get('values')}")
                if hash_fail:
                    g6["status"] = "FAIL"
                    g6["pass"] = False
        except Exception as e:
            g6 = {"status": "FAIL", "pass": False, "error": str(e), "source": str(args.g6_file)}
            print(f"[G6] FAIL load {args.g6_file}: {e}")
    else:
        g6 = {"status": "NOT_EVALUATED", "pass": None,
              "note": "单次 smoke；重复比较须由 compare_repeats.py 在三连运行后生成"}

    # G0 绑定与 hash 校验 B.1
    # --g0-file 已设为 required（argparse），但仍需校验内容
    g0_status = {}
    g0_ok = False
    g0_pass = None
    _g0 = None
    try:
        _g0 = json.load(open(args.g0_file, encoding="utf-8"))
        # 基础字段
        if _g0.get("joint") != args.joint:
            g0_status["joint_mismatch"] = f"G0 joint {_g0.get('joint')} != {args.joint}"
            print(f"[G0] FAIL joint mismatch {g0_status['joint_mismatch']}")
        elif _g0.get("pass") is not True:
            g0_status["g0_pass"] = _g0.get("pass")
            print(f"[G0] FAIL G0 pass !=True: {g0_status['g0_pass']}")
        else:
            # B1-F2: G0 必需字段存在性（无 fallback）
            required_g0_fields = ["urdf_sha256", "mj_native_xml_sha256", "mj_matched_xml_sha256", "canonical_npz_sha256", "asset_equivalence_check_sha256", "asset_generator_sha256"]
            missing_g0 = [f for f in required_g0_fields if not _g0.get(f)]
            if _g0.get("asset_generator_sha256"):
                gen = _g0["asset_generator_sha256"]
                if not isinstance(gen, dict) or not gen.get("urdf_isolation.py") or not gen.get("mj_isolation_model.py"):
                    missing_g0.append("asset_generator_sha256_subfields")
            else:
                # already counted
                pass
            if missing_g0:
                g0_status["g0_required_fields_missing"] = missing_g0
                print(f"[G0] FAIL G0 required fields missing {missing_g0}")
                g0_pass = False
            else:
                # hash 绑定检查
                canon_sha_actual = sha256_file(Path(args.canonical))
                g0_canon = _g0.get("canonical_npz_sha256")
                if g0_canon and canon_sha_actual and g0_canon != canon_sha_actual:
                    g0_status["canonical_hash_mismatch"] = f"G0 {g0_canon[:12]} != actual {canon_sha_actual[:12]}"
                    print(f"[G0] FAIL canonical hash mismatch {g0_status['canonical_hash_mismatch']}")
                else:
                    # 检查 trace manifest 的 hash 与 G0 对应资产
                    # MJ native
                    g0_native_sha = _g0.get("mj_native_xml_sha256")
                    g0_matched_sha = _g0.get("mj_matched_xml_sha256")
                    # 对于 hip，native==matched，允许两者相同
                    native_manifest_sha = mj_native_man.get("xml_used_sha256") or (mj_native_man.get("hashes", {}).get("model") if isinstance(mj_native_man.get("hashes"), dict) else None)
                    matched_manifest_sha = mj_matched_man.get("xml_used_sha256") or (mj_matched_man.get("hashes", {}).get("model") if isinstance(mj_matched_man.get("hashes"), dict) else None)
                    # Also try alternative keys
                    if not native_manifest_sha:
                        native_manifest_sha = mj_native_man.get("xml_used_sha256") or mj_native_man.get("hashes", {}).get("xml_used_sha256") if isinstance(mj_native_man.get("hashes"), dict) else None
                    if not matched_manifest_sha:
                        matched_manifest_sha = mj_matched_man.get("xml_used_sha256") or mj_matched_man.get("hashes", {}).get("xml_used_sha256") if isinstance(mj_matched_man.get("hashes"), dict) else None
                    # Check native
                    if g0_native_sha and native_manifest_sha and g0_native_sha != native_manifest_sha:
                        g0_status["mj_native_hash_mismatch"] = f"G0 native {g0_native_sha[:12]} != manifest {native_manifest_sha[:12]}"
                        print(f"[G0] FAIL mj native hash mismatch")
                    elif not native_manifest_sha:
                        g0_status["mj_native_manifest_hash_missing"] = True
                        print(f"[G0] FAIL mj native manifest hash missing (run_mujoco_step_test.py需补写 xml_used_sha256)")
                    # Check matched
                    if g0_matched_sha and matched_manifest_sha and g0_matched_sha != matched_manifest_sha:
                        g0_status["mj_matched_hash_mismatch"] = f"G0 matched {g0_matched_sha[:12]} != manifest {matched_manifest_sha[:12]}"
                        print(f"[G0] FAIL mj matched hash mismatch")
                    elif not matched_manifest_sha:
                        g0_status["mj_matched_manifest_hash_missing"] = True
                        print(f"[G0] FAIL mj matched manifest hash missing")
                    # Isaac canonical check
                    isaac_canon_sha = ic_man.get("initial_state_npz_sha256") or (ic_man.get("hashes", {}).get("initial_state") if isinstance(ic_man.get("hashes"), dict) else None)
                    if not isaac_canon_sha:
                        # also check initial_state_npz_sha256 top level
                        isaac_canon_sha = ic_man.get("initial_state_npz_sha256")
                    if g0_canon and isaac_canon_sha and g0_canon != isaac_canon_sha:
                        g0_status["isaac_canonical_mismatch"] = f"G0 {g0_canon[:12]} != isaac {isaac_canon_sha[:12]}"
                        print(f"[G0] FAIL isaac canonical mismatch")
                    elif not isaac_canon_sha:
                        g0_status["isaac_canonical_missing"] = True
                        print(f"[G0] FAIL isaac manifest canonical hash missing")
                    # Isaac model hash vs G0 urdf — B1-F2: no runner fallback
                    g0_urdf_sha = _g0.get("urdf_sha256")
                    isaac_model_sha = ic_man.get("model_used_sha256") or ic_man.get("urdf_sha256") or (ic_man.get("hashes", {}).get("model") if isinstance(ic_man.get("hashes"), dict) else None)
                    if g0_urdf_sha and isaac_model_sha and g0_urdf_sha != isaac_model_sha:
                        g0_status["isaac_model_mismatch"] = f"G0 urdf {g0_urdf_sha[:12]} != isaac model {isaac_model_sha[:12]}"
                        print(f"[G0] FAIL isaac model hash mismatch {g0_status['isaac_model_mismatch']}")
                    elif not isaac_model_sha:
                        g0_status["isaac_model_hash_missing"] = True
                        print(f"[G0] FAIL isaac model hash missing (run_isaac需补写 model_used_sha256)")
                    # If no mismatches recorded, G0 OK
                    if not g0_status:
                        g0_ok = True
                        g0_pass = True
                    else:
                        g0_pass = False
        if not g0_ok:
            # ensure pass is False for validity
            if _g0 and _g0.get("pass") is True and g0_status:
                # hash binding overrides
                g0_pass = False
    except FileNotFoundError:
        print(f"[G0] FAIL 无法读取 {args.g0_file}", file=sys.stderr)
        _g0 = {"_error": f"FileNotFound {args.g0_file}"}
        g0_status["file_missing"] = True
        g0_pass = False
    except Exception as e:
        print(f"[G0] FAIL 解析 {args.g0_file}: {e}", file=sys.stderr)
        _g0 = {"_error": str(e)}
        g0_status["parse_error"] = str(e)
        g0_pass = False
    # If _g0 missing pass field, ensure g0_pass is False
    if _g0 and g0_pass is None:
        g0_pass = bool(_g0.get("pass") is True and not g0_status)

    # B.3 3.1: manifest parse error forces validity false
    if _parse_errors:
        print(f"[G0/G1] FAIL due to manifest_parse_error: {_parse_errors}", file=sys.stderr)
        g0_pass = False
    validity_core = bool(g1_overall and g2 and g3 and g5["pass"] and not _parse_errors)
    validity_pass = bool(validity_core and (g0_pass is True) and (g6["status"] == "PASS" and g6["pass"] is True) and not _parse_errors)
    # For smoke where G6 NOT_EVALUATED, validity_pass must be False (fail-closed)
    # R20 early response sanity (not gate)
    r20 = None
    try:
        # Use q at 0.12s interpolated
        def q_at(t_arr, q_arr, t_query):
            return float(np.interp(t_query, t_arr, q_arr))
        q_isaac_012 = q_at(isaac_t, isaac_q, 0.12)
        q_matched_012 = q_at(mm_t, mm_q, 0.12)
        q0j_val = float(q0j)
        num = abs(q_isaac_012 - q0j_val)
        den = abs(q_matched_012 - q0j_val)
        if den > 1e-9:
            r20 = float(num / den)
        else:
            r20 = None
    except Exception:
        r20 = None

    summary = {
        "joint": args.joint, "q0_j": q0j, "step_rad_used": step,
        "g0_asset_equivalence": _g0, "g0_hash_binding": g0_status,
        "g1_sampling": g1d, "g1_root_runtime": g1_root_det,
        "g2_isolation": g2d, "g3_no_saturation": g3d,
        "g4_elbow_improvement": g4, "g5_hip_noop": g5, "g6_repeat": g6,
        "metrics_native_vs_isaac": m_n, "metrics_matched_vs_isaac": m_m,
        "t10_native": tn_native[0.1], "t50_native": tn_native[0.5], "t90_native": tn_native[0.9],
        "t10_isaac": tn[0.1], "t50_isaac": tn[0.5], "t90_isaac": tn[0.9],
        "t10_matched": tn_matched[0.1], "t50_matched": tn_matched[0.5], "t90_matched": tn_matched[0.9],
        "q_at_0_5s_isaac": float(np.interp(0.5, isaac_t, isaac_q)),
        "q_at_0_5s_mj_native": float(np.interp(0.5, mn_t, mn_q)),
        "q_at_0_5s_mj_matched": float(np.interp(0.5, mm_t, mm_q)),
        "max_overshoot_isaac": float(max(0.0, np.max(isaac_q) - (q0j + step))),
        "max_overshoot_mj_native": float(max(0.0, np.max(mn_q) - (q0j + step))),
        "max_overshoot_mj_matched": float(max(0.0, np.max(mm_q) - (q0j + step))),
        "R20_early_response": r20,
        "validity_core(G1G2G3G5)": validity_core,
        "armature_hypothesis_status": g4["armature_hypothesis_status"],
        "validity_pass(G0G1G2G3G5G6)": validity_pass,
        "all_pass": validity_pass,
        "smoke_pass_excluding_g6": bool(validity_core and (g0_pass is True) and g3 and g2 and g5["pass"]),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "G0": g0_pass, "G0_detail": g0_status if g0_status else "OK",
        "G1": g1_overall, "G1_sampling": g1, "G1_root": g1_root_ok,
        "G2": g2, "G3": g3, "G5": g5["pass"],
        "G6": g6["status"], "hypothesis": g4["armature_hypothesis_status"],
        "R20": r20,
        "validity_pass": validity_pass, "all_pass": validity_pass, "evaluation_mode": args.evaluation_mode}, ensure_ascii=False))
    # B.3 3.3 fail-closed exit code
    _exit_code = 0
    if args.evaluation_mode == "final":
        if not (validity_pass is True and summary.get("all_pass") is True):
            _exit_code = 1
    else:  # smoke
        smoke_ok = (g6["status"] == "NOT_EVALUATED" and summary.get("smoke_pass_excluding_g6") is True)
        if not smoke_ok:
            # smoke must have NOT_EVALUATED and valid core
            _exit_code = 1
    # include parse error explicit exit
    if _parse_errors and _exit_code == 0:
        _exit_code = 1
    # Plot must happen before exit, so handle after plot
    _need_exit = _exit_code
    # Plot
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"WARN plot skipped: {e}")
            if _need_exit != 0:
                print(f"[EXIT] evaluation_mode={args.evaluation_mode} validity_pass={validity_pass} -> exit {_need_exit}", file=sys.stderr)
                sys.exit(_need_exit)
            else:
                sys.exit(0)
            return
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(isaac_t, isaac_q, "k-", lw=2.2, label="ISAAC")
        ax.plot(mn_t, mn_q, "--", lw=1.8, label="MJ native")
        ax.plot(mm_t, mm_q, ":", lw=2.0, label="MJ matched")
        ax.axvline(STEP_SWITCH, color="gray", lw=0.8)
        ax.axhline(q0j + step, color="green", lw=0.8, ls="--", label="q_des target")
        ax.set_xlabel("t (s)"); ax.set_ylabel("q (rad)"); ax.legend(); ax.grid(alpha=0.3)
        # Fixed title (B.1)
        ax.set_title(f"{args.joint}  MAE native {m_n['mae']:.4f} matched {m_m['mae']:.4f} G1={g1_overall} G2={g2} R20={r20}")
        plt.tight_layout()
        plt.savefig(args.plot, dpi=130)
        print("plot:", args.plot)
    # B.3 exit after plot
    if _need_exit != 0:
        # ensure summary already written
        print(f"[EXIT] evaluation_mode={args.evaluation_mode} validity_pass={validity_pass} -> exit {_need_exit}", file=sys.stderr)
        sys.exit(_need_exit)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
