# B3.1 最终状态 — R4F B.1 + Lab 10s + N5-R2 + B3.1 相机/索引/验证修正 (2026-08-31)

B3.1 是最终状态。

> 本包为 **B3.1 最终状态**（基于 `R4F_B1_Lab10s_N5R2_B3_20260831.zip` SHA `317dc16af685df4c805bb7f1d15487cf17cef381ee8f7b0924ab790382a14a09` 的直接修正，仅重渲染视频与修正文档/验证，无重跑 N5-R2/G0/Lab/Gym/SONIC 数值实验）
> 解压后根目录即为本 README 所在目录；唯一权威哈希清单为根 `SHA256.txt`（64 位完整 SHA，LF 换行，相对路径，覆盖除自身外全部文件）

## 最终状态摘要

- **N5-R2 已完成 18 条原始证据归档**：`04_isolation_N5R2/elbow/isaac_r1..r3 + mj_native_r1..r3 + mj_matched_r1..r3` 与 `hip` 同构，各含 `step_trace*.npz + run_manifest.* + stdout.log + initial_state_readback_*.json`，G6 三重复逐点一致。
- **elbow/hip final gate 均为 `validity=true / all_pass=true`**：从包内 18 条原始数据重新生成 `04_isolation_N5R2/elbow_G6.json` 与 `hip_G6.json`（`pass==true b1_strict==true n_repeats==3 all_field_consistent==true all_hash_present_and_same==true`），再经 `code_B1_patch/scripts/n5_r_summarize.py --evaluation-mode final` 重算 `elbow_gates.json` `hip_gates.json`，`g0_asset_equivalence.pass==true g0_hash_binding=={} g1_sampling.pass==true g1_root_runtime.pass==true runtime_dof_count==1 runtime_joint_name==target g2/g3/g5 pass==true g6_repeat.status==PASS validity_pass==true all_pass==true`。`elbow MAE 0.006752→0.001704 改善74.8% t50 0.17743→0.16329 (Isaac 0.16382) hip 0.0` 可复现。
- **Lab 视频已通过全身可见门禁（B31.3 6.5m/4.0m 相机）**：`02_lab_dance9/isaac_10s` `eye 6.5m root+[0.35,-6.5,0.20] tgt 0.00` 与 `mj_10s` `distance 4.0m lookat 0.30` 首/中/尾帧均可见头、双手、双脚且四边≥5%边距、主体高度 52%/48%（40-80% 内且尺度对齐）、无黑帧，`video_visibility.json pass==true`，`media_manifest.json` 记录 `runner/video/frame` 完整 64 位哈希及 `candidate_max_diff 0.0`，`sidebyside_10s_labeled.mp4` 标签正确且时间轴一致，`02_lab_dance9/camera_smoke_pass` 6 张烟雾帧已人工复检（随包）。
- **SONIC 四格只引用真实证据**：`06_sonic/sonic_II_IM_MM_MI_ledger.csv` 为唯一当前台账（`II:C IM:D MM:C MI:C`），旧 `sonic_evidence_ledger.csv` 已移入 `historical/` 并首行标注 `DEPRECATED`；所有 `action_space_fix_before_after.npz` `zero_shot_IM.csv` `docs/assets/action_space_fix.mp4` `docs/assets/isaac_backtransfer_r2_11000.mp4` `0.5 rad overshoot` `2.1s fall` 等无来源路径/数字已删除或标 `unavailable`；ARCH 表已按 `SONIC_Training_Report.md` 修正为 `930 proprio (10×29+... ) +64 token →994 →29 action`。
- **当前根 SHA256 是唯一权威清单**：`SHA256.txt` 212 payload 项完整 64 位（含 `02_lab_dance9/camera_smoke_pass` 6 张烟雾及 `09_report/assets` 6 张报告内嵌图片），LF 换行，相对路径，无重复，无缺失，已通过新鲜解压 `sha256sum -c` 212/212 PASS 及 `final_package_validate.py` 全量只读校验。
- **延期项为 N6、20s、delay/gain、Gym 50Hz 等**：按 `2026-08-31-R4F-B3审查与B3.1最终收口清单` §2 `Won't` 明确延后，不影响本周 Must 闭环。

## 目录结构（相对）

- `02_lab_dance9/` Lab 5s/10s 双侧基线（含冻结数值 trace 与重渲染视频，`isaac_5s/mj_5s` 为同引擎 10s 前半前缀）
- `04_G0/` 资产等价门禁 G0
- `04_isolation_N5R2/` N5-R2 18 条 + G6 + final gate + plots
- `04_isolation_smoke/` smoke 单次（G6 NOT_EVALUATED 预期）
- `05_gym/` Gym 参考
- `06_sonic/` SONIC 四格（唯一当前 ledger，C/D）
- `09_report/` 主报告、索引、日志、验证；`assets/` 为报告自包含图片，避免预览器拦截 `../` 跨目录资源
- `code_B1_patch/` 门禁脚本（含 B.3/B3.1 补丁）+ evidence
- `historical/` 旧 README/报告/ledger 归档
- `SHA256.txt` 根哈希清单（LF）

## 关键命令（可复现）

```bash
# G6
python code_B1_patch/scripts/compare_repeats.py --joint right_elbow_pitch_joint --isaac-dirs 04_isolation_N5R2/elbow/isaac_r1,... --mj-native-dirs ... --mj-matched-dirs ... --out 04_isolation_N5R2/elbow_G6.json
# final gate
python code_B1_patch/scripts/n5_r_summarize.py --evaluation-mode final --joint right_elbow_pitch_joint --canonical 04_G0/canonical_initial_state_clean.npz --g0-file 04_G0/asset_equivalence_right_elbow_pitch_joint.json --isaac-trace 04_isolation_N5R2/elbow/isaac_r1/step_trace... --isaac-dir ... --mj-native ... --mj-matched ... --g6-file ... --out 04_isolation_N5R2/elbow_gates.json --plot 04_isolation_N5R2/plots/elbow.png
# 仅重渲染视频（数值冻结，B31.3 6.5m/4.0m）
# Isaac: run_isaac_onnx.py set_world_poses_from_view eye root+[0.35,-6.5,0.20] tgt root+[0,0,0.00] distance 6.5m
# MuJoCo: run_mujoco_onnx.py MjvCamera distance 4.0 elevation -10 lookat 0.30 (独立相机单次 update_scene)
```

## 哈希与打包

最终 ZIP：`R4F_B1_Lab10s_N5R2_B31_20260831.zip`（UTF-8 文件名，/ 归档路径，`ZIP entries 213` `SHA256.txt 212 payload`，Windows 7-Zip / WSL `unzip -t` / macOS `bsdtar` 均通过，解压后中文报告与 `09_report/assets` 图片正常，`sha256sum -c SHA256.txt` 212/212 PASS，`final_package_validate.py --work <解压目录>` 退出 0）

> ZIP 伴随文件 `R4F_B1_Lab10s_N5R2_B31_20260831.zip.sha256` 位于 ZIP 同目录（包外），不包含在 ZIP 内；包内 `SHA256.txt` 仅覆盖 payload。

> 旧 `README_B2.md`/`README.md (B.3)` 已移入 `historical/`；本 README 为 B3.1 唯一当前入口。
