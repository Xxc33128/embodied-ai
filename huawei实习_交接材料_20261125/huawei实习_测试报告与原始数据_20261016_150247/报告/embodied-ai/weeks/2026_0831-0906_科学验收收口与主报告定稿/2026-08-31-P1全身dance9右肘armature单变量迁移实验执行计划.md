# P1：Humanoid Lab 全身 dance_9 右肘 armature 单变量迁移实验执行计划

> 日期：2026-08-31  
> 平台：Windows 原生 Isaac Sim / Isaac Lab；WSL2 MuJoCo 3.2.7  
> 实验编号：`P1-FULLBODY-ELBOW-TRANSFER`  
> 对象：Humanoid Lab，`dance_9`，29 DOF  
> 用途：交给 Windows/WSL agent 逐步执行并返回完整证据包

## 0. 目标和最终决策

核心问题：

> MuJoCo 完整模型只将 `right_elbow_pitch_joint` 的 armature 从 `0.0685` 改为 Isaac 运行时的 `0.01` 后，完整 `dance_9` 的右肘轨迹是否更接近 Isaac，同时保持 root 和全身指标不恶化？

结果必须归入以下一种：

| 决策 | 含义 | 后续 |
|---|---|---|
| `SUPPORTED_TRANSFER_TO_FULLBODY` | 右肘改善且系统指标通过 | 进入 `ARM_ONLY`，再决定 `WAIST_ONLY` / N6 |
| `LOCAL_EFFECT_WITH_SYSTEM_TRADEOFF` | 右肘改善但系统指标恶化 | 分析 action/q_des 和身体耦合 |
| `ISOLATED_EFFECT_NOT_TRANSFERRED` | 右肘未达到最小改善 | 转查 PD 时序、frictionloss、dt、solver |
| `INVALID_RERUN` | 资产、哈希、初态或重复门禁失败 | 只修实验链 |

## 1. 预注册指标和阈值

固定主窗口为 `0–2s`。读取 matched 结果后不得修改。

```text
primary = right_elbow_pitch_joint q MAE vs Isaac, 0–2s
absolute_improvement = MAE_native - MAE_matched
relative_improvement = (MAE_native - MAE_matched) / MAE_native
```

右肘效应通过条件：

```text
relative_improvement >= 10%
AND absolute_improvement >= 1e-3 rad
AND 三次重复方向一致
```

系统保护条件：

```text
matched 不更早触发 failure predicate
root height RMSE 增量 <= max(native 的 10%, 0.002 m)
root roll RMSE 增量   <= max(native 的 10%, 0.005 rad)
root pitch RMSE 增量  <= max(native 的 10%, 0.005 rad)
全 29 关节 q RMSE 恶化不超过 10%
```

增加一个辅助高激励窗口：

1. 只读取冻结 Isaac reference；
2. 在右肘 `q_des` 上滑动 1 秒窗口；
3. 选择标准差最大的窗口；
4. 在加载 matched trace 前保存 `active_window.json`；
5. 后续不再改窗口。

以上阈值只用于本轮探索性工程决策。

## 2. 实验矩阵

| 条件 | 引擎 | 完整模型 | 右肘 armature | 次数 |
|---|---|---|---:|---:|
| `ISAAC-REF` | Isaac/PhysX | 已冻结 URDF | runtime ≈ `0.01` | 复用 1×10s |
| `MJ-NATIVE-R1/R2/R3` | MuJoCo | 原始 XML | `0.0685` | 3×10s |
| `MJ-MATCHED-R1/R2/R3` | MuJoCo | matched XML | `0.01` | 3×10s |

正式运行前先跑 native/matched 各 2 秒 smoke。Isaac 默认复用，触发第 12 节条件时再启动 Kit。

## 3. 冻结输入

### 3.1 期望哈希

| 输入 | SHA256 |
|---|---|
| policy | `30e1fdede1bc6485e45c04e4f60aacd8ac5611e2ed1e67b1e3bc5b757a7bd659` |
| motion | `bee3fe34a7ddefab4e0694a898a8b64e17cf89ac4850f1632843689539ec23fb` |
| native MuJoCo XML | `69e975bad858be6be4cedd0c91a6d98882a48dc419a5aab0c19b94c99e3bba22` |
| native YAML | `0fe5cd31b5b2db90a1b3dd03cc4f683961c8db7fd0d7be0bbe6d9026094b6c39` |
| 全身 canonical | `7fdf68af39c0d4be7bcf8cd90f5ddb7718b93db240d657e78b97ad0dae6745cf` |
| Isaac 10s trace | `be64b024e206b0221d17a6135e3d2e5a75ab583db98e827d98c6c965dcb2e79d` |
| MuJoCo 10s trace | `56eb572bbe4e0053a948711a6b83179b0a519e3827a3ad567eb2d2e543b38a82` |

### 3.2 canonical 硬规则

完整 `dance_9` 只允许使用 `7fdf...` canonical。

`04_G0/canonical_initial_state_clean.npz` 和 `code_B1_patch/evidence/canonical_initial_state_clean.npz` 的 SHA 为 `19bb...`，它们属于 1-DOF 隔离实验，本实验禁止使用。

找不到 `7fdf...` 时停止并报告 `BLOCKED_CANONICAL_MISSING`。

### 3.3 控制变量

| 项目 | 固定值 |
|---|---|
| policy / motion | 上表冻结文件 |
| mode / start frame | `aligned` / 0 |
| history | `repeat-first` |
| seed | 42 |
| duration | 10s |
| control dt | 0.02s |
| MuJoCo physics dt / ppc | 0.002s / 10 |
| Isaac physics dt / decimation | 0.005s / 4 |
| delay / gain | 0ms / 1.0 |
| Kp/Kd | ONNX metadata |
| initial root/q/dq | 同一 `7fdf...` canonical |
| solver/friction/contact | 原始值 |

## 4. 输出目录

PowerShell：

```powershell
$P1Root = "E:/sim2sim-week-2026-08-26/10_p1_fullbody_elbow"
$Dirs = @("00_inputs","01_asset","02_smoke/mj_native","02_smoke/mj_matched","03_runs/mj_native_r1","03_runs/mj_native_r2","03_runs/mj_native_r3","03_runs/mj_matched_r1","03_runs/mj_matched_r2","03_runs/mj_matched_r3","04_analysis/plots","05_media","06_checks","07_feedback")
$Dirs | ForEach-Object { New-Item -ItemType Directory -Force -Path "$P1Root/$_" }
```

最终结构至少包含：

```text
10_p1_fullbody_elbow/
├── 00_inputs/
│   ├── input_hashes.csv
│   ├── environment.txt
│   ├── isaac_reference/
│   ├── native_reference/
│   ├── canonical_initial_state.npz
│   └── frozen_run_mujoco_onnx.py
├── 01_asset/
│   ├── l7_29dof_neck_fixed_native.xml
│   ├── l7_29dof_neck_fixed_elbow_matched.xml
│   ├── mimic_dance_9_native.yaml
│   ├── mimic_dance_9_elbow_matched.yaml
│   ├── make_fullbody_elbow_matched.py
│   ├── audit_fullbody_elbow_assets.py
│   ├── fullbody_elbow_asset_audit.json
│   └── xml_target_diff.patch
├── 02_smoke/
├── 03_runs/
├── 04_analysis/
│   ├── active_window.json
│   ├── p1_metrics_per_run.csv
│   ├── p1_metrics_summary.csv
│   ├── p1_repeat_check.json
│   ├── p1_final_gate.json
│   ├── P1_result_summary.md
│   └── plots/
├── 05_media/isaac_native_matched_10s_labeled.mp4
├── 06_checks/
├── README_P1.md
└── SHA256.txt
```

## 5. Step P1-0：定位并冻结输入

### 5.1 定位 B3.1

```powershell
Get-ChildItem "E:/sim2sim-week-2026-08-26" -Recurse -Filter "policy_trace.npz" | Where-Object { $_.FullName -like "*02_lab_dance9*isaac_10s*policy_trace.npz" } | Select-Object FullName
```

设置并检查：

```powershell
$B31 = "填写已验收 B3.1 解压根目录"
$LabRepo = "E:/humanoid-lab"
$P1Root = "E:/sim2sim-week-2026-08-26/10_p1_fullbody_elbow"
Test-Path "$B31/02_lab_dance9/isaac_10s/policy_trace.npz"
Test-Path "$B31/02_lab_dance9/mj_10s/policy_trace.npz"
Test-Path "$LabRepo/scripts/experiments/run_mujoco_onnx.py"
```

任一结果为 `False` 时停止。

### 5.2 定位全身 canonical

```powershell
Get-ChildItem "E:/sim2sim-week-2026-08-26" -Recurse -Filter "canonical_initial_state.npz" | ForEach-Object { $h=(Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(); if($h -eq "7fdf68af39c0d4be7bcf8cd90f5ddb7718b93db240d657e78b97ad0dae6745cf"){$_.FullName} }
```

将唯一结果设置为 `$Canonical`。多个同哈希副本可以任选并记录来源。

### 5.3 复制证据

```powershell
Copy-Item "$B31/02_lab_dance9/isaac_10s" "$P1Root/00_inputs/isaac_reference" -Recurse -Force
Copy-Item "$B31/02_lab_dance9/mj_10s" "$P1Root/00_inputs/native_reference" -Recurse -Force
Copy-Item $Canonical "$P1Root/00_inputs/canonical_initial_state.npz" -Force
Copy-Item "$LabRepo/scripts/experiments/run_mujoco_onnx.py" "$P1Root/00_inputs/frozen_run_mujoco_onnx.py" -Force
Copy-Item "$LabRepo/deploy/l7_29dof_neck_fixed/l7_29dof_neck_fixed.xml" "$P1Root/01_asset/l7_29dof_neck_fixed_native.xml" -Force
Copy-Item "$LabRepo/deploy/src/era_rl_controller/configs/mimic_dance_9.yaml" "$P1Root/01_asset/mimic_dance_9_native.yaml" -Force
```

### 5.4 环境与哈希

保存以下输出：

```powershell
git -C $LabRepo rev-parse HEAD
git -C $LabRepo status --porcelain
& "E:/isaaclab_env51/Scripts/python.exe" --version
wsl -d Ubuntu-22.04 -- bash -lc 'uname -a; python3 --version; nvidia-smi'
```

对第 3.1 节全部输入计算 SHA256，写 `00_inputs/input_hashes.csv`。任一关键哈希不一致时停止。

## 6. Step P1-1：生成 matched XML 和 YAML

### 6.1 唯一允许的模型修改

原始右肘 joint 通过 `class="elbow_pitch"` 继承 `armature="0.0685"`。matched 在目标 joint 标签增加显式覆盖：

```xml
<joint name="right_elbow_pitch_joint"
       pos="0 0 0"
       axis="0 1 0"
       range="-2.36 0.7"
       class="elbow_pitch"
       armature="0.01"/>
```

严禁修改：

- `<default class="elbow_pitch">`，修改它会影响左右肘；
- 左肘；
- Kp/Kd、motor、gear、ctrlrange、damping、frictionloss；
- body mass/inertia/COM、geom、contact、solver、timestep；
- 官方原始 XML。

### 6.2 生成脚本要求

创建 `01_asset/make_fullbody_elbow_matched.py`：

1. UTF-8 读取 native XML；
2. 精确定位唯一 `right_elbow_pitch_joint` joint 标签；
3. 断言目标标签没有显式 armature；
4. 只插入 `armature="0.01"`；
5. 其他文本保持不变；
6. 输出 matched XML 和 unified diff；
7. diff 只允许包含目标 joint 行。

### 6.3 matched YAML

复制 native YAML，只把：

```yaml
robot:
  sim2sim:
    xml_path: "l7_29dof_neck_fixed/p1_generated/l7_29dof_neck_fixed_elbow_matched.xml"
```

保存为 `01_asset/mimic_dance_9_elbow_matched.yaml`。

语义审计确认唯一配置差异为 `robot.sim2sim.xml_path`。

### 6.4 部署副本

```powershell
$DeployGenerated = "E:/humanoid-lab/deploy/l7_29dof_neck_fixed/p1_generated"
New-Item -ItemType Directory -Force -Path $DeployGenerated
Copy-Item "$P1Root/01_asset/l7_29dof_neck_fixed_elbow_matched.xml" "$DeployGenerated/l7_29dof_neck_fixed_elbow_matched.xml" -Force
```

证据副本和部署副本 SHA 必须相同。

## 7. Step P1-2：单变量资产审计

创建 `01_asset/audit_fullbody_elbow_assets.py`，使用 MuJoCo 3.2.7 编译两个 XML。

必须检查：

- joint/body/geom/actuator 名称与顺序；
- `nq/nv/nu/nbody/njnt/ngeom`；
- native 右肘 armature = `0.0685`；
- matched 右肘 armature = `0.01`；
- 左肘和其他 28 关节 armature 不变；
- body mass/inertia/pose；
- joint pose/axis/range/type；
- damping/frictionloss；
- actuator gear/ctrlrange/forcerange/gain/dyn/bias；
- timestep/solver/iterations/tolerance；
- 编译后差异字段数量精确为 1。

输出 `fullbody_elbow_asset_audit.json`：

```json
{
  "pass": true,
  "target_joint": "right_elbow_pitch_joint",
  "native_armature": 0.0685,
  "matched_armature": 0.01,
  "changed_model_fields": ["dof_armature[right_elbow_pitch_joint]"],
  "other_armature_max_diff": 0.0,
  "mass_max_diff": 0.0,
  "inertia_max_diff": 0.0,
  "actuator_param_max_diff": 0.0,
  "native_xml_sha256": "...",
  "matched_xml_sha256": "..."
}
```

负测试：

1. 额外修改左肘，审计必须 FAIL；
2. 额外修改 timestep，审计必须 FAIL；
3. 删除右肘覆盖，审计必须 FAIL。

三个结果写入 `06_checks/asset_audit_negative_tests.txt`。审计未通过时禁止 smoke。

## 8. Step P1-3：冻结 runner 与环境

WSL：

```bash
cd /mnt/e/humanoid-lab
if [ -x "$HOME/.venv_lab_sim/bin/python" ]; then
  lab_python="$HOME/.venv_lab_sim/bin/python"
else
  lab_python="$HOME/projects/GR00T-WholeBodyControl/.venv_sim/bin/python"
fi
test -x "$lab_python"
"$lab_python" -c 'import numpy,yaml,onnxruntime,mujoco; print(mujoco.__version__)'
export MUJOCO_GL=egl
p1_root=/mnt/e/sim2sim-week-2026-08-26/10_p1_fullbody_elbow
runner="$p1_root/00_inputs/frozen_run_mujoco_onnx.py"
"$lab_python" -m py_compile "$runner"
"$lab_python" -m py_compile "$p1_root/01_asset/make_fullbody_elbow_matched.py"
"$lab_python" -m py_compile "$p1_root/01_asset/audit_fullbody_elbow_assets.py"
```

不得升级依赖。六次正式 run 必须执行同一 frozen runner。

native/matched 允许 config/model SHA 不同；runner/policy/motion/canonical/seed/dt/delay/gain 必须相同。

## 9. Step P1-4：2 秒 smoke

变量：

```bash
p1_root=/mnt/e/sim2sim-week-2026-08-26/10_p1_fullbody_elbow
canonical="$p1_root/00_inputs/canonical_initial_state.npz"
runner="$p1_root/00_inputs/frozen_run_mujoco_onnx.py"
native_cfg="$p1_root/01_asset/mimic_dance_9_native.yaml"
matched_cfg="$p1_root/01_asset/mimic_dance_9_elbow_matched.yaml"
```

native：

```bash
"$lab_python" "$runner" \
  --config "$native_cfg" --mode aligned --start-frame 0 \
  --history-init repeat-first --initial-state "$canonical" \
  --duration 2 --seed 42 --fixed-delay-ms 0 --gain-scale 1.0 \
  --output "$p1_root/02_smoke/mj_native" --offscreen
```

matched：

```bash
"$lab_python" "$runner" \
  --config "$matched_cfg" --mode aligned --start-frame 0 \
  --history-init repeat-first --initial-state "$canonical" \
  --duration 2 --seed 42 --fixed-delay-ms 0 --gain-scale 1.0 \
  --output "$p1_root/02_smoke/mj_matched" --record-video --offscreen
```

smoke 门禁：

- EXIT=0，100 cycles；
- fall reason 为空；
- 无 NaN/Inf；
- initial q/dq error ≤ `1e-6`；
- control dt=0.02，physics dt=0.002，ppc=10；
- native/matched runtime right elbow = `0.0685/0.01`；
- 其他 28 个 runtime armature 相同；
- policy/motion/canonical/runner hash 相同；
- matched 视频可见。

再把 native smoke 前 100 周期与冻结 MuJoCo reference 前 100 周期比较：

```text
q/dq/action/q_des/root_pos/root_quat 最大逐点差 <= 1e-6
```

超阈值时暂停，记录首次差异，检查 runner/config/XML/canonical/policy/motion。必要 runner 修订导致旧前缀无法复现时，执行第 12 节新 reference set。

## 10. Step P1-5：正式六次 MuJoCo run

native R1：

```bash
"$lab_python" "$runner" \
  --config "$native_cfg" --mode aligned --start-frame 0 \
  --history-init repeat-first --initial-state "$canonical" \
  --duration 10 --seed 42 --fixed-delay-ms 0 --gain-scale 1.0 \
  --output "$p1_root/03_runs/mj_native_r1" --record-video --offscreen
```

native R2/R3：

```bash
for rep in 2 3; do
  "$lab_python" "$runner" \
    --config "$native_cfg" --mode aligned --start-frame 0 \
    --history-init repeat-first --initial-state "$canonical" \
    --duration 10 --seed 42 --fixed-delay-ms 0 --gain-scale 1.0 \
    --output "$p1_root/03_runs/mj_native_r$rep"
done
```

matched R1：

```bash
"$lab_python" "$runner" \
  --config "$matched_cfg" --mode aligned --start-frame 0 \
  --history-init repeat-first --initial-state "$canonical" \
  --duration 10 --seed 42 --fixed-delay-ms 0 --gain-scale 1.0 \
  --output "$p1_root/03_runs/mj_matched_r1" --record-video --offscreen
```

matched R2/R3：

```bash
for rep in 2 3; do
  "$lab_python" "$runner" \
    --config "$matched_cfg" --mode aligned --start-frame 0 \
    --history-init repeat-first --initial-state "$canonical" \
    --duration 10 --seed 42 --fixed-delay-ms 0 --gain-scale 1.0 \
    --output "$p1_root/03_runs/mj_matched_r$rep"
done
```

每个 run 立即检查：

- `policy_trace.npz/run_manifest.yaml/summary.json/stdout.log`；
- 500 cycles，t 间隔 0.02s；
- 无 NaN/Inf；
- initial readback 和 runtime armature；
- hash 完整；
- R1 有视频。

失败目录不得覆盖。重试使用 `_retry1`，并在 `run_matrix.csv` 写明旧 run 无效原因。

## 11. Step P1-6：重复性、指标和门禁

### 11.1 重复性

native 和 matched 各自比较 R1/R2/R3：

- t、q、dq、q_des、action；
- root_pos/root_quat；
- 14-body actual/reference；
- fall reason 和 cycles；
- runner/policy/motion/config/model/canonical hash。

建议逐点阈值 `1e-7`。缺字段、少于三次或 hash 错配均 FAIL。

输出 `p1_repeat_check.json`。

### 11.2 指标脚本

创建只读 `p1_summarize.py`：

- 全部正式 trace 必须 500 周期；
- 时间差 ≤ `1e-6s`，不允许静默插值；
- 通过 manifest joint names 定位右肘；
- 按 0–2s、高激励窗口、0–10s 计算右肘 q MAE/RMSE；
- 同时计算 q_des、action、`q_des-q`；
- 计算 29 关节 q/q_des RMSE；
- 计算 root height、roll、pitch；
- 计算 14-body position/orientation 和 anchor；
- 计算 MuJoCo native/matched torque RMS/peak/saturation；
- 跨引擎 torque 标记 unavailable。

四元数姿态误差：

```text
angle = 2 * arccos(abs(dot(q1_normalized, q2_normalized)))
```

输出 `p1_metrics_per_run.csv` 和 `p1_metrics_summary.csv`。

### 11.3 图表

必须生成：

1. `elbow_q_0_2s.png`；
2. `elbow_q_active_window.png`；
3. `elbow_qdes_tracking_0_2s.png`；
4. `elbow_abs_error_vs_isaac.png`；
5. `joint_rmse_29.png`；
6. `root_10s.png`；
7. `body_anchor_10s.png`；
8. `native_matched_metric_bars.png`。

标注条件、armature、窗口、单位、三重复范围、physics/control dt。

### 11.4 final gate

`p1_final_gate.json` 分三部分：

```text
validity =
  input/hash/asset/config/runtime/repeat/time/joint/shape 全部通过

elbow_effect_pass =
  relative_improvement_0_2s >= 0.10
  AND absolute_improvement_0_2s >= 0.001
  AND repeat_direction_consistent

system_guard_pass =
  no_earlier_failure
  AND root guards
  AND all_joint_q_rmse_not_worse_over_10pct
```

决策：

| validity | effect | guard | decision |
|---|---|---|---|
| false | 任意 | 任意 | `INVALID_RERUN` |
| true | true | true | `SUPPORTED_TRANSFER_TO_FULLBODY` |
| true | true | false | `LOCAL_EFFECT_WITH_SYSTEM_TRADEOFF` |
| true | false | 任意 | `ISOLATED_EFFECT_NOT_TRANSFERRED` |

结论必须限定到当前 `dance_9`、当前初态和当前控制栈。禁止写“全身已对齐”“全部 gap 来自 armature”“所有关节都设为 0.01”或“全身改善 74.8%”。

## 12. Isaac 重跑分支

触发条件：

- reference 缺失或 SHA 不匹配；
- native 前缀无法复现且 runner 有必要修订；
- runner 改变接口、初态或字段语义。

PowerShell：

```powershell
$IsaacPython = "E:/isaaclab_env51/Scripts/python.exe"
$LabRepo = "E:/humanoid-lab"
$P1Root = "E:/sim2sim-week-2026-08-26/10_p1_fullbody_elbow"
$env:OMNI_KIT_ACCEPT_EULA = "YES"
Set-Location $LabRepo
& $IsaacPython "scripts/experiments/run_isaac_onnx.py" --task Tracking-Flat-L7_29Dof-v0 --policy "deploy/policy/dance_9/motion_anchor_obs_model_23500.onnx" --motion "deploy/policy/dance_9/dance_9.npz" --mode aligned --obs-source deploy-builder --start-frame 0 --history-init repeat-first --initial-state "$P1Root/00_inputs/canonical_initial_state.npz" --duration 10 --seed 42 --fixed-delay-ms 0 --gain-scale 1.0 --output "$P1Root/03_runs/isaac_reference_v2" --record-video
```

重跑后检查 hash、initial readback、runtime armature/Kp/Kd、500 cycles、fall 和视频。分析统一引用 v2。

## 13. 三栏视频

生成 `05_media/isaac_native_matched_10s_labeled.mp4`：

```text
左：Isaac reference，armature≈0.01，dt=0.005/control=0.02
中：MuJoCo native，armature=0.0685，dt=0.002/control=0.02
右：MuJoCo matched，armature=0.01，dt=0.002/control=0.02
```

要求 t=0 同步、速度一致、标注条件、保留异常段、全身可见、首中尾无黑帧。视频用于展示，结论使用 trace。

## 14. 失败处理

| 问题 | 处理 |
|---|---|
| matched XML 无法加载 | 检查标签、路径和 model hash |
| matched runtime 仍为 0.0685 | 检查目标 joint 显式属性和 config 指向 |
| 除右肘外存在差异 | 停止，重新生成并保留失败审计 |
| native 也 smoke 失败 | 检查环境/runner 回归 |
| 只有 matched 失败 | 检查资产，再记录局部修改的闭环影响 |
| cycle 0 已不同 | 检查 canonical/config/runner |
| 三重复不同 | 检查 hash、seed、线程、solver 和覆盖写入 |
| 证据不完整 | `validity=false`，不讨论物理结论 |

## 15. 反馈包

生成 `P1_fullbody_elbow_transfer_v1.zip`，包含：

- 本计划；
- native/matched XML 和 YAML；
- 生成、审计、汇总脚本；
- asset audit 和三个负测试；
- Isaac reference trace/manifest；
- 两个 smoke；
- 六次正式 MuJoCo trace/manifest/log；
- R1 视频；
- repeat check、指标 CSV、final gate；
- 八张图和三栏视频；
- `P1_result_summary.md`、`README_P1.md`；
- 根 `SHA256.txt` 和 ZIP companion SHA。

README 回答输入哈希、唯一模型差异、Isaac 是否复用、六次运行状态、主指标、系统保护结果、最终 decision 和结论限制。

新鲜解压后执行：

```bash
sha256sum -c SHA256.txt
```

只读验证器还需检查 trace 可加载、六个正式 run、三重复、asset audit、final gate、媒体链接和所有哈希。

## 16. 时间安排

| 阶段 | 预计 |
|---|---:|
| 输入与哈希 | 20–30 分钟 |
| XML/YAML 与审计 | 30–45 分钟 |
| smoke | 15–30 分钟 |
| 6×10s MuJoCo | 30–60 分钟 |
| 分析和图 | 45–90 分钟 |
| 视频和打包 | 30–60 分钟 |

总计约 3–5 小时。触发 Isaac 重跑时增加 Kit 时间。

## 17. 最终检查表

```text
[ ] policy/motion/XML/YAML/trace hash 命中
[ ] full-body canonical = 7fdf...
[ ] 未使用 isolation canonical 19bb...
[ ] 原始 XML 未覆盖
[ ] matched 只修改 right_elbow_pitch_joint
[ ] 未修改 default class elbow_pitch
[ ] 其他 28 个 runtime armature 一致
[ ] 三个负测试按预期 FAIL
[ ] native/matched smoke EXIT=0
[ ] native 前缀复现通过
[ ] native R1/R2/R3 完整
[ ] matched R1/R2/R3 完整
[ ] 每条 500 cycles
[ ] 三重复和 hash 门禁通过
[ ] active window 在读取 matched 前冻结
[ ] q/q_des/tracking/action 均报告
[ ] 29 关节、root、body、anchor 均报告
[ ] validity/effect/system 分开
[ ] 8 张图和三栏视频
[ ] SHA256 和新鲜解压验证通过
```

## 18. 完成后的路线

`SUPPORTED_TRANSFER_TO_FULLBODY`：

```text
ARM_ONLY → WAIST_ONLY → 决定是否进入 N6
```

`ISOLATED_EFFECT_NOT_TRANSFERRED`：

```text
PD 执行时序 → frictionloss → physics dt → solver
```

`LOCAL_EFFECT_WITH_SYSTEM_TRADEOFF`：先分析 action/q_des 和全身传播，再决定修改范围。

`INVALID_RERUN`：只修证据链和实验实现。

