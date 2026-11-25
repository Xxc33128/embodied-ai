# Phase 7 Windows 执行清单（2026-08-20）

## 0. 目标与边界

本阶段目标是证明 Isaac 与 MuJoCo 在同一个 29-DoF SONIC checkpoint 接口下数值等价，最终生成可复核的 `gate_report.json`。

本阶段的硬规则：

- **不修改任何物理参数**；
- 不上传 checkpoint、motion、STL、视频、原始 NPZ 和 `artifacts/`；
- 不向原项目 push，只向 `Xxc33128/GR00T-WholeBodyControl-alignment` 个人私库 push；
- Isaac 和 MuJoCo 的自然 rollout 不算同状态证据；
- interface gate 失败时，不进入物理 benchmark 和调参。

---

## 1. P0：立即撤销已泄露的 GitHub PAT

已出现过的 PAT 必须视为已泄露凭据，立即在 GitHub 撤销，不要继续使用。

撤销后检查：

- PowerShell history；
- 下载脚本、配置和日志；
- Markdown 文档；
- Git tracked/staged/untracked 文件；
- 本地 Git 历史。

可使用以下命令搜索常见 token 前缀：

```powershell
git grep -n -I -E "github_pat_|ghp_|gho_|ghu_|ghs_|ghr_"
Get-ChildItem -Recurse -File | Select-String -Pattern "github_pat_|ghp_|gho_|ghu_|ghs_|ghr_"
```

新 token 只能放在环境变量或 Windows Credential Manager，不能写入命令脚本、Markdown 或 Git remote URL。

---

## 2. 审查并提交当前 Windows 修复

### 2.1 确认分支和远端

```powershell
git branch --show-current
git remote -v
git status --short
git fetch personal
```

必须确认：

- 当前分支为 `feature/isaac-mujoco-alignment` 或专用 Windows topic branch；
- push remote 为个人私库 `personal`；
- 不存在对原项目的 push URL；
- 不使用 `git add -A`、`git clean`、`git stash` 或 force push。

### 2.2 提交前检查

```powershell
git diff --check
git diff --name-status
git status --short
```

不能进入 commit 的内容：

```text
gear_sonic_deploy/g1/meshes/*.STL
artifacts/**
*.npz
*.pt / *.pth / *.ckpt
*.pkl
视频和日志
任何 token
带 Windows 用户名的绝对路径
```

STL 格式转换脚本可以提交，但脚本里不能有 token 或本机绝对路径。

### 2.3 拆分原子 commit

建议拆为：

```text
fix(cross-engine): repair Windows capture runtime
feat(cross-engine): add deterministic capture tooling
```

仅显式 add 本次文件，例如：

```powershell
git add <capture-runtime-files>
git diff --cached --name-status
git commit -m "fix(cross-engine): repair Windows capture runtime"

git add <capture-tool-and-test-files>
git diff --cached --name-status
git commit -m "feat(cross-engine): add deterministic capture tooling"
```

提交后必须：

```powershell
git status --short
python -m pytest -q tests/cross_engine
python -m ruff check gear_sonic/cross_engine tests/cross_engine
```

`git status --short` 应为空。如果 mesh/artifacts 作为 ignored 文件存在，也不能出现在 staged diff 里。

push 前再确认私库：

```powershell
gh repo view Xxc33128/GR00T-WholeBodyControl-alignment --json visibility
git push personal HEAD:refs/heads/feature/isaac-mujoco-alignment
gh repo view Xxc33128/GR00T-WholeBodyControl-alignment --json visibility
```

两次都必须显示 `PRIVATE`。不要 force push。

---

## 3. 生成正式环境与资产指纹

在仓库干净之后生成 `fingerprint.json`。

本地创建一份不上传的：

```text
artifacts/cross_engine/<run_id>/fingerprint.input.local.json
```

内容示例：

```json
{
  "files": {
    "isaac_asset": "E:\\local\\g1.usd",
    "mujoco_model": "E:\\local\\g1_29dof.xml",
    "checkpoint": "E:\\local\\last.pt",
    "motion": "E:\\local\\sample_motion.pkl",
    "resolved_config": "E:\\local\\benchmark.json"
  },
  "settings": {
    "physics_dt_isaac": 0.005,
    "physics_dt_mujoco": 0.002,
    "control_dt": 0.02,
    "isaac_decimation": 4,
    "mujoco_decimation": 10,
    "seed": 7,
    "headless": true,
    "gpu": "<GPU model>",
    "driver": "<driver version>",
    "cuda": "<CUDA version>"
  }
}
```

执行：

```powershell
python -m gear_sonic.cross_engine.cli fingerprint `
  --input artifacts/cross_engine/<run_id>/fingerprint.input.local.json `
  --output artifacts/cross_engine/<run_id>/fingerprint.json
```

检查：

- `source_git.dirty` 必须为 `false`；
- `source_git.commit` 必须是刚提交的 40 位 SHA；
- `files` 只包含 label、size、SHA-256，不得包含本机路径；
- checkpoint 和 motion 的 SHA-256 后续必须在两引擎 metadata 中一致。

---

## 4. 补齐 Isaac collider 审计

当前 Isaac 审计输出 `Colliders = 0`。这不能作为“预期差异”直接接受：圆柱碰撞模型应该在 USD stage 中有 collision prim。

需要检查：

1. `UsdPhysics.CollisionAPI` 是否真正应用到 prim；
2. collider 是否位于 fixed child prim 下；
3. exporter 是否把 collider 归并到最近 canonical rigid ancestor；
4. `robot_path` 前缀是否与实际 stage path 一致；
5. cylinder/capsule/mesh collider 的 size、pose、material 与 contact flags 是否能取到。

修复后重跑：

```powershell
python -m gear_sonic.cross_engine.cli export-isaac `
  --device cuda:0 `
  --headless `
  --output artifacts/cross_engine/<run_id>/audit/isaac_model.json

python -m gear_sonic.cross_engine.cli export-mujoco `
  --model-xml gear_sonic_deploy/g1/g1_29dof.xml `
  --decimation 10 `
  --output artifacts/cross_engine/<run_id>/audit/mujoco_model.json

python -m gear_sonic.cross_engine.cli diff `
  --left artifacts/cross_engine/<run_id>/audit/isaac_model.json `
  --right artifacts/cross_engine/<run_id>/audit/mujoco_model.json `
  --csv artifacts/cross_engine/<run_id>/audit/model_diff.csv `
  --unmapped-json artifacts/cross_engine/<run_id>/audit/unmapped_parameters.json `
  --summary-markdown artifacts/cross_engine/<run_id>/audit/audit_summary.md
```

验收：

- Isaac collider 不能再是 0；
- 每个 collider 有明确 owner/body；
- 缺失的 material/contact 字段必须显式记录，不能静默当成 0；
- solver 只标记 semantic/behavioral，不声称数值等价。

collider 导出问题不一定阻止无接触的接口门禁，但在 Phase 8 正式物理 benchmark 前必须解决。

---

## 5. 实现真实 interface capture factory

建议新增两个库内 factory，但以实际环境结构为准：

```text
gear_sonic/cross_engine/capture_adapters/isaac.py:create_adapter
gear_sonic/cross_engine/capture_adapters/mujoco.py:create_adapter
```

每个 adapter 必须实现：

```python
class CaptureAdapter:
    def reset(self) -> None: ...
    def next_frame(self) -> Mapping[str, object]: ...
    def close(self) -> None: ...
```

采集器只接收 runtime 暴露的真实命名阶段，不允许 adapter 伪造缺失值。

### 5.1 每帧必须提供的接口阶段

- canonical 29 joint names/order；
- 5 个 named observation blocks；
- flattened actor observation；
- raw policy action；
- wrapper-clipped action；
- canonical action；
- target joint position；
- unclipped PD torque；
- applied/clipped torque；
- action effective substep；
- reference motion id/time/future indices；
- reset/step index。

### 5.2 observation/history 合同

```text
gravity_dir   = 3  × 10 = 30
base_ang_vel  = 3  × 10 = 30
joint_pos     = 29 × 10 = 290
joint_vel     = 29 × 10 = 290
last_action   = 29 × 10 = 290
flattened actor observation = 930
```

Isaac 配置中的 `actions` 如果映射到 trace 中的 `last_action`，必须在 adapter 中显式记录这个命名映射。

### 5.3 每帧必须提供的 metadata

```json
{
  "noise_disabled": true,
  "randomization_disabled": true,
  "auto_reset_disabled": true,
  "stochastic_policy_disabled": true,
  "source_git_commit": "<40 lowercase hex>",
  "checkpoint_sha256": "<64 lowercase hex>",
  "motion_sha256": "<64 lowercase hex>",
  "state_set_sha256": "<64 lowercase hex>",
  "engine_asset_sha256": "<64 lowercase hex>",
  "runtime_versions": {
    "simulator": "<actual version>"
  },
  "observation_history_lengths": {
    "gravity_dir": 10,
    "base_ang_vel": 10,
    "joint_pos": 10,
    "joint_vel": 10,
    "last_action": 10
  }
}
```

两引擎必须完全相同：

- `source_git_commit`；
- `checkpoint_sha256`；
- `motion_sha256`；
- `state_set_sha256`。

`engine_asset_sha256` 和 `runtime_versions` 允许不同，但必须真实且非空。

---

## 6. 构造 100-state 确定性状态集

状态集必须一次生成、序列化并计算 SHA-256，两引擎从同一份状态集读取。

每个状态至少包含：

- canonical root pose/orientation；
- canonical root linear/angular velocity；
- canonical 29-DoF q/qd；
- reference motion id/time/future indices；
- 完整 observation/action history；
- 指定的 raw policy action 或确定性 policy 输入；
- reset/step/history slot 信息。

100 个状态必须覆盖：

1. reset 首帧；
2. history 尚未填满；
3. history 刚好填满；
4. 普通站立/跟踪状态；
5. action 超过 ±1 但未超过 wrapper ±20；
6. wrapper clip 边界；
7. target joint position 触发限位；
8. torque 未饱和与刚好饱和；
9. 不同 reference time/future index 组合；
10. 四元数符号等价情况。

不允许：

- 用 100 帧自然 rollout 代替 100 个同状态输入；
- 两引擎各自 reset 后独立演化；
- 在 adapter 内静默修正输入；
- 用全零/合成 trace 冒充真实 runtime 证据。

状态集生成后：

1. 计算 `state_set_sha256`；
2. 补入两个 capture config；
3. 将 state-set 加入 fingerprint 的 `files`；
4. 重新生成最终 `fingerprint.json`。

---

## 7. 执行两侧 capture

```powershell
python -m gear_sonic.cross_engine.cli capture-interface `
  --engine isaac `
  --adapter gear_sonic.cross_engine.capture_adapters.isaac:create_adapter `
  --config artifacts/cross_engine/<run_id>/interface/isaac_capture.local.json `
  --output artifacts/cross_engine/<run_id>/interface/isaac_golden.npz `
  --frames 100

python -m gear_sonic.cross_engine.cli capture-interface `
  --engine mujoco `
  --adapter gear_sonic.cross_engine.capture_adapters.mujoco:create_adapter `
  --config artifacts/cross_engine/<run_id>/interface/mujoco_capture.local.json `
  --output artifacts/cross_engine/<run_id>/interface/mujoco_candidate.npz `
  --frames 100
```

两个命令必须都返回 0，且不得存在 `.error.json`。

---

## 8. 执行 interface gate

```powershell
python -m gear_sonic.cross_engine.cli gate-interface `
  --reference artifacts/cross_engine/<run_id>/interface/isaac_golden.npz `
  --candidate artifacts/cross_engine/<run_id>/interface/mujoco_candidate.npz `
  --report artifacts/cross_engine/<run_id>/interface/gate_report.json
```

固定阈值：

| 检查项 | 阈值 |
|---|---:|
| observation 及 named blocks | `max_abs <= 1e-5` |
| policy/raw/clipped/canonical action | `max_abs <= 1e-5` |
| target joint position | `max_abs <= 1e-6` |
| unclipped/applied torque | `abs <= 1e-4` 且 `rel <= 1e-5` |
| joint order/block order/history/timing/provenance | 必须精确相等 |

验收：

- CLI exit code 为 0；
- `gate_report.json` 中 `passed: true`；
- `reference_engine: isaac`；
- `candidate_engine: mujoco`；
- `frames: 100`；
- 所有 exact check 与 numeric block 都通过。

gate 失败时立即停止，不运行 Phase 8 成对物理 benchmark。

---

## 9. 需要回传的内容

可回传：

- 最终 Git commit SHA；
- `verify_windows_env.ps1` 的 PASS/FAIL 输出；
- 去除用户路径后的 `fingerprint.json`；
- `audit_summary.md`；
- `model_diff.csv` 和 `unmapped_parameters.json`（确认无机器路径后）；
- Isaac/MuJoCo collider 数量与导出问题说明；
- `gate_report.json`；
- capture/gate 的小型 Markdown 摘要；
- 失败时的脱敏错误信息。

不可回传：

- `isaac_golden.npz`、`mujoco_candidate.npz`；
- state-set 原始文件；
- checkpoint/motion；
- STL 网格；
- 视频；
- token；
- 未脱敏环境转储或绝对路径。

---

## 10. 完成回复模板

```text
Phase 7 Windows 执行结果

Git commit:
Git working tree clean: yes/no
GitHub repository visibility: PRIVATE/not-confirmed
Leaked PAT revoked: yes/no

Environment check: pass/fail
Isaac Sim:
Isaac Lab:
PhysX:
MuJoCo:
Python:
GPU / driver / CUDA:

Fingerprint generated: yes/no
source_git.dirty:
checkpoint SHA256:
motion SHA256:
state-set SHA256:

Isaac bodies/joints/actuators/colliders:
MuJoCo bodies/joints/actuators/colliders:
Isaac collider exporter fixed: yes/no

Isaac capture frames:
MuJoCo capture frames:
Interface gate: pass/fail
Gate report path:

Known failures:
Files returned:
Files intentionally kept local:
```

---

## 11. 最终停止条件

任一条发生即停止：

- PAT 未撤销或仓库仍能搜到 token；
- 工作树不干净；
- fingerprint `source_git.dirty != false`；
- checkpoint/motion/state-set SHA 在两侧不一致；
- observation 不是 930 维；
- capture 使用两条自然 rollout；
- capture 输出存在 `.error.json`；
- gate 的 engine role 不是 Isaac → MuJoCo；
- `gate_report.json` 不是 `passed: true`。

只有 Phase 7 gate 通过后，才可以进入 Phase 8 的 18-case 成对物理 benchmark。
