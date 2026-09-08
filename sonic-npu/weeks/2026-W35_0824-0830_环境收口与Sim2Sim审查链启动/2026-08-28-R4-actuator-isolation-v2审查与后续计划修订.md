# R4 actuator isolation v2 结果审查与后续计划修订

> **2026-08-28 smoke 更新：**`R4F_N5R_smoke_v1` 已完成初审，但 hip/elbow 的 G2 隔离和 gate 汇总器仍需修正。全量 N5-R 的启动条件以 [smoke v1 验收与全量运行前修订](/Users/xerxes3/Documents/huawei实习/2026-08-28-R4F-N5R-smoke-v1验收与全量运行前修订.md) 为准。

> 审查日期：2026-08-28  
> 审查对象：`R4_actuator_isolation_v2.zip`  
> ZIP SHA256：`66c4b4e295c8dab4004d9464f087aca96fb2a4905b44aed023b53347f920aa34`  
> 证据原则：包内 README、manifest 和汇总图中的结论只作为待验证陈述；本审查以脚本、原始 NPZ、runtime 表和独立复算为准。

---

## 1. 一句话结论

**R4 已经证明“MuJoCo 的 elbow armature 会显著改变 elbow 瞬态响应”，但还没有可靠证明“armature 对齐使 Isaac–MuJoCo 误差下降 55%”。**

当前不应直接进入 N6。先修正 Isaac 阶跃测试的前置激励污染、时间戳错相、固定基座和接触/关节隔离问题，再重跑 elbow/hip。N5 重跑通过后，N6 的 armature 干预也不能只改一个 elbow，而应覆盖 N4 找到的全部 17 个不一致关节，并按 arms/waist 分组做消融。

当前验收状态：

| 阶段 | 状态 | 说明 |
|---|---|---|
| N3.1 runner 修复 | PASS | joint/dof id 已修正；14/14 测试通过；脚本可编译 |
| N4 runtime 参数表 | PASS，带一项语义保留 | 29 行齐全，Kp/Kd 和数值限幅一致，关键 ID 正确；但“数值限幅相同”还不能单独证明两个引擎的隐式执行器语义完全相同 |
| N5-0 torque parity | 不通过跨引擎 parity 验收 | MuJoCo 在静态初态算 torque，Isaac 在推进 20 ms 后读取 torque，比较的不是同一状态和时刻 |
| N5 elbow/hip step | 仅作 B 级方向性证据 | elbow 干预效应和 hip no-op 对照有效；Isaac 曲线污染和采样错相使跨引擎定量指标暂不可用 |
| N6 整机实验 | HOLD | 必须先完成 N5-R 重跑 gate |

---

## 2. 当前包中可以接受的结果

### 2.1 包完整性与可复现材料

- ZIP CRC 检查通过；
- `SHA256.txt` 中全部文件校验通过；
- 官方 XML 的 SHA256 未被覆盖；
- elbow 干预 XML 与官方 XML 按行比较，只有 `right_elbow_pitch_joint` 增加了 `armature="0.01"`，其余内容未变；
- 包内 Python 脚本全部通过 `py_compile`；
- N3.1 测试结果为 14/14 PASS。

### 2.2 N4 runtime 参数结论

N4 表中的关键值可以接受：

| 项目 | Isaac | MuJoCo | 判断 |
|---|---:|---:|---|
| right elbow armature | 0.0100 | 0.0685 | 明确不一致 |
| right hip pitch armature | 0.0968 | 0.0968 | 一致，可作 no-op 对照 |
| right elbow Kp/Kd | 38.311 / 2.115 | 38.311 / 2.115 | 数值一致 |
| right hip Kp/Kd | 414.874 / 29.165 | 414.874 / 29.165 | 数值一致 |
| right elbow effort/ctrl limit | 95 | ±95 | 数值一致 |
| right hip effort/ctrl limit | 350 | ±350 | 数值一致 |
| right elbow joint/dof id | — | 26 / 31 | 已修正，可追溯 |
| right hip joint/dof id | — | 9 / 14 | 已修正，可追溯 |

更重要的是，29 个关节中有 **17 个 armature 不一致**：

- waist：3 个；
- shoulder pitch/roll：4 个；
- arm yaw：2 个；
- elbow pitch/yaw：4 个；
- wrist pitch/roll：4 个。

因此后续整机实验若只修改 `right_elbow_pitch_joint`，不能称为“armature matched 整机实验”。

### 2.3 MuJoCo 内部 elbow 干预具有因果意义

在 MuJoCo 内部，native 和 matched 两次运行只修改 elbow armature：

- native：0.0685；
- matched：0.0100；
- 阶跃后两条 elbow 轨迹平均绝对差约 `0.00802 rad`；
- 最大绝对差约 `0.01860 rad`；
- 最后一个已记录样本的差约 `0.00452 rad`；
- 全程没有 torque saturation。

这可以支持：

> elbow armature 是 MuJoCo elbow 瞬态响应的重要影响因子。

它暂时不能单独支持：

> armature 已经解释了 Isaac–MuJoCo elbow 误差的 55%。

### 2.4 hip no-op 对照有效

right hip pitch 在两侧 armature 都是 0.0968。MuJoCo 的 hip native 与 matched：

- `q` 数组逐点完全相同；
- torque 数组逐点完全相同；
- 汇总曲线完全重合。

这说明干预脚本没有在“数值未改变时”制造额外动力学差异，是一个有效的 negative control。但它不能证明 Isaac–MuJoCo 的其他动力学参数已经一致。

---

## 3. elbow/hip 阶跃对比图应该怎样解读

原图：`n5/n5_response_comparison.png`。

### 3.1 elbow 图中真正可见的现象

1. MuJoCo armature 从 0.0685 降到 0.01 后，elbow 上升更快；
2. 在阶跃后的前半段，matched 曲线比 native 更靠近当前 Isaac 曲线；
3. 大约 0.25 s 后，Isaac 与两条 MuJoCo 曲线仍有不同的过冲/收敛形状；
4. 因此即使重跑后结论方向不变，也更可能是“armature 解释早期瞬态的一部分”，而不是“armature 解释全部差异”。

### 3.2 hip 图中真正可见的现象

1. MuJoCo native 与 matched 完全重合，符合 no-op 预期；
2. Isaac 曲线在标记为 `t=0` 时已经是约 `0.69768 rad`，而共同初态是 `0.68033 rad`；
3. 两条 MuJoCo 曲线从真实共同初态 `0.68033 rad` 开始；
4. 因此 hip 图开头的约 `0.01735 rad` 差异不是正常的“同初态响应差”，而是 Isaac 记录流程已经提前推进过物理。

### 3.3 图上的时间轴目前不等价

Isaac NPZ：

- 25 个状态样本：`0.00, 0.02, ..., 0.48 s`；
- 每个状态都是一次 `env.step()` 完成后才记录；
- 标记为 `t=0` 的点不是初态，而是推进后的状态；
- 没有真实 `t=0` 初态，也没有真实 `t=0.50 s` 状态。

MuJoCo NPZ：

- 250 个状态样本：`0.000, 0.002, ..., 0.498 s`；
- 状态在 `mj_step()` 前记录；
- 第一个点是真实共同初态；
- 同样没有记录 `t=0.500 s` 的最终状态。

所以当前汇总中的 `q_at_0.5s` 和 `steady_state_diff_at_0.5s` 实际使用的是约 0.48 s 的样本；`t50` 和 MAE 又把 post-step Isaac 状态与 pre-step MuJoCo 状态直接按数组下标相减。这些数值应撤回，等重跑后重新生成。

---

## 4. 必须修正的问题

### P0-1：Isaac 阶跃被前面的 parity step 污染

`run_isaac_onnx.py` 当前流程是：

```text
写入 canonical q0/dq0
  → 先给被测关节 +0.1 rad，env.step 20 ms，生成 parity JSON
  → 不重置环境
  → 进入 0–0.1 s 保持 q0 的 step loop
```

于是正式阶跃记录开始前，关节已经吃过一次 +0.1 rad 激励。随后标记为 `t=0` 的第一点又是一次 20 ms hold step 后的状态。因此第一条已记录状态距离 canonical 初始化实际已经推进至少两个 control cycle。

原始证据：

| 关节 | canonical q0 | Isaac 第一个记录点 | 起点误差 |
|---|---:|---:|---:|
| right elbow pitch | 0.488362 | 0.491819 | +0.003457 rad |
| right hip pitch | 0.680333 | 0.697681 | +0.017348 rad |

**修法：parity 和 step 必须用两个全新进程分别运行。**不要依赖在同一环境中“再写一次状态”，因为 actuator target、内部 buffer 或 contact cache 也可能残留。

### P0-2：状态采样时刻和标签定义不同

Isaac 是 step 后记录，MuJoCo 是 step 前记录。必须统一为：

```text
t=0：先记录共同初态
对区间 [t, t+dt) 应用本周期命令
推进物理
在 t+dt 记录新状态
```

预期样本数：

- Isaac control-tick trace：26 个状态，`0.00 ... 0.50 s`；
- MuJoCo physics trace：251 个状态，`0.000 ... 0.500 s`。

命令最好单独保存为 `command_start_time`、`q_des_applied`，不要让状态时间和命令时间共用一个含糊字段。

### P0-3：当前实验并非真正“单关节隔离”

其余 28 个关节虽然 `q_des=q0`，但仍会受重力、耦合和 PD 误差影响。当前最大漂移甚至超过被测阶跃 0.1 rad：

| 条件 | 其他关节最大漂移 |
|---|---:|
| MuJoCo elbow | 0.1169 rad |
| Isaac elbow | 0.1345 rad |
| MuJoCo hip | 0.1214 rad |
| Isaac hip | 0.1385 rad |

最大漂移主要出现在 ankle pitch。这只能叫“全身耦合系统中的单关节命令”，不能叫“A 级单关节隔离实验”。

### P0-4：固定基座实现不一致

- MuJoCo：每 2 ms physics step 前把 root pose/velocity 写回；
- Isaac：每 20 ms control step 前写回一次，随后内部推进 4 个 5 ms physics substep。

两者不是相同约束。优先用两侧各自的固定根模型/约束；若做不到，至少要在 physics substep 级应用等价约束并记录 root position/orientation 的最大漂移。

### P0-5：MuJoCo “contacts=off” 没有证据支持

manifest 声称模型默认 `contype/conaffinity=0`，但官方 XML 中同时存在：

- 明确设置为 `contype=1, conaffinity=1` 的 collision default；
- 多个未显式关闭碰撞的 collision geom；
- 未关闭碰撞的 floor geom。

测试脚本没有主动关闭 contact，也没有记录 `ncon` 或 contact force。当前最多只能写“未验证接触状态”，不能写“无碰撞”。

### P0-6：N5-0 不是同状态 torque parity

MuJoCo parity 在 `q=q0, dq=0`、不推进物理时计算：

- elbow：3.8311 Nm；
- hip：41.4874 Nm。

Isaac parity 在推进 20 ms 后读取：

- elbow：`q=0.48942, dq=0.67640`，computed/applied=2.6340 Nm；
- hip：`q=0.69803, dq=0.49697`，computed/applied=20.1347 Nm。

它可以证明 Isaac 当时的 `computed_torque == applied_torque`，但不能与 MuJoCo 静态值直接比较。下一版应把“同状态 PD 指令一致”和“推进后 engine readback”拆成两个指标。

### P1：证据包缺少的材料

- 缺 elbow native parity JSON；
- 缺 hip matched 干预 XML 副本；
- matched parity JSON 只写了官方 XML hash，没有同时写 `xml_used_sha256`；
- trace 没有 contact count/contact force；
- Isaac trace 没有真实初态样本和 physics-substep 状态。

这些不是当前结论失效的首要原因，但重跑包应补齐。

---

## 5. 修订后的可执行计划

## Step R4-F：修复 N5 测试工具

预计 45–90 分钟。先改代码，不跑 N6。

### R4-F1：拆开 parity 与 step 进程

怎么做：

1. 保留 `--torque-parity-only`，单独启动一次进程生成 parity；
2. 增加 `--step-only`，这个进程不得先执行 parity；
3. `--step-only` 启动后 reset，再写入 canonical `q0`、`dq0=0`、root velocity=0；
4. 写入后立刻回读并保存 `initial_state_readback.json`；
5. 若任何关节 `|q-q0| > 1e-5 rad` 或 `|dq| > 1e-5 rad/s`，立即停止。

需要收集：

- 两个独立命令行和 stdout；
- 两个进程各自的 manifest；
- initial-state readback；
- policy/motion/XML/initial-state SHA256。

### R4-F2：统一状态/命令时间语义

怎么做：

1. 两侧都在推进物理前保存 `t=0` 初态；
2. 命令在 `[t,t+dt)` 生效，状态标记为推进后的 `t+dt`；
3. 精确保存 `t_state` 和 `t_command_start`；
4. 运行到并记录 `t_state=0.500 s`；
5. 汇总时用时间插值把 MuJoCo 状态插到 Isaac 的真实状态时间，不再使用 `q[::10]`；
6. summary 程序遇到 25/250 个旧式样本时直接 FAIL，不能静默出图。

验收：

```text
Isaac t_state: 0.00 ... 0.50，共 26 点
MuJoCo t_state: 0.000 ... 0.500，共 251 点
两侧第一个 q 与 q0 最大差 <= 1e-5 rad
两侧第一个 dq 最大值 <= 1e-5 rad/s
step command 的开始时间都严格为 0.100 s
```

### R4-F3：建立真正的 clean isolation 条件

本轮建议同时保留两种协议，避免把工程真实情况和因果诊断混在一起。

#### 协议 A：CLEAN_ISOLATED（主因果证据）

- gravity=0；
- contacts 明确关闭；
- root 用固定根模型或等价硬约束；
- 其余 28 个 DOF 用等价关节锁定方式固定在 q0；
- 被测关节 dq0=0；
- delay=0；
- Kp/Kd 与 N4 runtime 表一致；
- elbow 分别跑 native/matched；
- hip 分别跑 native/matched no-op。

验收：

```text
max contact count == 0
max contact force == 0
root position/orientation drift <= 1e-5
其他 28 关节最大漂移 <= 1e-4 rad
全程无 saturation；若发生则单独标记，不能混入主结论
```

#### 协议 B：COUPLED_GRAVITY（工程补充证据）

- 保留 gravity 和全身 PD；
- 只给一个关节阶跃命令；
- 明确命名为“single-command coupled response”；
- 记录所有关节漂移、root 漂移和 contact；
- 不用它声称“单关节隔离”，只用于观察 armature 改动在较真实条件下是否仍有相同方向。

如果时间只够完成一种协议，先完成协议 A。

### R4-F4：修正 torque parity 定义

输出两个层次：

1. `pd_contract_at_common_state`：共同 `q0/dq0/q_des` 下离线计算 raw torque 和 clipped torque；两侧必须逐关节一致；
2. `engine_torque_readback`：推进后记录当时真实 q/dq、computed torque、applied torque，只在相同状态/相同时间语义下比较。

主 gate：

```text
common-state raw torque max error <= 1e-5 Nm
common-state clipped torque max error <= 1e-5 Nm
本次 elbow/hip 主实验不得触发 saturation
```

如果 Isaac API 无法在不推进物理时给出 implicit actuator readback，不要伪装成同状态 applied parity；保留离线 contract 检查，并把 engine readback 明确标成推进后诊断值。

---

## Step N5-R：重跑 elbow/hip

预计 60–120 分钟，取决于固定根和关节锁定实现。

运行矩阵：

| 关节 | 条件 | armature | 用途 |
|---|---|---:|---|
| right elbow pitch | Isaac native | 0.0100 | 参考 |
| right elbow pitch | MuJoCo native | 0.0685 | 主对照 |
| right elbow pitch | MuJoCo matched | 0.0100 | 主干预 |
| right hip pitch | Isaac native | 0.0968 | 参考 |
| right hip pitch | MuJoCo native | 0.0968 | no-op 对照 |
| right hip pitch | MuJoCo matched | 0.0968 | no-op 对照 |

每个条件至少跑 3 次。确定性仿真中重复的最大 q 差应不超过 `1e-6 rad`；若不能复现，先排查随机化、solver warm start 和 reset 缓存。

每次必须保存：

- `t_state`、`t_command_start`；
- q、dq、q_des；
- raw/clipped/applied torque；
- saturation mask；
- root pose/velocity；
- 所有关节 q/dq；
- contact count、contact force；
- runtime armature/Kp/Kd/limit；
- 官方 XML 和实际 XML 的 SHA256；
- isolation mode、gravity、fixed-base 方法、joint-lock 方法。

重新计算的指标：

- response-window MAE、RMSE、max error；
- t10、t50、t90，阈值相对 `q0 + {0.1,0.5,0.9} × step` 定义，并做相邻样本线性插值；
- 最大 overshoot；
- 精确 `q(t=0.5s)`，不要称“稳态”除非另有稳态判据；
- torque MAE/max error；
- 其他关节漂移、root 漂移、contact、saturation；
- elbow native→matched 的误差改善率；
- hip native/matched 的最大逐点差。

N5-R 通过条件：

```text
G1 初态与时间语义全部通过
G2 clean isolation 的 root/contact/other-joint 阈值通过
G3 torque contract 通过且主实验无 saturation
G4 elbow matched 相比 native 的 MAE、RMSE、t50 至少两项稳定改善
G5 hip native 与 matched 最大逐点差 <= 1e-6 rad
G6 三次重复满足确定性阈值
```

只有 G1–G6 全部通过，才可以把结论升级为 A 级证据，并报告准确改善百分比。

---

## Step N6-R：整机短前缀 armature 分组消融

只有 N5-R 通过后执行。先跑 0.5 s，不直接跑 5 s/20 s。

根据 N4 的 17 个 armature mismatch，至少运行：

| 条件 | 修改范围 | 回答的问题 |
|---|---|---|
| NATIVE | 不修改 | 原始基线 |
| ARM_ONLY_MATCHED | 14 个双臂关节 | 上肢 armature 是否解释 cycle 1 的 elbow/arm 分叉 |
| WAIST_ONLY_MATCHED | 3 个腰部关节 | 腰部惯量差是否向全身传播 |
| ALL_17_MATCHED | 全部 17 个不一致关节 | armature 总体能解释多少整机前缀误差 |

每个干预 XML 必须：

1. 从同一官方 XML 自动生成；
2. 输出逐关节 old/new 值；
3. 保存官方/干预 XML SHA256；
4. runtime 回读确认 17 个目标值；
5. 断言未列入干预集合的关节数值不变。

整机 0.5 s 指标优先看：

- cycle 1–25 的 q/dq/action/q_des/torque；
- right elbow、双臂 RMS、waist RMS、全身 RMS；
- 首次超过阈值的 cycle 和 joint；
- root/body 误差；
- 是否改变 fall precursor；
- 相对 NATIVE 的误差改善率。

决策：

- 若 `ALL_17_MATCHED` 明显改善，进入 5 s 复验，并保留 arms/waist 消融解释贡献；
- 若只改善 elbow、全身指标不改善，结论限定为局部执行器差异，不把 armature 当作主要整机根因；
- 若 clean N5 有效而 N6 无效，下一步查 coupling/contact/integrator，而不是立即加 delay；
- delay 和 gain sweep 仍放最后，不能在基础动力学未对齐时提前使用。

---

## 6. 下一次反馈点与反馈包

最合适的下一次反馈点是：**R4-F 修改完成，并且只跑完一组 elbow CLEAN_ISOLATED smoke 后立即反馈。**不要等六个正式条件全部跑完，先用这个 smoke 检查时间轴、初态、隔离和数据格式。

建议反馈包：

```text
R4F_N5R_smoke_v1.zip
├── README.md
├── SHA256.txt
├── scripts/
├── checks/
│   ├── sampling_contract.json
│   ├── initial_state_gate.json
│   └── isolation_gate.json
├── n5_0/
│   ├── pd_contract_at_common_state.json
│   └── engine_torque_readback.json
├── elbow_clean_smoke/
│   ├── isaac_native/
│   ├── mujoco_native/
│   └── mujoco_matched/
├── plots/
│   ├── elbow_q.png
│   ├── elbow_dq.png
│   └── elbow_torque.png
└── manifests/
```

反馈前人工检查：

```text
[ ] 两侧曲线都从完全相同的 t=0/q0/dq0 开始
[ ] 图中确实存在 t=0.500 s 状态
[ ] Isaac step 进程没有先执行 parity
[ ] step 在 0.100 s 生效
[ ] contact=0 有 runtime 计数，不只是一句说明
[ ] 其他 28 关节漂移未超过阈值
[ ] root 约束方法和漂移已记录
[ ] summary 使用真实时间插值
[ ] README 没有使用“稳态”“A 级”“改善 55%”等未通过 gate 的表述
```

---

## 7. 当前可以写入周报的谨慎结论

可以写：

> Runtime 参数审计发现 Isaac 与 MuJoCo 在 29 个关节中的 17 个关节存在 armature 差异。MuJoCo 内部单变量干预显示，将 right elbow pitch 的 armature 从 0.0685 调整为 0.01 会显著改变其阶跃瞬态；right hip pitch 在 armature 数值不变时 native/matched 轨迹完全一致，支持干预脚本的 no-op 正确性。当前跨引擎阶跃数据存在 Isaac 前置激励和时间采样错相，精确误差改善比例需要在修正实验协议后重新测量。

暂时不要写：

> armature 对齐已经使跨引擎误差下降 55%，并已获得 A 级单关节隔离证据。
