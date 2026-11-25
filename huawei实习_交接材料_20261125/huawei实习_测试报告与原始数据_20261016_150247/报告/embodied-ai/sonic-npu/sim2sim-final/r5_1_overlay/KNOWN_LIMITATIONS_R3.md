# R3 已知限制（R4 记录，不影响 trace 有效性）

## P1-1 源码注释 vs 实际实现
执行过的 `p3_instrumented_delayed_actuator_r2.py` 文件头 docstring 仍保留早期"通过 DelayBuffer.compute / DelayBuffer 输出后"的描述，且类内 `compute()` 对 **velocity/effort** 仍走原生 `DelayedImplicitActuator` 的 `DelayBuffer`。
实际 **position** 目标（P3 研究对象）用的是 `collections.deque` 实现的显式 per-physics-step FIFO + repeat-first-q-des prefill，未调用原生 position DelayBuffer。
统一表述：
```
P3 explicit per-physics-step position-target FIFO
implementation = collections.deque
prehistory = repeat-first-q-des
position_delay_backend = explicit_deque
native_position_delay_buffer_bypassed = true
native_velocity_effort_delay_buffer_used = true
```
结论口径：本实验是「受控等时延注入」，不是对 Humanoid Lab 原生 DelayBuffer 的鲁棒性验证。

## P1-2 reset_count
runner manifest 的 `reset_count=0` 是常量，validator 不直接信任；无 reset 由 trace 独立推断：
policy_step=0..499、time_step=0..499、actuator physics-step index=1..2000 连续、done 全 0。最终写 `reset_observed=false, reset_evidence=independent_trace_counters`。

## P1-4 仓库 dirty
`repo_state.json`：Humanoid Lab commit c68c1e63、branch main、dirty=true、88 项改动（含 actuator.py / run_isaac_onnx.py / run_mujoco_onnx.py 等）；Isaac Lab 5c2ec81。结论依赖 artifact SHA256，不依赖 clean commit 假设。

## Isaac 0 ms 回归 runner 前版本
Isaac D00 基线为旧 runner 8f051161，D20 正式为 de872655（含 100-cycle checkpoint 块）。
checkpoint 块只做只读计算与 `if cyc==99` 落盘，物理上不写回 sim、不改 action，轨迹中性：D20 正式 trace（de872655）前 100 周期与 R2 probe（64365052，`01_delay_gate_r2` 复跑）逐点一致，q/q_des/action/root/body max diff = 0.0。故 Isaac 0 ms 证据沿用回归 runner 64365052（其 D00 对齐已验证）。MuJoCo 侧 0 ms 回归与正式 D20 同为 ab40ff04。
（对应"Isaac/MuJoCo D00/D20 runner 允许不同，前提是 0 ms 回归证据通过并被 validator 验证"）

## MJ 正式 D20 trace 为 P3 v1 复用
MuJoCo elbow-matched D20 10s trace 沿用 P3 v1（10-step PerStepDelayFifo repeat-first，从 trace 独立验证）；其 `q_des_effective_substep` 是「同一 q_des 在 control 周期内 10 个 physics step 恒等」的记录（非延迟 10 步的逐步值），validator 因此以 control 级 shift=1 + first 相等 + fifo_used + physics_dt/ppc 组合证明 20ms，不依赖 MJ substep 数组的逐步滞后。

## 绝对路径
执行期生成的 run_manifest.yaml 内含当时绝对 model/config 路径（如 E:\...、/mnt/e/...），属历史运行工件，未篡改。condition_index/ledger/validator 一律只用包内相对路径，且 validator 断言 condition index 无 `E:/` `/mnt/` 等标记。
