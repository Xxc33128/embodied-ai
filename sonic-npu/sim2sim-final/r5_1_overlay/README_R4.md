# R4 Final Gate — README

## 开头问答（按验收计划 §6）

1. **final validator 是否从解压包独立运行**: 是。`python 04_analysis/p3_validate.py --root .` 仅使用包内相对路径，在新鲜解压目录复跑通过（无 R1/R2 fallback、无绝对路径）。
2. **decision 是否为 VALID_P3**: 是（`04_analysis/p3_final_gate.json`，gates A–F 全 PASS，exit 0）。
3. **Isaac measured delay 是否为 4 steps / 20 ms**: 是。physics shift 唯一识别=4（0–20 搜索），shift max diff=0.0，第二优 diff=0.419 ≫ 阈值；control shift=1；physics index 连续 1..2000；physics_time==index×0.005。
4. **MuJoCo measured delay 是否为 10 steps / 20 ms**: 是。control shift 唯一=1（=10 physics steps × 0.002 s），shift max diff=0.0，fifo_used=true，repeat-first。
5. **四条件是否均为 500 周期、无 fall**: 是。四条件 policy_step=0..499、time_step=0..499、done 全 0、fall_reason 全空、summary cycles_done=500 fall=null；reset_observed=false 由 trace 计数器独立推断（P1-2）。
6. **0 ms baseline pairing 是否通过**: 是。Isaac R2 0ms(100cyc) vs D00 前 100 全字段 max diff 0.0（含 body quat/linvel/angvel）；MuJoCo 0ms 回归与新 D20 同 runner ab40ff04，vs D00 前 100 max diff 0.0；seed/start_frame/gain/mode/history/obs_source/canonical 四条件一致。
7. **negative tests 是否 8/8**: 是，且另含 untouched 对照（CONTROL=VALID_P3，证明 validator 非恒失败）。每个 case 复制整包→单项篡改→调用同一真实 validator→要求非零退出且 decision 精确匹配：
   T1 effective 篡改→INVALID_DELAY_IMPLEMENTATION；T2 删 q_des_effective_substep→INVALID_TRACE；T3 requested_delay_ms=15→INVALID_DELAY_IMPLEMENTATION；T4 canonical hash→INVALID_ARTIFACT_HASH；T5 model hash→INVALID_ARTIFACT_HASH；T6 删一周期→INVALID_TRACE；T7 physics index 不连续→INVALID_DELAY_IMPLEMENTATION；T8 seed parity→INVALID_BASELINE_PAIRING。详见 `06_checks/p3_negative_tests.json` 与 `06_checks/negative_test_logs.txt`。
8. **是否还有绝对路径或 fallback**: validator 与 condition index 无绝对路径、无 fallback。执行期 run_manifest.yaml 内部保留当时写入的绝对 model/config 路径（历史工件事实，未改动），已在 KNOWN_LIMITATIONS_R3.md 声明。

## 本包解决的 R4 前置问题

- **P0-1 自包含**：四条件 trace/manifest/summary + Isaac R2 0ms 回归 + MuJoCo 0ms 回归（同正式 runner）+ canonical/policy/motion/URDF/XML/执行源码全部入包；`run_delay_gate_r2/run_zero_regression_r2` 等依赖包外路径的旧脚本不再作为门禁入口，唯一门禁为 `p3_validate.py`。
- **P0-2 FIFO 单测**：新增 `01_delay_gate_r2/explicit_fifo_unit_test_r2.py`，直接导入正式执行的 `p3_instrumented_delayed_actuator_r2.py`（85a06b93…，stub 基类）验证 explicit deque：shift4、repeat-first prefill、pulse 落点、0ms 恒等、reset 无泄漏，7/7 PASS。原生 DelayBuffer 单测 `fifo_unit_test_r1.json` 保留为工程补充，不作正式实现证据。
- **P0-3 checkpoint 全哈希**：validator 用 64 位完整哈希比较 executed_sources 与 manifest（runner/actuator/model/policy/motion/canonical），不再检查非空或仅 8 位前缀（前缀另核对一次）。
- **P0-4 条件索引**：`00_inputs/p3_condition_index.json` 相对路径指向全部证据。
- **P1-1 表述统一**：全部文档写「显式 per-physics-step position-target FIFO（collections.deque）」，不宣称验证 Isaac 原生 DelayBuffer。
- **P1-3 body mapping**：`00_inputs/body_mapping.json` 交叉核对 ONNX metadata ↔ MJ manifest ↔ trace body 数组 (14)。
- **P1-4 provenance**：`runtime_dependency_ledger.json` 15 项全部哈希复算通过；`repo_state.json`（commit c68c1e63、dirty=true/88 项）与 D20 manifest 内嵌一致。

## 目录

- `00_inputs/` policy motion canonical isaac_model mj_elbow_model executed_sources runtime_dependencies + 三个 json
- `01_delay_gate_r2/` isaac 0ms 回归、mj 0ms 回归（同正式 runner）、explicit deque 单测(+脚本+结果)、历史 gate json
- `03_runs/` isaac_d00_10s_reference isaac_d20_10s_r2（trace+manifest+summary+checkpoint+video+frames） mj_elbow_d00_10s_reference mj_elbow_d20_10s
- `04_analysis/` p3_validate.py p3_final_gate.json
- `06_checks/` run_p3_negative_tests.py p3_negative_tests.json negative_test_logs.txt

## 结论口径

`VALID_P3` = 四条件 500 周期无 fall + 双侧延迟语义实测一致（Isaac 显式 FIFO 4×5ms，MJ 10×2ms，均 20ms）+ 输入/哈希/配对完整。延迟实验描述为**受控等时延注入**（受控实验结论），不外推到 Isaac 原生 DelayBuffer、其他动作、其他延迟档位。指标与媒体在 R5 产出。
