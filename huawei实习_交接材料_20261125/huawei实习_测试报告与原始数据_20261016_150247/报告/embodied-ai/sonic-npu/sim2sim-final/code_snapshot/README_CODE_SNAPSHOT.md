# 代码快照 README（code_snapshot，2026-09-02 追加）

本目录把本周各实验用到的**运行代码**集中为一份便于阅读的快照。它是**便利副本，不是溯源锚点**：
正式结论绑定的代码哈希以各证据包内部的副本为准（见每节"权威锚点"列）。哈希清单：`SHA256_CODE_SNAPSHOT.txt`
（LF，`sha256sum -c` 兼容，路径相对本目录）。

注意：`14_final_delivery/` 与 `r5_1_overlay/` 保持原样未动；本目录**不在**最终 ZIP
`Sim2Sim_week_final_R5_1_R6_R7_v1.zip`（`SHA256_FEEDBACK_PACKAGE.txt`，108 项）的哈希范围内。

## 目录 → 实验 → 运行环境

| 子目录 | 内容 | 实验 | 运行环境 | 权威锚点 |
|---|---|---|---|---|
| `g0i_interface/scripts_n2/` | `compare_interface.py`、`run_isaac_onnx.py`、`run_mujoco_onnx.py`、`checks/run_gate_tests.py`（T1–T7）；`configs_used/era_l7_29dof_used.py` | 接口 G0-I（g0_report_v2）+ N2 脚本加固 | Isaac 侧 Windows `E:\isaaclab_env51`（Py3.11.9 / Isaac Sim 5.1.0 headless d3d12）；MJ 侧 WSL `~/.venv_lab_sim`（Py3.10.12 / mujoco 3.2.7） | `feedback/R3_interface_gate_v2.zip` 内 `scripts/` 同名文件 |
| `n4_isolation/` | `n4_actuator_contract.py`（生成脚本）+ `actuator_runtime_contract.csv`（29 关节 requested vs runtime 对照表，即"readback 表"本体） | N4 | Isaac 真启动 readback（Windows Kit）；MJ 侧 WSL 编译 XML | `04_actuator_isolation/actuator_runtime_contract.csv`（本目录为同哈希复制件） |
| `n5_b31/scripts/` | `asset_equivalence_check.py`(31060B)、`compare_repeats.py`、`mj_isolation_model.py`、`n5_r_summarize.py`(B.1 920 行版)、`urdf_isolation.py`(V3)、`run_isaac_onnx.py`、`run_mujoco_onnx.py`、`run_mujoco_step_test.py` | 资产 G0-A + 单关节隔离 N5-R2（18 条 trace）+ G1–G6 门禁 | MJ 阶跃：WSL mujoco 3.2.7；Isaac：Windows Kit | B3.1 ZIP 内 `code_B1_patch/scripts/`（本目录即从 ZIP 解包，逐字节一致） |
| `n5_b31/lab_dance9/` | `generate_lab_plots.py`、`make_sidebyside.py`、`make_sidebyside_labeled.py` | Lab 10s 双侧基线 + 并排视频 | 分析/合成：Windows Py3.11（matplotlib/imageio/ffmpeg） | B3.1 ZIP 内 `02_lab_dance9/` |
| `n5_b31/gym/` | `sim2sim_record.py`（自包含 12-DOF 常量 + TorchScript JIT，不 import humanoid 包） | Gym 100Hz 带日志基线 | WSL（physics dt=0.001，`MUJOCO_GL=egl`） | B3.1 ZIP 内 `05_gym/`；与 `05_gym/sim2sim_record.py` 同哈希（16982B） |
| `n5_b31/report/` | `final_package_validate.py`（B3.1 十项只读校验） | B.2/B3.1 终验 | Windows Py3.11 | B3.1 ZIP 内 `09_report/` |
| `p1/` | `make_fullbody_elbow_matched.py`、`audit_fullbody_elbow_assets.py`、`frozen_run_mujoco_onnx.py`、`p1_summarize.py`（含 `validate()` 门禁）、`p1_validate.py` | P1 全身右肘单变量（6×10s） | 正式 6 次 MJ 跑：WSL mujoco 3.2.7；审计：WSL；分析/图：Windows | `P1_fullbody_elbow_transfer_v1.zip`（`8bb8ba96…`，202 项） |
| `p2/` | `make_fullbody_arm_matched.py`、`audit_fullbody_arm_assets.py`、`frozen_run_mujoco_onnx.py`（与 P1 同哈希）、`p2_summarize.py`、`p2_validate.py`、`make_p2_videos.py`、`run_negative_tests.py` | P2 14 臂关节组扩展（9×10s） | 同 P1 | `P2_fullbody_arm_only_v1_1.zip`（`a48749cb…`，346 项）。fail-closed 修订版另见 `../14_final_delivery/validation/p2_validate_r6.py` |
| `p3_executed/executed_sources/` | `frozen_run_isaac_delay_r2.py`、`p3_instrumented_delayed_actuator_r2.py`、`run_mujoco_onnx.py` | P3 0/20ms 受控延迟（正式执行源码，SHA 绑定进 `p3_final_gate`） | Isaac D20 正式跑：Windows Kit；MJ 四条件：WSL mujoco 3.2.7 | R4 树 `13_p3_delay20/00_inputs/executed_sources/`（runner `de872655…`、actuator `85a06b93…`、`ab40ff04…`，`SHA256_FULL_TREE.txt` 108 项） |
| `p3_executed/runtime_dependencies/` | `era_l7_29dof.py`、`l7_29dof_tracking_env_cfg.py`、`tracking_env_cfg.py`、`robots_actuator.py`、`mimic_dance_9.yaml` | P3 运行时依赖（env/actuator/canonical 配置） | 随上 | R4 树 `00_inputs/runtime_dependencies/`（15 项 ledger 全哈希） |

## 未重复收录（已在本交付目录内）

- P3 R4/R5 分析门禁：`../r5_1_overlay/04_analysis/p3_validate.py`、`p3_summarize_r2.py`、`06_checks/run_p3_negative_tests.py`、`01_delay_gate_r2/explicit_fifo_unit_test_r2.py`、`05_media/make_four_panel_r5.py`
- P0/P2/R6/R7 validator：`../14_final_delivery/validation/`（`validate_evidence_closure_r6.py`、`p2_validate_r6.py`、`audit_fullbody_arm_assets_r6.py`、`validate_final_delivery.py`）
- 大 trace / 视频 / 冻结输入：在 R4 树与各反馈 ZIP，不在本快照重复。

## 已知版本漂移说明

lab 工作区 `E:\humanoid-lab\scripts\experiments\` 中同名脚本在 B3.1 之后继续演化过
（如 `n5_r_summarize.py` 现为 ~19.7KB 精简版、`asset_equivalence_check.py` 19.2KB），
**与本周结论绑定的执行版本是 `n5_b31/scripts/` 里的大版本**；需要复现 N5-R2 数字时以本快照或 B3.1 ZIP 为准。
仓库当时状态：humanoid-lab `c68c1e63`（dirty 88 项）、Isaac Lab `5c2ec81`（dirty），见 R4 树 `repo_state.json`。
