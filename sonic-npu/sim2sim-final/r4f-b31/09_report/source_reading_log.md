# Source Reading Log (2026-08-28)

| source_path | exists | read_or_checked | purpose | key_fact_used | conflict_or_limitation |
|---|---|---|---|---|---|
| E:/sim2sim-week-2026-08-26/01_lab_interface_gate/aligned_v1/canonical_initial_state.npz | true | read | Lab canonical contract | joint_names 29, root pos -0.096 -1.92 1.00 quat 0.83 0.02 -0.05 0.55 | none |
| E:/humanoid-lab/deploy/src/era_rl_controller/configs/mimic_dance_9.yaml | true | read | Lab policy/motion/sim2sim config | control_dt 0.02 obs 770 | none |
| E:/humanoid-lab/deploy/src/era_rl_controller/era_rl_controller/rl_interfaces/mimic_rl_interface.py | true | read | Lab inference/action chain | ONNX metadata; `q_des=action*scale+default_q` | none |
| E:/humanoid-lab/deploy/src/era_rl_controller/era_rl_controller/rl_interfaces/hist_observations.py | true | read | Lab history construction | per-term ring buffers, configured order, flatten to 770 | initialization must be aligned |
| E:/humanoid-lab/source/.../l7_29dof_neck_fixed.urdf | true | checked | Lab URDF asset | 29 DOF, fix | none |
| E:/humanoid-lab/deploy/l7_29dof_neck_fixed/l7_29dof_neck_fixed.xml | true | checked | Lab MJCF asset | 29 hinge, d629... | reformatted to 69e9... but hash tracked |
| E:/humanoid-gym/humanoid/scripts/sim2sim.py | true | read | Gym viewer loop | 12 DOF, 705 obs, 0.25 scale | not used for baseline, viewer only |
| E:/humanoid-gym/humanoid/scripts/sim2sim_record.py | true | read | Gym self-contained 100Hz | 12 DOF, 100Hz, KPS 200/350/15 | new file, no Isaac Gym dep |
| E:/sim2sim-week-2026-08-26/02_lab_dance9/isaac_10s/policy_trace.npz | true | checked | Lab Isaac trace | 500 cycles, 770 obs, 29 action | video 0% black after fix |
| E:/sim2sim-week-2026-08-26/02_lab_dance9/mj_10s/policy_trace.npz | true | checked | Lab MuJoCo trace | 500 cycles, 770 obs | video 295 frames |
| E:/sim2sim-week-2026-08-26/04_actuator_isolation/n5_r/b1_patch/evidence/asset_equivalence_*.json | true | read | G0 gate | COM 5e-8, Ieff 0.075245 | dummy flange normalized |
| E:/sim2sim-week-2026-08-26/04_actuator_isolation/n5_r/smoke_b1_fixed/elbow_gates.json | true | read | Smoke gate | R20 0.94 | G6 NOT_EVALUATED |
| E:/sim2sim-week-2026-08-26/04_actuator_isolation/n5_r/N5R2_elbow_G6_v3.json | true | read | N5-R2 G6 | 3 repeats hash ok true | all diff 0 |
| E:/huawei/GR00T-WholeBodyControl/docs/EXPERIMENTS.md | false | NOT_FOUND (checked) | SONIC II/IM | II/IM source | use SONIC_Training_Report instead |
| E:/GR00T-WBC-alignment/docs/EXPERIMENTS.md | true | read | SONIC II/IM | E2/E3/E4 | no IM NPZ/video, write unavailable |
| E:/GR00T-WBC-alignment/SONIC_Training_Report.md | true | read | SONIC MM | g1_walk_final_r2_11000.mp4 | only MuJoCo |
| E:/GR00T-WBC-alignment/gear_sonic_deploy/policy/release/observation_config_sonic_release.yaml | true | read | SONIC deploy observations | token + 10-frame joint/velocity/action/gravity/ang-vel history | dimensions cross-checked in C++ |
| E:/GR00T-WBC-alignment/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/control_policy.hpp | true | read | SONIC policy I/O | input `obs_dict`, output `action`, validates 29 motors | control semantics elsewhere |
| E:/GR00T-WBC-alignment/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp | true | read | SONIC observation registry | joint pos/vel/action 290 each; gravity/ang-vel 30 each | deploy path only |
| E:/GR00T-WBC-alignment/docs/assets/g1_walk_final_r2_11000.mp4 | true | checked | SONIC MM video | exists | no log hash |
| E:/GR00T-WBC-alignment/docs/assets/isaac_official_weights.mp4 | false | NOT_FOUND | SONIC II | official weights | write unavailable, use report |
| E:/sim2sim-week-2026-08-26/00_static_evidence/evidence_ledger.csv | true | read | Evidence ledger | A/B/C/D | updated to A- for N5-R2 |

| E:/humanoid-gym/2026-08-26-三项目Sim2Sim两日实验执行计划.md | true | read | Original two-day plan (Must/Should/Won't, R0-R7 hard order) | Must 7 tasks, G0 gate | none |
| E:/humanoid-gym/2026-08-28-N5R2终验与最终行动清单.md | true | read | B.2 final checklist (18 items, final gate, video bbox) | B.2 closed, video 250/295 frames | none |
| E:/humanoid-gym/2026-08-31-R4F-B2-v4审查与B3修正执行清单.md | true | read | B.3 correction plan (Must 9 items, fail-closed, 64 hash, video, SONIC) | B.3 closed, 200 payload | none |
| E:/humanoid-gym/2026-08-31-R4F-B3审查与B3.1最终收口执行清单.md | true | read | B3.1 final closure (Must 8 items, 6.5m/4.0m camera, smoke 2s, re-render 10s diff 0, full LF SHA, read-only validator) | current execution basis, Isaac 6.5m/0.20 MuJoCo 4.0m/0.30, 6 frames 5% margins 52%/48% height, 212 payload (含报告自包含图片) | none |
| E:/huawei/GR00T-WholeBodyControl/docs/source/references/observation_config.md | true | checked | SONIC obs/history (930 dims) | 930 proprio +64 token->994->29 | not 770 |
