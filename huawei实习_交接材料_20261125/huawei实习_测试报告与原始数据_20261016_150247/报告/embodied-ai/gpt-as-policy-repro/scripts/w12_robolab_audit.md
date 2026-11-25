# W12 RoboLab 审计清单（2026-09-18，按计划 §6.3 W12）

## 已核对（E1，GPT-as-Policy@8f3d362b）
- checkpoint 契约：hybrid_rollout/robolab/pi05_server/checkpoint.py —— pi05_droid_jointpos，
  horizon=15、action_dim=8（禁止沿用 RoboDojo 50×14 契约）。

## 待核对（按计划顺序）
1. 原 RoboLab 源码锁定（NVlabs/RoboLab）与 10 任务×5 case 身份、指令、预算、评分。
2. 原版 Direct 入口是否公开；案例身份能否取得——若只有 Hybrid 源码，列出缺失
   文件/行为，重建实现须标"重建"，不得声称保留原代码。
3. Franka/DROID 资产与 checkpoint 身份（独立于 RoboDojo 的 50×32 契约）。
4. 独立 configs/robolab/ 与 src/gap_repro/robolab/；复用 transport/日志时验证接口差异。
5. 两路线 Direct/Hybrid 各 50 + Student-only 50；不与 RoboDojo 合并成功率。

## 如实状态
- W12 未开始执行：前置依赖 W8 transport（已完成）+ W11 runner（脚手架已建，
  正式执行待 W3 渲染 + 场景转换 + 服务编排 + 模型接入授权）。
- 本文件为审计清单，不构成完成声明。
