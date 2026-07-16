# Embodied AI Projects

## DreamZero — NPU 训练报告 & Full vs Action-Only 对比分析

- final_report.html — 3000 步完整训练+评测报告（自包含 HTML）
- eval_enhanced_comparison.html — 增强版 Full/AO/GT 同轴对比图
- base.py — Fix 7: resume 时真正加载 LoRA 适配器
- droid_16gpu.sh — 修复: 添加 $@ 透传 Hydra CLI override

核心发现:
- AO-Full gap 恒定 ~20-30 MSE，不随训练缩小
- Full 和 AO 预测曲线走势一致（同一权重），AO = Full + 常量偏置
- 视频去噪价值是场景依赖的：简单轨迹 AO≈Full，复杂轨迹需 Full
