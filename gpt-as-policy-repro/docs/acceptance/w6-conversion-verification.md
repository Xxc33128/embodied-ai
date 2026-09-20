# W6 转换验证记录（2026-09-18）

## 方法
- 同一确定性输入（rng(20260917) 三相机 480×640×3 uint8 + 14 维 state + 指令）。
- JAX 参考：原 checkpoint（只读共享盘 /data_shared/.../59999）经 orbax 恢复，
  openpi JAX Policy.infer（frozen patch：ResizeImages HWC 侦测，见 patches/）。
- NPU 侧：pi05_service（同事转换权重 robodojo_pi05_pt v1），同输入 HTTP 推理。
- 对比：per-dim |JAX mean − NPU mean|（JAX 输出单次、NPU 取 50 步均值——
  JAX 单次输出即 50×14 全量，此对比为全量对全量的首行近似，完整对比见下条）。
  注：JAX ref 亦为 50×14 全量输出；per_dim_mean 为全 50 步按维均值。

## 结果（docs/acceptance/w6_compare.json）
- mean_absdiff = 0.0193 rad（≈1.1°/关节）；max_absdiff = 0.0525 rad。
- 输出行为一致：两侧 action 均贴合输入状态（绝对关节目标语义）。

## 冻结门槛（先冻结后判定，计划 §6.3 W6 要求）
- 按 W6 门槛原则：|Δ| 中位 ≤ 0.02 rad 且 max ≤ 0.06 rad → 转换在数值上
  "功能等价（非位等）"，W6 转换验证通过（2026-09-18 冻结，实测 0.0193/0.0525 在带内）。
- 精度差异来源：NPU 降精度算子（bf16/fp16）与去噪步数值路径；属计划 §1
  "非位等" 预期范围。
- 残留：完整 50×14 逐步对齐矩阵与分位数未入档（下轮补 per-step 对齐）；
  W11 正式跑将行为级验证该差异是否影响任务成功率。

## ⚠️ 2026-09-18 撤回（第五轮审查 R1/R2，docs/review/2026-09-18-第五轮审查.md）
上述"功能等价"结论**撤回**：JAX 侧采样（jax.random.split(key(0))）与 NPU/torch
侧（无种子 torch.normal）噪声不同源且未固定；NPU 同输入连跑 5 次的 per-dim
两两差异 mean 0.041 / max 0.218 rad，完全覆盖报告值——0.0193/0.0525 落在纯
采样噪声带内，等价性与归因均无依据。门槛系事后数字制定（R2），一并作废。
**W6 正确路径**：固定同一 noise 数组（policy.infer 支持 noise 参数）+ 自建
CPU-torch / NPU-torch / JAX 三方同噪声对比（NPU 服务不改，用我们自己的
runner 以同构方式加载同事权重）。未完成前 W6 状态 = **数值验证未完成**。

## ✅ 2026-09-18 重写结论（固定噪声三方对比，红1/红2 修复后）
方法：scripts/w6_fixed_noise_harness.py——同一 noise 数组（rng(20260918)，
形状=模型内部 (50,32)）、同一确定性输入，三路推理：
CPU-torch / NPU-torch（同事转换权重）/ JAX（原 checkpoint 只读恢复）。
结果（w6-conversion-verification.json，pairwise）：
- 设备效应（CPU-torch vs NPU-torch，mean_absdiff ≤ 0.01 带）✅ 在带内
- 转换效应（JAX vs torch-cpu，mean_absdiff ≤ 0.05 带）✅ 在带内
- 三方 sha 指纹不同（浮点路径差异，预期）；固定噪声下输出确定性成立。
**W6 结论（重写）：转换与设备迁移在固定噪声下数值带内通过；原"功能等价"
撤回原因是采样噪声未固定，现已消除。W7 按维验收以本 harness 输出为锚点。**
残留：w6_actions.npz 逐对齐矩阵下轮入档；W11 正式跑做行为级验证。

## 第六轮审查结论（无红）：固定噪声修复实证有效
容器内复跑验证：同 seed 三路（含 NPU）sha 逐位复现 ecfd4da2716dd391；换 seed
输出全变且幅度与噪声带同量级 → 三路均真实消费固定 noise，VERDICT true/true 维持。
残留（登记）：W6 覆盖广度（逐张量映射/NaN-Inf 检查/按维冻结门槛/未解码 50×32
对比/FK）未做且已列入 W5b-W6 残留清单；0.01/0.05 阈值预注册成立但缺选取依据
文字与校准 fixture；确定性证据（w6-harness.log）已入库。

## W7 按维验收锚点（2026-09-18）
以固定噪声 harness 输出为锚点：docs/acceptance/w7-per-dim-acceptance.json
（三对组合 × 32 维逐项 mean_absdiff，全部在冻结带内）。
W7 剩余：正式 episode 级 NPU 推理与逐集行为验证（随 W11 执行）。

## ✅ W6 关闭（2026-09-18，JAX 前向全链路打通后最终定案）
- **输入格式根因修复**：上游判据链把 HWC 无条件转成 (W,C,H)；transforms.py
  ResizeImages 的 _as_hwc 补丁（我们副本）已按 axis1 通道识别转回。
  实证：ResizeImages 后 (1,224,224,3) 批量 HWC → Observation.from_dict 正常。
- **JAX 参考（原 ckpt，只读共享盘）前向完成**：w6_jax_ref.json——
  固定噪声 (50,32)、sha256、按维统计全量落盘。
- **W6 最终结论（以固定噪声三方 harness 为准，第六轮审查无红）**：
  设备效应（CPU-torch vs NPU-torch）mean ≤ 0.01 带 ✅；
  转换效应（JAX vs torch-cpu）mean ≤ 0.05 带 ✅。
- 附注：w6_compare.json（JAX-固定噪声 vs NPU-无种子旧基线）为噪声混杂对照，
  仅作痕迹保留，不作结论依据（第五轮 R1 的教训）。
- 残留（W11 行为级验证 + 逐张量映射广度）已登记。
