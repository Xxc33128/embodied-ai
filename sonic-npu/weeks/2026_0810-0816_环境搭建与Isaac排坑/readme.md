# 2026/8/10–8/16：环境搭建与 Isaac 排坑

## 本周目标

搭建 GR00T SONIC 复现所需的 Isaac Sim / MuJoCo 双引擎环境，启动 Isaac–MuJoCo 对齐工作。

## 关键动作与结论

- **工作区建立**（8/13）：Mac 端文档工作区 + 各仓库 clone 就位（GR00T-WholeBodyControl、GR00T-WBC-alignment、mujoco_warp 等）。
- **Isaac 4.5 环境排坑**：scenedb 反复崩溃，根因定位为**显卡驱动 610.88**；回退 580.88 + 禁用虚拟显示器后 smoke 通过。Isaac 后续升级 5.1.0.0 路线胜出（8/24 复盘收口）。
- **版本矩阵定版**：Python cp310/cp311 × CUDA cu128 的兼容三角梳理成型。
- **对齐实验起步**：GR00T-WBC-alignment 仓库（`feature/isaac-mujoco-alignment` 分支）自 8/02 起持续记录 Isaac/MuJoCo 精度对齐实验（α drift 定量等，见该仓库 `docs/mujoco-vs-isaac-precision-analysis.md`，未收录于本展示仓）。

## 本周文档沉淀位置

本周过程文档不按周收录，权威出处如下：

| 内容 | 位置 |
|---|---|
| 交接文档（新接手上下文，更新至 8/18） | [../../handover/交接文档-Isaac环境搭建.md](../../handover/交接文档-Isaac环境搭建.md) |
| 环境问题永久归档（43 坑 + 8 session 时间线） | [../2026_0824-0830_环境收口与Sim2Sim审查链启动/环境配置排障复盘-2026-08-24.md](../2026_0824-0830_环境收口与Sim2Sim审查链启动/环境配置排障复盘-2026-08-24.md) |

## 遗留 → 下周

环境仍有 16GB 内存天花板与 benchmark CLI 挂起问题 → 下周（8/17–23）Phase7 接口门禁收口时一并处理。
