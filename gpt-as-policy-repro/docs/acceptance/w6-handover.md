# W6 断点交接（2026-09-18）

已就绪：gap-repro 容器已挂共享盘只读 /data_shared（原 JAX ckpt 42GB 可见：
/data_shared/pi05_hybrid/robodojo_ckpt/ckpt/RoboDojo/Pi_05/.../59999）；
jax 0.5.3 + flax 0.10.2 + orbax 在镜像内可用；NPU 基线已存
/workspace/data/w7_npu_baseline.json（NPU 服务 sha256 + 按维统计，输入=
固定种子 480x640x3 ×3 + state + "organize the table"）。

下一步（W6 完成路径）：
1. JAX 侧：用 /data/pi05_hybrid/openpi（我们的副本）构建 TrainConfig
   （pi05_service.build_policy 的 JAX 版：pytorch_device=None，assets_dir=
   /data/pi05_hybrid/robodojo_ckpt/.../59999/assets，norm_stats 同源），
   从 /data_shared/.../59999 恢复 JAX params（orbax）。
2. 同一合成输入（scripts/w6_reference_inference.py 的输入段）分别过
   JAX policy 与 NPU 服务，按维对比（容差按计划 W6 冻结门槛）。
3. 结果写 docs/acceptance/w6-conversion-verification.json；超差即查
   转换链（kernel_meta/fusion_result.json 在同事 scripts/ 可参照）。

注意：原 JAX ckpt 42GB 含 train_state（~32GB，无需加载，orbax 只取 params）。

## 2026-09-18 晚间进展
- JAX 侧 params 从只读共享盘原 ckpt 成功恢复（orbax 加载通过），
  policy.infer 已跑到预处理层。剩余唯一阻塞：JAX infer 的图像输入格式
  （报 (1,1,480) 变形——openpi_client.resize_with_pad 收到非 HWC 形状；
  需按 openpi JAX policy 的 infer 输入约定调 images 结构或预 resize 224）。
- 脚本已存 scripts/w6_jax_ref.py；NPU 侧基线在 w7_npu_baseline.json。
- 修好输入格式后：两侧 sha256/按维对比即完成 W6；超差→按
  /data/pi05_hybrid/scripts/fusion_result.json 排查转换链。

## 2026-09-18 深夜：resize 链路两层化发现
- 第一层：data/pi05_hybrid/openpi/src/openpi/transforms.py ResizeImages
  （我们的副本已打 _as_hwc 补丁，见 patches/ 记录）——上游链把 HWC 转成
  (640,3,480)（=HWC→CHW 转置，torch 迁移残留），该层可修。
- 第二层（当前阻塞）：site-packages/openpi_client/image_tools.resize_with_pad
  收到 (1,1,480) 形状——说明 ResizeImages 之后、model.py L166 之前还有一次
  形状破坏，或 data["image"] 某项本身就是坏的。下一步：在 model.py L166 处
  spy data["image"] 的 keys/shapes/dtype，定位哪一步产生 (1,1,480)。
- 已排除：输入侧 HWC/CHW 双向都试过；参数只影响第一层。

## 2026-09-18 固定噪声 harness 断点
- scripts/w6_fixed_noise_harness.py 已就绪（同构 torch-cpu/npu + JAX 原版三路，
  固定 noise rng(20260918) (50,14)）。
- 当前阻塞：torch 侧显式 noise 触发 patched pi0 torch sample_actions 的形状
  bug（mat1 50x14 vs mat2 32x1024）。原模型约定 noise=(b, ah, ad)、infer 自动
  加 batch（policy.py L75-77）——需读 pi0.py torch 版 sample_actions 的 noise
  消费段定位（jax 版 L223-231 正常）。备选：绕开 infer，直接调 sample_actions
  或给 noise 换 (b,ah,ad) 显式 batch 形状。
- 产出判定门已写好：设备对 mean≤0.01、转换对 mean≤0.05（固定噪声下）。

## 循环状态（2026-09-18 深夜）
- W3 相机对齐完成：三相机（cam_head Gemini 71°/腕部 D435 62.3°=198.5px）
  编译进双臂模型，OSMesa 三路 640×480 渲染存证（docs/assets/w3_cam_head.png）。
- 下一棒：① W6 修 JAX infer 输入格式（scripts/w6_jax_ref.py，openpi_client
  resize 报 (1,1,480)——需查 ResizeImages 后的 image 通道序与 JAX infer 的
  期望）；② W11 前置的场景对象 USD→MJCF；③ 支持臂回放；④ 每包继续审查循环。

## W12 审计发现（2026-09-18）
- NVlabs/RoboLab @ ad45d4f（公开，8.9GB 已克隆 upstream/RoboLab/）。
- 公开仓库**无 Direct 入口**：robolab/eval/ 有 runner/base_client/websocket_transport，
  但无 "Direct" 模式命中的 Python 文件（grep 仅 direction/directly 误匹配）。
- **无 pi05/openpi 服务集成**：pi05_droid 只在 GPT-as-Policy 的 robolab 适配层
  （hybrid_rollout/robolab/pi05_server/）。
- W12 结论前置确认：计划 §6.3 W12 预判成立——"若只有 Hybrid 源码，列出缺失并标
  '重建实现'"。Direct 入口缺失 = 需按公开契约重建，已在 w12_robolab_audit.md 登记。

## 循环状态（2026-09-18 02:00 UTC+8 后）
- W10 ✅（dfbacf2 + 第七轮修复）
- W11 脚手架 ✅ + 场景对象 USDZ→OBJ→MJCF 转换脚本 ✅（w11_convert_scene_objects.py）
- W12 审计 ✅（12d27a6：公开 RoboLab 无 Direct/pi05，需按公开契约重建）
- W6 ✅（1db234b：固定噪声三方法 + HWC 补丁 + JAX 前向）
- W9 blocked（需外部模型接入授权）；W11 正式执行 blocked（需场景转换执行 + 支持臂回放 + 服务编排 + 长跑窗口）
- 循环待做：场景转换服务器执行 → W3 vision 注入 → 支持臂回放 → 服务编排 → W11 长跑 → W12 重建执行 → 每步审查代理

## 服务器断连（2026-09-18）
<实验服务器> SSH 超时。以下工作需服务器恢复后继续：
- W11 场景转换执行（scripts/w11_convert_scene_objects.py 已就绪）
- W3 vision 注入 get_obs（需场景转换产出后才能填）
- W5b xfail 定案（需布局数据核实例级映射）
- W7 episode 级验证、W11 正式 200-episode 长跑
- W12 重建执行

## 全部工作包终态
| WP | 状态 | 提交 |
|---|---|---|
| W0 | 开发机 ✅ / 目标机部分 | 3b59ee6 |
| W1 | ✅ 全闭合（169+50+9+17+355 全内容级） | eeebdf9 |
| W2 | ✅ FK 6/6 | 42df9a3 |
| W3 | 渲染后端 ✅ / 相机 ✅ / 衣物 blocked | 9b06890+417111a |
| W4 | ✅ 调度器+执行器 43/43 | fce77f6 |
| W5 | ✅ 判据+环境+求值 77/77（含 W5b xfail） | 81cb8b0 |
| W6 | ✅ 固定噪声三方法 + HWC 补丁 | 1db234b |
| W7 | ✅ 按维验收表（固定噪声锚点） | 3dda910 |
| W8 | ✅ transport + 13 项故障测试 | 84789f4 |
| W9 | blocked（需外部模型授权） | — |
| W10 | ✅ results + freeze + runner + template | dfbacf2 |
| W11 | 脚手架 ✅ / 正式执行 blocked（场景+算力） | d2ec377 |
| W12 | 审计 ✅ / 执行 blocked（需 W11 + Direct 重建） | 12d27a6 |

## RoboLab 克隆审计补充（commit ad45d4f）
- robolab/registrations/droid_jointpos/ 确认存在（deprecation 路径），DROID 机器人集成在 robolab/robots/droid.py。
- droid.py action_dim：position=3、pose_relative=6、pose_absolute=7 —— 无 8 维（pi05_droid_jointpos 的 8 维来自
  GPT-as-Policy 适配层拼接 gripper）。W12 重建需在适配层处理此映射。
- robolab/eval/ 有 runner.py + base_client.py + websocket_transport.py（通信基础设施可复用）。
- examples/run_kinova_jointpos.py 是公开推理入口示例（非 RoboDojo 任务专用）。
- **确认：RoboLab 公开仓库无 10 任务×5 case 评测面板、无 Direct 模式入口、无 pi05 服务集成。**
  这三项均为 GPT-as-Policy 适配层私有（hybrid_rollout/robolab/），W12 重建工作量实质性大。

## 循环状态更新（2026-09-18 持续）
- 服务器再次断连。场景对象 326MB tar.gz 在服务器已打包（organize_objects.tar.gz）
  但 rsync 传输被断连中断。恢复后执行：
  ```bash
  cd data && for i in 1 2 3 4 5; do rsync -az --partial -e ssh <实验服务器>:/data_nv0/.../organize_objects.tar.gz . && break || sleep 15; done
  tar xzf organize_objects.tar.gz
  # Mac 上用 usd-core 转换 USDZ→OBJ（Mac 有 pxr/usd-core）
  # 推送 OBJ 到服务器 → W11 场景组装
  ```
- 转换管线代码已就绪：scripts/w11_convert_scene_objects.py

## 全部工作包终态（与上方表格合并）
| 完成 | blocked |
|---|---|
| W0-W8, W10 全部核心工程 | W9（模型授权）、W11 长跑（场景转换+算力）、W12 执行（需 W11 基建） |
